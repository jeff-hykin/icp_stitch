//! Combined comparison rrd (port of gsc_pgo `scripts/make_rrd.py`): raw lidar cloud +
//! every `*_corrected*_accumulated` cloud in the db, each its own colored entity, plus
//! AprilTag landmarks (textured tag squares + medoid camera frustums) and trajectories.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use gtsam_shim::Pose3;
use kornia_apriltag::family::TagFamily;
use lcm_msgs::sensor_msgs::Image as LcmImage;
use rusqlite::Connection;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;

use crate::apriltags::{Detection, GlimpseGates, filter_glimpses};
use crate::artifacts::nearest_index;
use crate::detect::{CameraModel, resolve_camera_info};
use crate::mat3::{self, Mat3};
use crate::memory2;
use crate::recording::default_odom_edge;
use crate::se3;
use crate::tf::RecordingTf;

const SCAN_STRIDE: usize = 8;
const POINT_STRIDE: usize = 3;
const VOXEL: f64 = 0.10;
/// Cap each accumulated cloud so rerun stays responsive.
const MAX_RENDER_POINTS: usize = 200_000;
const TAG_SIZE_M: f64 = 0.10;
/// 36h11 incl. border is 8 modules; render each as 25 px.
const TAG_IMAGE_PX: usize = 200;
/// How far from a medoid glimpse to look for its color frame.
const CAMERA_MATCH_SEC: f64 = 0.2;
/// Image plane distance of the placed medoid views.
const FRUSTUM_PLANE_M: f32 = 0.6;
const DEFAULT_ROTATION_WEIGHT_M_PER_RAD: f64 = 0.5;
/// Clip outlier floors/ceilings out of the color range.
const Z_GRADIENT_PERCENTILES: (f64, f64) = (2.0, 98.0);
const POINT_RADIUS: f32 = 0.02;

// each entry fades between two distinct hues. Both endpoints are kept bright so the ramp
// reads as a change of color, not of brightness.
type Gradient = ([f64; 3], [f64; 3]);
const RAW_CLOUD_GRADIENT: Gradient = ([235.0, 45.0, 95.0], [250.0, 205.0, 60.0]);
const RAW_TRAJECTORY_GRADIENT: Gradient = ([255.0, 95.0, 165.0], [255.0, 240.0, 120.0]);
const CLOUD_GRADIENTS: [Gradient; 4] = [
    ([60.0, 110.0, 255.0], [90.0, 245.0, 180.0]),
    ([70.0, 200.0, 90.0], [235.0, 230.0, 70.0]),
    ([170.0, 90.0, 250.0], [250.0, 110.0, 170.0]),
    ([250.0, 130.0, 60.0], [245.0, 225.0, 130.0]),
];
const TRAJECTORY_GRADIENTS: [Gradient; 4] = [
    ([120.0, 175.0, 255.0], [140.0, 255.0, 220.0]),
    ([130.0, 245.0, 140.0], [245.0, 255.0, 130.0]),
    ([210.0, 150.0, 255.0], [255.0, 165.0, 210.0]),
    ([255.0, 175.0, 110.0], [255.0, 240.0, 180.0]),
];

/// Relaxed vs the detection defaults: landmarks are display markers, not PGO factors.
fn landmark_gates() -> GlimpseGates {
    GlimpseGates {
        min_sharpness: 25.0,
        max_reproj_px: 3.5,
        min_tag_px: 12.0,
        max_distance_m: 1.5,
        max_view_angle_deg: 65.0,
        max_linear_speed_mps: 1.5,
        max_angular_speed_dps: 150.0,
    }
}

/// Colors interpolated from the gradient's first color to its second, at fraction t in [0, 1].
fn ramp(gradient: &Gradient, t: f64) -> [u8; 3] {
    let (start, end) = gradient;
    [
        (start[0] + (end[0] - start[0]) * t) as u8,
        (start[1] + (end[1] - start[1]) * t) as u8,
        (start[2] + (end[2] - start[2]) * t) as u8,
    ]
}

/// numpy-style linear-interpolated percentile.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = q / 100.0 * (sorted.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (rank - low as f64)
}

/// Per-point colors fading across the gradient with height.
fn z_gradient_colors(points: &[[f32; 3]], gradient: &Gradient) -> Vec<[u8; 3]> {
    let mut z_values: Vec<f64> = points.iter().map(|p| p[2] as f64).collect();
    z_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let low = percentile(&z_values, Z_GRADIENT_PERCENTILES.0);
    let high = percentile(&z_values, Z_GRADIENT_PERCENTILES.1);
    let span = if high - low == 0.0 { 1.0 } else { high - low };
    points
        .iter()
        .map(|p| ramp(gradient, ((p[2] as f64 - low) / span).clamp(0.0, 1.0)))
        .collect()
}

/// (segments, colors) for a path fading across the gradient from start to finish.
fn gradient_trajectory(
    positions: &[[f32; 3]],
    gradient: &Gradient,
) -> (Vec<[[f32; 3]; 2]>, Vec<[u8; 3]>) {
    let count = positions.len().saturating_sub(1);
    let segments: Vec<[[f32; 3]; 2]> = (0..count)
        .map(|i| [positions[i], positions[i + 1]])
        .collect();
    let colors = (0..count)
        .map(|i| {
            let t = if count <= 1 {
                0.0
            } else {
                i as f64 / (count - 1) as f64
            };
            ramp(gradient, t)
        })
        .collect();
    (segments, colors)
}

/// RGB bitmap of the actual AprilTag (border + 6x6 code), for texturing its 3D placement.
fn tag_image(marker_id: i64) -> Vec<u8> {
    let family = TagFamily::tag36_h11();
    let cells = family.width_at_border as usize;
    let module_px = TAG_IMAGE_PX / cells;
    let code = family.code_data[marker_id as usize];
    let border = cells - 1;
    let mut rgb = vec![0u8; TAG_IMAGE_PX * TAG_IMAGE_PX * 3];
    for cell_y in 0..cells {
        for cell_x in 0..cells {
            let mut intensity = 0u8;
            if cell_x != 0 && cell_y != 0 && cell_x != border && cell_y != border {
                for bit in 0..family.nbits {
                    if family.bit_x[bit] == cell_x as i8 && family.bit_y[bit] == cell_y as i8 {
                        if (code >> (family.nbits - 1 - bit)) & 1 == 1 {
                            intensity = 255;
                        }
                        break;
                    }
                }
            }
            for row in cell_y * module_px..(cell_y + 1) * module_px {
                for col in cell_x * module_px..(cell_x + 1) * module_px {
                    let base = (row * TAG_IMAGE_PX + col) * 3;
                    rgb[base] = intensity;
                    rgb[base + 1] = intensity;
                    rgb[base + 2] = intensity;
                }
            }
        }
    }
    rgb
}

/// Spatial + rotational distance between two xyzquat poses.
fn pose_distance(a: &[f64; 7], b: &[f64; 7], rotation_weight_m_per_rad: f64) -> f64 {
    let translation = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let dot: f64 = (3..7).map(|i| a[i] * b[i]).sum();
    let rotation = 2.0 * dot.abs().min(1.0).acos();
    translation + rotation_weight_m_per_rad * rotation
}

/// The detection whose pose is most central (min total spatial+rotational distance
/// to the rest) -- a robust representative of the cluster.
fn cluster_medoid(cluster: &[&Detection]) -> usize {
    let mut best_index = 0;
    let mut best_cost = f64::INFINITY;
    for i in 0..cluster.len() {
        let cost: f64 = (0..cluster.len())
            .filter(|&j| j != i)
            .map(|j| {
                pose_distance(
                    &cluster[i].t_cam_marker,
                    &cluster[j].t_cam_marker,
                    DEFAULT_ROTATION_WEIGHT_M_PER_RAD,
                )
            })
            .sum();
        if cost < best_cost {
            best_cost = cost;
            best_index = i;
        }
    }
    best_index
}

fn odom_samples(connection: &Connection, stream: &str) -> Result<Vec<[f64; 8]>, String> {
    Ok(memory2::read_odometry(connection, stream, 1)?
        .iter()
        .map(|row| {
            [
                row.ts,
                row.translation[0],
                row.translation[1],
                row.translation[2],
                row.quaternion_xyzw[0],
                row.quaternion_xyzw[1],
                row.quaternion_xyzw[2],
                row.quaternion_xyzw[3],
            ]
        })
        .collect())
}

/// Voxel-thinned world-space accumulation of a lidar stream (register=true applies the tf).
fn accumulate(
    connection: &Connection,
    db_path: &Path,
    stream_name: &str,
    register: bool,
    store_tf: &RecordingTf,
    world_frame: &str,
    fallback_frame: &str,
) -> Result<Vec<[f32; 3]>, String> {
    let scans = memory2::read_scans(connection, stream_name, SCAN_STRIDE)
        .map_err(|_| format!("stream '{stream_name}' has no points in {}", db_path.display()))?;
    let mut all_points: Vec<[f32; 3]> = Vec::new();
    for scan in &scans {
        if scan.points.is_empty() {
            continue;
        }
        if register {
            let scan_frame = if scan.frame_id.is_empty() {
                fallback_frame
            } else {
                &scan.frame_id
            };
            let pose = store_tf.get(world_frame, scan_frame, scan.ts)?;
            for point in scan.points.iter().step_by(POINT_STRIDE) {
                let p = [point[0] as f64, point[1] as f64, point[2] as f64];
                let world = mat3::add(&mat3::mat_vec(&pose.rotation, &p), &pose.translation);
                all_points.push([world[0] as f32, world[1] as f32, world[2] as f32]);
            }
        } else {
            all_points.extend(scan.points.iter().step_by(POINT_STRIDE));
        }
    }
    if all_points.is_empty() {
        return Err(format!(
            "stream '{stream_name}' has no points in {}",
            db_path.display()
        ));
    }
    let mut seen: ahash::AHashSet<[i64; 3]> = ahash::AHashSet::new();
    let mut voxelized: Vec<[f32; 3]> = Vec::new();
    for point in &all_points {
        let key = [
            (point[0] as f64 / VOXEL).floor() as i64,
            (point[1] as f64 / VOXEL).floor() as i64,
            (point[2] as f64 / VOXEL).floor() as i64,
        ];
        if seen.insert(key) {
            voxelized.push(*point);
        }
    }
    if voxelized.len() > MAX_RENDER_POINTS {
        let step = voxelized.len() as f64 / MAX_RENDER_POINTS as f64;
        voxelized = (0..MAX_RENDER_POINTS)
            .map(|i| voxelized[(i as f64 * step) as usize])
            .collect();
    }
    Ok(voxelized)
}

struct Landmark {
    marker_id: i64,
    position: [f64; 3],
    rotation: Mat3,
    camera_pose: Pose3,
    ts: f64,
}

/// Per marker: the mean tag position, plus the medoid glimpse -- the detection whose
/// pose is most central -- which orients the tag square and places the camera frustum.
fn landmarks(
    connection: &Connection,
    streams: &[String],
    tag_stream: &str,
    gt_odom: &str,
    base_to_optical: &Pose3,
) -> Result<Vec<Landmark>, String> {
    let odom_rows = odom_samples(connection, gt_odom)?;
    if odom_rows.is_empty() || !streams.iter().any(|s| s == tag_stream) {
        return Ok(Vec::new());
    }
    let odom_times: Vec<f64> = odom_rows.iter().map(|row| row[0]).collect();
    let raw = crate::detect::read_raw_tag_stream(connection, tag_stream)?;
    let detections = filter_glimpses(&raw, &Default::default(), &landmark_gates());
    let mut by_marker: BTreeMap<i64, Vec<(&Detection, Pose3, Pose3)>> = BTreeMap::new();
    for detection in &detections {
        let row = &odom_rows[nearest_index(&odom_times, detection.ts)];
        let base_pose = se3::from_xyzquat(&row[1..]);
        let camera_pose = se3::compose(&base_pose, base_to_optical);
        let tag_in_world = se3::compose(&camera_pose, &se3::from_xyzquat(&detection.t_cam_marker));
        by_marker
            .entry(detection.marker_id)
            .or_default()
            .push((detection, tag_in_world, camera_pose));
    }
    let mut found = Vec::new();
    for (marker_id, glimpses) in &by_marker {
        let cluster: Vec<&Detection> = glimpses.iter().map(|g| g.0).collect();
        let medoid = cluster_medoid(&cluster);
        let (medoid_detection, medoid_tag, medoid_camera) = &glimpses[medoid];
        let mut position = [0.0f64; 3];
        for (_, tag_in_world, _) in glimpses {
            position = mat3::add(&position, &tag_in_world.translation);
        }
        for value in &mut position {
            *value /= glimpses.len() as f64;
        }
        found.push(Landmark {
            marker_id: *marker_id,
            position,
            rotation: medoid_tag.rotation,
            camera_pose: medoid_camera.clone(),
            ts: medoid_detection.ts,
        });
    }
    Ok(found)
}

/// Decode an LCM Image into (rgb bytes, width, height).
fn decode_rgb(image: &LcmImage) -> Result<(Vec<u8>, usize, usize), String> {
    let encoding = image.encoding.to_ascii_lowercase();
    if encoding == "jpeg" || encoding == "jpg" {
        let mut decoder = JpegDecoder::new(ZCursor::new(&image.data[..]));
        decoder
            .decode_headers()
            .map_err(|e| format!("jpeg header decode failed: {e:?}"))?;
        let info = decoder.info().ok_or("jpeg missing info")?;
        let pixels = decoder
            .decode()
            .map_err(|e| format!("jpeg decode failed: {e:?}"))?;
        let components = decoder.output_colorspace().map(|c| c.num_components()).unwrap_or(3);
        let (width, height) = (info.width as usize, info.height as usize);
        let rgb = match components {
            3 => pixels,
            1 => pixels.iter().flat_map(|&v| [v, v, v]).collect(),
            n => return Err(format!("jpeg with {n} components unsupported")),
        };
        return Ok((rgb, width, height));
    }
    let width = image.width as usize;
    let height = image.height as usize;
    let step = image.step as usize;
    let channels = match encoding.as_str() {
        "rgb8" | "bgr8" => 3,
        "rgba8" | "bgra8" => 4,
        "mono8" | "8uc1" => 1,
        other => return Err(format!("unsupported image encoding {other:?}")),
    };
    let row_bytes = if step > 0 { step } else { width * channels };
    let mut rgb = Vec::with_capacity(width * height * 3);
    let bgr = encoding.starts_with("bgr");
    for row in 0..height {
        for col in 0..width {
            let base = row * row_bytes + col * channels;
            if channels == 1 {
                let v = image.data[base];
                rgb.extend([v, v, v]);
            } else if bgr {
                rgb.extend([image.data[base + 2], image.data[base + 1], image.data[base]]);
            } else {
                rgb.extend([image.data[base], image.data[base + 1], image.data[base + 2]]);
            }
        }
    }
    Ok((rgb, width, height))
}

/// The camera frame nearest `ts` (within `CAMERA_MATCH_SEC`), or None.
fn camera_frame_at(
    connection: &Connection,
    camera_stream: &str,
    ts: f64,
) -> Option<(Vec<u8>, usize, usize)> {
    if camera_stream.contains('"') {
        return None;
    }
    let blob: Vec<u8> = connection
        .query_row(
            &format!(
                "SELECT blob.data FROM \"{camera_stream}\" AS meta \
                 JOIN \"{camera_stream}_blob\" AS blob ON meta.id = blob.id \
                 WHERE ABS(meta.ts - ?1) <= ?2 ORDER BY ABS(meta.ts - ?1) LIMIT 1"
            ),
            rusqlite::params![ts, CAMERA_MATCH_SEC],
            |row| row.get(0),
        )
        .ok()?;
    let blob = memory2::decompress_if_lz4(blob);
    let image = LcmImage::decode(&blob).ok()?;
    decode_rgb(&image).ok()
}

/// Place the camera's-eye image on a pinhole frustum at the pose it was taken from.
fn log_medoid_camera(
    rec: &rerun::RecordingStream,
    connection: &Connection,
    camera_stream: &str,
    entity: &str,
    camera_model: Option<&CameraModel>,
    camera_pose: &Pose3,
    ts: f64,
) -> Result<bool, String> {
    let Some(camera) = camera_model else {
        return Ok(false);
    };
    let Some((rgb, width, height)) = camera_frame_at(connection, camera_stream, ts) else {
        return Ok(false);
    };
    let translation = [
        camera_pose.translation[0] as f32,
        camera_pose.translation[1] as f32,
        camera_pose.translation[2] as f32,
    ];
    let rotation = &camera_pose.rotation;
    // rerun mat3x3 is column-major
    let mat3x3: [f32; 9] = [
        rotation[0][0] as f32,
        rotation[1][0] as f32,
        rotation[2][0] as f32,
        rotation[0][1] as f32,
        rotation[1][1] as f32,
        rotation[2][1] as f32,
        rotation[0][2] as f32,
        rotation[1][2] as f32,
        rotation[2][2] as f32,
    ];
    rec.log_static(
        entity,
        &rerun::Transform3D::from_translation(translation)
            .with_mat3x3(rerun::datatypes::Mat3x3(mat3x3)),
    )
    .map_err(|e| e.to_string())?;
    // rerun image_from_camera is column-major
    let k = rerun::datatypes::Mat3x3([
        camera.fx as f32,
        0.0,
        0.0,
        camera.skew as f32,
        camera.fy as f32,
        0.0,
        camera.cx as f32,
        camera.cy as f32,
        1.0,
    ]);
    rec.log_static(
        entity,
        &rerun::Pinhole::new(k)
            .with_resolution([width as f32, height as f32])
            .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
            .with_image_plane_distance(FRUSTUM_PLANE_M),
    )
    .map_err(|e| e.to_string())?;
    rec.log_static(
        format!("{entity}/rgb"),
        &rerun::Image::from_rgb24(rgb, [width as u32, height as u32]),
    )
    .map_err(|e| e.to_string())?;
    Ok(true)
}

/// Write `corrected_compare.rrd` next to the db and return its path.
#[allow(clippy::too_many_arguments)]
pub fn build(
    connection: &Connection,
    db_path: &Path,
    lidar_stream: &str,
    odom_stream: &str,
    tag_stream: &str,
    world_frame: &str,
    camera_stream: &str,
    camera_info_stream: &str,
) -> Result<PathBuf, String> {
    let out_path = db_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("corrected_compare.rrd");
    let odom_tf = default_odom_edge(connection, odom_stream);
    let body_frame = if odom_tf.is_empty() {
        world_frame.to_string()
    } else {
        odom_tf.split_once(':').map(|(_, c)| c.to_string()).unwrap_or_default()
    };
    let lidar_frame = body_frame.clone();
    let streams = memory2::list_streams(connection)?;

    let odom_rows = odom_samples(connection, odom_stream)?;
    let tf_samples = if streams.iter().any(|s| s == "tf") {
        memory2::read_tf(connection, "tf")?
    } else {
        Vec::new()
    };
    let mut store_tf = RecordingTf::from_samples(&tf_samples);
    if !odom_tf.is_empty() && streams.iter().any(|s| s == odom_stream) {
        let (parent, child) = odom_tf.split_once(':').unwrap_or((odom_tf.as_str(), ""));
        let trajectory: Vec<(f64, Pose3)> = odom_rows
            .iter()
            .map(|row| (row[0], se3::from_xyzquat(&row[1..])))
            .collect();
        store_tf.override_edge(parent, child, trajectory);
    }

    // base<-optical camera extrinsic, read from the tf tree
    let (camera_info, camera_info_tried) =
        resolve_camera_info(connection, camera_stream, camera_info_stream)?;
    let mut camera_model = None;
    let mut base_to_optical = None;
    match camera_info {
        None => println!("no CameraInfo stream among ['{}']", camera_info_tried.join("', '")),
        Some((model, frame_id)) => {
            let optical_frame = if frame_id.is_empty() {
                "camera_optical".to_string()
            } else {
                frame_id
            };
            let lookup_ts = odom_rows.last().map(|row| row[0]).unwrap_or(0.0);
            match store_tf.get(&body_frame, &optical_frame, lookup_ts) {
                Ok(pose) => base_to_optical = Some(pose),
                Err(_) => println!("no {body_frame} <- {optical_frame} tf edge"),
            }
            camera_model = Some(model);
        }
    }

    // the world-registered accumulated corrected clouds; the per-scan `*_corrected` streams
    // are stored sensor-relative, so they'd render as a blob without tf.
    let mut corrected_lidars: Vec<String> = streams
        .iter()
        .filter(|name| name.contains("_corrected") && name.ends_with("_accumulated"))
        .cloned()
        .collect();
    corrected_lidars.sort();
    println!("corrected accumulated lidar streams: {corrected_lidars:?}");

    let rec = rerun::RecordingStreamBuilder::new("corrected_compare")
        .save(&out_path)
        .map_err(|e| e.to_string())?;

    let raw_cloud = accumulate(
        connection,
        db_path,
        lidar_stream,
        true,
        &store_tf,
        world_frame,
        &lidar_frame,
    )?;
    rec.log_static(
        "raw/cloud",
        &rerun::Points3D::new(raw_cloud.iter().copied())
            .with_colors(z_gradient_colors(&raw_cloud, &RAW_CLOUD_GRADIENT))
            .with_radii([POINT_RADIUS]),
    )
    .map_err(|e| e.to_string())?;
    let raw_positions: Vec<[f32; 3]> = odom_rows
        .iter()
        .map(|row| [row[1] as f32, row[2] as f32, row[3] as f32])
        .collect();
    let (raw_segments, raw_colors) = gradient_trajectory(&raw_positions, &RAW_TRAJECTORY_GRADIENT);
    rec.log_static(
        "raw/trajectory",
        &rerun::LineStrips3D::new(raw_segments.iter().map(|s| s.to_vec())).with_colors(raw_colors),
    )
    .map_err(|e| e.to_string())?;

    let mut corrected_odoms: Vec<String> = streams
        .iter()
        .filter(|name| name.contains("_corrected") && name.contains("odom"))
        .cloned()
        .collect();
    corrected_odoms.sort();

    for (lidar_index, lidar_name) in corrected_lidars.iter().enumerate() {
        let gradient = &CLOUD_GRADIENTS[lidar_index % CLOUD_GRADIENTS.len()];
        let cloud = accumulate(
            connection,
            db_path,
            lidar_name,
            false,
            &store_tf,
            world_frame,
            &lidar_frame,
        )?;
        rec.log_static(
            format!("{lidar_name}/cloud"),
            &rerun::Points3D::new(cloud.iter().copied())
                .with_colors(z_gradient_colors(&cloud, gradient))
                .with_radii([POINT_RADIUS]),
        )
        .map_err(|e| e.to_string())?;
        println!("  logged {lidar_name}: {} pts", crate::artifacts::format_thousands(cloud.len()));
    }

    if let Some(corrected_odom) = corrected_odoms.first() {
        let corrected_positions: Vec<[f32; 3]> = odom_samples(connection, corrected_odom)?
            .iter()
            .map(|row| [row[1] as f32, row[2] as f32, row[3] as f32])
            .collect();
        let (segments, colors) =
            gradient_trajectory(&corrected_positions, &TRAJECTORY_GRADIENTS[0]);
        rec.log_static(
            "corrected/trajectory",
            &rerun::LineStrips3D::new(segments.iter().map(|s| s.to_vec())).with_colors(colors),
        )
        .map_err(|e| e.to_string())?;

        // landmarks placed against the first available corrected odometry
        match &base_to_optical {
            None => println!("no CameraInfo stream or optical tf edge -- skipping tag landmarks"),
            Some(base_to_optical) => {
                let found =
                    landmarks(connection, &streams, tag_stream, corrected_odom, base_to_optical)?;
                let mut images_logged = 0usize;
                let half = TAG_SIZE_M / 2.0;
                // marker-frame corners (aruco: x right, y up, z out), texcoord order TL TR BR BL
                let tag_corners =
                    [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]];
                for marker in &found {
                    let vertices: Vec<[f32; 3]> = tag_corners
                        .iter()
                        .map(|corner| {
                            let world = mat3::add(
                                &mat3::mat_vec(&marker.rotation, corner),
                                &marker.position,
                            );
                            [world[0] as f32, world[1] as f32, world[2] as f32]
                        })
                        .collect();
                    rec.log_static(
                        format!("landmarks/tag{}", marker.marker_id),
                        &rerun::Mesh3D::new(vertices)
                            .with_triangle_indices([[0u32, 1, 2], [0, 2, 3]])
                            .with_vertex_texcoords([[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
                            .with_albedo_texture_image(rerun::Image::from_rgb24(
                                tag_image(marker.marker_id),
                                [TAG_IMAGE_PX as u32, TAG_IMAGE_PX as u32],
                            )),
                    )
                    .map_err(|e| e.to_string())?;
                    if log_medoid_camera(
                        &rec,
                        connection,
                        camera_stream,
                        &format!("landmarks/tag{}/medoid_view", marker.marker_id),
                        camera_model.as_ref(),
                        &marker.camera_pose,
                        marker.ts,
                    )? {
                        images_logged += 1;
                    }
                }
                if !found.is_empty() {
                    rec.log_static(
                        "landmarks/labels",
                        &rerun::Points3D::new(found.iter().map(|marker| {
                            [
                                marker.position[0] as f32,
                                marker.position[1] as f32,
                                marker.position[2] as f32,
                            ]
                        }))
                        .with_colors([[255u8, 230, 0]])
                        .with_radii([0.005f32])
                        .with_labels(found.iter().map(|m| format!("tag{}", m.marker_id))),
                    )
                    .map_err(|e| e.to_string())?;
                    println!("  logged {} landmarks, {images_logged} medoid views", found.len());
                }
            }
        }
    }
    let _ = rec.flush_blocking();
    println!("wrote {}", out_path.display());
    Ok(out_path)
}

/// Build the comparison rrd and open it in a locally-installed `rerun` viewer if present.
#[allow(clippy::too_many_arguments)]
pub fn build_and_open_rrd(
    connection: &Connection,
    db_path: &Path,
    lidar_stream: &str,
    odom_stream: &str,
    tag_stream: &str,
    world_frame: &str,
    camera_stream: &str,
    camera_info_stream: &str,
) -> Result<(), String> {
    println!("building comparison rrd...");
    let rrd_path = build(
        connection,
        db_path,
        lidar_stream,
        odom_stream,
        tag_stream,
        world_frame,
        camera_stream,
        camera_info_stream,
    )?;
    match Command::new("rerun").arg(&rrd_path).spawn() {
        Ok(_) => println!("opened {}", rrd_path.display()),
        Err(_) => println!(
            "rerun not found on PATH; open manually: rerun {}",
            rrd_path.display()
        ),
    }
    Ok(())
}
