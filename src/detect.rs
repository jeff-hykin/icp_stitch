//! AprilTag detection over a memory2 color stream (port of the detect side of
//! dimos `apriltags.py` + `marker_pose.py`), on the pure-Rust kornia detector.
//!
//! The python path is cv2 aruco `detectMarkers` + `solvePnP(IPPE_SQUARE)`; here
//! kornia-apriltag finds the quads and a small planar PnP (homography init,
//! Gauss-Newton refine, plus the reflected-normal second IPPE candidate)
//! recovers the same camera<-marker pose against the same y-up marker frame.

use std::collections::HashMap;

use kornia_apriltag::family::TagFamilyKind;
use kornia_apriltag::{AprilTagDecoder, DecodeTagsConfig};
use kornia_image::allocator::CpuAllocator;
use kornia_image::{Image as KorniaImage, ImageSize};
use lcm_msgs::sensor_msgs::{CameraInfo, Image as LcmImage};
use lcm_msgs::{geometry_msgs, std_msgs};
use rusqlite::Connection;
use rusqlite::types::Value;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;

use crate::apriltags::{Detection, view_quality};
use crate::mat3::{self, Mat3, Vec3};
use crate::memory2;

const POSE_STAMPED_MODULE: &str = "dimos.msgs.geometry_msgs.PoseStamped.PoseStamped";
const UNDISTORT_ITERATIONS: usize = 5;

/// Pinhole + OpenCV radtan/rational distortion, from a CameraInfo K and D.
#[derive(Clone, Debug)]
pub struct CameraModel {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
    /// OpenCV order: k1, k2, p1, p2, k3, k4, k5, k6 (zero-padded).
    pub dist: [f64; 8],
}

impl CameraModel {
    pub fn from_info(k: &[f64; 9], d: &[f64]) -> CameraModel {
        let mut dist = [0.0; 8];
        for (slot, value) in dist.iter_mut().zip(d.iter()) {
            *slot = *value;
        }
        CameraModel {
            fx: k[0],
            skew: k[1],
            cx: k[2],
            fy: k[4],
            cy: k[5],
            dist,
        }
    }

    /// Pixel -> normalized image coordinates, inverting distortion the way
    /// `cv2.undistortPoints` does (fixed-count compensation iterations).
    pub fn undistort(&self, pixel: &[f64; 2]) -> [f64; 2] {
        let y0 = (pixel[1] - self.cy) / self.fy;
        let x0 = (pixel[0] - self.cx - self.skew * y0) / self.fx;
        let [k1, k2, p1, p2, k3, k4, k5, k6] = self.dist;
        let (mut x, mut y) = (x0, y0);
        for _ in 0..UNDISTORT_ITERATIONS {
            let r2 = x * x + y * y;
            let icdist = (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2)
                / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2);
            let delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            x = (x0 - delta_x) * icdist;
            y = (y0 - delta_y) * icdist;
        }
        [x, y]
    }

    /// Camera-frame point -> distorted pixel (`cv2.projectPoints` math).
    pub fn project(&self, point: &Vec3) -> [f64; 2] {
        let x = point[0] / point[2];
        let y = point[1] / point[2];
        let [k1, k2, p1, p2, k3, k4, k5, k6] = self.dist;
        let r2 = x * x + y * y;
        let radial = (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2)
            / (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2);
        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        [
            self.fx * xd + self.skew * yd + self.cx,
            self.fy * yd + self.cy,
        ]
    }
}

fn quote_ident(name: &str) -> Result<String, String> {
    if name.contains('"') {
        return Err(format!("illegal stream name {name:?}"));
    }
    Ok(format!("\"{name}\""))
}

/// First CameraInfo of `stream_name` as `(model, optical frame_id)`, or None
/// when the stream is absent or empty (python `read_camera_info`).
pub fn read_camera_info(
    connection: &Connection,
    stream_name: &str,
) -> Result<Option<(CameraModel, String)>, String> {
    if !memory2::list_streams(connection)?.iter().any(|s| s == stream_name) {
        return Ok(None);
    }
    let table = quote_ident(stream_name)?;
    let blob_table = quote_ident(&format!("{stream_name}_blob"))?;
    let sql = format!(
        "SELECT blob.data FROM {table} AS meta \
         JOIN {blob_table} AS blob ON meta.id = blob.id ORDER BY meta.ts LIMIT 1"
    );
    let mut statement = connection.prepare(&sql).map_err(|e| e.to_string())?;
    let mut rows = statement.query([]).map_err(|e| e.to_string())?;
    let Some(row) = rows.next().map_err(|e| e.to_string())? else {
        return Ok(None);
    };
    let data: Vec<u8> = row.get(0).map_err(|e| e.to_string())?;
    let info = CameraInfo::decode(&data).map_err(|e| e.to_string())?;
    Ok(Some((
        CameraModel::from_info(&info.K, &info.D),
        info.header.frame_id,
    )))
}

/// `(camera info, stream names tried)`. Without an override, rigs that name
/// their intrinsics after the image stream (`<camera>_camera_info`) resolve
/// before the generic `camera_info`.
pub fn resolve_camera_info(
    connection: &Connection,
    camera_stream: &str,
    override_stream: &str,
) -> Result<(Option<(CameraModel, String)>, Vec<String>), String> {
    let tried: Vec<String> = if override_stream.is_empty() {
        vec![format!("{camera_stream}_camera_info"), "camera_info".to_string()]
    } else {
        vec![override_stream.to_string()]
    };
    for stream_name in &tried {
        if let Some(found) = read_camera_info(connection, stream_name)? {
            return Ok((Some(found), tried));
        }
    }
    Ok((None, tried))
}

// ---- planar square PnP ---------------------------------------------------------

fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Homography h (h33 = 1) mapping `object` plane coords to `image` points,
/// from exactly 4 correspondences (direct 8x8 DLT solve).
fn homography_4pt(object: &[[f64; 2]; 4], image: &[[f64; 2]; 4]) -> Option<[[f64; 3]; 3]> {
    let mut a = nalgebra::SMatrix::<f64, 8, 8>::zeros();
    let mut b = nalgebra::SVector::<f64, 8>::zeros();
    for k in 0..4 {
        let [ox, oy] = object[k];
        let [u, v] = image[k];
        let row = 2 * k;
        a[(row, 0)] = ox;
        a[(row, 1)] = oy;
        a[(row, 2)] = 1.0;
        a[(row, 6)] = -u * ox;
        a[(row, 7)] = -u * oy;
        b[row] = u;
        a[(row + 1, 3)] = ox;
        a[(row + 1, 4)] = oy;
        a[(row + 1, 5)] = 1.0;
        a[(row + 1, 6)] = -v * ox;
        a[(row + 1, 7)] = -v * oy;
        b[row + 1] = v;
    }
    let h = a.lu().solve(&b)?;
    Some([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ])
}

/// Nearest rotation matrix (Frobenius) via SVD, det +1.
fn orthonormalize(m: &Mat3) -> Mat3 {
    let matrix = nalgebra::Matrix3::from_fn(|i, j| m[i][j]);
    let svd = matrix.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let mut d = nalgebra::Matrix3::identity();
    d[(2, 2)] = (u * v_t).determinant().signum();
    let r = u * d * v_t;
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = r[(i, j)];
        }
    }
    out
}

/// Initial (R, t) from a plane->normalized-image homography.
fn pose_from_homography(h: &[[f64; 3]; 3]) -> (Mat3, Vec3) {
    let h1 = [h[0][0], h[1][0], h[2][0]];
    let h2 = [h[0][1], h[1][1], h[2][1]];
    let h3 = [h[0][2], h[1][2], h[2][2]];
    let scale = 2.0 / (mat3::norm(&h1) + mat3::norm(&h2));
    let sign = if h3[2] * scale < 0.0 { -1.0 } else { 1.0 };
    let r1 = [h1[0] * scale * sign, h1[1] * scale * sign, h1[2] * scale * sign];
    let r2 = [h2[0] * scale * sign, h2[1] * scale * sign, h2[2] * scale * sign];
    let translation = [h3[0] * scale * sign, h3[1] * scale * sign, h3[2] * scale * sign];
    let r3 = cross(&r1, &r2);
    let rotation = orthonormalize(&[
        [r1[0], r2[0], r3[0]],
        [r1[1], r2[1], r3[1]],
        [r1[2], r2[2], r3[2]],
    ]);
    (rotation, translation)
}

/// The second IPPE candidate: the tag plane's normal reflected across the line
/// of sight to the tag center (the classic planar-pose ambiguity).
fn reflected_pose(rotation: &Mat3, translation: &Vec3) -> Option<(Mat3, Vec3)> {
    let distance = mat3::norm(translation);
    if distance < 1e-9 {
        return None;
    }
    let sight = [
        translation[0] / distance,
        translation[1] / distance,
        translation[2] / distance,
    ];
    let normal = [rotation[0][2], rotation[1][2], rotation[2][2]];
    let along = dot(&sight, &normal);
    let mirrored = [
        2.0 * along * sight[0] - normal[0],
        2.0 * along * sight[1] - normal[1],
        2.0 * along * sight[2] - normal[2],
    ];
    let axis = cross(&normal, &mirrored);
    let axis_norm = mat3::norm(&axis);
    if axis_norm < 1e-12 {
        return None;
    }
    let angle = dot(&normal, &mirrored).clamp(-1.0, 1.0).acos();
    let omega = [
        axis[0] / axis_norm * angle,
        axis[1] / axis_norm * angle,
        axis[2] / axis_norm * angle,
    ];
    Some((mat3::mat_mul(&crate::se3::so3_exp(&omega), rotation), *translation))
}

fn reprojection_rms(
    camera: &CameraModel,
    object: &[Vec3; 4],
    pixels: &[[f64; 2]; 4],
    rotation: &Mat3,
    translation: &Vec3,
) -> f64 {
    let mut total = 0.0;
    for (point, pixel) in object.iter().zip(pixels) {
        let cam = mat3::add(&mat3::mat_vec(rotation, point), translation);
        let projected = camera.project(&cam);
        let dx = projected[0] - pixel[0];
        let dy = projected[1] - pixel[1];
        total += dx * dx + dy * dy;
    }
    (total / 4.0).sqrt()
}

/// Levenberg-damped Gauss-Newton over [omega, delta_t] (left perturbation),
/// minimizing pixel reprojection with the full distortion model.
fn refine_pose(
    camera: &CameraModel,
    object: &[Vec3; 4],
    pixels: &[[f64; 2]; 4],
    rotation: &Mat3,
    translation: &Vec3,
) -> (Mat3, Vec3) {
    let mut rotation = *rotation;
    let mut translation = *translation;
    let residuals = |r: &Mat3, t: &Vec3| -> [f64; 8] {
        let mut out = [0.0; 8];
        for (k, (point, pixel)) in object.iter().zip(pixels).enumerate() {
            let cam = mat3::add(&mat3::mat_vec(r, point), t);
            let projected = camera.project(&cam);
            out[2 * k] = projected[0] - pixel[0];
            out[2 * k + 1] = projected[1] - pixel[1];
        }
        out
    };
    let step = 1e-6;
    for _ in 0..20 {
        let base = residuals(&rotation, &translation);
        let mut jacobian = nalgebra::SMatrix::<f64, 8, 6>::zeros();
        for parameter in 0..6 {
            let mut delta = [0.0; 6];
            delta[parameter] = step;
            let omega = [delta[0], delta[1], delta[2]];
            let plus_rotation = mat3::mat_mul(&crate::se3::so3_exp(&omega), &rotation);
            let plus_translation = [
                translation[0] + delta[3],
                translation[1] + delta[4],
                translation[2] + delta[5],
            ];
            let plus = residuals(&plus_rotation, &plus_translation);
            for row in 0..8 {
                jacobian[(row, parameter)] = (plus[row] - base[row]) / step;
            }
        }
        let residual_vec = nalgebra::SVector::<f64, 8>::from_row_slice(&base);
        let normal = jacobian.transpose() * jacobian
            + nalgebra::SMatrix::<f64, 6, 6>::identity() * 1e-9;
        let gradient = jacobian.transpose() * residual_vec;
        let Some(update) = normal.lu().solve(&(-gradient)) else {
            break;
        };
        let omega = [update[0], update[1], update[2]];
        rotation = mat3::mat_mul(&crate::se3::so3_exp(&omega), &rotation);
        translation = [
            translation[0] + update[3],
            translation[1] + update[4],
            translation[2] + update[5],
        ];
        if update.norm() < 1e-12 {
            break;
        }
    }
    (rotation, translation)
}

/// camera<-marker pose of a square tag from its 4 image corners, in the same
/// y-up marker frame as cv2 aruco / `solvePnP(IPPE_SQUARE)`. `corners` are the
/// oriented pixel corners in aruco order: TL, TR, BR, BL of the canonical tag.
/// Returns `(rotation, translation, reprojection RMS px)`.
pub fn solve_square_pnp(
    camera: &CameraModel,
    corners: &[[f64; 2]; 4],
    marker_length: f64,
) -> Option<(Mat3, Vec3, f64)> {
    let half = marker_length / 2.0;
    let object_2d = [[-half, half], [half, half], [half, -half], [-half, -half]];
    let object = [
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ];
    let normalized = [
        camera.undistort(&corners[0]),
        camera.undistort(&corners[1]),
        camera.undistort(&corners[2]),
        camera.undistort(&corners[3]),
    ];
    let homography = homography_4pt(&object_2d, &normalized)?;
    let (rotation, translation) = pose_from_homography(&homography);
    let mut best: Option<(Mat3, Vec3, f64)> = None;
    let mut candidates = vec![(rotation, translation)];
    if let Some(alternate) = reflected_pose(&rotation, &translation) {
        candidates.push(alternate);
    }
    for (candidate_rotation, candidate_translation) in candidates {
        let (refined_rotation, refined_translation) =
            refine_pose(camera, &object, corners, &candidate_rotation, &candidate_translation);
        if refined_translation[2] <= 0.0 {
            continue;
        }
        let rms = reprojection_rms(camera, &object, corners, &refined_rotation, &refined_translation);
        if best.as_ref().is_none_or(|(_, _, best_rms)| rms < *best_rms) {
            best = Some((refined_rotation, refined_translation, rms));
        }
    }
    best
}

// ---- image decode + glimpse diagnostics ----------------------------------------

fn gray_from_channels(data: &[u8], channels: usize, weights: [f64; 3]) -> Vec<u8> {
    data.chunks_exact(channels)
        .map(|px| {
            (weights[0] * px[0] as f64 + weights[1] * px[1] as f64 + weights[2] * px[2] as f64)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

const REC601: [f64; 3] = [0.299, 0.587, 0.114];
const REC601_BGR: [f64; 3] = [0.114, 0.587, 0.299];

/// Grayscale pixels + dimensions from an lcm Image (jpeg-compressed or raw).
pub fn decode_gray(image: &LcmImage) -> Result<(Vec<u8>, usize, usize), String> {
    if image.encoding == "jpeg" {
        let mut decoder = JpegDecoder::new(ZCursor::new(&image.data[..]));
        let pixels = decoder.decode().map_err(|e| e.to_string())?;
        let info = decoder.info().ok_or("jpeg missing header info")?;
        let (width, height) = (info.width as usize, info.height as usize);
        let colorspace = decoder
            .output_colorspace()
            .ok_or("jpeg missing colorspace")?;
        let channels = colorspace.num_components();
        let gray = if channels == 1 {
            pixels
        } else {
            gray_from_channels(&pixels, channels, REC601)
        };
        return Ok((gray, width, height));
    }
    let width = image.width as usize;
    let height = image.height as usize;
    let step = image.step as usize;
    let per_pixel = match image.encoding.as_str() {
        "mono8" | "8UC1" => 1,
        "rgb8" | "bgr8" => 3,
        "rgba8" | "bgra8" => 4,
        other => return Err(format!("unsupported image encoding {other:?}")),
    };
    if image.data.len() < step * height || step < width * per_pixel {
        return Err(format!(
            "image buffer too small: {} bytes for {width}x{height} {}",
            image.data.len(),
            image.encoding
        ));
    }
    let weights = if image.encoding.starts_with("bgr") {
        REC601_BGR
    } else {
        REC601
    };
    let mut gray = Vec::with_capacity(width * height);
    for row in 0..height {
        let line = &image.data[row * step..row * step + width * per_pixel];
        if per_pixel == 1 {
            gray.extend_from_slice(line);
        } else {
            gray.extend(gray_from_channels(line, per_pixel, weights));
        }
    }
    Ok((gray, width, height))
}

/// Tag side length in pixels: sqrt of the quad's (shoelace) image area.
pub fn tag_pixel_size(corners: &[[f64; 2]; 4]) -> f64 {
    let mut doubled_area = 0.0;
    for k in 0..4 {
        let [x0, y0] = corners[k];
        let [x1, y1] = corners[(k + 1) % 4];
        doubled_area += x0 * y1 - x1 * y0;
    }
    (doubled_area / 2.0).abs().sqrt()
}

/// Laplacian (3x3 cross kernel) variance over the tag's bounding box, computed
/// on the crop with reflect-101 borders — cv2's exact recipe; low = motion blur.
pub fn tag_sharpness(gray: &[u8], width: usize, height: usize, corners: &[[f64; 2]; 4]) -> f64 {
    let xs = corners.iter().map(|c| c[0]);
    let ys = corners.iter().map(|c| c[1]);
    let x_min = (xs.clone().fold(f64::INFINITY, f64::min).floor() as i64).max(0) as usize;
    let y_min = (ys.clone().fold(f64::INFINITY, f64::min).floor() as i64).max(0) as usize;
    let x_max = ((xs.fold(f64::NEG_INFINITY, f64::max).ceil() as i64).max(0) as usize).min(width);
    let y_max = ((ys.fold(f64::NEG_INFINITY, f64::max).ceil() as i64).max(0) as usize).min(height);
    if x_max.saturating_sub(x_min) < 2 || y_max.saturating_sub(y_min) < 2 {
        return 0.0;
    }
    let roi_width = (x_max - x_min) as isize;
    let roi_height = (y_max - y_min) as isize;
    let sample = |x: isize, y: isize| -> f64 {
        let x = if x < 0 { -x } else if x >= roi_width { 2 * roi_width - 2 - x } else { x };
        let y = if y < 0 { -y } else if y >= roi_height { 2 * roi_height - 2 - y } else { y };
        gray[(y_min + y as usize) * width + x_min + x as usize] as f64
    };
    let count = (roi_width * roi_height) as f64;
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for y in 0..roi_height {
        for x in 0..roi_width {
            let response = sample(x - 1, y) + sample(x + 1, y) + sample(x, y - 1)
                + sample(x, y + 1)
                - 4.0 * sample(x, y);
            sum += response;
            sum_squares += response * response;
        }
    }
    let mean = sum / count;
    sum_squares / count - mean * mean
}

/// Per-image (linear m/s, angular deg/s) keyed by ts bits, from consecutive
/// posed rows of the image stream's meta table. Second value is false when too
/// few frames carry poses to estimate motion (speed gates disabled).
fn camera_speeds(
    connection: &Connection,
    image_stream: &str,
) -> Result<(HashMap<u64, (f64, f64)>, bool), String> {
    let table = quote_ident(image_stream)?;
    let sql = format!(
        "SELECT ts, pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw \
         FROM {table} ORDER BY ts"
    );
    let mut statement = connection.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = statement
        .query_map([], |row| {
            let ts: f64 = row.get(0)?;
            let mut pose = [0.0; 7];
            let mut complete = true;
            for (index, slot) in pose.iter_mut().enumerate() {
                match row.get::<_, Option<f64>>(index + 1)? {
                    Some(value) => *slot = value,
                    None => complete = false,
                }
            }
            Ok((ts, complete.then_some(pose)))
        })
        .map_err(|e| e.to_string())?;
    let mut posed: Vec<(f64, [f64; 7])> = Vec::new();
    for row in rows {
        let (ts, pose) = row.map_err(|e| e.to_string())?;
        if let Some(pose) = pose {
            posed.push((ts, pose));
        }
    }
    posed.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut speeds = HashMap::new();
    for pair in posed.windows(2) {
        let (ts_a, pose_a) = pair[0];
        let (ts_b, pose_b) = pair[1];
        let dt = ts_b - ts_a;
        if dt <= 0.0 {
            continue;
        }
        let dx = pose_b[0] - pose_a[0];
        let dy = pose_b[1] - pose_a[1];
        let dz = pose_b[2] - pose_a[2];
        let linear = (dx * dx + dy * dy + dz * dz).sqrt() / dt;
        let quat_dot: f64 = (3..7).map(|i| pose_a[i] * pose_b[i]).sum();
        let angular = (2.0 * quat_dot.abs().min(1.0).acos()).to_degrees() / dt;
        speeds.insert(ts_b.to_bits(), (linear, angular));
    }
    Ok((speeds, posed.len() >= 2))
}

fn family_kind(dictionary: &str) -> Result<TagFamilyKind, String> {
    match dictionary {
        "DICT_APRILTAG_36h11" => Ok(TagFamilyKind::Tag36H11),
        "DICT_APRILTAG_36h10" => Ok(TagFamilyKind::Tag36H10),
        "DICT_APRILTAG_25h9" => Ok(TagFamilyKind::Tag25H9),
        "DICT_APRILTAG_16h5" => Ok(TagFamilyKind::Tag16H5),
        other => Err(format!("unsupported aruco dictionary {other:?}")),
    }
}

/// Oriented pixel corners of a kornia detection in aruco order (TL, TR, BR, BL
/// of the canonical tag). `quad.corners` ignore the decoded orientation, but
/// `quad.homography` includes it, so project the canonical tag corners.
fn oriented_corners(detection: &kornia_apriltag::decoder::Detection) -> [[f64; 2]; 4] {
    // C-apriltag detection.p tag coords, y-down bitmap frame.
    let tag_corners = [[-1.0f32, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]];
    let mut out = [[0.0; 2]; 4];
    // p[0]=BL, p[1]=BR, p[2]=TR, p[3]=TL in the y-up marker frame; reverse to
    // aruco TL,TR,BR,BL.
    for (slot, tag_corner) in out.iter_mut().rev().zip(tag_corners) {
        let point = detection.quad.homography_project(tag_corner[0], tag_corner[1]);
        *slot = [point.x as f64, point.y as f64];
    }
    out
}

/// Every valid-PnP tag detection over `image_stream`, unfiltered, with gate
/// diagnostics: `(detections, speed gate available, image count)`.
pub fn detect_raw_detections(
    connection: &Connection,
    camera: &CameraModel,
    image_stream: &str,
    marker_length: f64,
    dictionary: &str,
) -> Result<(Vec<Detection>, bool, usize), String> {
    let kind = family_kind(dictionary)?;
    let (speed_by_ts, speed_available) = camera_speeds(connection, image_stream)?;
    let table = quote_ident(image_stream)?;
    let blob_table = quote_ident(&format!("{image_stream}_blob"))?;
    let sql = format!(
        "SELECT meta.ts, blob.data FROM {table} AS meta \
         JOIN {blob_table} AS blob ON meta.id = blob.id ORDER BY meta.ts"
    );
    let mut statement = connection.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = statement
        .query_map([], |row| {
            let ts: f64 = row.get(0)?;
            let data: Vec<u8> = row.get(1)?;
            Ok((ts, data))
        })
        .map_err(|e| e.to_string())?;
    let mut raw_detections = Vec::new();
    let mut image_count = 0usize;
    let mut decoder: Option<(AprilTagDecoder, ImageSize)> = None;
    for row in rows {
        let (ts, data) = row.map_err(|e| e.to_string())?;
        image_count += 1;
        let Ok(message) = LcmImage::decode(&data) else {
            continue;
        };
        let Ok((gray, width, height)) = decode_gray(&message) else {
            continue;
        };
        let size = ImageSize { width, height };
        let needs_new = decoder
            .as_ref()
            .is_none_or(|(_, existing)| existing.width != width || existing.height != height);
        if needs_new {
            let config = DecodeTagsConfig::new(vec![kind.clone()]);
            decoder = Some((
                AprilTagDecoder::new(config, size).map_err(|e| e.to_string())?,
                size,
            ));
        }
        let (active_decoder, _) = decoder.as_mut().unwrap();
        let kornia_image = KorniaImage::<u8, 1, CpuAllocator>::new(size, gray.clone(), CpuAllocator)
            .map_err(|e| e.to_string())?;
        let found = active_decoder.decode(&kornia_image).map_err(|e| e.to_string())?;
        active_decoder.clear();
        for tag in &found {
            let corners = oriented_corners(tag);
            let Some((rotation, translation, reproj_px)) =
                solve_square_pnp(camera, &corners, marker_length)
            else {
                continue;
            };
            let [qw, qx, qy, qz] = mat3::quat_from_mat(&rotation);
            let t_cam_marker = [
                translation[0],
                translation[1],
                translation[2],
                qx,
                qy,
                qz,
                qw,
            ];
            let (distance_m, view_angle_deg) = view_quality(&t_cam_marker);
            let (lin_speed, ang_speed) =
                speed_by_ts.get(&ts.to_bits()).copied().unwrap_or((-1.0, -1.0));
            raw_detections.push(Detection {
                ts,
                marker_id: tag.id as i64,
                t_cam_marker,
                sharpness: tag_sharpness(&gray, width, height, &corners),
                reproj_px,
                tag_px: tag_pixel_size(&corners),
                distance_m,
                view_angle_deg,
                lin_speed,
                ang_speed,
            });
        }
    }
    Ok((raw_detections, speed_available, image_count))
}

// ---- tag stream persistence ----------------------------------------------------

fn sec_nsec(ts: f64) -> (i32, i32) {
    let seconds = ts as i64;
    (seconds as i32, ((ts - seconds as f64) * 1_000_000_000.0) as i32)
}

fn encode_pose_stamped(ts: f64, pose: &[f64; 7]) -> Vec<u8> {
    let (sec, nsec) = sec_nsec(ts);
    lcm_msgs::geometry_msgs::PoseStamped {
        header: std_msgs::Header {
            seq: 0,
            stamp: std_msgs::Time { sec, nsec },
            frame_id: String::new(),
        },
        pose: geometry_msgs::Pose {
            position: geometry_msgs::Point {
                x: pose[0],
                y: pose[1],
                z: pose[2],
            },
            orientation: geometry_msgs::Quaternion {
                x: pose[3],
                y: pose[4],
                z: pose[5],
                w: pose[6],
            },
        },
    }
    .encode()
}

/// (Re)write a PoseStamped tag stream, with the numeric gate-diagnostic tags
/// when `diagnostics` (python `_write_tag_stream`).
pub fn write_tag_stream(
    connection: &Connection,
    stream_name: &str,
    detections: &[Detection],
    diagnostics: bool,
) -> Result<(), String> {
    if memory2::list_streams(connection)?.iter().any(|s| s == stream_name) {
        memory2::delete_stream(connection, stream_name)?;
    }
    memory2::create_stream(connection, stream_name, POSE_STAMPED_MODULE)?;
    for detection in detections {
        let pose = detection.t_cam_marker;
        let mut tags: Vec<(&str, String)> = vec![("marker_id", detection.marker_id.to_string())];
        if diagnostics {
            let (distance, view_angle) = view_quality(&pose);
            let speed_known = detection.lin_speed >= 0.0;
            tags.extend([
                ("sharpness", format!("{:?}", detection.sharpness)),
                ("reproj_px", format!("{:?}", detection.reproj_px)),
                ("tag_px", format!("{:?}", detection.tag_px)),
                ("distance_m", format!("{distance:?}")),
                ("view_angle_deg", format!("{view_angle:?}")),
                ("lin_speed", format!("{:?}", if speed_known { detection.lin_speed } else { -1.0 })),
                ("ang_speed", format!("{:?}", if speed_known { detection.ang_speed } else { -1.0 })),
            ]);
        }
        memory2::append_json_tags(
            connection,
            stream_name,
            detection.ts,
            Some(pose),
            &tags,
            &encode_pose_stamped(detection.ts, &pose),
        )?;
    }
    Ok(())
}

fn tag_number(value: Value, key: &str) -> Result<Option<f64>, String> {
    match value {
        Value::Null => Ok(None),
        Value::Integer(number) => Ok(Some(number as f64)),
        Value::Real(number) => Ok(Some(number)),
        Value::Text(text) => text
            .parse::<f64>()
            .map(Some)
            .map_err(|_| format!("non-numeric tag {key}={text:?}")),
        Value::Blob(_) => Err(format!("unexpected blob tag {key}")),
    }
}

/// Read a persisted raw tag stream back into `Detection`s
/// (python `read_raw_tag_stream`).
pub fn read_raw_tag_stream(
    connection: &Connection,
    stream_name: &str,
) -> Result<Vec<Detection>, String> {
    let table = quote_ident(stream_name)?;
    let blob_table = quote_ident(&format!("{stream_name}_blob"))?;
    let keys = ["marker_id", "sharpness", "reproj_px", "tag_px", "lin_speed", "ang_speed"];
    let tag_columns: Vec<String> = keys
        .iter()
        .map(|key| format!("json_extract(meta.tags, '$.{key}')"))
        .collect();
    let sql = format!(
        "SELECT meta.ts, blob.data, {} FROM {table} AS meta \
         JOIN {blob_table} AS blob ON meta.id = blob.id ORDER BY meta.ts",
        tag_columns.join(", ")
    );
    let mut statement = connection.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = statement
        .query_map([], |row| {
            let ts: f64 = row.get(0)?;
            let data: Vec<u8> = row.get(1)?;
            let mut tag_values = Vec::with_capacity(keys.len());
            for index in 0..keys.len() {
                tag_values.push(row.get::<_, Value>(index + 2)?);
            }
            Ok((ts, data, tag_values))
        })
        .map_err(|e| e.to_string())?;
    let mut detections = Vec::new();
    for row in rows {
        let (ts, data, tag_values) = row.map_err(|e| e.to_string())?;
        let message = lcm_msgs::geometry_msgs::PoseStamped::decode(&data)
            .map_err(|e| format!("{stream_name}: {e}"))?;
        let mut numbers = [None; 6];
        for ((slot, value), key) in numbers.iter_mut().zip(tag_values).zip(keys) {
            *slot = tag_number(value, key)?;
        }
        let required = |index: usize| {
            numbers[index].ok_or_else(|| format!("{stream_name}: missing tag {:?}", keys[index]))
        };
        let lin_speed = numbers[4].unwrap_or(-1.0);
        let ang_speed = numbers[5].unwrap_or(-1.0);
        let t_cam_marker = [
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ];
        let (distance_m, view_angle_deg) = view_quality(&t_cam_marker);
        detections.push(Detection {
            ts,
            marker_id: required(0)? as i64,
            t_cam_marker,
            sharpness: required(1)?,
            reproj_px: required(2)?,
            tag_px: required(3)?,
            distance_m,
            view_angle_deg,
            lin_speed: if lin_speed < 0.0 { -1.0 } else { lin_speed },
            ang_speed: if lin_speed < 0.0 { -1.0 } else { ang_speed },
        });
    }
    Ok(detections)
}

/// Detect + persist the unfiltered tag stream when absent; returns whether it
/// exists (python `ensure_raw_tag_stream`).
pub fn ensure_raw_tag_stream(
    connection: &Connection,
    camera: Option<&CameraModel>,
    raw_stream: &str,
    image_stream: &str,
    marker_length: f64,
    dictionary: &str,
) -> Result<bool, String> {
    let streams = memory2::list_streams(connection)?;
    if streams.iter().any(|s| s == raw_stream) {
        return Ok(true);
    }
    let Some(camera) = camera else {
        return Ok(false);
    };
    if !streams.iter().any(|s| s == image_stream) {
        return Ok(false);
    }
    println!(
        "{raw_stream} missing -- detecting AprilTags over {image_stream} \
         (tag_size={marker_length:?} m, dict={dictionary})..."
    );
    let (raw_detections, _, image_count) =
        detect_raw_detections(connection, camera, image_stream, marker_length, dictionary)?;
    write_tag_stream(connection, raw_stream, &raw_detections, true)?;
    println!(
        "wrote {raw_stream}: {} detections over {image_count} images",
        raw_detections.len()
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_apriltag::family::TagFamily;

    fn test_camera() -> CameraModel {
        CameraModel {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
            dist: [0.0; 8],
        }
    }

    #[test]
    fn undistort_project_roundtrip() {
        let mut camera = test_camera();
        camera.dist = [-0.28, 0.07, 0.001, -0.0005, 0.0, 0.0, 0.0, 0.0];
        for pixel in [[100.0, 80.0], [320.0, 240.0], [500.0, 400.0]] {
            let [x, y] = camera.undistort(&pixel);
            let reprojected = camera.project(&[x, y, 1.0]);
            assert!((reprojected[0] - pixel[0]).abs() < 1e-3, "{pixel:?} -> {reprojected:?}");
            assert!((reprojected[1] - pixel[1]).abs() < 1e-3);
        }
    }

    /// Intensity of the tag36h11 bitmap for `id` at tag coords (y-down, border
    /// at ±1, white margin to ±1.25); None = outside the tag (background).
    fn tag_intensity(family: &TagFamily, id: usize, tag_x: f64, tag_y: f64) -> Option<u8> {
        let margin = family.total_width as f64 / family.width_at_border as f64;
        if tag_x.abs() > margin || tag_y.abs() > margin {
            return None;
        }
        if tag_x.abs() > 1.0 || tag_y.abs() > 1.0 {
            return Some(255);
        }
        let cells = family.width_at_border as f64;
        let cell_x = ((tag_x / 2.0 + 0.5) * cells).floor() as i64;
        let cell_y = ((tag_y / 2.0 + 0.5) * cells).floor() as i64;
        let cell_x = cell_x.clamp(0, family.width_at_border as i64 - 1) as i8;
        let cell_y = cell_y.clamp(0, family.width_at_border as i64 - 1) as i8;
        let border = family.width_at_border as i8 - 1;
        if cell_x == 0 || cell_y == 0 || cell_x == border || cell_y == border {
            return Some(0);
        }
        let code = family.code_data[id];
        for bit in 0..family.nbits {
            if family.bit_x[bit] == cell_x && family.bit_y[bit] == cell_y {
                let set = (code >> (family.nbits - 1 - bit)) & 1 == 1;
                return Some(if set { 255 } else { 0 });
            }
        }
        Some(0)
    }

    /// Render the tag at a camera<-tag pose (aruco y-up marker frame) through
    /// the pinhole camera, supersampled 3x3.
    fn render_tag(
        family: &TagFamily,
        id: usize,
        camera: &CameraModel,
        rotation: &Mat3,
        translation: &Vec3,
        width: usize,
        height: usize,
        marker_length: f64,
    ) -> Vec<u8> {
        let half = marker_length / 2.0;
        let rotation_t = mat3::transpose(rotation);
        let normal = [rotation[0][2], rotation[1][2], rotation[2][2]];
        let plane_offset = dot(translation, &normal);
        let mut pixels = vec![0u8; width * height];
        let offsets = [-1.0 / 3.0, 0.0, 1.0 / 3.0];
        for pixel_y in 0..height {
            for pixel_x in 0..width {
                let mut total = 0.0;
                for dy in offsets {
                    for dx in offsets {
                        let ray = [
                            (pixel_x as f64 + dx - camera.cx) / camera.fx,
                            (pixel_y as f64 + dy - camera.cy) / camera.fy,
                            1.0,
                        ];
                        let along = dot(&ray, &normal);
                        let mut value = 128.0;
                        if along.abs() > 1e-9 {
                            let depth = plane_offset / along;
                            if depth > 0.0 {
                                let hit = [
                                    ray[0] * depth - translation[0],
                                    ray[1] * depth - translation[1],
                                    ray[2] * depth - translation[2],
                                ];
                                let local = mat3::mat_vec(&rotation_t, &hit);
                                // aruco y-up -> bitmap y-down
                                let tag_x = local[0] / half;
                                let tag_y = -local[1] / half;
                                if let Some(intensity) = tag_intensity(family, id, tag_x, tag_y) {
                                    value = intensity as f64;
                                }
                            }
                        }
                        total += value;
                    }
                }
                pixels[pixel_y * width + pixel_x] = (total / 9.0).round() as u8;
            }
        }
        pixels
    }

    #[test]
    fn synthetic_tag_detection_recovers_pose() {
        let camera = test_camera();
        let family = TagFamily::tag36_h11();
        let marker_length = 0.1;
        // aruco pose of a camera-facing tag: ~180 deg about x (tag +z toward
        // the viewer), plus a small perturbation
        let omega = [0.15, -0.1, 0.05];
        let facing = crate::se3::so3_exp(&[std::f64::consts::PI, 0.0, 0.0]);
        let rotation = mat3::mat_mul(&crate::se3::so3_exp(&omega), &facing);
        let translation = [0.03, -0.02, 0.5];
        let (width, height) = (640, 480);
        let gray = render_tag(
            &family, 0, &camera, &rotation, &translation, width, height, marker_length,
        );
        let size = ImageSize { width, height };
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]);
        let mut decoder = AprilTagDecoder::new(config, size).unwrap();
        let image = KorniaImage::<u8, 1, CpuAllocator>::new(size, gray.clone(), CpuAllocator).unwrap();
        let found = decoder.decode(&image).unwrap();
        assert_eq!(found.len(), 1, "expected exactly one detection");
        assert_eq!(found[0].id, 0);
        let corners = oriented_corners(&found[0]);
        let (solved_rotation, solved_translation, reproj) =
            solve_square_pnp(&camera, &corners, marker_length).unwrap();
        assert!(reproj < 1.0, "reproj rms {reproj}");
        for axis in 0..3 {
            assert!(
                (solved_translation[axis] - translation[axis]).abs() < 0.005,
                "translation {solved_translation:?} vs {translation:?}"
            );
        }
        let angle_error = mat3::angular_distance(&solved_rotation, &rotation);
        assert!(angle_error < 0.05, "rotation off by {angle_error} rad");
        // diagnostics on the same detection
        assert!(tag_pixel_size(&corners) > 24.0);
        assert!(tag_sharpness(&gray, width, height, &corners) > 60.0);
    }

    #[test]
    fn tag_stream_write_read_roundtrip() {
        let connection = Connection::open_in_memory().unwrap();
        let detections = vec![
            Detection {
                ts: 10.5,
                marker_id: 3,
                t_cam_marker: [0.1, -0.2, 0.7, 0.0, 0.0, 0.0, 1.0],
                sharpness: 120.0,
                reproj_px: 0.4,
                tag_px: 55.0,
                distance_m: 0.0,
                view_angle_deg: 0.0,
                lin_speed: 0.2,
                ang_speed: 12.0,
            },
            Detection {
                ts: 11.0,
                marker_id: 7,
                t_cam_marker: [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                sharpness: 90.0,
                reproj_px: 1.1,
                tag_px: 30.0,
                distance_m: 0.0,
                view_angle_deg: 0.0,
                lin_speed: -1.0,
                ang_speed: -1.0,
            },
        ];
        write_tag_stream(&connection, "raw_april_tags", &detections, true).unwrap();
        // numeric tag types in the stored JSON, python-style
        let marker_type: String = connection
            .query_row(
                "SELECT type FROM (SELECT json_extract(tags, '$.marker_id') AS v, \
                 json_type(tags, '$.marker_id') AS type FROM raw_april_tags LIMIT 1)",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(marker_type, "integer");
        let back = read_raw_tag_stream(&connection, "raw_april_tags").unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].marker_id, 3);
        assert!((back[0].sharpness - 120.0).abs() < 1e-12);
        assert!((back[0].lin_speed - 0.2).abs() < 1e-12);
        assert!((back[0].t_cam_marker[2] - 0.7).abs() < 1e-12);
        assert_eq!(back[1].lin_speed, -1.0);
        assert_eq!(back[1].ang_speed, -1.0);
        // ensure() sees the stream and leaves it alone
        assert!(ensure_raw_tag_stream(&connection, None, "raw_april_tags", "color_image", 0.1, "DICT_APRILTAG_36h11").unwrap());
        // absent stream + no camera -> false
        assert!(!ensure_raw_tag_stream(&connection, None, "other_tags", "color_image", 0.1, "DICT_APRILTAG_36h11").unwrap());
    }
}
