// Copyright 2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Write PGO artifacts back into a recording db: corrected odom/lidar, deformation
//! nodes, pose graph, aggregated .pc2.lcm, and raycast-accumulated maps.
//! Faithful port of `gsc_pgo/utils/artifacts.py`, byte-compatible on the wire
//! (same lcm blobs, same sqlite schema, same print strings).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use gtsam_shim::Pose3;
use lcm_msgs::geometry_msgs;
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::{PointCloud2, PointField};
use lcm_msgs::std_msgs;
use rusqlite::Connection;

use crate::memory2::{self, ScanRow};
use crate::msgs::{DeformationNode, Edge, Graph3D, Node3D, PoseStamped, tf_id_for};
use crate::pgo::OdomPoseRow;
use crate::pointcloud::KdTree;
use crate::se3;
use crate::tf::RecordingTf;
use crate::voxel_ray_tracer::{self, Config as RayConfig, VoxelMap};

// aggregated .pc2.lcm
const LCM_CHUNK_SCANS: usize = 1000; // collapse buffered scans this often to bound memory
const LCM_OUTLIER_NN: usize = 20; // statistical outlier removal: neighbor count
const LCM_OUTLIER_STD: f64 = 2.0; // ...and std-ratio threshold (lower = more aggressive)

// progress logging cadence
const ODOM_LOG_EVERY: usize = 20000;
const SCAN_LOG_EVERY: usize = 2000;

pub(crate) const ODOMETRY_MODULE: &str = "dimos.msgs.nav_msgs.Odometry.Odometry";
pub(crate) const POINTCLOUD2_MODULE: &str = "dimos.msgs.sensor_msgs.PointCloud2.PointCloud2";
const GRAPH3D_MODULE: &str = "dimos.navigation.jnav.msgs.Graph3D.Graph3D";
const DEFORMATION_NODE_MODULE: &str = "dimos.navigation.jnav.msgs.DeformationNode.DeformationNode";

fn format_thousands(value: usize) -> String {
    // python's f"{value:,}"
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// SE(3) interpolation of the keyframe corrections at an arbitrary timestamp.
pub fn interpolate_correction(keyframe_times: &[f64], corrections: &[Pose3], ts: f64) -> Pose3 {
    if ts <= keyframe_times[0] {
        return corrections[0];
    }
    if ts >= keyframe_times[keyframe_times.len() - 1] {
        return corrections[corrections.len() - 1];
    }
    let after = keyframe_times.partition_point(|&t| t < ts);
    let before = after - 1;
    let alpha = (ts - keyframe_times[before]) / (keyframe_times[after] - keyframe_times[before]);
    let delta = se3::logmap(&se3::between(&corrections[before], &corrections[after]));
    let scaled = [
        alpha * delta[0],
        alpha * delta[1],
        alpha * delta[2],
        alpha * delta[3],
        alpha * delta[4],
        alpha * delta[5],
    ];
    se3::compose(&corrections[before], &se3::expmap(&scaled))
}

/// Index of the sample in sorted `times` closest to `ts` (trajectory_metrics'
/// `nearest_index`: searchsorted, clamp, step back if the previous is closer).
pub fn nearest_index(times: &[f64], ts: f64) -> usize {
    let mut index = times.partition_point(|&t| t < ts);
    if index >= times.len() {
        index = times.len() - 1;
    }
    if index > 0 && (times[index - 1] - ts).abs() < (times[index] - ts).abs() {
        index -= 1;
    }
    index
}

fn ros_stamp(ts: f64) -> std_msgs::Time {
    std_msgs::Time {
        sec: ts as i32,
        nsec: ((ts - ts.trunc()) * 1e9) as i32,
    }
}

fn xyz_fields() -> Vec<PointField> {
    ["x", "y", "z", "intensity"]
        .iter()
        .enumerate()
        .map(|(index, name)| PointField {
            name: name.to_string(),
            offset: (index * 4) as i32,
            datatype: 7, // FLOAT32
            count: 1,
        })
        .collect()
}

/// `PointCloud2.from_numpy(...).lcm_encode()`: x/y/z/intensity float32 rows,
/// intensity zeros when the cloud carries none.
pub fn encode_pointcloud2(
    points: &[[f32; 3]],
    intensities: Option<&[f32]>,
    frame_id: &str,
    ts: f64,
) -> Vec<u8> {
    let mut message = PointCloud2 {
        header: std_msgs::Header {
            seq: 0,
            stamp: ros_stamp(ts),
            frame_id: frame_id.to_string(),
        },
        fields: xyz_fields(),
        is_bigendian: false,
        is_dense: true,
        point_step: 16,
        ..Default::default()
    };
    if points.is_empty() {
        message.height = 0;
        message.width = 0;
        message.row_step = 0;
        return message.encode();
    }
    message.height = 1;
    message.width = points.len() as i32;
    message.row_step = 16 * message.width;
    let mut data = Vec::with_capacity(points.len() * 16);
    for (index, point) in points.iter().enumerate() {
        for value in point {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let intensity = intensities.map_or(0.0, |values| values[index]);
        data.extend_from_slice(&intensity.to_le_bytes());
    }
    message.data = data;
    message.encode()
}

pub(crate) fn encode_odometry(
    ts: f64,
    frame_id: &str,
    child_frame_id: &str,
    pose: &[f64; 7],
) -> Vec<u8> {
    let [x, y, z, qx, qy, qz, qw] = *pose;
    let mut message = Odometry {
        header: std_msgs::Header {
            seq: 0,
            stamp: ros_stamp(ts),
            frame_id: frame_id.to_string(),
        },
        child_frame_id: child_frame_id.to_string(),
        ..Default::default()
    };
    message.pose.pose = geometry_msgs::Pose {
        position: geometry_msgs::Point { x, y, z },
        orientation: geometry_msgs::Quaternion {
            x: qx,
            y: qy,
            z: qz,
            w: qw,
        },
    };
    message.encode()
}

/// Per keyframe: the raw pose then the optimized pose, so tf.get can replay the correction.
pub fn write_deformation_nodes(
    connection: &Connection,
    name: &str,
    keyframe_times: &[f64],
    raw_poses: &[Pose3],
    optimized: &[Pose3],
    world_frame: &str,
    body_frame: &str,
) -> Result<(), String> {
    if memory2::list_streams(connection)?.iter().any(|s| s == name) {
        memory2::delete_stream(connection, name)?;
    }
    memory2::create_stream(connection, name, DEFORMATION_NODE_MODULE)?;
    let edge_id = tf_id_for(world_frame, body_frame);
    for index in 0..keyframe_times.len() {
        let node_ts = keyframe_times[index];
        for pose in [&raw_poses[index], &optimized[index]] {
            let tuple = se3::pose_tuple(pose);
            let node = DeformationNode {
                id: index as u64,
                tf_id: edge_id,
                pose: PoseStamped {
                    ts: node_ts,
                    frame_id: world_frame.to_string(),
                    position: [tuple[0], tuple[1], tuple[2]],
                    orientation: [tuple[3], tuple[4], tuple[5], tuple[6]],
                },
            };
            memory2::append(
                connection,
                name,
                node_ts,
                None,
                &[
                    ("tf_id", edge_id.to_string()),
                    ("id", index.to_string()),
                ],
                &node.encode(),
            )?;
        }
    }
    println!("wrote {name}: {} keyframes (raw+optimized)", keyframe_times.len());
    Ok(())
}

/// The optimized keyframe nodes + sequential odom edges as a Graph3D.
pub fn write_pose_graph(
    connection: &Connection,
    name: &str,
    keyframe_times: &[f64],
    optimized: &[Pose3],
    world_frame: &str,
) -> Result<(), String> {
    if memory2::list_streams(connection)?.iter().any(|s| s == name) {
        memory2::delete_stream(connection, name)?;
    }
    memory2::create_stream(connection, name, GRAPH3D_MODULE)?;
    let num_keyframes = keyframe_times.len();
    let nodes: Vec<Node3D> = (0..num_keyframes)
        .map(|index| {
            let tuple = se3::pose_tuple(&optimized[index]);
            Node3D {
                pose: PoseStamped {
                    ts: keyframe_times[index],
                    frame_id: world_frame.to_string(),
                    position: [tuple[0], tuple[1], tuple[2]],
                    orientation: [tuple[3], tuple[4], tuple[5], tuple[6]],
                },
                id: index as u64,
                metadata_id: 0,
            }
        })
        .collect();
    let edges: Vec<Edge> = (0..num_keyframes.saturating_sub(1))
        .map(|index| Edge {
            start_id: index as u64,
            end_id: (index + 1) as u64,
            timestamp: keyframe_times[index + 1],
            metadata_id: 0,
        })
        .collect();
    let graph_ts = keyframe_times[num_keyframes - 1];
    let graph = Graph3D {
        ts: graph_ts,
        nodes,
        edges,
    };
    let edge_count = graph.edges.len();
    memory2::append(connection, name, graph_ts, None, &[], &graph.encode())?;
    println!("wrote {name}: {num_keyframes} nodes, {edge_count} edges");
    Ok(())
}

/// Corrected trajectory as `world_frame -> corrected_odom_frame` odometry, i.e. the tf
/// edge the per-scan corrected clouds hang on.
pub fn write_corrected_odom(
    connection: &Connection,
    name: &str,
    odom_rows: &[OdomPoseRow],
    keyframe_times: &[f64],
    corrections: &[Pose3],
    world_frame: &str,
    corrected_odom_frame: &str,
) -> Result<(), String> {
    if memory2::list_streams(connection)?.iter().any(|s| s == name) {
        memory2::delete_stream(connection, name)?;
    }
    memory2::create_stream(connection, name, ODOMETRY_MODULE)?;
    println!("writing {name} ({} poses)...", odom_rows.len());
    let started = Instant::now();
    for (count, row) in odom_rows.iter().enumerate() {
        let ts = row[0];
        let raw = se3::from_xyzquat(&row[1..]);
        let corrected =
            se3::compose(&interpolate_correction(keyframe_times, corrections, ts), &raw);
        let tuple = se3::pose_tuple(&corrected);
        let blob = encode_odometry(ts, world_frame, corrected_odom_frame, &tuple);
        memory2::append(connection, name, ts, Some(tuple), &[], &blob)?;
        if (count + 1) % ODOM_LOG_EVERY == 0 {
            println!(
                "  {}/{} poses, {:.0}s",
                count + 1,
                odom_rows.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "wrote {name}: {} poses in {:.0}s",
        odom_rows.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Merge chunks, voxel-downsample with open3d semantics (cells relative to the
/// min bound, per-cell centroid), averaging intensity like o3d's color channel.
fn voxel_downsample_chunks(
    points_chunks: &[Vec<[f32; 3]>],
    intensity_chunks: &[Vec<f32>],
    voxel: f64,
) -> (Vec<[f32; 3]>, Option<Vec<f32>>) {
    let carry = !intensity_chunks.is_empty();
    let points: Vec<[f64; 3]> = points_chunks
        .iter()
        .flatten()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let intensities: Vec<f64> = intensity_chunks
        .iter()
        .flatten()
        .map(|&v| v as f64)
        .collect();
    if points.is_empty() {
        return (Vec::new(), carry.then(Vec::new));
    }
    let mut min_bound = points[0];
    for point in &points {
        for axis in 0..3 {
            min_bound[axis] = min_bound[axis].min(point[axis]);
        }
    }
    struct Cell {
        sum: [f64; 3],
        intensity_sum: f64,
        count: usize,
    }
    let mut cells: BTreeMap<[i64; 3], Cell> = BTreeMap::new();
    for (index, point) in points.iter().enumerate() {
        let key = [
            ((point[0] - min_bound[0]) / voxel).floor() as i64,
            ((point[1] - min_bound[1]) / voxel).floor() as i64,
            ((point[2] - min_bound[2]) / voxel).floor() as i64,
        ];
        let cell = cells.entry(key).or_insert(Cell {
            sum: [0.0; 3],
            intensity_sum: 0.0,
            count: 0,
        });
        for axis in 0..3 {
            cell.sum[axis] += point[axis];
        }
        if carry {
            cell.intensity_sum += intensities[index];
        }
        cell.count += 1;
    }
    let mut out_points = Vec::with_capacity(cells.len());
    let mut out_intensities = carry.then(|| Vec::with_capacity(cells.len()));
    for cell in cells.values() {
        let inv = 1.0 / cell.count as f64;
        out_points.push([
            (cell.sum[0] * inv) as f32,
            (cell.sum[1] * inv) as f32,
            (cell.sum[2] * inv) as f32,
        ]);
        if let Some(values) = out_intensities.as_mut() {
            values.push((cell.intensity_sum * inv) as f32);
        }
    }
    (out_points, out_intensities)
}

/// open3d `remove_statistical_outlier(nb_neighbors, std_ratio)`: keep points
/// whose mean knn distance (self included, as o3d searches the cloud itself)
/// is below `cloud_mean + std_ratio * sample_std`.
fn remove_statistical_outlier(
    points: &[[f32; 3]],
    intensities: Option<&[f32]>,
    nb_neighbors: usize,
    std_ratio: f64,
) -> (Vec<[f32; 3]>, Option<Vec<f32>>) {
    if points.is_empty() {
        return (Vec::new(), intensities.map(|_| Vec::new()));
    }
    let tree = KdTree::build(points);
    let mean_distances: Vec<f64> = points
        .iter()
        .map(|point| {
            let neighbors = tree.knn(point, nb_neighbors);
            if neighbors.is_empty() {
                return -1.0;
            }
            let total: f64 = neighbors
                .iter()
                .map(|(_, sq_dist)| (*sq_dist as f64).sqrt())
                .sum();
            total / neighbors.len() as f64
        })
        .collect();
    let valid: Vec<f64> = mean_distances.iter().copied().filter(|&d| d > 0.0).collect();
    if valid.is_empty() {
        return (Vec::new(), intensities.map(|_| Vec::new()));
    }
    let cloud_mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let sq_sum: f64 = valid.iter().map(|d| (d - cloud_mean) * (d - cloud_mean)).sum();
    let std_dev = if valid.len() > 1 {
        (sq_sum / (valid.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    let threshold = cloud_mean + std_ratio * std_dev;
    let mut out_points = Vec::new();
    let mut out_intensities = intensities.map(|_| Vec::new());
    for (index, &distance) in mean_distances.iter().enumerate() {
        if distance > 0.0 && distance < threshold {
            out_points.push(points[index]);
            if let (Some(out), Some(values)) = (out_intensities.as_mut(), intensities) {
                out.push(values[index]);
            }
        }
    }
    (out_points, out_intensities)
}

/// Per-scan corrected clouds into the db, stored body-relative on `corrected_odom_frame`
/// so rerun/tf can place them via `<odom>_corrected`; if `lcm_path`, also one aggregated,
/// world-baked .pc2.lcm (a single fused cloud has no tf to hang on).
#[allow(clippy::too_many_arguments)]
pub fn write_corrected_lidar(
    connection: &Connection,
    name: &str,
    scans: &[ScanRow],
    odom_rows: &[OdomPoseRow],
    keyframe_times: &[f64],
    corrections: &[Pose3],
    world_points: &dyn Fn(&ScanRow) -> Result<Vec<[f64; 3]>, String>,
    lcm_path: Option<&Path>,
    lcm_voxel: f64,
    world_frame: &str,
    corrected_odom_frame: &str,
) -> Result<(), String> {
    if memory2::list_streams(connection)?.iter().any(|s| s == name) {
        memory2::delete_stream(connection, name)?;
    }
    memory2::create_stream(connection, name, POINTCLOUD2_MODULE)?;
    let odom_times: Vec<f64> = odom_rows.iter().map(|row| row[0]).collect();
    let mut aggregated_points: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut aggregated_intensities: Vec<Vec<f32>> = Vec::new();
    let mut buffered_points: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut buffered_intensities: Vec<Vec<f32>> = Vec::new();
    let mut have_intensities = false;

    println!("writing {name} (corrected lidar)...");
    let started = Instant::now();
    let mut scan_count = 0usize;
    for scan in scans {
        scan_count += 1;
        let ts = scan.ts;
        let correction = interpolate_correction(keyframe_times, corrections, ts);
        let raw_row = &odom_rows[nearest_index(&odom_times, ts)];
        let raw_pose = se3::from_xyzquat(&raw_row[1..]);
        let points = world_points(scan)?;
        // body-relative points: place them via the corrected-odom tf, not baked into world
        let raw_rotation_t = crate::mat3::transpose(&raw_pose.rotation);
        let body_points: Vec<[f32; 3]> = points
            .iter()
            .map(|point| {
                let shifted = [
                    point[0] - raw_pose.translation[0],
                    point[1] - raw_pose.translation[1],
                    point[2] - raw_pose.translation[2],
                ];
                // (p - t) @ R == Rᵀ (p - t)
                let body = crate::mat3::mat_vec(&raw_rotation_t, &shifted);
                [body[0] as f32, body[1] as f32, body[2] as f32]
            })
            .collect();
        let blob = encode_pointcloud2(
            &body_points,
            scan.intensities.as_deref(),
            corrected_odom_frame,
            ts,
        );
        let pose = se3::pose_tuple(&se3::compose(&correction, &raw_pose));
        memory2::append(connection, name, ts, Some(pose), &[], &blob)?;
        if lcm_path.is_some() {
            let corrected_points: Vec<[f32; 3]> = points
                .iter()
                .map(|point| {
                    let world = crate::mat3::add(
                        &crate::mat3::mat_vec(&correction.rotation, point),
                        &correction.translation,
                    );
                    [world[0] as f32, world[1] as f32, world[2] as f32]
                })
                .collect();
            buffered_points.push(corrected_points);
            if let Some(values) = &scan.intensities {
                have_intensities = true;
                buffered_intensities.push(values.clone());
            }
            if buffered_points.len() >= LCM_CHUNK_SCANS {
                let (points_out, intensities_out) = voxel_downsample_chunks(
                    &buffered_points,
                    if have_intensities { &buffered_intensities } else { &[] },
                    lcm_voxel,
                );
                aggregated_points.push(points_out);
                if let Some(values) = intensities_out {
                    aggregated_intensities.push(values);
                }
                buffered_points = Vec::new();
                buffered_intensities = Vec::new();
            }
        }
        if scan_count % SCAN_LOG_EVERY == 0 {
            println!("  {scan_count} scans, {:.0}s", started.elapsed().as_secs_f64());
        }
    }
    println!(
        "wrote {name}: {scan_count} scans in {:.0}s",
        started.elapsed().as_secs_f64()
    );

    if let Some(lcm_path) = lcm_path {
        if !buffered_points.is_empty() {
            let (points_out, intensities_out) = voxel_downsample_chunks(
                &buffered_points,
                if have_intensities { &buffered_intensities } else { &[] },
                lcm_voxel,
            );
            aggregated_points.push(points_out);
            if let Some(values) = intensities_out {
                aggregated_intensities.push(values);
            }
        }
        write_aggregated_lcm(
            &aggregated_points,
            if have_intensities { &aggregated_intensities } else { &[] },
            lcm_voxel,
            lcm_path,
            odom_times[0],
            world_frame,
        )?;
    }
    Ok(())
}

/// Final unified voxel pass + statistical outlier removal into a single .pc2.lcm cloud.
pub fn write_aggregated_lcm(
    points_chunks: &[Vec<[f32; 3]>],
    intensity_chunks: &[Vec<f32>],
    voxel: f64,
    lcm_path: &Path,
    stamp: f64,
    world_frame: &str,
) -> Result<(), String> {
    let (points, intensities) = voxel_downsample_chunks(points_chunks, intensity_chunks, voxel);
    println!(
        "aggregating .pc2.lcm: {} pts after voxel, removing outliers...",
        format_thousands(points.len())
    );
    let (merged_points, merged_intensities) = remove_statistical_outlier(
        &points,
        intensities.as_deref(),
        LCM_OUTLIER_NN,
        LCM_OUTLIER_STD,
    );
    let blob = encode_pointcloud2(
        &merged_points,
        merged_intensities.as_deref(),
        world_frame,
        stamp,
    );
    std::fs::write(lcm_path, &blob).map_err(|e| e.to_string())?;
    println!(
        "wrote {}: 1 aggregated cloud, {} pts (voxel {voxel} m)",
        lcm_path.display(),
        format_thousands(merged_points.len())
    );
    Ok(())
}

/// Raycast `scans` (a lidar stream) into one `<in_stream>_accumulated` cloud, carving
/// free space along every ray so dynamic objects and registration ghosts get cleared,
/// not smeared.
pub fn raycast_accumulate(
    connection: &Connection,
    in_stream: &str,
    scans: &[ScanRow],
    store_tf: &RecordingTf,
    world_frame: &str,
    voxel: f64,
    max_range: f64,
) -> Result<(), String> {
    let out_stream = format!("{in_stream}_accumulated");
    let config = RayConfig::with_defaults(voxel as f32, max_range as f32);
    let mut map = VoxelMap::default();
    let mut live = Default::default();
    let mut scan_count = 0usize;
    let mut last_ts = 0.0f64;
    let started = Instant::now();
    for scan in scans {
        if scan.points.is_empty() {
            continue;
        }
        let transform = store_tf.get(world_frame, &scan.frame_id, scan.ts)?;
        let origin = (
            transform.translation[0] as f32,
            transform.translation[1] as f32,
            transform.translation[2] as f32,
        );
        let points: Vec<(f32, f32, f32)> = scan
            .points
            .iter()
            .map(|point| {
                let world = crate::mat3::add(
                    &crate::mat3::mat_vec(
                        &transform.rotation,
                        &[point[0] as f64, point[1] as f64, point[2] as f64],
                    ),
                    &transform.translation,
                );
                (world[0] as f32, world[1] as f32, world[2] as f32)
            })
            .collect();
        live = voxel_ray_tracer::update_map(&mut map, origin, &points, &config);
        last_ts = scan.ts;
        scan_count += 1;
        if scan_count % SCAN_LOG_EVERY == 0 {
            println!(
                "  {scan_count} scans, {} voxels, {:.0}s",
                format_thousands(map.healthy_count()),
                started.elapsed().as_secs_f64()
            );
        }
    }
    let accumulated: Vec<[f32; 3]> =
        voxel_ray_tracer::emit_points(&map, config.voxel_size, None, 0, &live)
            .into_iter()
            .map(|(x, y, z)| [x, y, z])
            .collect();
    if memory2::list_streams(connection)?.iter().any(|s| *s == out_stream) {
        memory2::delete_stream(connection, &out_stream)?;
    }
    memory2::create_stream(connection, &out_stream, POINTCLOUD2_MODULE)?;
    let blob = encode_pointcloud2(&accumulated, None, world_frame, last_ts);
    memory2::append(connection, &out_stream, last_ts, None, &[], &blob)?;
    println!(
        "wrote {out_stream}: {} pts from {scan_count} scans in {:.0}s",
        format_thousands(accumulated.len()),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory2::read_scans;

    fn identity() -> Pose3 {
        Pose3::identity()
    }

    fn translation(x: f64, y: f64, z: f64) -> Pose3 {
        Pose3::from_translation([x, y, z])
    }

    #[test]
    fn interpolation_clamps_and_lerps() {
        let times = [0.0, 1.0];
        let corrections = [identity(), translation(2.0, 0.0, 0.0)];
        let before = interpolate_correction(&times, &corrections, -1.0);
        assert!(before.translation[0].abs() < 1e-12);
        let after = interpolate_correction(&times, &corrections, 5.0);
        assert!((after.translation[0] - 2.0).abs() < 1e-12);
        let mid = interpolate_correction(&times, &corrections, 0.5);
        assert!((mid.translation[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn nearest_index_steps_back() {
        let times = [0.0, 1.0, 2.0];
        assert_eq!(nearest_index(&times, -5.0), 0);
        assert_eq!(nearest_index(&times, 0.4), 0);
        assert_eq!(nearest_index(&times, 0.6), 1);
        assert_eq!(nearest_index(&times, 99.0), 2);
    }

    #[test]
    fn outlier_removal_drops_the_far_point() {
        let mut points: Vec<[f32; 3]> = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                points.push([i as f32 * 0.1, j as f32 * 0.1, 0.0]);
            }
        }
        points.push([100.0, 100.0, 100.0]);
        let (kept, _) = remove_statistical_outlier(&points, None, 5, 2.0);
        assert_eq!(kept.len(), 100);
        assert!(kept.iter().all(|p| p[0] < 50.0));
    }

    #[test]
    fn voxel_downsample_averages_intensity() {
        let chunks = vec![vec![[0.0f32, 0.0, 0.0], [0.01, 0.01, 0.01], [1.0, 0.0, 0.0]]];
        let intensity = vec![vec![10.0f32, 20.0, 5.0]];
        let (points, intensities) = voxel_downsample_chunks(&chunks, &intensity, 0.5);
        assert_eq!(points.len(), 2);
        let intensities = intensities.unwrap();
        assert!(intensities.contains(&15.0) && intensities.contains(&5.0));
    }

    #[test]
    fn corrected_streams_round_trip_through_python_schema() {
        let connection = Connection::open_in_memory().unwrap();
        let times = [0.0, 10.0];
        let raw = [identity(), translation(1.0, 0.0, 0.0)];
        let optimized = [identity(), translation(1.5, 0.0, 0.0)];
        write_deformation_nodes(
            &connection,
            "tf_nodes_corrected",
            &times,
            &raw,
            &optimized,
            "world",
            "base",
        )
        .unwrap();
        write_pose_graph(&connection, "pose_graph", &times, &optimized, "world").unwrap();

        let corrections = [identity(), translation(0.5, 0.0, 0.0)];
        let odom_rows: Vec<OdomPoseRow> = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        write_corrected_odom(
            &connection,
            "odom_corrected",
            &odom_rows,
            &times,
            &corrections,
            "world",
            "odom_corrected_frame",
        )
        .unwrap();

        // registry rows exist with the exact python payload modules
        let streams = memory2::list_streams(&connection).unwrap();
        assert_eq!(
            streams,
            vec!["odom_corrected", "pose_graph", "tf_nodes_corrected"]
        );
        let config: String = connection
            .query_row(
                "SELECT config FROM _streams WHERE name = 'odom_corrected'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(config.contains("\"payload_module\": \"dimos.msgs.nav_msgs.Odometry.Odometry\""));
        assert!(config.contains("\"codec_id\": \"lcm\""));

        // deformation nodes: 2 rows per keyframe, jsonb tags queryable
        let tagged: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM \"tf_nodes_corrected\" \
                 WHERE json_extract(tags, '$.id') = '1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tagged, 2);

        // corrected odom: read back through our own reader; poses = correction ∘ raw
        let rows = memory2::read_odometry(&connection, "odom_corrected", 1).unwrap();
        assert_eq!(rows.len(), 2);
        assert!((rows[1].translation[0] - 1.5).abs() < 1e-9);
        assert_eq!(rows[1].frame_id, "world");
        assert_eq!(rows[1].child_frame_id, "odom_corrected_frame");
        // rtree rows only exist for posed streams
        let rtree_rows: i64 = connection
            .query_row("SELECT COUNT(*) FROM \"odom_corrected_rtree\"", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(rtree_rows, 2);

        // corrected lidar round trip incl. intensity
        let scans = vec![ScanRow {
            ts: 5.0,
            points: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            intensities: Some(vec![7.0, 8.0]),
            frame_id: "lidar".to_string(),
        }];
        let world_points = |scan: &ScanRow| -> Result<Vec<[f64; 3]>, String> {
            Ok(scan
                .points
                .iter()
                .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect())
        };
        let lcm_file = std::env::temp_dir().join("icp_stitch_test_agg.pc2.lcm");
        write_corrected_lidar(
            &connection,
            "lidar_corrected",
            &scans,
            &odom_rows,
            &times,
            &corrections,
            &world_points,
            Some(&lcm_file),
            0.1,
            "world",
            "odom_corrected_frame",
        )
        .unwrap();
        let read = read_scans(&connection, "lidar_corrected", 1).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].points.len(), 2);
        assert_eq!(read[0].intensities.as_ref().unwrap(), &vec![7.0, 8.0]);
        assert_eq!(read[0].frame_id, "odom_corrected_frame");
        // ts=5 ties between odom rows; searchsorted keeps the later one
        // (ts=10, x=1), so body x = input x - 1
        assert!((read[0].points[0][0] - 0.0).abs() < 1e-5);
        // aggregated lcm file decodes as a PointCloud2; a 2-point cloud is
        // fully dropped by o3d-semantics outlier removal (std=0 → strict <
        // threshold fails), exercising the empty-cloud encode branch
        let bytes = std::fs::read(&lcm_file).unwrap();
        let decoded = PointCloud2::decode(&bytes).unwrap();
        assert_eq!(decoded.header.frame_id, "world");
        assert_eq!(decoded.width, 0);
        assert_eq!(decoded.fields.len(), 4);
        std::fs::remove_file(&lcm_file).ok();
    }
}
