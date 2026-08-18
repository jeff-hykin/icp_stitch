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

//! Recording-specific stream + odom-edge resolution (`recording_scans.py`) and
//! the in-place go2-legacy normalize (`go2_legacy.py`): legacy go2 recordings
//! store lidar pre-registered in a fake world frame and odometry as bare
//! `PoseStamped`; this derives sensor-frame `l1_cloud`, a static
//! `base_link -> l1_link` tf, and a proper `go2_odometry` stream.

use gtsam_shim::Pose3;
use lcm_msgs::geometry_msgs;
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::std_msgs;
use lcm_msgs::tf2_msgs::TFMessage;
use rusqlite::Connection;

use crate::artifacts::{self, encode_pointcloud2};
use crate::mat3;
use crate::memory2::{self, read_odometry, read_raw_rows, read_scans, read_tf};
use crate::pgo::OdomPoseRow;
use crate::se3;
use crate::tf::RecordingTf;

/// (odom stream, lidar-fallback candidates). First pair whose odom stream exists wins, so a
/// recording never mixes rigs (e.g. fastlio + pointlio). Ordered mid360 rig -> go2 -> generic.
pub const STREAM_PAIRS: [(&str, &[&str]); 4] = [
    ("pointlio_odometry", &["pointlio_lidar"]),
    ("fastlio_odometry", &["fastlio_lidar"]),
    ("go2_odom", &["go2_lidar", "l1_lidar", "lidar"]),
    ("odom", &["lidar"]),
];

const TF_STREAM: &str = "tf";
// go2's stuff is in a fake world frame, so we correct that with a new db stream
const LEGACY_ODOM_STREAMS: [&str; 2] = ["odom", "go2_odom"];
const GO2_CORRECTED_LIDAR_STREAM_NAME: &str = "l1_cloud";
const GO2_CORRECTED_LIDAR_FRAME: &str = "l1_link";
// fallbacks when a bare PoseStamped odom carries no world/base frame in its header
const DEFAULT_WORLD_FRAME: &str = "world";
const DEFAULT_BASE_FRAME: &str = "base_link";
// proper "Odometry" type instead of Pose
const GO2_CORRECTED_ODOMETRY_STREAM_NAME: &str = "go2_odometry";
const LOG_EVERY: usize = 5000;

const POSE_STAMPED_MODULE: &str = "dimos.msgs.geometry_msgs.PoseStamped.PoseStamped";
const TFMESSAGE_MODULE: &str = "dimos.msgs.tf2_msgs.TFMessage.TFMessage";

/// `(odom_stream, lidar_stream)` defaults from what a recording actually has.
pub fn resolve_streams(available: &[String], odom: &str, lidar: &str) -> (String, String) {
    let has = |name: &str| available.iter().any(|s| s == name);
    let odom = if odom.is_empty() {
        STREAM_PAIRS
            .iter()
            .map(|(name, _)| *name)
            .find(|name| has(name))
            .unwrap_or("odom")
            .to_string()
    } else {
        odom.to_string()
    };
    let lidar = if lidar.is_empty() {
        let candidates: &[&str] = STREAM_PAIRS
            .iter()
            .find(|(name, _)| *name == odom)
            .map(|(_, candidates)| *candidates)
            .unwrap_or(&["lidar"]);
        candidates
            .iter()
            .copied()
            .find(|name| has(name))
            .unwrap_or(candidates[0])
            .to_string()
    } else {
        lidar.to_string()
    };
    (odom, lidar)
}

fn payload_module(connection: &Connection, stream: &str) -> Option<String> {
    connection
        .query_row(
            "SELECT json_extract(config, '$.payload_module') FROM _streams WHERE name = ?",
            [stream],
            |row| row.get(0),
        )
        .ok()
}

fn first_blob(connection: &Connection, stream: &str) -> Option<Vec<u8>> {
    if stream.contains('"') {
        return None;
    }
    connection
        .query_row(
            &format!(
                "SELECT blob.data FROM \"{stream}\" AS meta \
                 JOIN \"{stream}_blob\" AS blob ON meta.id = blob.id \
                 ORDER BY meta.ts LIMIT 1"
            ),
            [],
            |row| row.get(0),
        )
        .ok()
}

/// `"parent:child"` from the odom stream's own header, or `""` if it has no child frame
/// (e.g. `PoseStamped` odometry).
pub fn default_odom_edge(connection: &Connection, odom_stream: &str) -> String {
    let Some(blob) = first_blob(connection, odom_stream) else {
        return String::new();
    };
    let Ok(message) = Odometry::decode(&blob) else {
        return String::new();
    };
    if message.child_frame_id.is_empty() {
        return String::new();
    }
    format!("{}:{}", message.header.frame_id, message.child_frame_id)
}

/// True when `odom_stream` is a go2-legacy bare-`Pose` odom (not `Odometry`).
fn is_go2_legacy(connection: &Connection, odom_stream: &str) -> bool {
    if !LEGACY_ODOM_STREAMS.contains(&odom_stream) {
        return false;
    }
    payload_module(connection, odom_stream).as_deref() == Some(POSE_STAMPED_MODULE)
}

/// `(N, 8)` `ts, x, y, z, qx, qy, qz, qw` from the odom `Pose` payloads, NaN-filtered.
fn odom_pose_rows(connection: &Connection, odom_stream: &str) -> Result<Vec<OdomPoseRow>, String> {
    let rows = read_odometry(connection, odom_stream, 1)?;
    Ok(rows
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
        .filter(|row| row.iter().all(|value| value.is_finite()))
        .collect())
}

/// Write `source_lidar` into `l1_cloud` (`l1_link` frame), un-registering via `tf`.
///
/// `tf` carries an ephemeral `world_frame -> l1_link` edge (the odom trajectory), so a
/// world-registered scan is pulled back into the sensor frame by the tf chain rather than
/// hand-rolled quat math; scans already in a sensor frame pass through unchanged. Either way
/// the latched odom pose is kept on the row so `p_world = pose * p_l1` reconstructs the map.
fn write_l1_cloud(
    connection: &Connection,
    source_lidar: &str,
    odom_rows: &[OdomPoseRow],
    world_frame: &str,
    tf: &RecordingTf,
) -> Result<(), String> {
    let odom_times: Vec<f64> = odom_rows.iter().map(|row| row[0]).collect();
    if memory2::list_streams(connection)?
        .iter()
        .any(|s| s == GO2_CORRECTED_LIDAR_STREAM_NAME)
    {
        memory2::delete_stream(connection, GO2_CORRECTED_LIDAR_STREAM_NAME)?;
    }
    memory2::create_stream(
        connection,
        GO2_CORRECTED_LIDAR_STREAM_NAME,
        artifacts::POINTCLOUD2_MODULE,
    )?;
    let mut count = 0usize;
    for scan in read_scans(connection, source_lidar, 1)? {
        let scan_ts = scan.ts;
        // searchsorted(side="right") - 1, clamped: the pose latched at scan time
        let latched = odom_times
            .partition_point(|&t| t <= scan_ts)
            .saturating_sub(1);
        let xyzquat: [f64; 7] = odom_rows[latched][1..].try_into().unwrap();
        let l1_points: Vec<[f32; 3]> = if scan.frame_id == world_frame {
            let world_to_sensor = tf.get(GO2_CORRECTED_LIDAR_FRAME, world_frame, scan_ts)?;
            scan.points
                .iter()
                .map(|point| {
                    let moved = mat3::add(
                        &mat3::mat_vec(
                            &world_to_sensor.rotation,
                            &[point[0] as f64, point[1] as f64, point[2] as f64],
                        ),
                        &world_to_sensor.translation,
                    );
                    [moved[0] as f32, moved[1] as f32, moved[2] as f32]
                })
                .collect()
        } else {
            scan.points.clone()
        };
        let blob = encode_pointcloud2(
            &l1_points,
            scan.intensities.as_deref(),
            GO2_CORRECTED_LIDAR_FRAME,
            scan_ts,
        );
        memory2::append(
            connection,
            GO2_CORRECTED_LIDAR_STREAM_NAME,
            scan_ts,
            Some(xyzquat),
            &[],
            &blob,
        )?;
        count += 1;
        if count % LOG_EVERY == 0 {
            println!("  {GO2_CORRECTED_LIDAR_STREAM_NAME}: {count} scans...");
        }
    }
    println!(
        "wrote {GO2_CORRECTED_LIDAR_STREAM_NAME}: {count} scans \
         in {GO2_CORRECTED_LIDAR_FRAME} frame"
    );
    Ok(())
}

fn encode_static_tf_edge(stamp: f64, base_frame: &str) -> Vec<u8> {
    let message = TFMessage {
        transforms: vec![lcm_msgs::geometry_msgs::TransformStamped {
            header: std_msgs::Header {
                seq: 1,
                stamp: std_msgs::Time {
                    sec: stamp as i32,
                    nsec: ((stamp - stamp.trunc()) * 1e9) as i32,
                },
                frame_id: base_frame.to_string(),
            },
            child_frame_id: GO2_CORRECTED_LIDAR_FRAME.to_string(),
            transform: geometry_msgs::Transform {
                translation: geometry_msgs::Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                rotation: geometry_msgs::Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
            },
        }],
    };
    message.encode()
}

/// Write exactly one identity `base_frame -> l1_link` edge so the sensor frame joins the
/// tf tree. `l1_link` is only ever introduced here, so any pre-existing edge into it is a
/// leftover from an earlier normalize run; strip all of them first (rewriting the stream, the
/// only removal the store API offers) so re-runs never accumulate duplicates.
fn write_static_tf(connection: &Connection, stamp: f64, base_frame: &str) -> Result<(), String> {
    if !memory2::list_streams(connection)?
        .iter()
        .any(|s| s == TF_STREAM)
    {
        memory2::create_stream(connection, TF_STREAM, TFMESSAGE_MODULE)?;
    }
    let rows = read_raw_rows(connection, TF_STREAM)?;
    let mut kept = Vec::new();
    let mut stale = 0usize;
    for row in rows {
        let has_l1_edge = TFMessage::decode(&row.blob)
            .map(|message| {
                message
                    .transforms
                    .iter()
                    .any(|edge| edge.child_frame_id == GO2_CORRECTED_LIDAR_FRAME)
            })
            .unwrap_or(false);
        if has_l1_edge {
            stale += 1;
        } else {
            kept.push(row);
        }
    }
    if stale > 0 {
        memory2::delete_stream(connection, TF_STREAM)?;
        memory2::create_stream(connection, TF_STREAM, TFMESSAGE_MODULE)?;
        for row in &kept {
            memory2::append_raw(
                connection,
                TF_STREAM,
                row.ts,
                row.pose,
                &row.tags_json,
                &row.blob,
            )?;
        }
    }
    memory2::append(
        connection,
        TF_STREAM,
        stamp,
        None,
        &[("child_frame", GO2_CORRECTED_LIDAR_FRAME.to_string())],
        &encode_static_tf_edge(stamp, base_frame),
    )?;
    println!(
        "wrote static tf {base_frame} -> {GO2_CORRECTED_LIDAR_FRAME} (identity); \
         removed {stale} stale duplicate(s)"
    );
    Ok(())
}

/// Rewrite the bare-`Pose` odom as a proper `world -> l1_link` `Odometry` stream.
fn write_go2_odometry(
    connection: &Connection,
    odom_rows: &[OdomPoseRow],
    world_frame: &str,
) -> Result<(), String> {
    if memory2::list_streams(connection)?
        .iter()
        .any(|s| s == GO2_CORRECTED_ODOMETRY_STREAM_NAME)
    {
        memory2::delete_stream(connection, GO2_CORRECTED_ODOMETRY_STREAM_NAME)?;
    }
    memory2::create_stream(
        connection,
        GO2_CORRECTED_ODOMETRY_STREAM_NAME,
        artifacts::ODOMETRY_MODULE,
    )?;
    for row in odom_rows {
        let stamp = row[0];
        let pose: [f64; 7] = row[1..].try_into().unwrap();
        let blob = artifacts::encode_odometry(
            stamp,
            world_frame,
            GO2_CORRECTED_LIDAR_FRAME,
            &pose,
        );
        memory2::append(
            connection,
            GO2_CORRECTED_ODOMETRY_STREAM_NAME,
            stamp,
            Some(pose),
            &[],
            &blob,
        )?;
    }
    println!(
        "wrote {GO2_CORRECTED_ODOMETRY_STREAM_NAME}: {} poses",
        odom_rows.len()
    );
    Ok(())
}

/// Convert a legacy go2 recording in place, returning `(odom_tf, odom, lidar)` to use.
///
/// On a go2-legacy recording this derives `l1_cloud`, a static `base_link -> l1_link`
/// tf, and a `go2_odometry` stream, then returns `("<world>:l1_link", "go2_odometry",
/// "l1_cloud")`. Any other recording is untouched and its inputs are returned unchanged.
pub fn normalize_go2_legacy(
    connection: &Connection,
    odom_tf: &str,
    odom_stream: &str,
    lidar_stream: &str,
) -> Result<(String, String, String), String> {
    let unchanged = || {
        (
            odom_tf.to_string(),
            odom_stream.to_string(),
            lidar_stream.to_string(),
        )
    };
    if !is_go2_legacy(connection, odom_stream) {
        return Ok(unchanged());
    }
    let odom_rows = odom_pose_rows(connection, odom_stream)?;
    if odom_rows.is_empty() {
        return Ok(unchanged());
    }
    let (mut world_frame, base_frame) = match odom_tf.split_once(':') {
        Some((world, base)) => (world.to_string(), base.to_string()),
        None => (odom_tf.to_string(), String::new()),
    };
    // a bare PoseStamped odom has no child_frame_id, so default_odom_edge hands us an empty
    // edge; fall back to the odom's own header frame (its parent) and the go2 base link, else
    // go2_odometry + the static tf get written in an empty frame and every world<->l1_link
    // lookup fails (raw map + comparison rrd come out empty).
    if world_frame.is_empty() {
        let header_frame = read_odometry(connection, odom_stream, 1)?
            .first()
            .map(|row| row.frame_id.clone())
            .unwrap_or_default();
        world_frame = if header_frame.is_empty() {
            DEFAULT_WORLD_FRAME.to_string()
        } else {
            header_frame
        };
    }
    let base_frame = if base_frame.is_empty() {
        DEFAULT_BASE_FRAME.to_string()
    } else {
        base_frame
    };
    println!("go2 legacy recording: deriving l1_cloud / go2_odometry / l1_link tf");
    // Ephemeral world->l1_link edge (the odom trajectory) so write_l1_cloud un-registers
    // world-framed scans through the tf chain instead of hand-rolled quat math.
    let tf_samples = if memory2::list_streams(connection)?
        .iter()
        .any(|s| s == TF_STREAM)
    {
        read_tf(connection, TF_STREAM)?
    } else {
        Vec::new()
    };
    let mut tf = RecordingTf::from_samples(&tf_samples);
    let trajectory: Vec<(f64, Pose3)> = odom_rows
        .iter()
        .map(|row| (row[0], se3::from_xyzquat(&row[1..])))
        .collect();
    tf.override_edge(&world_frame, GO2_CORRECTED_LIDAR_FRAME, trajectory);
    write_l1_cloud(connection, lidar_stream, &odom_rows, &world_frame, &tf)?;
    write_static_tf(connection, odom_rows[0][0], &base_frame)?;
    write_go2_odometry(connection, &odom_rows, &world_frame)?;
    Ok((
        format!("{world_frame}:{GO2_CORRECTED_LIDAR_FRAME}"),
        GO2_CORRECTED_ODOMETRY_STREAM_NAME.to_string(),
        GO2_CORRECTED_LIDAR_STREAM_NAME.to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory2::read_scans;

    #[test]
    fn stream_pairs_pick_by_rig() {
        let available: Vec<String> = ["fastlio_odometry", "fastlio_lidar", "lidar"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            resolve_streams(&available, "", ""),
            ("fastlio_odometry".to_string(), "fastlio_lidar".to_string())
        );
        // explicit odom wins; lidar falls back through the candidate list
        let go2: Vec<String> = ["go2_odom", "lidar"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            resolve_streams(&go2, "", ""),
            ("go2_odom".to_string(), "lidar".to_string())
        );
        assert_eq!(
            resolve_streams(&go2, "odom", ""),
            ("odom".to_string(), "lidar".to_string())
        );
        // nothing available: generic defaults
        assert_eq!(
            resolve_streams(&[], "", ""),
            ("odom".to_string(), "lidar".to_string())
        );
    }

    fn pose_stamped_blob(ts: f64, frame_id: &str, xyz: [f64; 3]) -> Vec<u8> {
        let message = lcm_msgs::geometry_msgs::PoseStamped {
            header: std_msgs::Header {
                seq: 1,
                stamp: std_msgs::Time {
                    sec: ts as i32,
                    nsec: 0,
                },
                frame_id: frame_id.to_string(),
            },
            pose: geometry_msgs::Pose {
                position: geometry_msgs::Point {
                    x: xyz[0],
                    y: xyz[1],
                    z: xyz[2],
                },
                orientation: geometry_msgs::Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
            },
        };
        message.encode()
    }

    #[test]
    fn normalize_untouched_for_non_legacy() {
        let connection = Connection::open_in_memory().unwrap();
        memory2::create_stream(&connection, "pointlio_odometry", artifacts::ODOMETRY_MODULE)
            .unwrap();
        let result =
            normalize_go2_legacy(&connection, "world:base", "pointlio_odometry", "pointlio_lidar")
                .unwrap();
        assert_eq!(
            result,
            (
                "world:base".to_string(),
                "pointlio_odometry".to_string(),
                "pointlio_lidar".to_string()
            )
        );
    }

    #[test]
    fn normalize_derives_l1_streams() {
        let connection = Connection::open_in_memory().unwrap();
        // legacy odom: bare PoseStamped moving along +x in "world"
        memory2::create_stream(&connection, "odom", POSE_STAMPED_MODULE).unwrap();
        for (index, ts) in [0.0f64, 1.0, 2.0].iter().enumerate() {
            let blob = pose_stamped_blob(*ts, "world", [index as f64, 0.0, 0.0]);
            memory2::append(&connection, "odom", *ts, None, &[], &blob).unwrap();
        }
        // lidar pre-registered in "world": one point at the robot's x=1 pose + 1m forward
        memory2::create_stream(&connection, "lidar", artifacts::POINTCLOUD2_MODULE).unwrap();
        let scan = encode_pointcloud2(&[[2.0, 0.0, 0.0]], None, "world", 1.0);
        memory2::append(&connection, "lidar", 1.0, None, &[], &scan).unwrap();

        let (odom_tf, odom, lidar) =
            normalize_go2_legacy(&connection, "", "odom", "lidar").unwrap();
        assert_eq!(odom_tf, "world:l1_link");
        assert_eq!(odom, "go2_odometry");
        assert_eq!(lidar, "l1_cloud");

        // scan un-registered into the sensor frame: 1m forward of the x=1 pose
        let scans = read_scans(&connection, "l1_cloud", 1).unwrap();
        assert_eq!(scans.len(), 1);
        assert_eq!(scans[0].frame_id, "l1_link");
        assert!((scans[0].points[0][0] - 1.0).abs() < 1e-5);
        // odometry readable as proper Odometry with the l1_link child
        let odom_rows = read_odometry(&connection, "go2_odometry", 1).unwrap();
        assert_eq!(odom_rows.len(), 3);
        assert_eq!(odom_rows[1].child_frame_id, "l1_link");
        // static tf edge present exactly once, and re-running does not duplicate it
        let tf_samples = read_tf(&connection, TF_STREAM).unwrap();
        let l1_edges = |samples: &[crate::memory2::TfSample]| {
            samples
                .iter()
                .filter(|s| s.child == "l1_link")
                .count()
        };
        assert_eq!(l1_edges(&tf_samples), 1);
        normalize_go2_legacy(&connection, "", "odom", "lidar").unwrap();
        let tf_again = read_tf(&connection, TF_STREAM).unwrap();
        assert_eq!(l1_edges(&tf_again), 1);
    }
}
