//! AprilTag-loop-closed + ICP-refined ground-truth post-processing for a recording.
//!
//! Rust port of dimos gsc_pgo `scripts/post_process.py` with the same CLI.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use gtsam_shim::Pose3;
use rusqlite::{Connection, OpenFlags};

use icp_stitch::apriltags::{self, GlimpseGates};
use icp_stitch::{artifacts, detect, helpers, mat3, memory2, pgo, recording, se3, tf};

const ABOUT: &str = "\
AprilTag-loop-closed + ICP-refined ground-truth post-processing for a recording.

Two-stage solve turns drifty odometry into a ground-truth trajectory:
  1. GTSAM tag PGO: anisotropic odometry between-factors (stiff roll/pitch + gravity z
     anchor, loose yaw) + quality-weighted AprilTag landmark factors fix macro drift.
  2. ICP loop closures between spatially-close / temporally-distant lidar submaps anchor
     local geometry, then re-solve.

Outputs written back into the recording db: <odom>_corrected, <lidar>_corrected,
tf_deformation_nodes_corrected, pose_graph, and raycast-accumulated maps; plus an
aggregated <lidar>_corrected.pc2.lcm and a comparison rrd opened in rerun.

--db is the recording .db file; its parent dir is where the .pc2.lcm outputs land. Camera
intrinsics come from the recording's CameraInfo stream (auto-detected) and the base<-optical
extrinsic from its tf tree; stream/frame defaults auto-detect the rig. With no CameraInfo
stream the AprilTag stage is skipped and ICP loop closures alone drive the PGO; a go2 recording
can get one with --helper add_go2_camera_info.

Usage:
  icp_stitch --db PATH.db [--no-odom | --no-lidar] [options]
  icp_stitch --db PATH.db --helper add_go2_camera_info";

#[derive(Clone, Copy, clap::ValueEnum)]
enum Helper {
    /// write the static go2 front-camera 720p intrinsics as a `camera_info` stream
    #[value(name = "add_go2_camera_info")]
    AddGo2CameraInfo,
}

#[derive(Parser)]
#[command(name = "icp_stitch", about = ABOUT)]
struct Args {
    /// recording .db file
    #[arg(long)]
    db: PathBuf,
    /// run a one-shot recording fixup on --db and exit, instead of solving
    #[arg(long = "helper", alias = "helpers", value_enum)]
    helper: Option<Helper>,
    /// input lidar stream (auto if unset)
    #[arg(long, default_value = "")]
    lidar: String,
    /// input odometry stream (auto if unset)
    #[arg(long, default_value = "")]
    odom: String,
    /// unfiltered AprilTag stream
    #[arg(long, default_value = "raw_april_tags")]
    tags: String,
    /// image stream to detect tags on
    #[arg(long, default_value = "color_image")]
    camera: String,
    /// CameraInfo stream (K + distortion); when unset, tries '<camera>_camera_info' then 'camera_info'
    #[arg(long = "camera-info-stream", default_value = "")]
    camera_info_stream: String,
    /// optical frame the tag detections sit in; the base<-optical extrinsic is read from the tf
    /// tree (falls back to this when CameraInfo carries no frame_id)
    #[arg(long = "tag-frame", default_value = "camera_optical")]
    tag_frame: String,
    /// base<-optical camera extrinsic 'x y z qx qy qz qw' (meters + quaternion), used only when
    /// the recording has no tf tree to resolve it. Mid360 rig: '0.3 0 0 -0.5 0.5 -0.5 0.5'
    #[arg(long = "base-optical", default_value = "")]
    base_optical: String,
    /// AprilTag edge length (m)
    #[arg(long = "tag-size", default_value_t = 0.10)]
    tag_size: f64,
    #[arg(long = "dict", default_value = "DICT_APRILTAG_36h11")]
    dictionary: String,
    /// comma/space-separated moving tag ids
    #[arg(long = "ignore-tags", default_value = "")]
    ignore_tags: String,
    #[arg(long = "corrected-suffix", default_value = "_corrected")]
    corrected_suffix: String,
    #[arg(long, default_value = "")]
    suffix: String,
    #[arg(long = "world-frame", default_value = "world")]
    world_frame: String,
    /// child frame the corrected odom/lidar hang on (tf-driven, not world-baked)
    #[arg(long = "corrected-odom-frame", default_value = "corrected_odom")]
    corrected_odom_frame: String,
    /// 'parent:child' edge the odom overrides
    #[arg(long = "odom-tf", default_value = "")]
    odom_tf: String,
    /// max one ICP loop closure per this many meters of odom path (<=0 disables thinning)
    #[arg(long = "closure-spacing", default_value_t = 2.0)]
    closure_spacing: f64,
    #[arg(long = "no-odom", action = clap::ArgAction::SetFalse)]
    write_odom: bool,
    #[arg(long = "no-lidar", action = clap::ArgAction::SetFalse)]
    write_lidar: bool,
    #[arg(long = "no-icp", action = clap::ArgAction::SetFalse)]
    icp: bool,
    #[arg(long = "no-lcm", action = clap::ArgAction::SetFalse)]
    lcm: bool,
    #[arg(long = "no-rrd", action = clap::ArgAction::SetFalse)]
    rrd: bool,
    #[arg(long = "no-accum", action = clap::ArgAction::SetFalse)]
    accum: bool,
    #[arg(long = "lcm-voxel", default_value_t = 0.05)]
    lcm_voxel: f64,
    #[arg(long = "accum-voxel", default_value_t = 0.05)]
    accum_voxel: f64,
    #[arg(long = "accum-max-range", default_value_t = 20.0)]
    accum_max_range: f64,

    // solve tuning: keyframe / factor-noise / ICP knobs (see pgo::Tuning)
    #[arg(long = "keyframe-translation-m", default_value_t = 0.5, help = "default 0.5", help_heading = "solve tuning")]
    keyframe_translation_m: f64,
    #[arg(long = "keyframe-rotation-deg", default_value_t = 10.0, help = "default 10", help_heading = "solve tuning")]
    keyframe_rotation_deg: f64,
    #[arg(long = "lm-max-iterations", default_value_t = 200, help = "default 200", help_heading = "solve tuning")]
    lm_max_iterations: i32,
    #[arg(long = "odom-rot-roll-pitch-var", default_value_t = 1e-8, help = "default 1e-08", help_heading = "solve tuning")]
    odom_rot_roll_pitch_var: f64,
    #[arg(long = "odom-rot-yaw-var", default_value_t = 1e-5, help = "default 1e-05", help_heading = "solve tuning")]
    odom_rot_yaw_var: f64,
    #[arg(long = "odom-trans-xy-var", default_value_t = 1e-4, help = "default 0.0001", help_heading = "solve tuning")]
    odom_trans_xy_var: f64,
    #[arg(long = "odom-trans-z-var", default_value_t = 1e-6, help = "default 1e-06", help_heading = "solve tuning")]
    odom_trans_z_var: f64,
    #[arg(long = "icp-radius-m", default_value_t = 4.0, help = "default 4", help_heading = "solve tuning")]
    icp_radius_m: f64,
    #[arg(long = "icp-min-dt-s", default_value_t = 25.0, help = "default 25", help_heading = "solve tuning")]
    icp_min_dt_s: f64,
    #[arg(long = "icp-max-corr-m", default_value_t = 0.6, help = "default 0.6", help_heading = "solve tuning")]
    icp_max_corr_m: f64,
    #[arg(long = "icp-voxel-m", default_value_t = 0.15, help = "default 0.15", help_heading = "solve tuning")]
    icp_voxel_m: f64,
    #[arg(long = "icp-fitness-min", default_value_t = 0.45, help = "default 0.45", help_heading = "solve tuning")]
    icp_fitness_min: f64,
    #[arg(long = "icp-rmse-max-m", default_value_t = 0.25, help = "default 0.25", help_heading = "solve tuning")]
    icp_rmse_max_m: f64,
    #[arg(long = "icp-huber-delta", default_value_t = 1.345, help = "default 1.345", help_heading = "solve tuning")]
    icp_huber_delta: f64,
    #[arg(long = "icp-rot-var", default_value_t = 4e-4, help = "default 0.0004", help_heading = "solve tuning")]
    icp_rot_var: f64,
    #[arg(long = "icp-trans-var", default_value_t = 2.5e-3, help = "default 0.0025", help_heading = "solve tuning")]
    icp_trans_var: f64,
    #[arg(long = "submap-half-s", default_value_t = 1.0, help = "default 1", help_heading = "solve tuning")]
    submap_half_s: f64,
}

impl Args {
    fn tuning(&self) -> pgo::Tuning {
        pgo::Tuning {
            keyframe_translation_m: self.keyframe_translation_m,
            keyframe_rotation_deg: self.keyframe_rotation_deg,
            lm_max_iterations: self.lm_max_iterations,
            odom_rot_roll_pitch_var: self.odom_rot_roll_pitch_var,
            odom_rot_yaw_var: self.odom_rot_yaw_var,
            odom_trans_xy_var: self.odom_trans_xy_var,
            odom_trans_z_var: self.odom_trans_z_var,
            icp_radius_m: self.icp_radius_m,
            icp_min_dt_s: self.icp_min_dt_s,
            icp_max_corr_m: self.icp_max_corr_m,
            icp_voxel_m: self.icp_voxel_m,
            icp_fitness_min: self.icp_fitness_min,
            icp_rmse_max_m: self.icp_rmse_max_m,
            icp_huber_delta: self.icp_huber_delta,
            icp_rot_var: self.icp_rot_var,
            icp_trans_var: self.icp_trans_var,
            submap_half_s: self.submap_half_s,
        }
    }
}

/// Parse a `'x y z qx qy qz qw'` base<-optical extrinsic (None if empty).
fn parse_base_optical(spec: &str) -> Result<Option<Pose3>, String> {
    if spec.trim().is_empty() {
        return Ok(None);
    }
    let values: Vec<f64> = spec
        .replace(',', " ")
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|_| format!("--base-optical: not a number: {token:?}"))
        })
        .collect::<Result<_, _>>()?;
    if values.len() != 7 {
        return Err(format!(
            "--base-optical needs 7 numbers 'x y z qx qy qz qw', got {}: '{spec}'",
            values.len()
        ));
    }
    Ok(Some(se3::from_xyzquat(&values)))
}

/// base<-optical extrinsic: explicit `--base-optical`, else the recording's tf tree.
fn resolve_base_optical(
    store_tf: &tf::RecordingTf,
    body_frame: &str,
    optical_frame: &str,
    ts: f64,
    cli_spec: &str,
) -> Result<Pose3, String> {
    if let Some(pose) = parse_base_optical(cli_spec)? {
        return Ok(pose);
    }
    store_tf.get(body_frame, optical_frame, ts).map_err(|error| {
        format!(
            "cannot resolve the camera extrinsic '{body_frame}' <- '{optical_frame}' from the \
             tf tree; pass --base-optical 'x y z qx qy qz qw' for this rig. ({error})"
        )
    })
}

/// numpy-style median (mean of the two middles for even lengths).
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn main() {
    if let Err(message) = run(Args::parse()) {
        eprintln!("{message}");
        exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let tuning = args.tuning();
    let db_path = &args.db;
    if db_path.is_dir() {
        return Err(format!(
            "--db must be a .db file, not a directory: {}",
            db_path.display()
        ));
    }
    let rec_dir = db_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let connection = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_WRITE)
        .map_err(|error| format!("cannot open {}: {error}", db_path.display()))?;

    if let Some(helper) = args.helper {
        return match helper {
            Helper::AddGo2CameraInfo => {
                helpers::add_go2_camera_info(&connection, &args.camera)
            }
        };
    }

    // resolve stream/frame defaults from what the recording actually has
    let streams = memory2::list_streams(&connection)?;
    let (odom_stream, lidar_stream) = recording::resolve_streams(&streams, &args.odom, &args.lidar);
    let odom_tf = if args.odom_tf.is_empty() {
        recording::default_odom_edge(&connection, &odom_stream)
    } else {
        args.odom_tf.clone()
    };
    // legacy go2 recordings are massaged into the generic shape here; every other rig is a no-op
    let (odom_tf, odom_stream, lidar_stream) =
        recording::normalize_go2_legacy(&connection, &odom_tf, &odom_stream, &lidar_stream)?;
    let body_frame = if odom_tf.is_empty() {
        args.world_frame.clone()
    } else {
        odom_tf
            .split_once(':')
            .map(|(_, child)| child.to_string())
            .ok_or_else(|| format!("--odom-tf must be 'parent:child', got '{odom_tf}'"))?
    };
    let ignore_tags: HashSet<i64> = args
        .ignore_tags
        .replace(',', " ")
        .split_whitespace()
        .map(|token| {
            token
                .parse::<i64>()
                .map_err(|_| format!("--ignore-tags: not an integer: {token:?}"))
        })
        .collect::<Result<_, _>>()?;

    let (camera_info, camera_info_tried) =
        detect::resolve_camera_info(&connection, &args.camera, &args.camera_info_stream)?;
    let (camera_model, optical_frame) = match camera_info {
        None => {
            println!(
                "WARNING: no CameraInfo stream among ['{}'] -- AprilTag stage skipped; \
                 ICP + odom only. If this is a go2 recording, add the static front-camera \
                 intrinsics first with --helper add_go2_camera_info, then re-run.",
                camera_info_tried.join("', '")
            );
            (None, args.tag_frame.clone())
        }
        Some((model, frame_id)) => {
            let frame = if frame_id.is_empty() {
                args.tag_frame.clone()
            } else {
                frame_id
            };
            (Some(model), frame)
        }
    };
    let tags_available = detect::ensure_raw_tag_stream(
        &connection,
        camera_model.as_ref(),
        &args.tags,
        &args.camera,
        args.tag_size,
        &args.dictionary,
    )?;
    if !tags_available {
        println!(
            "no AprilTag data ('{}' absent) -- running tag-free (odom + ICP only)",
            args.tags
        );
    }

    println!("recording: {}", rec_dir.display());
    println!(
        "streams: tags={} odom={} lidar={} -> {}{}",
        args.tags, odom_stream, lidar_stream, args.corrected_suffix, args.suffix
    );

    // gate tags, pick keyframes, keep one best factor per keyframe x marker
    let raw_detections = if tags_available {
        detect::read_raw_tag_stream(&connection, &args.tags)?
    } else {
        Vec::new()
    };
    let detections =
        apriltags::filter_glimpses(&raw_detections, &ignore_tags, &GlimpseGates::default());
    let odom_full = memory2::read_odometry(&connection, &odom_stream, 1).map_err(|error| {
        if error.contains("decoded no") {
            format!(
                "odom stream '{odom_stream}' is empty in {}",
                db_path.display()
            )
        } else {
            error
        }
    })?;
    let odom_rows: Vec<pgo::OdomPoseRow> = odom_full
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
        .collect();
    let (_indices, keyframe_poses, keyframe_times) = pgo::select_keyframes(&odom_rows, &tuning);
    let best_factors = pgo::best_factor_per_keyframe_marker(&detections, &keyframe_times);
    if !raw_detections.is_empty() {
        pgo::report_revisits(&raw_detections, &best_factors);
    }

    // tf tree with the odom edge overridden by the odom trajectory (python
    // RecordingTF.from_store(store, odom_tf=..., odom_stream=...))
    let streams = memory2::list_streams(&connection)?;
    let tf_samples = if streams.iter().any(|s| s == "tf") {
        memory2::read_tf(&connection, "tf")?
    } else {
        Vec::new()
    };
    let mut store_tf = tf::RecordingTf::from_samples(&tf_samples);
    if !odom_tf.is_empty() && streams.iter().any(|s| s == &odom_stream) {
        let (parent, child) = odom_tf
            .split_once(':')
            .map(|(parent, child)| (parent.to_string(), child.to_string()))
            .unwrap_or((odom_tf.clone(), String::new()));
        let trajectory: Vec<(f64, Pose3)> = odom_full
            .iter()
            .map(|row| {
                (
                    row.ts,
                    Pose3 {
                        rotation: row.rotation,
                        translation: row.translation,
                    },
                )
            })
            .collect();
        store_tf.override_edge(&parent, &child, trajectory);
    }
    let world_points = |scan: &memory2::ScanRow| -> Result<Vec<[f64; 3]>, String> {
        let pose = store_tf.get(&args.world_frame, &scan.frame_id, scan.ts)?;
        Ok(scan
            .points
            .iter()
            .map(|point| {
                let point = [point[0] as f64, point[1] as f64, point[2] as f64];
                mat3::add(&mat3::mat_vec(&pose.rotation, &point), &pose.translation)
            })
            .collect())
    };

    // base<-optical camera extrinsic: --base-optical override, else the tf tree
    let mut base_optical = Pose3::identity();
    if !best_factors.is_empty() {
        let odom_times: Vec<f64> = odom_rows.iter().map(|row| row[0]).collect();
        base_optical = resolve_base_optical(
            &store_tf,
            &body_frame,
            &optical_frame,
            // mid-run, not odom_rows[0]: tf typically starts a fraction of a second after
            // odometry, and a past-only lookup before the first tf sample breaks the chain
            median(&odom_times),
            &args.base_optical,
        )?;
    }

    // stage 1: tag PGO
    println!(
        "building factor graph over {} keyframes...",
        keyframe_poses.len()
    );
    let (mut graph, values, seen_markers) =
        pgo::build_tag_graph(&keyframe_poses, &best_factors, &base_optical, &tuning)?;
    println!("solving stage 1 (tag PGO)...");
    let mut estimate = pgo::solve(&graph, &values, &tuning)?;
    let raw_keyframe_poses = keyframe_poses.clone();

    // stage 2: ICP loop closures
    let scans = if args.icp || args.write_lidar {
        memory2::read_scans(&connection, &lidar_stream, 1)?
    } else {
        Vec::new()
    };
    if args.icp {
        let accepted = pgo::add_icp_closures(
            &mut graph,
            &estimate,
            &scans,
            &keyframe_poses,
            &keyframe_times,
            &world_points,
            args.closure_spacing,
            &tuning,
        )?;
        if accepted > 0 {
            println!("solving stage 2 (tag PGO + ICP closures)...");
            estimate = pgo::solve(&graph, &estimate, &tuning)?;
        }
    }

    // per-keyframe corrections
    let optimized: Vec<Pose3> = (0..keyframe_poses.len())
        .map(|index| {
            estimate
                .pose3(index as u64)
                .ok_or_else(|| format!("estimate missing keyframe pose {index}"))
        })
        .collect::<Result<_, _>>()?;
    let corrections: Vec<Pose3> = optimized
        .iter()
        .zip(&raw_keyframe_poses)
        .map(|(optimized_pose, raw_pose)| se3::compose(optimized_pose, &se3::inverse(raw_pose)))
        .collect();
    let max_shift = corrections
        .iter()
        .map(|correction| mat3::norm(&correction.translation))
        .fold(0.0f64, f64::max);
    println!(
        "PGO: {} keyframes, {} tag factors over {} markers, max correction shift {max_shift:.1} m",
        keyframe_poses.len(),
        best_factors.len(),
        seen_markers.len()
    );

    // persist PGO artifacts
    artifacts::write_deformation_nodes(
        &connection,
        &format!(
            "tf_deformation_nodes{}{}",
            args.corrected_suffix, args.suffix
        ),
        &keyframe_times,
        &raw_keyframe_poses,
        &optimized,
        &args.world_frame,
        &body_frame,
    )?;
    artifacts::write_pose_graph(
        &connection,
        &format!("pose_graph{}", args.suffix),
        &keyframe_times,
        &optimized,
        &args.world_frame,
    )?;

    let corrected_odom_out = format!("{odom_stream}{}{}", args.corrected_suffix, args.suffix);
    if args.write_odom {
        artifacts::write_corrected_odom(
            &connection,
            &corrected_odom_out,
            &odom_rows,
            &keyframe_times,
            &corrections,
            &args.world_frame,
            &args.corrected_odom_frame,
        )?;
    }

    if args.write_lidar {
        let lidar_out = format!("{lidar_stream}{}{}", args.corrected_suffix, args.suffix);
        let lcm_path = args.lcm.then(|| rec_dir.join(format!("{lidar_out}.pc2.lcm")));
        artifacts::write_corrected_lidar(
            &connection,
            &lidar_out,
            &scans,
            &odom_rows,
            &keyframe_times,
            &corrections,
            &world_points,
            lcm_path.as_deref(),
            args.lcm_voxel,
            &args.world_frame,
            &args.corrected_odom_frame,
        )?;
        if args.accum {
            artifacts::raycast_accumulate(
                &connection,
                &lidar_stream,
                &scans,
                &store_tf,
                &args.world_frame,
                args.accum_voxel,
                args.accum_max_range,
            )?;
            if memory2::list_streams(&connection)?
                .iter()
                .any(|s| s == &corrected_odom_out)
            {
                // tf that places the corrected-odom-framed per-scan clouds back into the
                // world; also supplies the ray origin for the corrected raycast.
                let corrected_odom_full =
                    memory2::read_odometry(&connection, &corrected_odom_out, 1)?;
                let mut corrected_store_tf = tf::RecordingTf::from_samples(&tf_samples);
                let trajectory: Vec<(f64, Pose3)> = corrected_odom_full
                    .iter()
                    .map(|row| {
                        (
                            row.ts,
                            Pose3 {
                                rotation: row.rotation,
                                translation: row.translation,
                            },
                        )
                    })
                    .collect();
                corrected_store_tf.override_edge(
                    &args.world_frame,
                    &args.corrected_odom_frame,
                    trajectory,
                );
                let corrected_scans = memory2::read_scans(&connection, &lidar_out, 1)?;
                artifacts::raycast_accumulate(
                    &connection,
                    &lidar_out,
                    &corrected_scans,
                    &corrected_store_tf,
                    &args.world_frame,
                    args.accum_voxel,
                    args.accum_max_range,
                )?;
            } else {
                println!(
                    "WARNING: no corrected odom stream (--no-odom?) -- skipping corrected lidar accumulation"
                );
            }
        }
        if args.rrd {
            icp_stitch::rrd::build_and_open_rrd(
                &connection,
                db_path,
                &lidar_stream,
                &odom_stream,
                &args.tags,
                &args.world_frame,
                &args.camera,
                &args.camera_info_stream,
            )?;
        }
    }
    Ok(())
}
