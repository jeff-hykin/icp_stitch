//! GTSAM tag-PGO + ICP-loop-closure solve pipeline (port of `offline_pgo.py`).

use crate::apriltags::Detection;
use crate::mat3::{self, Mat3, Vec3};
use crate::memory2::ScanRow;
use crate::pointcloud::KdTree;
use crate::se3;
use gtsam_shim::{FactorGraph, NoiseModel, Pose3, Values};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

// tag revisit report
pub const VISIT_GAP_S: f64 = 30.0;
pub const MIN_VISITS_FOR_LOOP: usize = 2;

// Quality weighting: planar-PnP pose error grows ~quadratically with range,
// and reproj_px is a direct misfit proxy.
const REF_DISTANCE_M: f64 = 0.4;
const REF_REPROJ_PX: f64 = 1.0;

// progress logging cadence
const ODOM_LOG_EVERY: usize = 20000;

fn gravity_anchor_noise() -> NoiseModel {
    NoiseModel::diagonal_variances(&[1e-8, 1e-8, 1e-6, 1e-8, 1e-8, 1e-8])
}

/// Every knob of the offline solve, defaults matching the tag-rig recordings
/// it was built on (== the python `Tuning` dataclass defaults).
#[derive(Clone, Debug)]
pub struct Tuning {
    pub keyframe_translation_m: f64,
    pub keyframe_rotation_deg: f64,
    pub lm_max_iterations: i32,
    /// Odometry between-factor variances (anisotropic): stiff roll/pitch and
    /// z, looser yaw and xy so the graph absorbs drift as heading error.
    pub odom_rot_roll_pitch_var: f64,
    pub odom_rot_yaw_var: f64,
    pub odom_trans_xy_var: f64,
    pub odom_trans_z_var: f64,
    /// Corrected positions must be within this to be a revisit candidate...
    pub icp_radius_m: f64,
    /// ...and at least this far apart in time (a real revisit, not adjacency).
    pub icp_min_dt_s: f64,
    pub icp_max_corr_m: f64,
    pub icp_voxel_m: f64,
    pub icp_fitness_min: f64,
    pub icp_rmse_max_m: f64,
    pub icp_huber_delta: f64,
    pub icp_rot_var: f64,
    pub icp_trans_var: f64,
    /// Accumulate scans within +/- this of a keyframe time into its submap.
    pub submap_half_s: f64,
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning {
            keyframe_translation_m: 0.5,
            keyframe_rotation_deg: 10.0,
            lm_max_iterations: 200,
            odom_rot_roll_pitch_var: 1e-8,
            odom_rot_yaw_var: 1e-5,
            odom_trans_xy_var: 1e-4,
            odom_trans_z_var: 1e-6,
            icp_radius_m: 4.0,
            icp_min_dt_s: 25.0,
            icp_max_corr_m: 0.6,
            icp_voxel_m: 0.15,
            icp_fitness_min: 0.45,
            icp_rmse_max_m: 0.25,
            icp_huber_delta: 1.345,
            icp_rot_var: 4e-4,
            icp_trans_var: 2.5e-3,
            submap_half_s: 1.0,
        }
    }
}

impl Tuning {
    pub fn odom_noise(&self) -> NoiseModel {
        NoiseModel::diagonal_variances(&[
            self.odom_rot_roll_pitch_var,
            self.odom_rot_roll_pitch_var,
            self.odom_rot_yaw_var,
            self.odom_trans_xy_var,
            self.odom_trans_xy_var,
            self.odom_trans_z_var,
        ])
    }

    pub fn icp_noise(&self) -> NoiseModel {
        let base = NoiseModel::diagonal_variances(&[
            self.icp_rot_var,
            self.icp_rot_var,
            self.icp_rot_var,
            self.icp_trans_var,
            self.icp_trans_var,
            self.icp_trans_var,
        ]);
        NoiseModel::robust_huber(self.icp_huber_delta, &base)
    }
}

/// One odometry row as fed to the solve: `[ts, x, y, z, qx, qy, qz, qw]`.
pub type OdomPoseRow = [f64; 8];

pub fn pose_from_row(row: &OdomPoseRow) -> Pose3 {
    se3::from_xyzquat(&row[1..])
}

/// Keyframe indices (into `odom_rows`) where the robot moved past the
/// translation/rotation thresholds, plus their poses and times.
pub fn select_keyframes(
    odom_rows: &[OdomPoseRow],
    tuning: &Tuning,
) -> (Vec<usize>, Vec<Pose3>, Vec<f64>) {
    let mut indices = vec![0usize];
    let mut previous = pose_from_row(&odom_rows[0]);
    for (row_index, row) in odom_rows.iter().enumerate().skip(1) {
        let current = pose_from_row(row);
        let moved = mat3::norm(&mat3::sub(&current.translation, &previous.translation));
        let relative = se3::between(&previous, &current);
        let omega = se3::so3_log(&relative.rotation);
        let turned = mat3::norm(&omega).to_degrees();
        if moved > tuning.keyframe_translation_m || turned > tuning.keyframe_rotation_deg {
            indices.push(row_index);
            previous = current;
        }
    }
    let poses: Vec<Pose3> = indices.iter().map(|&i| pose_from_row(&odom_rows[i])).collect();
    let times: Vec<f64> = indices.iter().map(|&i| odom_rows[i][0]).collect();
    (indices, poses, times)
}

/// Index of the keyframe time closest to `ts` (argmin of |times - ts|,
/// first minimum wins like `np.argmin`).
fn closest_keyframe(keyframe_times: &[f64], ts: f64) -> usize {
    let mut best = 0usize;
    let mut best_gap = f64::INFINITY;
    for (index, &time) in keyframe_times.iter().enumerate() {
        let gap = (time - ts).abs();
        if gap < best_gap {
            best_gap = gap;
            best = index;
        }
    }
    best
}

/// One factor per (keyframe, marker): the filtered detection with the lowest
/// reproj error. BTreeMap so downstream iteration matches python's
/// `sorted(best_factors.items())`.
pub fn best_factor_per_keyframe_marker(
    detections: &[Detection],
    keyframe_times: &[f64],
) -> BTreeMap<(usize, i64), Detection> {
    let mut best: BTreeMap<(usize, i64), Detection> = BTreeMap::new();
    for detection in detections {
        let keyframe = closest_keyframe(keyframe_times, detection.ts);
        let key = (keyframe, detection.marker_id);
        match best.get(&key) {
            Some(existing) if detection.reproj_px >= existing.reproj_px => {}
            _ => {
                best.insert(key, detection.clone());
            }
        }
    }
    best
}

/// Number of temporally-separated visits in a list of timestamps.
pub fn count_visits(times: &[f64]) -> usize {
    let mut times = times.to_vec();
    times.sort_by(|a, b| a.total_cmp(b));
    let mut visits = 1usize;
    let mut last = times[0];
    for &value in &times[1..] {
        if value - last > VISIT_GAP_S {
            visits += 1;
        }
        last = value;
    }
    visits
}

/// Print per-marker raw viewings + filtered revisits, flagging tags with no
/// loop closure.
pub fn report_revisits(
    detections: &[Detection],
    best_factors: &BTreeMap<(usize, i64), Detection>,
) {
    let mut raw_by_marker: BTreeMap<i64, usize> = BTreeMap::new();
    for detection in detections {
        *raw_by_marker.entry(detection.marker_id).or_default() += 1;
    }
    let mut visit_times: HashMap<i64, Vec<f64>> = HashMap::new();
    for ((_keyframe, marker_id), detection) in best_factors {
        visit_times.entry(*marker_id).or_default().push(detection.ts);
    }
    println!("{:>4} | {:>12} | {:>17}", "tag", "raw viewings", "filtered revisits");
    let mut not_revisited: Vec<i64> = Vec::new();
    for (&marker_id, &raw_count) in &raw_by_marker {
        let visits = visit_times
            .get(&marker_id)
            .map(|times| count_visits(times))
            .unwrap_or(0);
        let flag = if visits >= MIN_VISITS_FOR_LOOP {
            ""
        } else {
            "   <-- NOT REVISITED"
        };
        println!("{marker_id:>4} | {raw_count:>12} | {visits:>10} visit(s){flag}");
        if visits < MIN_VISITS_FOR_LOOP {
            not_revisited.push(marker_id);
        }
    }
    if not_revisited.is_empty() {
        println!("\ntags with no loop-closure constraint: none\n");
    } else {
        println!("\ntags with no loop-closure constraint: {not_revisited:?}\n");
    }
}

/// Range/reproj-inflated Gaussian covariance for a single AprilTag landmark
/// factor. `covariance[:3,:3]` rotates diag(0.04, 0.04, 0.0025) into the tag
/// frame, `[3:,3:]` rotates diag(0.0025, 0.0025, 0.25).
pub fn tag_noise(tag_rotation: &Mat3, distance_m: f64, reproj_px: f64) -> NoiseModel {
    let scale = ((distance_m.max(0.2) / REF_DISTANCE_M).powi(2)
        * (reproj_px.max(0.5) / REF_REPROJ_PX).powi(2))
    .max(0.25);
    let rotate = |diagonal: [f64; 3]| -> Mat3 {
        let mut scaled = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                scaled[i][j] = tag_rotation[i][j] * diagonal[j];
            }
        }
        mat3::mat_mul(&scaled, &mat3::transpose(tag_rotation))
    };
    let rot_block = rotate([0.04, 0.04, 0.0025]);
    let trans_block = rotate([0.0025, 0.0025, 0.25]);
    let mut covariance = [0.0f64; 36];
    for i in 0..3 {
        for j in 0..3 {
            covariance[i * 6 + j] = rot_block[i][j] * scale;
            covariance[(i + 3) * 6 + (j + 3)] = trans_block[i][j] * scale;
        }
    }
    NoiseModel::gaussian_covariance(&covariance)
}

/// Stage-1 graph: sequential odom between-factors + AprilTag landmark
/// factors. Pose keys are the plain keyframe indices; landmarks are
/// `Symbol('l', marker_id)`.
pub fn build_tag_graph(
    keyframe_poses: &[Pose3],
    best_factors: &BTreeMap<(usize, i64), Detection>,
    base_optical: &Pose3,
    tuning: &Tuning,
) -> Result<(FactorGraph, Values, HashSet<i64>), String> {
    let mut graph = FactorGraph::new();
    let mut values = Values::new();
    let odom_noise = tuning.odom_noise();
    for (index, pose) in keyframe_poses.iter().enumerate() {
        values
            .insert_pose3(index as u64, pose)
            .map_err(|e| e.to_string())?;
        if index == 0 {
            graph
                .add_prior_pose3(0, pose, &gravity_anchor_noise())
                .map_err(|e| e.to_string())?;
        } else {
            let relative = se3::between(&keyframe_poses[index - 1], pose);
            graph
                .add_between_pose3(index as u64 - 1, index as u64, &relative, &odom_noise)
                .map_err(|e| e.to_string())?;
        }
    }
    let mut seen_markers: HashSet<i64> = HashSet::new();
    for ((keyframe, marker_id), detection) in best_factors {
        let base_tag = se3::compose(base_optical, &se3::from_xyzquat(&detection.t_cam_marker));
        let landmark_key = gtsam_shim::symbol_key('l', *marker_id as u64);
        if seen_markers.insert(*marker_id) {
            values
                .insert_pose3(
                    landmark_key,
                    &se3::compose(&keyframe_poses[*keyframe], &base_tag),
                )
                .map_err(|e| e.to_string())?;
        }
        graph
            .add_between_pose3(
                *keyframe as u64,
                landmark_key,
                &base_tag,
                &tag_noise(&base_tag.rotation, detection.distance_m, detection.reproj_px),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok((graph, values, seen_markers))
}

/// Batch Levenberg-Marquardt with `tuning.lm_max_iterations`.
pub fn solve(graph: &FactorGraph, values: &Values, tuning: &Tuning) -> Result<Values, String> {
    let (optimized, _, _) = graph
        .lm_optimize(values, tuning.lm_max_iterations)
        .map_err(|e| e.to_string())?;
    Ok(optimized)
}

// ---------------------------------------------------------------------------
// Submaps + point-to-plane ICP (open3d `registration_icp` semantics)
// ---------------------------------------------------------------------------

/// Body-frame, voxel-downsampled lidar submap with per-point normals.
pub struct Submap {
    pub points: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    points_f32: Vec<[f32; 3]>,
}

/// open3d `voxel_down_sample`: cells indexed relative to the cloud's min
/// bound, each occupied cell emits the centroid of its points.
fn o3d_voxel_downsample(points: &[[f64; 3]], voxel: f64) -> Vec<[f64; 3]> {
    if points.is_empty() || voxel <= 0.0 {
        return points.to_vec();
    }
    let mut min_bound = [f64::INFINITY; 3];
    for p in points {
        for i in 0..3 {
            min_bound[i] = min_bound[i].min(p[i]);
        }
    }
    let mut cells: BTreeMap<(i64, i64, i64), ([f64; 3], u32)> = BTreeMap::new();
    for p in points {
        let key = (
            ((p[0] - min_bound[0]) / voxel).floor() as i64,
            ((p[1] - min_bound[1]) / voxel).floor() as i64,
            ((p[2] - min_bound[2]) / voxel).floor() as i64,
        );
        let entry = cells.entry(key).or_insert(([0.0; 3], 0));
        for i in 0..3 {
            entry.0[i] += p[i];
        }
        entry.1 += 1;
    }
    cells
        .values()
        .map(|(sum, count)| {
            let n = *count as f64;
            [sum[0] / n, sum[1] / n, sum[2] / n]
        })
        .collect()
}

/// open3d hybrid-search normal estimation (`KDTreeSearchParamHybrid`): up to
/// `max_nn` nearest neighbours within `radius`, smallest eigenvector of the
/// neighbourhood covariance; `(0, 0, 1)` when the neighbourhood is too small.
/// Orientation is arbitrary (point-to-plane residuals square it away).
fn estimate_normals_hybrid(points: &[[f64; 3]], radius: f64, max_nn: usize) -> Vec<[f64; 3]> {
    use rayon::prelude::*;
    let points_f32: Vec<[f32; 3]> = points
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let tree = KdTree::build(&points_f32);
    let radius_sq = (radius * radius) as f32;
    points_f32
        .par_iter()
        .map(|query| {
            let neighbors: Vec<(usize, f32)> = tree
                .knn(query, max_nn)
                .into_iter()
                .filter(|&(_, d)| d <= radius_sq)
                .collect();
            if neighbors.len() < 3 {
                return [0.0, 0.0, 1.0];
            }
            let mut mean = [0.0f64; 3];
            for &(idx, _) in &neighbors {
                for i in 0..3 {
                    mean[i] += points[idx][i];
                }
            }
            let count = neighbors.len() as f64;
            for value in &mut mean {
                *value /= count;
            }
            let mut covariance = [[0.0f64; 3]; 3];
            for &(idx, _) in &neighbors {
                let d = mat3::sub(&points[idx], &mean);
                for i in 0..3 {
                    for j in 0..3 {
                        covariance[i][j] += d[i] * d[j];
                    }
                }
            }
            let (_, vectors) = mat3::jacobi_eigen(covariance);
            // Smallest eigenvalue's eigenvector = column 0.
            [vectors[0][0], vectors[1][0], vectors[2][0]]
        })
        .collect()
}

/// Body-frame, voxel-downsampled, normal-estimated lidar submap per involved
/// keyframe. `world_points` maps a scan to world-frame f64 points (the tf
/// lookup lives with the caller).
pub fn build_submaps(
    scans: &[ScanRow],
    keyframe_indices: &HashSet<usize>,
    keyframe_poses: &[Pose3],
    keyframe_times: &[f64],
    world_points: &dyn Fn(&ScanRow) -> Result<Vec<[f64; 3]>, String>,
    tuning: &Tuning,
) -> Result<HashMap<usize, Submap>, String> {
    let mut chunks: HashMap<usize, Vec<[f64; 3]>> = HashMap::new();
    let started = Instant::now();
    for (scan_count, scan) in scans.iter().enumerate() {
        if scan_count > 0 && scan_count % ODOM_LOG_EVERY == 0 {
            println!(
                "  read {scan_count} scans, {:.0}s",
                started.elapsed().as_secs_f64()
            );
        }
        let keyframe = closest_keyframe(keyframe_times, scan.ts);
        if !keyframe_indices.contains(&keyframe)
            || (keyframe_times[keyframe] - scan.ts).abs() > tuning.submap_half_s
        {
            continue;
        }
        let pose = &keyframe_poses[keyframe];
        let rotation_t = mat3::transpose(&pose.rotation);
        let chunk = chunks.entry(keyframe).or_default();
        for world in world_points(scan)? {
            // (world - t) @ R  ==  R^T (world - t) per point.
            let body = mat3::mat_vec(&rotation_t, &mat3::sub(&world, &pose.translation));
            chunk.push(body);
        }
    }
    let mut clouds: HashMap<usize, Submap> = HashMap::new();
    for (keyframe, points) in chunks {
        if points.is_empty() {
            continue;
        }
        let points = o3d_voxel_downsample(&points, tuning.icp_voxel_m);
        let normals = estimate_normals_hybrid(&points, 0.5, 30);
        let points_f32 = points
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        clouds.insert(
            keyframe,
            Submap {
                points,
                normals,
                points_f32,
            },
        );
    }
    Ok(clouds)
}

/// Registration outcome, open3d `RegistrationResult` semantics:
/// `fitness` = correspondences / source points, `inlier_rmse` = RMS Euclidean
/// correspondence distance.
pub struct RegistrationResult {
    pub fitness: f64,
    pub inlier_rmse: f64,
    pub transform: Pose3,
}

fn evaluate_registration(
    source: &[[f64; 3]],
    target: &Submap,
    target_tree: &KdTree,
    transform: &Pose3,
    max_corr: f64,
) -> (Vec<(usize, usize)>, f64, f64) {
    let max_corr_sq = (max_corr * max_corr) as f32;
    let mut correspondences = Vec::new();
    let mut squared_sum = 0.0f64;
    for (source_index, point) in source.iter().enumerate() {
        let moved = mat3::add(&mat3::mat_vec(&transform.rotation, point), &transform.translation);
        let query = [moved[0] as f32, moved[1] as f32, moved[2] as f32];
        if let Some((target_index, dist_sq)) = target_tree.nearest(&query) {
            if dist_sq <= max_corr_sq {
                correspondences.push((source_index, target_index));
                let d = mat3::sub(&moved, &target.points[target_index]);
                squared_sum += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            }
        }
    }
    let fitness = if source.is_empty() {
        0.0
    } else {
        correspondences.len() as f64 / source.len() as f64
    };
    let rmse = if correspondences.is_empty() {
        0.0
    } else {
        (squared_sum / correspondences.len() as f64).sqrt()
    };
    (correspondences, fitness, rmse)
}

/// Solve the 6x6 normal equations `jtj x = -jtr` by Gaussian elimination with
/// partial pivoting; `None` on a (near-)singular system.
fn solve_6x6(jtj: &[[f64; 6]; 6], jtr: &[f64; 6]) -> Option<[f64; 6]> {
    let mut a = *jtj;
    let mut b = [0.0f64; 6];
    for i in 0..6 {
        b[i] = -jtr[i];
    }
    for col in 0..6 {
        let mut pivot = col;
        for row in (col + 1)..6 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..6 {
            let factor = a[row][col] / a[col][col];
            for k in col..6 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0f64; 6];
    for row in (0..6).rev() {
        let mut sum = b[row];
        for k in (row + 1)..6 {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

fn rot_x(angle: f64) -> Mat3 {
    let (s, c) = angle.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

/// open3d `TransformVector6dToMatrix4d`: rotation = Rz(g) * Ry(b) * Rx(a),
/// translation = (tx, ty, tz) for x = [a, b, g, tx, ty, tz].
fn pose_from_xi(xi: &[f64; 6]) -> Pose3 {
    let rotation = mat3::mat_mul(
        &mat3::mat_mul(&mat3::rot_z(xi[2]), &mat3::rot_y(xi[1])),
        &rot_x(xi[0]),
    );
    Pose3 {
        rotation,
        translation: [xi[3], xi[4], xi[5]],
    }
}

fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// open3d `registration_icp` with `TransformationEstimationPointToPlane` and
/// default `ICPConvergenceCriteria` (max 30 iterations, relative fitness and
/// relative RMSE thresholds 1e-6). Per iteration: point-to-plane Gauss-Newton
/// step from the current correspondences (residual `(R p + t - q) . n`,
/// jacobian `[(R p + t) x n ; n]`), applied as a small-angle Euler update.
pub fn icp_point_to_plane(
    source: &Submap,
    target: &Submap,
    initial: &Pose3,
    max_corr: f64,
) -> RegistrationResult {
    const MAX_ITERATIONS: usize = 30;
    const RELATIVE_FITNESS: f64 = 1e-6;
    const RELATIVE_RMSE: f64 = 1e-6;
    let target_tree = KdTree::build(&target.points_f32);
    let mut transform = *initial;
    let (mut correspondences, mut fitness, mut rmse) =
        evaluate_registration(&source.points, target, &target_tree, &transform, max_corr);
    for _ in 0..MAX_ITERATIONS {
        if correspondences.is_empty() {
            break;
        }
        let mut jtj = [[0.0f64; 6]; 6];
        let mut jtr = [0.0f64; 6];
        for &(source_index, target_index) in &correspondences {
            let moved = mat3::add(
                &mat3::mat_vec(&transform.rotation, &source.points[source_index]),
                &transform.translation,
            );
            let normal = target.normals[target_index];
            let residual = (moved[0] - target.points[target_index][0]) * normal[0]
                + (moved[1] - target.points[target_index][1]) * normal[1]
                + (moved[2] - target.points[target_index][2]) * normal[2];
            let rotational = cross(&moved, &normal);
            let jacobian = [
                rotational[0],
                rotational[1],
                rotational[2],
                normal[0],
                normal[1],
                normal[2],
            ];
            for i in 0..6 {
                for j in 0..6 {
                    jtj[i][j] += jacobian[i] * jacobian[j];
                }
                jtr[i] += jacobian[i] * residual;
            }
        }
        let Some(xi) = solve_6x6(&jtj, &jtr) else {
            break;
        };
        let update = pose_from_xi(&xi);
        transform = Pose3 {
            rotation: mat3::mat_mul(&update.rotation, &transform.rotation),
            translation: mat3::add(
                &mat3::mat_vec(&update.rotation, &transform.translation),
                &update.translation,
            ),
        };
        let (new_correspondences, new_fitness, new_rmse) =
            evaluate_registration(&source.points, target, &target_tree, &transform, max_corr);
        let converged = (new_fitness - fitness).abs() < RELATIVE_FITNESS
            && (new_rmse - rmse).abs() < RELATIVE_RMSE;
        correspondences = new_correspondences;
        fitness = new_fitness;
        rmse = new_rmse;
        if converged {
            break;
        }
    }
    RegistrationResult {
        fitness,
        inlier_rmse: rmse,
        transform,
    }
}

/// Keep at most one index pair per `section_length` meters of path. Greedy in
/// the given pair order; `section_length <= 0` keeps everything.
pub fn thin_pairs_by_path_section(
    pairs: &[(usize, usize)],
    positions: &[Vec3],
    section_length: f64,
) -> Vec<(usize, usize)> {
    if section_length <= 0.0 {
        return pairs.to_vec();
    }
    let mut arc_length = vec![0.0f64; positions.len()];
    for i in 1..positions.len() {
        arc_length[i] =
            arc_length[i - 1] + mat3::norm(&mat3::sub(&positions[i], &positions[i - 1]));
    }
    let sections: Vec<i64> = arc_length
        .iter()
        .map(|&arc| (arc / section_length).floor() as i64)
        .collect();
    let mut used: HashSet<i64> = HashSet::new();
    let mut kept = Vec::new();
    for &(first, second) in pairs {
        if used.contains(&sections[first]) || used.contains(&sections[second]) {
            continue;
        }
        used.insert(sections[first]);
        used.insert(sections[second]);
        kept.push((first, second));
    }
    kept
}

/// Stage 2: register spatially-close / temporally-distant submaps, add loop
/// factors. Returns the number of accepted closures.
#[allow(clippy::too_many_arguments)]
pub fn add_icp_closures(
    graph: &mut FactorGraph,
    estimate: &Values,
    scans: &[ScanRow],
    keyframe_poses: &[Pose3],
    keyframe_times: &[f64],
    world_points: &dyn Fn(&ScanRow) -> Result<Vec<[f64; 3]>, String>,
    closure_spacing: f64,
    tuning: &Tuning,
) -> Result<usize, String> {
    let num_keyframes = keyframe_poses.len();
    let corrected_poses: Vec<Pose3> = (0..num_keyframes)
        .map(|index| {
            estimate
                .pose3(index as u64)
                .ok_or_else(|| format!("estimate missing keyframe pose {index}"))
        })
        .collect::<Result<_, String>>()?;
    let positions: Vec<Vec3> = corrected_poses.iter().map(|p| p.translation).collect();

    let radius_sq = tuning.icp_radius_m * tuning.icp_radius_m;
    let mut candidate_pairs: Vec<(usize, usize)> = Vec::new();
    for first in 0..num_keyframes {
        for second in (first + 1)..num_keyframes {
            let d = mat3::sub(&positions[second], &positions[first]);
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > radius_sq {
                continue;
            }
            if (keyframe_times[first] - keyframe_times[second]).abs() >= tuning.icp_min_dt_s {
                candidate_pairs.push((first, second));
            }
        }
    }
    candidate_pairs.sort_by(|a, b| {
        let da = mat3::norm(&mat3::sub(&positions[a.0], &positions[a.1]));
        let db = mat3::norm(&mat3::sub(&positions[b.0], &positions[b.1]));
        da.total_cmp(&db)
    });
    let total_candidates = candidate_pairs.len();
    let candidate_pairs =
        thin_pairs_by_path_section(&candidate_pairs, &positions, closure_spacing);
    println!(
        "ICP stage: thinned {total_candidates} -> {} pairs (one per {closure_spacing} m of path)",
        candidate_pairs.len()
    );
    let involved: HashSet<usize> = candidate_pairs
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .collect();
    println!(
        "ICP stage: {} candidate pairs over {} keyframes",
        candidate_pairs.len(),
        involved.len()
    );
    if candidate_pairs.is_empty() {
        return Ok(0);
    }

    println!("ICP stage: reading lidar submaps...");
    let clouds = build_submaps(
        scans,
        &involved,
        keyframe_poses,
        keyframe_times,
        world_points,
        tuning,
    )?;
    println!(
        "ICP stage: built {} submaps, registering {} pairs...",
        clouds.len(),
        candidate_pairs.len()
    );

    let mut accepted = 0usize;
    let icp_noise = tuning.icp_noise();
    let started = Instant::now();
    for (pair_index, &(first, second)) in candidate_pairs.iter().enumerate() {
        if pair_index > 0 && pair_index % 5000 == 0 {
            println!(
                "  {pair_index}/{} pairs, {accepted} accepted, {:.0}s",
                candidate_pairs.len(),
                started.elapsed().as_secs_f64()
            );
        }
        let (Some(target), Some(source)) = (clouds.get(&first), clouds.get(&second)) else {
            continue;
        };
        let initial_guess = se3::between(&corrected_poses[first], &corrected_poses[second]);
        let result = icp_point_to_plane(source, target, &initial_guess, tuning.icp_max_corr_m);
        if result.fitness >= tuning.icp_fitness_min && result.inlier_rmse <= tuning.icp_rmse_max_m
        {
            graph
                .add_between_pose3(first as u64, second as u64, &result.transform, &icp_noise)
                .map_err(|e| e.to_string())?;
            accepted += 1;
        }
    }
    println!(
        "ICP stage: accepted {accepted}/{} loop closures",
        candidate_pairs.len()
    );
    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(ts: f64, x: f64, y: f64) -> OdomPoseRow {
        [ts, x, y, 0.0, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn keyframes_by_translation() {
        let rows = vec![
            row(0.0, 0.0, 0.0),
            row(1.0, 0.2, 0.0),
            row(2.0, 0.6, 0.0),
            row(3.0, 0.7, 0.0),
            row(4.0, 1.3, 0.0),
        ];
        let (indices, poses, times) = select_keyframes(&rows, &Tuning::default());
        assert_eq!(indices, vec![0, 2, 4]);
        assert_eq!(poses.len(), 3);
        assert_eq!(times, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn visits_split_on_gap() {
        assert_eq!(count_visits(&[0.0, 1.0, 2.0]), 1);
        assert_eq!(count_visits(&[0.0, 1.0, 100.0, 101.0, 300.0]), 3);
    }

    #[test]
    fn thinning_keeps_one_pair_per_section() {
        let positions: Vec<Vec3> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
        // Sections at length 2: indices 0-1 -> 0, 2-3 -> 1, 4-5 -> 2, ...
        let pairs = vec![(0, 9), (2, 5), (1, 8)];
        let kept = thin_pairs_by_path_section(&pairs, &positions, 2.0);
        // (0,9) claims sections 0 and 4; (2,5) claims 1 and 2; (1,8) collides.
        assert_eq!(kept, vec![(0, 9), (2, 5)]);
        assert_eq!(thin_pairs_by_path_section(&pairs, &positions, 0.0), pairs);
    }

    #[test]
    fn tag_graph_solves_and_pulls_revisit_together() {
        // Keyframes drift +x; the same tag is seen at kf 0 and kf 3 with the
        // same camera-frame pose, so the solve should pull kf 3 back.
        let keyframe_poses: Vec<Pose3> = [0.0, 1.0, 2.0, 0.3]
            .iter()
            .map(|&x| Pose3::from_translation([x, 0.0, 0.0]))
            .collect();
        let detection = |ts: f64| Detection {
            ts,
            marker_id: 7,
            t_cam_marker: [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
            sharpness: 100.0,
            reproj_px: 0.5,
            tag_px: 40.0,
            distance_m: 0.5,
            view_angle_deg: 0.0,
            lin_speed: -1.0,
            ang_speed: -1.0,
        };
        let mut best = BTreeMap::new();
        best.insert((0usize, 7i64), detection(0.0));
        best.insert((3usize, 7i64), detection(3.0));
        let tuning = Tuning::default();
        let (graph, values, seen) =
            build_tag_graph(&keyframe_poses, &best, &Pose3::identity(), &tuning).unwrap();
        assert_eq!(seen, HashSet::from([7]));
        assert_eq!(graph.len(), 1 + 3 + 2);
        let estimate = solve(&graph, &values, &tuning).unwrap();
        let kf3 = estimate.pose3(3).unwrap();
        // Identical sightings pull kf3 toward kf0. The odom chain (3 between
        // factors, xy var 1e-4) against the two tag factors (x var ~1e-3
        // each) puts the Gaussian posterior at x ~= 0.26 of the 0.3 guess.
        assert!(
            (kf3.translation[0] - 0.26).abs() < 0.02,
            "kf3 {:?}",
            kf3.translation
        );
        assert!(estimate.pose3(gtsam_shim::symbol_key('l', 7)).is_some());
    }

    #[test]
    fn icp_recovers_small_offset() {
        // A dense L-shaped wall pattern so point-to-plane is well constrained.
        let mut points: Vec<[f64; 3]> = Vec::new();
        for i in 0..40 {
            for j in 0..10 {
                points.push([i as f64 * 0.05, 0.0, j as f64 * 0.05]);
                points.push([0.0, i as f64 * 0.05, j as f64 * 0.05]);
                points.push([i as f64 * 0.05, j as f64 * 0.05, 0.0]);
            }
        }
        let normals = estimate_normals_hybrid(&points, 0.5, 30);
        let points_f32 = points
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let target = Submap {
            points: points.clone(),
            normals: normals.clone(),
            points_f32,
        };
        // Source = target shifted by a small offset; ICP should undo it.
        let offset = [0.04, -0.03, 0.02];
        let shifted: Vec<[f64; 3]> = points.iter().map(|p| mat3::add(p, &offset)).collect();
        let shifted_f32 = shifted
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let source = Submap {
            points: shifted,
            normals,
            points_f32: shifted_f32,
        };
        let result = icp_point_to_plane(&source, &target, &Pose3::identity(), 0.6);
        assert!(result.fitness > 0.9, "fitness {}", result.fitness);
        assert!(result.inlier_rmse < 0.02, "rmse {}", result.inlier_rmse);
        for i in 0..3 {
            assert!(
                (result.transform.translation[i] + offset[i]).abs() < 0.01,
                "translation {:?}",
                result.transform.translation
            );
        }
    }
}
