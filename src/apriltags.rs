//! AprilTag glimpse types + per-glimpse quality gating (port of the read/filter
//! side of dimos `apriltags.py`; detection itself lives in `detect.rs`).

use crate::mat3;
use std::collections::HashSet;

/// One raw AprilTag glimpse from the `raw_april_tags` stream.
#[derive(Clone, Debug)]
pub struct Detection {
    pub ts: f64,
    pub marker_id: i64,
    /// camera<-marker pose [x, y, z, qx, qy, qz, qw]
    pub t_cam_marker: [f64; 7],
    pub sharpness: f64,
    pub reproj_px: f64,
    pub tag_px: f64,
    /// Annotated by `filter_glimpses` (from the pose); raw rows may carry them too.
    pub distance_m: f64,
    pub view_angle_deg: f64,
    /// -1.0 encodes "unknown" (no odometry context at detection time).
    pub lin_speed: f64,
    pub ang_speed: f64,
}

/// Per-glimpse quality gates (defaults = apriltags.py module defaults).
#[derive(Clone, Debug)]
pub struct GlimpseGates {
    pub min_sharpness: f64,
    pub max_reproj_px: f64,
    pub min_tag_px: f64,
    pub max_distance_m: f64,
    pub max_view_angle_deg: f64,
    pub max_linear_speed_mps: f64,
    pub max_angular_speed_dps: f64,
}

impl Default for GlimpseGates {
    fn default() -> GlimpseGates {
        GlimpseGates {
            min_sharpness: 60.0,
            max_reproj_px: 2.0,
            min_tag_px: 24.0,
            max_distance_m: 1.0,
            max_view_angle_deg: 45.0,
            max_linear_speed_mps: 0.5,
            max_angular_speed_dps: 50.0,
        }
    }
}

/// (distance_m, view_angle_deg) for a tag pose in the camera optical frame.
/// View angle is between the line of sight and the tag's surface normal
/// (the rotation's third column); 0 = head-on.
pub fn view_quality(t_cam_marker: &[f64; 7]) -> (f64, f64) {
    let translation = [t_cam_marker[0], t_cam_marker[1], t_cam_marker[2]];
    let distance = mat3::norm(&translation);
    let [qx, qy, qz, qw] = [t_cam_marker[3], t_cam_marker[4], t_cam_marker[5], t_cam_marker[6]];
    let rotation = mat3::mat_from_quat(&[qw, qx, qy, qz]);
    let normal = [rotation[0][2], rotation[1][2], rotation[2][2]];
    let line_of_sight = [
        translation[0] / (distance + 1e-9),
        translation[1] / (distance + 1e-9),
        translation[2] / (distance + 1e-9),
    ];
    let cos_angle = (line_of_sight[0] * normal[0]
        + line_of_sight[1] * normal[1]
        + line_of_sight[2] * normal[2])
        .abs()
        .min(1.0);
    (distance, cos_angle.acos().to_degrees())
}

/// None when a glimpse passes every gate, else the rejection reason.
pub fn glimpse_passes(detection: &Detection, gates: &GlimpseGates) -> Option<&'static str> {
    if detection.sharpness < gates.min_sharpness {
        return Some("blur");
    }
    if detection.reproj_px > gates.max_reproj_px {
        return Some("reproj");
    }
    if detection.tag_px < gates.min_tag_px {
        return Some("small");
    }
    let (distance, view_angle) = view_quality(&detection.t_cam_marker);
    if distance > gates.max_distance_m {
        return Some("far");
    }
    if view_angle > gates.max_view_angle_deg {
        return Some("oblique");
    }
    let speed_known = detection.lin_speed >= 0.0;
    if speed_known
        && (detection.lin_speed > gates.max_linear_speed_mps
            || detection.ang_speed > gates.max_angular_speed_dps)
    {
        return Some("motion");
    }
    None
}

/// Per-glimpse gating with NO clustering: every clean sighting survives,
/// annotated with distance_m / view_angle_deg for downstream quality weighting.
pub fn filter_glimpses(
    raw_detections: &[Detection],
    exclude_tags: &HashSet<i64>,
    gates: &GlimpseGates,
) -> Vec<Detection> {
    let mut kept = Vec::new();
    for detection in raw_detections {
        if exclude_tags.contains(&detection.marker_id) {
            continue;
        }
        if glimpse_passes(detection, gates).is_some() {
            continue;
        }
        let (distance_m, view_angle_deg) = view_quality(&detection.t_cam_marker);
        kept.push(Detection {
            distance_m,
            view_angle_deg,
            ..detection.clone()
        });
    }
    kept
}
