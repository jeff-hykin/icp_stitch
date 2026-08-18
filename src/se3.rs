//! SE(3) helpers over `gtsam_shim::Pose3` (plain rotation matrix + translation),
//! matching gtsam's Pose3 Expmap/Logmap conventions (tangent = [omega, v]).

use crate::mat3::{self, Mat3, Vec3};
use gtsam_shim::Pose3;

pub fn from_xyzquat(row: &[f64]) -> Pose3 {
    // input is xyzw; mat_from_quat wants wxyz
    let quaternion = [row[6], row[3], row[4], row[5]];
    Pose3 {
        rotation: mat3::mat_from_quat(&quaternion),
        translation: [row[0], row[1], row[2]],
    }
}

pub fn quaternion_xyzw(pose: &Pose3) -> [f64; 4] {
    let [w, x, y, z] = mat3::quat_from_mat(&pose.rotation);
    [x, y, z, w]
}

pub fn compose(a: &Pose3, b: &Pose3) -> Pose3 {
    Pose3 {
        rotation: mat3::mat_mul(&a.rotation, &b.rotation),
        translation: mat3::add(&mat3::mat_vec(&a.rotation, &b.translation), &a.translation),
    }
}

pub fn inverse(pose: &Pose3) -> Pose3 {
    let rotation_t = mat3::transpose(&pose.rotation);
    let translation = mat3::mat_vec(&rotation_t, &pose.translation);
    Pose3 {
        rotation: rotation_t,
        translation: [-translation[0], -translation[1], -translation[2]],
    }
}

/// `a.between(b)` = a⁻¹ ∘ b.
pub fn between(a: &Pose3, b: &Pose3) -> Pose3 {
    compose(&inverse(a), b)
}

fn skew(v: &Vec3) -> Mat3 {
    [
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ]
}

fn mat_scale(m: &Mat3, s: f64) -> Mat3 {
    let mut out = *m;
    for row in &mut out {
        for value in row {
            *value *= s;
        }
    }
    out
}

fn mat_add(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = *a;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] += b[i][j];
        }
    }
    out
}

pub fn so3_log(rotation: &Mat3) -> Vec3 {
    let trace = rotation[0][0] + rotation[1][1] + rotation[2][2];
    let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    let axis_raw = [
        rotation[2][1] - rotation[1][2],
        rotation[0][2] - rotation[2][0],
        rotation[1][0] - rotation[0][1],
    ];
    if angle < 1e-10 {
        return [axis_raw[0] / 2.0, axis_raw[1] / 2.0, axis_raw[2] / 2.0];
    }
    if (std::f64::consts::PI - angle).abs() < 1e-6 {
        // Near pi the off-diagonal difference vanishes; recover the axis from
        // the diagonal of (R + I) / 2 = axis axisᵀ at exactly pi.
        let axis = [
            ((rotation[0][0] + 1.0) / 2.0).max(0.0).sqrt(),
            ((rotation[1][1] + 1.0) / 2.0).max(0.0).sqrt(),
            ((rotation[2][2] + 1.0) / 2.0).max(0.0).sqrt(),
        ];
        let signs = [
            1.0,
            if rotation[0][1] + rotation[1][0] >= 0.0 { 1.0 } else { -1.0 },
            if rotation[0][2] + rotation[2][0] >= 0.0 { 1.0 } else { -1.0 },
        ];
        let mut out = [0.0; 3];
        for i in 0..3 {
            out[i] = angle * axis[i] * signs[i];
        }
        return out;
    }
    let scale = angle / (2.0 * angle.sin());
    [axis_raw[0] * scale, axis_raw[1] * scale, axis_raw[2] * scale]
}

pub fn so3_exp(omega: &Vec3) -> Mat3 {
    let angle = mat3::norm(omega);
    if angle < 1e-10 {
        return mat_add(&mat3::identity(), &skew(omega));
    }
    let k = skew(&[omega[0] / angle, omega[1] / angle, omega[2] / angle]);
    let k2 = mat3::mat_mul(&k, &k);
    mat_add(
        &mat_add(&mat3::identity(), &mat_scale(&k, angle.sin())),
        &mat_scale(&k2, 1.0 - angle.cos()),
    )
}

/// Left Jacobian V of SO(3): Expmap t = V v.
fn so3_left_jacobian(omega: &Vec3) -> Mat3 {
    let angle = mat3::norm(omega);
    let omega_hat = skew(omega);
    let omega_hat2 = mat3::mat_mul(&omega_hat, &omega_hat);
    if angle < 1e-8 {
        return mat_add(&mat3::identity(), &mat_scale(&omega_hat, 0.5));
    }
    let a = (1.0 - angle.cos()) / (angle * angle);
    let b = (angle - angle.sin()) / (angle * angle * angle);
    mat_add(
        &mat_add(&mat3::identity(), &mat_scale(&omega_hat, a)),
        &mat_scale(&omega_hat2, b),
    )
}

fn mat_inverse(m: &Mat3) -> Mat3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv_det = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

/// gtsam `Pose3::Logmap`: tangent [omega(3), v(3)].
pub fn logmap(pose: &Pose3) -> [f64; 6] {
    let omega = so3_log(&pose.rotation);
    let v = mat3::mat_vec(&mat_inverse(&so3_left_jacobian(&omega)), &pose.translation);
    [omega[0], omega[1], omega[2], v[0], v[1], v[2]]
}

/// gtsam `Pose3::Expmap`.
pub fn expmap(xi: &[f64; 6]) -> Pose3 {
    let omega = [xi[0], xi[1], xi[2]];
    let v = [xi[3], xi[4], xi[5]];
    Pose3 {
        rotation: so3_exp(&omega),
        translation: mat3::mat_vec(&so3_left_jacobian(&omega), &v),
    }
}

/// [x, y, z, qx, qy, qz, qw]
pub fn pose_tuple(pose: &Pose3) -> [f64; 7] {
    let q = quaternion_xyzw(pose);
    let t = pose.translation;
    [t[0], t[1], t[2], q[0], q[1], q[2], q[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_pose_close(a: &Pose3, b: &Pose3) {
        for i in 0..3 {
            assert!((a.translation[i] - b.translation[i]).abs() < 1e-9);
            for j in 0..3 {
                assert!((a.rotation[i][j] - b.rotation[i][j]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn log_exp_roundtrip() {
        let pose = from_xyzquat(&[1.0, -2.0, 0.5, 0.1, 0.2, -0.3, 0.927]);
        let normalized = {
            let q = [0.1, 0.2, -0.3, 0.927];
            let n = (q.iter().map(|x| x * x).sum::<f64>()).sqrt();
            from_xyzquat(&[1.0, -2.0, 0.5, q[0] / n, q[1] / n, q[2] / n, q[3] / n])
        };
        let _ = pose;
        assert_pose_close(&expmap(&logmap(&normalized)), &normalized);
    }

    #[test]
    fn compose_inverse_is_identity() {
        let q = [0.0, 0.0, 0.3826834f64, 0.9238795];
        let n = (q.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let pose = from_xyzquat(&[3.0, 1.0, -0.7, q[0] / n, q[1] / n, q[2] / n, q[3] / n]);
        assert_pose_close(&compose(&pose, &inverse(&pose)), &Pose3::identity());
    }
}
