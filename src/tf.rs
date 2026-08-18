//! Whole-recording tf lookup, mirroring dimos' `RecordingTF`: every edge of the
//! recorded `tf` stream is buffered once, each lookup latches the newest sample
//! at-or-before the query time (never a future one), and chains are resolved by
//! BFS over the undirected frame graph. An optional edge override replaces the
//! recorded localization (all time-varying edges) with a fed trajectory while
//! keeping static edges (sensor mounts).

use crate::mat3;
use crate::memory2::TfSample;
use crate::se3;
use gtsam_shim::Pose3;
use std::collections::{HashMap, HashSet, VecDeque};

const STATIC_POSE_TOLERANCE: f64 = 1e-6;

pub struct RecordingTf {
    edges: HashMap<(String, String), Vec<(f64, Pose3)>>,
}

fn pose_from_sample(translation: &[f64; 3], quaternion_xyzw: &[f64; 4]) -> Pose3 {
    let [qx, qy, qz, qw] = *quaternion_xyzw;
    Pose3 {
        rotation: mat3::mat_from_quat(&[qw, qx, qy, qz]),
        translation: *translation,
    }
}

fn edge_is_static(samples: &[(f64, Pose3)]) -> bool {
    let Some((_, first)) = samples.first() else {
        return true;
    };
    samples.iter().all(|(_, pose)| {
        let dt = mat3::norm(&mat3::sub(&pose.translation, &first.translation));
        let dr: f64 = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| (pose.rotation[i][j] - first.rotation[i][j]).abs())
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        dt <= STATIC_POSE_TOLERANCE && dr <= 10.0 * STATIC_POSE_TOLERANCE
    })
}

impl RecordingTf {
    pub fn from_samples(samples: &[TfSample]) -> RecordingTf {
        let mut edges: HashMap<(String, String), Vec<(f64, Pose3)>> = HashMap::new();
        for sample in samples {
            edges
                .entry((sample.parent.clone(), sample.child.clone()))
                .or_default()
                .push((
                    sample.ts,
                    pose_from_sample(&sample.translation, &sample.quaternion_xyzw),
                ));
        }
        for samples in edges.values_mut() {
            samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        RecordingTf { edges }
    }

    /// Replace `parent -> child` with a sampled trajectory, dropping every
    /// time-varying recorded edge so the fed edge is the only localization.
    pub fn override_edge(&mut self, parent: &str, child: &str, trajectory: Vec<(f64, Pose3)>) {
        self.edges.retain(|_, samples| edge_is_static(samples));
        let mut trajectory = trajectory;
        trajectory.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.edges
            .insert((parent.to_string(), child.to_string()), trajectory);
    }

    pub fn frames(&self) -> HashSet<String> {
        let mut names = HashSet::new();
        for (parent, child) in self.edges.keys() {
            names.insert(parent.clone());
            names.insert(child.clone());
        }
        names
    }

    pub fn has_edge(&self, parent: &str, child: &str) -> bool {
        self.edges
            .contains_key(&(parent.to_string(), child.to_string()))
    }

    /// Newest sample at-or-before `ts` on one recorded edge.
    fn latch(&self, parent: &str, child: &str, ts: f64) -> Option<Pose3> {
        let samples = self.edges.get(&(parent.to_string(), child.to_string()))?;
        let index = samples.partition_point(|(sample_ts, _)| *sample_ts <= ts);
        if index == 0 {
            return None;
        }
        Some(samples[index - 1].1.clone())
    }

    fn step(&self, from: &str, to: &str, ts: f64) -> Option<Pose3> {
        if let Some(pose) = self.latch(from, to, ts) {
            return Some(pose);
        }
        self.latch(to, from, ts).map(|pose| se3::inverse(&pose))
    }

    fn neighbors(&self, frame: &str) -> Vec<String> {
        let mut out = Vec::new();
        for (parent, child) in self.edges.keys() {
            if parent == frame {
                out.push(child.clone());
            }
            if child == frame {
                out.push(parent.clone());
            }
        }
        out
    }

    /// `parent <- child` at `ts`, chaining edges by BFS. Errors (showing the
    /// loaded tree) when the chain cannot be resolved — a broken tf tree is a
    /// hard failure, not a soft miss.
    pub fn get(&self, parent: &str, child: &str, ts: f64) -> Result<Pose3, String> {
        if parent == child {
            return Ok(Pose3::identity());
        }
        if let Some(pose) = self.step(parent, child, ts) {
            return Ok(pose);
        }
        let mut queue: VecDeque<(String, Pose3)> = VecDeque::new();
        let mut visited: HashSet<String> = HashSet::new();
        queue.push_back((parent.to_string(), Pose3::identity()));
        visited.insert(parent.to_string());
        while let Some((frame, accumulated)) = queue.pop_front() {
            if frame == child {
                return Ok(accumulated);
            }
            for neighbor in self.neighbors(&frame) {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor.clone());
                if let Some(step) = self.step(&frame, &neighbor, ts) {
                    queue.push_back((neighbor, se3::compose(&accumulated, &step)));
                }
            }
        }
        let mut tree: Vec<String> = self
            .edges
            .keys()
            .map(|(p, c)| format!("{p}->{c}"))
            .collect();
        tree.sort();
        Err(format!(
            "tf lookup failed:\n  tree: {}\n  timestamp: {ts}\n  failed query: {parent:?} <- {child:?}",
            if tree.is_empty() { "<none>".to_string() } else { tree.join(", ") }
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ts: f64, parent: &str, child: &str, translation: [f64; 3]) -> TfSample {
        TfSample {
            ts,
            parent: parent.to_string(),
            child: child.to_string(),
            translation,
            quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn chains_and_latches_past_only() {
        let tf = RecordingTf::from_samples(&[
            sample(0.0, "world", "odom", [1.0, 0.0, 0.0]),
            sample(0.0, "odom", "base", [0.0, 1.0, 0.0]),
            sample(5.0, "odom", "base", [0.0, 2.0, 0.0]),
        ]);
        let early = tf.get("world", "base", 1.0).unwrap();
        assert_eq!(early.translation, [1.0, 1.0, 0.0]);
        let late = tf.get("world", "base", 10.0).unwrap();
        assert_eq!(late.translation, [1.0, 2.0, 0.0]);
        assert!(tf.get("world", "base", -1.0).is_err());
        let inverse = tf.get("base", "world", 1.0).unwrap();
        assert_eq!(inverse.translation, [-1.0, -1.0, 0.0]);
    }

    #[test]
    fn override_drops_dynamic_edges_only() {
        let mut tf = RecordingTf::from_samples(&[
            sample(0.0, "base", "sensor", [0.5, 0.0, 0.0]),
            sample(0.0, "world", "base", [0.0, 0.0, 0.0]),
            sample(1.0, "world", "base", [9.0, 0.0, 0.0]),
        ]);
        tf.override_edge(
            "world",
            "base",
            vec![(
                0.0,
                Pose3::from_translation([2.0, 0.0, 0.0]),
            )],
        );
        let pose = tf.get("world", "sensor", 3.0).unwrap();
        assert_eq!(pose.translation, [2.5, 0.0, 0.0]);
    }
}
