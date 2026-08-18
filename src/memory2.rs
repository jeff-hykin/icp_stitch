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

//! Minimal reader for a dimos memory2 SQLite recording, decoding the raw
//! LCM blobs the eval binary feeds through GscPgo.
//!
//! Each stream `S` is two tables: `"S"` (metadata: id, ts, ...) and `"S_blob"`
//! (id, data BLOB). The blob is exactly `value.lcm_encode()` — an 8-byte
//! fingerprint followed by the packed message — so a stream is discriminated by
//! trying each candidate decoder and letting the fingerprint check reject
//! mismatches. Pose values in the metadata columns are deprecated; everything
//! here comes from decoding the blob.

use crate::mat3::{self, Mat3, Vec3};
use lcm_msgs::geometry_msgs::PoseStamped;
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::PointCloud2;
use lcm_msgs::tf2_msgs::TFMessage;
use rusqlite::Connection;

/// One odometry sample: world pose at a timestamp. `frame_id` is the pose's
/// reference (parent) frame; `child_frame_id` is the body frame the pose moves
/// — the frame the paired lidar cloud must already be in (no tf is applied).
/// `child_frame_id` is empty for `PoseStamped` payloads, which carry no child.
pub struct OdomRow {
    pub ts: f64,
    pub translation: Vec3,
    pub rotation: Mat3,
    pub quaternion_xyzw: [f64; 4],
    pub frame_id: String,
    pub child_frame_id: String,
}

/// One lidar scan: xyz points at a timestamp. `intensities` is present only
/// when the cloud has a float32 `intensity` field (aligned with `points`).
pub struct ScanRow {
    pub ts: f64,
    pub points: Vec<[f32; 3]>,
    pub intensities: Option<Vec<f32>>,
    pub frame_id: String,
}

fn quote_ident(name: &str) -> Result<String, String> {
    if name.contains('"') {
        return Err(format!("illegal stream name {name:?}"));
    }
    Ok(format!("\"{name}\""))
}

/// Join a stream's metadata + blob tables on id, ordered by timestamp, and hand
/// each `(ts, blob)` to `decode`. Rows the decoder returns `None` for are
/// skipped (e.g. a corrupt frame), so a single bad message never aborts a run.
///
/// `stride` keeps every `stride`-th row (1 = all). The stride is applied here,
/// before decoding, so strided-out blobs are never decoded or retained — a huge
/// recording can be subsampled without materializing every scan in memory.
fn read_stream<T>(
    connection: &Connection,
    stream: &str,
    stride: usize,
    mut decode: impl FnMut(f64, &[u8]) -> Option<T>,
) -> Result<Vec<T>, String> {
    let stride = stride.max(1);
    let table = quote_ident(stream)?;
    let blob_table = quote_ident(&format!("{stream}_blob"))?;
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
    let mut out = Vec::new();
    for (index, row) in rows.enumerate() {
        if index % stride != 0 {
            continue;
        }
        let (ts, data) = row.map_err(|e| e.to_string())?;
        if let Some(value) = decode(ts, &data) {
            out.push(value);
        }
    }
    Ok(out)
}

fn odom_from_odometry(ts: f64, message: &Odometry) -> OdomRow {
    let pose = &message.pose.pose;
    odom_from_pose(
        ts,
        [pose.position.x, pose.position.y, pose.position.z],
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        message.header.frame_id.clone(),
        message.child_frame_id.clone(),
    )
}

fn odom_from_pose_stamped(ts: f64, message: &PoseStamped) -> OdomRow {
    let pose = &message.pose;
    odom_from_pose(
        ts,
        [pose.position.x, pose.position.y, pose.position.z],
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        message.header.frame_id.clone(),
        String::new(),
    )
}

fn odom_from_pose(
    ts: f64,
    translation: Vec3,
    quaternion_xyzw: [f64; 4],
    frame_id: String,
    child_frame_id: String,
) -> OdomRow {
    let [qx, qy, qz, qw] = quaternion_xyzw;
    OdomRow {
        ts,
        translation,
        rotation: mat3::mat_from_quat(&[qw, qx, qy, qz]),
        quaternion_xyzw,
        frame_id,
        child_frame_id,
    }
}

/// Read an odometry stream, accepting either `nav_msgs/Odometry` or
/// `geometry_msgs/PoseStamped` payloads (legacy go2 recordings store the
/// latter). The fingerprint check makes the choice unambiguous per blob.
pub fn read_odometry(
    connection: &Connection,
    stream: &str,
    stride: usize,
) -> Result<Vec<OdomRow>, String> {
    let rows = read_stream(connection, stream, stride, |ts, data| {
        if let Ok(message) = Odometry::decode(data) {
            return Some(odom_from_odometry(ts, &message));
        }
        if let Ok(message) = PoseStamped::decode(data) {
            return Some(odom_from_pose_stamped(ts, &message));
        }
        None
    })?;
    if rows.is_empty() {
        return Err(format!(
            "odom stream {stream:?} decoded no Odometry/PoseStamped rows"
        ));
    }
    Ok(rows)
}

/// FLOAT32 in sensor_msgs/PointField.
const POINT_FIELD_FLOAT32: u8 = 7;

/// Extract xyz points (and, when present as a float32 field, intensities) from
/// a PointCloud2 (mirrors the module's `extract_xyz` / `intensities_f32`).
fn extract_xyz(message: &PointCloud2) -> Option<(Vec<[f32; 3]>, Option<Vec<f32>>)> {
    let mut offset_x = None;
    let mut offset_y = None;
    let mut offset_z = None;
    let mut offset_intensity = None;
    for field in &message.fields {
        match field.name.as_str() {
            "x" => offset_x = Some(field.offset as usize),
            "y" => offset_y = Some(field.offset as usize),
            "z" => offset_z = Some(field.offset as usize),
            "intensity" if field.datatype == POINT_FIELD_FLOAT32 => {
                offset_intensity = Some(field.offset as usize)
            }
            _ => {}
        }
    }
    let (offset_x, offset_y, offset_z) = (offset_x?, offset_y?, offset_z?);
    let step = message.point_step as usize;
    if step == 0 {
        return None;
    }
    let point_count = (message.width as usize) * (message.height as usize);
    let data = &message.data;
    let mut points = Vec::with_capacity(point_count);
    let mut intensities = offset_intensity.map(|_| Vec::with_capacity(point_count));
    let read = |base: usize, offset: usize| -> f32 {
        let bytes = &data[base + offset..base + offset + 4];
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    };
    for index in 0..point_count {
        let base = index * step;
        if base + step > data.len() {
            break;
        }
        points.push([
            read(base, offset_x),
            read(base, offset_y),
            read(base, offset_z),
        ]);
        if let (Some(values), Some(offset)) = (intensities.as_mut(), offset_intensity) {
            values.push(read(base, offset));
        }
    }
    Some((points, intensities))
}

/// One tf edge sample. `ts` comes from the TransformStamped header (falling
/// back to the row ts when the stamp is zero).
pub struct TfSample {
    pub ts: f64,
    pub parent: String,
    pub child: String,
    pub translation: Vec3,
    pub quaternion_xyzw: [f64; 4],
}

/// Read a tf stream of TFMessage rows, flattened to per-edge samples.
pub fn read_tf(connection: &Connection, stream: &str) -> Result<Vec<TfSample>, String> {
    let nested = read_stream(connection, stream, 1, |row_ts, data| {
        let message = TFMessage::decode(data).ok()?;
        let samples: Vec<TfSample> = message
            .transforms
            .iter()
            .map(|entry| {
                let stamp =
                    entry.header.stamp.sec as f64 + entry.header.stamp.nsec as f64 / 1e9;
                TfSample {
                    ts: if stamp > 0.0 { stamp } else { row_ts },
                    parent: entry.header.frame_id.clone(),
                    child: entry.child_frame_id.clone(),
                    translation: [
                        entry.transform.translation.x,
                        entry.transform.translation.y,
                        entry.transform.translation.z,
                    ],
                    quaternion_xyzw: [
                        entry.transform.rotation.x,
                        entry.transform.rotation.y,
                        entry.transform.rotation.z,
                        entry.transform.rotation.w,
                    ],
                }
            })
            .collect();
        Some(samples)
    })?;
    Ok(nested.into_iter().flatten().collect())
}

/// Read a lidar stream of PointCloud2 scans.
pub fn read_scans(
    connection: &Connection,
    stream: &str,
    stride: usize,
) -> Result<Vec<ScanRow>, String> {
    let rows = read_stream(connection, stream, stride, |ts, data| {
        let message = PointCloud2::decode(data).ok()?;
        let (points, intensities) = extract_xyz(&message)?;
        Some(ScanRow {
            ts,
            points,
            intensities,
            frame_id: message.header.frame_id.clone(),
        })
    })?;
    if rows.is_empty() {
        return Err(format!(
            "lidar stream {stream:?} decoded no PointCloud2 rows"
        ));
    }
    Ok(rows)
}

// ---- write side ----------------------------------------------------------------
//
// Mirrors dimos memory2's SqliteBackend exactly (registry row, table DDL, jsonb
// tags, rtree rows) so python `Store(db).stream(name, T)` reads what we write.

/// The `_streams` registry config for a stream created by this tool. Field
/// order and spacing match python `json.dumps` of the dict SqliteBackend
/// serializes, so a byte-compare against python-written registries passes.
fn stream_config_json(payload_module: &str) -> String {
    format!(
        "{{\"payload_module\": \"{payload_module}\", \"codec_id\": \"lcm\", \
         \"eager_blobs\": false, \"page_size\": 256, \
         \"blob_store\": {{\"class\": \"dimos.memory2.blobstore.sqlite.SqliteBlobStore\", \"config\": {{\"path\": null}}}}, \
         \"vector_store\": {{\"class\": \"dimos.memory2.vectorstore.sqlite.SqliteVectorStore\", \"config\": {{\"path\": null}}}}, \
         \"notifier\": {{\"class\": \"dimos.memory2.notifier.subject.SubjectNotifier\", \"config\": {{}}}}}}"
    )
}

/// Stream names present in the `_streams` registry (empty when the registry
/// table itself is absent).
pub fn list_streams(connection: &Connection) -> Result<Vec<String>, String> {
    let mut statement = match connection.prepare("SELECT name FROM _streams ORDER BY name") {
        Ok(statement) => statement,
        Err(_) => return Ok(Vec::new()),
    };
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

/// Drop a stream's tables and registry row (`SqliteBackend.delete_stream`).
pub fn delete_stream(connection: &Connection, name: &str) -> Result<(), String> {
    for suffix in ["", "_blob", "_vec", "_rtree"] {
        let full = quote_ident(&format!("{name}{suffix}"))?;
        connection
            .execute_batch(&format!("DROP TABLE IF EXISTS {full}"))
            .map_err(|e| e.to_string())?;
    }
    connection
        .execute("DELETE FROM _streams WHERE name = ?", [name])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Create a stream: registry row + meta/rtree/blob tables. No `_vec` table —
/// python creates it lazily on first vector insert, which never happens here.
pub fn create_stream(
    connection: &Connection,
    name: &str,
    payload_module: &str,
) -> Result<(), String> {
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS _streams (name TEXT PRIMARY KEY, config TEXT NOT NULL)",
        )
        .map_err(|e| e.to_string())?;
    connection
        .execute(
            "INSERT OR REPLACE INTO _streams (name, config) VALUES (?, ?)",
            [name, &stream_config_json(payload_module)],
        )
        .map_err(|e| e.to_string())?;
    let table = quote_ident(name)?;
    let rtree = quote_ident(&format!("{name}_rtree"))?;
    let blob = quote_ident(&format!("{name}_blob"))?;
    connection
        .execute_batch(&format!(
            "CREATE TABLE IF NOT EXISTS {table} (\
                id INTEGER PRIMARY KEY AUTOINCREMENT, \
                ts REAL NOT NULL, \
                value NUMERIC, \
                pose_x REAL, pose_y REAL, pose_z REAL, \
                pose_qx REAL, pose_qy REAL, pose_qz REAL, pose_qw REAL, \
                tags BLOB DEFAULT (jsonb('{{}}')));\
             CREATE VIRTUAL TABLE IF NOT EXISTS {rtree} USING rtree(\
                id, x_min, x_max, y_min, y_max, z_min, z_max);\
             CREATE TABLE IF NOT EXISTS {blob} (\
                id INTEGER PRIMARY KEY, data BLOB NOT NULL);"
        ))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Append one observation: meta row (+ tag indexes), blob at the same id, and
/// an rtree row when a pose is present. `tags` order is preserved into the
/// stored JSON (python dict insertion order).
pub fn append(
    connection: &Connection,
    name: &str,
    ts: f64,
    pose: Option<[f64; 7]>,
    tags: &[(&str, String)],
    blob: &[u8],
) -> Result<(), String> {
    let json_tags: Vec<(&str, String)> = tags
        .iter()
        .map(|(key, value)| (*key, format!("\"{value}\"")))
        .collect();
    append_json_tags(connection, name, ts, pose, &json_tags, blob)
}

/// Like `append`, but tag values are raw JSON (numbers keep their type, as
/// python json.dumps writes them for e.g. `marker_id` / diagnostics tags).
pub fn append_json_tags(
    connection: &Connection,
    name: &str,
    ts: f64,
    pose: Option<[f64; 7]>,
    tags: &[(&str, String)],
    blob: &[u8],
) -> Result<(), String> {
    let table = quote_ident(name)?;
    for (key, _) in tags {
        if key.contains('"') || key.contains('\'') {
            return Err(format!("illegal tag key {key:?}"));
        }
        let index_name = quote_ident(&format!("{name}_tag_{key}"))?;
        connection
            .execute_batch(&format!(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table}(json_extract(tags, '$.{key}'))"
            ))
            .map_err(|e| e.to_string())?;
    }
    let tags_json = if tags.is_empty() {
        "{}".to_string()
    } else {
        // python json.dumps spacing: {"key": value, ...}
        let body: Vec<String> = tags
            .iter()
            .map(|(key, value)| format!("\"{key}\": {value}"))
            .collect();
        format!("{{{}}}", body.join(", "))
    };
    append_raw(connection, name, ts, pose, &tags_json, blob)
}

/// One stream row with everything the meta/blob tables carry, for lossless
/// stream rewrites (e.g. stripping stale tf edges).
pub struct RawRow {
    pub ts: f64,
    pub pose: Option<[f64; 7]>,
    pub tags_json: String,
    pub blob: Vec<u8>,
}

/// All rows of a stream in ts order, with pose columns, tags (as JSON text),
/// and the raw blob.
pub fn read_raw_rows(connection: &Connection, stream: &str) -> Result<Vec<RawRow>, String> {
    let table = quote_ident(stream)?;
    let blob_table = quote_ident(&format!("{stream}_blob"))?;
    let sql = format!(
        "SELECT meta.ts, meta.pose_x, meta.pose_y, meta.pose_z, \
                meta.pose_qx, meta.pose_qy, meta.pose_qz, meta.pose_qw, \
                json(meta.tags), blob.data \
         FROM {table} AS meta JOIN {blob_table} AS blob ON meta.id = blob.id \
         ORDER BY meta.ts"
    );
    let mut statement = connection.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = statement
        .query_map([], |row| {
            let ts: f64 = row.get(0)?;
            let pose_x: Option<f64> = row.get(1)?;
            let pose = match pose_x {
                Some(px) => Some([
                    px,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ]),
                None => None,
            };
            let tags_json: String = row.get(8)?;
            let blob: Vec<u8> = row.get(9)?;
            Ok(RawRow {
                ts,
                pose,
                tags_json,
                blob,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

/// `append` with the tags already serialized (used for lossless row rewrites;
/// jsonb normalizes the text, so spacing differences are immaterial).
pub fn append_raw(
    connection: &Connection,
    name: &str,
    ts: f64,
    pose: Option<[f64; 7]>,
    tags_json: &str,
    blob: &[u8],
) -> Result<(), String> {
    let table = quote_ident(name)?;
    let pose_values: [Option<f64>; 7] = match pose {
        Some(p) => p.map(Some),
        None => [None; 7],
    };
    connection
        .execute(
            &format!(
                "INSERT INTO {table} (ts, value, pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, tags) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, jsonb(?))"
            ),
            rusqlite::params![
                ts,
                Option::<f64>::None,
                pose_values[0],
                pose_values[1],
                pose_values[2],
                pose_values[3],
                pose_values[4],
                pose_values[5],
                pose_values[6],
                tags_json,
            ],
        )
        .map_err(|e| e.to_string())?;
    let row_id = connection.last_insert_rowid();
    let blob_table = quote_ident(&format!("{name}_blob"))?;
    connection
        .execute(
            &format!("INSERT INTO {blob_table} (id, data) VALUES (?, ?)"),
            rusqlite::params![row_id, blob],
        )
        .map_err(|e| e.to_string())?;
    if let Some([px, py, pz, ..]) = pose {
        let rtree = quote_ident(&format!("{name}_rtree"))?;
        connection
            .execute(
                &format!(
                    "INSERT INTO {rtree} (id, x_min, x_max, y_min, y_max, z_min, z_max) \
                     VALUES (?, ?, ?, ?, ?, ?, ?)"
                ),
                rusqlite::params![row_id, px, px, py, py, pz, pz],
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_config_matches_python_registry_bytes() {
        // Byte-for-byte the config SqliteBackend json.dumps'd into a real
        // recording db (go2_china_office.db, stream go2_odometry_corrected).
        let expected = "{\"payload_module\": \"dimos.msgs.nav_msgs.Odometry.Odometry\", \
                        \"codec_id\": \"lcm\", \"eager_blobs\": false, \"page_size\": 256, \
                        \"blob_store\": {\"class\": \"dimos.memory2.blobstore.sqlite.SqliteBlobStore\", \"config\": {\"path\": null}}, \
                        \"vector_store\": {\"class\": \"dimos.memory2.vectorstore.sqlite.SqliteVectorStore\", \"config\": {\"path\": null}}, \
                        \"notifier\": {\"class\": \"dimos.memory2.notifier.subject.SubjectNotifier\", \"config\": {}}}";
        assert_eq!(
            stream_config_json("dimos.msgs.nav_msgs.Odometry.Odometry"),
            expected
        );
    }
}
