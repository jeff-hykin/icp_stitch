//! The recording the pipeline reads from and writes into: a memory2 `.db` or an `.mcap`.
//!
//! Every stream is handed to the decoders as the LCM blob a `.db` row holds, so the two
//! formats share one decode path: an `.mcap`'s ROS 2 CDR messages are re-encoded to that
//! blob (`cdr::to_lcm`) and its dimos LCM messages already are one.
//!
//! In an `.mcap` a stream is a topic, named with or without its leading slash, or the
//! way `dtk mcap_to_db` names it (leading slash dropped, the rest joined with `_`), so
//! the same command line works on a recording and on its `.db` conversion. A topic may
//! sit on several channel ids (files edited in place repeat them); all are read. `tf`
//! also takes in `/tf_static`, which is where a ROS 2 rig keeps its sensor mounts.
//!
//! Outputs are appended to the `.mcap` as new CDR topics with ros2msg schemas (`/<name>`),
//! each in chunks of its own, committed with a fresh summary whenever a stream is done
//! (when the next one starts, something is read back, or the run finishes). A re-run
//! removes the previous run's outputs first, giving their space back.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use lcm_msgs::geometry_msgs::PoseStamped;
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::{CameraInfo, CompressedImage, Image, PointCloud2};
use lcm_msgs::tf2_msgs::TFMessage;
use rusqlite::types::Value;
use rusqlite::{Connection, OpenFlags};

use crate::cdr;
use crate::mcap::{record, Channel, McapFile};
use crate::memory2::{self, decompress_if_lz4, RawRow, Rows};
use crate::msgs;

const ODOMETRY_MODULE: &str = "dimos.msgs.nav_msgs.Odometry.Odometry";
const POINTCLOUD2_MODULE: &str = "dimos.msgs.sensor_msgs.PointCloud2.PointCloud2";
const TFMESSAGE_MODULE: &str = "dimos.msgs.tf2_msgs.TFMessage.TFMessage";
const CAMERA_INFO_MODULE: &str = "dimos.msgs.sensor_msgs.CameraInfo.CameraInfo";
const POSE_STAMPED_MODULE: &str = "dimos.msgs.geometry_msgs.PoseStamped.PoseStamped";
const GRAPH3D_MODULE: &str = "dimos.navigation.jnav.msgs.Graph3D.Graph3D";
const DEFORMATION_NODE_MODULE: &str = "dimos.navigation.jnav.msgs.DeformationNode.DeformationNode";

/// Flush an output's buffered messages into a chunk at about this size.
const CHUNK_TARGET: usize = 4 << 20;
/// How far a header stamp may sit from its log time when windowing by log time.
const STAMP_SLACK_NS: u64 = 5_000_000_000;

/// The tag keys a raw AprilTag stream carries, in `detect::read_raw_tag_stream`'s order.
pub const TAG_KEYS: [&str; 6] = ["marker_id", "sharpness", "reproj_px", "tag_px", "lin_speed", "ang_speed"];

pub enum Store {
    Db(Connection),
    Mcap(McapStore),
}

/// `.mcap` by its magic, so a renamed file still opens as what it is.
fn is_mcap_file(path: &Path) -> bool {
    let mut magic = [0u8; 8];
    std::fs::File::open(path)
        .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut magic))
        .is_ok_and(|_| &magic == crate::mcap::MAGIC)
}

impl Store {
    pub fn open(path: &Path) -> Result<Store, String> {
        if path.is_dir() {
            return Err(format!("the recording must be a .db or .mcap file, not a directory: {}", path.display()));
        }
        if is_mcap_file(path) {
            return Ok(Store::Mcap(McapStore::open(path)?));
        }
        let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)
            .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        Ok(Store::Db(connection))
    }

    pub fn is_mcap(&self) -> bool {
        matches!(self, Store::Mcap(_))
    }

    /// The sqlite connection, for the few things only a `.db` has.
    pub fn connection(&self) -> Option<&Connection> {
        match self {
            Store::Db(connection) => Some(connection),
            Store::Mcap(_) => None,
        }
    }

    pub fn list_streams(&self) -> Result<Vec<String>, String> {
        match self {
            Store::Db(connection) => memory2::list_streams(connection),
            Store::Mcap(mcap) => mcap.list_streams(),
        }
    }

    pub fn has_stream(&self, name: &str) -> Result<bool, String> {
        Ok(self.list_streams()?.iter().any(|s| s == name))
    }

    /// The memory2 payload module a stream holds, when it says.
    pub fn payload_module(&self, stream: &str) -> Option<String> {
        match self {
            Store::Db(connection) => connection
                .query_row(
                    "SELECT json_extract(config, '$.payload_module') FROM _streams WHERE name = ?",
                    [stream],
                    |row| row.get(0),
                )
                .ok(),
            Store::Mcap(mcap) => mcap.payload_module(stream),
        }
    }

    /// The earliest row of a stream: `(ts, LCM blob)`.
    pub fn first_blob(&self, stream: &str) -> Option<(f64, Vec<u8>)> {
        match self {
            Store::Db(connection) => {
                if stream.contains('"') {
                    return None;
                }
                connection
                    .query_row(
                        &format!(
                            "SELECT meta.ts, blob.data FROM \"{stream}\" AS meta \
                             JOIN \"{stream}_blob\" AS blob ON meta.id = blob.id \
                             ORDER BY meta.ts LIMIT 1"
                        ),
                        [],
                        |row| Ok((row.get::<_, f64>(0)?, row.get::<_, Vec<u8>>(1)?)),
                    )
                    .ok()
                    .map(|(ts, data)| (ts, decompress_if_lz4(data)))
            }
            Store::Mcap(mcap) => mcap.first_blob(stream).ok().flatten(),
        }
    }

    /// The row nearest `ts`, within `max_dt` seconds.
    pub fn nearest_blob(&self, stream: &str, ts: f64, max_dt: f64) -> Option<Vec<u8>> {
        match self {
            Store::Db(connection) => {
                if stream.contains('"') {
                    return None;
                }
                connection
                    .query_row(
                        &format!(
                            "SELECT blob.data FROM \"{stream}\" AS meta \
                             JOIN \"{stream}_blob\" AS blob ON meta.id = blob.id \
                             WHERE ABS(meta.ts - ?1) <= ?2 ORDER BY ABS(meta.ts - ?1) LIMIT 1"
                        ),
                        rusqlite::params![ts, max_dt],
                        |row| row.get::<_, Vec<u8>>(0),
                    )
                    .ok()
                    .map(decompress_if_lz4)
            }
            Store::Mcap(mcap) => mcap.nearest_blob(stream, ts, max_dt).ok().flatten(),
        }
    }

    /// `(ts, x y z qx qy qz qw)` of every row that carries a pose in its metadata. An
    /// `.mcap` has no per-message pose, so it has none.
    pub fn row_poses(&self, stream: &str) -> Result<Vec<(f64, [f64; 7])>, String> {
        let Store::Db(connection) = self else {
            return Ok(Vec::new());
        };
        if stream.contains('"') {
            return Err(format!("illegal stream name {stream:?}"));
        }
        let sql = format!(
            "SELECT ts, pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw FROM \"{stream}\" ORDER BY ts"
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
        let mut posed = Vec::new();
        for row in rows {
            if let (ts, Some(pose)) = row.map_err(|e| e.to_string())? {
                posed.push((ts, pose));
            }
        }
        Ok(posed)
    }

    /// A raw AprilTag stream's rows: `(ts, PoseStamped LCM blob, the TAG_KEYS values)`.
    pub fn read_tag_rows(&self, stream: &str) -> Result<Vec<(f64, Vec<u8>, Vec<Value>)>, String> {
        match self {
            Store::Db(connection) => {
                if stream.contains('"') {
                    return Err(format!("illegal stream name {stream:?}"));
                }
                let tag_columns: Vec<String> =
                    TAG_KEYS.iter().map(|key| format!("json_extract(meta.tags, '$.{key}')")).collect();
                let sql = format!(
                    "SELECT meta.ts, blob.data, {} FROM \"{stream}\" AS meta \
                     JOIN \"{stream}_blob\" AS blob ON meta.id = blob.id ORDER BY meta.ts",
                    tag_columns.join(", ")
                );
                let mut statement = connection.prepare(&sql).map_err(|e| e.to_string())?;
                let rows = statement
                    .query_map([], |row| {
                        let mut values = Vec::with_capacity(TAG_KEYS.len());
                        for index in 0..TAG_KEYS.len() {
                            values.push(row.get::<_, Value>(index + 2)?);
                        }
                        Ok((row.get::<_, f64>(0)?, decompress_if_lz4(row.get::<_, Vec<u8>>(1)?), values))
                    })
                    .map_err(|e| e.to_string())?;
                rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
            }
            Store::Mcap(mcap) => mcap.read_tag_rows(stream),
        }
    }

    // ---- writing ----

    /// Clear a previous run's outputs before writing new ones. On an `.mcap` this removes
    /// them all at once, so the space they held at the end of the file is given back; a
    /// `.db` replaces each stream as it is written, so this does nothing there.
    pub fn prepare_outputs(&self, names: &[String]) -> Result<(), String> {
        match self {
            Store::Db(_) => Ok(()),
            Store::Mcap(mcap) => mcap.remove(names),
        }
    }

    pub fn delete_stream(&self, name: &str) -> Result<(), String> {
        match self {
            Store::Db(connection) => memory2::delete_stream(connection, name),
            Store::Mcap(mcap) => mcap.remove(&[name.to_string()]),
        }
    }

    pub fn create_stream(&self, name: &str, payload_module: &str) -> Result<(), String> {
        match self {
            Store::Db(connection) => memory2::create_stream(connection, name, payload_module),
            Store::Mcap(mcap) => mcap.create(name, payload_module),
        }
    }

    pub fn append(&self, name: &str, ts: f64, pose: Option<[f64; 7]>, tags: &[(&str, String)], blob: &[u8]) -> Result<(), String> {
        match self {
            Store::Db(connection) => memory2::append(connection, name, ts, pose, tags, blob),
            Store::Mcap(mcap) => mcap.append(name, ts, &[], blob),
        }
    }

    pub fn append_json_tags(
        &self,
        name: &str,
        ts: f64,
        pose: Option<[f64; 7]>,
        tags: &[(&str, String)],
        blob: &[u8],
    ) -> Result<(), String> {
        match self {
            Store::Db(connection) => memory2::append_json_tags(connection, name, ts, pose, tags, blob),
            Store::Mcap(mcap) => mcap.append(name, ts, tags, blob),
        }
    }

    pub fn read_raw_rows(&self, stream: &str) -> Result<Vec<RawRow>, String> {
        match self {
            Store::Db(connection) => memory2::read_raw_rows(connection, stream),
            Store::Mcap(_) => Err("rewriting a recorded stream row by row needs a .db".into()),
        }
    }

    pub fn append_raw(&self, name: &str, ts: f64, pose: Option<[f64; 7]>, tags_json: &str, blob: &[u8]) -> Result<(), String> {
        match self {
            Store::Db(connection) => memory2::append_raw(connection, name, ts, pose, tags_json, blob),
            Store::Mcap(_) => Err("rewriting a recorded stream row by row needs a .db".into()),
        }
    }

    /// Write out everything still buffered and leave the file complete.
    pub fn finish(&self) -> Result<(), String> {
        match self {
            Store::Db(_) => Ok(()),
            Store::Mcap(mcap) => mcap.flush(),
        }
    }
}

impl Rows for Store {
    fn read_stream<T>(&self, stream: &str, stride: usize, decode: impl FnMut(f64, &[u8]) -> Option<T>) -> Result<Vec<T>, String> {
        match self {
            Store::Db(connection) => connection.read_stream(stream, stride, decode),
            Store::Mcap(mcap) => mcap.read_stream(stream, stride, decode),
        }
    }
}

// ---- the .mcap side -------------------------------------------------------------------------

/// How a channel's payloads become LCM blobs.
#[derive(Clone)]
enum Encoding {
    Cdr(String),
    Lcm,
    Unknown,
}

struct Output {
    module: String,
    channel: Option<u16>,
    records: Vec<u8>,
    entries: Vec<(u64, u64)>,
    sequence: u32,
}

pub struct McapStore {
    file: RefCell<McapFile>,
    outputs: RefCell<BTreeMap<String, Output>>,
}

/// `dtk mcap_to_db`'s stream name for a topic.
pub fn stream_name_for_topic(topic: &str) -> String {
    topic.trim_start_matches('/').replace('/', "_")
}

/// `(fingerprint, offset of the header stamp)` per stamped dimos LCM type. LCM hoists
/// array lengths ahead of the struct (a PointCloud2's stamp sits at 20, not 12), so each
/// offset is found by encoding a message with a marker stamp rather than assumed.
fn lcm_stamp_offsets() -> &'static Vec<([u8; 8], usize)> {
    static OFFSETS: OnceLock<Vec<([u8; 8], usize)>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        let marker = lcm_msgs::std_msgs::Time { sec: 0x1234_5678, nsec: 0x0abc_def0 };
        let mut needle = 0x1234_5678i32.to_be_bytes().to_vec();
        needle.extend_from_slice(&0x0abc_def0i32.to_be_bytes());
        let header = lcm_msgs::std_msgs::Header { stamp: marker, ..Default::default() };
        let mut tf = TFMessage::default();
        tf.transforms.push(lcm_msgs::geometry_msgs::TransformStamped { header: header.clone(), ..Default::default() });
        let encoded = [
            Odometry { header: header.clone(), ..Default::default() }.encode(),
            PointCloud2 { header: header.clone(), ..Default::default() }.encode(),
            CameraInfo { header: header.clone(), ..Default::default() }.encode(),
            Image { header: header.clone(), ..Default::default() }.encode(),
            PoseStamped { header: header.clone(), ..Default::default() }.encode(),
            CompressedImage { header, ..Default::default() }.encode(),
            tf.encode(),
        ];
        encoded
            .iter()
            .map(|bytes| {
                let at = bytes.windows(8).position(|window| window == needle.as_slice()).unwrap();
                (bytes[..8].try_into().unwrap(), at)
            })
            .collect()
    })
}

/// The header stamp of a dimos LCM message (big-endian, no alignment), or the first
/// transform's for a TFMessage; None for a type without one, or a zero stamp.
pub fn lcm_header_stamp(data: &[u8]) -> Option<f64> {
    let int = |at: usize| data.get(at..at + 4).map(|b| i32::from_be_bytes(b.try_into().unwrap()));
    let fingerprint: [u8; 8] = data.get(..8)?.try_into().ok()?;
    let &(_, at) = lcm_stamp_offsets().iter().find(|(print, _)| *print == fingerprint)?;
    if fingerprint == TFMessage::default().encode()[..8] && int(8)? <= 0 {
        return None;
    }
    let stamp = int(at)? as f64 + int(at + 4)? as f64 / 1e9;
    (stamp != 0.0).then_some(stamp)
}

fn seconds_to_ns(ts: f64) -> u64 {
    (ts * 1e9).round().max(0.0) as u64
}

fn tag_number(tags: &[(&str, String)], key: &str) -> f64 {
    tags.iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, value)| value.trim_matches('"').parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// An LCM blob of a stream's payload module as `(CDR schema name, CDR bytes)`.
fn to_cdr(module: &str, tags: &[(&str, String)], blob: &[u8]) -> Result<(&'static str, Vec<u8>), String> {
    let bad = |what: &str, error: std::io::Error| format!("not a {what} blob: {error}");
    Ok(match module {
        ODOMETRY_MODULE => (cdr::ODOMETRY, cdr::encode_odometry(&Odometry::decode(blob).map_err(|e| bad("nav_msgs.Odometry", e))?)),
        POINTCLOUD2_MODULE => (
            cdr::POINT_CLOUD2,
            cdr::encode_point_cloud2(&PointCloud2::decode(blob).map_err(|e| bad("sensor_msgs.PointCloud2", e))?),
        ),
        TFMESSAGE_MODULE => (cdr::TF_MESSAGE, cdr::encode_tf_message(&TFMessage::decode(blob).map_err(|e| bad("tf2_msgs.TFMessage", e))?)),
        CAMERA_INFO_MODULE => (
            cdr::CAMERA_INFO,
            cdr::encode_camera_info(&CameraInfo::decode(blob).map_err(|e| bad("sensor_msgs.CameraInfo", e))?),
        ),
        POSE_STAMPED_MODULE => {
            let message = PoseStamped::decode(blob).map_err(|e| bad("geometry_msgs.PoseStamped", e))?;
            if !tags.iter().any(|(key, _)| *key == "marker_id") {
                return Ok((cdr::POSE_STAMPED, cdr::encode_pose_stamped(&message)));
            }
            let detection = cdr::AprilTagDetection {
                header: message.header,
                pose: message.pose,
                marker_id: tag_number(tags, "marker_id") as i64,
                sharpness: tag_number(tags, "sharpness"),
                reproj_px: tag_number(tags, "reproj_px"),
                tag_px: tag_number(tags, "tag_px"),
                distance_m: tag_number(tags, "distance_m"),
                view_angle_deg: tag_number(tags, "view_angle_deg"),
                lin_speed: tag_number(tags, "lin_speed"),
                ang_speed: tag_number(tags, "ang_speed"),
            };
            (cdr::APRIL_TAG_DETECTION, cdr::encode_april_tag_detection(&detection))
        }
        GRAPH3D_MODULE => (cdr::GRAPH3D, cdr::encode_graph3d(&msgs::Graph3D::decode(blob).map_err(|e| bad("Graph3D", e))?)),
        DEFORMATION_NODE_MODULE => (
            cdr::DEFORMATION_NODE,
            cdr::encode_deformation_node(&msgs::DeformationNode::decode(blob).map_err(|e| bad("DeformationNode", e))?),
        ),
        other => return Err(format!("no CDR encoding for {other} outputs")),
    })
}

/// The memory2 payload module a ROS 2 schema corresponds to.
fn module_for_schema(schema: &str) -> Option<String> {
    let (package, rest) = schema.split_once("/msg/")?;
    Some(format!("dimos.msgs.{package}.{rest}.{rest}"))
}

impl McapStore {
    pub fn open(path: &Path) -> Result<McapStore, String> {
        Ok(McapStore { file: RefCell::new(McapFile::open(path)?), outputs: RefCell::new(BTreeMap::new()) })
    }

    /// The channel ids a stream name stands for, and each one's encoding.
    fn channels(&self, stream: &str) -> Vec<(u16, Encoding)> {
        let file = self.file.borrow();
        let summary = &file.summary;
        let topics: Vec<&str> = summary.channels.values().map(|c| c.topic.as_str()).collect();
        let slashed = format!("/{stream}");
        let wanted: Vec<String> = if topics.contains(&stream) {
            vec![stream.to_string()]
        } else if topics.contains(&slashed.as_str()) {
            vec![slashed]
        } else {
            let mut matches: Vec<String> =
                topics.iter().filter(|t| stream_name_for_topic(t) == stream).map(|t| t.to_string()).collect();
            matches.dedup();
            matches
        };
        let mut wanted: HashSet<String> = wanted.into_iter().collect();
        if stream == "tf" && !wanted.is_empty() {
            for topic in ["/tf_static", "tf_static"] {
                if topics.contains(&topic) {
                    wanted.insert(topic.to_string());
                }
            }
        }
        summary
            .channels
            .values()
            .filter(|channel| wanted.contains(&channel.topic))
            .map(|channel| {
                let schema = summary.schemas.get(&channel.schema_id);
                let encoding = match channel.message_encoding.as_str() {
                    "cdr" => schema.map_or(Encoding::Unknown, |s| Encoding::Cdr(s.name.clone())),
                    "lcm" => Encoding::Lcm,
                    _ => Encoding::Unknown,
                };
                (channel.id, encoding)
            })
            .collect()
    }

    fn list_streams(&self) -> Result<Vec<String>, String> {
        self.flush()?;
        let file = self.file.borrow();
        let mut names: Vec<String> = file.summary.channels.values().map(|c| stream_name_for_topic(&c.topic)).collect();
        names.sort();
        names.dedup();
        Ok(names)
    }

    fn payload_module(&self, stream: &str) -> Option<String> {
        let channels = self.channels(stream);
        let (id, encoding) = channels.first()?;
        match encoding {
            Encoding::Cdr(schema) => module_for_schema(schema),
            Encoding::Lcm => {
                let file = self.file.borrow();
                let lcm_type = file.summary.channels.get(id)?.meta("lcm_type")?.to_string();
                let (package, name) = lcm_type.split_once('.')?;
                Some(format!("dimos.msgs.{package}.{name}.{name}"))
            }
            Encoding::Unknown => None,
        }
    }

    /// Visit a stream's messages (log-time order) within a log-time window as
    /// `(ts, log time ns, payload)`, ts being the header stamp, or the log time when the
    /// stamp is zero or the type has none. The payload is still in its channel's encoding.
    fn visit(
        &self,
        stream: &str,
        window: (u64, u64),
        mut each: impl FnMut(f64, u64, &Encoding, &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        self.flush()?;
        let channels = self.channels(stream);
        let ids: HashSet<u16> = channels.iter().map(|(id, _)| *id).collect();
        let encodings: BTreeMap<u16, Encoding> = channels.into_iter().collect();
        self.file.borrow_mut().for_each_message(&ids, window.0, window.1, |message| {
            let encoding = &encodings[&message.channel_id];
            let stamp = match encoding {
                Encoding::Cdr(schema) => cdr::header_stamp(schema, message.data),
                Encoding::Lcm => lcm_header_stamp(message.data),
                Encoding::Unknown => None,
            };
            let ts = stamp.unwrap_or(message.log_time as f64 / 1e9);
            each(ts, message.log_time, encoding, message.data)
        })
    }

    fn to_lcm(encoding: &Encoding, data: &[u8]) -> Option<Vec<u8>> {
        match encoding {
            Encoding::Cdr(schema) => cdr::to_lcm(schema, data).ok(),
            Encoding::Lcm => Some(decompress_if_lz4(data.to_vec())),
            Encoding::Unknown => None,
        }
    }

    fn read_stream<T>(&self, stream: &str, stride: usize, mut decode: impl FnMut(f64, &[u8]) -> Option<T>) -> Result<Vec<T>, String> {
        let stride = stride.max(1);
        let everything = (0, u64::MAX);
        if stride == 1 {
            let mut out: Vec<(f64, T)> = Vec::new();
            self.visit(stream, everything, |ts, _, encoding, data| {
                if let Some(value) = Self::to_lcm(encoding, data).and_then(|blob| decode(ts, &blob)) {
                    out.push((ts, value));
                }
                Ok(())
            })?;
            // a .db reads in ts order; chunks come in log-time order, which a stamp can disagree with
            out.sort_by(|a, b| a.0.total_cmp(&b.0));
            return Ok(out.into_iter().map(|(_, value)| value).collect());
        }
        // strided: the stride counts rows in ts order, as the .db query does, so find the
        // kept rows from their stamps first and decode only those
        let mut stamps: Vec<(f64, usize)> = Vec::new();
        self.visit(stream, everything, |ts, _, _, _| {
            stamps.push((ts, stamps.len()));
            Ok(())
        })?;
        stamps.sort_by(|a, b| a.0.total_cmp(&b.0));
        let kept: HashSet<usize> = stamps.iter().step_by(stride).map(|(_, arrival)| *arrival).collect();
        let mut out: Vec<(f64, T)> = Vec::new();
        let mut arrival = 0usize;
        self.visit(stream, everything, |ts, _, encoding, data| {
            if kept.contains(&arrival) {
                if let Some(value) = Self::to_lcm(encoding, data).and_then(|blob| decode(ts, &blob)) {
                    out.push((ts, value));
                }
            }
            arrival += 1;
            Ok(())
        })?;
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(out.into_iter().map(|(_, value)| value).collect())
    }

    fn first_blob(&self, stream: &str) -> Result<Option<(f64, Vec<u8>)>, String> {
        const STOP: &str = "\u{0}stop";
        let mut best: Option<(f64, u64, Vec<u8>)> = None;
        let result = self.visit(stream, (0, u64::MAX), |ts, log_time, encoding, data| {
            if let Some((_, best_log, _)) = &best {
                if log_time > best_log.saturating_add(STAMP_SLACK_NS) {
                    return Err(STOP.into());
                }
            }
            if best.as_ref().is_none_or(|(best_ts, _, _)| ts < *best_ts) {
                if let Some(blob) = Self::to_lcm(encoding, data) {
                    best = Some((ts, log_time, blob));
                }
            }
            Ok(())
        });
        match result {
            Err(error) if error != STOP => Err(error),
            _ => Ok(best.map(|(ts, _, blob)| (ts, blob))),
        }
    }

    fn nearest_blob(&self, stream: &str, ts: f64, max_dt: f64) -> Result<Option<Vec<u8>>, String> {
        let slack = seconds_to_ns(max_dt) + STAMP_SLACK_NS;
        let center = seconds_to_ns(ts);
        let mut best: Option<(f64, Vec<u8>)> = None;
        self.visit(stream, (center.saturating_sub(slack), center.saturating_add(slack)), |stamp, _, encoding, data| {
            let distance = (stamp - ts).abs();
            if distance <= max_dt && best.as_ref().is_none_or(|(d, _)| distance < *d) {
                if let Some(blob) = Self::to_lcm(encoding, data) {
                    best = Some((distance, blob));
                }
            }
            Ok(())
        })?;
        Ok(best.map(|(_, blob)| blob))
    }

    fn read_tag_rows(&self, stream: &str) -> Result<Vec<(f64, Vec<u8>, Vec<Value>)>, String> {
        let mut rows = Vec::new();
        self.visit(stream, (0, u64::MAX), |ts, _, encoding, data| {
            let Encoding::Cdr(schema) = encoding else {
                return Err(format!("'{stream}' is not a {} topic", cdr::APRIL_TAG_DETECTION));
            };
            if schema != cdr::APRIL_TAG_DETECTION {
                return Err(format!("'{stream}' holds {schema}, not {}", cdr::APRIL_TAG_DETECTION));
            }
            let detection = cdr::decode_april_tag_detection(data)?;
            let real = |value: f64| if value.is_nan() { Value::Null } else { Value::Real(value) };
            let values = vec![
                Value::Integer(detection.marker_id),
                real(detection.sharpness),
                real(detection.reproj_px),
                real(detection.tag_px),
                real(detection.lin_speed),
                real(detection.ang_speed),
            ];
            let blob = PoseStamped { header: detection.header, pose: detection.pose }.encode();
            rows.push((ts, blob, values));
            Ok(())
        })?;
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(rows)
    }

    // ---- writing ----

    fn remove(&self, names: &[String]) -> Result<(), String> {
        self.flush()?;
        let mut ids = HashSet::new();
        for name in names {
            self.outputs.borrow_mut().remove(name);
            ids.extend(self.channels(name).into_iter().map(|(id, _)| id));
        }
        self.file.borrow_mut().remove_channels(&ids)
    }

    fn create(&self, name: &str, module: &str) -> Result<(), String> {
        self.flush()?;
        if !self.channels(name).is_empty() {
            return Err(format!("'{name}' is already a topic in this recording; remove it before writing it"));
        }
        let output = Output { module: module.to_string(), channel: None, records: Vec::new(), entries: Vec::new(), sequence: 0 };
        self.outputs.borrow_mut().insert(name.to_string(), output);
        Ok(())
    }

    /// Declare an output's channel (with the schema its first message needs).
    fn open_channel(&self, name: &str, schema: &str) -> Result<(u16, Vec<u8>), String> {
        let text = cdr::schema_text(schema).ok_or_else(|| format!("no schema text for {schema}"))?;
        let mut file = self.file.borrow_mut();
        let schema_id = file.add_schema(schema, "ros2msg", text.as_bytes());
        file.add_channel(Channel {
            id: 0,
            schema_id,
            topic: format!("/{name}"),
            message_encoding: "cdr".into(),
            metadata: vec![("written_by".into(), "icp_stitch".into())],
        })
    }

    fn append(&self, name: &str, ts: f64, tags: &[(&str, String)], blob: &[u8]) -> Result<(), String> {
        let mut outputs = self.outputs.borrow_mut();
        let output = outputs.get_mut(name).ok_or_else(|| format!("'{name}' was never created"))?;
        let (schema, payload) = to_cdr(&output.module, tags, blob)?;
        if output.channel.is_none() {
            let (id, prelude) = self.open_channel(name, schema)?;
            output.channel = Some(id);
            output.records = prelude;
        }
        let channel = output.channel.unwrap();
        let log_time = seconds_to_ns(ts);
        let mut body = Vec::with_capacity(22 + payload.len());
        body.extend_from_slice(&channel.to_le_bytes());
        body.extend_from_slice(&output.sequence.to_le_bytes());
        body.extend_from_slice(&log_time.to_le_bytes());
        body.extend_from_slice(&log_time.to_le_bytes());
        body.extend_from_slice(&payload);
        output.entries.push((log_time, output.records.len() as u64));
        output.records.extend_from_slice(&record(0x05, &body));
        output.sequence = output.sequence.wrapping_add(1);
        if output.records.len() >= CHUNK_TARGET {
            let (records, entries) = (std::mem::take(&mut output.records), std::mem::take(&mut output.entries));
            self.file.borrow_mut().append_chunk(channel, &records, &entries)?;
        }
        Ok(())
    }

    /// Chunk whatever outputs have buffered, declare the ones still empty, and commit.
    /// An output stays open after this: a later append continues it in a new chunk.
    fn flush(&self) -> Result<(), String> {
        let names: Vec<String> = self.outputs.borrow().keys().cloned().collect();
        for name in names {
            let (module, channel) = {
                let outputs = self.outputs.borrow();
                (outputs[&name].module.clone(), outputs[&name].channel)
            };
            let Some(channel) = channel else {
                // created and never written: still a (zero-message) topic, as in a .db
                let (id, _) = self.open_channel(&name, schema_for_module(&module))?;
                if let Some(stats) = self.file.borrow_mut().summary.statistics.as_mut() {
                    stats.channel_message_counts.insert(id, 0);
                }
                self.outputs.borrow_mut().get_mut(&name).unwrap().channel = Some(id);
                continue;
            };
            let (records, entries) = {
                let mut outputs = self.outputs.borrow_mut();
                let output = outputs.get_mut(&name).unwrap();
                (std::mem::take(&mut output.records), std::mem::take(&mut output.entries))
            };
            if !entries.is_empty() {
                self.file.borrow_mut().append_chunk(channel, &records, &entries)?;
            }
        }
        self.file.borrow_mut().commit()
    }
}

/// The schema an output of `module` is written with (a tagged PoseStamped stream becomes
/// AprilTagDetection on its first message; an empty one stays PoseStamped).
fn schema_for_module(module: &str) -> &'static str {
    match module {
        ODOMETRY_MODULE => cdr::ODOMETRY,
        POINTCLOUD2_MODULE => cdr::POINT_CLOUD2,
        TFMESSAGE_MODULE => cdr::TF_MESSAGE,
        CAMERA_INFO_MODULE => cdr::CAMERA_INFO,
        GRAPH3D_MODULE => cdr::GRAPH3D,
        DEFORMATION_NODE_MODULE => cdr::DEFORMATION_NODE,
        _ => cdr::POSE_STAMPED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_names_follow_mcap_to_db() {
        assert_eq!(stream_name_for_topic("/pointlio_odometry"), "pointlio_odometry");
        assert_eq!(stream_name_for_topic("/realsense/color_image/compressed"), "realsense_color_image_compressed");
        assert_eq!(stream_name_for_topic("global_map_aggregated"), "global_map_aggregated");
    }

    #[test]
    fn lcm_stamps_by_fingerprint() {
        let mut odometry = Odometry::default();
        odometry.header.stamp.sec = 1790629091;
        odometry.header.stamp.nsec = 500_000_000;
        assert_eq!(lcm_header_stamp(&odometry.encode()), Some(1790629091.5));
        // the layout mcap_edit.py documents: fingerprint, seq, then the stamp
        assert_eq!(&odometry.encode()[..8], &0x94e17f94648cf0e4u64.to_be_bytes());
        let mut cloud = PointCloud2::default();
        cloud.header.stamp.sec = 7;
        assert_eq!(&cloud.encode()[..8], &0xf5eb3da1c2853175u64.to_be_bytes());
        assert_eq!(lcm_header_stamp(&cloud.encode()), Some(7.0));
        let mut tf = TFMessage::default();
        assert_eq!(lcm_header_stamp(&tf.encode()), None);
        tf.transforms.push(Default::default());
        tf.transforms[0].header.stamp.sec = 9;
        assert_eq!(lcm_header_stamp(&tf.encode()), Some(9.0));
    }

    #[test]
    fn tagged_pose_stamped_becomes_a_tag_detection() {
        let pose = PoseStamped::default().encode();
        let tags = [("marker_id", "17".to_string()), ("sharpness", "12.5".to_string())];
        let (schema, bytes) = to_cdr(POSE_STAMPED_MODULE, &tags, &pose).unwrap();
        assert_eq!(schema, cdr::APRIL_TAG_DETECTION);
        let detection = cdr::decode_april_tag_detection(&bytes).unwrap();
        assert_eq!((detection.marker_id, detection.sharpness), (17, 12.5));
        assert!(detection.tag_px.is_nan());
        assert_eq!(to_cdr(POSE_STAMPED_MODULE, &[], &pose).unwrap().0, cdr::POSE_STAMPED);
    }
}
