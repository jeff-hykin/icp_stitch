//! ROS 2 CDR (XCDR1, plain) for the message types icp_stitch reads from and writes into
//! an `.mcap`: decoded into the same `lcm_msgs` structs the `.db` path decodes, and
//! encoded from them, plus the ros2msg schema text a reader such as Foxglove needs.
//!
//! Alignment counts from the start of the body, after the 4-byte encapsulation header;
//! counting it from byte 0 turns every float past the first string into a denormal.

use lcm_msgs::geometry_msgs::{
    Point, Pose, PoseStamped, PoseWithCovariance, Quaternion, Transform, TransformStamped,
    Twist, TwistWithCovariance, Vector3,
};
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::{
    CameraInfo, Image, PointCloud2, PointField, RegionOfInterest,
};
use lcm_msgs::std_msgs::{Header, Time};
use lcm_msgs::tf2_msgs::TFMessage;

use crate::msgs;

pub const TF_MESSAGE: &str = "tf2_msgs/msg/TFMessage";
pub const ODOMETRY: &str = "nav_msgs/msg/Odometry";
pub const POINT_CLOUD2: &str = "sensor_msgs/msg/PointCloud2";
pub const CAMERA_INFO: &str = "sensor_msgs/msg/CameraInfo";
pub const IMAGE: &str = "sensor_msgs/msg/Image";
pub const COMPRESSED_IMAGE: &str = "sensor_msgs/msg/CompressedImage";
pub const POSE_STAMPED: &str = "geometry_msgs/msg/PoseStamped";
pub const APRIL_TAG_DETECTION: &str = "dimos_msgs/msg/AprilTagDetection";
pub const GRAPH3D: &str = "dimos_msgs/msg/Graph3D";
pub const DEFORMATION_NODE: &str = "dimos_msgs/msg/DeformationNode";

// ---- reading -----------------------------------------------------------------------

pub struct CdrReader<'a> {
    data: &'a [u8],
    at: usize,
    little: bool,
}

impl<'a> CdrReader<'a> {
    pub fn new(data: &'a [u8]) -> Result<CdrReader<'a>, String> {
        if data.len() < 4 {
            return Err(format!("CDR message is {} bytes, shorter than its header", data.len()));
        }
        Ok(CdrReader { data, at: 4, little: data[1] & 1 == 1 })
    }

    fn align(&mut self, width: usize) {
        let body = self.at - 4;
        self.at += (width - body % width) % width;
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(count).filter(|end| *end <= self.data.len()).ok_or_else(|| {
            format!("CDR message ends at {} bytes, wanted {count} more at {}", self.data.len(), self.at)
        })?;
        let bytes = &self.data[self.at..end];
        self.at = end;
        Ok(bytes)
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], String> {
        self.align(N);
        let mut bytes: [u8; N] = self.take(N)?.try_into().unwrap();
        if !self.little {
            bytes.reverse();
        }
        Ok(bytes)
    }

    pub fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    pub fn bool(&mut self) -> Result<bool, String> {
        Ok(self.u8()? != 0)
    }

    pub fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }

    pub fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.fixed()?))
    }

    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }

    pub fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.fixed()?))
    }

    pub fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.fixed()?))
    }

    pub fn string(&mut self) -> Result<String, String> {
        let length = self.u32()? as usize;
        let raw = self.take(length)?;
        let raw = raw.strip_suffix(&[0]).unwrap_or(raw);
        String::from_utf8(raw.to_vec()).map_err(|error| format!("CDR string: {error}"))
    }

    pub fn f64s<const N: usize>(&mut self) -> Result<[f64; N], String> {
        let mut out = [0.0; N];
        for slot in out.iter_mut() {
            *slot = self.f64()?;
        }
        Ok(out)
    }

    /// A `sequence<uint8>`, copied.
    pub fn byte_sequence(&mut self) -> Result<Vec<u8>, String> {
        let length = self.u32()? as usize;
        Ok(self.take(length)?.to_vec())
    }
}

fn read_time(r: &mut CdrReader) -> Result<Time, String> {
    Ok(Time { sec: r.i32()?, nsec: r.u32()? as i32 })
}

fn read_header(r: &mut CdrReader) -> Result<Header, String> {
    Ok(Header { seq: 0, stamp: read_time(r)?, frame_id: r.string()? })
}

fn read_vector3(r: &mut CdrReader) -> Result<Vector3, String> {
    let [x, y, z] = r.f64s()?;
    Ok(Vector3 { x, y, z })
}

fn read_quaternion(r: &mut CdrReader) -> Result<Quaternion, String> {
    let [x, y, z, w] = r.f64s()?;
    Ok(Quaternion { x, y, z, w })
}

fn read_pose(r: &mut CdrReader) -> Result<Pose, String> {
    let [x, y, z] = r.f64s()?;
    Ok(Pose { position: Point { x, y, z }, orientation: read_quaternion(r)? })
}

pub fn decode_tf_message(data: &[u8]) -> Result<TFMessage, String> {
    let mut r = CdrReader::new(data)?;
    let count = r.u32()? as usize;
    let mut transforms = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let header = read_header(&mut r)?;
        let child_frame_id = r.string()?;
        let translation = read_vector3(&mut r)?;
        let rotation = read_quaternion(&mut r)?;
        transforms.push(TransformStamped {
            header,
            child_frame_id,
            transform: Transform { translation, rotation },
        });
    }
    Ok(TFMessage { transforms })
}

pub fn decode_odometry(data: &[u8]) -> Result<Odometry, String> {
    let mut r = CdrReader::new(data)?;
    let header = read_header(&mut r)?;
    let child_frame_id = r.string()?;
    let pose = read_pose(&mut r)?;
    let pose_covariance = r.f64s::<36>()?;
    let linear = read_vector3(&mut r)?;
    let angular = read_vector3(&mut r)?;
    let twist_covariance = r.f64s::<36>()?;
    Ok(Odometry {
        header,
        child_frame_id,
        pose: PoseWithCovariance { pose, covariance: pose_covariance },
        twist: TwistWithCovariance { twist: Twist { linear, angular }, covariance: twist_covariance },
    })
}

pub fn decode_point_cloud2(data: &[u8]) -> Result<PointCloud2, String> {
    let mut r = CdrReader::new(data)?;
    let header = read_header(&mut r)?;
    let height = r.u32()? as i32;
    let width = r.u32()? as i32;
    let field_count = r.u32()? as usize;
    let mut fields = Vec::with_capacity(field_count.min(64));
    for _ in 0..field_count {
        let name = r.string()?;
        let offset = r.u32()? as i32;
        let datatype = r.u8()?;
        let count = r.u32()? as i32;
        fields.push(PointField { name, offset, datatype, count });
    }
    let is_bigendian = r.bool()?;
    let point_step = r.u32()? as i32;
    let row_step = r.u32()? as i32;
    let data = r.byte_sequence()?;
    let is_dense = r.bool()?;
    Ok(PointCloud2 { header, height, width, fields, is_bigendian, point_step, row_step, data, is_dense })
}

pub fn decode_camera_info(data: &[u8]) -> Result<CameraInfo, String> {
    let mut r = CdrReader::new(data)?;
    let header = read_header(&mut r)?;
    let height = r.u32()? as i32;
    let width = r.u32()? as i32;
    let distortion_model = r.string()?;
    let d_count = r.u32()? as usize;
    let mut distortion = Vec::with_capacity(d_count.min(64));
    for _ in 0..d_count {
        distortion.push(r.f64()?);
    }
    let k = r.f64s::<9>()?;
    let rectification = r.f64s::<9>()?;
    let projection = r.f64s::<12>()?;
    let binning_x = r.u32()? as i32;
    let binning_y = r.u32()? as i32;
    let roi = RegionOfInterest {
        x_offset: r.u32()? as i32,
        y_offset: r.u32()? as i32,
        height: r.u32()? as i32,
        width: r.u32()? as i32,
        do_rectify: r.bool()?,
    };
    Ok(CameraInfo {
        header,
        height,
        width,
        distortion_model,
        D: distortion,
        K: k,
        R: rectification,
        P: projection,
        binning_x,
        binning_y,
        roi,
    })
}

pub fn decode_image(data: &[u8]) -> Result<Image, String> {
    let mut r = CdrReader::new(data)?;
    let header = read_header(&mut r)?;
    let height = r.u32()? as i32;
    let width = r.u32()? as i32;
    let encoding = r.string()?;
    let is_bigendian = r.u8()?;
    let step = r.u32()? as i32;
    let data = r.byte_sequence()?;
    Ok(Image { header, height, width, encoding, is_bigendian, step, data })
}

/// A `CompressedImage` as the `Image` the tag detector takes: dimos stores a jpeg frame
/// as an `Image` whose encoding is `jpeg`; any other format keeps its own name.
pub fn decode_compressed_image(data: &[u8]) -> Result<Image, String> {
    let mut r = CdrReader::new(data)?;
    let header = read_header(&mut r)?;
    let format = r.string()?;
    let data = r.byte_sequence()?;
    let lower = format.to_ascii_lowercase();
    let encoding = if lower.contains("jpeg") || lower.contains("jpg") { "jpeg".to_string() } else { format };
    Ok(Image { header, encoding, data, ..Default::default() })
}

pub fn decode_pose_stamped(data: &[u8]) -> Result<PoseStamped, String> {
    let mut r = CdrReader::new(data)?;
    Ok(PoseStamped { header: read_header(&mut r)?, pose: read_pose(&mut r)? })
}

/// One raw AprilTag sighting: the camera<-marker pose plus the gate diagnostics the
/// `.db` path keeps in a row's tags (NaN where a diagnostic was not recorded).
#[derive(Debug, Clone, PartialEq)]
pub struct AprilTagDetection {
    pub header: Header,
    pub pose: Pose,
    pub marker_id: i64,
    pub sharpness: f64,
    pub reproj_px: f64,
    pub tag_px: f64,
    pub distance_m: f64,
    pub view_angle_deg: f64,
    pub lin_speed: f64,
    pub ang_speed: f64,
}

pub fn decode_april_tag_detection(data: &[u8]) -> Result<AprilTagDetection, String> {
    let mut r = CdrReader::new(data)?;
    Ok(AprilTagDetection {
        header: read_header(&mut r)?,
        pose: read_pose(&mut r)?,
        marker_id: r.i64()?,
        sharpness: r.f64()?,
        reproj_px: r.f64()?,
        tag_px: r.f64()?,
        distance_m: r.f64()?,
        view_angle_deg: r.f64()?,
        lin_speed: r.f64()?,
        ang_speed: r.f64()?,
    })
}

/// A CDR message of a known schema re-encoded as the LCM blob a memory2 `.db` row holds
/// for the same message, so every decoder downstream is shared with the `.db` path.
pub fn to_lcm(schema: &str, data: &[u8]) -> Result<Vec<u8>, String> {
    Ok(match schema {
        TF_MESSAGE => decode_tf_message(data)?.encode(),
        ODOMETRY => decode_odometry(data)?.encode(),
        POINT_CLOUD2 => decode_point_cloud2(data)?.encode(),
        CAMERA_INFO => decode_camera_info(data)?.encode(),
        IMAGE => decode_image(data)?.encode(),
        COMPRESSED_IMAGE => decode_compressed_image(data)?.encode(),
        POSE_STAMPED => decode_pose_stamped(data)?.encode(),
        APRIL_TAG_DETECTION => {
            let detection = decode_april_tag_detection(data)?;
            PoseStamped { header: detection.header, pose: detection.pose }.encode()
        }
        other => return Err(format!("no CDR decoder for {other}")),
    })
}

/// The header stamp in seconds (the first transform's for a TFMessage), read without
/// decoding the rest; None when the type has no header up front or the stamp is zero.
pub fn header_stamp(schema: &str, data: &[u8]) -> Option<f64> {
    let mut r = CdrReader::new(data).ok()?;
    if schema == TF_MESSAGE {
        let count = r.u32().ok()?;
        if count == 0 {
            return None;
        }
    } else if !matches!(
        schema,
        ODOMETRY | POINT_CLOUD2 | CAMERA_INFO | IMAGE | COMPRESSED_IMAGE | POSE_STAMPED | APRIL_TAG_DETECTION
    ) {
        return None;
    }
    let sec = r.i32().ok()?;
    let nanosec = r.u32().ok()?;
    let stamp = sec as f64 + nanosec as f64 / 1e9;
    (stamp != 0.0).then_some(stamp)
}

// ---- writing -----------------------------------------------------------------------

pub struct CdrWriter {
    out: Vec<u8>,
}

impl Default for CdrWriter {
    fn default() -> Self {
        CdrWriter::new()
    }
}

impl CdrWriter {
    /// Little-endian plain CDR (`CDR_LE`), what ROS 2 writes on every common platform.
    pub fn new() -> CdrWriter {
        CdrWriter { out: vec![0x00, 0x01, 0x00, 0x00] }
    }

    fn align(&mut self, width: usize) {
        let body = self.out.len() - 4;
        let padding = (width - body % width) % width;
        self.out.extend(std::iter::repeat_n(0u8, padding));
    }

    pub fn u8(&mut self, value: u8) {
        self.out.push(value);
    }

    pub fn bool(&mut self, value: bool) {
        self.out.push(value as u8);
    }

    pub fn u32(&mut self, value: u32) {
        self.align(4);
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn i32(&mut self, value: i32) {
        self.align(4);
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn u64(&mut self, value: u64) {
        self.align(8);
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn i64(&mut self, value: i64) {
        self.align(8);
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn f64(&mut self, value: f64) {
        self.align(8);
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn f64s(&mut self, values: &[f64]) {
        for value in values {
            self.f64(*value);
        }
    }

    pub fn string(&mut self, value: &str) {
        self.u32(value.len() as u32 + 1);
        self.out.extend_from_slice(value.as_bytes());
        self.out.push(0);
    }

    pub fn byte_sequence(&mut self, bytes: &[u8]) {
        self.u32(bytes.len() as u32);
        self.out.extend_from_slice(bytes);
    }

    pub fn finish(self) -> Vec<u8> {
        self.out
    }
}

fn write_header(w: &mut CdrWriter, header: &Header) {
    w.i32(header.stamp.sec);
    w.u32(header.stamp.nsec as u32);
    w.string(&header.frame_id);
}

fn write_pose(w: &mut CdrWriter, pose: &Pose) {
    let (p, q) = (&pose.position, &pose.orientation);
    w.f64s(&[p.x, p.y, p.z, q.x, q.y, q.z, q.w]);
}

/// A dimos `PoseStamped` (float seconds) as a `geometry_msgs/PoseStamped`.
fn write_local_pose_stamped(w: &mut CdrWriter, pose: &msgs::PoseStamped) {
    w.i32(pose.ts.floor() as i32);
    w.u32(((pose.ts - pose.ts.floor()) * 1e9) as u32);
    w.string(&pose.frame_id);
    w.f64s(&pose.position);
    w.f64s(&pose.orientation);
}

pub fn encode_tf_message(message: &TFMessage) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.u32(message.transforms.len() as u32);
    for entry in &message.transforms {
        write_header(&mut w, &entry.header);
        w.string(&entry.child_frame_id);
        let (t, q) = (&entry.transform.translation, &entry.transform.rotation);
        w.f64s(&[t.x, t.y, t.z, q.x, q.y, q.z, q.w]);
    }
    w.finish()
}

pub fn encode_odometry(message: &Odometry) -> Vec<u8> {
    let mut w = CdrWriter::new();
    write_header(&mut w, &message.header);
    w.string(&message.child_frame_id);
    write_pose(&mut w, &message.pose.pose);
    w.f64s(&message.pose.covariance);
    let (linear, angular) = (&message.twist.twist.linear, &message.twist.twist.angular);
    w.f64s(&[linear.x, linear.y, linear.z, angular.x, angular.y, angular.z]);
    w.f64s(&message.twist.covariance);
    w.finish()
}

pub fn encode_point_cloud2(message: &PointCloud2) -> Vec<u8> {
    let mut w = CdrWriter::new();
    write_header(&mut w, &message.header);
    w.u32(message.height as u32);
    w.u32(message.width as u32);
    w.u32(message.fields.len() as u32);
    for field in &message.fields {
        w.string(&field.name);
        w.u32(field.offset as u32);
        w.u8(field.datatype);
        w.u32(field.count as u32);
    }
    w.bool(message.is_bigendian);
    w.u32(message.point_step as u32);
    w.u32(message.row_step as u32);
    w.byte_sequence(&message.data);
    w.bool(message.is_dense);
    w.finish()
}

pub fn encode_camera_info(message: &CameraInfo) -> Vec<u8> {
    let mut w = CdrWriter::new();
    write_header(&mut w, &message.header);
    w.u32(message.height as u32);
    w.u32(message.width as u32);
    w.string(&message.distortion_model);
    w.u32(message.D.len() as u32);
    w.f64s(&message.D);
    w.f64s(&message.K);
    w.f64s(&message.R);
    w.f64s(&message.P);
    w.u32(message.binning_x as u32);
    w.u32(message.binning_y as u32);
    let roi = &message.roi;
    for value in [roi.x_offset, roi.y_offset, roi.height, roi.width] {
        w.u32(value as u32);
    }
    w.bool(roi.do_rectify);
    w.finish()
}

pub fn encode_pose_stamped(message: &PoseStamped) -> Vec<u8> {
    let mut w = CdrWriter::new();
    write_header(&mut w, &message.header);
    write_pose(&mut w, &message.pose);
    w.finish()
}

pub fn encode_april_tag_detection(message: &AprilTagDetection) -> Vec<u8> {
    let mut w = CdrWriter::new();
    write_header(&mut w, &message.header);
    write_pose(&mut w, &message.pose);
    w.i64(message.marker_id);
    w.f64s(&[
        message.sharpness,
        message.reproj_px,
        message.tag_px,
        message.distance_m,
        message.view_angle_deg,
        message.lin_speed,
        message.ang_speed,
    ]);
    w.finish()
}

pub fn encode_graph3d(graph: &msgs::Graph3D) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.f64(graph.ts);
    w.u32(graph.nodes.len() as u32);
    for node in &graph.nodes {
        write_local_pose_stamped(&mut w, &node.pose);
        w.u64(node.id);
        w.u64(node.metadata_id);
    }
    w.u32(graph.edges.len() as u32);
    for edge in &graph.edges {
        w.u64(edge.start_id);
        w.u64(edge.end_id);
        w.f64(edge.timestamp);
        w.u64(edge.metadata_id);
    }
    w.finish()
}

pub fn encode_deformation_node(node: &msgs::DeformationNode) -> Vec<u8> {
    let mut w = CdrWriter::new();
    w.u64(node.id);
    w.u64(node.tf_id);
    write_local_pose_stamped(&mut w, &node.pose);
    w.finish()
}

// ---- schemas -----------------------------------------------------------------------

const RULE: &str = "================================================================================";

/// Every message definition a schema below needs, keyed by its `MSG:` name.
fn definition(name: &str) -> &'static str {
    match name {
        "builtin_interfaces/Time" => "int32 sec\nuint32 nanosec",
        "std_msgs/Header" => "builtin_interfaces/Time stamp\nstring frame_id",
        "geometry_msgs/Vector3" | "geometry_msgs/Point" => "float64 x\nfloat64 y\nfloat64 z",
        "geometry_msgs/Quaternion" => "float64 x\nfloat64 y\nfloat64 z\nfloat64 w",
        "geometry_msgs/Pose" => "geometry_msgs/Point position\ngeometry_msgs/Quaternion orientation",
        "geometry_msgs/PoseStamped" => "std_msgs/Header header\ngeometry_msgs/Pose pose",
        "geometry_msgs/PoseWithCovariance" => "geometry_msgs/Pose pose\nfloat64[36] covariance",
        "geometry_msgs/Twist" => "geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular",
        "geometry_msgs/TwistWithCovariance" => "geometry_msgs/Twist twist\nfloat64[36] covariance",
        "geometry_msgs/Transform" => "geometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation",
        "geometry_msgs/TransformStamped" => {
            "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/Transform transform"
        }
        "sensor_msgs/PointField" => {
            "uint8 INT8=1\nuint8 UINT8=2\nuint8 INT16=3\nuint8 UINT16=4\nuint8 INT32=5\nuint8 UINT32=6\n\
             uint8 FLOAT32=7\nuint8 FLOAT64=8\nstring name\nuint32 offset\nuint8 datatype\nuint32 count"
        }
        "sensor_msgs/RegionOfInterest" => {
            "uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\nbool do_rectify"
        }
        "dimos_msgs/Node3D" => "geometry_msgs/PoseStamped pose\nuint64 id\nuint64 metadata_id",
        "dimos_msgs/Edge3D" => "uint64 start_id\nuint64 end_id\nfloat64 timestamp\nuint64 metadata_id",
        _ => "",
    }
}

fn compose(main: &str, dependencies: &[&str]) -> String {
    let mut text = main.to_string();
    for name in dependencies {
        text.push_str(&format!("\n{RULE}\nMSG: {name}\n{}", definition(name)));
    }
    text.push('\n');
    text
}

/// The ros2msg schema text for a type this module writes.
pub fn schema_text(schema: &str) -> Option<String> {
    const HEADER: [&str; 2] = ["std_msgs/Header", "builtin_interfaces/Time"];
    const POSE: [&str; 3] = ["geometry_msgs/Pose", "geometry_msgs/Point", "geometry_msgs/Quaternion"];
    let pose_stamped: Vec<&str> = ["geometry_msgs/PoseStamped"].into_iter().chain(HEADER).chain(POSE).collect();
    Some(match schema {
        TF_MESSAGE => compose(
            "geometry_msgs/TransformStamped[] transforms",
            &[
                "geometry_msgs/TransformStamped",
                "geometry_msgs/Transform",
                "geometry_msgs/Vector3",
                "geometry_msgs/Quaternion",
                HEADER[0],
                HEADER[1],
            ],
        ),
        ODOMETRY => compose(
            "std_msgs/Header header\nstring child_frame_id\n\
             geometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist",
            &[
                "geometry_msgs/PoseWithCovariance",
                POSE[0],
                POSE[1],
                POSE[2],
                "geometry_msgs/TwistWithCovariance",
                "geometry_msgs/Twist",
                "geometry_msgs/Vector3",
                HEADER[0],
                HEADER[1],
            ],
        ),
        POINT_CLOUD2 => compose(
            "std_msgs/Header header\nuint32 height\nuint32 width\nsensor_msgs/PointField[] fields\n\
             bool is_bigendian\nuint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense",
            &["sensor_msgs/PointField", HEADER[0], HEADER[1]],
        ),
        CAMERA_INFO => compose(
            "std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\n\
             float64[] d\nfloat64[9] k\nfloat64[9] r\nfloat64[12] p\nuint32 binning_x\n\
             uint32 binning_y\nsensor_msgs/RegionOfInterest roi",
            &["sensor_msgs/RegionOfInterest", HEADER[0], HEADER[1]],
        ),
        POSE_STAMPED => compose(definition("geometry_msgs/PoseStamped"), &[HEADER[0], HEADER[1], POSE[0], POSE[1], POSE[2]]),
        APRIL_TAG_DETECTION => compose(
            "# One raw AprilTag sighting: pose is camera <- marker in header.frame_id, plus the\n\
             # gate diagnostics icp_stitch filters on (NaN when not recorded)\n\
             std_msgs/Header header\ngeometry_msgs/Pose pose\nint64 marker_id\nfloat64 sharpness\n\
             float64 reproj_px\nfloat64 tag_px\nfloat64 distance_m\nfloat64 view_angle_deg\n\
             float64 lin_speed\nfloat64 ang_speed",
            &[HEADER[0], HEADER[1], POSE[0], POSE[1], POSE[2]],
        ),
        GRAPH3D => compose(
            "# icp_stitch's pose graph: optimized keyframes and the odometry edges between them\n\
             float64 ts\ndimos_msgs/Node3D[] nodes\ndimos_msgs/Edge3D[] edges",
            &[&["dimos_msgs/Node3D", "dimos_msgs/Edge3D"][..], &pose_stamped].concat(),
        ),
        DEFORMATION_NODE => compose(
            "# One keyframe of the tf deformation: the raw pose, then the optimized one, per id\n\
             uint64 id\nuint64 tf_id\ngeometry_msgs/PoseStamped pose",
            &pose_stamped,
        ),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(frame: &str) -> Header {
        Header { seq: 0, stamp: Time { sec: 1790629091, nsec: 123_456_789 }, frame_id: frame.into() }
    }

    #[test]
    fn odometry_round_trips_and_aligns_after_strings() {
        let mut message = Odometry { header: header("odom"), child_frame_id: "mid360_link".into(), ..Default::default() };
        message.pose.pose.position = Point { x: 1.5, y: -2.0, z: 0.25 };
        message.pose.pose.orientation = Quaternion { x: 0.0, y: 0.0, z: 0.6, w: 0.8 };
        message.pose.covariance[35] = 9.0;
        message.twist.twist.angular.z = 0.5;
        let bytes = encode_odometry(&message);
        // "odom\0" ends the header at body offset 8+4+5 = 17, so the child frame's length
        // pads to 20; the pose's first double then sits at a multiple of 8 in the body
        assert_eq!(&bytes[4 + 8..4 + 12], &5u32.to_le_bytes());
        assert_eq!(decode_odometry(&bytes).unwrap(), message);
        assert_eq!(header_stamp(ODOMETRY, &bytes), Some(1790629091.123456789));
        let lcm = to_lcm(ODOMETRY, &bytes).unwrap();
        assert_eq!(Odometry::decode(&lcm).unwrap(), message);
    }

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len()).step_by(2).map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap()).collect()
    }

    #[test]
    fn reads_a_point_cloud_written_by_rosbags() {
        // rosbags (ROS2_HUMBLE typestore) serialize_cdr of a PointCloud2: stamp (5, 7), frame
        // 'w', one FLOAT32 field 'x', one point 1.0f; then re-encoding gives the same bytes
        let bytes = hex(
            "00010000050000000700000002000000770000000100000001000000010000000200000078000000\
             000000000700000001000000000000000400000004000000040000000000803f01",
        );
        let cloud = decode_point_cloud2(&bytes).unwrap();
        assert_eq!((cloud.header.stamp.sec, cloud.header.stamp.nsec), (5, 7));
        assert_eq!(cloud.header.frame_id, "w");
        assert_eq!(cloud.fields[0].name, "x");
        assert_eq!(cloud.fields[0].datatype, 7);
        assert_eq!(cloud.data, 1.0f32.to_le_bytes());
        assert!(cloud.is_dense);
        assert_eq!(encode_point_cloud2(&cloud), bytes);
    }

    #[test]
    fn big_endian_cdr_decodes() {
        let mut bytes = vec![0x00, 0x00, 0x00, 0x00]; // CDR_BE
        bytes.extend(1u32.to_be_bytes()); // one transform
        bytes.extend(10i32.to_be_bytes());
        bytes.extend(0u32.to_be_bytes());
        bytes.extend(2u32.to_be_bytes());
        bytes.extend(b"a\0");
        bytes.extend([0, 0]); // pad to 4
        bytes.extend(2u32.to_be_bytes());
        bytes.extend(b"b\0");
        bytes.extend([0, 0, 0, 0, 0, 0]); // pad to 8
        for value in [1.0f64, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0] {
            bytes.extend(value.to_be_bytes());
        }
        let message = decode_tf_message(&bytes).unwrap();
        assert_eq!(message.transforms[0].header.frame_id, "a");
        assert_eq!(message.transforms[0].child_frame_id, "b");
        assert_eq!(message.transforms[0].transform.translation.y, 2.0);
        assert_eq!(header_stamp(TF_MESSAGE, &bytes), Some(10.0));
    }

    #[test]
    fn point_cloud_and_camera_info_round_trip() {
        let cloud = PointCloud2 {
            header: header("mid360_link"),
            height: 1,
            width: 2,
            fields: vec![
                PointField { name: "x".into(), offset: 0, datatype: 7, count: 1 },
                PointField { name: "intensity".into(), offset: 12, datatype: 7, count: 1 },
            ],
            is_bigendian: false,
            point_step: 16,
            row_step: 32,
            data: (0..32).collect(),
            is_dense: true,
        };
        assert_eq!(decode_point_cloud2(&encode_point_cloud2(&cloud)).unwrap(), cloud);
        let info = CameraInfo {
            header: header("camera_optical"),
            height: 480,
            width: 640,
            distortion_model: "plumb_bob".into(),
            D: vec![0.1, -0.2, 0.0, 0.0, 0.01],
            K: [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0],
            P: [1.0; 12],
            binning_x: 1,
            roi: RegionOfInterest { width: 640, do_rectify: true, ..Default::default() },
            ..Default::default()
        };
        assert_eq!(decode_camera_info(&encode_camera_info(&info)).unwrap(), info);
    }

    #[test]
    fn compressed_jpeg_becomes_a_jpeg_image() {
        let mut w = CdrWriter::new();
        write_header(&mut w, &header("color"));
        w.string("rgb8; jpeg compressed bgr8");
        w.byte_sequence(&[0xFF, 0xD8, 0xFF]);
        let image = decode_compressed_image(&w.finish()).unwrap();
        assert_eq!(image.encoding, "jpeg");
        assert_eq!(image.data, vec![0xFF, 0xD8, 0xFF]);
        assert_eq!(image.header.frame_id, "color");
    }

    #[test]
    fn tag_detection_round_trips() {
        let detection = AprilTagDetection {
            header: header("camera_optical"),
            pose: Pose { position: Point { x: 0.1, y: 0.2, z: 1.0 }, orientation: Quaternion { w: 1.0, ..Default::default() } },
            marker_id: 17,
            sharpness: 120.5,
            reproj_px: 0.3,
            tag_px: 42.0,
            distance_m: 1.02,
            view_angle_deg: 12.0,
            lin_speed: -1.0,
            ang_speed: f64::NAN,
        };
        let back = decode_april_tag_detection(&encode_april_tag_detection(&detection)).unwrap();
        assert_eq!(back.marker_id, 17);
        assert_eq!(back.tag_px, 42.0);
        assert!(back.ang_speed.is_nan());
    }

    #[test]
    fn truncated_message_is_an_error_not_a_panic() {
        let bytes = encode_odometry(&Odometry { header: header("odom"), ..Default::default() });
        for cut in [0, 3, 10, bytes.len() - 1] {
            assert!(decode_odometry(&bytes[..cut]).is_err());
        }
    }

    #[test]
    fn schemas_name_every_dependency_once() {
        for schema in [TF_MESSAGE, ODOMETRY, POINT_CLOUD2, CAMERA_INFO, POSE_STAMPED, APRIL_TAG_DETECTION, GRAPH3D, DEFORMATION_NODE] {
            let text = schema_text(schema).unwrap();
            let names: Vec<&str> = text.lines().filter_map(|line| line.strip_prefix("MSG: ")).collect();
            let mut unique = names.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(names.len(), unique.len(), "{schema}");
            for name in names {
                assert!(!definition(name).is_empty(), "{schema} needs {name}");
            }
        }
    }
}
