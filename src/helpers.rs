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

//! One-shot recording fixups run by `--helper` instead of the solve.

use lcm_msgs::sensor_msgs::{CameraInfo, Image as LcmImage, RegionOfInterest};
use lcm_msgs::std_msgs;
use rusqlite::Connection;

use crate::detect::sec_nsec;
use crate::memory2;

/// go2 front camera, 720p, from dimos `robot/unitree/go2/front_camera_720.yaml`
/// (the same calibration the live `Go2Connection` publishes).
const GO2_FRONT_720_WIDTH: i32 = 1280;
const GO2_FRONT_720_HEIGHT: i32 = 720;
const GO2_FRONT_720_K: [f64; 9] = [
    797.4756164864929,
    0.0,
    643.5352167821186,
    0.0,
    796.4872112769983,
    349.2783605343087,
    0.0,
    0.0,
    1.0,
];
const GO2_FRONT_720_D: [f64; 4] = [
    -0.07309428880537933,
    -0.02341140740909078,
    -0.0069305931780026956,
    0.009238684474464793,
];
const GO2_FRONT_720_DISTORTION_MODEL: &str = "equidistant";

const CAMERA_INFO_MODULE: &str = "dimos.msgs.sensor_msgs.CameraInfo.CameraInfo";
const CAMERA_INFO_STREAM: &str = "camera_info";
const TF_STREAM: &str = "tf";

fn first_image(connection: &Connection, stream: &str) -> Result<(f64, LcmImage), String> {
    if !memory2::list_streams(connection)?.iter().any(|s| s == stream) {
        return Err(format!("no '{stream}' stream in this recording"));
    }
    if stream.contains('"') {
        return Err(format!("illegal stream name {stream:?}"));
    }
    let (ts, data) = connection
        .query_row(
            &format!(
                "SELECT meta.ts, blob.data FROM \"{stream}\" AS meta \
                 JOIN \"{stream}_blob\" AS blob ON meta.id = blob.id \
                 ORDER BY meta.ts LIMIT 1"
            ),
            [],
            |row| Ok((row.get::<_, f64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .map_err(|_| format!("'{stream}' is empty"))?;
    let data = memory2::decompress_if_lz4(data);
    let image = LcmImage::decode(&data).map_err(|error| {
        format!("'{stream}' does not hold sensor_msgs.Image payloads ({error})")
    })?;
    Ok((ts, image))
}

/// Write the static go2 front-camera intrinsics into a recording that has none,
/// so the AprilTag stage has something to work with. Additive and idempotent:
/// the stream is rewritten on a re-run and nothing else in the db is touched.
pub fn add_go2_camera_info(connection: &Connection, camera_stream: &str) -> Result<(), String> {
    let (ts, image) = first_image(connection, camera_stream)?;
    if image.width != GO2_FRONT_720_WIDTH || image.height != GO2_FRONT_720_HEIGHT {
        return Err(format!(
            "'{camera_stream}' is {}x{}, but the go2 calibration is for {GO2_FRONT_720_WIDTH}x\
             {GO2_FRONT_720_HEIGHT} -- these intrinsics do not apply to it",
            image.width, image.height
        ));
    }
    let frame_id = image.header.frame_id.clone();
    if frame_id.is_empty() {
        return Err(format!(
            "'{camera_stream}' images carry no frame_id, so the CameraInfo would have no optical \
             frame to hang the tag detections on"
        ));
    }
    let (sec, nsec) = sec_nsec(ts);
    let info = CameraInfo {
        header: std_msgs::Header {
            seq: 0,
            stamp: std_msgs::Time { sec, nsec },
            frame_id: frame_id.clone(),
        },
        height: GO2_FRONT_720_HEIGHT,
        width: GO2_FRONT_720_WIDTH,
        distortion_model: GO2_FRONT_720_DISTORTION_MODEL.to_string(),
        D: GO2_FRONT_720_D.to_vec(),
        K: GO2_FRONT_720_K,
        R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P: [
            GO2_FRONT_720_K[0],
            0.0,
            GO2_FRONT_720_K[2],
            0.0,
            0.0,
            GO2_FRONT_720_K[4],
            GO2_FRONT_720_K[5],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        binning_x: 0,
        binning_y: 0,
        roi: RegionOfInterest {
            x_offset: 0,
            y_offset: 0,
            height: 0,
            width: 0,
            do_rectify: false,
        },
    };
    memory2::delete_stream(connection, CAMERA_INFO_STREAM)?;
    memory2::create_stream(connection, CAMERA_INFO_STREAM, CAMERA_INFO_MODULE)?;
    memory2::append(connection, CAMERA_INFO_STREAM, ts, None, &[], &info.encode())?;
    println!(
        "wrote {CAMERA_INFO_STREAM}: go2 front camera 720p ({GO2_FRONT_720_DISTORTION_MODEL}) in \
         frame '{frame_id}'"
    );
    if !memory2::list_streams(connection)?.iter().any(|s| s == TF_STREAM) {
        println!(
            "NOTE: this recording has no '{TF_STREAM}' stream, so the base<-'{frame_id}' extrinsic \
             cannot be resolved -- the solve still needs --base-optical 'x y z qx qy qz qw'."
        );
    }
    Ok(())
}
