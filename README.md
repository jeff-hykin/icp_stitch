# icp_stitch

Offline loop-closure post-processing for dimos memory2 recordings — AprilTag pose-graph
optimization + point-to-plane ICP stitching — modelled after `gsc_pgo/scripts/post_process.py`,
written entirely in Rust.

Given a recording `.db` (memory2 sqlite store) it:

1. resolves the odometry / lidar / tag / camera streams (go2-legacy recordings are normalized automatically)
2. detects AprilTags (36h11) in the color stream if a raw tag stream is missing
3. selects keyframes, builds a factor graph (odometry betweens + tag landmark factors), solves with Levenberg-Marquardt (GTSAM)
4. finds revisit pairs and adds point-to-plane ICP loop-closure factors, then re-solves
5. writes the results back into the db: `<odom>_corrected`, `<lidar>_corrected`,
   `tf_deformation_nodes_corrected`, `pose_graph`, accumulated raycast maps,
   a `<lidar>_corrected.pc2.lcm` aggregate, and a `corrected_compare.rrd` rerun comparison

## Install

One line (requires [nix](https://nixos.org/download/) with flakes enabled):

```sh
nix profile install github:jeff-hykin/icp_stitch
```

Or run it without installing:

```sh
nix run github:jeff-hykin/icp_stitch -- --db path/to/recording.db
```

GTSAM (pinned to the same `develop` revision dimos uses) is built/fetched by nix via
[gtsam_shim](https://github.com/jeff-hykin/gtsam_shim); no system dependencies needed.

## Usage

```sh
icp_stitch --db path/to/recording.db
```

`--db` takes a filesystem path to the sqlite recording. Streams and frames are
auto-detected; every stage can be overridden or disabled:

```text
--db <DB>                    recording db path
--odom <ODOM>                odometry stream (auto-detected)
--lidar <LIDAR>              lidar stream (auto-detected)
--tags <TAGS>                raw AprilTag stream (default raw_april_tags)
--camera <CAMERA>            color image stream, used to detect tags when missing
--camera-info-stream <S>     CameraInfo stream (default camera_info)
--odom-tf <PARENT:CHILD>     tf edge the odometry stream drives (auto-resolved)
--world-frame <FRAME>        world frame label (default world)
--base-optical "x y z qx qy qz qw"   camera extrinsic override
--marker-length <M>          tag side length in meters (default 0.1)
--dict <DICT>                tag dictionary (default DICT_APRILTAG_36h11)
--ignore-tags <IDS>          comma-separated marker ids to skip
--corrected-suffix <SUF>     suffix for output streams (default _corrected)
--no-odom / --no-lidar       skip writing corrected odometry / lidar
--no-icp                     tag PGO only, no ICP closures
--no-lcm / --no-rrd / --no-accum   skip the .pc2.lcm / .rrd / accumulated maps
```

Plus a `solve tuning` group exposing every solver constant (`--keyframe-spacing-m`,
`--icp-radius-m`, `--icp-fitness-min`, `--closure-spacing`, ...); see `icp_stitch --help`.

The corrected streams are written back into the same db (only derived streams are
added — the recording itself is never modified), so the run is repeatable and the
outputs are regenerable.

## Helpers

`--helper <NAME>` runs a one-shot fixup on `--db` and exits instead of solving:

```sh
icp_stitch --db path/to/go2_recording.db --helper add_go2_camera_info
```

`add_go2_camera_info` writes the static go2 front-camera 720p intrinsics — the same
`front_camera_720.yaml` calibration the live dimos `Go2Connection` publishes, fisheye
`equidistant` model — as a `camera_info` stream in whatever frame the color images
carry. go2 recordings ship without one, which is why the solve skips their AprilTag
stage. It is additive and idempotent, and refuses to run if the images are not 720p.

## Development

```sh
nix develop   # cargo + rustc + GTSAM env
cargo test
```

On macOS, running the debug binary outside `nix build` needs the GTSAM dylib on the
fallback path: `export DYLD_FALLBACK_LIBRARY_PATH="$GTSAM_LIB_DIR"`.

## Provenance

Port of the open-source dimos `gsc_pgo` offline pipeline
([dimensionalOS/dimos#2587](https://github.com/dimensionalOS/dimos/pull/2587)):
`post_process.py`, `offline_pgo.py`, `make_rrd.py`, and their supporting utilities,
re-implemented in Rust (rusqlite, kornia-apriltag, GTSAM via gtsam_shim, rerun SDK).
