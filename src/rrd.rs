//! Comparison .rrd output (python `utils/recording.py build_and_open_rrd` port).

use std::path::Path;

#[allow(clippy::too_many_arguments)]
pub fn build_and_open_rrd(
    _db_path: &Path,
    _lidar_stream: &str,
    _odom_stream: &str,
    _tags_stream: &str,
    _world_frame: &str,
    _camera_stream: &str,
    _camera_info_stream: &str,
) -> Result<(), String> {
    println!("WARNING: rrd output not implemented yet -- skipping (pass --no-rrd to silence)");
    Ok(())
}
