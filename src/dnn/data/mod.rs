// Model weight data module.
//
// Weight data is downloaded at build time by build.rs to the `model-data/`
// directory (gitignored). Models are loaded at runtime via the `parse_weights`
// + model init function path.
//
// The C data files (fargan_data.c, pitchdnn_data.c, etc.) are also copied
// to opus-ffi/opus-c/dnn/ so the C reference build can compile with DNN.

use std::path::PathBuf;

/// SHA256 hash of the current model weight tarball.
pub const MODEL_HASH: &str = "a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e";

/// Base URL for model weight downloads.
pub const MODEL_URL: &str = "https://media.xiph.org/opus/models";

/// Get the path to the downloaded model data directory.
/// Returns None if weights haven't been downloaded yet.
pub fn model_data_dir() -> Option<PathBuf> {
    // The model-data dir is at the crate root (same level as Cargo.toml).
    // At runtime, we locate it relative to the source file.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = crate_dir.join("model-data");
    if dir.join(".extracted").exists() {
        Some(dir)
    } else {
        None
    }
}

/// Get the path to a specific model data file (e.g., "dnn/pitchdnn_data.c").
pub fn model_data_file(relative_path: &str) -> Option<PathBuf> {
    let dir = model_data_dir()?;
    let path = dir.join(relative_path);
    if path.exists() { Some(path) } else { None }
}
