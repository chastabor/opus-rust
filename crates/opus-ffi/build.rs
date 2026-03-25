use std::env;
use std::path::PathBuf;

fn main() {
    // Build C libopus from the vendored submodule using cmake.
    let dst = cmake::Config::new("opus-c")
        .define("OPUS_BUILD_PROGRAMS", "OFF")
        .define("OPUS_BUILD_TESTING", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("OPUS_DRED", "OFF")
        .define("OPUS_OSCE", "OFF")
        // Enable custom modes to expose FFT alloc/free and MDCT init/clear.
        .define("OPUS_CUSTOM_MODES", "ON")
        // Note: keeping float build (default) for compatibility with correctness tests.
        // With FIXED_POINT=ON, all SILK primitives (Burg, A2NLSF, NLSF encode) are
        // verified identical between C and Rust.
        .build();

    let lib_dir = dst.join("lib");
    let include_dir = dst.join("include").join("opus");

    // Tell cargo to link the static library.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=opus");
    println!("cargo:rustc-link-lib=m");

    // Compile wrapper.c (non-variadic CTL shims) and celt_wrapper.c
    // (CELT internal function shims for cross-validation).
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    cc::Build::new()
        .file(manifest_dir.join("src/wrapper.c"))
        .file(manifest_dir.join("src/celt_wrapper.c"))
        .include(&include_dir)
        .include(manifest_dir.join("opus-c/include"))
        .include(manifest_dir.join("opus-c/celt"))
        .compile("opus_wrapper");
}
