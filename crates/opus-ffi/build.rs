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
        .build();

    let lib_dir = dst.join("lib");
    let include_dir = dst.join("include").join("opus");

    // Tell cargo to link the static library.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=opus");
    println!("cargo:rustc-link-lib=m");

    // Compile wrapper.c (non-variadic CTL shims).
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    cc::Build::new()
        .file(manifest_dir.join("src/wrapper.c"))
        .include(&include_dir)
        .include(manifest_dir.join("opus-c/include"))
        .compile("opus_wrapper");
}
