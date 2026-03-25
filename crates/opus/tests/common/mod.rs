//! Shared test utilities for opus integration tests.

/// Sample rate used across tests.
pub const SAMPLE_RATE: i32 = 48000;

/// Generate a mono sine wave into `buf` at the given frequency and amplitude.
/// `offset` is the sample offset for phase continuity across frames.
pub fn gen_sine(buf: &mut [f32], offset: usize, freq: f32, amp: f32) {
    for i in 0..buf.len() {
        buf[i] = amp
            * (2.0 * std::f32::consts::PI * freq * (i + offset) as f32 / SAMPLE_RATE as f32)
                .sin();
    }
}

/// Generate a stereo sine wave with separate L/R frequencies.
pub fn gen_stereo_sine(
    buf: &mut [f32],
    samples: usize,
    offset: usize,
    freq_l: f32,
    freq_r: f32,
    amp: f32,
) {
    for i in 0..samples {
        let t = (i + offset) as f32 / SAMPLE_RATE as f32;
        buf[i * 2] = amp * (2.0 * std::f32::consts::PI * freq_l * t).sin();
        buf[i * 2 + 1] = amp * (2.0 * std::f32::consts::PI * freq_r * t).sin();
    }
}

/// Compute the RMS (root-mean-square) of a float buffer.
pub fn rms(buf: &[f32]) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    let sum: f64 = buf.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum / buf.len() as f64).sqrt()
}

/// Generate a sine wave as a new Vec. Frequency `freq` at sample rate `fs`.
pub fn gen_sine_vec(len: usize, freq: f32, fs: f32, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin())
        .collect()
}

/// Generate pseudo-random noise using LCG.
pub fn gen_noise(len: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

/// Assert two f32 values are close within tolerance.
pub fn assert_f32_close(rust: f32, c: f32, tol: f32, name: &str) {
    let diff = (rust - c).abs();
    assert!(
        diff <= tol,
        "{}: Rust={} C={} diff={} (tol={})",
        name, rust, c, diff, tol
    );
}

/// Assert two f32 slices match within tolerance. Reports worst element.
pub fn assert_f32_slice_close(rust: &[f32], c: &[f32], tol: f32, name: &str) {
    assert_eq!(rust.len(), c.len(), "{}: length mismatch {} vs {}", name, rust.len(), c.len());
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for i in 0..rust.len() {
        let diff = (rust[i] - c[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    assert!(
        max_diff <= tol,
        "{}: max diff={} at [{}] (Rust={} C={}), tol={}",
        name, max_diff, max_idx, rust[max_idx], c[max_idx], tol
    );
}
