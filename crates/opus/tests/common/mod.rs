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
