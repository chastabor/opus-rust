//! Cross-validation tests for CELT pitch analysis functions.

mod common;

use common::{assert_f32_slice_close, gen_noise, gen_sine_vec};
use opus_celt::pitch;
use opus_ffi::*;

// ── celt_pitch_xcorr ──

#[test]
fn pitch_xcorr_sine() {
    let len = 240;
    let max_pitch = 128;
    let x = gen_sine_vec(len, 440.0, 48000.0, 0.5);
    let y = gen_sine_vec(len + max_pitch, 440.0, 48000.0, 0.5);
    let mut rust_xcorr = vec![0.0f32; max_pitch];
    let mut c_xcorr = vec![0.0f32; max_pitch];
    pitch::celt_pitch_xcorr(&x, &y, &mut rust_xcorr, len, max_pitch);
    c_celt_pitch_xcorr(&x, &y, &mut c_xcorr, len, max_pitch);
    assert_f32_slice_close(&rust_xcorr, &c_xcorr, 1e-2, "pitch_xcorr(sine)");
}

#[test]
fn pitch_xcorr_noise() {
    let len = 120;
    let max_pitch = 64;
    let x = gen_noise(len, 42);
    let y = gen_noise(len + max_pitch, 99);
    let mut rust_xcorr = vec![0.0f32; max_pitch];
    let mut c_xcorr = vec![0.0f32; max_pitch];
    pitch::celt_pitch_xcorr(&x, &y, &mut rust_xcorr, len, max_pitch);
    c_celt_pitch_xcorr(&x, &y, &mut c_xcorr, len, max_pitch);
    assert_f32_slice_close(&rust_xcorr, &c_xcorr, 1e-2, "pitch_xcorr(noise)");
}

// ── pitch_downsample (mono) ──

#[test]
fn pitch_downsample_sine_mono() {
    let len = 240;
    let input_len = len * 2 + 1;
    let signal = gen_sine_vec(input_len, 440.0, 48000.0, 0.5);
    let mut rust_lp = vec![0.0f32; len];
    let mut c_signal = signal.clone();
    let mut c_lp = vec![0.0f32; len];
    pitch::pitch_downsample(&[&signal], &mut rust_lp, len, 1);
    c_pitch_downsample_mono(&mut c_signal, &mut c_lp, len);
    assert_f32_slice_close(&rust_lp, &c_lp, 1e-2, "pitch_downsample(sine, mono)");
}

#[test]
fn pitch_downsample_noise_mono() {
    let len = 120;
    let input_len = len * 2 + 1;
    let signal = gen_noise(input_len, 42);
    let mut rust_lp = vec![0.0f32; len];
    let mut c_signal = signal.clone();
    let mut c_lp = vec![0.0f32; len];
    pitch::pitch_downsample(&[&signal], &mut rust_lp, len, 1);
    c_pitch_downsample_mono(&mut c_signal, &mut c_lp, len);
    assert_f32_slice_close(&rust_lp, &c_lp, 1e-2, "pitch_downsample(noise, mono)");
}

// ── remove_doubling ──

#[test]
fn remove_doubling_periodic_signal() {
    let maxperiod = 1024;
    let n = 480;
    let total = maxperiod + n;
    let signal = gen_sine_vec(total, 110.0, 48000.0, 0.5);
    let mut rust_t0: usize = 200;
    let mut c_t0: i32 = 200;
    let mut c_signal = signal.clone();

    let rust_gain = pitch::remove_doubling(&signal, maxperiod, 30, n, &mut rust_t0, 200, 0.5);
    let c_gain = c_remove_doubling(&mut c_signal, maxperiod, 30, n, &mut c_t0, 200, 0.5);

    let diff = (rust_t0 as i32 - c_t0).abs();
    assert!(diff <= 2, "remove_doubling pitch: Rust T0={} C T0={} (diff={})", rust_t0, c_t0, diff);
    let gain_diff = (rust_gain - c_gain).abs();
    assert!(gain_diff < 0.2, "remove_doubling gain: Rust={:.4} C={:.4} diff={:.4}", rust_gain, c_gain, gain_diff);
}
