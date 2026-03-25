//! FFI cross-validation tests for Layer 0 float DSP functions.
//! Each test calls both the Rust port and the C reference with the
//! same input and asserts matching output.

use opus_ffi::*;
use opus_silk::encoder_flp::dsp::*;

const ORDER: usize = 16;

// ---- Test helpers ----

fn gen_sine(len: usize, freq: f32, fs: f32) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin())
        .collect()
}

fn gen_noise(len: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

fn assert_f64_eq(rust: f64, c: f64, tol: f64, name: &str) {
    let diff = (rust - c).abs();
    assert!(
        diff <= tol,
        "{}: Rust={} C={} diff={} (tol={})",
        name, rust, c, diff, tol
    );
}

fn assert_f32_eq(rust: f32, c: f32, tol: f32, name: &str) {
    let diff = (rust - c).abs();
    assert!(
        diff <= tol,
        "{}: Rust={} C={} diff={} (tol={})",
        name, rust, c, diff, tol
    );
}

fn assert_f32_slice_eq(rust: &[f32], c: &[f32], tol: f32, name: &str) {
    assert_eq!(rust.len(), c.len(), "{}: length mismatch", name);
    for i in 0..rust.len() {
        let diff = (rust[i] - c[i]).abs();
        assert!(
            diff <= tol,
            "{}[{}]: Rust={} C={} diff={} (tol={})",
            name, i, rust[i], c[i], diff, tol
        );
    }
}

// ---- Energy ----

#[test]
fn energy_flp_sine() {
    let signal = gen_sine(320, 440.0, 16000.0);
    let rust = silk_energy_flp(&signal);
    let c = c_silk_energy_flp(&signal);
    assert_f64_eq(rust, c, 1e-4, "energy_flp(sine)");
}

#[test]
fn energy_flp_noise() {
    let signal = gen_noise(256, 42);
    let rust = silk_energy_flp(&signal);
    let c = c_silk_energy_flp(&signal);
    assert_f64_eq(rust, c, 1e-4, "energy_flp(noise)");
}

// ---- Inner Product ----

#[test]
fn inner_product_flp_sine() {
    let a = gen_sine(320, 440.0, 16000.0);
    let b = gen_sine(320, 880.0, 16000.0);
    let rust = silk_inner_product_flp(&a, &b);
    let c = c_silk_inner_product_flp(&a, &b);
    assert_f64_eq(rust, c, 1e-4, "inner_product_flp(sine)");
}

#[test]
fn inner_product_flp_self() {
    let a = gen_noise(200, 99);
    let rust = silk_inner_product_flp(&a, &a);
    let c = c_silk_inner_product_flp(&a, &a);
    assert_f64_eq(rust, c, 1e-4, "inner_product_flp(self)");
}

// ---- Autocorrelation ----

#[test]
fn autocorrelation_flp_matches() {
    let signal = gen_sine(160, 440.0, 16000.0);
    let mut rust_res = [0.0f32; 17];
    let mut c_res = [0.0f32; 17];
    silk_autocorrelation_flp(&mut rust_res, &signal, 17);
    c_silk_autocorrelation_flp(&mut c_res, &signal, 17);
    assert_f32_slice_eq(&rust_res, &c_res, 1e-2, "autocorrelation_flp");
}

// ---- Schur ----

#[test]
fn schur_flp_matches() {
    let signal = gen_sine(160, 440.0, 16000.0);
    let mut auto_corr = [0.0f32; ORDER + 1];
    c_silk_autocorrelation_flp(&mut auto_corr, &signal, ORDER + 1);

    let mut rust_rc = [0.0f32; ORDER];
    let rust_nrg = silk_schur_flp(&mut rust_rc, &auto_corr, ORDER);

    let mut c_rc = [0.0f32; ORDER];
    let c_nrg = c_silk_schur_flp(&mut c_rc, &auto_corr, ORDER);

    assert_f32_eq(rust_nrg, c_nrg, 1e-2, "schur_flp nrg");
    assert_f32_slice_eq(&rust_rc, &c_rc, 1e-6, "schur_flp rc");
}

// ---- K2A ----

#[test]
fn k2a_flp_matches() {
    // Get reflection coefficients from Schur
    let signal = gen_sine(160, 440.0, 16000.0);
    let mut auto_corr = [0.0f32; ORDER + 1];
    c_silk_autocorrelation_flp(&mut auto_corr, &signal, ORDER + 1);
    let mut rc = [0.0f32; ORDER];
    c_silk_schur_flp(&mut rc, &auto_corr, ORDER);

    let mut rust_a = [0.0f32; ORDER];
    silk_k2a_flp(&mut rust_a, &rc, ORDER);

    let mut c_a = [0.0f32; ORDER];
    c_silk_k2a_flp(&mut c_a, &rc, ORDER);

    assert_f32_slice_eq(&rust_a, &c_a, 1e-6, "k2a_flp");
}

// ---- BWExpander ----

#[test]
fn bwexpander_flp_matches() {
    let mut rust_ar = [0.9f32, 0.5, -0.3, 0.1, 0.8, -0.4, 0.2, 0.6,
                       0.3, -0.1, 0.05, 0.7, -0.6, 0.4, -0.2, 0.15];
    let mut c_ar = rust_ar;

    silk_bwexpander_flp(&mut rust_ar, ORDER, 0.95);
    c_silk_bwexpander_flp(&mut c_ar, ORDER, 0.95);

    assert_f32_slice_eq(&rust_ar, &c_ar, 1e-6, "bwexpander_flp");
}

// ---- Apply Sine Window ----

#[test]
fn apply_sine_window_flp_rising() {
    let signal = gen_sine(64, 440.0, 16000.0);
    let mut rust_out = vec![0.0f32; 64];
    let mut c_out = vec![0.0f32; 64];

    silk_apply_sine_window_flp(&mut rust_out, &signal, 1, 64);
    c_silk_apply_sine_window_flp(&mut c_out, &signal, 1, 64);

    assert_f32_slice_eq(&rust_out, &c_out, 1e-5, "sine_window(rising)");
}

#[test]
fn apply_sine_window_flp_falling() {
    let signal = gen_sine(64, 440.0, 16000.0);
    let mut rust_out = vec![0.0f32; 64];
    let mut c_out = vec![0.0f32; 64];

    silk_apply_sine_window_flp(&mut rust_out, &signal, 2, 64);
    c_silk_apply_sine_window_flp(&mut c_out, &signal, 2, 64);

    assert_f32_slice_eq(&rust_out, &c_out, 1e-5, "sine_window(falling)");
}

// ---- Scale Copy Vector ----

#[test]
fn scale_copy_vector_flp_matches() {
    let input = gen_noise(100, 77);
    let mut rust_out = vec![0.0f32; 100];
    let mut c_out = vec![0.0f32; 100];

    silk_scale_copy_vector_flp(&mut rust_out, &input, 0.75, 100);
    c_silk_scale_copy_vector_flp(&mut c_out, &input, 0.75, 100);

    assert_f32_slice_eq(&rust_out, &c_out, 1e-6, "scale_copy_vector_flp");
}

// ---- LPC Analysis Filter ----

#[test]
fn lpc_analysis_filter_flp_matches() {
    let signal = gen_sine(320, 440.0, 16000.0);
    // Get LPC from schur + k2a
    let mut auto_corr = [0.0f32; ORDER + 1];
    c_silk_autocorrelation_flp(&mut auto_corr, &signal, ORDER + 1);
    let mut rc = [0.0f32; ORDER];
    c_silk_schur_flp(&mut rc, &auto_corr, ORDER);
    let mut lpc = [0.0f32; ORDER];
    c_silk_k2a_flp(&mut lpc, &rc, ORDER);

    let mut rust_res = vec![0.0f32; 320];
    let mut c_res = vec![0.0f32; 320];

    silk_lpc_analysis_filter_flp(&mut rust_res, &lpc, &signal, 320, ORDER);
    c_silk_lpc_analysis_filter_flp(&mut c_res, &lpc, &signal, 320, ORDER);

    assert_f32_slice_eq(&rust_res, &c_res, 1e-3, "lpc_analysis_filter_flp");
}

// ---- LPC Inverse Prediction Gain ----

#[test]
fn lpc_inv_pred_gain_flp_matches() {
    // Get stable LPC coefficients
    let signal = gen_sine(160, 440.0, 16000.0);
    let mut auto_corr = [0.0f32; ORDER + 1];
    c_silk_autocorrelation_flp(&mut auto_corr, &signal, ORDER + 1);
    let mut rc = [0.0f32; ORDER];
    c_silk_schur_flp(&mut rc, &auto_corr, ORDER);
    let mut lpc = [0.0f32; ORDER];
    c_silk_k2a_flp(&mut lpc, &rc, ORDER);

    let rust_gain = silk_lpc_inverse_pred_gain_flp(&lpc, ORDER);
    let c_gain = c_silk_lpc_inverse_pred_gain_flp(&lpc, ORDER);

    assert_f32_eq(rust_gain, c_gain, 1e-6, "lpc_inv_pred_gain_flp");
}
