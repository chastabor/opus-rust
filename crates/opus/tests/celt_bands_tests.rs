//! Cross-validation tests for CELT band processing functions.

mod common;

use common::{assert_f32_slice_close, gen_noise, gen_sine_vec};
use opus_celt::bands;
use opus_celt::mode::CeltMode;
use opus_ffi::*;

// ── compute_band_energies ──

#[test]
fn compute_band_energies_sine_mono() {
    let m = CeltMode::get_mode();
    let lm = 0usize;
    let mm = 1usize << lm;
    let eff_end = m.nb_ebands;
    let c = 1;
    let freq_size = mm * m.ebands[eff_end] as usize;
    let freq = gen_sine_vec(freq_size, 1000.0, 48000.0, 0.5);

    let mut rust_band_e = vec![0.0f32; eff_end * c];
    let mut c_band_e = vec![0.0f32; eff_end * c];
    bands::compute_band_energies(m, &freq, &mut rust_band_e, eff_end, c, lm as i32);
    c_compute_band_energies(&freq, &mut c_band_e, eff_end, c, lm);
    assert_f32_slice_close(&rust_band_e, &c_band_e, 1e-4, "compute_band_energies(sine, mono)");
}

#[test]
fn compute_band_energies_noise_mono() {
    let m = CeltMode::get_mode();
    let lm = 0usize;
    let mm = 1usize << lm;
    let eff_end = m.nb_ebands;
    let c = 1;
    let freq_size = mm * m.ebands[eff_end] as usize;
    let freq = gen_noise(freq_size, 42);

    let mut rust_band_e = vec![0.0f32; eff_end * c];
    let mut c_band_e = vec![0.0f32; eff_end * c];
    bands::compute_band_energies(m, &freq, &mut rust_band_e, eff_end, c, lm as i32);
    c_compute_band_energies(&freq, &mut c_band_e, eff_end, c, lm);
    assert_f32_slice_close(&rust_band_e, &c_band_e, 1e-4, "compute_band_energies(noise, mono)");
}

// ── normalise_bands + denormalise_bands roundtrip ──

#[test]
fn normalise_denormalise_roundtrip() {
    let m = CeltMode::get_mode();
    let lm = 0usize;
    let mm = 1usize << lm;
    let eff_end = m.nb_ebands;
    let c = 1;
    let freq_size = mm * m.ebands[eff_end] as usize;
    let freq = gen_noise(freq_size, 42);

    let mut band_e = vec![0.0f32; eff_end * c];
    bands::compute_band_energies(m, &freq, &mut band_e, eff_end, c, lm as i32);

    let mut rust_norm = vec![0.0f32; freq_size];
    let mut c_norm = vec![0.0f32; freq_size];
    bands::normalise_bands(m, &freq, &mut rust_norm, &band_e, eff_end, c, mm);
    c_normalise_bands(&freq, &mut c_norm, &band_e, eff_end, c, mm);
    assert_f32_slice_close(&rust_norm, &c_norm, 1e-4, "normalise_bands");

    let mut band_log_e = vec![0.0f32; eff_end * c];
    bands::amp2_log2(m, eff_end, eff_end, &band_e, &mut band_log_e, c);

    let full_n = mm * m.short_mdct_size;
    let mut rust_freq = vec![0.0f32; full_n * c];
    let mut c_freq = vec![0.0f32; full_n * c];
    bands::denormalise_bands(m, &rust_norm, &mut rust_freq, &band_log_e, 0, eff_end, mm, 1, false);
    c_denormalise_bands(&c_norm, &mut c_freq, &band_log_e, 0, eff_end, mm, 1, false);
    assert_f32_slice_close(&rust_freq, &c_freq, 1e-2, "denormalise_bands");
}
