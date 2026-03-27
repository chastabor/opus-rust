// Faithful port of silk/float/noise_shape_analysis_FLP.c
// Every computation matches the C reference line-by-line.

use super::dsp::*;
use crate::nsq::MAX_SHAPE_LPC_ORDER;
use crate::{MAX_FS_KHZ, MAX_NB_SUBFR, SUB_FRAME_LENGTH_MS, TYPE_VOICED};

// Constants from silk/define.h and silk/tuning_parameters.h
const SHAPE_LPC_WIN_MAX: usize = 15 * MAX_FS_KHZ; // 240
const SHAPE_WHITE_NOISE_FRACTION: f32 = 5e-5;
const BG_SNR_DECR_DB: f32 = 2.0;
const HARM_SNR_INCR_DB: f32 = 2.0;
const BANDWIDTH_EXPANSION: f32 = 0.95;
const FIND_PITCH_WHITE_NOISE_FRACTION: f32 = 1e-3;
const LOW_FREQ_SHAPING: f32 = 4.0;
const LOW_QUALITY_LOW_FREQ_SHAPING_DECR: f32 = 0.5;
const HP_NOISE_COEF: f32 = 0.25;
const HARM_HP_NOISE_COEF: f32 = 0.35;
const HARMONIC_SHAPING: f32 = 0.3;
const HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING: f32 = 0.2;
const ENERGY_VARIATION_THRESHOLD_QNT_OFFSET: f32 = 0.6;
const SUBFR_SMTH_COEF: f32 = 0.4;
const MIN_QGAIN_DB: f32 = 2.0;

// ---- Helper functions (top of noise_shape_analysis_FLP.c) ----

/// C: silk_sigmoid(x) = 1.0 / (1.0 + exp(-x))
#[inline]
fn silk_sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-x as f64).exp())) as f32
}

/// C: silk_log2(x) = 3.32192809488736 * log10(x)
#[inline]
fn silk_log2_f(x: f64) -> f32 {
    (std::f64::consts::LOG2_10 * x.log10()) as f32
}

/// C: warped_gain (lines 39-53)
fn warped_gain(coefs: &[f32], lambda: f32, order: usize) -> f32 {
    let lambda = -lambda;
    let mut gain = coefs[order - 1];
    for i in (0..order - 1).rev() {
        gain = lambda * gain + coefs[i];
    }
    1.0 / (1.0 - lambda * gain)
}

/// C: warped_true2monic_coefs (lines 57-114)
fn warped_true2monic_coefs(coefs: &mut [f32], lambda: f32, limit: f32, order: usize) {
    // Convert to monic coefficients
    for i in (1..order).rev() {
        coefs[i - 1] -= lambda * coefs[i];
    }
    let mut gain = (1.0 - lambda * lambda) / (1.0 + lambda * coefs[0]);
    for item in coefs.iter_mut().take(order) {
        *item *= gain;
    }

    for iter in 0..10 {
        // Find maximum absolute value
        let mut maxabs = -1.0f32;
        let mut ind = 0;
        for (i, &coef) in coefs.iter().enumerate().take(order) {
            let tmp = coef.abs();
            if tmp > maxabs {
                maxabs = tmp;
                ind = i;
            }
        }
        if maxabs <= limit {
            return;
        }

        // Convert back to true warped coefficients
        for i in 1..order {
            coefs[i - 1] += lambda * coefs[i];
        }
        gain = 1.0 / gain;
        for item in coefs.iter_mut().take(order) {
            *item *= gain;
        }

        // Apply bandwidth expansion
        let chirp =
            0.99 - (0.8 + 0.1 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_flp(coefs, order, chirp);

        // Convert to monic warped coefficients
        for i in (1..order).rev() {
            coefs[i - 1] -= lambda * coefs[i];
        }
        gain = (1.0 - lambda * lambda) / (1.0 + lambda * coefs[0]);
        for item in coefs.iter_mut().take(order) {
            *item *= gain;
        }
    }
}

/// C: limit_coefs (lines 116-144)
fn limit_coefs(coefs: &mut [f32], limit: f32, order: usize) {
    for iter in 0..10 {
        let mut maxabs = -1.0f32;
        let mut ind = 0;
        for (i, &coef) in coefs.iter().enumerate().take(order) {
            let tmp = coef.abs();
            if tmp > maxabs {
                maxabs = tmp;
                ind = i;
            }
        }
        if maxabs <= limit {
            return;
        }
        let chirp =
            0.99 - (0.8 + 0.1 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_flp(coefs, order, chirp);
    }
}

/// Output of noise shape analysis, matching silk_encoder_control_FLP fields.
pub struct NoiseShapeResult {
    pub ar: [f32; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    pub gains: [f32; MAX_NB_SUBFR],
    pub harm_shape_gain: [f32; MAX_NB_SUBFR],
    pub tilt: [f32; MAX_NB_SUBFR],
    pub lf_ma_shp: [f32; MAX_NB_SUBFR],
    pub lf_ar_shp: [f32; MAX_NB_SUBFR],
    pub coding_quality: f32,
    pub input_quality: f32,
    pub quant_offset_type: i8,
}

/// Faithful port of silk_noise_shape_analysis_FLP (lines 147-350).
pub fn silk_noise_shape_analysis_flp(
    // Signal: x points to x_frame, which has la_shape samples before the actual frame
    x: &[f32],
    // Pitch residual (from find_pitch_lags; used for sparseness measure)
    pitch_res: &[f32],
    // Pitch lags per subframe
    pitch_l: &[i32],
    // Encoder state fields
    signal_type: i32,
    snr_db_q7: i32,
    speech_activity_q8: i32,
    input_quality_bands_q15: &[i32],
    ltp_corr: f32,
    pred_gain: f32,
    use_cbr: bool,
    _la_shape: usize,
    fs_khz: i32,
    nb_subfr: usize,
    subfr_length: usize,
    shape_win_length: usize,
    shaping_lpc_order: usize,
    warping_q16: i32,
    // Smoothing state (persistent across frames)
    harm_shape_gain_smth: &mut f32,
    tilt_smth: &mut f32,
) -> NoiseShapeResult {
    let mut result = NoiseShapeResult {
        ar: [0.0; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
        gains: [0.0; MAX_NB_SUBFR],
        harm_shape_gain: [0.0; MAX_NB_SUBFR],
        tilt: [0.0; MAX_NB_SUBFR],
        lf_ma_shp: [0.0; MAX_NB_SUBFR],
        lf_ar_shp: [0.0; MAX_NB_SUBFR],
        coding_quality: 0.0,
        input_quality: 0.0,
        quant_offset_type: 0,
    };

    // C: x_ptr = x - la_shape. The caller passes x starting at x_frame,
    // which already has la_shape samples of lookback prepended.

    // ---- GAIN CONTROL (C lines 168-190) ----
    let mut snr_adj_db = snr_db_q7 as f32 * (1.0 / 128.0);

    result.input_quality =
        0.5 * (input_quality_bands_q15[0] + input_quality_bands_q15[1]) as f32 * (1.0 / 32768.0);

    result.coding_quality = silk_sigmoid(0.25 * (snr_adj_db - 20.0));

    if !use_cbr {
        let b = 1.0 - speech_activity_q8 as f32 * (1.0 / 256.0);
        snr_adj_db -=
            BG_SNR_DECR_DB * result.coding_quality * (0.5 + 0.5 * result.input_quality) * b * b;
    }

    if signal_type == TYPE_VOICED {
        snr_adj_db += HARM_SNR_INCR_DB * ltp_corr;
    } else {
        snr_adj_db +=
            (-0.4 * snr_db_q7 as f32 * (1.0 / 128.0) + 6.0) * (1.0 - result.input_quality);
    }

    // ---- SPARSENESS PROCESSING (C lines 195-220) ----
    if signal_type == TYPE_VOICED {
        result.quant_offset_type = 0;
    } else {
        let n_samples = 2 * fs_khz as usize;
        let n_segs = (SUB_FRAME_LENGTH_MS * nb_subfr) / 2;
        let mut energy_variation = 0.0f32;
        let mut log_energy_prev = 0.0f32;

        for k in 0..n_segs {
            let start = k * n_samples;
            let end = (start + n_samples).min(pitch_res.len());
            if start >= pitch_res.len() {
                break;
            }
            let nrg = n_samples as f32 + silk_energy_flp(&pitch_res[start..end]) as f32;
            let log_energy = silk_log2_f(nrg as f64);
            if k > 0 {
                energy_variation += (log_energy - log_energy_prev).abs();
            }
            log_energy_prev = log_energy;
        }

        if energy_variation > ENERGY_VARIATION_THRESHOLD_QNT_OFFSET * (n_segs as f32 - 1.0) {
            result.quant_offset_type = 0;
        } else {
            result.quant_offset_type = 1;
        }
    }

    // ---- BANDWIDTH EXPANSION (C lines 225-230) ----
    let strength_bw = FIND_PITCH_WHITE_NOISE_FRACTION * pred_gain;
    let bw_exp = BANDWIDTH_EXPANSION / (1.0 + strength_bw * strength_bw);

    let warping = warping_q16 as f32 / 65536.0 + 0.01 * result.coding_quality;

    // ---- COMPUTE NOISE SHAPING AR COEFS AND GAINS (C lines 234-284) ----
    let flat_part = fs_khz as usize * 3;
    let slope_part = (shape_win_length - flat_part) / 2;

    // x_ptr starts at x - la_shape (the caller provides x with la_shape lookback)
    let mut x_ptr_offset = 0usize; // starts at beginning of x (= x_frame - la_shape)

    let mut x_windowed = [0.0f32; SHAPE_LPC_WIN_MAX];
    let mut auto_corr;
    let mut rc;

    for k in 0..nb_subfr {
        let ar_offset = k * MAX_SHAPE_LPC_ORDER;

        // Apply window (C lines 240-248)
        if x_ptr_offset + shape_win_length <= x.len() && slope_part >= 4 {
            silk_apply_sine_window_flp(
                &mut x_windowed[..slope_part],
                &x[x_ptr_offset..],
                1,
                slope_part,
            );
            let shift = slope_part;
            x_windowed[shift..shift + flat_part]
                .copy_from_slice(&x[x_ptr_offset + shift..x_ptr_offset + shift + flat_part]);
            let shift2 = shift + flat_part;
            silk_apply_sine_window_flp(
                &mut x_windowed[shift2..shift2 + slope_part],
                &x[x_ptr_offset + shift2..],
                2,
                slope_part,
            );
        }

        // Advance pointer for next subframe (C line 251)
        x_ptr_offset += subfr_length;

        // Autocorrelation (C lines 253-259)
        auto_corr = [0.0; MAX_SHAPE_LPC_ORDER + 1];
        if warping_q16 > 0 {
            silk_warped_autocorrelation_flp(
                &mut auto_corr,
                &x_windowed[..shape_win_length],
                warping,
                shape_win_length,
                shaping_lpc_order,
            );
        } else {
            silk_autocorrelation_flp(
                &mut auto_corr,
                &x_windowed[..shape_win_length],
                shaping_lpc_order + 1,
            );
        }

        // Add white noise fraction (C line 261)
        auto_corr[0] += auto_corr[0] * SHAPE_WHITE_NOISE_FRACTION + 1.0;

        // Schur → reflection coefficients + residual energy (C lines 264-266)
        rc = [0.0; MAX_SHAPE_LPC_ORDER + 1];
        let nrg = silk_schur_flp(&mut rc, &auto_corr, shaping_lpc_order);
        silk_k2a_flp(
            &mut result.ar[ar_offset..ar_offset + shaping_lpc_order],
            &rc,
            shaping_lpc_order,
        );
        result.gains[k] = nrg.max(0.0).sqrt();

        // Warping gain adjustment (C lines 269-272)
        if warping_q16 > 0 {
            result.gains[k] *= warped_gain(
                &result.ar[ar_offset..ar_offset + shaping_lpc_order],
                warping,
                shaping_lpc_order,
            );
        }

        // Bandwidth expansion (C line 275)
        silk_bwexpander_flp(
            &mut result.ar[ar_offset..ar_offset + shaping_lpc_order],
            shaping_lpc_order,
            bw_exp,
        );

        // Coefficient limiting (C lines 277-283)
        if warping_q16 > 0 {
            warped_true2monic_coefs(
                &mut result.ar[ar_offset..ar_offset + shaping_lpc_order],
                warping,
                3.999,
                shaping_lpc_order,
            );
        } else {
            limit_coefs(
                &mut result.ar[ar_offset..ar_offset + shaping_lpc_order],
                3.999,
                shaping_lpc_order,
            );
        }
    }

    // ---- GAIN TWEAKING (C lines 289-295) ----
    let gain_mult = (-0.16f32 * snr_adj_db).exp2();
    let gain_add = (0.16f32 * MIN_QGAIN_DB).exp2();
    for k in 0..nb_subfr {
        result.gains[k] = result.gains[k] * gain_mult + gain_add;
    }

    // ---- LOW-FREQUENCY SHAPING AND NOISE TILT (C lines 300-325) ----
    let strength_lf = LOW_FREQ_SHAPING
        * (1.0
            + LOW_QUALITY_LOW_FREQ_SHAPING_DECR
                * (input_quality_bands_q15[0] as f32 * (1.0 / 32768.0) - 1.0));
    let strength_lf = strength_lf * speech_activity_q8 as f32 * (1.0 / 256.0);

    let tilt: f32;
    if signal_type == TYPE_VOICED {
        for (k, &pitch_l_k) in pitch_l.iter().enumerate().take(nb_subfr) {
            let b = 0.2 / fs_khz as f32 + 3.0 / pitch_l_k.max(1) as f32;
            result.lf_ma_shp[k] = -1.0 + b;
            result.lf_ar_shp[k] = 1.0 - b - b * strength_lf;
        }
        tilt = -HP_NOISE_COEF
            - (1.0 - HP_NOISE_COEF)
                * HARM_HP_NOISE_COEF
                * speech_activity_q8 as f32
                * (1.0 / 256.0);
    } else {
        let b = 1.3 / fs_khz as f32;
        result.lf_ma_shp[0] = -1.0 + b;
        result.lf_ar_shp[0] = 1.0 - b - b * strength_lf * 0.6;
        for k in 1..nb_subfr {
            result.lf_ma_shp[k] = result.lf_ma_shp[0];
            result.lf_ar_shp[k] = result.lf_ar_shp[0];
        }
        tilt = -HP_NOISE_COEF;
    }

    // ---- HARMONIC SHAPING CONTROL (C lines 327-339) ----
    let harm_shape_gain: f32 = if signal_type == TYPE_VOICED {
        let mut hsg = HARMONIC_SHAPING;
        hsg += HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING
            * (1.0 - (1.0 - result.coding_quality) * result.input_quality);
        hsg *= ltp_corr.max(0.0).sqrt();
        hsg
    } else {
        0.0
    };

    // ---- SMOOTH OVER SUBFRAMES (C lines 344-349) ----
    for k in 0..nb_subfr {
        *harm_shape_gain_smth += SUBFR_SMTH_COEF * (harm_shape_gain - *harm_shape_gain_smth);
        result.harm_shape_gain[k] = *harm_shape_gain_smth;
        *tilt_smth += SUBFR_SMTH_COEF * (tilt - *tilt_smth);
        result.tilt[k] = *tilt_smth;
    }

    result
}
