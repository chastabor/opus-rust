// Port of silk/float/noise_shape_analysis_FLP.c
// Computes spectral shaping filter parameters and per-subframe gains.
// This is the largest single analysis component in the float encoder.

use super::dsp::*;
use crate::*;

// Constants from silk/tuning_parameters.h
const SHAPE_WHITE_NOISE_FRACTION: f32 = 5e-5;
const LOW_FREQ_SHAPING: f32 = 4.0;
const HARM_SNR_INCR_DB: f32 = 2.0;
const BG_SNR_DECR_DB: f32 = 2.0;
const BANDWIDTH_EXPANSION: f32 = 0.95;
const FIND_PITCH_WHITE_NOISE_FRACTION: f32 = 1e-3;
const HARMONIC_SHAPING: f32 = 0.3;
const HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING: f32 = 0.2;
const HP_NOISE_COEF: f32 = 0.25;
const HARM_HP_NOISE_COEF: f32 = 0.35;
const FREQ_RESPONSE_LOW_UP: f32 = 0.5;
const FREQ_RESPONSE_LOW_DOWN: f32 = 0.3;
const MIN_QGAIN_DB: f32 = 2.0;

const LA_SHAPE_MS: usize = 5;
const SILK_MAX_SHAPE_LPC_ORDER: usize = 24;

/// Noise shape analysis output (matching silk_encoder_control_FLP fields).
pub struct NoiseShapeResult {
    pub ar: [f32; MAX_NB_SUBFR * SILK_MAX_SHAPE_LPC_ORDER],
    pub gains: [f32; MAX_NB_SUBFR],
    pub harm_shape_gain: [f32; MAX_NB_SUBFR],
    pub tilt: [f32; MAX_NB_SUBFR],
    pub lf_ma_shp: [f32; MAX_NB_SUBFR],
    pub lf_ar_shp: [f32; MAX_NB_SUBFR],
    pub coding_quality: f32,
    pub input_quality: f32,
    pub quant_offset_type: i8,
}

/// Helper: warped_gain (noise_shape_analysis_FLP.c lines 39-53)
/// Evaluates warped filter at DC to compute frequency warping compensation.
fn warped_gain(coefs: &[f32], lambda: f32, order: usize) -> f32 {
    let lambda = -lambda;
    let mut gain = coefs[order - 1];
    for i in (0..order - 1).rev() {
        gain = lambda * gain + coefs[i];
    }
    1.0 / (1.0 - lambda * gain)
}

/// Helper: limit_coefs (noise_shape_analysis_FLP.c lines 116-144)
/// Limits AR coefficients via iterative bandwidth expansion.
fn limit_coefs(ar: &mut [f32], limit: f32, order: usize) {
    let inv_limit = 1.0 / limit;
    for _iter in 0..10 {
        let mut max_abs = 0.0f32;
        for i in 0..order {
            max_abs = max_abs.max(ar[i].abs());
        }
        if max_abs <= limit {
            break;
        }
        let chirp = 0.999 - 0.8 * (1.0 - inv_limit * limit / max_abs);
        silk_bwexpander_flp(ar, order, chirp);
    }
}

/// Float noise shape analysis.
///
/// Port of silk_noise_shape_analysis_FLP (noise_shape_analysis_FLP.c).
///
/// Computes per-subframe: AR shaping filter, gains, tilt, harmonic shaping,
/// LF shaping, coding/input quality, and quantizer offset type.
pub fn silk_noise_shape_analysis_flp(
    x: &[f32],                             // I: signal with la_shape lookback
    pitch_lags: &[i32],                    // I: pitch lags [nb_subfr]
    is_voiced: bool,
    snr_db_q7: i32,
    speech_activity_q8: i32,
    input_quality_bands_q15: &[i32],       // I: from VAD [4 bands]
    input_tilt_q15: i32,
    fs_khz: i32,
    nb_subfr: usize,
    subfr_length: usize,
    shaping_lpc_order: usize,
    warping_q16: i32,
    prev_harm_smth: &mut f32,
    prev_tilt_smth: &mut f32,
    ltp_corr: f32,                         // I: pitch correlation (from find_pitch_lags)
) -> NoiseShapeResult {
    let la_shape = LA_SHAPE_MS * fs_khz as usize;

    // Input/coding quality
    let input_quality = 0.5 * (input_quality_bands_q15[0] + input_quality_bands_q15[1]) as f32
        / 32768.0;
    let snr_adj_db_base = snr_db_q7 as f32 / 128.0;
    let coding_quality = 1.0 / (1.0 + (-0.25 * (snr_adj_db_base - 20.0)).exp()); // sigmoid

    // SNR adjustment
    let mut snr_adj_db = snr_adj_db_base;
    if !is_voiced {
        // Background SNR decrement for low activity
        let b = 1.0 - speech_activity_q8 as f32 / 256.0;
        snr_adj_db -= BG_SNR_DECR_DB * coding_quality * (0.5 + 0.5 * input_quality) * b * b;
    }
    if is_voiced {
        snr_adj_db += HARM_SNR_INCR_DB * ltp_corr;
    }

    // Bandwidth expansion factor
    let bw_exp = BANDWIDTH_EXPANSION;
    let warping = warping_q16 as f32 / 65536.0;

    // Shape window length
    let shape_win_length = (3 * fs_khz as usize).max(1);
    let slope_part = ((shape_win_length + 2 * la_shape - shape_win_length) / 2 / 4) * 4;
    let slope_part = slope_part.max(4);
    let flat_part = shape_win_length;

    let mut result = NoiseShapeResult {
        ar: [0.0; MAX_NB_SUBFR * SILK_MAX_SHAPE_LPC_ORDER],
        gains: [0.0; MAX_NB_SUBFR],
        harm_shape_gain: [0.0; MAX_NB_SUBFR],
        tilt: [0.0; MAX_NB_SUBFR],
        lf_ma_shp: [0.0; MAX_NB_SUBFR],
        lf_ar_shp: [0.0; MAX_NB_SUBFR],
        coding_quality,
        input_quality,
        quant_offset_type: 0,
    };

    // Per-subframe processing
    // x_ptr starts at x (which already includes la_shape lookback from caller)
    let mut x_ptr_offset = 0usize;

    for k in 0..nb_subfr {
        let ar_offset = k * SILK_MAX_SHAPE_LPC_ORDER;

        // Window the input signal
        let win_len = (2 * slope_part + flat_part).min(x.len().saturating_sub(x_ptr_offset));
        let mut x_windowed = vec![0.0f32; win_len];
        if win_len >= 4 && x_ptr_offset + win_len <= x.len() {
            // Apply rising sine window
            let rising = slope_part.min(win_len);
            silk_apply_sine_window_flp(&mut x_windowed[..rising], &x[x_ptr_offset..], 1, (rising / 4) * 4);

            // Flat part (copy)
            let flat_start = rising;
            let flat_end = (flat_start + flat_part).min(win_len);
            for i in flat_start..flat_end {
                x_windowed[i] = x[x_ptr_offset + i];
            }

            // Falling sine window
            let fall_start = flat_end;
            let fall_len = (win_len - fall_start).min(slope_part);
            if fall_len >= 4 {
                silk_apply_sine_window_flp(
                    &mut x_windowed[fall_start..fall_start + (fall_len / 4) * 4],
                    &x[x_ptr_offset + fall_start..],
                    2,
                    (fall_len / 4) * 4,
                );
            }
        }

        // Autocorrelation
        let analysis_len = win_len;
        let mut auto_corr = vec![0.0f32; shaping_lpc_order + 1];
        if analysis_len > shaping_lpc_order {
            silk_autocorrelation_flp(&mut auto_corr, &x_windowed[..analysis_len], shaping_lpc_order + 1);
        }

        // Add white noise fraction
        auto_corr[0] += auto_corr[0] * SHAPE_WHITE_NOISE_FRACTION + 1.0;

        // Schur recursion → reflection coefficients + residual energy
        let mut rc = vec![0.0f32; shaping_lpc_order];
        let nrg = silk_schur_flp(&mut rc, &auto_corr, shaping_lpc_order);

        // Convert reflection coefficients to LPC
        silk_k2a_flp(&mut result.ar[ar_offset..ar_offset + shaping_lpc_order], &rc, shaping_lpc_order);

        // Initial gain = sqrt(residual energy)
        result.gains[k] = nrg.max(0.0).sqrt();

        // Warping gain adjustment
        if warping_q16 > 0 {
            result.gains[k] *= warped_gain(
                &result.ar[ar_offset..ar_offset + shaping_lpc_order],
                warping,
                shaping_lpc_order,
            );
        }

        // Bandwidth expansion
        silk_bwexpander_flp(&mut result.ar[ar_offset..ar_offset + shaping_lpc_order], shaping_lpc_order, bw_exp);

        // Coefficient limiting
        limit_coefs(&mut result.ar[ar_offset..ar_offset + shaping_lpc_order], 3.999, shaping_lpc_order);

        x_ptr_offset += subfr_length;
    }

    // Gain tweaking: gain_mult and gain_add (C: lines 290-295)
    let gain_mult = 2.0f32.powf(-0.16 * snr_adj_db);
    let gain_add = 2.0f32.powf(0.16 * MIN_QGAIN_DB);
    for k in 0..nb_subfr {
        result.gains[k] = result.gains[k] * gain_mult + gain_add;
    }

    // Low-frequency shaping
    for k in 0..nb_subfr {
        let strength = LOW_FREQ_SHAPING;
        if is_voiced {
            let lag = pitch_lags[k];
            let freq = fs_khz as f32 * 1000.0 / lag.max(1) as f32;
            let ratio = freq / (fs_khz as f32 * 1000.0);
            result.lf_ma_shp[k] = -FREQ_RESPONSE_LOW_UP * strength * ratio;
            result.lf_ar_shp[k] = FREQ_RESPONSE_LOW_DOWN * strength * ratio;
        }
    }

    // Harmonic shaping and tilt
    for k in 0..nb_subfr {
        if is_voiced {
            let harm_shp_gain = if coding_quality > 0.5 {
                HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING
            } else {
                HARMONIC_SHAPING
            };
            result.harm_shape_gain[k] = harm_shp_gain;
        }

        // Tilt: spectral tilt from input analysis
        result.tilt[k] = input_tilt_q15 as f32 / 32768.0;
    }

    // Smooth over subframes (exponential smoothing)
    let smooth_coef = 0.25f32;
    for k in 0..nb_subfr {
        *prev_harm_smth += smooth_coef * (result.harm_shape_gain[k] - *prev_harm_smth);
        result.harm_shape_gain[k] = *prev_harm_smth;

        *prev_tilt_smth += smooth_coef * (result.tilt[k] - *prev_tilt_smth);
        result.tilt[k] = *prev_tilt_smth;
    }

    // Quantizer offset type
    result.quant_offset_type = if is_voiced { 0 } else { 1 };

    result
}
