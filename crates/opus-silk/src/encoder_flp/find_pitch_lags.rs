// Port of silk/float/find_pitch_lags_FLP.c
// Performs pitch analysis: windowed LPC → LPC residual → pitch search.
// The core pitch search reuses the existing fixed-point implementation.

use super::dsp::*;
use crate::pitch_analysis;
use crate::{
    MAX_LPC_ORDER, MAX_NB_SUBFR, TYPE_NO_VOICE_ACTIVITY, TYPE_UNVOICED, TYPE_VOICED, silk_sat16,
};

const FIND_PITCH_BANDWIDTH_EXPANSION: f32 = 0.99;
const FIND_PITCH_WHITE_NOISE_FRACTION: f32 = 1e-3;

/// Output of pitch lag analysis.
pub struct PitchLagsResult {
    pub pitch_l: [i32; MAX_NB_SUBFR], // pitch lags per subframe
    pub lag_index: i16,               // lag index for entropy coding
    pub contour_index: i8,            // contour index
    pub ltp_corr: f32,                // normalized pitch correlation
    pub signal_type: i32,             // TYPE_VOICED or TYPE_UNVOICED
    pub pred_gain: f32,               // LPC prediction gain
    pub res_pitch: Vec<f32>,          // LPC residual (buf_len samples, for LTP analysis)
}

/// Float pitch lag analysis.
/// Port of silk_find_pitch_lags_FLP (find_pitch_lags_FLP.c).
///
/// `x` points to x_frame (start of analysis frame in x_buf).
/// The function accesses x - ltp_mem_length for the LPC analysis.
pub fn silk_find_pitch_lags_flp(
    x_buf: &[f32], // full x_buf (includes ltp_mem + la_shape + frame)
    ltp_mem_length: usize,
    frame_length: usize,
    la_pitch: usize,
    pitch_lpc_win_length: usize,
    pitch_estimation_lpc_order: usize,
    fs_khz: i32,
    nb_subfr: usize,
    complexity: i32,
    prev_lag: i32,
    prev_signal_type: i32,
    speech_activity_q8: i32,
    input_tilt_q15: i32,
    first_frame_after_reset: bool,
    prev_ltp_corr: f32,
) -> PitchLagsResult {
    let mut result = PitchLagsResult {
        pitch_l: [0; MAX_NB_SUBFR],
        lag_index: 0,
        contour_index: 0,
        ltp_corr: prev_ltp_corr,
        signal_type: TYPE_UNVOICED,
        pred_gain: 1.0,
        res_pitch: Vec::new(),
    };

    // buf_len = la_pitch + frame_length + ltp_mem_length
    let buf_len = la_pitch + frame_length + ltp_mem_length;

    // x_buf_ptr = start of signal for pitch analysis
    // In C: x_buf = x - ltp_mem_length, x is x_frame + la_shape
    // The caller should provide x_buf starting from the beginning of x_buf
    // x_frame is at offset ltp_mem_length in x_buf

    // Window the signal for LPC analysis
    let x_buf_start = buf_len.saturating_sub(pitch_lpc_win_length);
    let mut wsig = vec![0.0f32; pitch_lpc_win_length];

    if x_buf_start + pitch_lpc_win_length <= x_buf.len() && la_pitch >= 4 {
        // First la_pitch samples: rising sine window
        silk_apply_sine_window_flp(&mut wsig[..la_pitch], &x_buf[x_buf_start..], 1, la_pitch);

        // Middle: direct copy
        let mid_len = pitch_lpc_win_length - 2 * la_pitch;
        wsig[la_pitch..la_pitch + mid_len]
            .copy_from_slice(&x_buf[x_buf_start + la_pitch..x_buf_start + la_pitch + mid_len]);

        // Last la_pitch samples: falling sine window
        let last_start = la_pitch + mid_len;
        silk_apply_sine_window_flp(
            &mut wsig[last_start..last_start + la_pitch],
            &x_buf[x_buf_start + last_start..],
            2,
            la_pitch,
        );
    }

    // Autocorrelation
    let mut auto_corr = [0.0f32; MAX_LPC_ORDER + 1];
    silk_autocorrelation_flp(&mut auto_corr, &wsig, pitch_estimation_lpc_order + 1);

    // Add white noise fraction
    auto_corr[0] += auto_corr[0] * FIND_PITCH_WHITE_NOISE_FRACTION + 1.0;

    // Schur → reflection coefficients
    let mut refl_coef = [0.0f32; MAX_LPC_ORDER];
    let res_nrg = silk_schur_flp(&mut refl_coef, &auto_corr, pitch_estimation_lpc_order);

    // Prediction gain
    result.pred_gain = auto_corr[0] / res_nrg.max(1.0);

    // Convert to LPC
    let mut a = [0.0f32; MAX_LPC_ORDER];
    silk_k2a_flp(&mut a, &refl_coef, pitch_estimation_lpc_order);

    // Bandwidth expansion
    silk_bwexpander_flp(
        &mut a,
        pitch_estimation_lpc_order,
        FIND_PITCH_BANDWIDTH_EXPANSION,
    );

    // LPC analysis filter → residual
    let mut res = vec![0.0f32; buf_len];
    if buf_len <= x_buf.len() {
        silk_lpc_analysis_filter_flp(&mut res, &a, x_buf, buf_len, pitch_estimation_lpc_order);
    }

    // Store float residual for LTP analysis
    result.res_pitch = res;

    let mut res_i16 = vec![0i16; buf_len];
    for i in 0..buf_len {
        res_i16[i] = silk_sat16(result.res_pitch[i].round() as i32);
    }

    // Call pitch estimator (reuses existing fixed-point implementation)
    if prev_signal_type != TYPE_NO_VOICE_ACTIVITY && !first_frame_after_reset {
        // Derive pitch estimation thresholds from complexity (matching C)
        let pe_complexity = match complexity {
            0 | 2 => 0,
            1 | 3..=7 => 1,
            _ => 2,
        };
        let search_thres1_q16: i32 = match complexity {
            0 | 2 => 52429,
            1 | 3 => 49807,
            4 | 5 => 48497,
            6 | 7 => 47186,
            _ => 45875,
        };
        // Threshold for pitch estimator (C: find_pitch_lags_FLP.c lines 91-96)
        let thrhld = 0.6f32
            - 0.004 * pitch_estimation_lpc_order as f32
            - 0.1 * speech_activity_q8 as f32 / 256.0
            - 0.15 * (prev_signal_type >> 1) as f32
            - 0.1 * input_tilt_q15 as f32 / 32768.0;
        let search_thres2_q13 = (thrhld * 8192.0).round() as i32;

        // The C float encoder calls silk_pitch_analysis_core_FLP which internally
        // converts to fixed-point. We call the fixed-point core directly.
        let mut ltp_corr_q15 = 0i32;
        let ret = pitch_analysis::silk_pitch_analysis_core(
            &res_i16,
            &mut result.pitch_l,
            &mut result.lag_index,
            &mut result.contour_index,
            &mut ltp_corr_q15,
            prev_lag,
            search_thres1_q16,
            search_thres2_q13,
            fs_khz,
            pe_complexity,
            nb_subfr as i32,
        );

        if ret == 0 {
            result.signal_type = TYPE_VOICED;
            // Convert ltp_corr from Q15 to float
            result.ltp_corr = ltp_corr_q15 as f32 / 32768.0;
        } else {
            result.signal_type = TYPE_UNVOICED;
        }
    }

    result
}
