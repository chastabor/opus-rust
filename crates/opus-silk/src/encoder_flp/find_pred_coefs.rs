// Port of silk/float/find_pred_coefs_FLP.c: silk_find_pred_coefs_FLP
// Orchestrates the LPC/LTP prediction pipeline for the float encoder.
// Handles both voiced (LTP analysis) and unvoiced paths.

use super::dsp::silk_scale_copy_vector_flp;
use super::find_lpc::silk_find_lpc_flp;
use super::find_ltp::silk_find_ltp_flp;
use super::find_ltp::silk_ltp_analysis_filter_flp;
use super::quant_ltp_gains::silk_quant_ltp_gains_flp;
use super::residual_energy::silk_residual_energy_flp;
use super::wrappers::silk_process_nlsfs_flp;
use crate::*;

use crate::MAX_PREDICTION_POWER_GAIN;
const MAX_PREDICTION_POWER_GAIN_AFTER_RESET: f32 = 1e2; // silk/define.h

/// Result from find_pred_coefs, including LTP state needed downstream.
pub struct PredCoefsResult {
    pub ltp_coef: [f32; MAX_NB_SUBFR * LTP_ORDER],
    pub ltp_pred_cod_gain: f32,
    pub sum_log_gain_q7: i32,
}

/// Float prediction coefficient analysis.
///
/// Port of silk_find_pred_coefs_FLP (find_pred_coefs_FLP.c).
///
/// For voiced frames: LTP analysis → quantize LTP gains → LTP filter → Burg
/// For unvoiced frames: scale by invGains → Burg
///
/// Both paths then: NLSF quantization → residual energy
///
/// `x_buf` is the full analysis buffer (ltp_mem + la_shape + frame).
/// `x_frame_offset` is where x_frame starts in x_buf.
/// `la_shape` is the lookback before x_frame for the actual frame data.
/// The actual frame starts at `x_buf[x_frame_offset + la_shape]`.
pub fn silk_find_pred_coefs_flp(
    x_buf: &[f32],                         // I: full x_buf
    x_frame_offset: usize,                // I: offset of x_frame in x_buf
    la_shape: usize,                       // I: la_shape lookback
    res_pitch: &[f32],                     // I: LPC residual from pitch analysis (full buffer)
    res_pitch_frame_offset: usize,         // I: offset where frame starts in res_pitch
    pitch_lags: &[i32; MAX_NB_SUBFR],     // I: pitch lags per subframe
    gains: &[f32],                         // I: per-subframe gains from noise_shape
    coding_quality: f32,                   // I: from noise_shape
    signal_type: i32,                      // I: TYPE_VOICED or TYPE_UNVOICED
    sum_log_gain_q7_in: i32,               // I: cumulative log gain from previous frames
    indices: &mut SideInfoIndices,
    prev_nlsf_q15: &mut [i16],
    predict_lpc_order: usize,
    subfr_length: usize,
    nb_subfr: usize,
    first_frame_after_reset: bool,
    use_interpolated_nlsfs: bool,
    speech_activity_q8: i32,
    nlsf_cb: &NlsfCbStruct,
    n_survivors: i32,
    // Outputs
    pred_coef: &mut [[f32; MAX_LPC_ORDER]; 2],
    res_nrg: &mut [f32; MAX_NB_SUBFR],
    nlsf_q15_out: &mut [i16],
) -> PredCoefsResult {
    let d = predict_lpc_order;
    let burg_subfr = d + subfr_length;

    // invGains
    let mut inv_gains = [0.0f32; MAX_NB_SUBFR];
    for i in 0..nb_subfr {
        inv_gains[i] = 1.0 / gains[i].max(1e-12);
    }

    // LPC_in_pre buffer
    const MAX_LPC_PRE_LEN: usize = MAX_NB_SUBFR * (MAX_LPC_ORDER + crate::MAX_SUB_FRAME_LENGTH);
    let mut lpc_in_pre = [0.0f32; MAX_LPC_PRE_LEN];

    let mut result = PredCoefsResult {
        ltp_coef: [0.0; MAX_NB_SUBFR * LTP_ORDER],
        ltp_pred_cod_gain: 0.0,
        sum_log_gain_q7: sum_log_gain_q7_in,
    };

    // The actual frame data starts at x_buf[x_frame_offset + la_shape]
    let frame_start = x_frame_offset + la_shape;

    if signal_type == TYPE_VOICED {
        // ---- VOICED path ----

        // LTP analysis: compute correlation matrices from pitch residual
        let mut xx_ltp = [0.0f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
        let mut x_x_ltp = [0.0f32; MAX_NB_SUBFR * LTP_ORDER];

        silk_find_ltp_flp(
            &mut xx_ltp,
            &mut x_x_ltp,
            res_pitch,
            res_pitch_frame_offset,
            pitch_lags,
            subfr_length,
            nb_subfr,
        );

        // Quantize LTP gains
        let mut cbk_index = [0i8; MAX_NB_SUBFR];
        let mut periodicity_index = 0i8;

        silk_quant_ltp_gains_flp(
            &mut result.ltp_coef,
            &mut cbk_index,
            &mut periodicity_index,
            &mut result.sum_log_gain_q7,
            &mut result.ltp_pred_cod_gain,
            &xx_ltp,
            &x_x_ltp,
            subfr_length as i32,
            nb_subfr,
        );

        // Store LTP indices
        indices.ltp_index = cbk_index;
        indices.per_index = periodicity_index;

        // LTP analysis filter: create gain-normalized LTP residual
        // C: silk_LTP_analysis_filter_FLP(LPC_in_pre, x - predictLPCOrder, ...)
        // x = x_frame, so x - predictLPCOrder = x_buf[x_frame_offset + la_shape - d]
        let x_ltp_offset = frame_start.saturating_sub(d);
        silk_ltp_analysis_filter_flp(
            &mut lpc_in_pre,
            x_buf,
            x_ltp_offset,
            &result.ltp_coef,
            pitch_lags,
            &inv_gains,
            subfr_length,
            nb_subfr,
            d, // pre_length = predictLPCOrder
        );
    } else {
        // ---- UNVOICED path ----
        // C: x_ptr = x - predictLPCOrder, advancing by subfr_length per subframe
        // x = x_frame + la_shape (the actual frame)
        let x = &x_buf[frame_start..];
        let mut pre_idx = 0usize;
        for i in 0..nb_subfr {
            let x_start = i * subfr_length;
            let src_start = if x_start >= d { x_start - d } else { 0 };
            let copy_len = burg_subfr.min(x.len().saturating_sub(src_start));
            silk_scale_copy_vector_flp(
                &mut lpc_in_pre[pre_idx..pre_idx + copy_len],
                &x[src_start..src_start + copy_len],
                inv_gains[i],
                copy_len,
            );
            pre_idx += burg_subfr;
        }

        result.ltp_pred_cod_gain = 0.0;
        result.sum_log_gain_q7 = 0;
    }

    // minInvGain (C: find_pred_coefs_FLP.c lines 97-100)
    let min_inv_gain = if first_frame_after_reset {
        1.0 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET
    } else {
        let base = (result.ltp_pred_cod_gain / 3.0).exp2() / MAX_PREDICTION_POWER_GAIN;
        base / (0.25 + 0.75 * coding_quality)
    };

    // Find LPC via Burg + NLSF interpolation
    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    silk_find_lpc_flp(
        &mut nlsf_q15,
        &mut indices.nlsf_interp_coef_q2,
        &lpc_in_pre,
        min_inv_gain,
        d,
        nb_subfr,
        burg_subfr,
        use_interpolated_nlsfs,
        first_frame_after_reset,
        prev_nlsf_q15,
    );

    // Quantize NLSFs → PredCoef[2]
    silk_process_nlsfs_flp(
        pred_coef,
        &mut nlsf_q15,
        prev_nlsf_q15,
        indices,
        nlsf_cb,
        d,
        n_survivors,
        indices.signal_type as i32,
        speech_activity_q8,
        nb_subfr as i32,
        first_frame_after_reset,
        use_interpolated_nlsfs,
    );

    // Compute residual energy
    silk_residual_energy_flp(
        res_nrg,
        &lpc_in_pre,
        pred_coef,
        gains,
        subfr_length,
        nb_subfr,
        d,
    );

    // Save NLSFs for next frame
    nlsf_q15_out[..d].copy_from_slice(&nlsf_q15[..d]);
    prev_nlsf_q15[..d].copy_from_slice(&nlsf_q15[..d]);

    result
}
