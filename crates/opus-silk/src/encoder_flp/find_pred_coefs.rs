// Port of silk/float/find_pred_coefs_FLP.c: silk_find_pred_coefs_FLP
// Orchestrates the LPC/LTP prediction pipeline for the float encoder.
// This is the central analysis function that produces LPC, NLSFs, and residual energy.

use super::dsp::silk_scale_copy_vector_flp;
use super::find_lpc::silk_find_lpc_flp;
use super::wrappers::silk_process_nlsfs_flp;
use super::residual_energy::silk_residual_energy_flp;
use crate::*;

use crate::MAX_PREDICTION_POWER_GAIN;
const MAX_PREDICTION_POWER_GAIN_AFTER_RESET: f32 = 1e2; // silk/define.h

/// Float prediction coefficient analysis (unvoiced path only for now).
///
/// Port of silk_find_pred_coefs_FLP (find_pred_coefs_FLP.c).
///
/// Computes:
/// - gain-normalized input (LPC_in_pre)
/// - LPC via Burg + NLSF interpolation
/// - NLSF quantization + interpolation → PredCoef[2]
/// - Per-subframe residual energy (for process_gains)
pub fn silk_find_pred_coefs_flp(
    // Per-frame analysis inputs
    x: &[f32],                            // I: float signal (x_frame with la_shape lookback)
    gains: &[f32],                         // I: per-subframe gains from noise_shape
    coding_quality: f32,                   // I: from noise_shape
    ltp_pred_cod_gain: f32,                // I: LTP prediction coding gain (0 for unvoiced)
    // Encoder state (read/write)
    indices: &mut SideInfoIndices,
    prev_nlsf_q15: &mut [i16],            // I/O: previous frame's quantized NLSFs
    // Config
    predict_lpc_order: usize,
    subfr_length: usize,
    nb_subfr: usize,
    first_frame_after_reset: bool,
    use_interpolated_nlsfs: bool,
    speech_activity_q8: i32,
    nlsf_cb: &NlsfCbStruct,
    n_survivors: i32,
    // Outputs
    pred_coef: &mut [[f32; MAX_LPC_ORDER]; 2],  // O: float LPC [2 halves]
    res_nrg: &mut [f32; MAX_NB_SUBFR],           // O: residual energy per subframe
    nlsf_q15_out: &mut [i16],                     // O: quantized NLSFs
) {
    let d = predict_lpc_order;
    let burg_subfr = d + subfr_length;  // subfr_length including D preceding samples

    // Step 1: Compute invGains (C: invGains[i] = 1.0 / Gains[i])
    let mut inv_gains = [0.0f32; MAX_NB_SUBFR];
    for i in 0..nb_subfr {
        inv_gains[i] = 1.0 / gains[i].max(1e-12);
    }

    // Step 2: Create gain-normalized LPC_in_pre (unvoiced path)
    // C: silk_scale_copy_vector_FLP(x_pre_ptr, x_ptr, invGains[i], ...)
    // x points to x_frame which has la_shape lookback before the actual frame.
    // In the C code: x_ptr = x - predictLPCOrder, advancing by subfr_length per subframe.
    // Stack-allocated: max 4 * (16+80) = 384 floats = 1536 bytes
    const MAX_LPC_PRE_LEN: usize = MAX_NB_SUBFR * (MAX_LPC_ORDER + crate::MAX_SUB_FRAME_LENGTH);
    let mut lpc_in_pre = [0.0f32; MAX_LPC_PRE_LEN];
    {
        let mut pre_idx = 0usize;
        // x layout: [... la_shape lookback ... | frame data ...]
        // The caller provides x starting from the appropriate offset.
        // For unvoiced: x_ptr starts at x - predictLPCOrder, i.e.,
        // predictLPCOrder samples before each subframe.
        for i in 0..nb_subfr {
            let x_start = i * subfr_length; // start of subframe i's data in x
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
    }

    // Set LTP coefficients to zero (unvoiced path)
    // (Voiced path would fill these via silk_find_LTP_FLP + silk_quant_LTP_gains_FLP)

    // Step 3: minInvGain (C: find_pred_coefs_FLP.c lines 97-100)
    let min_inv_gain = if first_frame_after_reset {
        1.0 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET
    } else {
        let base = (ltp_pred_cod_gain / 3.0).exp2() / MAX_PREDICTION_POWER_GAIN;
        base / (0.25 + 0.75 * coding_quality)
    };

    // Step 4: Find LPC via Burg + NLSF interpolation
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

    // Step 5: Quantize NLSFs → PredCoef[2] (C: silk_process_NLSFs_FLP)
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

    // Step 6: Compute residual energy using quantized LPC
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
}
