// Port of silk/float/process_gains_FLP.c: silk_process_gains_FLP
// Floors gains using residual energy, quantizes, computes lambda.

use crate::gain_quant;
use crate::tables::SILK_QUANTIZATION_OFFSETS_Q10;
use crate::*;

// Tuning constants from silk/tuning_parameters.h
const LAMBDA_OFFSET: f32 = 1.2;
const LAMBDA_SPEECH_ACT: f32 = -0.2;
const LAMBDA_DELAYED_DECISIONS: f32 = -0.05;
const LAMBDA_INPUT_QUALITY: f32 = -0.1;
const LAMBDA_CODING_QUALITY: f32 = -0.2;
const LAMBDA_QUANT_OFFSET: f32 = 0.8;

/// Float gain processing: floor, quantize, compute lambda.
///
/// Port of silk_process_gains_FLP (process_gains_FLP.c).
///
/// On input, `gains` contains the noise_shape gains (pre-floor).
/// On output, `gains` contains the quantized gains (post-floor, post-quant).
/// Returns lambda (rate-distortion tradeoff parameter).
pub fn silk_process_gains_flp(
    gains: &mut [f32; MAX_NB_SUBFR], // I/O: per-subframe gains
    res_nrg: &[f32; MAX_NB_SUBFR],   // I: residual energy per subframe
    indices: &mut SideInfoIndices,
    last_gain_index: &mut i8,
    snr_db_q7: i32,
    nb_subfr: usize,
    signal_type: i32,
    ltp_pred_cod_gain: f32,
    n_states_delayed_decision: i32,
    speech_activity_q8: i32,
    input_quality: f32,
    coding_quality: f32,
    cond_coding: bool,
    input_tilt_q15: i32,
    subfr_length: usize,
) -> f32 {
    // Gain reduction for voiced (LTP coding gain high)
    if signal_type == TYPE_VOICED {
        let s = 1.0 - 0.5 * silk_sigmoid_f32(0.25 * (ltp_pred_cod_gain - 12.0));
        for item in gains.iter_mut().take(nb_subfr) {
            *item *= s;
        }
    }

    // Limit the quantized signal
    // InvMaxSqrVal = pow(2, 0.33 * (21 - SNR_dB)) / subfr_length
    let snr_db = snr_db_q7 as f32 / 128.0;
    let inv_max_sqr_val = (0.33f32 * (21.0 - snr_db)).exp2() / subfr_length as f32;

    for k in 0..nb_subfr {
        let gain = gains[k];
        let new_gain = (gain * gain + res_nrg[k] * inv_max_sqr_val).sqrt();
        gains[k] = new_gain.min(32767.0);
    }

    // Convert to Q16 for quantization
    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    for k in 0..nb_subfr {
        gains_q16[k] = (gains[k] * 65536.0).round() as i32;
    }

    // Save unquantized gains
    let _gains_unq_q16 = gains_q16;
    let _last_gain_index_prev = *last_gain_index;

    // Quantize gains
    gain_quant::silk_gains_quant(
        &mut indices.gains_indices,
        &mut gains_q16,
        last_gain_index,
        cond_coding,
        nb_subfr,
    );

    // Convert quantized gains back to float
    for k in 0..nb_subfr {
        gains[k] = gains_q16[k] as f32 / 65536.0;
    }

    // Set quantizer offset for voiced signals
    if signal_type == TYPE_VOICED {
        if ltp_pred_cod_gain + input_tilt_q15 as f32 * (1.0 / 32768.0) > 1.0 {
            indices.quant_offset_type = 0;
        } else {
            indices.quant_offset_type = 1;
        }
    }

    // Compute lambda (quantizer boundary adjustment)
    let quant_offset = SILK_QUANTIZATION_OFFSETS_Q10[(signal_type >> 1) as usize]
        [indices.quant_offset_type as usize] as f32
        / 1024.0;

    LAMBDA_OFFSET
        + LAMBDA_DELAYED_DECISIONS * n_states_delayed_decision as f32
        + LAMBDA_SPEECH_ACT * speech_activity_q8 as f32 * (1.0 / 256.0)
        + LAMBDA_INPUT_QUALITY * input_quality
        + LAMBDA_CODING_QUALITY * coding_quality
        + LAMBDA_QUANT_OFFSET * quant_offset
}

/// Simple sigmoid approximation matching C silk_sigmoid.
fn silk_sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
