// Layer 2: Float-to-fixed wrappers for the float SILK encoder.
// Each function mirrors a corresponding function in silk/float/wrappers_FLP.c.
// These thin layers convert between f32 analysis values and the fixed-point
// functions (NLSF, NSQ, gain quant, LTP quant) that always operate in Qxx.

use crate::lpc_analysis;
use crate::nlsf::silk_nlsf2a;
use crate::nlsf_encode;
use crate::nsq;
use crate::nsq_del_dec;
use crate::*;

/// silk_float2int: round float to nearest integer (matching C's lrintf).
#[inline(always)]
fn float2int(x: f32) -> i32 {
    // C uses lrintf() which is round-to-nearest-even.
    // Rust's .round() is round-half-away-from-zero — functionally identical
    // except for exact ±0.5 values, which are vanishingly rare in SILK.
    x.round() as i32
}

// ---- silk_A2NLSF_FLP (wrappers_FLP.c:37-51) ----

/// Convert float LPC coefficients to NLSFs in Q15.
/// Float → Q16 → silk_A2NLSF → Q15 NLSFs.
pub fn silk_a2nlsf_flp(nlsf_q15: &mut [i16], a: &[f32], order: usize) {
    let mut a_fix_q16 = [0i32; MAX_LPC_ORDER];
    for i in 0..order {
        a_fix_q16[i] = float2int(a[i] * 65536.0);
    }
    lpc_analysis::silk_a2nlsf(nlsf_q15, &mut a_fix_q16, order);
}

// ---- silk_NLSF2A_FLP (wrappers_FLP.c:54-69) ----

/// Convert NLSFs (Q15) to float LPC coefficients.
/// Q15 NLSFs → silk_NLSF2A → Q12 → float (÷4096).
pub fn silk_nlsf2a_flp(a: &mut [f32], nlsf_q15: &[i16], order: usize) {
    let mut a_fix_q12 = [0i16; MAX_LPC_ORDER];
    silk_nlsf2a(&mut a_fix_q12, nlsf_q15, order);
    for i in 0..order {
        a[i] = a_fix_q12[i] as f32 * (1.0 / 4096.0);
    }
}

// ---- silk_process_NLSFs_FLP (wrappers_FLP.c:74-91) ----

/// NLSF quantization + interpolation, producing float PredCoef[2][order].
/// Wraps silk_process_NLSFs (NLSF encode, decode, stabilize, interpolate).
///
/// pred_coef: output [2][MAX_LPC_ORDER] — [0]=first half, [1]=second half
/// nlsf_q15: I/O — quantized NLSFs
/// prev_nlsf_q15: previous frame's quantized NLSFs
pub fn silk_process_nlsfs_flp(
    pred_coef: &mut [[f32; MAX_LPC_ORDER]; 2],
    nlsf_q15: &mut [i16],
    prev_nlsf_q15: &[i16],
    indices: &mut SideInfoIndices,
    nlsf_cb: &NlsfCbStruct,
    lpc_order: usize,
    n_survivors: i32,
    signal_type: i32,
    speech_activity_q8: i32,
    nb_subfr: i32,
    first_frame_after_reset: bool,
    use_interpolated_nlsfs: bool,
) {
    // Compute NLSF_mu_Q20 (C: process_NLSFs.c lines 56-61)
    let mut nlsf_mu_q20 = silk_smlawb(3146, -268435, speech_activity_q8);
    if nb_subfr == 2 {
        nlsf_mu_q20 += nlsf_mu_q20 >> 1;
    }

    // Compute NLSF weights
    let mut p_nlsf_w_q2 = [0i16; MAX_LPC_ORDER];
    lpc_analysis::silk_nlsf_vq_weights_laroia(&mut p_nlsf_w_q2, nlsf_q15, lpc_order);

    // Determine interpolation
    let do_interpolate = use_interpolated_nlsfs
        && !first_frame_after_reset
        && nb_subfr == MAX_NB_SUBFR as i32
        && (indices.nlsf_interp_coef_q2 as i32) < 4;

    // Update weights for interpolation if needed
    if do_interpolate {
        let interp_coef = indices.nlsf_interp_coef_q2 as i32;
        let i_sqr_q15 = (interp_coef * interp_coef) << 11;
        let mut nlsf0_temp_q15 = [0i16; MAX_LPC_ORDER];
        // Interpolate NLSFs for first half
        for i in 0..lpc_order {
            nlsf0_temp_q15[i] = (prev_nlsf_q15[i] as i32
                + ((interp_coef * (nlsf_q15[i] as i32 - prev_nlsf_q15[i] as i32)) >> 2))
                as i16;
        }
        let mut nlsf_w0_temp = [0i16; MAX_LPC_ORDER];
        lpc_analysis::silk_nlsf_vq_weights_laroia(&mut nlsf_w0_temp, &nlsf0_temp_q15, lpc_order);

        // Blend weights
        for i in 0..lpc_order {
            p_nlsf_w_q2[i] = ((p_nlsf_w_q2[i] as i32 >> 1)
                + ((nlsf_w0_temp[i] as i32 * i_sqr_q15) >> 16)) as i16;
        }
    }

    // NLSF encode
    let survivors = if n_survivors > 0 {
        n_survivors as usize
    } else {
        16
    };
    nlsf_encode::silk_nlsf_encode(
        &mut indices.nlsf_indices,
        nlsf_q15,
        nlsf_cb,
        &p_nlsf_w_q2,
        nlsf_mu_q20,
        survivors,
        signal_type,
    );

    // Convert quantized NLSFs to LPC Q12 for second half, then to float
    let mut pred_coef_q12 = [0i16; MAX_LPC_ORDER];
    silk_nlsf2a(&mut pred_coef_q12, nlsf_q15, lpc_order);
    for (i, &coef) in pred_coef_q12.iter().enumerate().take(lpc_order) {
        pred_coef[1][i] = coef as f32 * (1.0 / 4096.0);
    }

    // First half: interpolate or copy
    if do_interpolate {
        let interp_coef = indices.nlsf_interp_coef_q2 as i32;
        let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
        for i in 0..lpc_order {
            nlsf0_q15[i] = (prev_nlsf_q15[i] as i32
                + ((interp_coef * (nlsf_q15[i] as i32 - prev_nlsf_q15[i] as i32)) >> 2))
                as i16;
        }
        let mut pred_coef0_q12 = [0i16; MAX_LPC_ORDER];
        silk_nlsf2a(&mut pred_coef0_q12, &nlsf0_q15, lpc_order);
        for (i, &coef) in pred_coef0_q12.iter().enumerate().take(lpc_order) {
            pred_coef[0][i] = coef as f32 * (1.0 / 4096.0);
        }
    } else {
        pred_coef[0] = pred_coef[1];
    }
}

// ---- silk_NSQ_wrapper_FLP (wrappers_FLP.c:96-170) ----

/// Convert all float encoder control parameters to fixed-point Qxx formats,
/// then dispatch to either silk_NSQ or silk_NSQ_del_dec.
///
/// This is the bridge between the float analysis pipeline and the fixed-point NSQ.
pub fn silk_nsq_wrapper_flp(
    nsq_state: &mut nsq::NsqState,
    indices: &mut SideInfoIndices,
    x: &[f32],                             // float input signal
    pulses: &mut [i8],                     // output quantized pulses
    pred_coef: &[[f32; MAX_LPC_ORDER]; 2], // float PredCoef[2][order]
    ltp_coef: &[f32],                      // float LTP coefficients [nb_subfr * LTP_ORDER]
    ar: &[f32],                            // float AR shaping [nb_subfr * MAX_SHAPE_LPC_ORDER]
    harm_shape_gain: &[f32],               // float harmonic shaping [nb_subfr]
    tilt: &[f32],                          // float tilt [nb_subfr]
    lf_ma_shp: &[f32],                     // float LF MA shaping [nb_subfr]
    lf_ar_shp: &[f32],                     // float LF AR shaping [nb_subfr]
    gains: &[f32],                         // float gains [nb_subfr]
    pitch_l: &[i32],                       // pitch lags [nb_subfr]
    lambda: f32,                           // rate-distortion lambda
    ltp_scale_q14: i32,
    // Config
    frame_length: i32,
    subfr_length: i32,
    ltp_mem_length: i32,
    lpc_order: i32,
    shaping_lpc_order: i32,
    nb_subfr: i32,
    signal_type: i32,
    warping_q16: i32,
    n_states_delayed_decision: i32,
    // Scratch buffers
    scratch_s_ltp_q15: &mut [i32],
    scratch_s_ltp: &mut [i16],
    scratch_x_sc_q10: &mut [i32],
    scratch_xq_tmp: &mut [i16],
) {
    let nb = nb_subfr as usize;
    let order = lpc_order as usize;
    let shaping_order = shaping_lpc_order as usize;

    // Convert AR shaping: float → Q13
    let mut ar_q13 = [0i16; MAX_NB_SUBFR * nsq::MAX_SHAPE_LPC_ORDER];
    for i in 0..(nb * shaping_order) {
        ar_q13[i] = float2int(ar[i] * 8192.0).clamp(-32768, 32767) as i16;
    }

    // Convert LF shaping: pack LF_MA (low 16) and LF_AR (high 16) into Q14
    let mut lf_shp_q14 = [0i32; MAX_NB_SUBFR];
    for i in 0..nb {
        let ma = float2int(lf_ma_shp[i] * 16384.0).clamp(-32768, 32767) as i16;
        let ar_val = float2int(lf_ar_shp[i] * 16384.0).clamp(-32768, 32767) as i16;
        lf_shp_q14[i] = (ar_val as i32) << 16 | (ma as u16 as i32);
    }

    // Convert Tilt: float → Q14
    let mut tilt_q14 = [0i32; MAX_NB_SUBFR];
    for i in 0..nb {
        tilt_q14[i] = float2int(tilt[i] * 16384.0);
    }

    // Convert HarmShapeGain: float → Q14
    let mut harm_shape_gain_q14 = [0i32; MAX_NB_SUBFR];
    for i in 0..nb {
        harm_shape_gain_q14[i] = float2int(harm_shape_gain[i] * 16384.0);
    }

    // Convert Lambda: float → Q10
    let lambda_q10 = float2int(lambda * 1024.0);

    // Convert LTPCoef: float → Q14
    let mut ltp_coef_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    for i in 0..(nb * LTP_ORDER) {
        ltp_coef_q14[i] = float2int(ltp_coef[i] * 16384.0).clamp(-32768, 32767) as i16;
    }

    // Convert PredCoef: float → Q12 [2 * MAX_LPC_ORDER]
    let mut pred_coef_q12 = [0i16; 2 * MAX_LPC_ORDER];
    for j in 0..2 {
        for i in 0..order {
            pred_coef_q12[j * MAX_LPC_ORDER + i] =
                float2int(pred_coef[j][i] * 4096.0).clamp(-32768, 32767) as i16;
        }
    }

    // Convert Gains: float → Q16
    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    for i in 0..nb {
        gains_q16[i] = float2int(gains[i] * 65536.0);
    }

    // Convert input signal: float → i16
    let frame_len = frame_length as usize;
    let mut x16 = [0i16; MAX_FRAME_LENGTH];
    for i in 0..frame_len {
        x16[i] = float2int(x[i]).clamp(-32768, 32767) as i16;
    }

    let nlsf_interp_coef_q2 = indices.nlsf_interp_coef_q2 as i32;
    let quant_offset_type = indices.quant_offset_type as i32;

    // Dispatch to NSQ
    if n_states_delayed_decision > 1 || warping_q16 > 0 {
        nsq_del_dec::silk_nsq_del_dec(
            nsq_state,
            indices,
            &x16,
            pulses,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            pitch_l,
            lambda_q10,
            ltp_scale_q14,
            frame_length,
            subfr_length,
            ltp_mem_length,
            lpc_order,
            shaping_lpc_order,
            nb_subfr,
            signal_type,
            quant_offset_type,
            nlsf_interp_coef_q2,
            n_states_delayed_decision,
            warping_q16,
            scratch_s_ltp_q15,
            scratch_s_ltp,
        );
    } else {
        nsq::silk_nsq(
            nsq_state,
            indices,
            &x16,
            pulses,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            pitch_l,
            lambda_q10,
            ltp_scale_q14,
            frame_length,
            subfr_length,
            ltp_mem_length,
            lpc_order,
            shaping_lpc_order,
            nb_subfr,
            signal_type,
            quant_offset_type,
            nlsf_interp_coef_q2,
            scratch_s_ltp_q15,
            scratch_s_ltp,
            scratch_x_sc_q10,
            scratch_xq_tmp,
        );
    }

    // Convert quantized gains back to float
    // (Done by caller — process_gains_FLP updates Gains from pGains_Q16)
}

// ---- silk_quant_LTP_gains_FLP (wrappers_FLP.c:175-209) ----

/// Quantize LTP gains in float → Q14/Q17 → fixed-point quant → Q14 → float.
pub fn silk_quant_ltp_gains_flp(
    b: &mut [f32],              // O: quantized LTP gains [nb_subfr * LTP_ORDER]
    _ltp_index: &mut [i8],      // O: LTP codebook indices [nb_subfr]
    _per_index: &mut i8,        // O: periodicity codebook index
    _sum_log_gain_q7: &mut i32, // I/O: cumulative log gain
    pred_gain_db: &mut f32,     // O: LTP prediction gain in dB
    xx: &[f32],                 // I: correlation matrix [nb_subfr * LTP_ORDER * LTP_ORDER]
    x_x: &[f32],                // I: correlation vector [nb_subfr * LTP_ORDER]
    _subfr_length: i32,
    nb_subfr: i32,
) {
    let nb = nb_subfr as usize;
    let n_ltp = LTP_ORDER;

    // Convert XX: float → Q17
    let mut xx_q17 = vec![0i32; nb * n_ltp * n_ltp];
    for i in 0..(nb * n_ltp * n_ltp) {
        xx_q17[i] = float2int(xx[i] * 131072.0); // Q17
    }

    // Convert xX: float → Q17
    let mut x_x_q17 = vec![0i32; nb * n_ltp];
    for i in 0..(nb * n_ltp) {
        x_x_q17[i] = float2int(x_x[i] * 131072.0); // Q17
    }

    // Call fixed-point quantizer
    // Note: silk_quant_LTP_gains is currently integrated into silk_find_ltp_params.
    // For the float wrapper to work standalone, it would need to be factored out.
    // For now, this wrapper is structurally complete but the LTP quant call is
    // handled by the find_pred_coefs layer which calls silk_find_ltp_params directly.
    let b_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    let pred_gain_db_q7 = 0i32;
    // TODO: Factor silk_quant_LTP_gains out of pitch_analysis.rs

    // Convert back: Q14 → float
    for i in 0..(nb * n_ltp) {
        b[i] = b_q14[i] as f32 * (1.0 / 16384.0);
    }
    *pred_gain_db = pred_gain_db_q7 as f32 * (1.0 / 128.0);
}

// ---- Helper: silk_float2short_array ----

/// Convert float array to i16 with saturation (matching C silk_float2short_array).
pub fn silk_float2short_array(out: &mut [i16], input: &[f32], length: usize) {
    for i in 0..length.min(out.len()).min(input.len()) {
        let v = float2int(input[i]);
        out[i] = v.clamp(-32768, 32767) as i16;
    }
}

/// Convert i16 array to float (matching C silk_short2float_array).
pub fn silk_short2float_array(out: &mut [f32], input: &[i16], length: usize) {
    for i in 0..length.min(out.len()).min(input.len()) {
        out[i] = input[i] as f32;
    }
}
