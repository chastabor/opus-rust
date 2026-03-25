// Port of SILK Noise Shaping Quantizer (NSQ) from silk/NSQ.c
//
// This implements the core scalar quantization with noise shaping
// for the SILK encoder. Faithfully ported from the C reference.

use crate::nlsf;
use crate::tables::*;
use crate::*;

// Constants from silk/define.h
pub const MAX_SHAPE_LPC_ORDER: usize = 24;
pub const NSQ_LPC_BUF_LENGTH: usize = MAX_LPC_ORDER;
pub const HARM_SHAPE_FIR_TAPS: usize = 3;

const NSQ_BUF_SIZE: usize = 2 * MAX_FRAME_LENGTH; // 640

/// NSQ state -- matches silk_nsq_state from silk/structs.h
/// Uses fixed-size arrays (not Vec) so Clone is a single memcpy with zero heap allocation.
#[derive(Clone)]
pub struct NsqState {
    pub xq: [i16; NSQ_BUF_SIZE],
    pub s_ltp_shp_q14: [i32; NSQ_BUF_SIZE],
    pub s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    pub s_lf_ar_shp_q14: i32,
    pub s_diff_shp_q14: i32,
    pub lag_prev: i32,
    pub s_ltp_buf_idx: i32,
    pub s_ltp_shp_buf_idx: i32,
    pub rand_seed: i32,
    pub prev_gain_q16: i32,
    pub rewhite_flag: i32,
}

impl Default for NsqState {
    fn default() -> Self {
        Self::new()
    }
}

impl NsqState {
    pub fn new() -> Self {
        Self {
            xq: [0i16; NSQ_BUF_SIZE],
            s_ltp_shp_q14: [0i32; NSQ_BUF_SIZE],
            s_lpc_q14: [0i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            s_ar2_q14: [0i32; MAX_SHAPE_LPC_ORDER],
            s_lf_ar_shp_q14: 0,
            s_diff_shp_q14: 0,
            lag_prev: 100,
            s_ltp_buf_idx: 0,
            s_ltp_shp_buf_idx: 0,
            rand_seed: 0,
            prev_gain_q16: 0, // C reference: zeroed by silk_memset in init_encoder
            rewhite_flag: 0,
        }
    }
}

/// Scale states for the NSQ (matching silk_nsq_scale_states)
#[allow(clippy::too_many_arguments)]
fn silk_nsq_scale_states(
    nsq: &mut NsqState,
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
    subfr_length: usize,
    ltp_mem_length: usize,
) {
    let lag = pitch_l[subfr] as usize;
    let inv_gain_q31 = silk_inverse32_varq(gains_q16[subfr].max(1), 47);

    // Scale input
    let inv_gain_q26 = silk_rshift_round(inv_gain_q31, 5);
    for i in 0..subfr_length {
        x_sc_q10[i] = silk_smulww_correct(x16[i] as i32, inv_gain_q26);
    }

    // After rewhitening the LTP state is un-scaled, so scale with inv_gain
    if nsq.rewhite_flag != 0 {
        let mut ig_q31 = inv_gain_q31;
        if subfr == 0 {
            // Do LTP downscaling
            ig_q31 = silk_smulwb(ig_q31, ltp_scale_q14) << 2;
        }
        let start = (nsq.s_ltp_buf_idx as usize).saturating_sub(lag + LTP_ORDER / 2);
        let end = nsq.s_ltp_buf_idx as usize;
        for i in start..end {
            if i < s_ltp_q15.len() && i < s_ltp.len() {
                s_ltp_q15[i] = silk_smulwb(ig_q31, s_ltp[i] as i32);
            }
        }
    }

    // Adjust for changing gain
    if gains_q16[subfr] != nsq.prev_gain_q16 {
        let gain_adj_q16 = silk_div32_varq(nsq.prev_gain_q16, gains_q16[subfr], 16);

        // Scale long-term shaping state
        let shp_start = (nsq.s_ltp_shp_buf_idx as usize).saturating_sub(ltp_mem_length);
        let shp_end = nsq.s_ltp_shp_buf_idx as usize;
        for i in shp_start..shp_end {
            if i < nsq.s_ltp_shp_q14.len() {
                nsq.s_ltp_shp_q14[i] = silk_smulww_correct(gain_adj_q16, nsq.s_ltp_shp_q14[i]);
            }
        }

        // Scale long-term prediction state
        if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
            let ltp_start = (nsq.s_ltp_buf_idx as usize).saturating_sub(lag + LTP_ORDER / 2);
            let ltp_end = nsq.s_ltp_buf_idx as usize;
            for i in ltp_start..ltp_end {
                if i < s_ltp_q15.len() {
                    s_ltp_q15[i] = silk_smulww_correct(gain_adj_q16, s_ltp_q15[i]);
                }
            }
        }

        nsq.s_lf_ar_shp_q14 = silk_smulww_correct(gain_adj_q16, nsq.s_lf_ar_shp_q14);
        nsq.s_diff_shp_q14 = silk_smulww_correct(gain_adj_q16, nsq.s_diff_shp_q14);

        // Scale short-term prediction and shaping states
        for i in 0..NSQ_LPC_BUF_LENGTH {
            nsq.s_lpc_q14[i] = silk_smulww_correct(gain_adj_q16, nsq.s_lpc_q14[i]);
        }
        for i in 0..MAX_SHAPE_LPC_ORDER {
            nsq.s_ar2_q14[i] = silk_smulww_correct(gain_adj_q16, nsq.s_ar2_q14[i]);
        }

        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}

/// Per-sample noise shape quantizer (matching silk_noise_shape_quantizer)
#[allow(clippy::too_many_arguments)]
fn silk_noise_shape_quantizer(
    nsq: &mut NsqState,
    signal_type: i32,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    xq: &mut [i16],
    s_ltp_q15: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_fir_packed_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: usize,
    shaping_lpc_order: usize,
    predict_lpc_order: usize,
) {
    let gain_q10 = gain_q16 >> 6;

    // Pointer to short-term AR state -- psLPC_Q14
    // In C: psLPC_Q14 = &NSQ->sLPC_Q14[NSQ_LPC_BUF_LENGTH - 1]
    // We use an index offset into s_lpc_q14
    let mut ps_lpc_idx = NSQ_LPC_BUF_LENGTH - 1;

    for i in 0..length {
        // Generate dither
        nsq.rand_seed = silk_rand(nsq.rand_seed);

        // Short-term prediction
        // C: silk_SMULWB(buf_Q14, coef_Q12) = (buf * (int16)coef) >> 16 → Q10
        let mut lpc_pred_q10: i64 = 0;
        for (j, a_q12_item) in a_q12.iter().enumerate().take(predict_lpc_order) {
            lpc_pred_q10 += (nsq.s_lpc_q14[ps_lpc_idx - j] as i64) * (*a_q12_item as i64);
        }
        let lpc_pred_q10 = (lpc_pred_q10 >> 16) as i32;

        // Long-term prediction
        let ltp_pred_q13 = if signal_type == TYPE_VOICED {
            let pred_lag_base = nsq.s_ltp_buf_idx as i64 - lag as i64 + (LTP_ORDER / 2) as i64;
            let mut acc: i64 = 2; // rounding bias
            for (k, b_q14_item) in b_q14.iter().enumerate().take(LTP_ORDER) {
                let idx = (pred_lag_base - k as i64) as usize;
                if idx < s_ltp_q15.len() {
                    // silk_SMLAWB: a + ((b * (c as i16)) >> 16)
                    acc += ((s_ltp_q15[idx] as i64) * (*b_q14_item as i64)) >> 16;
                }
            }
            acc as i32
        } else {
            0
        };

        // Noise shape feedback
        let mut n_ar_q12: i64 = 0;
        // AR noise shaping feedback using s_ar2_q14 and s_diff_shp_q14
        // silk_NSQ_noise_shape_feedback_loop
        for j in (0..shaping_lpc_order).rev() {
            n_ar_q12 += (nsq.s_ar2_q14[j] as i64) * (ar_shp_q13[j] as i64);
        }
        let mut n_ar_q12 = (n_ar_q12 >> 14) as i32;

        // Tilt
        // silk_SMLAWB(n_AR_Q12, NSQ->sLF_AR_shp_Q14, Tilt_Q14)
        n_ar_q12 = silk_smlawb(n_ar_q12, nsq.s_lf_ar_shp_q14, tilt_q14);

        // LF shaping
        let shp_idx = nsq.s_ltp_shp_buf_idx as usize;
        let prev_shp = if shp_idx > 0 && shp_idx - 1 < nsq.s_ltp_shp_q14.len() {
            nsq.s_ltp_shp_q14[shp_idx - 1]
        } else {
            0
        };
        // silk_SMULWB: (a * (b as i16)) >> 16
        let n_lf_q12 = silk_smulwb(prev_shp, lf_shp_q14);
        // silk_SMLAWT: a + ((b * (c >> 16)) >> 16)
        let n_lf_q12 = n_lf_q12
            .wrapping_add(((nsq.s_lf_ar_shp_q14 as i64 * (lf_shp_q14 as i64 >> 16)) >> 16) as i32);

        // Combine prediction and noise shaping signals
        let mut tmp1 = (lpc_pred_q10 << 2).wrapping_sub(n_ar_q12); // Q12
        tmp1 = tmp1.wrapping_sub(n_lf_q12); // Q12

        let r_q10 = if lag > 0 {
            // Symmetric, packed FIR coefficients for harmonic shaping
            let shp_lag_base =
                (nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS as i32 / 2) as usize;
            let shp0 = if shp_lag_base < nsq.s_ltp_shp_q14.len() {
                nsq.s_ltp_shp_q14[shp_lag_base]
            } else {
                0
            };
            let shp_m2 = if shp_lag_base >= 2 && shp_lag_base - 2 < nsq.s_ltp_shp_q14.len() {
                nsq.s_ltp_shp_q14[shp_lag_base - 2]
            } else {
                0
            };
            let shp_m1 = if shp_lag_base >= 1 && shp_lag_base - 1 < nsq.s_ltp_shp_q14.len() {
                nsq.s_ltp_shp_q14[shp_lag_base - 1]
            } else {
                0
            };

            let n_ltp_q13 = silk_smulwb(silk_add_sat32(shp0, shp_m2), harm_shape_fir_packed_q14);
            let n_ltp_q13 = n_ltp_q13.wrapping_add(
                ((shp_m1 as i64 * (harm_shape_fir_packed_q14 as i64 >> 16)) >> 16) as i32,
            );
            let n_ltp_q13 = n_ltp_q13 << 1;

            let tmp2 = ltp_pred_q13.wrapping_sub(n_ltp_q13); // Q13
            let combined = tmp2.wrapping_add(tmp1.wrapping_shl(1)); // Q13
            let combined_q10 = silk_rshift_round(combined, 3); // Q10

            x_sc_q10[i].wrapping_sub(combined_q10)
        } else {
            let combined_q10 = silk_rshift_round(tmp1, 2); // Q10
            x_sc_q10[i].wrapping_sub(combined_q10)
        };

        // Flip sign depending on dither
        let r_q10 = if nsq.rand_seed < 0 { -r_q10 } else { r_q10 };
        let r_q10 = r_q10.clamp(-(31 << 10), 30 << 10);

        // Find two quantization level candidates and measure their rate-distortion
        let q1_q10_initial = r_q10 - offset_q10;
        let mut q1_q0 = q1_q10_initial >> 10;

        if lambda_q10 > 2048 {
            let rdo_offset = lambda_q10 / 2 - 512;
            if q1_q10_initial > rdo_offset {
                q1_q0 = (q1_q10_initial - rdo_offset) >> 10;
            } else if q1_q10_initial < -rdo_offset {
                q1_q0 = (q1_q10_initial + rdo_offset) >> 10;
            } else if q1_q10_initial < 0 {
                q1_q0 = -1;
            } else {
                q1_q0 = 0;
            }
        }

        let (q1_q10, q2_q10, rd1_q20, rd2_q20);

        // Rate-distortion: C reference uses silk_SMULBB (truncates to i16) and
        // silk_SMLABB (truncates b,c to i16). We must match this behavior exactly.
        #[inline(always)]
        fn smulbb(a: i32, b: i32) -> i32 {
            (a as i16 as i32) * (b as i16 as i32)
        }
        #[inline(always)]
        fn smlabb(a: i32, b: i32, c: i32) -> i32 {
            a + (b as i16 as i32) * (c as i16 as i32)
        }

        if q1_q0 > 0 {
            let q1 = (q1_q0 << 10) - QUANT_LEVEL_ADJUST_Q10 + offset_q10;
            let q2 = q1 + 1024;
            q1_q10 = q1;
            q2_q10 = q2;
            rd1_q20 = smulbb(q1, lambda_q10);
            rd2_q20 = smulbb(q2, lambda_q10);
        } else if q1_q0 == 0 {
            let q1 = offset_q10;
            let q2 = q1 + 1024 - QUANT_LEVEL_ADJUST_Q10;
            q1_q10 = q1;
            q2_q10 = q2;
            rd1_q20 = smulbb(q1, lambda_q10);
            rd2_q20 = smulbb(q2, lambda_q10);
        } else if q1_q0 == -1 {
            let q2 = offset_q10;
            let q1 = q2 - 1024 + QUANT_LEVEL_ADJUST_Q10;
            q1_q10 = q1;
            q2_q10 = q2;
            rd1_q20 = smulbb(-q1, lambda_q10);
            rd2_q20 = smulbb(q2, lambda_q10);
        } else {
            // q1_q0 < -1
            let q1 = (q1_q0 << 10) + QUANT_LEVEL_ADJUST_Q10 + offset_q10;
            let q2 = q1 + 1024;
            q1_q10 = q1;
            q2_q10 = q2;
            rd1_q20 = smulbb(-q1, lambda_q10);
            rd2_q20 = smulbb(-q2, lambda_q10);
        };

        let rr1 = r_q10 - q1_q10;
        let rd1 = smlabb(rd1_q20, rr1, rr1);
        let rr2 = r_q10 - q2_q10;
        let rd2 = smlabb(rd2_q20, rr2, rr2);

        let q1_q10_final = if rd2 < rd1 { q2_q10 } else { q1_q10 };

        pulses[i] = silk_rshift_round(q1_q10_final, 10) as i8;

        // Excitation
        let mut exc_q14 = q1_q10_final << 4;
        if nsq.rand_seed < 0 {
            exc_q14 = -exc_q14;
        }

        // Add predictions
        let lpc_exc_q14 = exc_q14.wrapping_add(ltp_pred_q13 << 1);
        let xq_q14 = lpc_exc_q14.wrapping_add(lpc_pred_q10 << 4);

        // Scale XQ back to normal level before saving
        xq[i] = silk_sat16(silk_rshift_round(silk_smulww_correct(xq_q14, gain_q10), 8));

        // Update states
        ps_lpc_idx += 1;
        if ps_lpc_idx < nsq.s_lpc_q14.len() {
            nsq.s_lpc_q14[ps_lpc_idx] = xq_q14;
        }

        // Noise shaping state updates
        nsq.s_diff_shp_q14 = xq_q14.wrapping_sub(x_sc_q10[i] << 4);
        let s_lf_ar_shp_q14 = nsq.s_diff_shp_q14.wrapping_sub(n_ar_q12 << 2);
        nsq.s_lf_ar_shp_q14 = s_lf_ar_shp_q14;

        let shp_buf_idx = nsq.s_ltp_shp_buf_idx as usize;
        if shp_buf_idx < nsq.s_ltp_shp_q14.len() {
            nsq.s_ltp_shp_q14[shp_buf_idx] = s_lf_ar_shp_q14.wrapping_sub(n_lf_q12 << 2);
        }
        let ltp_buf_idx = nsq.s_ltp_buf_idx as usize;
        if ltp_buf_idx < s_ltp_q15.len() {
            s_ltp_q15[ltp_buf_idx] = lpc_exc_q14 << 1;
        }

        // Update AR shaping state: shift s_ar2_q14 (memmove is faster than element loop)
        if shaping_lpc_order > 1 {
            nsq.s_ar2_q14.copy_within(0..shaping_lpc_order - 1, 1);
        }
        if shaping_lpc_order > 0 {
            nsq.s_ar2_q14[0] = nsq.s_diff_shp_q14;
        }

        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        // Make dither dependent on quantized signal
        nsq.rand_seed = nsq.rand_seed.wrapping_add(pulses[i] as i32);
    }

    // Update LPC synth buffer: copy tail to head
    let copy_len = NSQ_LPC_BUF_LENGTH;
    let src_start = length;
    if src_start + copy_len <= nsq.s_lpc_q14.len() {
        for j in 0..copy_len {
            nsq.s_lpc_q14[j] = nsq.s_lpc_q14[src_start + j];
        }
    }
}

/// Main NSQ entry point.
///
/// Port of silk_NSQ_c from silk/NSQ.c.
/// Performs noise-shaped quantization of the input signal, producing
/// quantized pulse signal and reconstructed signal.
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq(
    nsq: &mut NsqState,
    indices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
    // Config
    frame_length: i32,
    subfr_length: i32,
    ltp_mem_length: i32,
    lpc_order: i32,
    shaping_lpc_order: i32,
    nb_subfr: i32,
    signal_type: i32,
    quant_offset_type: i32,
    nlsf_interp_coef_q2: i32,
    // Scratch buffers (caller-provided, avoids per-frame heap allocation)
    scratch_s_ltp_q15: &mut [i32],
    scratch_s_ltp: &mut [i16],
    scratch_x_sc_q10: &mut [i32],
    scratch_xq_tmp: &mut [i16],
) {
    let frame_len = frame_length as usize;
    let subfr_len = subfr_length as usize;
    let ltp_mem_len = ltp_mem_length as usize;
    let lpc_ord = lpc_order as usize;
    let shaping_ord = shaping_lpc_order as usize;

    nsq.rand_seed = indices.seed as i32;

    // Set unvoiced lag to the previous one, overwrite later for voiced
    let mut lag = nsq.lag_prev;

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(signal_type >> 1) as usize]
        [quant_offset_type as usize] as i32;

    let lsf_interpolation_flag = if nlsf_interp_coef_q2 == 4 { 0 } else { 1 };

    let total_len = ltp_mem_len + frame_len;
    let s_ltp_q15 = &mut scratch_s_ltp_q15[..total_len];
    let s_ltp = &mut scratch_s_ltp[..total_len];
    let x_sc_q10 = &mut scratch_x_sc_q10[..subfr_len];
    for v in s_ltp_q15.iter_mut() {
        *v = 0;
    }
    for v in s_ltp.iter_mut() {
        *v = 0;
    }
    for v in x_sc_q10.iter_mut() {
        *v = 0;
    }

    // Set up pointers to start of sub frame
    nsq.s_ltp_shp_buf_idx = ltp_mem_length;
    nsq.s_ltp_buf_idx = ltp_mem_length;
    let mut pxq_offset = ltp_mem_len;
    let mut x16_offset = 0usize;
    let mut pulses_offset = 0usize;

    for k in 0..nb_subfr as usize {
        // Select A_Q12 coefficients: ((k >> 1) | (1 - LSF_interpolation_flag)) * MAX_LPC_ORDER
        let a_q12_offset = ((k >> 1) | (1 - lsf_interpolation_flag as usize)) * MAX_LPC_ORDER;
        let a_q12 = &pred_coef_q12[a_q12_offset..a_q12_offset + lpc_ord];
        let b_q14 = &ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER];
        let ar_shp_q13 = &ar_q13[k * MAX_SHAPE_LPC_ORDER..(k + 1) * MAX_SHAPE_LPC_ORDER];

        // Noise shape parameters
        let harm_gain = harm_shape_gain_q14[k];
        let mut harm_shape_fir_packed_q14 = harm_gain >> 2;
        harm_shape_fir_packed_q14 |= (harm_gain >> 1) << 16;

        nsq.rewhite_flag = 0;
        if signal_type == TYPE_VOICED {
            lag = pitch_l[k];

            // Re-whitening
            let rewhite_cond = k & (3 - (lsf_interpolation_flag as usize * 2));
            if rewhite_cond == 0 {
                let start_idx =
                    (ltp_mem_len as i32 - lag - lpc_order - LTP_ORDER as i32 / 2) as usize;

                // silk_LPC_analysis_filter
                nlsf::silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx..],
                    &nsq.xq[(start_idx + k * subfr_len)..],
                    a_q12,
                    ltp_mem_len - start_idx,
                    lpc_ord,
                );

                nsq.rewhite_flag = 1;
                nsq.s_ltp_buf_idx = ltp_mem_length;
            }
        }

        silk_nsq_scale_states(
            nsq,
            &x16[x16_offset..],
            x_sc_q10,
            s_ltp,
            s_ltp_q15,
            k,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            signal_type,
            subfr_len,
            ltp_mem_len,
        );

        // Use caller-provided scratch for xq output to avoid double-borrowing nsq
        let xq_tmp = &mut scratch_xq_tmp[..subfr_len];
        for v in xq_tmp.iter_mut() {
            *v = 0;
        }

        silk_noise_shape_quantizer(
            nsq,
            signal_type,
            x_sc_q10,
            &mut pulses[pulses_offset..],
            xq_tmp,
            s_ltp_q15,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_fir_packed_q14,
            tilt_q14[k],
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            subfr_len,
            shaping_ord,
            lpc_ord,
        );

        // Copy temporary xq output into nsq.xq at the right offset
        let xq_end = (pxq_offset + subfr_len).min(nsq.xq.len());
        let copy_len = xq_end - pxq_offset;
        nsq.xq[pxq_offset..pxq_offset + copy_len].copy_from_slice(&xq_tmp[..copy_len]);

        x16_offset += subfr_len;
        pulses_offset += subfr_len;
        pxq_offset += subfr_len;
    }

    // Update lagPrev for next frame
    nsq.lag_prev = pitch_l[nb_subfr as usize - 1];

    // Save quantized speech and noise shaping signals: shift buffers
    // memmove(xq, xq + frame_length, ltp_mem_length)
    nsq.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    nsq.s_ltp_shp_q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}
