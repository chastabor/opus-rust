// Port of silk/decode_core.c - Core decoder (inverse NSQ: LTP + LPC)

use crate::tables::*;
use crate::*;

/// Core decoder. Performs inverse NSQ operation LTP + LPC
pub fn silk_decode_core(
    ps_dec: &mut ChannelState,
    ps_dec_ctrl: &DecoderControl,
    xq: &mut [i16],
    pulses: &[i16],
) {
    // Guard against division by zero from malformed bitstream (C reference uses silk_assert)
    if ps_dec.prev_gain_q16 == 0 {
        ps_dec.prev_gain_q16 = 1;
    }

    let frame_length = ps_dec.frame_length as usize;
    let subfr_length = ps_dec.subfr_length as usize;
    let lpc_order = ps_dec.lpc_order as usize;
    let ltp_mem_length = ps_dec.ltp_mem_length as usize;

    let mut s_ltp = [0i16; MAX_LTP_MEM_LENGTH];
    let mut s_ltp_q15 = [0i32; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH];
    let mut res_q14 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut s_lpc_q14 = [0i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_dec.indices.signal_type >> 1) as usize]
        [ps_dec.indices.quant_offset_type as usize] as i32;

    let nlsf_interpolation_flag = (ps_dec.indices.nlsf_interp_coef_q2 as i32) < 4;

    // Decode excitation
    let mut rand_seed = ps_dec.indices.seed as i32;
    for (i, &pulse) in pulses.iter().enumerate().take(frame_length) {
        rand_seed = silk_rand(rand_seed);
        ps_dec.exc_q14[i] = (pulse as i32) << 14;
        if ps_dec.exc_q14[i] > 0 {
            ps_dec.exc_q14[i] -= QUANT_LEVEL_ADJUST_Q10 << 4;
        } else if ps_dec.exc_q14[i] < 0 {
            ps_dec.exc_q14[i] += QUANT_LEVEL_ADJUST_Q10 << 4;
        }
        ps_dec.exc_q14[i] += offset_q10 << 4;
        if rand_seed < 0 {
            ps_dec.exc_q14[i] = -ps_dec.exc_q14[i];
        }
        rand_seed = rand_seed.wrapping_add(pulses[i] as i32);
    }

    // Copy LPC state
    s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&ps_dec.s_lpc_q14_buf);

    let mut pexc_offset = 0usize;
    let mut pxq_offset = 0usize;
    let mut s_ltp_buf_idx = ltp_mem_length;

    // Local copies for pitch parameters that may be modified during PLC transition
    let mut pitch_l = ps_dec_ctrl.pitch_l;
    let mut ltp_coef_q14 = [0i16; LTP_ORDER * MAX_NB_SUBFR];
    ltp_coef_q14[..LTP_ORDER * ps_dec.nb_subfr as usize]
        .copy_from_slice(&ps_dec_ctrl.ltp_coef_q14[..LTP_ORDER * ps_dec.nb_subfr as usize]);

    // Loop over subframes
    let mut lag = 0i32;
    for (k, pitch_l_k) in pitch_l
        .iter_mut()
        .enumerate()
        .take(ps_dec.nb_subfr as usize)
    {
        let a_q12 = &ps_dec_ctrl.pred_coef_q12[k >> 1];
        let mut a_q12_tmp = [0i16; MAX_LPC_ORDER];
        a_q12_tmp[..lpc_order].copy_from_slice(&a_q12[..lpc_order]);

        let b_q14_base = k * LTP_ORDER;
        let mut signal_type = ps_dec.indices.signal_type as i32;

        let gain_q10 = ps_dec_ctrl.gains_q16[k] >> 6;
        let inv_gain_q31 = silk_inverse32_varq(ps_dec_ctrl.gains_q16[k], 47);

        // Calculate gain adjustment factor
        let gain_adj_q16 = if ps_dec_ctrl.gains_q16[k] != ps_dec.prev_gain_q16 {
            let adj = silk_div32_varq(ps_dec.prev_gain_q16, ps_dec_ctrl.gains_q16[k], 16);
            for item in s_lpc_q14.iter_mut().take(MAX_LPC_ORDER) {
                *item = silk_smulww_correct(adj, *item);
            }
            adj
        } else {
            1 << 16
        };

        ps_dec.prev_gain_q16 = ps_dec_ctrl.gains_q16[k];

        // Avoid abrupt transition from voiced PLC to unvoiced normal decoding
        if ps_dec.loss_cnt > 0
            && ps_dec.prev_signal_type == TYPE_VOICED
            && ps_dec.indices.signal_type as i32 != TYPE_VOICED
            && k < MAX_NB_SUBFR / 2
        {
            for i in 0..LTP_ORDER {
                ltp_coef_q14[b_q14_base + i] = 0;
            }
            ltp_coef_q14[b_q14_base + LTP_ORDER / 2] = (0.25f64 * 16384.0) as i16;
            signal_type = TYPE_VOICED;
            *pitch_l_k = ps_dec.lag_prev;
        }

        if signal_type == TYPE_VOICED {
            lag = *pitch_l_k;

            // Re-whitening
            if k == 0 || (k == 2 && nlsf_interpolation_flag) {
                let start_idx = (ltp_mem_length as i32
                    - lag
                    - ps_dec.lpc_order
                    - (LTP_ORDER as i32) / 2)
                    .max(0) as usize;

                if k == 2 {
                    // Copy decoded samples so far into outBuf for rewhitening
                    let dst_start = ltp_mem_length;
                    for (i, &xq_val) in xq.iter().enumerate().take(2 * subfr_length) {
                        if dst_start + i < ps_dec.out_buf.len() {
                            ps_dec.out_buf[dst_start + i] = xq_val;
                        }
                    }
                }

                // LPC analysis filter
                // C: silk_LPC_analysis_filter(&sLTP[start_idx], &psDec->outBuf[start_idx + k * subfr_length], ...)
                // The C function writes to sLTP starting at start_idx, reading from outBuf at the same-offset range.
                // Our Rust function writes to out[d..d+len]. We need it to write to s_ltp[start_idx..].
                let filter_len = ltp_mem_length - start_idx;
                let out_buf_offset = start_idx + k * subfr_length;
                silk_lpc_analysis_filter_offset(
                    &mut s_ltp,
                    start_idx,
                    &ps_dec.out_buf,
                    out_buf_offset,
                    &a_q12_tmp,
                    filter_len,
                    lpc_order,
                );

                let mut inv_gain_for_ltp = inv_gain_q31;
                if k == 0 {
                    // Do LTP downscaling to reduce inter-packet dependency
                    inv_gain_for_ltp =
                        silk_smulwb(inv_gain_for_ltp, ps_dec_ctrl.ltp_scale_q14) << 2;
                }

                let lag_plus = lag as usize + LTP_ORDER / 2;
                for i in 0..lag_plus.min(ltp_mem_length).min(s_ltp_buf_idx) {
                    let src_idx = ltp_mem_length - i - 1;
                    s_ltp_q15[s_ltp_buf_idx - i - 1] =
                        silk_smulwb(inv_gain_for_ltp, s_ltp[src_idx] as i32);
                }
            } else if gain_adj_q16 != (1 << 16) {
                let lag_plus = lag as usize + LTP_ORDER / 2;
                for i in 0..lag_plus.min(s_ltp_buf_idx) {
                    s_ltp_q15[s_ltp_buf_idx - i - 1] =
                        silk_smulww_correct(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx - i - 1]);
                }
            }
        }

        // Long-term prediction
        if signal_type == TYPE_VOICED {
            let pred_lag_base = s_ltp_buf_idx as i32 - lag + (LTP_ORDER as i32) / 2;
            for (i, res_q14_i) in res_q14.iter_mut().enumerate().take(subfr_length) {
                let pred_idx = (pred_lag_base + i as i32) as usize;

                let mut ltp_pred_q13: i32 = 2; // bias
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_idx],
                    ltp_coef_q14[b_q14_base] as i32,
                );
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_idx - 1],
                    ltp_coef_q14[b_q14_base + 1] as i32,
                );
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_idx - 2],
                    ltp_coef_q14[b_q14_base + 2] as i32,
                );
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_idx - 3],
                    ltp_coef_q14[b_q14_base + 3] as i32,
                );
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_idx - 4],
                    ltp_coef_q14[b_q14_base + 4] as i32,
                );

                *res_q14_i = ps_dec.exc_q14[pexc_offset + i].wrapping_add(ltp_pred_q13 << 1);
                s_ltp_q15[s_ltp_buf_idx] = *res_q14_i << 1;
                s_ltp_buf_idx += 1;
            }
        }

        // Short-term prediction (LPC synthesis)
        let use_res = signal_type == TYPE_VOICED;
        for i in 0..subfr_length {
            // Bias: silk_RSHIFT(psDec->LPC_order, 1)
            let mut lpc_pred_q10: i32 = lpc_order as i32 >> 1;

            for j in 0..lpc_order.min(10) {
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 1 - j],
                    a_q12_tmp[j] as i32,
                );
            }
            if lpc_order == 16 {
                for j in 10..16 {
                    lpc_pred_q10 = silk_smlawb(
                        lpc_pred_q10,
                        s_lpc_q14[MAX_LPC_ORDER + i - 1 - j],
                        a_q12_tmp[j] as i32,
                    );
                }
            }

            let exc = if use_res {
                res_q14[i]
            } else {
                ps_dec.exc_q14[pexc_offset + i]
            };
            s_lpc_q14[MAX_LPC_ORDER + i] = silk_add_sat32(exc, silk_lshift_sat32(lpc_pred_q10, 4));

            xq[pxq_offset + i] = silk_sat16(silk_rshift_round(
                silk_smulww_correct(s_lpc_q14[MAX_LPC_ORDER + i], gain_q10),
                8,
            ));
        }

        // Update LPC filter state
        s_lpc_q14.copy_within(subfr_length..subfr_length + MAX_LPC_ORDER, 0);
        pexc_offset += subfr_length;
        pxq_offset += subfr_length;
    }

    // Save LPC state
    ps_dec
        .s_lpc_q14_buf
        .copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
}

/// LPC analysis filter with explicit offsets matching C behavior
/// C: silk_LPC_analysis_filter(&sLTP[start_idx], &psDec->outBuf[start_idx + k*subfr_length], A_Q12, len, order)
/// Writes to out[out_offset + d .. out_offset + d + len]
/// Reads from input[in_offset .. in_offset + len]  (needs in_offset - d .. in_offset + len accessible)
fn silk_lpc_analysis_filter_offset(
    out: &mut [i16],
    out_offset: usize,
    input: &[i16],
    in_offset: usize,
    b_q12: &[i16],
    len: usize,
    d: usize,
) {
    for ix in d..len {
        let in_ix = in_offset + ix;
        let mut out32_q12: i32 = if in_ix < input.len() {
            (input[in_ix] as i32) << 12
        } else {
            0
        };
        for (j, &b_q12_j) in b_q12.iter().enumerate().take(d) {
            let in_j = in_offset + ix - j - 1;
            if in_j < input.len() {
                out32_q12 -= (b_q12_j as i32) * (input[in_j] as i32);
            }
        }
        let out_ix = out_offset + ix;
        if out_ix < out.len() {
            out[out_ix] = silk_sat16(silk_rshift_round(out32_q12, 12));
        }
    }
}
