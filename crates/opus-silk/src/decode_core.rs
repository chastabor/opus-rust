// Port of silk/decode_core.c - Core decoder (inverse NSQ: LTP + LPC)

use crate::*;
use crate::tables::*;

/// Core decoder. Performs inverse NSQ operation LTP + LPC
pub fn silk_decode_core(
    ps_dec: &mut ChannelState,
    ps_dec_ctrl: &DecoderControl,
    xq: &mut [i16],
    pulses: &[i16],
) {
    assert!(ps_dec.prev_gain_q16 != 0);

    let frame_length = ps_dec.frame_length as usize;
    let subfr_length = ps_dec.subfr_length as usize;
    let lpc_order = ps_dec.lpc_order as usize;
    let ltp_mem_length = ps_dec.ltp_mem_length as usize;

    let mut s_ltp = vec![0i16; ltp_mem_length];
    let mut s_ltp_q15 = vec![0i32; ltp_mem_length + frame_length];
    let mut res_q14 = vec![0i32; subfr_length];
    let mut s_lpc_q14 = vec![0i32; subfr_length + MAX_LPC_ORDER];

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10
        [(ps_dec.indices.signal_type >> 1) as usize]
        [ps_dec.indices.quant_offset_type as usize] as i32;

    let nlsf_interpolation_flag = (ps_dec.indices.nlsf_interp_coef_q2 as i32) < 4;

    // Decode excitation
    let mut rand_seed = ps_dec.indices.seed as i32;
    for i in 0..frame_length {
        rand_seed = silk_rand(rand_seed);
        ps_dec.exc_q14[i] = (pulses[i] as i32) << 14;
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

    // Loop over subframes
    for k in 0..ps_dec.nb_subfr as usize {
        let a_q12 = &ps_dec_ctrl.pred_coef_q12[k >> 1];
        let b_q14 = &ps_dec_ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER];
        let mut signal_type = ps_dec.indices.signal_type as i32;

        let gain_q10 = ps_dec_ctrl.gains_q16[k] >> 6;
        let inv_gain_q31 = silk_inverse32_varq(ps_dec_ctrl.gains_q16[k], 47);

        // Calculate gain adjustment factor
        let gain_adj_q16 = if ps_dec_ctrl.gains_q16[k] != ps_dec.prev_gain_q16 {
            let adj = silk_div32_varq(ps_dec.prev_gain_q16, ps_dec_ctrl.gains_q16[k], 16);
            for i in 0..MAX_LPC_ORDER {
                s_lpc_q14[i] = silk_smulww_correct(adj, s_lpc_q14[i]);
            }
            adj
        } else {
            1 << 16
        };

        ps_dec.prev_gain_q16 = ps_dec_ctrl.gains_q16[k];

        // Avoid abrupt transition from voiced PLC to unvoiced normal decoding
        let mut b_q14_local = [0i16; LTP_ORDER];
        b_q14_local.copy_from_slice(&b_q14[..LTP_ORDER]);

        if ps_dec.loss_cnt > 0 && ps_dec.prev_signal_type == TYPE_VOICED
            && ps_dec.indices.signal_type as i32 != TYPE_VOICED
            && k < MAX_NB_SUBFR / 2
        {
            b_q14_local.fill(0);
            b_q14_local[LTP_ORDER / 2] = (0.25f64 * 16384.0) as i16; // SILK_FIX_CONST(0.25, 14)
            signal_type = TYPE_VOICED;
        }

        let mut lag = 0i32;

        if signal_type == TYPE_VOICED {
            lag = ps_dec_ctrl.pitch_l[k];

            // Re-whitening
            if k == 0 || (k == 2 && nlsf_interpolation_flag) {
                let start_idx = ltp_mem_length as i32 - lag - ps_dec.lpc_order - (LTP_ORDER as i32) / 2;
                let start_idx = start_idx.max(0) as usize;

                if k == 2 {
                    let src_start = ltp_mem_length;
                    for i in 0..2 * subfr_length {
                        if src_start + i < ps_dec.out_buf.len() {
                            ps_dec.out_buf[src_start + i] = xq[i];
                        }
                    }
                }

                // LPC analysis filter
                nlsf::silk_lpc_analysis_filter(
                    &mut s_ltp,
                    &ps_dec.out_buf[k * subfr_length..],
                    a_q12,
                    ltp_mem_length - start_idx,
                    lpc_order,
                );

                let mut inv_gain_for_ltp = inv_gain_q31;
                if k == 0 {
                    inv_gain_for_ltp = (silk_smulwb(inv_gain_for_ltp, ps_dec_ctrl.ltp_scale_q14 as i32)) << 2;
                }

                let lag_plus = lag as usize + LTP_ORDER / 2;
                for i in 0..lag_plus.min(ltp_mem_length) {
                    let src_idx = ltp_mem_length - i - 1;
                    s_ltp_q15[s_ltp_buf_idx - i - 1] = silk_smulwb(inv_gain_for_ltp, s_ltp[src_idx] as i32);
                }
            } else {
                if gain_adj_q16 != (1 << 16) {
                    let lag_plus = lag as usize + LTP_ORDER / 2;
                    for i in 0..lag_plus.min(s_ltp_buf_idx) {
                        s_ltp_q15[s_ltp_buf_idx - i - 1] =
                            silk_smulww_correct(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx - i - 1]);
                    }
                }
            }
        }

        // Long-term prediction
        if signal_type == TYPE_VOICED {
            let pred_lag_base = s_ltp_buf_idx as i32 - lag + (LTP_ORDER as i32) / 2;
            for i in 0..subfr_length {
                let pred_idx = (pred_lag_base + i as i32) as usize;
                let mut ltp_pred_q13: i32 = 2; // bias

                // Check bounds before accessing
                if pred_idx >= 4 {
                    ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_idx], b_q14_local[0] as i32);
                    ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_idx - 1], b_q14_local[1] as i32);
                    ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_idx - 2], b_q14_local[2] as i32);
                    ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_idx - 3], b_q14_local[3] as i32);
                    ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_idx - 4], b_q14_local[4] as i32);
                }

                res_q14[i] = ps_dec.exc_q14[pexc_offset + i].wrapping_add(ltp_pred_q13 << 1);
                s_ltp_q15[s_ltp_buf_idx] = res_q14[i] << 1;
                s_ltp_buf_idx += 1;
            }
        }

        // Short-term prediction (LPC synthesis)
        let use_res = signal_type == TYPE_VOICED;
        for i in 0..subfr_length {
            let mut lpc_pred_q10: i32 = lpc_order as i32 >> 1; // bias

            for j in 0..lpc_order.min(10) {
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 1 - j],
                    a_q12[j] as i32,
                );
            }
            if lpc_order == 16 {
                for j in 10..16 {
                    lpc_pred_q10 = silk_smlawb(
                        lpc_pred_q10,
                        s_lpc_q14[MAX_LPC_ORDER + i - 1 - j],
                        a_q12[j] as i32,
                    );
                }
            }

            let exc = if use_res { res_q14[i] } else { ps_dec.exc_q14[pexc_offset + i] };
            s_lpc_q14[MAX_LPC_ORDER + i] = silk_add_sat32(exc, silk_lshift_sat32(lpc_pred_q10, 4));

            xq[pxq_offset + i] = silk_sat16(silk_rshift_round(
                silk_smulww_correct(s_lpc_q14[MAX_LPC_ORDER + i], gain_q10), 8
            ));
        }

        // Update LPC filter state
        s_lpc_q14.copy_within(subfr_length..subfr_length + MAX_LPC_ORDER, 0);
        pexc_offset += subfr_length;
        pxq_offset += subfr_length;
    }

    // Save LPC state
    ps_dec.s_lpc_q14_buf.copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
}
