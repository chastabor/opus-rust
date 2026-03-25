// Port of silk/CNG.c - Comfort Noise Generation

use crate::*;

/// Reset CNG state
pub fn cng_reset(ps_dec: &mut ChannelState) {
    let lpc_order = ps_dec.lpc_order.max(1) as usize;
    let step_q15 = 32767 / (lpc_order as i32 + 1);
    let mut acc_q15 = 0i32;
    for i in 0..lpc_order {
        acc_q15 += step_q15;
        ps_dec.s_cng.cng_smth_nlsf_q15[i] = acc_q15 as i16;
    }
    ps_dec.s_cng.cng_smth_gain_q16 = 0;
    ps_dec.s_cng.rand_seed = 3176576;
}

/// CNG excitation generator
fn silk_cng_exc(exc_q14: &mut [i32], exc_buf_q14: &[i32], length: usize, rand_seed: &mut i32) {
    let mut exc_mask = CNG_BUF_MASK_MAX;
    while exc_mask > length {
        exc_mask >>= 1;
    }

    let mut seed = *rand_seed;
    for item in exc_q14.iter_mut().take(length) {
        seed = silk_rand(seed);
        let idx = ((seed >> 24) as usize) & exc_mask;
        if idx < exc_buf_q14.len() {
            *item = exc_buf_q14[idx];
        }
    }
    *rand_seed = seed;
}

/// Updates CNG estimate, and applies CNG when packet was lost
pub fn silk_cng(
    ps_dec: &mut ChannelState,
    ps_dec_ctrl: &DecoderControl,
    frame: &mut [i16],
    length: usize,
) {
    let lpc_order = ps_dec.lpc_order as usize;

    if ps_dec.fs_khz != ps_dec.s_cng.fs_khz {
        cng_reset(ps_dec);
        ps_dec.s_cng.fs_khz = ps_dec.fs_khz;
    }

    if ps_dec.loss_cnt == 0 && ps_dec.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
        // Smooth LSFs
        for i in 0..lpc_order {
            ps_dec.s_cng.cng_smth_nlsf_q15[i] = (ps_dec.s_cng.cng_smth_nlsf_q15[i] as i32
                + silk_smulwb(
                    ps_dec.prev_nlsf_q15[i] as i32 - ps_dec.s_cng.cng_smth_nlsf_q15[i] as i32,
                    CNG_NLSF_SMTH_Q16,
                )) as i16;
        }

        // Find subframe with highest gain
        let mut max_gain_q16 = 0i32;
        let mut subfr = 0usize;
        for i in 0..ps_dec.nb_subfr as usize {
            if ps_dec_ctrl.gains_q16[i] > max_gain_q16 {
                max_gain_q16 = ps_dec_ctrl.gains_q16[i];
                subfr = i;
            }
        }

        // Update CNG excitation buffer
        let subfr_len = ps_dec.subfr_length as usize;
        let nb = ps_dec.nb_subfr as usize;

        // Move old data forward
        let move_len = (nb - 1) * subfr_len;
        let buf_len = ps_dec.s_cng.cng_exc_buf_q14.len();
        if subfr_len + move_len <= buf_len {
            ps_dec
                .s_cng
                .cng_exc_buf_q14
                .copy_within(0..move_len, subfr_len);
        }
        // Copy new subframe excitation
        let src_offset = subfr * subfr_len;
        for i in 0..subfr_len.min(buf_len) {
            if src_offset + i < ps_dec.exc_q14.len() {
                ps_dec.s_cng.cng_exc_buf_q14[i] = ps_dec.exc_q14[src_offset + i];
            }
        }

        // Smooth gains
        for i in 0..nb {
            ps_dec.s_cng.cng_smth_gain_q16 += silk_smulwb(
                ps_dec_ctrl.gains_q16[i] - ps_dec.s_cng.cng_smth_gain_q16,
                CNG_GAIN_SMTH_Q16,
            );
            if silk_smulww_correct(ps_dec.s_cng.cng_smth_gain_q16, CNG_GAIN_SMTH_THRESHOLD_Q16)
                > ps_dec_ctrl.gains_q16[i]
            {
                ps_dec.s_cng.cng_smth_gain_q16 = ps_dec_ctrl.gains_q16[i];
            }
        }
    }

    // Add CNG when packet is lost
    if ps_dec.loss_cnt > 0 {
        let mut cng_sig_q14 = vec![0i32; length + MAX_LPC_ORDER];

        // Generate CNG excitation
        let mut gain_q16 = silk_smulww_correct(
            ps_dec.s_plc.rand_scale_q14 as i32,
            ps_dec.s_plc.prev_gain_q16[1],
        );
        gain_q16 = silk_smulww_correct(gain_q16, gain_q16);
        let smth_sq = silk_smulww_correct(
            ps_dec.s_cng.cng_smth_gain_q16,
            ps_dec.s_cng.cng_smth_gain_q16,
        );
        gain_q16 = silk_sub_lshift32(smth_sq, gain_q16, 5).max(0);
        let gain_q16 = silk_sqrt_approx(gain_q16) << 8;
        let gain_q10 = gain_q16 >> 6;

        silk_cng_exc(
            &mut cng_sig_q14[MAX_LPC_ORDER..],
            &ps_dec.s_cng.cng_exc_buf_q14,
            length,
            &mut ps_dec.s_cng.rand_seed,
        );

        // Convert CNG NLSF to filter
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        nlsf::silk_nlsf2a(&mut a_q12, &ps_dec.s_cng.cng_smth_nlsf_q15, lpc_order);

        // CNG synthesis
        cng_sig_q14[..MAX_LPC_ORDER].copy_from_slice(&ps_dec.s_cng.cng_synth_state);

        for i in 0..length {
            let mut lpc_pred_q10 = (lpc_order as i32) >> 1;
            for j in 0..lpc_order.min(10) {
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    cng_sig_q14[MAX_LPC_ORDER + i - 1 - j],
                    a_q12[j] as i32,
                );
            }
            if lpc_order == 16 {
                for j in 10..16 {
                    lpc_pred_q10 = silk_smlawb(
                        lpc_pred_q10,
                        cng_sig_q14[MAX_LPC_ORDER + i - 1 - j],
                        a_q12[j] as i32,
                    );
                }
            }

            cng_sig_q14[MAX_LPC_ORDER + i] = silk_add_sat32(
                cng_sig_q14[MAX_LPC_ORDER + i],
                silk_lshift_sat32(lpc_pred_q10, 4),
            );

            let cng_sample = silk_sat16(silk_rshift_round(
                silk_smulww_correct(cng_sig_q14[MAX_LPC_ORDER + i], gain_q10),
                8,
            ));
            frame[i] = (frame[i] as i32)
                .saturating_add(cng_sample as i32)
                .clamp(-32768, 32767) as i16;
        }

        ps_dec
            .s_cng
            .cng_synth_state
            .copy_from_slice(&cng_sig_q14[length..length + MAX_LPC_ORDER]);
    } else {
        ps_dec.s_cng.cng_synth_state[..lpc_order].fill(0);
    }
}
