// Port of silk/PLC.c - Packet Loss Concealment

use crate::*;

const BWE_COEF_Q16: i32 = 64881; // 0.99 in Q16
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469;
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565;
const RAND_BUF_SIZE: usize = 128;
const RAND_BUF_MASK: usize = RAND_BUF_SIZE - 1;

const HARM_ATT_Q15: [i16; 2] = [32440, 31130];
const PLC_RAND_ATTENUATE_V_Q15: [i16; 2] = [31130, 26214];
const PLC_RAND_ATTENUATE_UV_Q15: [i16; 2] = [32440, 29491];

/// Reset PLC state
pub fn plc_reset(ps_dec: &mut ChannelState) {
    ps_dec.s_plc.pitch_l_q8 = (ps_dec.frame_length << 8) >> 1;
    ps_dec.s_plc.prev_gain_q16 = [1 << 16, 1 << 16];
    ps_dec.s_plc.subfr_length = 20;
    ps_dec.s_plc.nb_subfr = 2;
}

/// PLC main entry
pub fn silk_plc(
    ps_dec: &mut ChannelState,
    ps_dec_ctrl: &mut DecoderControl,
    frame: &mut [i16],
    lost: bool,
) {
    if ps_dec.fs_khz != ps_dec.s_plc.fs_khz {
        plc_reset(ps_dec);
        ps_dec.s_plc.fs_khz = ps_dec.fs_khz;
    }

    if lost {
        silk_plc_conceal(ps_dec, ps_dec_ctrl, frame);
        ps_dec.loss_cnt += 1;
    } else {
        silk_plc_update(ps_dec, ps_dec_ctrl);
    }
}

/// Update PLC state from a good frame
fn silk_plc_update(ps_dec: &mut ChannelState, ps_dec_ctrl: &DecoderControl) {
    ps_dec.prev_signal_type = ps_dec.indices.signal_type as i32;
    let mut ltp_gain_q14 = 0i32;

    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        let nb_subfr = ps_dec.nb_subfr as usize;
        let subfr_len = ps_dec.subfr_length as usize;
        let last_pitch = ps_dec_ctrl.pitch_l[nb_subfr - 1];

        let mut j = 0;
        while j * subfr_len < last_pitch as usize {
            if j >= nb_subfr { break; }
            let mut temp = 0i32;
            for i in 0..LTP_ORDER {
                temp += ps_dec_ctrl.ltp_coef_q14[(nb_subfr - 1 - j) * LTP_ORDER + i] as i32;
            }
            if temp > ltp_gain_q14 {
                ltp_gain_q14 = temp;
                ps_dec.s_plc.ltp_coef_q14.copy_from_slice(
                    &ps_dec_ctrl.ltp_coef_q14[(nb_subfr - 1 - j) * LTP_ORDER..(nb_subfr - j) * LTP_ORDER]
                );
                ps_dec.s_plc.pitch_l_q8 = ps_dec_ctrl.pitch_l[nb_subfr - 1 - j] << 8;
            }
            j += 1;
        }

        // Concentrate LTP gain to center tap
        ps_dec.s_plc.ltp_coef_q14.fill(0);
        ps_dec.s_plc.ltp_coef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;

        // Limit LTP coefs
        if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
            let scale_q10 = silk_div32((V_PITCH_GAIN_START_MIN_Q14) << 10, ltp_gain_q14.max(1));
            for i in 0..LTP_ORDER {
                ps_dec.s_plc.ltp_coef_q14[i] =
                    ((ps_dec.s_plc.ltp_coef_q14[i] as i32 * scale_q10) >> 10) as i16;
            }
        } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
            let scale_q14 = silk_div32(V_PITCH_GAIN_START_MAX_Q14 << 14, ltp_gain_q14.max(1));
            for i in 0..LTP_ORDER {
                ps_dec.s_plc.ltp_coef_q14[i] =
                    ((ps_dec.s_plc.ltp_coef_q14[i] as i32 * scale_q14) >> 14) as i16;
            }
        }
    } else {
        ps_dec.s_plc.pitch_l_q8 = (ps_dec.fs_khz * 18) << 8;
        ps_dec.s_plc.ltp_coef_q14.fill(0);
    }

    // Save LPC coefficients and gains
    let lpc_order = ps_dec.lpc_order as usize;
    ps_dec.s_plc.prev_lpc_q12[..lpc_order]
        .copy_from_slice(&ps_dec_ctrl.pred_coef_q12[1][..lpc_order]);
    ps_dec.s_plc.prev_ltp_scale_q14 = ps_dec_ctrl.ltp_scale_q14 as i16;

    let nb = ps_dec.nb_subfr as usize;
    ps_dec.s_plc.prev_gain_q16[0] = ps_dec_ctrl.gains_q16[nb - 2];
    ps_dec.s_plc.prev_gain_q16[1] = ps_dec_ctrl.gains_q16[nb - 1];

    ps_dec.s_plc.subfr_length = ps_dec.subfr_length;
    ps_dec.s_plc.nb_subfr = ps_dec.nb_subfr;
}

/// Conceal a lost frame
fn silk_plc_conceal(
    ps_dec: &mut ChannelState,
    ps_dec_ctrl: &mut DecoderControl,
    frame: &mut [i16],
) {
    let frame_length = ps_dec.frame_length as usize;
    let _subfr_length = ps_dec.subfr_length as usize;
    let lpc_order = ps_dec.lpc_order as usize;
    let ltp_mem_length = ps_dec.ltp_mem_length as usize;
    let _nb_subfr = ps_dec.nb_subfr;

    let prev_gain_q10 = [
        ps_dec.s_plc.prev_gain_q16[0] >> 6,
        ps_dec.s_plc.prev_gain_q16[1] >> 6,
    ];

    if ps_dec.first_frame_after_reset {
        ps_dec.s_plc.prev_lpc_q12.fill(0);
    }

    // Set up attenuation gains
    let att_idx = (ps_dec.loss_cnt as usize).min(1);
    let _harm_gain_q15 = HARM_ATT_Q15[att_idx] as i32;
    let _rand_gain_q15 = if ps_dec.prev_signal_type == TYPE_VOICED {
        PLC_RAND_ATTENUATE_V_Q15[att_idx] as i32
    } else {
        PLC_RAND_ATTENUATE_UV_Q15[att_idx] as i32
    };

    // BWE of LPC coefficients
    nlsf::silk_bwexpander(&mut ps_dec.s_plc.prev_lpc_q12, lpc_order, BWE_COEF_Q16);
    let a_q12: Vec<i16> = ps_dec.s_plc.prev_lpc_q12[..lpc_order].to_vec();

    // Compute rand_scale_q14
    let mut rand_scale_q14 = ps_dec.s_plc.rand_scale_q14;
    if ps_dec.loss_cnt == 0 {
        rand_scale_q14 = 1 << 14;
        if ps_dec.prev_signal_type == TYPE_VOICED {
            let mut sum = 0i16;
            for i in 0..LTP_ORDER {
                sum = sum.wrapping_add(ps_dec.s_plc.ltp_coef_q14[i]);
            }
            rand_scale_q14 = (1i16 << 14).wrapping_sub(sum);
            rand_scale_q14 = rand_scale_q14.max(3277);
            rand_scale_q14 = ((rand_scale_q14 as i32 * ps_dec.s_plc.prev_ltp_scale_q14 as i32) >> 14) as i16;
        }
    }

    let mut rand_seed = ps_dec.s_plc.rand_seed;
    let lag = silk_rshift_round(ps_dec.s_plc.pitch_l_q8, 8);

    // Simplified PLC: generate output using LPC synthesis
    let mut s_lpc_q14 = vec![0i32; ltp_mem_length + frame_length + MAX_LPC_ORDER];
    let lpc_base = ltp_mem_length - MAX_LPC_ORDER;
    s_lpc_q14[lpc_base..lpc_base + MAX_LPC_ORDER]
        .copy_from_slice(&ps_dec.s_lpc_q14_buf);

    // Simple concealment: attenuated random noise through LPC
    for i in 0..frame_length {
        rand_seed = silk_rand(rand_seed);

        // Simple noise generation
        let exc_q14 = if ps_dec.prev_signal_type == TYPE_VOICED {
            // For voiced: use pitch-period repetition (simplified)
            let idx = (rand_seed >> 25) as usize & RAND_BUF_MASK;
            let exc_idx = idx.min(frame_length.saturating_sub(1));
            silk_smulwb(ps_dec.exc_q14[exc_idx], rand_scale_q14 as i32) << 2
        } else {
            silk_smulwb(ps_dec.exc_q14[i % frame_length], rand_scale_q14 as i32) << 2
        };

        // LPC synthesis
        let mut lpc_pred_q10 = (lpc_order as i32) >> 1;
        for j in 0..lpc_order {
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[lpc_base + MAX_LPC_ORDER + i - 1 - j],
                a_q12[j] as i32,
            );
        }

        s_lpc_q14[lpc_base + MAX_LPC_ORDER + i] = silk_add_sat32(
            exc_q14,
            silk_lshift_sat32(lpc_pred_q10, 4),
        );

        frame[i] = silk_sat16(silk_rshift_round(
            silk_smulww_correct(s_lpc_q14[lpc_base + MAX_LPC_ORDER + i], prev_gain_q10[1]),
            8,
        ));
    }

    // Save LPC state
    ps_dec.s_lpc_q14_buf.copy_from_slice(
        &s_lpc_q14[lpc_base + frame_length..lpc_base + frame_length + MAX_LPC_ORDER]
    );

    // Update PLC state
    ps_dec.s_plc.rand_seed = rand_seed;
    ps_dec.s_plc.rand_scale_q14 = rand_scale_q14;
    for i in 0..MAX_NB_SUBFR {
        ps_dec_ctrl.pitch_l[i] = lag;
    }
}

/// Glue concealed frames with new good frames
pub fn silk_plc_glue_frames(
    ps_dec: &mut ChannelState,
    frame: &mut [i16],
    length: usize,
) {
    if ps_dec.loss_cnt > 0 {
        let mut energy = 0i32;
        let mut shift = 0i32;
        silk_sum_sqr_shift(&mut energy, &mut shift, &frame[..length], length);
        ps_dec.s_plc.conc_energy = energy;
        ps_dec.s_plc.conc_energy_shift = shift;
        ps_dec.s_plc.last_frame_lost = true;
    } else {
        if ps_dec.s_plc.last_frame_lost {
            let mut energy = 0i32;
            let mut energy_shift = 0i32;
            silk_sum_sqr_shift(&mut energy, &mut energy_shift, &frame[..length], length);

            let mut conc_energy = ps_dec.s_plc.conc_energy;
            let conc_shift = ps_dec.s_plc.conc_energy_shift;

            if energy_shift > conc_shift {
                conc_energy >>= energy_shift - conc_shift;
            } else if energy_shift < conc_shift {
                energy >>= conc_shift - energy_shift;
            }

            if energy > conc_energy {
                let lz = silk_clz32(conc_energy) - 1;
                let lz = lz.max(0);
                conc_energy <<= lz;
                energy >>= (24 - lz).max(0);

                let frac_q24 = silk_div32(conc_energy, energy.max(1));
                let mut gain_q16 = silk_sqrt_approx(frac_q24) << 4;
                let slope_q16 = silk_div32((1 << 16) - gain_q16, length as i32).max(0) << 2;

                for i in 0..length {
                    frame[i] = silk_smulwb(gain_q16, frame[i] as i32) as i16;
                    gain_q16 += slope_q16;
                    if gain_q16 > (1 << 16) {
                        break;
                    }
                }
            }
        }
        ps_dec.s_plc.last_frame_lost = false;
    }
}
