// Port of silk/stereo_decode_pred.c, silk/stereo_MS_to_LR.c

use crate::tables::*;
use crate::*;
use opus_range_coder::EcCtx;

/// Decode mid/side predictors
pub fn silk_stereo_decode_pred(ps_range_dec: &mut EcCtx, pred_q13: &mut [i32; 2]) {
    let n = ps_range_dec.dec_icdf(&SILK_STEREO_PRED_JOINT_ICDF, 8) as i32;
    let ix0_2 = n / 5;
    let ix1_2 = n - 5 * ix0_2;

    let mut ix = [[0i32; 3]; 2];
    ix[0][2] = ix0_2;
    ix[1][2] = ix1_2;

    for ix_ch in &mut ix {
        ix_ch[0] = ps_range_dec.dec_icdf(&SILK_UNIFORM3_ICDF, 8) as i32;
        ix_ch[1] = ps_range_dec.dec_icdf(&SILK_UNIFORM5_ICDF, 8) as i32;
    }

    // Dequantize
    // SILK_FIX_CONST(0.5 / STEREO_QUANT_SUB_STEPS, 16) = 0.1 * 65536 = 6554
    const STEP_SCALE_Q16: i32 = 6554;

    for ch in 0..2 {
        ix[ch][0] += 3 * ix[ch][2];
        let low_q13 = SILK_STEREO_PRED_QUANT_Q13[ix[ch][0] as usize] as i32;
        let next_q13 = SILK_STEREO_PRED_QUANT_Q13[(ix[ch][0] + 1) as usize] as i32;
        let step_q13 = silk_smulwb(next_q13 - low_q13, STEP_SCALE_Q16);
        pred_q13[ch] = silk_smlabb(low_q13, step_q13, 2 * ix[ch][1] + 1);
    }

    // Subtract second from first predictor
    pred_q13[0] -= pred_q13[1];
}

/// Decode mid-only flag
pub fn silk_stereo_decode_mid_only(ps_range_dec: &mut EcCtx) -> i32 {
    ps_range_dec.dec_icdf(&SILK_STEREO_ONLY_CODE_MID_ICDF, 8) as i32
}

/// Convert adaptive Mid/Side representation to Left/Right stereo signal
pub fn silk_stereo_ms_to_lr(
    state: &mut StereoDecState,
    x1: &mut [i16], // mid signal (with 2-sample buffer at start)
    x2: &mut [i16], // side signal (with 2-sample buffer at start)
    pred_q13: &[i32; 2],
    fs_khz: i32,
    frame_length: usize,
) {
    // Buffering
    x1[0] = state.s_mid[0];
    x1[1] = state.s_mid[1];
    x2[0] = state.s_side[0];
    x2[1] = state.s_side[1];
    state.s_mid[0] = x1[frame_length];
    state.s_mid[1] = x1[frame_length + 1];
    state.s_side[0] = x2[frame_length];
    state.s_side[1] = x2[frame_length + 1];

    // Interpolate predictors and add prediction to side channel
    let mut pred0_q13 = state.pred_prev_q13[0] as i32;
    let mut pred1_q13 = state.pred_prev_q13[1] as i32;
    let interp_len = (STEREO_INTERP_LEN_MS * fs_khz) as usize;
    let denom_q16 = (1 << 16) / (interp_len as i32).max(1);
    let delta0_q13 = silk_rshift_round(
        silk_smulbb(pred_q13[0] - state.pred_prev_q13[0] as i32, denom_q16),
        16,
    );
    let delta1_q13 = silk_rshift_round(
        silk_smulbb(pred_q13[1] - state.pred_prev_q13[1] as i32, denom_q16),
        16,
    );

    for n in 0..interp_len.min(frame_length) {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        let sum = (x1[n] as i32 + x1[n + 2] as i32 + ((x1[n + 1] as i32) << 1)) << 9;
        let mut out = silk_smlawb((x2[n + 1] as i32) << 8, sum, pred0_q13);
        out = silk_smlawb(out, (x1[n + 1] as i32) << 11, pred1_q13);
        x2[n + 1] = silk_sat16(silk_rshift_round(out, 8));
    }

    let pred0_q13_final = pred_q13[0];
    let pred1_q13_final = pred_q13[1];
    for n in interp_len..frame_length {
        let sum = (x1[n] as i32 + x1[n + 2] as i32 + ((x1[n + 1] as i32) << 1)) << 9;
        let mut out = silk_smlawb((x2[n + 1] as i32) << 8, sum, pred0_q13_final);
        out = silk_smlawb(out, (x1[n + 1] as i32) << 11, pred1_q13_final);
        x2[n + 1] = silk_sat16(silk_rshift_round(out, 8));
    }

    state.pred_prev_q13[0] = pred_q13[0] as i16;
    state.pred_prev_q13[1] = pred_q13[1] as i16;

    // Convert to left/right signals
    for n in 0..frame_length {
        let sum = x1[n + 1] as i32 + x2[n + 1] as i32;
        let diff = x1[n + 1] as i32 - x2[n + 1] as i32;
        x1[n + 1] = silk_sat16(sum);
        x2[n + 1] = silk_sat16(diff);
    }
}
