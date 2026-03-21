// Port of silk/NLSF_decode.c, silk/NLSF2A.c, silk/NLSF_stabilize.c,
// silk/LPC_inv_pred_gain.c, silk/bwexpander.c, silk/LPC_fit.c

use crate::*;
use crate::tables::*;

const QA: i32 = 16;
const SILK_MAX_ORDER_LPC: usize = 16;

/// NLSF vector decoder
pub fn silk_nlsf_decode(
    p_nlsf_q15: &mut [i16],
    nlsf_indices: &[i8],
    cb: &NlsfCbStruct,
) {
    let order = cb.order as usize;

    // Unpack entropy table indices and predictor
    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    nlsf_unpack(&mut ec_ix, &mut pred_q8, cb, nlsf_indices[0] as usize);

    // Predictive residual dequantizer
    let mut res_q10 = [0i16; MAX_LPC_ORDER];
    nlsf_residual_dequant(
        &mut res_q10,
        &nlsf_indices[1..],
        &pred_q8,
        cb.quant_step_size_q16 as i32,
        order,
    );

    // Apply inverse square-rooted weights and add to output
    let cb1_offset = nlsf_indices[0] as usize * order;
    for i in 0..order {
        let cb_element = cb.cb1_nlsf_q8[cb1_offset + i] as i32;
        let wght = cb.cb1_wght_q9[cb1_offset + i] as i32;
        let res = res_q10[i] as i32;

        // silk_ADD_LSHIFT32(silk_DIV32_16(silk_LSHIFT(res, 14), wght), cb_element, 7)
        let nlsf_q15_tmp = ((res << 14) / wght) + (cb_element << 7);
        p_nlsf_q15[i] = nlsf_q15_tmp.clamp(0, 32767) as i16;
    }

    // NLSF stabilization
    silk_nlsf_stabilize(p_nlsf_q15, cb.delta_min_q15, order);
}

fn nlsf_unpack(
    ec_ix: &mut [i16; MAX_LPC_ORDER],
    pred_q8: &mut [u8; MAX_LPC_ORDER],
    cb: &NlsfCbStruct,
    cb1_index: usize,
) {
    let order = cb.order as usize;
    let ec_sel_offset = cb1_index * order / 2;

    for i in (0..order).step_by(2) {
        let entry = cb.ec_sel[ec_sel_offset + i / 2];
        ec_ix[i] = (((entry >> 1) & 7) as i16) * (2 * NLSF_QUANT_MAX_AMPLITUDE as i16 + 1);
        pred_q8[i] = cb.pred_q8[i + ((entry & 1) as usize) * (order - 1)];
        ec_ix[i + 1] = (((entry >> 5) & 7) as i16) * (2 * NLSF_QUANT_MAX_AMPLITUDE as i16 + 1);
        pred_q8[i + 1] = cb.pred_q8[i + (((entry >> 4) & 1) as usize) * (order - 1) + 1];
    }
}

fn nlsf_residual_dequant(
    x_q10: &mut [i16],
    indices: &[i8],
    pred_coef_q8: &[u8],
    quant_step_size_q16: i32,
    order: usize,
) {
    let mut out_q10: i32 = 0;
    for i in (0..order).rev() {
        // C: pred_Q10 = silk_RSHIFT(silk_SMULBB(out_Q10, (opus_int16)pred_coef_Q8[i]), 8);
        // silk_SMULBB takes lower 16 bits of both operands
        let pred_q10 = ((out_q10 as i16 as i32) * (pred_coef_q8[i] as i32)) >> 8;
        out_q10 = (indices[i] as i32) << 10;
        if out_q10 > 0 {
            out_q10 -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if out_q10 < 0 {
            out_q10 += NLSF_QUANT_LEVEL_ADJ_Q10;
        }
        out_q10 = silk_smlawb(pred_q10, out_q10, quant_step_size_q16);
        x_q10[i] = out_q10 as i16;
    }
}

/// NLSF stabilizer
pub fn silk_nlsf_stabilize(nlsf_q15: &mut [i16], n_delta_min_q15: &[i16], l: usize) {
    const MAX_LOOPS: usize = 20;

    for _loop in 0..MAX_LOOPS {
        // Find smallest distance
        let mut min_diff_q15 = nlsf_q15[0] as i32 - n_delta_min_q15[0] as i32;
        let mut min_idx = 0usize;

        for i in 1..l {
            let diff = nlsf_q15[i] as i32 - (nlsf_q15[i - 1] as i32 + n_delta_min_q15[i] as i32);
            if diff < min_diff_q15 {
                min_diff_q15 = diff;
                min_idx = i;
            }
        }

        // Last element
        let diff = (1 << 15) - (nlsf_q15[l - 1] as i32 + n_delta_min_q15[l] as i32);
        if diff < min_diff_q15 {
            min_diff_q15 = diff;
            min_idx = l;
        }

        if min_diff_q15 >= 0 {
            return;
        }

        if min_idx == 0 {
            nlsf_q15[0] = n_delta_min_q15[0];
        } else if min_idx == l {
            nlsf_q15[l - 1] = ((1 << 15) - n_delta_min_q15[l] as i32) as i16;
        } else {
            let mut min_center_q15 = 0i32;
            for k in 0..min_idx {
                min_center_q15 += n_delta_min_q15[k] as i32;
            }
            min_center_q15 += (n_delta_min_q15[min_idx] as i32) >> 1;

            let mut max_center_q15 = 1i32 << 15;
            for k in ((min_idx + 1)..=l).rev() {
                max_center_q15 -= n_delta_min_q15[k] as i32;
            }
            max_center_q15 -= (n_delta_min_q15[min_idx] as i32) >> 1;

            let center_freq_q15 = silk_rshift_round(
                nlsf_q15[min_idx - 1] as i32 + nlsf_q15[min_idx] as i32, 1
            ).clamp(min_center_q15, max_center_q15) as i16;

            nlsf_q15[min_idx - 1] = center_freq_q15 - ((n_delta_min_q15[min_idx] as i16) >> 1);
            nlsf_q15[min_idx] = nlsf_q15[min_idx - 1] + n_delta_min_q15[min_idx];
        }
    }

    // Fallback: insertion sort and enforce distances
    // Simple insertion sort
    for i in 1..l {
        let mut j = i;
        while j > 0 && nlsf_q15[j] < nlsf_q15[j - 1] {
            nlsf_q15.swap(j, j - 1);
            j -= 1;
        }
    }

    nlsf_q15[0] = nlsf_q15[0].max(n_delta_min_q15[0]);
    for i in 1..l {
        nlsf_q15[i] = nlsf_q15[i].max(nlsf_q15[i - 1].saturating_add(n_delta_min_q15[i]));
    }
    nlsf_q15[l - 1] = nlsf_q15[l - 1].min(((1 << 15) - n_delta_min_q15[l] as i32) as i16);
    for i in (0..l - 1).rev() {
        nlsf_q15[i] = nlsf_q15[i].min(nlsf_q15[i + 1] - n_delta_min_q15[i + 1]);
    }
}

/// Convert NLSF parameters to AR prediction filter coefficients
pub fn silk_nlsf2a(a_q12: &mut [i16], nlsf: &[i16], d: usize) {
    // Ordering tables for best numerical accuracy
    const ORDERING16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    const ORDERING10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

    let ordering: &[usize] = if d == 16 { &ORDERING16 } else { &ORDERING10 };

    let mut cos_lsf_qa = [0i32; SILK_MAX_ORDER_LPC];

    // Convert LSFs to 2*cos(LSF) using piecewise linear curve
    for k in 0..d {
        let f_int = (nlsf[k] as i32) >> (15 - 7);
        let f_frac = nlsf[k] as i32 - (f_int << (15 - 7));

        let cos_val = SILK_LSF_COS_TAB_FIX_Q12[f_int as usize] as i32;
        let delta = SILK_LSF_COS_TAB_FIX_Q12[f_int as usize + 1] as i32 - cos_val;

        cos_lsf_qa[ordering[k]] = silk_rshift_round(
            (cos_val << 8) + delta * f_frac,
            20 - QA,
        );
    }

    let dd = d >> 1;

    // Generate even and odd polynomials
    let mut p = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];
    let mut q = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];

    nlsf2a_find_poly(&mut p, &cos_lsf_qa, 0, dd);
    nlsf2a_find_poly(&mut q, &cos_lsf_qa, 1, dd);

    // Convert to filter coefficients
    let mut a32_qa1 = [0i32; SILK_MAX_ORDER_LPC];
    for k in 0..dd {
        let ptmp = p[k + 1] + p[k];
        let qtmp = q[k + 1] - q[k];
        a32_qa1[k] = -qtmp - ptmp;
        a32_qa1[d - k - 1] = qtmp - ptmp;
    }

    // Convert to Q12
    silk_lpc_fit(a_q12, &mut a32_qa1, 12, QA + 1, d);

    // Check stability and apply bandwidth expansion if needed
    for i in 0..MAX_LPC_STABILIZE_ITERATIONS {
        if silk_lpc_inverse_pred_gain(a_q12, d) != 0 {
            break;
        }
        silk_bwexpander_32(&mut a32_qa1, d, 65536 - (2 << i));
        for k in 0..d {
            a_q12[k] = silk_rshift_round(a32_qa1[k], QA + 1 - 12) as i16;
        }
    }
}

fn nlsf2a_find_poly(out: &mut [i32], c_lsf: &[i32], offset: usize, dd: usize) {
    out[0] = 1 << QA;
    out[1] = -c_lsf[offset];
    for k in 1..dd {
        let ftmp = c_lsf[2 * k + offset];
        out[k + 1] = (out[k - 1] << 1) - silk_rshift_round64(
            out[k] as i64 * ftmp as i64, QA
        ) as i32;
        for n in (2..=k).rev() {
            out[n] += out[n - 2] - silk_rshift_round64(
                out[n - 1] as i64 * ftmp as i64, QA
            ) as i32;
        }
        out[1] -= ftmp;
    }
}

/// Convert int32 coefficients to int16 with bandwidth expansion if needed
fn silk_lpc_fit(
    a_qout: &mut [i16],
    a_qin: &mut [i32],
    qout: i32,
    qin: i32,
    d: usize,
) {
    let mut idx = 0usize;
    for _iter in 0..10 {
        let mut maxabs = 0i32;
        for k in 0..d {
            let absval = a_qin[k].abs();
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }
        maxabs = silk_rshift_round(maxabs, qin - qout);

        if maxabs > i16::MAX as i32 {
            maxabs = maxabs.min(163838);
            let chirp_q16 = 65470 - ((maxabs - i16::MAX as i32) << 14)
                / ((maxabs * (idx as i32 + 1)) >> 2).max(1);
            silk_bwexpander_32(a_qin, d, chirp_q16);
        } else {
            // Converged
            for k in 0..d {
                a_qout[k] = silk_rshift_round(a_qin[k], qin - qout) as i16;
            }
            return;
        }
    }

    // Last iteration: clip
    for k in 0..d {
        a_qout[k] = silk_sat16(silk_rshift_round(a_qin[k], qin - qout));
        a_qin[k] = (a_qout[k] as i32) << (qin - qout);
    }
}

/// Compute inverse of LPC prediction gain
pub fn silk_lpc_inverse_pred_gain(a_q12: &[i16], order: usize) -> i32 {
    const QA_IPG: i32 = 24;
    const A_LIMIT: i32 = (0.99975f64 * (1i64 << QA_IPG) as f64) as i32;

    let mut atmp_qa = [0i32; SILK_MAX_ORDER_LPC];
    let mut dc_resp = 0i32;

    for k in 0..order {
        dc_resp += a_q12[k] as i32;
        atmp_qa[k] = (a_q12[k] as i32) << (QA_IPG - 12);
    }
    if dc_resp >= 4096 {
        return 0;
    }

    let mut inv_gain_q30 = 1i32 << 30;

    for k in (1..order).rev() {
        if atmp_qa[k] > A_LIMIT || atmp_qa[k] < -A_LIMIT {
            return 0;
        }

        let rc_q31 = -(atmp_qa[k] << (31 - QA_IPG));
        let rc_mult1_q30 = (1i32 << 30) - silk_smmul(rc_q31, rc_q31);
        if rc_mult1_q30 <= (1 << 15) {
            return 0;
        }

        inv_gain_q30 = silk_smmul(inv_gain_q30, rc_mult1_q30) << 2;
        if inv_gain_q30 < ((1.0f64 / MAX_PREDICTION_POWER_GAIN as f64) * (1i64 << 30) as f64) as i32 {
            return 0;
        }

        let mult2q = 32 - silk_clz32(rc_mult1_q30.abs());
        let rc_mult2 = silk_inverse32_varq(rc_mult1_q30, mult2q + 30);

        for n in 0..(k + 1) >> 1 {
            let tmp1 = atmp_qa[n];
            let tmp2 = atmp_qa[k - n - 1];
            let mul_frac_q = |a32: i32, b32: i32, q: i32| -> i32 {
                silk_rshift_round64(a32 as i64 * b32 as i64, q) as i32
            };
            let t64_1 = silk_rshift_round64(
                (tmp1.saturating_sub(mul_frac_q(tmp2, rc_q31, 31))) as i64 * rc_mult2 as i64,
                mult2q,
            );
            if t64_1 > i32::MAX as i64 || t64_1 < i32::MIN as i64 {
                return 0;
            }
            atmp_qa[n] = t64_1 as i32;

            let t64_2 = silk_rshift_round64(
                (tmp2.saturating_sub(mul_frac_q(tmp1, rc_q31, 31))) as i64 * rc_mult2 as i64,
                mult2q,
            );
            if t64_2 > i32::MAX as i64 || t64_2 < i32::MIN as i64 {
                return 0;
            }
            atmp_qa[k - n - 1] = t64_2 as i32;
        }
    }

    if atmp_qa[0] > A_LIMIT || atmp_qa[0] < -A_LIMIT {
        return 0;
    }

    let rc_q31 = -(atmp_qa[0] << (31 - QA_IPG));
    let rc_mult1_q30 = (1i32 << 30) - silk_smmul(rc_q31, rc_q31);
    inv_gain_q30 = silk_smmul(inv_gain_q30, rc_mult1_q30) << 2;

    if inv_gain_q30 < ((1.0f64 / MAX_PREDICTION_POWER_GAIN as f64) * (1i64 << 30) as f64) as i32 {
        return 0;
    }

    inv_gain_q30
}

/// Bandwidth expander for i16 coefficients (matching C reference exactly)
/// NB: Uses silk_RSHIFT_ROUND(silk_MUL(...), 16) instead of silk_SMULWB
/// to avoid bias that can lead to unstable filters.
pub fn silk_bwexpander(ar: &mut [i16], d: usize, chirp_q16: i32) {
    let mut chirp = chirp_q16;
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for i in 0..d.saturating_sub(1) {
        ar[i] = silk_rshift_round(chirp.wrapping_mul(ar[i] as i32), 16) as i16;
        chirp += silk_rshift_round(chirp.wrapping_mul(chirp_minus_one_q16), 16);
    }
    if d > 0 {
        ar[d - 1] = silk_rshift_round(chirp.wrapping_mul(ar[d - 1] as i32), 16) as i16;
    }
}

/// Bandwidth expander for i32 coefficients
pub fn silk_bwexpander_32(ar: &mut [i32], d: usize, chirp_q16: i32) {
    let mut chirp = chirp_q16;
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for i in 0..d.saturating_sub(1) {
        ar[i] = silk_smulww_correct(chirp, ar[i]);
        chirp += silk_rshift_round(chirp.wrapping_mul(chirp_minus_one_q16), 16);
    }
    if d > 0 {
        ar[d - 1] = silk_smulww_correct(chirp, ar[d - 1]);
    }
}

/// LPC analysis filter
pub fn silk_lpc_analysis_filter(
    out: &mut [i16],
    input: &[i16],
    b_q12: &[i16],
    len: usize,
    d: usize,
) {
    for ix in d..len {
        let mut out32_q12: i32 = (input[ix] as i32) << 12;
        for j in 0..d {
            out32_q12 -= (b_q12[j] as i32) * (input[ix - j - 1] as i32);
        }
        out[ix] = silk_sat16(silk_rshift_round(out32_q12, 12));
    }
}
