// Port of SILK NLSF encoding functions:
// - silk_NLSF_VQ (codebook error computation)
// - silk_NLSF_del_dec_quant (delayed-decision trellis quantizer)
// - silk_NLSF_encode (top-level NLSF encoder)
//
// Ported from silk/NLSF_VQ.c, silk/NLSF_del_dec_quant.c, silk/NLSF_encode.c

use crate::*;
use crate::nlsf::*;

// Constants from silk/define.h
const NLSF_QUANT_DEL_DEC_STATES_LOG2: usize = 2;
const NLSF_QUANT_DEL_DEC_STATES: usize = 1 << NLSF_QUANT_DEL_DEC_STATES_LOG2;
const NLSF_QUANT_MAX_AMPLITUDE_EXT: i32 = 10;

// NLSF_QUANT_LEVEL_ADJ = 0.1 in Q10 = 102
const NLSF_QUANT_LEVEL_ADJ_Q10: i32 = 102;

/// NLSF VQ error computation for first-stage codebook search.
///
/// Port of silk_NLSF_VQ from silk/NLSF_VQ.c.
/// For each codebook vector, computes weighted absolute predictive quantization error.
fn silk_nlsf_vq(
    err_q24: &mut [i32],
    in_q15: &[i16],
    cb_nlsf_q8: &[u8],
    cb_wght_q9: &[i16],
    n_vectors: usize,
    order: usize,
) {
    for i in 0..n_vectors {
        let cb_offset = i * order;
        let mut sum_error_q24: i32 = 0;
        let mut pred_q24: i32 = 0;

        // Loop backward in pairs (matching C reference)
        let mut m = order as i32 - 2;
        while m >= 0 {
            let m_u = m as usize;

            // Compute weighted absolute predictive quantization error for index m + 1
            let diff_q15 = (in_q15[m_u + 1] as i32) - ((cb_nlsf_q8[cb_offset + m_u + 1] as i32) << 7);
            let diffw_q24 = diff_q15.wrapping_mul(cb_wght_q9[cb_offset + m_u + 1] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24 - (pred_q24 >> 1)).abs());
            pred_q24 = diffw_q24;

            // Compute weighted absolute predictive quantization error for index m
            let diff_q15 = (in_q15[m_u] as i32) - ((cb_nlsf_q8[cb_offset + m_u] as i32) << 7);
            let diffw_q24 = diff_q15.wrapping_mul(cb_wght_q9[cb_offset + m_u] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24 - (pred_q24 >> 1)).abs());
            pred_q24 = diffw_q24;

            m -= 2;
        }
        err_q24[i] = sum_error_q24;
    }
}

/// Delayed-decision trellis quantizer for NLSF residuals.
///
/// Port of silk_NLSF_del_dec_quant from silk/NLSF_del_dec_quant.c.
/// Uses a multi-state trellis with bounded path tracking.
/// Returns RD value in Q25.
fn silk_nlsf_del_dec_quant(
    indices: &mut [i8],
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i16,
    mu_q20: i32,
    order: i16,
) -> i32 {
    let order_usize = order as usize;

    // Pre-compute output tables
    let table_size = 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize;
    let mut out0_q10_table = vec![0i32; table_size];
    let mut out1_q10_table = vec![0i32; table_size];

    for iv in -NLSF_QUANT_MAX_AMPLITUDE_EXT..NLSF_QUANT_MAX_AMPLITUDE_EXT {
        let mut out0: i32 = iv << 10;
        let mut out1: i32 = out0 + 1024;

        if iv > 0 {
            out0 -= NLSF_QUANT_LEVEL_ADJ_Q10;
            out1 -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if iv == 0 {
            out1 -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if iv == -1 {
            out0 += NLSF_QUANT_LEVEL_ADJ_Q10;
        } else {
            out0 += NLSF_QUANT_LEVEL_ADJ_Q10;
            out1 += NLSF_QUANT_LEVEL_ADJ_Q10;
        }

        let idx = (iv + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize;
        out0_q10_table[idx] = ((out0 as i64 * quant_step_size_q16 as i64) >> 16) as i32;
        out1_q10_table[idx] = ((out1 as i64 * quant_step_size_q16 as i64) >> 16) as i32;
    }

    let mut n_states: usize = 1;
    let mut rd_q25 = [0i32; 2 * NLSF_QUANT_DEL_DEC_STATES];
    let mut prev_out_q10 = [0i16; 2 * NLSF_QUANT_DEL_DEC_STATES];
    let mut ind: [[i8; MAX_LPC_ORDER]; NLSF_QUANT_DEL_DEC_STATES] =
        [[0i8; MAX_LPC_ORDER]; NLSF_QUANT_DEL_DEC_STATES];
    let mut ind_sort = [0usize; NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_min_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_max_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];

    rd_q25[0] = 0;
    prev_out_q10[0] = 0;

    // Process coefficients in reverse order (matching C reference)
    for i_rev in (0..order_usize).rev() {
        let rates_q5_offset = ec_ix[i_rev] as usize;
        let in_q10 = x_q10[i_rev] as i32;

        for j in 0..n_states {
            let pred_q10 = (((pred_coef_q8[i_rev] as i16 as i32) * (prev_out_q10[j] as i32)) >> 8) as i32;
            let res_q10 = in_q10 - pred_q10;
            let mut ind_tmp = ((inv_quant_step_size_q6 as i32) * res_q10) >> 16;
            ind_tmp = ind_tmp.clamp(-NLSF_QUANT_MAX_AMPLITUDE_EXT, NLSF_QUANT_MAX_AMPLITUDE_EXT - 1);
            ind[j][i_rev] = ind_tmp as i8;

            // Compute outputs for ind_tmp and ind_tmp + 1
            let table_idx = (ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize;
            let out0 = (out0_q10_table[table_idx] as i16 as i32 + pred_q10) as i16;
            let out1 = (out1_q10_table[table_idx] as i16 as i32 + pred_q10) as i16;

            prev_out_q10[j] = out0;
            prev_out_q10[j + n_states] = out1;

            // Compute RD for ind_tmp and ind_tmp + 1
            let rate0_q5: i32;
            let rate1_q5: i32;

            if ind_tmp + 1 >= NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp + 1 == NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_q5 = ec_rates_q5[rates_q5_offset + (ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                    rate1_q5 = 280;
                } else {
                    rate0_q5 = 280 - 43 * NLSF_QUANT_MAX_AMPLITUDE + 43 * ind_tmp;
                    rate1_q5 = rate0_q5 + 43;
                }
            } else if ind_tmp <= -NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp == -NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_q5 = 280;
                    rate1_q5 = ec_rates_q5[rates_q5_offset + (ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                } else {
                    rate0_q5 = 280 + 43 * NLSF_QUANT_MAX_AMPLITUDE - 43 * ind_tmp;
                    rate1_q5 = rate0_q5 - 43;
                }
            } else {
                rate0_q5 = ec_rates_q5[rates_q5_offset + (ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                rate1_q5 = ec_rates_q5[rates_q5_offset + (ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
            }

            let rd_tmp_q25 = rd_q25[j];
            let diff0 = in_q10 - out0 as i32;
            rd_q25[j] = rd_tmp_q25
                .wrapping_add(diff0.wrapping_mul(diff0).wrapping_mul(w_q5[i_rev] as i32))
                .wrapping_add(mu_q20.wrapping_mul(rate0_q5));
            let diff1 = in_q10 - out1 as i32;
            rd_q25[j + n_states] = rd_tmp_q25
                .wrapping_add(diff1.wrapping_mul(diff1).wrapping_mul(w_q5[i_rev] as i32))
                .wrapping_add(mu_q20.wrapping_mul(rate1_q5));
        }

        if n_states <= NLSF_QUANT_DEL_DEC_STATES / 2 {
            // Double number of states and copy
            for j in 0..n_states {
                ind[j + n_states][i_rev] = ind[j][i_rev] + 1;
            }
            n_states <<= 1;
            for j in n_states..NLSF_QUANT_DEL_DEC_STATES {
                ind[j][i_rev] = ind[j - n_states][i_rev];
            }
        } else {
            // Sort lower and upper half of RD_Q25, pairwise
            for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                if rd_q25[j] > rd_q25[j + NLSF_QUANT_DEL_DEC_STATES] {
                    rd_max_q25[j] = rd_q25[j];
                    rd_min_q25[j] = rd_q25[j + NLSF_QUANT_DEL_DEC_STATES];
                    rd_q25[j] = rd_min_q25[j];
                    rd_q25[j + NLSF_QUANT_DEL_DEC_STATES] = rd_max_q25[j];
                    let tmp = prev_out_q10[j];
                    prev_out_q10[j] = prev_out_q10[j + NLSF_QUANT_DEL_DEC_STATES];
                    prev_out_q10[j + NLSF_QUANT_DEL_DEC_STATES] = tmp;
                    ind_sort[j] = j + NLSF_QUANT_DEL_DEC_STATES;
                } else {
                    rd_min_q25[j] = rd_q25[j];
                    rd_max_q25[j] = rd_q25[j + NLSF_QUANT_DEL_DEC_STATES];
                    ind_sort[j] = j;
                }
            }

            // Compare highest RD of winning half with lowest of losing half
            loop {
                let mut min_max_q25 = i32::MAX;
                let mut max_min_q25 = 0i32;
                let mut ind_min_max = 0usize;
                let mut ind_max_min = 0usize;

                for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                    if min_max_q25 > rd_max_q25[j] {
                        min_max_q25 = rd_max_q25[j];
                        ind_min_max = j;
                    }
                    if max_min_q25 < rd_min_q25[j] {
                        max_min_q25 = rd_min_q25[j];
                        ind_max_min = j;
                    }
                }

                if min_max_q25 >= max_min_q25 {
                    break;
                }

                ind_sort[ind_max_min] = ind_sort[ind_min_max] ^ NLSF_QUANT_DEL_DEC_STATES;
                rd_q25[ind_max_min] = rd_q25[ind_min_max + NLSF_QUANT_DEL_DEC_STATES];
                prev_out_q10[ind_max_min] = prev_out_q10[ind_min_max + NLSF_QUANT_DEL_DEC_STATES];
                rd_min_q25[ind_max_min] = 0;
                rd_max_q25[ind_min_max] = i32::MAX;
                ind[ind_max_min] = ind[ind_min_max];
            }

            // Increment index if it comes from the upper half
            for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                ind[j][i_rev] += (ind_sort[j] >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
            }
        }
    }

    // Last sample: find winner, copy indices and return RD value
    let mut ind_tmp = 0usize;
    let mut min_q25 = i32::MAX;
    for j in 0..(2 * NLSF_QUANT_DEL_DEC_STATES) {
        if min_q25 > rd_q25[j] {
            min_q25 = rd_q25[j];
            ind_tmp = j;
        }
    }

    for j in 0..order_usize {
        indices[j] = ind[ind_tmp & (NLSF_QUANT_DEL_DEC_STATES - 1)][j];
    }
    indices[0] += (ind_tmp >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;

    if min_q25 < 0 {
        min_q25 = 0;
    }
    min_q25
}

/// Insertion sort with index tracking (find top n elements)
fn insertion_sort_increasing(arr: &mut [i32], idx: &mut [i32], len: usize) {
    for i in 0..len {
        idx[i] = i as i32;
    }
    for i in 1..len {
        let val = arr[i];
        let id = idx[i];
        let mut j = i;
        while j > 0 && arr[j - 1] > val {
            arr[j] = arr[j - 1];
            idx[j] = idx[j - 1];
            j -= 1;
        }
        arr[j] = val;
        idx[j] = id;
    }
}

/// Top-level NLSF encoder.
///
/// Port of silk_NLSF_encode from silk/NLSF_encode.c.
/// Returns RD value in Q25.
pub fn silk_nlsf_encode(
    nlsf_indices: &mut [i8],
    p_nlsf_q15: &mut [i16],
    nlsf_cb: &NlsfCbStruct,
    p_w_q2: &[i16],
    nlsf_mu_q20: i32,
    n_survivors: usize,
    signal_type: i32,
) -> i32 {
    let order = nlsf_cb.order as usize;
    let n_vectors = nlsf_cb.n_vectors as usize;

    // NLSF stabilization
    silk_nlsf_stabilize(p_nlsf_q15, nlsf_cb.delta_min_q15, order);

    // First stage: VQ
    let mut err_q24 = vec![0i32; n_vectors];
    silk_nlsf_vq(
        &mut err_q24,
        p_nlsf_q15,
        nlsf_cb.cb1_nlsf_q8,
        nlsf_cb.cb1_wght_q9,
        n_vectors,
        order,
    );

    // Sort the quantization errors
    let mut temp_indices1 = vec![0i32; n_vectors];
    insertion_sort_increasing(&mut err_q24, &mut temp_indices1, n_vectors);

    let mut rd_q25 = vec![0i32; n_survivors];
    let mut temp_indices2 = vec![0i8; n_survivors * MAX_LPC_ORDER];

    // Loop over survivors
    for s in 0..n_survivors {
        let ind1 = temp_indices1[s] as usize;

        // Residual after first stage
        let cb_offset = ind1 * order;
        let mut res_q10 = [0i16; MAX_LPC_ORDER];
        let mut w_adj_q5 = [0i16; MAX_LPC_ORDER];

        for i in 0..order {
            let nlsf_tmp_q15 = (nlsf_cb.cb1_nlsf_q8[cb_offset + i] as i32) << 7;
            let w_tmp_q9 = nlsf_cb.cb1_wght_q9[cb_offset + i] as i32;
            let diff = p_nlsf_q15[i] as i32 - nlsf_tmp_q15;
            res_q10[i] = ((diff.wrapping_mul(w_tmp_q9)) >> 14) as i16;
            w_adj_q5[i] = silk_div32_varq(
                p_w_q2[i] as i32,
                w_tmp_q9.wrapping_mul(w_tmp_q9),
                21,
            ) as i16;
        }

        // Unpack entropy table indices and predictor for current CB1 index
        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred_q8 = [0u8; MAX_LPC_ORDER];
        nlsf_unpack(&mut ec_ix, &mut pred_q8, nlsf_cb, ind1);

        // Trellis quantizer
        let trellis_offset = s * MAX_LPC_ORDER;
        rd_q25[s] = silk_nlsf_del_dec_quant(
            &mut temp_indices2[trellis_offset..],
            &res_q10,
            &w_adj_q5,
            &pred_q8,
            &ec_ix,
            nlsf_cb.ec_rates_q5,
            nlsf_cb.quant_step_size_q16 as i32,
            nlsf_cb.inv_quant_step_size_q6,
            nlsf_mu_q20,
            nlsf_cb.order,
        );

        // Add rate for first stage
        let icdf_offset = ((signal_type >> 1) as usize) * n_vectors;
        let prob_q8 = if ind1 == 0 {
            256 - nlsf_cb.cb1_icdf[icdf_offset + ind1] as i32
        } else {
            nlsf_cb.cb1_icdf[icdf_offset + ind1 - 1] as i32
                - nlsf_cb.cb1_icdf[icdf_offset + ind1] as i32
        };
        let bits_q7 = (8 << 7) - silk_lin2log(prob_q8);
        rd_q25[s] = silk_smlabb(rd_q25[s], bits_q7, nlsf_mu_q20 >> 2);
    }

    // Find the lowest rate-distortion error
    let mut best_index = 0usize;
    let mut best_rd = rd_q25[0];
    for s in 1..n_survivors {
        if rd_q25[s] < best_rd {
            best_rd = rd_q25[s];
            best_index = s;
        }
    }

    nlsf_indices[0] = temp_indices1[best_index] as i8;
    let best_offset = best_index * MAX_LPC_ORDER;
    for i in 0..order {
        nlsf_indices[i + 1] = temp_indices2[best_offset + i];
    }

    // Decode back to get the quantized NLSFs
    silk_nlsf_decode(p_nlsf_q15, nlsf_indices, nlsf_cb);

    best_rd
}
