// Port of silk/quant_LTP_gains.c + silk/VQ_WMat_EC.c + float wrapper from wrappers_FLP.c
// Quantizes LTP gains using codebook VQ search with rate-distortion optimization.

use crate::tables::*;
use crate::{silk_lin2log, silk_log2lin, LTP_ORDER, MAX_NB_SUBFR};

const MAX_SUM_LOG_GAIN_DB: f32 = 250.0;

/// C-matching silk_SMLAWB: a + ((b >> 16) * (i16)(c) + (((b & 0xFFFF) * (i16)(c)) >> 16))
#[inline(always)]
fn silk_smlawb_local(a: i32, b: i32, c: i32) -> i32 {
    let c16 = c as i16 as i32;
    a + (b >> 16) * c16 + (((b & 0x0000FFFF) * c16) >> 16)
}

/// Entropy-constrained matrix-weighted VQ for 5-element LTP vectors.
/// Port of silk_VQ_WMat_EC_c (VQ_WMat_EC.c).
fn silk_vq_wmat_ec(
    ind: &mut i8,
    res_nrg_q15: &mut i32,
    rate_dist_q8: &mut i32,
    gain_q7: &mut i32,
    xx_q17: &[i32],      // [LTP_ORDER * LTP_ORDER] correlation matrix
    x_x_q17: &[i32],     // [LTP_ORDER] correlation vector
    cb_q7: &[[i8; 5]],   // codebook
    cb_gain_q7: &[u8],   // codebook effective gains
    cl_q5: &[u8],        // code lengths
    subfr_len: i32,
    max_gain_q7: i32,
    l: usize,            // codebook size
) {
    // Negate and shift xX to Q24
    let mut neg_x_x_q24 = [0i32; 5];
    for i in 0..5 {
        neg_x_x_q24[i] = -(x_x_q17[i] << 7);
    }

    *rate_dist_q8 = i32::MAX;
    *res_nrg_q15 = i32::MAX;
    *ind = 0;

    for k in 0..l {
        let cb_row = &cb_q7[k];
        let gain_tmp_q7 = cb_gain_q7[k] as i32;

        // Penalty for too large gain
        let penalty = (gain_tmp_q7 - max_gain_q7).max(0) << 11;

        // Quantization error: 1 - 2*xX*cb + cb'*XX*cb
        // Start with 1.001 in Q15
        let mut sum1_q15: i32 = 32800; // SILK_FIX_CONST(1.001, 15)

        // Row 0 of XX — silk_MLA(a, b, c) = a + b*c (plain 32-bit multiply)
        let mut sum2_q24 = neg_x_x_q24[0]
            + xx_q17[1] * cb_row[1] as i32
            + xx_q17[2] * cb_row[2] as i32
            + xx_q17[3] * cb_row[3] as i32
            + xx_q17[4] * cb_row[4] as i32;
        sum2_q24 <<= 1;
        sum2_q24 += xx_q17[0] * cb_row[0] as i32;
        sum1_q15 = silk_smlawb_local(sum1_q15, sum2_q24, cb_row[0] as i32);

        // Row 1 of XX
        sum2_q24 = neg_x_x_q24[1]
            + xx_q17[7] * cb_row[2] as i32
            + xx_q17[8] * cb_row[3] as i32
            + xx_q17[9] * cb_row[4] as i32;
        sum2_q24 <<= 1;
        sum2_q24 += xx_q17[6] * cb_row[1] as i32;
        sum1_q15 = silk_smlawb_local(sum1_q15, sum2_q24, cb_row[1] as i32);

        // Row 2 of XX
        sum2_q24 = neg_x_x_q24[2]
            + xx_q17[13] * cb_row[3] as i32
            + xx_q17[14] * cb_row[4] as i32;
        sum2_q24 <<= 1;
        sum2_q24 += xx_q17[12] * cb_row[2] as i32;
        sum1_q15 = silk_smlawb_local(sum1_q15, sum2_q24, cb_row[2] as i32);

        // Row 3 of XX
        sum2_q24 = neg_x_x_q24[3]
            + xx_q17[19] * cb_row[4] as i32;
        sum2_q24 <<= 1;
        sum2_q24 += xx_q17[18] * cb_row[3] as i32;
        sum1_q15 = silk_smlawb_local(sum1_q15, sum2_q24, cb_row[3] as i32);

        // Row 4 of XX
        sum2_q24 = neg_x_x_q24[4] << 1;
        sum2_q24 += xx_q17[24] * cb_row[4] as i32;
        sum1_q15 = silk_smlawb_local(sum1_q15, sum2_q24, cb_row[4] as i32);

        if sum1_q15 >= 0 {
            // Translate residual energy to bits: high-rate assumption (6dB = 1 bit/sample)
            let bits_res_q8 = (subfr_len as i32) * (silk_lin2log(sum1_q15 + penalty) - (15 << 7));
            // Reduce codelength by half ("-1"): cl_Q5[k] << (3-1)
            let bits_tot_q8 = bits_res_q8 + ((cl_q5[k] as i32) << 2);
            if bits_tot_q8 <= *rate_dist_q8 {
                *rate_dist_q8 = bits_tot_q8;
                *res_nrg_q15 = sum1_q15 + penalty;
                *ind = k as i8;
                *gain_q7 = gain_tmp_q7;
            }
        }
    }
}

/// Quantize LTP gains (port of silk_quant_LTP_gains from quant_LTP_gains.c).
///
/// Searches 3 codebooks with different rate/distortion tradeoffs.
pub fn silk_quant_ltp_gains(
    b_q14: &mut [i16; MAX_NB_SUBFR * LTP_ORDER],
    cbk_index: &mut [i8; MAX_NB_SUBFR],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_db_q7: &mut i32,
    xx_q17: &[i32],   // [nb_subfr * LTP_ORDER * LTP_ORDER]
    x_x_q17: &[i32],  // [nb_subfr * LTP_ORDER]
    subfr_len: i32,
    nb_subfr: usize,
) {
    let order = LTP_ORDER;
    let gain_safety: i32 = 51; // SILK_FIX_CONST(0.4, 7) = 0.4 * 128 = 51.2 ≈ 51

    let mut min_rate_dist_q7 = i32::MAX;
    let mut best_sum_log_gain_q7 = 0i32;
    let mut best_res_nrg_q15 = 0i32;
    let mut temp_idx = [0i8; MAX_NB_SUBFR];

    for k in 0..3 {
        let cl_ptr = SILK_LTP_GAIN_BITS_Q5_PTRS[k];
        let cbk_ptr = SILK_LTP_VQ_PTRS_Q7[k];
        let cbk_gain_ptr = SILK_LTP_VQ_GAIN_PTRS_Q7[k];
        let cbk_size = SILK_LTP_VQ_SIZES[k];

        let mut res_nrg_q15 = 0i32;
        let mut rate_dist_q7 = 0i32;
        let mut sum_log_gain_tmp_q7 = *sum_log_gain_q7;

        for j in 0..nb_subfr {
            // MAX_SUM_LOG_GAIN_DB / 6.0 in Q7 = 250/6 * 128 ≈ 5333
            let max_sum_log_gain_q7: i32 = ((MAX_SUM_LOG_GAIN_DB / 6.0) * 128.0) as i32;
            let max_gain_q7 =
                silk_log2lin(max_sum_log_gain_q7 - sum_log_gain_tmp_q7 + (7 << 7)) - gain_safety;

            let mut ind = 0i8;
            let mut res_nrg_q15_subfr = 0i32;
            let mut rate_dist_q7_subfr = 0i32;
            let mut gain_q7 = 0i32;

            let xx_offset = j * order * order;
            let x_x_offset = j * order;

            silk_vq_wmat_ec(
                &mut ind,
                &mut res_nrg_q15_subfr,
                &mut rate_dist_q7_subfr,
                &mut gain_q7,
                &xx_q17[xx_offset..xx_offset + order * order],
                &x_x_q17[x_x_offset..x_x_offset + order],
                cbk_ptr,
                cbk_gain_ptr,
                cl_ptr,
                subfr_len,
                max_gain_q7,
                cbk_size,
            );

            temp_idx[j] = ind;

            // Saturating add for residual energy and rate-distortion
            res_nrg_q15 = res_nrg_q15.saturating_add(res_nrg_q15_subfr);
            rate_dist_q7 = rate_dist_q7.saturating_add(rate_dist_q7_subfr);

            sum_log_gain_tmp_q7 = 0i32.max(
                sum_log_gain_tmp_q7 + silk_lin2log(gain_safety + gain_q7) - (7 << 7),
            );
        }

        if rate_dist_q7 <= min_rate_dist_q7 {
            min_rate_dist_q7 = rate_dist_q7;
            *periodicity_index = k as i8;
            cbk_index[..nb_subfr].copy_from_slice(&temp_idx[..nb_subfr]);
            best_sum_log_gain_q7 = sum_log_gain_tmp_q7;
            best_res_nrg_q15 = res_nrg_q15;
        }
    }

    // Extract quantized LTP coefficients from the best codebook
    let best_cbk = SILK_LTP_VQ_PTRS_Q7[*periodicity_index as usize];
    for j in 0..nb_subfr {
        let row = &best_cbk[cbk_index[j] as usize];
        for i in 0..order {
            b_q14[j * order + i] = (row[i] as i16) << 7;
        }
    }

    // Normalize residual energy
    let norm_res_nrg_q15 = if nb_subfr == 2 {
        best_res_nrg_q15 >> 1
    } else {
        best_res_nrg_q15 >> 2
    };

    *sum_log_gain_q7 = best_sum_log_gain_q7;
    *pred_gain_db_q7 = -3 * (silk_lin2log(norm_res_nrg_q15) - (15 << 7));
}

/// Float wrapper for silk_quant_LTP_gains (port of silk_quant_LTP_gains_FLP
/// from wrappers_FLP.c).
///
/// Converts float correlation matrices to Q17, calls the fixed-point quantizer,
/// and converts results back to float.
pub fn silk_quant_ltp_gains_flp(
    b: &mut [f32; MAX_NB_SUBFR * LTP_ORDER], // O: quantized LTP gains (float)
    cbk_index: &mut [i8; MAX_NB_SUBFR],      // O: codebook indices
    periodicity_index: &mut i8,                // O: periodicity index
    sum_log_gain_q7: &mut i32,                 // I/O: cumulative max prediction gain
    pred_gain_db: &mut f32,                    // O: LTP prediction coding gain (dB)
    xx: &[f32],                                // I: correlation matrices [nb_subfr * 25]
    x_x: &[f32],                               // I: correlation vectors [nb_subfr * 5]
    subfr_len: i32,
    nb_subfr: usize,
) {
    let order = LTP_ORDER;

    // Convert XX to Q17 (× 131072)
    let n_xx = nb_subfr * order * order;
    let mut xx_q17 = vec![0i32; n_xx];
    for i in 0..n_xx {
        xx_q17[i] = (xx[i] * 131072.0).round() as i32;
    }

    // Convert xX to Q17
    let n_x_x = nb_subfr * order;
    let mut x_x_q17 = vec![0i32; n_x_x];
    for i in 0..n_x_x {
        x_x_q17[i] = (x_x[i] * 131072.0).round() as i32;
    }

    let mut b_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    let mut pred_gain_db_q7 = 0i32;

    silk_quant_ltp_gains(
        &mut b_q14,
        cbk_index,
        periodicity_index,
        sum_log_gain_q7,
        &mut pred_gain_db_q7,
        &xx_q17,
        &x_x_q17,
        subfr_len,
        nb_subfr,
    );

    // Convert B_Q14 to float
    for i in 0..(nb_subfr * order) {
        b[i] = b_q14[i] as f32 / 16384.0;
    }

    *pred_gain_db = pred_gain_db_q7 as f32 / 128.0;
}
