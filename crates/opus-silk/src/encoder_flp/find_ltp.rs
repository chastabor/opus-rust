// Port of silk/float/find_LTP_FLP.c, corrMatrix_FLP.c, LTP_analysis_filter_FLP.c
// LTP analysis for voiced frames: computes correlation matrices/vectors for LTP
// quantization, and the LTP analysis filter for computing LTP residuals.

use super::dsp::{silk_energy_flp, silk_inner_product_flp};
use crate::{LTP_ORDER, MAX_NB_SUBFR};

const LTP_CORR_INV_MAX: f32 = 0.03;

/// Correlation vector X'*t (port of silk_corrVector_FLP).
///
/// `x` points to x[0..order+L-1], where columns of X are successive lags.
/// `t` points to the target vector of length L (the residual for LTP).
/// Result: Xt[lag] = sum_i x[order-1-lag+i] * t[i], for i=0..L-1.
pub fn silk_corr_vector_flp(
    x: &[f32],      // [order-1+L] — lag_ptr from find_LTP
    t: &[f32],      // [L] — r_ptr (residual at subframe start)
    l: usize,       // subframe length
    order: usize,   // LTP_ORDER (5)
    xt: &mut [f32], // [order] output
) {
    for (lag, xt_lag) in xt.iter_mut().enumerate().take(order) {
        // ptr1 = x + (order-1) - lag = x + (order-1-lag)
        let col_start = order - 1 - lag;
        *xt_lag = silk_inner_product_flp(&x[col_start..col_start + l], &t[..l]) as f32;
    }
}

/// Correlation matrix X'*X (port of silk_corrMatrix_FLP).
///
/// `x` points to x[0..order+L-1]. XX is order×order output (row-major).
pub fn silk_corr_matrix_flp(
    x: &[f32],      // [order-1+L]
    l: usize,       // subframe length
    order: usize,   // LTP_ORDER (5)
    xx: &mut [f32], // [order*order] output, row-major
) {
    // ptr1 = x + (order-1), i.e., column 0 starts at index (order-1)
    let col0 = order - 1;

    // Diagonal: XX[j][j] = energy of column j
    let mut energy = silk_energy_flp(&x[col0..col0 + l]);
    xx[0] = energy as f32; // XX[0][0]
    for j in 1..order {
        // Update energy: add x[col0-j]^2, subtract x[col0+L-j]^2
        energy += x[col0 - j] as f64 * x[col0 - j] as f64
            - x[col0 + l - j] as f64 * x[col0 + l - j] as f64;
        xx[j * order + j] = energy as f32;
    }

    // Off-diagonal: XX[0][lag] = inner_product(col0, col0-lag)
    for lag in 1..order {
        let col_lag = col0 - lag; // first sample of column `lag`
        let mut e = silk_inner_product_flp(&x[col0..col0 + l], &x[col_lag..col_lag + l]);
        xx[lag * order] = e as f32; // XX[lag][0]
        xx[lag] = e as f32; // XX[0][lag] (symmetric)

        for j in 1..(order - lag) {
            // Update: add contribution from new samples
            e += x[col0 - j] as f64 * x[col_lag - j] as f64
                - x[col0 + l - j] as f64 * x[col_lag + l - j] as f64;
            xx[(lag + j) * order + j] = e as f32;
            xx[j * order + (lag + j)] = e as f32;
        }
    }
}

/// Float LTP analysis (port of silk_find_LTP_FLP).
///
/// Computes weighted correlation matrices (XX) and vectors (xX) for LTP
/// gain quantization, one set per subframe.
///
/// `res` is the LPC residual from pitch analysis, starting at the beginning
/// of the frame (i.e., res[0] is the first residual sample of subframe 0).
/// The residual must have enough preceding history for the pitch lags
/// (at least max(lag) + LTP_ORDER/2 samples before the frame).
///
/// In the C code: r_ptr starts at res_pitch (which points to frame start
/// in the residual buffer, with ltp_mem_length of history behind it).
pub fn silk_find_ltp_flp(
    xx_out: &mut [f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER],
    x_x_out: &mut [f32; MAX_NB_SUBFR * LTP_ORDER],
    res: &[f32],             // residual buffer (with history before frame start)
    res_frame_offset: usize, // offset where frame starts in res buffer
    lag: &[i32; MAX_NB_SUBFR],
    subfr_length: usize,
    nb_subfr: usize,
) {
    let order = LTP_ORDER;

    for (k, &lag_k) in lag.iter().enumerate().take(nb_subfr) {
        // r_ptr = start of subframe k's residual
        let r_start = res_frame_offset + k * subfr_length;

        // lag_ptr = r_ptr - (lag[k] + LTP_ORDER/2)
        // This gives the lagged signal aligned for the 5-tap filter
        let lag_offset = lag_k as usize + order / 2;
        let lag_start = r_start - lag_offset;

        // Compute correlation matrix XX for this subframe
        let xx_start = k * order * order;
        silk_corr_matrix_flp(
            &res[lag_start..],
            subfr_length,
            order,
            &mut xx_out[xx_start..xx_start + order * order],
        );

        // Compute correlation vector xX for this subframe
        let x_x_start = k * order;
        silk_corr_vector_flp(
            &res[lag_start..],
            &res[r_start..r_start + subfr_length],
            subfr_length,
            order,
            &mut x_x_out[x_x_start..x_x_start + order],
        );

        // Normalize by energy
        let xx = silk_energy_flp(&res[r_start..r_start + subfr_length + order]) as f32;
        let temp = 1.0f32
            / xx.max(
                LTP_CORR_INV_MAX * 0.5 * (xx_out[xx_start] + xx_out[xx_start + order * order - 1])
                    + 1.0,
            );

        // Scale XX and xX
        for i in 0..(order * order) {
            xx_out[xx_start + i] *= temp;
        }
        for i in 0..order {
            x_x_out[x_x_start + i] *= temp;
        }
    }
}

/// LTP analysis filter (port of silk_LTP_analysis_filter_FLP).
///
/// Computes gain-normalized LTP residual for each subframe:
///   LTP_res[i] = invGain * (x[i] - sum_j B[j] * x_lag[LTP_ORDER/2 - j])
///
/// `x_buf` is the full signal buffer. `x_offset` is where x_frame - pre_length
/// starts (i.e., x_buf[x_offset] = first sample for subframe 0 including
/// pre_length leading samples). The buffer must have enough history before
/// x_offset for the pitch lag access.
///
/// Output: `ltp_res` has nb_subfr * (pre_length + subfr_length) samples.
#[allow(clippy::too_many_arguments)]
pub fn silk_ltp_analysis_filter_flp(
    ltp_res: &mut [f32],
    x_buf: &[f32],                       // full signal buffer
    x_offset: usize,                     // offset where x - pre_length starts
    b: &[f32; MAX_NB_SUBFR * LTP_ORDER], // LTP coefficients per subframe
    pitch_l: &[i32; MAX_NB_SUBFR],
    inv_gains: &[f32; MAX_NB_SUBFR],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    let order = LTP_ORDER;
    let seg_len = subfr_length + pre_length;

    for k in 0..nb_subfr {
        // x_ptr in C starts at x + k*subfr_length (x = x_frame - pre_length)
        // In our buffer: x_buf[x_offset + k * subfr_length]
        let x_ptr_base = x_offset + k * subfr_length;
        let lag = pitch_l[k] as usize;
        let inv_gain = inv_gains[k];
        let b_k = &b[k * order..(k + 1) * order];

        let res_start = k * seg_len;

        for i in 0..seg_len {
            let mut val = x_buf[x_ptr_base + i];
            let center = x_ptr_base + i - lag + order / 2;
            for j in 0..order {
                val -= b_k[j] * x_buf[center - j];
            }
            ltp_res[res_start + i] = val * inv_gain;
        }
    }
}
