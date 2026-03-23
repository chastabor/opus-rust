// Simplified SILK pitch analysis functions for the encoder.
// This provides a basic pitch estimation suitable for a working encoder,
// not the full 3-stage hierarchical search from the reference.
//
// Functions:
// - silk_pitch_analysis_simple: simplified pitch estimation
// - silk_find_pitch_contour: find best pitch contour index
// - silk_find_ltp_params: find best LTP (Long-Term Prediction) parameters

use crate::*;
use crate::tables::*;

/// Simplified pitch estimation.
///
/// Algorithm:
/// 1. Define search range: min_lag = PE_MIN_LAG_MS * fs_khz, max_lag = PE_MAX_LAG_MS * fs_khz
/// 2. Compute normalized autocorrelation for each lag in the search range
/// 3. Find the lag with highest normalized autocorrelation
/// 4. If peak > 0.3, declare voiced; else unvoiced
/// 5. Set all subframe pitch lags to the found lag (uniform pitch for simplicity)
/// 6. Return voiced flag
pub fn silk_pitch_analysis_simple(
    input: &[i16],
    pitch_lags: &mut [i32],
    fs_khz: i32,
    nb_subfr: i32,
    frame_length: i32,
) -> bool {
    let min_lag = (PE_MIN_LAG_MS * fs_khz) as usize;
    let max_lag = (PE_MAX_LAG_MS * fs_khz) as usize;
    let input_len = input.len();

    // We need at least max_lag + frame_length samples
    if input_len < max_lag + frame_length as usize {
        // Not enough data, declare unvoiced
        for k in 0..nb_subfr as usize {
            pitch_lags[k] = min_lag as i32;
        }
        return false;
    }

    // Analyze the last frame_length samples
    let analysis_start = input_len - frame_length as usize;

    // Compute energy of the analysis window for normalization
    let mut energy_q0: i64 = 0;
    for i in analysis_start..input_len {
        energy_q0 += (input[i] as i64) * (input[i] as i64);
    }

    if energy_q0 == 0 {
        // Silent frame
        for k in 0..nb_subfr as usize {
            pitch_lags[k] = min_lag as i32;
        }
        return false;
    }

    // Find best lag by computing normalized cross-correlation
    let mut best_lag = min_lag;
    let mut best_corr_q16: i64 = 0; // normalized correlation in Q16
    let frame_len = frame_length as usize;

    for lag in min_lag..=max_lag {
        if analysis_start < lag {
            continue;
        }

        // Cross-correlation between analysis window and lagged version
        let mut cross_corr: i64 = 0;
        let mut lag_energy: i64 = 0;

        for i in 0..frame_len {
            let x = input[analysis_start + i] as i64;
            let y = input[analysis_start + i - lag] as i64;
            cross_corr += x * y;
            lag_energy += y * y;
        }

        if lag_energy == 0 {
            continue;
        }

        // Normalized correlation: cross_corr / sqrt(energy * lag_energy)
        // We compare cross_corr^2 * 65536 / (energy * lag_energy) to avoid sqrt
        // But for simplicity, compute cross_corr * 65536 / lag_energy
        // (normalized by lag energy only, relative comparison is valid since
        // source energy is constant)
        let norm_corr = if cross_corr > 0 {
            // Compute cross_corr^2 / (energy * lag_energy) in Q16
            // Use 64-bit arithmetic carefully
            let numer = (cross_corr as i128) * (cross_corr as i128) * (1i128 << 16);
            let denom = (energy_q0 as i128) * (lag_energy as i128);
            if denom > 0 {
                (numer / denom) as i64
            } else {
                0
            }
        } else {
            0
        };

        if norm_corr > best_corr_q16 {
            best_corr_q16 = norm_corr;
            best_lag = lag;
        }
    }

    // Voicing threshold: 0.3^2 in Q16 = 0.09 * 65536 ~= 5898
    let voiced = best_corr_q16 > 5898;

    // Set all subframe lags to the best lag (uniform pitch)
    for k in 0..nb_subfr as usize {
        pitch_lags[k] = best_lag as i32;
    }

    voiced
}

/// Find best pitch contour index and lag index for encoding.
///
/// The contour index encodes small per-subframe pitch lag offsets relative to a
/// base lag. Try each contour from the appropriate table, find the one that best
/// matches the actual pitch_lags. The lag_index is `(base_lag - min_lag)`.
pub fn silk_find_pitch_contour(
    contour_index: &mut i8,
    lag_index: &mut i16,
    pitch_lags: &[i32],
    fs_khz: i32,
    nb_subfr: i32,
) {
    let min_lag = PE_MIN_LAG_MS * fs_khz;

    // Base lag is the first subframe lag
    let base_lag = pitch_lags[0];
    *lag_index = (base_lag - min_lag) as i16;

    let nb_subfr_usize = nb_subfr as usize;

    // Find the contour that best matches the actual pitch lags.
    // We handle the 4-subfr and 2-subfr cases separately to avoid
    // type issues with arrays of different inner sizes.
    let mut best_contour = 0i8;
    let mut best_error = i64::MAX;

    if nb_subfr == MAX_NB_SUBFR as i32 {
        // 20ms frame, 4 subframes -- use SILK_CB_LAGS_STAGE3 ([i8; 34] x 4)
        let n_contours = PE_NB_CBKS_STAGE3_MAX;
        for c in 0..n_contours {
            let mut error: i64 = 0;
            for k in 0..nb_subfr_usize {
                let predicted_lag = base_lag + SILK_CB_LAGS_STAGE3[k][c] as i32;
                let diff = (pitch_lags[k] - predicted_lag) as i64;
                error += diff * diff;
            }
            if error < best_error {
                best_error = error;
                best_contour = c as i8;
            }
        }
    } else {
        // 10ms frame, 2 subframes -- use SILK_CB_LAGS_STAGE3_10_MS ([i8; 12] x 2)
        let n_contours = PE_NB_CBKS_STAGE3_10MS;
        for c in 0..n_contours {
            let mut error: i64 = 0;
            for k in 0..nb_subfr_usize {
                let predicted_lag = base_lag + SILK_CB_LAGS_STAGE3_10_MS[k][c] as i32;
                let diff = (pitch_lags[k] - predicted_lag) as i64;
                error += diff * diff;
            }
            if error < best_error {
                best_error = error;
                best_contour = c as i8;
            }
        }
    }

    *contour_index = best_contour;
}

/// Find best LTP (Long-Term Prediction) parameters.
///
/// Simplified LTP: For each subframe, compute the LPC residual, then for each LTP
/// codebook (3 codebooks with 8/16/32 entries), find the best entry by minimizing
/// prediction error. Select the codebook (per_index) and entry (ltp_index[k]) with
/// lowest total error.
pub fn silk_find_ltp_params(
    ltp_index: &mut [i8],
    per_index: &mut i8,
    pitch_lags: &[i32],
    input: &[i16],
    pred_coef_q12: &[i16],
    subfr_length: i32,
    nb_subfr: i32,
    ltp_mem_length: i32,
    lpc_order: i32,
) {
    let subfr_len = subfr_length as usize;
    let nb_subfr_usize = nb_subfr as usize;
    let lpc_ord = lpc_order as usize;

    // First, compute LPC residual for the entire frame
    let total_len = (nb_subfr * subfr_length) as usize;
    let offset = ltp_mem_length as usize; // samples available before the frame

    // We need input[offset..offset+total_len] as the frame
    // and input[0..offset] as the history

    let mut residual = vec![0i32; total_len];

    // Compute LPC residual: r[n] = x[n] - sum(a[k] * x[n-k-1])
    for n in 0..total_len {
        let abs_n = offset + n;
        let mut pred: i64 = 0;
        for k in 0..lpc_ord {
            if abs_n >= k + 1 {
                pred += (pred_coef_q12[k] as i64) * (input[abs_n - k - 1] as i64);
            }
        }
        residual[n] = (input[abs_n] as i32) - ((pred >> 12) as i32);
    }

    // For each LTP codebook, compute total error across all subframes
    let codebook_sizes: [usize; NB_LTP_CBKS] = [8, 16, 32];
    let codebooks: [&[[i8; 5]]; NB_LTP_CBKS] = [
        &SILK_LTP_GAIN_VQ_0,
        &SILK_LTP_GAIN_VQ_1,
        &SILK_LTP_GAIN_VQ_2,
    ];

    let mut best_total_error = i64::MAX;
    let mut best_per = 0usize;
    let mut best_ltp_indices = [0i8; MAX_NB_SUBFR];

    for cbk in 0..NB_LTP_CBKS {
        let n_entries = codebook_sizes[cbk];
        let codebook = codebooks[cbk];
        let mut total_error: i64 = 0;
        let mut cbk_ltp_indices = [0i8; MAX_NB_SUBFR];

        for sf in 0..nb_subfr_usize {
            let lag = pitch_lags[sf] as usize;
            let sf_start = sf * subfr_len;

            // Find best codebook entry for this subframe
            let mut best_entry_error = i64::MAX;
            let mut best_entry = 0usize;

            for entry in 0..n_entries {
                let b = &codebook[entry]; // [i8; 5] LTP coefficients in Q7

                let mut error: i64 = 0;
                for n in 0..subfr_len {
                    let abs_n = sf_start + n;
                    // LTP prediction: sum(b[k] * residual[n - lag + 2 - k]) for k=0..4
                    let mut ltp_pred: i64 = 0;
                    for k in 0..LTP_ORDER {
                        let lag_idx = abs_n as i64 - lag as i64 + 2 - k as i64;
                        if lag_idx >= 0 && (lag_idx as usize) < total_len {
                            ltp_pred += (b[k] as i64) * (residual[lag_idx as usize] as i64);
                        }
                    }
                    ltp_pred >>= 7; // Q7 -> Q0

                    let err = residual[abs_n] as i64 - ltp_pred;
                    error += err * err;
                }

                if error < best_entry_error {
                    best_entry_error = error;
                    best_entry = entry;
                }
            }

            cbk_ltp_indices[sf] = best_entry as i8;
            total_error += best_entry_error;
        }

        if total_error < best_total_error {
            best_total_error = total_error;
            best_per = cbk;
            best_ltp_indices = cbk_ltp_indices;
        }
    }

    *per_index = best_per as i8;
    for sf in 0..nb_subfr_usize {
        ltp_index[sf] = best_ltp_indices[sf];
    }
}
