// Port of silk/gain_quant.c - Gain scalar quantization with hysteresis

use crate::*;

const OFFSET: i32 = (MIN_QGAIN_DB * 128) / 6 + 16 * 128;
const SCALE_Q16: i32 = (65536 * (N_LEVELS_QGAIN - 1)) / (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6);
const INV_SCALE_Q16: i32 =
    (65536 * (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6)) / (N_LEVELS_QGAIN - 1);

/// Gain scalar quantization with hysteresis, uniform on log scale
/// Port of silk_gains_quant from silk/gain_quant.c
pub fn silk_gains_quant(
    ind: &mut [i8],
    gain_q16: &mut [i32],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        // Convert to log scale, scale, floor()
        ind[k] = silk_smulwb(SCALE_Q16, silk_lin2log(gain_q16[k]) - OFFSET) as i8;

        // Round towards previous quantized gain (hysteresis)
        if (ind[k] as i32) < (*prev_ind as i32) {
            ind[k] = (ind[k] as i32 + 1) as i8;
        }
        ind[k] = (ind[k] as i32).clamp(0, N_LEVELS_QGAIN - 1) as i8;

        if k == 0 && !conditional {
            // Full index: clamp to [prev_ind + MIN_DELTA, N_LEVELS-1]
            // Note: lower bound CAN be negative when prev_ind is small (matching C)
            ind[k] = (ind[k] as i32)
                .clamp(*prev_ind as i32 + MIN_DELTA_GAIN_QUANT, N_LEVELS_QGAIN - 1)
                as i8;
            *prev_ind = ind[k];
        } else {
            // Delta index
            let mut cur_ind = ind[k] as i32 - *prev_ind as i32;

            // Double the quantization step size for large gain increases
            let double_step_size_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + *prev_ind as i32;
            if cur_ind > double_step_size_threshold {
                cur_ind =
                    double_step_size_threshold + ((cur_ind - double_step_size_threshold + 1) >> 1);
            }

            cur_ind = cur_ind.clamp(MIN_DELTA_GAIN_QUANT, MAX_DELTA_GAIN_QUANT);

            // Accumulate deltas (matching C: no clamp to 0 in else branch)
            if cur_ind > double_step_size_threshold {
                *prev_ind = ((*prev_ind as i32 + (cur_ind << 1) - double_step_size_threshold)
                    .min(N_LEVELS_QGAIN - 1)) as i8;
            } else {
                *prev_ind = (*prev_ind as i32 + cur_ind) as i8;
            }

            // Shift to make non-negative
            ind[k] = (cur_ind - MIN_DELTA_GAIN_QUANT) as i8;
        }

        // Scale and convert to linear scale
        gain_q16[k] =
            silk_log2lin((silk_smulwb(INV_SCALE_Q16, *prev_ind as i32) + OFFSET).min(3967));
    }
}

/// Gains scalar dequantization, matching silk_gains_dequant from silk/gain_quant.c.
/// This is the inverse of silk_gains_quant.
pub fn silk_gains_dequant(
    gain_q16: &mut [i32],
    ind: &[i8],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        if k == 0 && !conditional {
            // Absolute index: clamp to not go below prev_ind - 16
            *prev_ind = ind[k].max(*prev_ind - 16);
        } else {
            // Delta index
            let ind_tmp = ind[k] as i32 + MIN_DELTA_GAIN_QUANT;
            let double_step_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + *prev_ind as i32;
            if ind_tmp > double_step_threshold {
                *prev_ind = (*prev_ind as i32 + (ind_tmp << 1) - double_step_threshold) as i8;
            } else {
                *prev_ind = (*prev_ind as i32 + ind_tmp) as i8;
            }
        }
        *prev_ind = (*prev_ind).clamp(0, (N_LEVELS_QGAIN - 1) as i8);

        gain_q16[k] =
            silk_log2lin((silk_smulwb(INV_SCALE_Q16, *prev_ind as i32) + OFFSET).min(3967));
    }
}

/// Compute unique identifier of gain indices vector
pub fn silk_gains_id(ind: &[i8], nb_subfr: usize) -> i32 {
    let mut gains_id: i32 = 0;
    for item in ind.iter().take(nb_subfr) {
        gains_id = silk_add_lshift32(*item as i32, gains_id, 8);
    }
    gains_id
}
