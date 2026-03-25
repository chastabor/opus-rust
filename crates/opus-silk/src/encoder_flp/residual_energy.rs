// Port of silk/float/residual_energy_FLP.c: silk_residual_energy_FLP
// Computes per-subframe residual energy using quantized LPC and gain scaling.

use super::dsp::{silk_lpc_analysis_filter_flp, silk_energy_flp};
use crate::{MAX_NB_SUBFR, MAX_LPC_ORDER, MAX_SUB_FRAME_LENGTH};

/// Compute per-subframe residual energies.
/// Uses a[0] for subframes 0-1, a[1] for subframes 2-3 (if nb_subfr == 4).
/// Each energy is scaled by gains[k]^2.
///
/// Port of silk_residual_energy_FLP (residual_energy_FLP.c).
pub fn silk_residual_energy_flp(
    nrgs: &mut [f32; MAX_NB_SUBFR],
    x: &[f32],                            // gain-normalized input (lpc_in_pre)
    a: &[[f32; MAX_LPC_ORDER]; 2],        // LPC coefficients [2 halves][order]
    gains: &[f32],                         // per-subframe gains
    subfr_length: usize,
    nb_subfr: usize,
    lpc_order: usize,
) {
    let shift = lpc_order + subfr_length;  // offset per subframe in x

    // First half (subframes 0-1): filter with a[0]
    // Stack-allocated: max 2*(16+80) = 192 floats = 768 bytes
    const MAX_FILTER_LEN: usize = 2 * (MAX_LPC_ORDER + MAX_SUB_FRAME_LENGTH);
    let filter_len = 2 * shift;
    let mut lpc_res = [0.0f32; MAX_FILTER_LEN];
    if filter_len <= x.len() {
        silk_lpc_analysis_filter_flp(&mut lpc_res, &a[0], x, filter_len, lpc_order);
    }

    let lpc_res_offset = lpc_order;
    nrgs[0] = gains[0] * gains[0]
        * silk_energy_flp(&lpc_res[lpc_res_offset..lpc_res_offset + subfr_length]) as f32;
    nrgs[1] = gains[1] * gains[1]
        * silk_energy_flp(&lpc_res[lpc_res_offset + shift..lpc_res_offset + shift + subfr_length]) as f32;

    // Second half (subframes 2-3): filter with a[1]
    if nb_subfr == 4 {
        let x_offset = 2 * shift;
        let mut lpc_res2 = [0.0f32; MAX_FILTER_LEN];
        if x_offset + filter_len <= x.len() {
            silk_lpc_analysis_filter_flp(
                &mut lpc_res2, &a[1], &x[x_offset..], filter_len, lpc_order,
            );
        }
        nrgs[2] = gains[2] * gains[2]
            * silk_energy_flp(&lpc_res2[lpc_res_offset..lpc_res_offset + subfr_length]) as f32;
        nrgs[3] = gains[3] * gains[3]
            * silk_energy_flp(&lpc_res2[lpc_res_offset + shift..lpc_res_offset + shift + subfr_length]) as f32;
    }
}
