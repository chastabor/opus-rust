// Port of silk/float/find_LPC_FLP.c: silk_find_LPC_FLP
// Burg LPC analysis with NLSF interpolation coefficient search.

use super::dsp::{silk_lpc_analysis_filter_flp, silk_energy_flp};
use super::wrappers::{silk_a2nlsf_flp, silk_nlsf2a_flp};
use crate::lpc_analysis::silk_burg_modified_flp;
use crate::{MAX_LPC_ORDER, MAX_NB_SUBFR, silk_interpolate_i16};

/// Find LPC coefficients via Burg analysis with optional NLSF interpolation.
///
/// Port of silk_find_LPC_FLP (find_LPC_FLP.c).
///
/// Sets `nlsf_interp_coef_q2` on the output and fills `nlsf_q15`.
pub fn silk_find_lpc_flp(
    nlsf_q15: &mut [i16],                 // O: NLSFs in Q15
    nlsf_interp_coef_q2: &mut i8,         // O: interpolation coefficient (0-4)
    x: &[f32],                            // I: input signal (gain-normalized lpc_in_pre)
    min_inv_gain: f32,                     // I: minimum inverse prediction gain
    predict_lpc_order: usize,
    nb_subfr: usize,
    subfr_length_with_d: usize,           // subfr_length + predictLPCOrder
    use_interpolated_nlsfs: bool,
    first_frame_after_reset: bool,
    prev_nlsf_q15: &[i16],               // I: previous frame's quantized NLSFs
) {
    let mut a = [0.0f32; MAX_LPC_ORDER];

    // Default: no interpolation
    *nlsf_interp_coef_q2 = 4;

    // Burg AR analysis for the full frame
    let mut res_nrg = silk_burg_modified_flp(
        &mut a, x, min_inv_gain, subfr_length_with_d, nb_subfr, predict_lpc_order,
    );

    if use_interpolated_nlsfs && !first_frame_after_reset && nb_subfr == MAX_NB_SUBFR {
        // Optimal solution for last 10ms
        let mut a_tmp = [0.0f32; MAX_LPC_ORDER];
        let half_offset = (MAX_NB_SUBFR / 2) * subfr_length_with_d;
        res_nrg -= silk_burg_modified_flp(
            &mut a_tmp, &x[half_offset..], min_inv_gain,
            subfr_length_with_d, MAX_NB_SUBFR / 2, predict_lpc_order,
        );

        // Convert second-half LPC to NLSFs
        silk_a2nlsf_flp(nlsf_q15, &a_tmp, predict_lpc_order);

        // Search over interpolation indices
        let mut res_nrg_2nd = f32::MAX;
        const MAX_FILTER_LEN: usize = 2 * (MAX_LPC_ORDER + crate::MAX_SUB_FRAME_LENGTH);
        let filter_len = 2 * subfr_length_with_d;
        let mut lpc_res = [0.0f32; MAX_FILTER_LEN];

        for k in (0..=3i32).rev() {
            // Interpolate NLSFs for first half
            let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
            silk_interpolate_i16(
                &mut nlsf0_q15, prev_nlsf_q15, nlsf_q15,
                k, predict_lpc_order,
            );

            // Convert to float LPC for residual energy evaluation
            let mut a_interp = [0.0f32; MAX_LPC_ORDER];
            silk_nlsf2a_flp(&mut a_interp, &nlsf0_q15, predict_lpc_order);

            // Calculate residual energy with NLSF interpolation
            silk_lpc_analysis_filter_flp(&mut lpc_res, &a_interp, x, filter_len, predict_lpc_order);

            // C: energy(LPC_res + predictLPCOrder, subfr_length_with_d - predictLPCOrder) per half
            let range_len = subfr_length_with_d - predict_lpc_order;
            let nrg0 = silk_energy_flp(
                &lpc_res[predict_lpc_order..predict_lpc_order + range_len],
            );
            let nrg1 = silk_energy_flp(
                &lpc_res[predict_lpc_order + subfr_length_with_d
                    ..predict_lpc_order + subfr_length_with_d + range_len],
            );
            let res_nrg_interp = (nrg0 + nrg1) as f32;

            // Determine whether current interpolated NLSFs are best so far
            if res_nrg_interp < res_nrg {
                res_nrg = res_nrg_interp;
                *nlsf_interp_coef_q2 = k as i8;
            } else if res_nrg_interp > res_nrg_2nd {
                // No reason to continue — residual energies will climb
                break;
            }
            res_nrg_2nd = res_nrg_interp;
        }
    }

    if *nlsf_interp_coef_q2 == 4 {
        // No interpolation — convert full-frame Burg LPC to NLSFs
        silk_a2nlsf_flp(nlsf_q15, &a, predict_lpc_order);
    }
}

