// Port of silk/encode_indices.c - Encode side-information parameters to payload
// This is the mirror of decode_indices.rs: every dec_icdf becomes enc_icdf.

use crate::nlsf::nlsf_unpack;
use crate::tables::*;
use crate::*;
use opus_range_coder::EcCtx;

/// Encode side-information parameters to payload.
/// Mirror of silk_decode_indices -- produces a bitstream the decoder can read.
#[allow(clippy::too_many_arguments)]
pub fn silk_encode_indices(
    indices: &SideInfoIndices,
    enc: &mut EcCtx,
    _frame_index: i32,
    encode_lbrr: bool,
    cond_coding: i32,
    nb_subfr: i32,
    nlsf_cb_sel: NlsfCbSel,
    pitch_contour_icdf_sel: PitchContourSel,
    pitch_lag_low_bits_sel: PitchLagLowBitsSel,
    fs_khz: i32,
    ec_prev_signal_type: i32,
    ec_prev_lag_index: i16,
) {
    // Encode signal type and quantizer offset
    let type_offset = 2 * indices.signal_type as i32 + indices.quant_offset_type as i32;
    if encode_lbrr || type_offset >= 2 {
        enc.enc_icdf((type_offset - 2) as usize, &SILK_TYPE_OFFSET_VAD_ICDF, 8);
    } else {
        enc.enc_icdf(type_offset as usize, &SILK_TYPE_OFFSET_NO_VAD_ICDF, 8);
    }

    // Encode gains
    if cond_coding == CODE_CONDITIONALLY {
        // Conditional coding: delta gain
        enc.enc_icdf(indices.gains_indices[0] as usize, &SILK_DELTA_GAIN_ICDF, 8);
    } else {
        // Independent coding: MSB bits followed by 3 LSBs
        let gain_val = indices.gains_indices[0] as i32;
        enc.enc_icdf(
            (gain_val >> 3) as usize,
            &SILK_GAIN_ICDF[indices.signal_type as usize],
            8,
        );
        enc.enc_icdf((gain_val & 7) as usize, &SILK_UNIFORM8_ICDF, 8);
    }

    // Remaining subframes: delta gains
    for i in 1..nb_subfr as usize {
        enc.enc_icdf(indices.gains_indices[i] as usize, &SILK_DELTA_GAIN_ICDF, 8);
    }

    // Encode NLSF indices
    let nlsf_cb = get_nlsf_cb(nlsf_cb_sel);
    let signal_type_half = (indices.signal_type >> 1) as usize;
    let n_vectors = nlsf_cb.n_vectors as usize;
    let order = nlsf_cb.order as usize;

    enc.enc_icdf(
        indices.nlsf_indices[0] as usize,
        &nlsf_cb.cb1_icdf[signal_type_half * n_vectors..],
        8,
    );

    // Unpack entropy table indices and predictor
    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        nlsf_cb,
        indices.nlsf_indices[0] as usize,
    );

    for (i, ec_ix_item) in ec_ix.iter().enumerate().take(order) {
        let nlsf_idx = indices.nlsf_indices[i + 1] as i32;
        if nlsf_idx >= NLSF_QUANT_MAX_AMPLITUDE {
            enc.enc_icdf(
                (2 * NLSF_QUANT_MAX_AMPLITUDE) as usize,
                &nlsf_cb.ec_icdf[*ec_ix_item as usize..],
                8,
            );
            enc.enc_icdf(
                (nlsf_idx - NLSF_QUANT_MAX_AMPLITUDE) as usize,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else if nlsf_idx <= -NLSF_QUANT_MAX_AMPLITUDE {
            enc.enc_icdf(0, &nlsf_cb.ec_icdf[*ec_ix_item as usize..], 8);
            enc.enc_icdf(
                (-nlsf_idx - NLSF_QUANT_MAX_AMPLITUDE) as usize,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else {
            enc.enc_icdf(
                (nlsf_idx + NLSF_QUANT_MAX_AMPLITUDE) as usize,
                &nlsf_cb.ec_icdf[*ec_ix_item as usize..],
                8,
            );
        }
    }

    // Encode NLSF interpolation factor
    if nb_subfr == MAX_NB_SUBFR as i32 {
        enc.enc_icdf(
            indices.nlsf_interp_coef_q2 as usize,
            &SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
            8,
        );
    }

    if indices.signal_type as i32 == TYPE_VOICED {
        // Encode pitch lags
        let mut encode_absolute_lag_index = true;
        if cond_coding == CODE_CONDITIONALLY && ec_prev_signal_type == TYPE_VOICED {
            let delta_lag_index = indices.lag_index as i32 - ec_prev_lag_index as i32;
            if (-8..=11).contains(&delta_lag_index) {
                // Delta encoding
                enc.enc_icdf((delta_lag_index + 9) as usize, &SILK_PITCH_DELTA_ICDF, 8);
                encode_absolute_lag_index = false;
            } else {
                // Signal absolute coding by sending delta_lagIndex = 0
                enc.enc_icdf(0, &SILK_PITCH_DELTA_ICDF, 8);
            }
        }

        if encode_absolute_lag_index {
            // Absolute encoding: split into high bits and low bits
            let pitch_high_bits = indices.lag_index as i32 / (fs_khz >> 1);
            let pitch_low_bits = indices.lag_index as i32 - pitch_high_bits * (fs_khz >> 1);

            enc.enc_icdf(pitch_high_bits as usize, &SILK_PITCH_LAG_ICDF, 8);

            let pitch_lag_low_bits_icdf = match pitch_lag_low_bits_sel {
                PitchLagLowBitsSel::Uniform4 => &SILK_UNIFORM4_ICDF[..],
                PitchLagLowBitsSel::Uniform6 => &SILK_UNIFORM6_ICDF[..],
                PitchLagLowBitsSel::Uniform8 => &SILK_UNIFORM8_ICDF[..],
            };
            enc.enc_icdf(pitch_low_bits as usize, pitch_lag_low_bits_icdf, 8);
        }

        // Encode pitch contour
        let pitch_contour_icdf = match pitch_contour_icdf_sel {
            PitchContourSel::Nb => &SILK_PITCH_CONTOUR_NB_ICDF[..],
            PitchContourSel::Wb => &SILK_PITCH_CONTOUR_ICDF[..],
            PitchContourSel::Nb10ms => &SILK_PITCH_CONTOUR_10_MS_NB_ICDF[..],
            PitchContourSel::Wb10ms => &SILK_PITCH_CONTOUR_10_MS_ICDF[..],
        };
        enc.enc_icdf(indices.contour_index as usize, pitch_contour_icdf, 8);

        // Encode LTP gains
        enc.enc_icdf(indices.per_index as usize, &SILK_LTP_PER_INDEX_ICDF, 8);
        for k in 0..nb_subfr as usize {
            enc.enc_icdf(
                indices.ltp_index[k] as usize,
                SILK_LTP_GAIN_ICDF_PTRS[indices.per_index as usize],
                8,
            );
        }

        // Encode LTP scaling
        if cond_coding == CODE_INDEPENDENTLY {
            enc.enc_icdf(indices.ltp_scale_index as usize, &SILK_LTPSCALE_ICDF, 8);
        }
    }

    // Encode seed
    enc.enc_icdf(indices.seed as usize, &SILK_UNIFORM4_ICDF, 8);
}
