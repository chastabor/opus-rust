// Port of silk/decode_indices.c

use crate::nlsf::nlsf_unpack;
use crate::tables::*;
use crate::*;
use opus_range_coder::EcCtx;

/// Decode side-information parameters from payload
pub fn silk_decode_indices(
    ps_dec: &mut ChannelState,
    ps_range_dec: &mut EcCtx,
    frame_index: i32,
    decode_lbrr: bool,
    cond_coding: i32,
) {
    // Decode signal type and quantizer offset
    let ix = if decode_lbrr || ps_dec.vad_flags[frame_index as usize] != 0 {
        ps_range_dec.dec_icdf(&SILK_TYPE_OFFSET_VAD_ICDF, 8) as i32 + 2
    } else {
        ps_range_dec.dec_icdf(&SILK_TYPE_OFFSET_NO_VAD_ICDF, 8) as i32
    };
    ps_dec.indices.signal_type = (ix >> 1) as i8;
    ps_dec.indices.quant_offset_type = (ix & 1) as i8;

    // Decode gains
    if cond_coding == CODE_CONDITIONALLY {
        ps_dec.indices.gains_indices[0] = ps_range_dec.dec_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    } else {
        ps_dec.indices.gains_indices[0] = ((ps_range_dec
            .dec_icdf(&SILK_GAIN_ICDF[ps_dec.indices.signal_type as usize], 8)
            as i8)
            << 3)
            .wrapping_add(ps_range_dec.dec_icdf(&SILK_UNIFORM8_ICDF, 8) as i8);
    }

    for i in 1..ps_dec.nb_subfr as usize {
        ps_dec.indices.gains_indices[i] = ps_range_dec.dec_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    }

    // Decode LSF indices
    let nlsf_cb = get_nlsf_cb(ps_dec.nlsf_cb);
    let signal_type_half = (ps_dec.indices.signal_type >> 1) as usize;
    let n_vectors = nlsf_cb.n_vectors as usize;
    let order = nlsf_cb.order as usize;

    ps_dec.indices.nlsf_indices[0] =
        ps_range_dec.dec_icdf(&nlsf_cb.cb1_icdf[signal_type_half * n_vectors..], 8) as i8;

    // Unpack entropy table indices and predictor
    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        nlsf_cb,
        ps_dec.indices.nlsf_indices[0] as usize,
    );

    for (i, &ec_ix_i) in ec_ix.iter().enumerate().take(order) {
        let mut ix_val = ps_range_dec.dec_icdf(&nlsf_cb.ec_icdf[ec_ix_i as usize..], 8) as i32;
        if ix_val == 0 {
            ix_val -= ps_range_dec.dec_icdf(&SILK_NLSF_EXT_ICDF, 8) as i32;
        } else if ix_val == 2 * NLSF_QUANT_MAX_AMPLITUDE {
            ix_val += ps_range_dec.dec_icdf(&SILK_NLSF_EXT_ICDF, 8) as i32;
        }
        ps_dec.indices.nlsf_indices[i + 1] = (ix_val - NLSF_QUANT_MAX_AMPLITUDE) as i8;
    }

    // Decode LSF interpolation factor
    if ps_dec.nb_subfr == MAX_NB_SUBFR as i32 {
        ps_dec.indices.nlsf_interp_coef_q2 =
            ps_range_dec.dec_icdf(&SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8;
    } else {
        ps_dec.indices.nlsf_interp_coef_q2 = 4;
    }

    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        // Decode pitch lags
        let mut decode_absolute_lag_index = true;
        if cond_coding == CODE_CONDITIONALLY && ps_dec.ec_prev_signal_type == TYPE_VOICED {
            let delta_lag_index = ps_range_dec.dec_icdf(&SILK_PITCH_DELTA_ICDF, 8) as i32;
            if delta_lag_index > 0 {
                let delta = delta_lag_index - 9;
                ps_dec.indices.lag_index = (ps_dec.ec_prev_lag_index as i32 + delta) as i16;
                decode_absolute_lag_index = false;
            }
        }
        if decode_absolute_lag_index {
            ps_dec.indices.lag_index = (ps_range_dec.dec_icdf(&SILK_PITCH_LAG_ICDF, 8) as i32
                * (ps_dec.fs_khz >> 1)) as i16;
            ps_dec.indices.lag_index +=
                ps_range_dec.dec_icdf(ps_dec.get_pitch_lag_low_bits_icdf(), 8) as i16;
        }
        ps_dec.ec_prev_lag_index = ps_dec.indices.lag_index;

        // Decode pitch contour
        ps_dec.indices.contour_index =
            ps_range_dec.dec_icdf(ps_dec.get_pitch_contour_icdf(), 8) as i8;

        // Decode LTP gains
        ps_dec.indices.per_index = ps_range_dec.dec_icdf(&SILK_LTP_PER_INDEX_ICDF, 8) as i8;

        for k in 0..ps_dec.nb_subfr as usize {
            ps_dec.indices.ltp_index[k] = ps_range_dec.dec_icdf(
                SILK_LTP_GAIN_ICDF_PTRS[ps_dec.indices.per_index as usize],
                8,
            ) as i8;
        }

        // Decode LTP scaling
        if cond_coding == CODE_INDEPENDENTLY {
            ps_dec.indices.ltp_scale_index = ps_range_dec.dec_icdf(&SILK_LTPSCALE_ICDF, 8) as i8;
        } else {
            ps_dec.indices.ltp_scale_index = 0;
        }
    }
    ps_dec.ec_prev_signal_type = ps_dec.indices.signal_type as i32;

    // Decode seed
    ps_dec.indices.seed = ps_range_dec.dec_icdf(&SILK_UNIFORM4_ICDF, 8) as i8;
}
