// Port of silk/float/encode_frame_FLP.c: silk_encode_frame_FLP
// Top-level per-frame encoder that orchestrates all analysis and quantization.
// This is the heart of the SILK encoder.

use opus_range_coder::EcCtx;
use super::wrappers::*;
use super::noise_shape::*;
use super::find_pitch_lags::silk_find_pitch_lags_flp;
use super::find_pred_coefs::silk_find_pred_coefs_flp;
use super::process_gains::silk_process_gains_flp;
use crate::*;
use crate::nsq::NsqState;
use crate::encode_indices;
use crate::encode_pulses;
use crate::tables::SILK_LTP_SCALES_TABLE_Q14;

use crate::LA_SHAPE_MS;

/// Encode one SILK frame using the float analysis pipeline.
///
/// Port of silk_encode_frame_FLP (encode_frame_FLP.c).
///
/// Returns the number of payload bytes written.
pub fn silk_encode_frame_flp(
    // Persistent encoder state
    x_buf: &mut [f32],                      // I/O: float analysis buffer (ltp_mem + la_shape + frame)
    nsq_state: &mut NsqState,
    indices: &mut SideInfoIndices,
    prev_nlsf_q15: &mut [i16],
    prev_signal_type: &mut i32,
    prev_lag: &mut i32,
    first_frame_after_reset: &mut bool,
    last_gain_index: &mut i8,
    prev_harm_smth: &mut f32,
    prev_tilt_smth: &mut f32,
    speech_activity_q8: i32,
    input_quality_bands_q15: &[i32],
    input_tilt_q15: i32,
    snr_db_q7: i32,
    // Input signal (i16 from inputBuf)
    input_buf: &[i16],
    // Config
    fs_khz: i32,
    nb_subfr: i32,
    subfr_length: i32,
    frame_length: i32,
    ltp_mem_length: i32,
    predict_lpc_order: i32,
    shaping_lpc_order: i32,
    shape_win_length: i32,
    la_pitch: i32,
    pitch_lpc_win_length: i32,
    pitch_estimation_lpc_order: i32,
    warping_q16: i32,
    complexity: i32,
    nlsf_cb: &NlsfCbStruct,
    _max_bits: i32,
    // Range coder
    enc: &mut EcCtx,
    // Scratch buffers
    scratch_s_ltp_q15: &mut [i32],
    scratch_s_ltp: &mut [i16],
    scratch_x_sc_q10: &mut [i32],
    scratch_xq_tmp: &mut [i16],
) -> i32 {
    let nb = nb_subfr as usize;
    let sfr_len = subfr_length as usize;
    let frame_len = frame_length as usize;
    let ltp_mem = ltp_mem_length as usize;
    let lpc_order = predict_lpc_order as usize;
    let shaping_order = shaping_lpc_order as usize;
    let la_shape = LA_SHAPE_MS * fs_khz as usize;

    // ---- Step 1: Copy new frame into x_buf ----
    let x_frame_offset = ltp_mem; // x_frame = x_buf + ltp_mem
    // Copy i16 input to float at x_frame + la_shape
    let dst_start = x_frame_offset + la_shape;
    for i in 0..frame_len.min(input_buf.len()) {
        if dst_start + i < x_buf.len() {
            x_buf[dst_start + i] = input_buf[i] as f32;
        }
    }

    // ---- Step 2: Pitch analysis (C: silk_find_pitch_lags_FLP) ----
    let pitch_result = silk_find_pitch_lags_flp(
        x_buf,
        ltp_mem,
        frame_len,
        la_pitch as usize,
        pitch_lpc_win_length as usize,
        pitch_estimation_lpc_order as usize,
        fs_khz,
        nb,
        complexity,
        *prev_lag,
        *prev_signal_type,
        speech_activity_q8,
        input_tilt_q15,
        *first_frame_after_reset,
        0.0, // prev_ltp_corr — TODO: persist across frames
    );
    let pitch_lags = pitch_result.pitch_l;
    indices.signal_type = pitch_result.signal_type as i8;
    indices.lag_index = pitch_result.lag_index;
    indices.contour_index = pitch_result.contour_index;
    let ltp_corr = pitch_result.ltp_corr;
    let pred_gain = pitch_result.pred_gain;

    // ---- Step 3: Noise shape analysis ----
    let x_frame = &x_buf[x_frame_offset..];

    // For noise_shape: pitch_res starts at ltp_mem_length into the residual
    // (the frame portion of the LPC residual)
    let pitch_res_for_ns = if pitch_result.res_pitch.len() > ltp_mem {
        &pitch_result.res_pitch[ltp_mem..pitch_result.res_pitch.len().min(ltp_mem + frame_len)]
    } else {
        // Fallback: use input signal
        &x_frame[la_shape..(la_shape + frame_len).min(x_frame.len())]
    };

    let ns_result = silk_noise_shape_analysis_flp(
        x_frame,
        pitch_res_for_ns,
        &pitch_lags,
        indices.signal_type as i32,
        snr_db_q7,
        speech_activity_q8,
        input_quality_bands_q15,
        ltp_corr,
        pred_gain,
        false, // use_cbr
        la_shape,
        fs_khz,
        nb,
        sfr_len,
        shape_win_length as usize,
        shaping_order,
        warping_q16,
        prev_harm_smth,
        prev_tilt_smth,
    );

    // ---- Step 4: Find prediction coefficients (with LTP for voiced) ----
    let use_interpolated = complexity >= 5;
    let n_survivors = match complexity {
        0 => 2, 1 => 3, 2 => 2, 3 => 4, 4 | 5 => 6, 6 | 7 => 8, _ => 16,
    } as i32;

    let mut pred_coef = [[0.0f32; MAX_LPC_ORDER]; 2];
    let mut res_nrg = [0.0f32; MAX_NB_SUBFR];
    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];

    // Pass full x_buf and offsets so find_pred_coefs can access LTP history
    let pred_result = silk_find_pred_coefs_flp(
        x_buf,
        x_frame_offset,
        la_shape,
        &pitch_result.res_pitch,
        ltp_mem, // frame starts at this offset in res_pitch
        &pitch_lags,
        &ns_result.gains,
        ns_result.coding_quality,
        indices.signal_type as i32,
        0, // sum_log_gain_q7 — TODO: persist across frames
        indices,
        prev_nlsf_q15,
        lpc_order,
        sfr_len,
        nb,
        *first_frame_after_reset,
        use_interpolated,
        speech_activity_q8,
        nlsf_cb,
        n_survivors,
        &mut pred_coef,
        &mut res_nrg,
        &mut nlsf_q15,
    );

    // ---- Step 5: Process gains ----
    let mut gains = ns_result.gains;
    let cond_coding = !*first_frame_after_reset && false; // first encode in packet: independent
    let n_states_del_dec = match complexity {
        0 | 1 => 1i32, 2..=5 => 2, 6 | 7 => 3, _ => 4,
    };

    let lambda = silk_process_gains_flp(
        &mut gains,
        &res_nrg,
        indices,
        last_gain_index,
        snr_db_q7,
        nb,
        indices.signal_type as i32,
        pred_result.ltp_pred_cod_gain,
        n_states_del_dec,
        speech_activity_q8,
        ns_result.input_quality,
        ns_result.coding_quality,
        cond_coding,
        input_tilt_q15,
        sfr_len,
    );

    indices.quant_offset_type = ns_result.quant_offset_type;
    indices.seed = 0; // TODO: frame counter & 3

    // LTP scale: minimum scaling (full implementation needs packet loss rate)
    indices.ltp_scale_index = 0;
    let ltp_scale_q14 = if indices.signal_type == TYPE_VOICED as i8 {
        SILK_LTP_SCALES_TABLE_Q14[0] as i32
    } else {
        0
    };

    // ---- Step 6: NSQ (via float wrapper) ----
    let mut pulses = [0i8; MAX_FRAME_LENGTH];
    let x_for_nsq = &x_buf[x_frame_offset + la_shape..];

    silk_nsq_wrapper_flp(
        nsq_state,
        indices,
        x_for_nsq,
        &mut pulses,
        &pred_coef,
        &pred_result.ltp_coef,
        &ns_result.ar,
        &ns_result.harm_shape_gain,
        &ns_result.tilt,
        &ns_result.lf_ma_shp,
        &ns_result.lf_ar_shp,
        &gains,
        &pitch_lags,
        lambda,
        ltp_scale_q14,
        frame_length,
        subfr_length,
        ltp_mem_length,
        predict_lpc_order,
        shaping_lpc_order,
        nb_subfr,
        indices.signal_type as i32,
        warping_q16,
        n_states_del_dec,
        scratch_s_ltp_q15,
        scratch_s_ltp,
        scratch_x_sc_q10,
        scratch_xq_tmp,
    );

    // ---- Step 7: Encode indices + pulses ----
    let cond_coding_int = if cond_coding { CODE_CONDITIONALLY } else { CODE_INDEPENDENTLY };

    // Select proper pitch contour table based on fs_khz and nb_subfr
    // C: fs_kHz == 8 → NB tables, else WB tables; type_offset == 2 → 10ms
    let pitch_contour_sel = match (fs_khz == 8, nb_subfr == 2) {
        (true, true)   => PitchContourSel::Nb10ms,
        (true, false)  => PitchContourSel::Nb,
        (false, true)  => PitchContourSel::Wb10ms,
        (false, false) => PitchContourSel::Wb,
    };

    encode_indices::silk_encode_indices(
        indices, enc, 0, false, cond_coding_int,
        nb_subfr,
        if fs_khz <= 12 { NlsfCbSel::NbMb } else { NlsfCbSel::Wb },
        pitch_contour_sel,
        match fs_khz >> 1 {
            4 => PitchLagLowBitsSel::Uniform4,
            6 => PitchLagLowBitsSel::Uniform6,
            _ => PitchLagLowBitsSel::Uniform8,
        },
        fs_khz,
        *prev_signal_type,
        0, // prev_lag_index
    );
    encode_pulses::silk_encode_pulses(
        enc, &pulses, indices.signal_type as i32,
        indices.quant_offset_type as i32, frame_length,
    );

    // ---- Step 8: Update state for next frame ----
    // Shift x_buf: discard oldest frame, keep history
    let keep_len = ltp_mem + la_shape;
    if keep_len + frame_len <= x_buf.len() {
        x_buf.copy_within(frame_len..frame_len + keep_len, 0);
    }

    *prev_signal_type = indices.signal_type as i32;
    *prev_lag = pitch_lags[nb - 1];
    *first_frame_after_reset = false;

    // Return payload bytes
    (enc.tell() + 7) >> 3
}
