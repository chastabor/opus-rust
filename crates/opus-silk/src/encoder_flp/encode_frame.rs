// Port of silk/float/encode_frame_FLP.c: silk_encode_frame_FLP
// Top-level per-frame encoder that orchestrates all analysis and quantization.
// This is the heart of the SILK encoder.

use opus_range_coder::EcCtx;
use super::wrappers::*;
use super::noise_shape::*;
use super::find_pred_coefs::silk_find_pred_coefs_flp;
use super::process_gains::silk_process_gains_flp;
use crate::*;
use crate::nsq::NsqState;
use crate::encode_indices;
use crate::encode_pulses;

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

    // ---- Step 2: Pitch analysis (simplified: unvoiced for now) ----
    // TODO: Port silk_find_pitch_lags_FLP for voiced detection
    let pitch_lags = [0i32; MAX_NB_SUBFR];
    indices.signal_type = TYPE_UNVOICED as i8;
    indices.quant_offset_type = 1; // unvoiced default

    // ---- Step 3: Noise shape analysis ----
    // C: x_frame = x_buf + ltp_mem_length. noise_shape receives x_frame and accesses
    // x - la_shape internally. We pass x starting at x_frame (= x_buf + ltp_mem).
    let x_frame = &x_buf[x_frame_offset..];

    // For unvoiced, pitch_res = input signal (C uses LPC residual from pitch analysis)
    // Convert the frame portion to float for the sparseness measure
    let pitch_res_start = la_shape;
    let pitch_res_end = (la_shape + frame_len).min(x_frame.len());
    let pitch_res = &x_frame[pitch_res_start..pitch_res_end];

    let ns_result = silk_noise_shape_analysis_flp(
        x_frame,                       // signal with la_shape lookback
        pitch_res,                     // pitch residual (= input for unvoiced)
        &pitch_lags,                   // pitch lags per subframe
        indices.signal_type as i32,    // signal type
        snr_db_q7,
        speech_activity_q8,
        input_quality_bands_q15,
        0.0,                           // ltp_corr (0 for unvoiced)
        0.0,                           // pred_gain (0 before pitch analysis)
        false,                         // use_cbr
        la_shape,
        fs_khz,
        nb,
        sfr_len,
        shape_win_length as usize,     // shapeWinLength
        shaping_order,                 // shapingLPCOrder
        warping_q16,
        prev_harm_smth,
        prev_tilt_smth,
    );

    // ---- Step 4: Find prediction coefficients ----
    let use_interpolated = complexity >= 5;
    let n_survivors = match complexity {
        0 => 2, 1 => 3, 2 => 2, 3 => 4, 4 | 5 => 6, 6 | 7 => 8, _ => 16,
    } as i32;

    let mut pred_coef = [[0.0f32; MAX_LPC_ORDER]; 2];
    let mut res_nrg = [0.0f32; MAX_NB_SUBFR];
    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];

    // x for find_pred_coefs: starts at x_frame + la_shape (the actual frame)
    let x_for_pred = &x_buf[x_frame_offset + la_shape..];

    silk_find_pred_coefs_flp(
        x_for_pred,
        &ns_result.gains,
        ns_result.coding_quality,
        0.0, // ltp_pred_cod_gain (unvoiced)
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
    // TODO: track n_frames_encoded for proper cond_coding

    let lambda = silk_process_gains_flp(
        &mut gains,
        &res_nrg,
        indices,
        last_gain_index,
        snr_db_q7,
        nb,
        indices.signal_type as i32,
        0.0, // ltp_pred_cod_gain
        match complexity { 0 | 1 => 1, 2..=5 => 2, 6 | 7 => 3, _ => 4 },
        speech_activity_q8,
        ns_result.input_quality,
        ns_result.coding_quality,
        cond_coding,
        input_tilt_q15,
        sfr_len,
    );

    indices.quant_offset_type = ns_result.quant_offset_type;
    indices.seed = 0; // TODO: frame counter & 3

    // ---- Step 6: NSQ (via float wrapper) ----
    let mut pulses = [0i8; MAX_FRAME_LENGTH];
    let n_states_del_dec = match complexity {
        0 | 1 => 1i32, 2..=5 => 2, 6 | 7 => 3, _ => 4,
    };

    // LTP coefficients (zero for unvoiced)
    let ltp_coef = [0.0f32; MAX_NB_SUBFR * LTP_ORDER];

    silk_nsq_wrapper_flp(
        nsq_state,
        indices,
        x_for_pred,
        &mut pulses,
        &pred_coef,
        &ltp_coef,
        &ns_result.ar,
        &ns_result.harm_shape_gain,
        &ns_result.tilt,
        &ns_result.lf_ma_shp,
        &ns_result.lf_ar_shp,
        &gains,
        &pitch_lags,
        lambda,
        0, // ltp_scale_q14 (0 for unvoiced)
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
    encode_indices::silk_encode_indices(
        indices, enc, 0, false, cond_coding_int,
        nb_subfr,
        if fs_khz <= 12 { NlsfCbSel::NbMb } else { NlsfCbSel::Wb },
        PitchContourSel::Nb, // TODO: proper pitch contour
        PitchLagLowBitsSel::Uniform4,
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
