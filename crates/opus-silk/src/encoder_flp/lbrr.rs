// Port of silk_LBRR_encode_FLP from silk/float/encode_frame_FLP.c
// Low Bitrate Redundancy encoding: re-encodes the frame with boosted gains
// for packet loss resilience. Uses a separate NSQ state copy.

use super::wrappers::silk_nsq_wrapper_flp;
use crate::gain_quant;
use crate::nsq::NsqState;
use crate::*;

/// Speech activity threshold for LBRR (C: LBRR_SPEECH_ACTIVITY_THRES = 0.3)
const LBRR_SPEECH_ACTIVITY_THRES_Q8: i32 = 77; // 0.3 * 256 = 76.8 → 77

/// LBRR encoding state fields that persist across frames.
pub struct LbrrState {
    pub enabled: bool,
    pub gain_increases: i32,
    pub flags: [i32; MAX_FRAMES_PER_PACKET],
    pub indices: [SideInfoIndices; MAX_FRAMES_PER_PACKET],
    pub pulses: [[i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
    pub prev_last_gain_index: i8,
}

impl LbrrState {
    pub fn new() -> Self {
        Self {
            enabled: false,
            gain_increases: 7,
            flags: [0; MAX_FRAMES_PER_PACKET],
            indices: std::array::from_fn(|_| SideInfoIndices::default()),
            pulses: [[0; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
            prev_last_gain_index: 10,
        }
    }

    /// Reset at the start of a new packet (C: nFramesEncoded == 0).
    pub fn reset_for_packet(&mut self, last_gain_index: i8) {
        self.flags = [0; MAX_FRAMES_PER_PACKET];
        self.prev_last_gain_index = last_gain_index;
    }
}

/// Port of silk_LBRR_encode_FLP.
///
/// Re-encodes the current frame at a lower bitrate (higher gains) for
/// redundancy. Uses a cloned NSQ state so the main encoder is unaffected.
///
/// Called after process_gains and before the main NSQ.
#[allow(clippy::too_many_arguments)]
pub fn silk_lbrr_encode_flp(
    lbrr: &mut LbrrState,
    nsq_state: &NsqState,      // main NSQ state (cloned, not modified)
    indices: &SideInfoIndices, // main frame indices
    x_frame: &[f32],           // input signal for NSQ
    // Noise shaping params (from noise_shape analysis)
    pred_coef: &[[f32; MAX_LPC_ORDER]; 2],
    ltp_coef: &[f32],
    ar: &[f32],
    harm_shape_gain: &[f32],
    tilt: &[f32],
    lf_ma_shp: &[f32],
    lf_ar_shp: &[f32],
    gains: &[f32], // original gains from process_gains
    pitch_lags: &[i32; MAX_NB_SUBFR],
    lambda: f32,
    ltp_scale_q14: i32,
    // Config
    frame_length: i32,
    subfr_length: i32,
    ltp_mem_length: i32,
    predict_lpc_order: i32,
    shaping_lpc_order: i32,
    nb_subfr: i32,
    signal_type: i32,
    warping_q16: i32,
    n_states_del_dec: i32,
    cond_coding: i32,
    n_frames_encoded: usize,
    speech_activity_q8: i32,
    // Scratch buffers (separate from main NSQ scratch)
    scratch_s_ltp_q15: &mut [i32],
    scratch_s_ltp: &mut [i16],
    scratch_x_sc_q10: &mut [i32],
    scratch_xq_tmp: &mut [i16],
) {
    // Guard: n_frames_encoded must be within packet bounds
    if n_frames_encoded >= MAX_FRAMES_PER_PACKET {
        return;
    }

    if !lbrr.enabled {
        lbrr.flags[n_frames_encoded] = 0;
        return;
    }

    if speech_activity_q8 <= LBRR_SPEECH_ACTIVITY_THRES_Q8 {
        lbrr.flags[n_frames_encoded] = 0;
        return;
    }

    lbrr.flags[n_frames_encoded] = 1;

    // Clone NSQ state for LBRR (main state unaffected)
    let mut lbrr_nsq = nsq_state.clone();

    // Copy current indices
    let mut lbrr_indices = indices.clone();

    // Save original gains
    let mut lbrr_gains = [0.0f32; MAX_NB_SUBFR];
    lbrr_gains[..nb_subfr as usize].copy_from_slice(&gains[..nb_subfr as usize]);

    // Boost gains for LBRR
    if n_frames_encoded == 0 || lbrr.flags[n_frames_encoded - 1] == 0 {
        // First frame in packet or previous frame not LBRR coded
        lbrr.prev_last_gain_index = lbrr.prev_last_gain_index; // keep current

        lbrr_indices.gains_indices[0] = ((lbrr_indices.gains_indices[0] as i32
            + lbrr.gain_increases)
            .min(N_LEVELS_QGAIN - 1)) as i8;
    }

    // Dequantize to get LBRR gains in sync with decoder
    let mut lbrr_gains_q16 = [0i32; MAX_NB_SUBFR];
    let mut lbrr_prev_idx = lbrr.prev_last_gain_index;
    gain_quant::silk_gains_dequant(
        &mut lbrr_gains_q16,
        &lbrr_indices.gains_indices,
        &mut lbrr_prev_idx,
        cond_coding == CODE_CONDITIONALLY,
        nb_subfr as usize,
    );

    // Convert Q16 gains to float for NSQ wrapper
    for k in 0..nb_subfr as usize {
        lbrr_gains[k] = lbrr_gains_q16[k] as f32 / 65536.0;
    }

    // Run NSQ with LBRR gains
    let mut lbrr_pulses = [0i8; MAX_FRAME_LENGTH];
    silk_nsq_wrapper_flp(
        &mut lbrr_nsq,
        &mut lbrr_indices,
        x_frame,
        &mut lbrr_pulses,
        pred_coef,
        ltp_coef,
        ar,
        harm_shape_gain,
        tilt,
        lf_ma_shp,
        lf_ar_shp,
        &lbrr_gains,
        pitch_lags,
        lambda,
        ltp_scale_q14,
        frame_length,
        subfr_length,
        ltp_mem_length,
        predict_lpc_order,
        shaping_lpc_order,
        nb_subfr,
        signal_type,
        warping_q16,
        n_states_del_dec,
        scratch_s_ltp_q15,
        scratch_s_ltp,
        scratch_x_sc_q10,
        scratch_xq_tmp,
    );

    // Store LBRR results
    lbrr.indices[n_frames_encoded] = lbrr_indices;
    let fl = frame_length as usize;
    lbrr.pulses[n_frames_encoded][..fl].copy_from_slice(&lbrr_pulses[..fl]);
}
