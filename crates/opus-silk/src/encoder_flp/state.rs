// Float SILK encoder persistent state.
// Mirrors silk_encoder_state_FLP from silk/float/structs_FLP.h.

use crate::*;
use crate::nsq::NsqState;
use crate::vad;
use super::lbrr::LbrrState;

use crate::LA_SHAPE_MS;
const MAX_SILK_X_BUF: usize = 2 * MAX_FRAME_LENGTH + LA_SHAPE_MS * MAX_FS_KHZ;

/// Persistent state for the float SILK encoder (one per channel).
pub struct SilkEncoderStateFlp {
    // Configuration (set by init/set_fs)
    pub fs_khz: i32,
    pub nb_subfr: i32,
    pub frame_length: i32,
    pub subfr_length: i32,
    pub ltp_mem_length: i32,
    pub predict_lpc_order: i32,
    pub shaping_lpc_order: i32,
    pub shape_win_length: i32,
    pub la_pitch: i32,
    pub pitch_lpc_win_length: i32,
    pub pitch_estimation_lpc_order: i32,
    pub warping_q16: i32,
    pub nlsf_cb_sel: NlsfCbSel,

    // Float analysis buffer (C: x_buf)
    pub x_buf: Vec<f32>,

    // NSQ state (fixed-point, used via wrapper)
    pub nsq_state: NsqState,

    // Side info indices
    pub indices: SideInfoIndices,

    // Previous-frame memory
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub prev_signal_type: i32,
    pub prev_lag: i32,
    pub first_frame_after_reset: bool,
    pub last_gain_index: i8,

    // Noise shape smoothing state
    pub prev_harm_smth: f32,
    pub prev_tilt_smth: f32,

    // VAD state
    pub vad_state: vad::VadState,
    pub speech_activity_q8: i32,
    pub input_quality_bands_q15: [i32; 4],
    pub input_tilt_q15: i32,
    pub snr_db_q7: i32,

    // Frame counter
    pub n_frames_encoded: i32,
    pub n_frames_per_packet: i32,

    // Packet loss / LBRR config
    pub packet_loss_perc: i32,
    pub lbrr: LbrrState,

    // Scratch buffers (avoid per-frame allocation)
    pub scratch_s_ltp_q15: Vec<i32>,
    pub scratch_s_ltp: Vec<i16>,
    pub scratch_x_sc_q10: Vec<i32>,
    pub scratch_xq_tmp: Vec<i16>,

    // LBRR scratch buffers (separate from main NSQ)
    pub lbrr_scratch_s_ltp_q15: Vec<i32>,
    pub lbrr_scratch_s_ltp: Vec<i16>,
    pub lbrr_scratch_x_sc_q10: Vec<i32>,
    pub lbrr_scratch_xq_tmp: Vec<i16>,
}

impl SilkEncoderStateFlp {
    pub fn new() -> Self {
        Self {
            fs_khz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            predict_lpc_order: 0,
            shaping_lpc_order: 0,
            shape_win_length: 0,
            la_pitch: 0,
            pitch_lpc_win_length: 0,
            pitch_estimation_lpc_order: 0,

            warping_q16: 0,
            nlsf_cb_sel: NlsfCbSel::NbMb,
            x_buf: vec![0.0; MAX_SILK_X_BUF],
            nsq_state: NsqState::new(),
            indices: SideInfoIndices::default(),
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            prev_lag: 0,
            first_frame_after_reset: true,
            last_gain_index: 10,
            prev_harm_smth: 0.0,
            prev_tilt_smth: 0.0,
            vad_state: vad::VadState::default(),
            speech_activity_q8: 128,
            input_quality_bands_q15: [0; 4],
            input_tilt_q15: 0,
            snr_db_q7: 0,
            n_frames_encoded: 0,
            n_frames_per_packet: 1,
            packet_loss_perc: 0,
            lbrr: LbrrState::new(),
            scratch_s_ltp_q15: vec![0; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH],
            scratch_s_ltp: vec![0; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH],
            scratch_x_sc_q10: vec![0; MAX_SUB_FRAME_LENGTH],
            scratch_xq_tmp: vec![0; MAX_SUB_FRAME_LENGTH],
            lbrr_scratch_s_ltp_q15: vec![0; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH],
            lbrr_scratch_s_ltp: vec![0; MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH],
            lbrr_scratch_x_sc_q10: vec![0; MAX_SUB_FRAME_LENGTH],
            lbrr_scratch_xq_tmp: vec![0; MAX_SUB_FRAME_LENGTH],
        }
    }

    /// Configure for a given sample rate and payload size.
    pub fn set_fs(&mut self, fs_khz: i32, payload_size_ms: i32) {
        if self.fs_khz != fs_khz {
            self.first_frame_after_reset = true;
            self.prev_nlsf_q15 = [0; MAX_LPC_ORDER];
            self.last_gain_index = 10;
        }

        self.fs_khz = fs_khz;
        self.nb_subfr = if payload_size_ms == 10 { 2 } else { MAX_NB_SUBFR as i32 };
        self.subfr_length = SUB_FRAME_LENGTH_MS as i32 * fs_khz;
        self.frame_length = self.nb_subfr * self.subfr_length;
        self.ltp_mem_length = LTP_MEM_LENGTH_MS as i32 * fs_khz;
        // C: shapeWinLength = SUB_FRAME_LENGTH_MS * fs_kHz + 2 * la_shape
        let la_shape = LA_SHAPE_MS as i32 * fs_khz;
        self.shape_win_length = SUB_FRAME_LENGTH_MS as i32 * fs_khz + 2 * la_shape;
        // Pitch analysis params (C: control_codec.c lines 285-290)
        self.la_pitch = 2 * fs_khz; // LA_PITCH_MS=2
        let pitch_lpc_win_ms = if self.nb_subfr == 2 { 14 } else { 24 }; // 10+2*2 or 20+2*2
        self.pitch_lpc_win_length = pitch_lpc_win_ms * fs_khz;
        self.pitch_estimation_lpc_order = if fs_khz == 8 { 8 } else { 16 };

        if fs_khz <= 12 {
            self.predict_lpc_order = MIN_LPC_ORDER as i32;
            self.shaping_lpc_order = 16; // MAX_SHAPE_LPC_ORDER for NB/MB
            self.nlsf_cb_sel = NlsfCbSel::NbMb;
        } else {
            self.predict_lpc_order = MAX_LPC_ORDER as i32;
            self.shaping_lpc_order = 24; // MAX_SHAPE_LPC_ORDER for WB
            self.nlsf_cb_sel = NlsfCbSel::Wb;
        }
    }
}
