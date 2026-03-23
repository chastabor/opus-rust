// Top-level SILK encoder
// Port of silk/enc_API.c, silk/control_codec.c (simplified)

use opus_range_coder::EcCtx;
use crate::*;
use crate::tables::*;
use crate::nlsf::*;
use crate::gain_quant;
use crate::encode_indices;
use crate::encode_pulses;
use crate::nlsf_encode;
use crate::lpc_analysis;
use crate::pitch_analysis;
use crate::nsq::{self, NsqState, MAX_SHAPE_LPC_ORDER};
use crate::vad;
use crate::noise_shape_analysis;

/// Encoder control parameters (from API)
#[derive(Clone)]
pub struct SilkEncControl {
    /// API sampling rate in Hz (8000, 12000, 16000, 24000, 48000)
    pub api_sample_rate: i32,
    /// Maximum internal sample rate in Hz
    pub max_internal_fs_hz: i32,
    /// Payload size in milliseconds (10 or 20)
    pub payload_size_ms: i32,
    /// Target bitrate in bits per second
    pub bitrate_bps: i32,
    /// Encoder complexity (0-10)
    pub complexity: i32,
    /// Whether to use in-band FEC (LBRR)
    pub use_in_band_fec: bool,
    /// Expected packet loss percentage (0-100)
    pub packet_loss_percentage: i32,
}

impl Default for SilkEncControl {
    fn default() -> Self {
        Self {
            api_sample_rate: 16000,
            max_internal_fs_hz: 16000,
            payload_size_ms: 20,
            bitrate_bps: 25000,
            complexity: 2,
            use_in_band_fec: false,
            packet_loss_percentage: 0,
        }
    }
}

// Maximum buffer sizes for SILK encoder scratch (16kHz, 20ms frame)
const MAX_SILK_FRAME: usize = MAX_FRAME_LENGTH; // 320
const MAX_SILK_SUBFR: usize = MAX_SUB_FRAME_LENGTH; // 80
const MAX_SILK_LTP_MEM: usize = 320; // LTP_MEM_LENGTH_MS * MAX_FS_KHZ
const MAX_SILK_TOTAL: usize = MAX_SILK_LTP_MEM + MAX_SILK_FRAME; // 640
const MAX_SILK_WIN: usize = 240; // shape_win_length max (15 * 16)

/// Pre-allocated scratch buffers for SILK encoder (avoids per-frame heap allocations).
#[derive(Default)]
pub struct SilkScratch {
    pub analysis_buf: Vec<i16>,   // MAX_SILK_TOTAL
    pub vad_x: Vec<i16>,         // ~800
    pub x_windowed: Vec<i16>,    // MAX_SILK_WIN
    pub nsq_s_ltp_q15: Vec<i32>, // MAX_SILK_TOTAL
    pub nsq_s_ltp: Vec<i16>,     // MAX_SILK_TOTAL
    pub nsq_x_sc_q10: Vec<i32>,  // MAX_SILK_SUBFR
    pub nsq_xq_tmp: Vec<i16>,    // MAX_SILK_SUBFR
    // LBRR NSQ scratch buffers (separate from main NSQ to avoid conflicts)
    pub lbrr_nsq_s_ltp_q15: Vec<i32>, // MAX_SILK_TOTAL
    pub lbrr_nsq_s_ltp: Vec<i16>,     // MAX_SILK_TOTAL
    pub lbrr_nsq_x_sc_q10: Vec<i32>,  // MAX_SILK_SUBFR
    pub lbrr_nsq_xq_tmp: Vec<i16>,    // MAX_SILK_SUBFR
}

impl SilkScratch {
    fn new() -> Self {
        Self {
            analysis_buf: vec![0i16; MAX_SILK_TOTAL],
            vad_x: vec![0i16; 800],
            x_windowed: vec![0i16; MAX_SILK_WIN],
            nsq_s_ltp_q15: vec![0i32; MAX_SILK_TOTAL],
            nsq_s_ltp: vec![0i16; MAX_SILK_TOTAL],
            nsq_x_sc_q10: vec![0i32; MAX_SILK_SUBFR],
            nsq_xq_tmp: vec![0i16; MAX_SILK_SUBFR],
            lbrr_nsq_s_ltp_q15: vec![0i32; MAX_SILK_TOTAL],
            lbrr_nsq_s_ltp: vec![0i16; MAX_SILK_TOTAL],
            lbrr_nsq_x_sc_q10: vec![0i32; MAX_SILK_SUBFR],
            lbrr_nsq_xq_tmp: vec![0i16; MAX_SILK_SUBFR],
        }
    }
}

/// Per-channel encoder state
struct EncChannelState {
    // Frame configuration
    fs_khz: i32,
    nb_subfr: i32,
    frame_length: i32,
    subfr_length: i32,
    ltp_mem_length: i32,
    lpc_order: i32,

    // NLSF codebook selection
    nlsf_cb_sel: NlsfCbSel,
    pitch_contour_sel: PitchContourSel,
    pitch_lag_low_bits_sel: PitchLagLowBitsSel,

    // Previous frame state
    prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    last_gain_index: i8,
    prev_signal_type: i32,
    ec_prev_signal_type: i32,
    ec_prev_lag_index: i16,
    first_frame_after_reset: bool,

    // Indices
    indices: SideInfoIndices,

    // NSQ state
    nsq_state: NsqState,

    // VAD state
    vad_state: vad::VadState,
    speech_activity_q8: i32,
    snr_db_q7: i32,

    // Noise shape state (smoothing across frames)
    prev_tilt_smth_q16: i32,
    prev_harm_smth_q16: i32,
    shaping_lpc_order: i32,
    warping_q16: i32,

    // Input buffer (for LPC analysis history)
    input_buf: Vec<i16>,
    input_buf_idx: usize,

    // Frame counter
    n_frames_encoded: i32,

    // Pitch analysis state (for lag tracking across frames)
    prev_lag: i32,
    ltp_corr_q15: i32,

    // LBRR state
    lbrr_enabled: bool,
    lbrr_gain_increases: i32,
    lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],
    lbrr_prev_last_gain_index: i8,
    indices_lbrr: [SideInfoIndices; MAX_FRAMES_PER_PACKET],
    pulses_lbrr: [[i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],

    // Previous packet LBRR data (written at start of next packet)
    prev_lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],
    prev_indices_lbrr: [SideInfoIndices; MAX_FRAMES_PER_PACKET],
    prev_pulses_lbrr: [[i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
    prev_lbrr_any: bool,
    prev_n_frames_per_packet: i32,
}

impl EncChannelState {
    fn new() -> Self {
        Self {
            fs_khz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: 0,
            nlsf_cb_sel: NlsfCbSel::NbMb,
            pitch_contour_sel: PitchContourSel::Nb,
            pitch_lag_low_bits_sel: PitchLagLowBitsSel::Uniform4,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            last_gain_index: 10,
            prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            first_frame_after_reset: true,
            indices: SideInfoIndices::default(),
            nsq_state: NsqState::new(),
            vad_state: vad::VadState::default(),
            speech_activity_q8: 128,
            snr_db_q7: 0,
            prev_tilt_smth_q16: 0,
            prev_harm_smth_q16: 0,
            shaping_lpc_order: 16,
            warping_q16: 0,
            input_buf: vec![0i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
            input_buf_idx: 0,
            n_frames_encoded: 0,
            prev_lag: 0,
            ltp_corr_q15: 0,

            // LBRR state
            lbrr_enabled: false,
            lbrr_gain_increases: 0,
            lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            lbrr_prev_last_gain_index: 10,
            indices_lbrr: Default::default(),
            pulses_lbrr: [[0i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],

            // Previous packet LBRR data
            prev_lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            prev_indices_lbrr: Default::default(),
            prev_pulses_lbrr: [[0i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
            prev_lbrr_any: false,
            prev_n_frames_per_packet: 1,
        }
    }

    fn set_fs(&mut self, fs_khz: i32, payload_size_ms: i32) {
        // Reset state on sample rate change
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

        if fs_khz == 8 || fs_khz == 12 {
            self.lpc_order = MIN_LPC_ORDER as i32;
            self.nlsf_cb_sel = NlsfCbSel::NbMb;
        } else {
            self.lpc_order = MAX_LPC_ORDER as i32;
            self.nlsf_cb_sel = NlsfCbSel::Wb;
        }

        if fs_khz == 8 {
            self.pitch_contour_sel = if self.nb_subfr == MAX_NB_SUBFR as i32 {
                PitchContourSel::Nb
            } else {
                PitchContourSel::Nb10ms
            };
        } else {
            self.pitch_contour_sel = if self.nb_subfr == MAX_NB_SUBFR as i32 {
                PitchContourSel::Wb
            } else {
                PitchContourSel::Wb10ms
            };
        }

        self.pitch_lag_low_bits_sel = match fs_khz {
            16 => PitchLagLowBitsSel::Uniform8,
            12 => PitchLagLowBitsSel::Uniform6,
            _ => PitchLagLowBitsSel::Uniform4,
        };

        // Shaping LPC order based on complexity (from silk/control_codec.c)
        // and warping for bilinear transform
        self.shaping_lpc_order = 16; // default, updated in encode() based on complexity
        self.warping_q16 = 0;        // default, updated based on complexity >= 4
    }
}

/// Top-level SILK encoder (mono)
pub struct SilkEncoder {
    state: EncChannelState,
    initialized: bool,
    scratch: SilkScratch,
}

impl SilkEncoder {
    /// Create a new SILK encoder
    pub fn new() -> Self {
        Self {
            state: EncChannelState::new(),
            initialized: false,
            scratch: SilkScratch::new(),
        }
    }

    /// Encode one SILK frame.
    ///
    /// Returns the number of bytes written, or a negative error code.
    /// The encoder writes into the provided range coder `enc`.
    pub fn encode(
        &mut self,
        control: &SilkEncControl,
        enc: &mut EcCtx,
        samples: &[i16],
    ) -> i32 {
        // Take scratch buffers to avoid borrow conflicts with self.state
        let mut scratch = std::mem::take(&mut self.scratch);
        let cs = &mut self.state;

        // Determine internal sampling rate
        let fs_khz = match control.max_internal_fs_hz {
            ..=8000 => 8,
            8001..=12000 => 12,
            _ => 16,
        };

        // Initialize / reconfigure if needed
        if !self.initialized || cs.fs_khz != fs_khz {
            cs.set_fs(fs_khz, control.payload_size_ms);
            self.initialized = true;
        }

        let frame_length = cs.frame_length as usize;
        let subfr_length = cs.subfr_length as usize;
        let nb_subfr = cs.nb_subfr as usize;
        let lpc_order = cs.lpc_order as usize;
        let ltp_mem_length = cs.ltp_mem_length as usize;

        // Number of frames per packet
        let n_frames_per_packet: i32 = 1;

        // Ensure we have enough input
        if samples.len() < frame_length {
            return -1;
        }

        // ====== LBRR setup (from silk/control_codec.c silk_setup_LBRR) ======
        let lbrr_in_previous = cs.lbrr_enabled;
        cs.lbrr_enabled = control.use_in_band_fec && control.packet_loss_percentage > 0;
        if cs.lbrr_enabled {
            // 13107 = SILK_FIX_CONST(0.2, 16) -- maps percentage to gain increase
            cs.lbrr_gain_increases = if !lbrr_in_previous {
                7
            } else {
                (7 - silk_smulwb(control.packet_loss_percentage, 13107)).max(2)
            };
        }

        // At the start of a new packet (n_frames_encoded == 0), reset LBRR flags
        if cs.n_frames_encoded == 0 {
            cs.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
            cs.lbrr_prev_last_gain_index = cs.last_gain_index;
        }

        // Build analysis buffer: history + current frame (use scratch)
        let analysis_buf_len = ltp_mem_length + frame_length;
        let analysis_buf = &mut scratch.analysis_buf[..analysis_buf_len];
        // Copy history from input_buf
        let history_len = ltp_mem_length.min(cs.input_buf_idx);
        if history_len > 0 {
            let src_start = cs.input_buf_idx - history_len;
            analysis_buf[ltp_mem_length - history_len..ltp_mem_length]
                .copy_from_slice(&cs.input_buf[src_start..cs.input_buf_idx]);
        }
        // Copy current frame
        analysis_buf[ltp_mem_length..ltp_mem_length + frame_length]
            .copy_from_slice(&samples[..frame_length]);

        // Update input buffer with current frame for next call's history
        if cs.input_buf_idx + frame_length > cs.input_buf.len() {
            // Shift buffer
            let shift = cs.input_buf_idx + frame_length - cs.input_buf.len();
            cs.input_buf.copy_within(shift..cs.input_buf_idx, 0);
            cs.input_buf_idx -= shift;
        }
        let start = cs.input_buf_idx;
        let end = start + frame_length;
        if end <= cs.input_buf.len() {
            cs.input_buf[start..end].copy_from_slice(&samples[..frame_length]);
            cs.input_buf_idx = end;
        }

        // ====== Analysis ======

        // 1. LPC analysis: compute autocorrelation -> Levinson-Durbin -> LPC coefficients
        let mut corr = [0i32; MAX_LPC_ORDER + 1];
        lpc_analysis::silk_autocorrelation(
            &mut corr,
            &analysis_buf[ltp_mem_length..],
            frame_length,
            lpc_order,
        );

        let mut a_q16 = [0i32; MAX_LPC_ORDER];
        let _pred_gain = lpc_analysis::silk_levinson_durbin(&mut a_q16, &corr, lpc_order);

        // Convert to Q12 for the filter
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        for i in 0..lpc_order {
            a_q12[i] = silk_rshift_round(a_q16[i], 4) as i16;
        }

        // 2. Convert LPC to NLSF
        let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
        lpc_analysis::silk_a2nlsf(&mut nlsf_q15, &mut a_q16, lpc_order);

        // 3. Quantize NLSFs
        let mut w_q2 = [0i16; MAX_LPC_ORDER];
        lpc_analysis::silk_nlsf_vq_weights_laroia(&mut w_q2, &nlsf_q15, lpc_order);

        let nlsf_cb = get_nlsf_cb(cs.nlsf_cb_sel);
        cs.indices.nlsf_indices = [0i8; MAX_LPC_ORDER + 1];
        // NLSF mu from C reference: 6 for voiced, 4 for unvoiced (scaled by 2^20)
        // Use previous signal type since pitch analysis hasn't run yet for this frame
        let prev_voiced = cs.prev_signal_type == TYPE_VOICED;
        let nlsf_mu_q20 = if prev_voiced { 6 << 20 >> 2 } else { 4 << 20 >> 2 };
        // Number of NLSF survivors based on complexity (from silk/control_codec.c)
        let nlsf_survivors = match control.complexity {
            0 => 2,
            1 => 3,
            2 => 2,
            3 => 4,
            4 | 5 => 6,
            6 | 7 => 8,
            _ => 16,
        };
        nlsf_encode::silk_nlsf_encode(
            &mut cs.indices.nlsf_indices,
            &mut nlsf_q15,
            nlsf_cb,
            &w_q2,
            nlsf_mu_q20, nlsf_survivors,
            cs.indices.signal_type as i32,
        );

        // 4. Convert quantized NLSFs back to LPC for filtering
        let mut pred_coef_q12 = [0i16; 2 * MAX_LPC_ORDER];
        silk_nlsf2a(&mut pred_coef_q12[MAX_LPC_ORDER..MAX_LPC_ORDER + lpc_order], &nlsf_q15, lpc_order);
        // For interpolation: use the same coefficients for both halves (simplified)
        // Copy second half to first half via temp to satisfy borrow checker
        let mut tmp_coefs = [0i16; MAX_LPC_ORDER];
        tmp_coefs[..lpc_order].copy_from_slice(&pred_coef_q12[MAX_LPC_ORDER..MAX_LPC_ORDER + lpc_order]);
        pred_coef_q12[..lpc_order].copy_from_slice(&tmp_coefs[..lpc_order]);

        cs.indices.nlsf_interp_coef_q2 = 4; // No interpolation (simplified)

        // 5. Pitch analysis using full 3-stage hierarchical search
        let mut pitch_lags = [0i32; MAX_NB_SUBFR];

        // Derive pitch estimation complexity from encoder complexity (matches C reference)
        let pe_complexity = match control.complexity {
            0 => 0,
            1 => 1,
            2 => 0,
            3 => 1,
            4..=7 => 1,
            _ => 2,
        };

        // Derive search threshold from complexity (matches C reference)
        let search_thres1_q16 = match control.complexity {
            0 | 2 => 52429,   // SILK_FIX_CONST(0.8, 16)
            1 | 3 => 49807,   // SILK_FIX_CONST(0.76, 16)
            4 | 5 => 48497,   // SILK_FIX_CONST(0.74, 16)
            6 | 7 => 47186,   // SILK_FIX_CONST(0.72, 16)
            _ => 45875,       // SILK_FIX_CONST(0.7, 16)
        };
        let search_thres2_q13 = 2458; // SILK_FIX_CONST(0.3, 13)

        let ret = pitch_analysis::silk_pitch_analysis_core(
            &analysis_buf[..analysis_buf_len],
            &mut pitch_lags,
            &mut cs.indices.lag_index,
            &mut cs.indices.contour_index,
            &mut cs.ltp_corr_q15,
            cs.prev_lag,
            search_thres1_q16,
            search_thres2_q13,
            cs.fs_khz,
            pe_complexity,
            cs.nb_subfr,
        );

        let voiced = ret == 0;

        // Set signal type
        if voiced {
            cs.indices.signal_type = TYPE_VOICED as i8;
            cs.prev_lag = pitch_lags[nb_subfr - 1]; // Track lag for next frame
        } else {
            cs.indices.signal_type = TYPE_UNVOICED as i8;
            cs.prev_lag = 0;
        }
        cs.indices.quant_offset_type = 0; // Low offset

        // 6. LTP analysis (for voiced frames)
        // Note: lag_index and contour_index are already set by silk_pitch_analysis_core
        if voiced {
            // LTP analysis: find per_index and ltp_index
            pitch_analysis::silk_find_ltp_params(
                &mut cs.indices.ltp_index,
                &mut cs.indices.per_index,
                &pitch_lags,
                &analysis_buf,
                &pred_coef_q12[MAX_LPC_ORDER..],
                cs.subfr_length,
                cs.nb_subfr,
                cs.ltp_mem_length,
                cs.lpc_order,
            );

            cs.indices.ltp_scale_index = 0;
        }

        // 7. Gain estimation
        let mut gains_q16 = [0i32; MAX_NB_SUBFR];
        for k in 0..nb_subfr {
            let offset = k * subfr_length;
            let end_idx = (offset + subfr_length).min(frame_length);
            let mut energy: i64 = 0;
            for i in offset..end_idx {
                let s = samples[i] as i64;
                energy += s * s;
            }
            // RMS energy as gain
            let rms = ((energy / subfr_length as i64).max(1) as f64).sqrt() as i32;
            gains_q16[k] = (rms.max(1)) << 6; // Scale up to reasonable Q16 range
        }

        // Determine coding mode
        let cond_coding = if cs.n_frames_encoded > 0 && !cs.first_frame_after_reset {
            CODE_CONDITIONALLY
        } else {
            CODE_INDEPENDENTLY
        };

        // Quantize gains
        gain_quant::silk_gains_quant(
            &mut cs.indices.gains_indices,
            &mut gains_q16,
            &mut cs.last_gain_index,
            cond_coding == CODE_CONDITIONALLY,
            nb_subfr,
        );

        // 8. NSQ - Noise Shaping Quantization
        let mut pulses = [0i8; MAX_FRAME_LENGTH];

        // Build LTP coefficients from indices
        let mut ltp_coef_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
        if voiced {
            let cbk = SILK_LTP_VQ_PTRS_Q7[cs.indices.per_index as usize];
            for k in 0..nb_subfr {
                let entry = &cbk[cs.indices.ltp_index[k] as usize];
                for j in 0..LTP_ORDER {
                    ltp_coef_q14[k * LTP_ORDER + j] = (entry[j] as i16) << 7;
                }
            }
        }

        // Set complexity-dependent shaping parameters
        cs.shaping_lpc_order = match control.complexity {
            0 => 12,
            1 | 3 => 14,
            2 => 12,
            4 | 5 => 16,
            6 | 7 => 20,
            _ => 24,
        };
        cs.warping_q16 = if control.complexity >= 4 {
            ((0.015 * cs.fs_khz as f64) * 65536.0) as i32
        } else {
            0
        };

        // Run VAD for speech activity estimation
        let mut quality_bands_q15 = [0i32; 4];
        let mut tilt_q15 = 0i32;
        vad::silk_vad_get_sa_q8(
            &mut cs.vad_state,
            &mut cs.speech_activity_q8,
            &mut cs.snr_db_q7,
            &mut quality_bands_q15,
            &mut tilt_q15,
            &samples[..frame_length],
            cs.frame_length,
        );

        // Noise shape analysis: compute spectral shaping filter parameters
        let shape_result = noise_shape_analysis::silk_noise_shape_analysis(
            &samples[..frame_length],
            &pitch_lags,
            voiced,
            &mut cs.prev_tilt_smth_q16,
            &mut cs.prev_harm_smth_q16,
            cs.fs_khz,
            cs.nb_subfr,
            cs.subfr_length,
            cs.frame_length,
            cs.lpc_order,
            cs.shaping_lpc_order,
            cs.warping_q16,
            cs.speech_activity_q8,
            10000, // coding_quality_q14 (moderate default)
            cs.snr_db_q7,
        );

        // Use noise shape results
        let ar_q13 = shape_result.ar_q13;
        let harm_shape_gain_q14 = &shape_result.harm_shape_gain_q14[..nb_subfr];
        let tilt_q14 = &shape_result.tilt_q14[..nb_subfr];
        let lf_shp_q14 = &shape_result.lf_shp_q14[..nb_subfr];
        let lambda_q10 = shape_result.lambda_q10;

        // Use noise shape analysis quant offset type
        cs.indices.quant_offset_type = shape_result.quant_offset_type;

        let ltp_scale_q14 = if cond_coding == CODE_INDEPENDENTLY {
            SILK_LTP_SCALES_TABLE_Q14[cs.indices.ltp_scale_index as usize] as i32
        } else {
            SILK_LTP_SCALES_TABLE_Q14[0] as i32
        };

        // Set random seed
        cs.indices.seed = (cs.n_frames_encoded & 3) as i8;

        // ====== LBRR encoding (before main NSQ, so we can clone NSQ state) ======
        let frame_idx = cs.n_frames_encoded as usize;
        if cs.lbrr_enabled {
            // Only encode LBRR for frames with sufficient speech activity
            // Threshold: speech_activity_q8 > 0.3 * 256 = 76
            if cs.speech_activity_q8 > 76 {
                cs.lbrr_flags[frame_idx] = 1;

                // Copy current indices for LBRR
                cs.indices_lbrr[frame_idx] = cs.indices.clone();

                // Boost first subframe gain index by lbrr_gain_increases
                let mut lbrr_gains_indices = cs.indices.gains_indices;
                if cond_coding == CODE_INDEPENDENTLY {
                    // Absolute coding: boost the absolute gain index
                    lbrr_gains_indices[0] = ((lbrr_gains_indices[0] as i32
                        + cs.lbrr_gain_increases)
                        .min(N_LEVELS_QGAIN - 1)) as i8;
                } else {
                    // Delta coding: boost the delta to increase gain
                    lbrr_gains_indices[0] = ((lbrr_gains_indices[0] as i32
                        + cs.lbrr_gain_increases)
                        .min(MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT)) as i8;
                }
                cs.indices_lbrr[frame_idx].gains_indices = lbrr_gains_indices;

                // Dequantize LBRR gains
                let mut lbrr_gains_q16 = [0i32; MAX_NB_SUBFR];
                let mut lbrr_prev_gain_idx = cs.lbrr_prev_last_gain_index;
                gain_quant::silk_gains_dequant(
                    &mut lbrr_gains_q16,
                    &lbrr_gains_indices,
                    &mut lbrr_prev_gain_idx,
                    cond_coding == CODE_CONDITIONALLY,
                    nb_subfr,
                );

                // Clone NSQ state for LBRR (does not affect main encoder state)
                let mut lbrr_nsq_state = cs.nsq_state.clone();
                let mut lbrr_indices = cs.indices_lbrr[frame_idx].clone();

                // Read NSQ config values for LBRR
                let lbrr_signal_type = cs.indices.signal_type as i32;
                let lbrr_quant_offset_type = cs.indices.quant_offset_type as i32;
                let lbrr_nlsf_interp_coef_q2 = cs.indices.nlsf_interp_coef_q2 as i32;

                // Run NSQ with LBRR gains
                let mut lbrr_pulses = [0i8; MAX_FRAME_LENGTH];
                nsq::silk_nsq(
                    &mut lbrr_nsq_state,
                    &mut lbrr_indices,
                    &samples[..frame_length],
                    &mut lbrr_pulses,
                    &pred_coef_q12,
                    &ltp_coef_q14,
                    &ar_q13,
                    harm_shape_gain_q14,
                    tilt_q14,
                    lf_shp_q14,
                    &lbrr_gains_q16,
                    &pitch_lags,
                    lambda_q10,
                    ltp_scale_q14,
                    cs.frame_length,
                    cs.subfr_length,
                    cs.ltp_mem_length,
                    cs.lpc_order,
                    MAX_SHAPE_LPC_ORDER as i32,
                    cs.nb_subfr,
                    lbrr_signal_type,
                    lbrr_quant_offset_type,
                    lbrr_nlsf_interp_coef_q2,
                    &mut scratch.lbrr_nsq_s_ltp_q15,
                    &mut scratch.lbrr_nsq_s_ltp,
                    &mut scratch.lbrr_nsq_x_sc_q10,
                    &mut scratch.lbrr_nsq_xq_tmp,
                );

                // Store LBRR pulses
                cs.pulses_lbrr[frame_idx][..frame_length]
                    .copy_from_slice(&lbrr_pulses[..frame_length]);
            } else {
                cs.lbrr_flags[frame_idx] = 0;
            }
        } else {
            cs.lbrr_flags[frame_idx] = 0;
        }

        // ====== Main NSQ ======

        // Read values before borrowing cs.indices mutably
        let nsq_signal_type = cs.indices.signal_type as i32;
        let nsq_quant_offset_type = cs.indices.quant_offset_type as i32;
        let nsq_nlsf_interp_coef_q2 = cs.indices.nlsf_interp_coef_q2 as i32;
        let nsq_frame_length = cs.frame_length;
        let nsq_subfr_length = cs.subfr_length;
        let nsq_ltp_mem_length = cs.ltp_mem_length;
        let nsq_lpc_order = cs.lpc_order;
        let nsq_nb_subfr = cs.nb_subfr;

        nsq::silk_nsq(
            &mut cs.nsq_state,
            &mut cs.indices,
            &samples[..frame_length],
            &mut pulses,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            harm_shape_gain_q14,
            tilt_q14,
            lf_shp_q14,
            &gains_q16,
            &pitch_lags,
            lambda_q10,
            ltp_scale_q14,
            nsq_frame_length,
            nsq_subfr_length,
            nsq_ltp_mem_length,
            nsq_lpc_order,
            MAX_SHAPE_LPC_ORDER as i32,
            nsq_nb_subfr,
            nsq_signal_type,
            nsq_quant_offset_type,
            nsq_nlsf_interp_coef_q2,
            &mut scratch.nsq_s_ltp_q15,
            &mut scratch.nsq_s_ltp,
            &mut scratch.nsq_x_sc_q10,
            &mut scratch.nsq_xq_tmp,
        );

        // ====== Bitstream writing ======

        // Encode VAD flags (simplified: always set VAD = 1 for voiced, 0 for unvoiced)
        // The VAD flags are typically encoded at the packet level, not per-frame here.
        // For this simplified encoder, the caller handles that.

        // Encode side information indices
        let prev_sig_type = cs.ec_prev_signal_type;
        let prev_lag_idx = cs.ec_prev_lag_index;
        encode_indices::silk_encode_indices(
            &cs.indices,
            enc,
            0, // frame_index
            false, // encode_lbrr
            cond_coding,
            cs.nb_subfr,
            cs.nlsf_cb_sel,
            cs.pitch_contour_sel,
            cs.pitch_lag_low_bits_sel,
            cs.fs_khz,
            prev_sig_type,
            prev_lag_idx,
        );

        // Update ec_prev state (after encode_indices, matching what the decoder expects)
        cs.ec_prev_signal_type = cs.indices.signal_type as i32;
        if voiced {
            cs.ec_prev_lag_index = cs.indices.lag_index;
        }

        // Encode excitation pulses
        encode_pulses::silk_encode_pulses(
            enc,
            &pulses,
            cs.indices.signal_type as i32,
            cs.indices.quant_offset_type as i32,
            cs.frame_length,
        );

        // Update state for next frame
        cs.prev_nlsf_q15[..lpc_order].copy_from_slice(&nlsf_q15[..lpc_order]);
        cs.prev_signal_type = cs.indices.signal_type as i32;
        cs.first_frame_after_reset = false;
        cs.n_frames_encoded += 1;

        // If this is the last frame in the packet, save LBRR data for next packet
        if cs.n_frames_encoded >= n_frames_per_packet {
            // Move current packet's LBRR data to "previous" storage
            cs.prev_lbrr_flags = cs.lbrr_flags;
            cs.prev_indices_lbrr = cs.indices_lbrr.clone();
            cs.prev_pulses_lbrr = cs.pulses_lbrr;
            cs.prev_lbrr_any = cs.lbrr_flags[..n_frames_per_packet as usize]
                .iter()
                .any(|&f| f != 0);
            cs.prev_n_frames_per_packet = n_frames_per_packet;

            // Reset frame counter for next packet
            cs.n_frames_encoded = 0;
        }

        // Put scratch buffers back
        self.scratch = scratch;

        0 // Success
    }

    /// Write LBRR data from the previous packet into the bitstream.
    ///
    /// This should be called at the start of encoding a new packet, after writing
    /// the LBRR flag bit. Returns true if LBRR data was written.
    ///
    /// The LBRR data written here is from the PREVIOUS packet. The first packet
    /// encoded never has LBRR data (no previous packet to reference).
    pub fn write_lbrr_data(
        &self,
        enc: &mut EcCtx,
    ) -> bool {
        let cs = &self.state;

        if !cs.prev_lbrr_any {
            return false;
        }

        let n_frames = cs.prev_n_frames_per_packet as usize;

        // If more than one frame per packet, encode per-frame LBRR flags
        if n_frames > 1 {
            let lbrr_flags_icdf = if n_frames == 2 {
                &SILK_LBRR_FLAGS_2_ICDF[..]
            } else {
                &SILK_LBRR_FLAGS_3_ICDF[..]
            };

            // Compute combined LBRR symbol: binary encoding of per-frame flags
            // For the iCDF tables: symbol = combined_flags - 1 (since 0 means no LBRR,
            // which is never encoded here because prev_lbrr_any is true)
            let mut lbrr_symbol = 0usize;
            for i in 0..n_frames {
                lbrr_symbol |= (cs.prev_lbrr_flags[i] as usize) << i;
            }
            enc.enc_icdf(lbrr_symbol - 1, lbrr_flags_icdf, 8);
        }

        // Encode LBRR indices and pulses for each flagged frame
        for i in 0..n_frames {
            if cs.prev_lbrr_flags[i] != 0 {
                let lbrr_cond_coding = if i > 0 {
                    CODE_CONDITIONALLY
                } else {
                    CODE_INDEPENDENTLY
                };

                encode_indices::silk_encode_indices(
                    &cs.prev_indices_lbrr[i],
                    enc,
                    i as i32,
                    true, // encode_lbrr = true
                    lbrr_cond_coding,
                    cs.nb_subfr,
                    cs.nlsf_cb_sel,
                    cs.pitch_contour_sel,
                    cs.pitch_lag_low_bits_sel,
                    cs.fs_khz,
                    // For LBRR ec_prev state: use previous LBRR frame's signal type/lag
                    if i > 0 { cs.prev_indices_lbrr[i - 1].signal_type as i32 } else { 0 },
                    if i > 0 { cs.prev_indices_lbrr[i - 1].lag_index } else { 0 },
                );

                encode_pulses::silk_encode_pulses(
                    enc,
                    &cs.prev_pulses_lbrr[i],
                    cs.prev_indices_lbrr[i].signal_type as i32,
                    cs.prev_indices_lbrr[i].quant_offset_type as i32,
                    cs.frame_length,
                );
            }
        }

        true
    }

    /// Encode a complete SILK packet with proper header (VAD + LBRR flags).
    ///
    /// This is the high-level API that handles the packet header format:
    /// 1. Write VAD flag (1 bit) and LBRR flag (1 bit) as placeholders
    /// 2. If LBRR data exists from previous packet, write it
    /// 3. Encode the current frame
    /// 4. Patch initial bits to correct VAD + LBRR flag values
    pub fn encode_packet(
        &mut self,
        control: &SilkEncControl,
        enc: &mut EcCtx,
        samples: &[i16],
    ) -> i32 {
        // Write placeholder VAD flag + LBRR flag (will be patched later)
        enc.enc_bit_logp(false, 1); // VAD flag placeholder
        enc.enc_bit_logp(false, 1); // LBRR flag placeholder

        // Write LBRR data from previous packet (if any)
        let has_lbrr = self.write_lbrr_data(enc);

        // Encode the current frame
        let result = self.encode(control, enc, samples);
        if result != 0 {
            return result;
        }

        // Determine actual VAD flag from the encoded frame
        let actual_vad = self.state.prev_signal_type != TYPE_NO_VOICE_ACTIVITY;

        // Patch the initial VAD + LBRR bits
        // Bit layout in first byte: bit 7 = VAD flag, bit 6 = LBRR flag
        // enc_patch_initial_bits patches the MSBs of the first byte
        let flags_byte = (actual_vad as u32) | ((has_lbrr as u32) << 1);
        enc.enc_patch_initial_bits(flags_byte, 2);

        0
    }
}

impl Default for SilkEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{SilkDecoder, SilkDecControl};
    use opus_range_coder::EcCtx;

    #[test]
    fn test_silk_encoder_create() {
        let enc = SilkEncoder::new();
        // Encoder is not initialized until encode() is called
        assert_eq!(enc.state.fs_khz, 0);
        assert!(!enc.initialized);
    }

    #[test]
    fn test_silk_encode_silence() {
        let mut enc = SilkEncoder::new();
        let control = SilkEncControl {
            api_sample_rate: 16000,
            max_internal_fs_hz: 16000,
            payload_size_ms: 20,
            bitrate_bps: 20000,
            complexity: 0,
            use_in_band_fec: false,
            packet_loss_percentage: 0,
        };
        let samples = vec![0i16; 320]; // 20ms at 16kHz
        let mut range_enc = EcCtx::enc_init(1275);

        // Write VAD flag (0 = no voice activity) and LBRR flag (0)
        range_enc.enc_bit_logp(false, 1); // VAD flag
        range_enc.enc_bit_logp(false, 1); // LBRR flag

        let result = enc.encode(&control, &mut range_enc, &samples);
        assert_eq!(result, 0);

        range_enc.enc_done();
        let nbytes = ((range_enc.tell() + 7) >> 3) as usize;
        assert!(nbytes > 0, "Should produce some bytes");
    }

    #[test]
    fn test_silk_encode_decode_roundtrip() {
        let mut enc = SilkEncoder::new();
        let control = SilkEncControl {
            api_sample_rate: 16000,
            max_internal_fs_hz: 16000,
            payload_size_ms: 20,
            bitrate_bps: 20000,
            complexity: 0,
            use_in_band_fec: false,
            packet_loss_percentage: 0,
        };

        // Generate a simple 200Hz tone at 16kHz
        let n = 320;
        let mut samples = vec![0i16; n];
        for i in 0..n {
            samples[i] = (5000.0 * (2.0 * std::f64::consts::PI * 200.0 * i as f64 / 16000.0).sin()) as i16;
        }

        let mut range_enc = EcCtx::enc_init(1275);

        // Write packet header: VAD flag + LBRR flag
        range_enc.enc_bit_logp(true, 1);  // VAD flag = 1 (voice activity)
        range_enc.enc_bit_logp(false, 1); // LBRR flag = 0

        let result = enc.encode(&control, &mut range_enc, &samples);
        assert_eq!(result, 0, "Encode should succeed");

        range_enc.enc_done();

        // Get encoded bytes
        let nbytes = (range_enc.tell() + 7) >> 3;
        let buf = range_enc.buf[..nbytes as usize].to_vec();
        assert!(buf.len() > 2, "Should produce a non-trivial bitstream");

        // Decode
        let mut dec = SilkDecoder::new();
        let mut dec_control = SilkDecControl {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: 16000,
            internal_sample_rate: 16000,
            payload_size_ms: 20,
            prev_pitch_lag: 0,
        };

        let mut range_dec = EcCtx::dec_init(&buf);
        let mut decoded = vec![0i16; n];
        let mut n_samples_out = 0i32;

        let ret = dec.decode(
            &mut dec_control,
            0, // not lost
            true, // new packet
            &mut range_dec,
            &mut decoded,
            &mut n_samples_out,
        );

        assert_eq!(ret, 0, "Decode should succeed on encoder output");
        assert_eq!(n_samples_out, n as i32, "Should decode correct number of samples");

        // Verify decoded signal has energy (not all zeros)
        let energy: i64 = decoded.iter().map(|&x| x as i64 * x as i64).sum();
        assert!(energy > 0, "Decoded signal should have non-zero energy");
    }
}
