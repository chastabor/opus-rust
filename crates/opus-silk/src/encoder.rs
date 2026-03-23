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
}

impl Default for SilkEncControl {
    fn default() -> Self {
        Self {
            api_sample_rate: 16000,
            max_internal_fs_hz: 16000,
            payload_size_ms: 20,
            bitrate_bps: 25000,
            complexity: 2,
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
}

impl SilkEncoder {
    /// Create a new SILK encoder
    pub fn new() -> Self {
        Self {
            state: EncChannelState::new(),
            initialized: false,
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

        // Ensure we have enough input
        if samples.len() < frame_length {
            return -1;
        }

        // Build analysis buffer: history + current frame
        let analysis_buf_len = ltp_mem_length + frame_length;
        let mut analysis_buf = vec![0i16; analysis_buf_len];
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
        let mut corr = vec![0i32; lpc_order + 1];
        lpc_analysis::silk_autocorrelation(
            &mut corr,
            &analysis_buf[ltp_mem_length..],
            frame_length,
            lpc_order,
        );

        let mut a_q16 = vec![0i32; lpc_order];
        let _pred_gain = lpc_analysis::silk_levinson_durbin(&mut a_q16, &corr, lpc_order);

        // Convert to Q12 for the filter
        let mut a_q12 = vec![0i16; lpc_order];
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
        let mut pred_coef_q12 = vec![0i16; 2 * MAX_LPC_ORDER];
        silk_nlsf2a(&mut pred_coef_q12[MAX_LPC_ORDER..MAX_LPC_ORDER + lpc_order], &nlsf_q15, lpc_order);
        // For interpolation: use the same coefficients for both halves (simplified)
        // Copy second half to first half via temp to satisfy borrow checker
        let mut tmp_coefs = [0i16; MAX_LPC_ORDER];
        tmp_coefs[..lpc_order].copy_from_slice(&pred_coef_q12[MAX_LPC_ORDER..MAX_LPC_ORDER + lpc_order]);
        pred_coef_q12[..lpc_order].copy_from_slice(&tmp_coefs[..lpc_order]);

        cs.indices.nlsf_interp_coef_q2 = 4; // No interpolation (simplified)

        // 5. Pitch analysis
        let mut pitch_lags = [0i32; MAX_NB_SUBFR];
        let voiced = pitch_analysis::silk_pitch_analysis_simple(
            &analysis_buf,
            &mut pitch_lags,
            cs.fs_khz,
            cs.nb_subfr,
            cs.frame_length,
        );

        // Set signal type
        if voiced {
            cs.indices.signal_type = TYPE_VOICED as i8;
        } else {
            cs.indices.signal_type = TYPE_UNVOICED as i8;
        }
        cs.indices.quant_offset_type = 0; // Low offset

        // 6. Pitch contour and LTP (for voiced frames)
        if voiced {
            pitch_analysis::silk_find_pitch_contour(
                &mut cs.indices.contour_index,
                &mut cs.indices.lag_index,
                &pitch_lags,
                cs.fs_khz,
                cs.nb_subfr,
            );

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
        let mut pulses = vec![0i8; frame_length];

        // Build LTP coefficients from indices
        let mut ltp_coef_q14 = vec![0i16; nb_subfr * LTP_ORDER];
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
            1 => 14,
            2 | 3 => if control.complexity >= 3 { 14 } else { 12 },
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
        let harm_shape_gain_q14: Vec<i32> = shape_result.harm_shape_gain_q14[..nb_subfr].to_vec();
        let tilt_q14: Vec<i32> = shape_result.tilt_q14[..nb_subfr].to_vec();
        let lf_shp_q14: Vec<i32> = shape_result.lf_shp_q14[..nb_subfr].to_vec();
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
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
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

        0 // Success
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
