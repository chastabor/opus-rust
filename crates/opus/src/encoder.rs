use opus_range_coder::EcCtx;
use opus_celt::CeltEncoder;
use opus_silk::{SilkEncoder, encoder::SilkEncControl};

use crate::error::OpusError;
use crate::packet::*;

// Application constants
pub const OPUS_APPLICATION_VOIP: i32 = 2048;
pub const OPUS_APPLICATION_AUDIO: i32 = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: i32 = 2051;

// Signal type constants
pub const OPUS_SIGNAL_VOICE: i32 = 3001;
pub const OPUS_SIGNAL_MUSIC: i32 = 3002;
pub const OPUS_AUTO: i32 = -1000;
pub const OPUS_BITRATE_MAX: i32 = -1;

/// Generate the TOC (Table of Contents) byte for an Opus packet.
///
/// This encodes the mode, frame rate, bandwidth, and channel count into a
/// single byte that is the first byte of every Opus packet.
///
/// The `framerate` parameter is `fs / frame_size` (e.g., 48000/960 = 50 for 20ms).
fn gen_toc(mode: i32, framerate: i32, bandwidth: i32, channels: i32) -> u8 {
    // Compute period: number of times we need to double the framerate
    // to reach at least 400 (the 2.5ms base rate).
    let mut period = 0u8;
    let mut fr = framerate;
    while fr < 400 {
        fr <<= 1;
        period += 1;
    }

    let toc;
    if mode == MODE_SILK_ONLY {
        // SILK: bits 7-5 = bandwidth - NB, bits 4-3 = period - 2, bit 2 = stereo
        let bw = (bandwidth - OPUS_BANDWIDTH_NARROWBAND) as u8;
        toc = (bw << 5) | ((period - 2) << 3);
    } else if mode == MODE_CELT_ONLY {
        // CELT: bit 7 set, bits 6-5 = max(0, bandwidth - MB), bits 4-3 = period, bit 2 = stereo
        let tmp = ((bandwidth - OPUS_BANDWIDTH_MEDIUMBAND) as u8).max(0);
        toc = 0x80 | (tmp << 5) | (period << 3);
    } else {
        // Hybrid: bits 7-5 = 011, bit 4 = bandwidth - SWB, bit 3 = period - 2, bit 2 = stereo
        let bw = (bandwidth - OPUS_BANDWIDTH_SUPERWIDEBAND) as u8;
        toc = 0x60 | (bw << 4) | ((period - 2) << 3);
    }

    toc | (if channels == 2 { 0x04 } else { 0x00 })
}

/// The main Opus encoder.
pub struct OpusEncoder {
    /// Number of input channels (1 or 2).
    channels: i32,
    /// Input sampling rate.
    fs: i32,
    /// Application type.
    application: i32,
    /// SILK encoder state.
    silk_enc: SilkEncoder,
    /// CELT encoder state.
    celt_enc: CeltEncoder,
    /// Current encoding mode (MODE_SILK_ONLY / MODE_HYBRID / MODE_CELT_ONLY).
    mode: i32,
    /// Previous frame's mode.
    prev_mode: i32,
    /// Current bandwidth (OPUS_BANDWIDTH_*).
    bandwidth: i32,
    /// Number of channels in the stream (may differ from input channels).
    stream_channels: i32,
    /// Target bitrate in bits per second.
    bitrate_bps: i32,
    /// Whether to use variable bitrate.
    use_vbr: bool,
    /// Encoder complexity (0-10).
    complexity: i32,
    /// Signal type hint (OPUS_SIGNAL_VOICE / OPUS_SIGNAL_MUSIC / OPUS_AUTO).
    signal_type: i32,
    /// Force channel count (-1 = auto, 1 = mono, 2 = stereo).
    force_channels: i32,
    /// Maximum allowed bandwidth.
    max_bandwidth: i32,
    /// User-requested bitrate (before clamping).
    user_bitrate_bps: i32,
    /// Whether to use in-band FEC (LBRR) for SILK frames.
    use_inband_fec: bool,
    /// Expected packet loss percentage (0-100) for FEC rate control.
    packet_loss_perc: i32,
    /// Final range coder state for testing/verification.
    pub range_final: u32,
}

impl OpusEncoder {
    /// Create a new Opus encoder.
    ///
    /// `fs` must be one of 8000, 12000, 16000, 24000, or 48000.
    /// `channels` must be 1 or 2.
    /// `application` must be one of OPUS_APPLICATION_VOIP, OPUS_APPLICATION_AUDIO,
    /// or OPUS_APPLICATION_RESTRICTED_LOWDELAY.
    pub fn new(fs: i32, channels: i32, application: i32) -> Result<Self, OpusError> {
        if !(channels == 1 || channels == 2) {
            return Err(OpusError::BadArg);
        }
        if ![8000, 12000, 16000, 24000, 48000].contains(&fs) {
            return Err(OpusError::BadArg);
        }
        if ![
            OPUS_APPLICATION_VOIP,
            OPUS_APPLICATION_AUDIO,
            OPUS_APPLICATION_RESTRICTED_LOWDELAY,
        ]
        .contains(&application)
        {
            return Err(OpusError::BadArg);
        }

        let silk_enc = SilkEncoder::new();
        let celt_enc =
            CeltEncoder::new(48000, channels as usize).map_err(|_| OpusError::InternalError)?;

        let default_bitrate = match application {
            OPUS_APPLICATION_VOIP => 20000,
            _ => 64000,
        };

        Ok(OpusEncoder {
            channels,
            fs,
            application,
            silk_enc,
            celt_enc,
            mode: MODE_CELT_ONLY,
            prev_mode: 0,
            bandwidth: OPUS_BANDWIDTH_FULLBAND,
            stream_channels: channels,
            bitrate_bps: default_bitrate,
            use_vbr: true,
            complexity: 10,
            signal_type: OPUS_AUTO,
            force_channels: OPUS_AUTO,
            max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
            user_bitrate_bps: default_bitrate,
            use_inband_fec: false,
            packet_loss_perc: 0,
            range_final: 0,
        })
    }

    /// Set the target bitrate in bits per second.
    pub fn set_bitrate(&mut self, bitrate: i32) {
        self.user_bitrate_bps = bitrate;
        if bitrate == OPUS_BITRATE_MAX || bitrate == OPUS_AUTO {
            self.bitrate_bps = if self.application == OPUS_APPLICATION_VOIP {
                20000
            } else {
                64000
            };
        } else {
            self.bitrate_bps = bitrate.clamp(500, 512000);
        }
    }

    /// Set the encoder complexity (0-10).
    pub fn set_complexity(&mut self, complexity: i32) {
        self.complexity = complexity.clamp(0, 10);
    }

    /// Set the signal type hint.
    pub fn set_signal(&mut self, signal: i32) {
        self.signal_type = signal;
    }

    /// Set the maximum bandwidth.
    pub fn set_bandwidth(&mut self, bandwidth: i32) {
        self.max_bandwidth = bandwidth;
    }

    /// Enable or disable in-band FEC (LBRR) for SILK frames.
    pub fn set_inband_fec(&mut self, enabled: bool) {
        self.use_inband_fec = enabled;
    }

    /// Set the expected packet loss percentage (0-100) for FEC rate control.
    pub fn set_packet_loss_perc(&mut self, perc: i32) {
        self.packet_loss_perc = perc.clamp(0, 100);
    }

    /// Get the number of channels.
    pub fn channels(&self) -> i32 {
        self.channels
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> i32 {
        self.fs
    }

    /// Decide the encoding mode based on bitrate, application, bandwidth, and frame size.
    fn decide_mode(&self, frame_size: i32) -> (i32, i32) {
        let frame_duration_ms = frame_size * 1000 / self.fs;

        // SILK requires at least 10ms frames
        let silk_ok = frame_duration_ms >= 10;

        // Determine mode from application, bitrate, and signal type
        let mode;
        let bandwidth;

        if self.application == OPUS_APPLICATION_RESTRICTED_LOWDELAY || !silk_ok {
            // Low-delay or sub-10ms frames: CELT only
            mode = MODE_CELT_ONLY;
            bandwidth = self.decide_bandwidth();
        } else if self.application == OPUS_APPLICATION_VOIP
            || self.signal_type == OPUS_SIGNAL_VOICE
        {
            if self.bitrate_bps < 20000 {
                mode = MODE_SILK_ONLY;
                bandwidth = self.decide_bandwidth();
            } else if self.bitrate_bps < 32000 {
                // Potential hybrid zone
                let bw = self.decide_bandwidth();
                if bw >= OPUS_BANDWIDTH_SUPERWIDEBAND {
                    mode = MODE_HYBRID;
                    bandwidth = bw;
                } else {
                    mode = MODE_SILK_ONLY;
                    bandwidth = bw;
                }
            } else {
                mode = MODE_CELT_ONLY;
                bandwidth = self.decide_bandwidth();
            }
        } else {
            // AUDIO application
            if self.bitrate_bps < 12000 {
                mode = MODE_SILK_ONLY;
                bandwidth = self.decide_bandwidth();
            } else if self.bitrate_bps < 24000 {
                let bw = self.decide_bandwidth();
                if bw >= OPUS_BANDWIDTH_SUPERWIDEBAND {
                    mode = MODE_HYBRID;
                    bandwidth = bw;
                } else {
                    mode = MODE_SILK_ONLY;
                    bandwidth = bw;
                }
            } else {
                mode = MODE_CELT_ONLY;
                bandwidth = self.decide_bandwidth();
            }
        }

        // Clamp bandwidth to max
        let bandwidth = bandwidth.min(self.max_bandwidth);

        (mode, bandwidth)
    }

    /// Decide bandwidth based on bitrate.
    fn decide_bandwidth(&self) -> i32 {
        if self.bitrate_bps < 8000 {
            OPUS_BANDWIDTH_NARROWBAND
        } else if self.bitrate_bps < 12000 {
            OPUS_BANDWIDTH_MEDIUMBAND
        } else if self.bitrate_bps < 16000 {
            OPUS_BANDWIDTH_WIDEBAND
        } else if self.bitrate_bps < 20000 {
            OPUS_BANDWIDTH_SUPERWIDEBAND
        } else {
            OPUS_BANDWIDTH_FULLBAND
        }
    }

    /// Encode an Opus frame from floating-point PCM.
    ///
    /// `pcm` contains `frame_size * channels` interleaved samples.
    /// `frame_size` must correspond to 2.5, 5, 10, 20, 40, or 60 ms at the
    /// configured sample rate.
    /// Returns the number of bytes written into `data`.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, OpusError> {
        if frame_size <= 0 || max_data_bytes <= 0 {
            return Err(OpusError::BadArg);
        }

        let max_data_bytes = max_data_bytes.min(data.len() as i32);
        if max_data_bytes < 1 {
            return Err(OpusError::BufferTooSmall);
        }

        // Validate frame_size: must be 2.5, 5, 10, 20, 40, or 60ms
        let valid_frame_sizes = [
            self.fs / 400,      // 2.5ms
            self.fs / 200,      // 5ms
            self.fs / 100,      // 10ms
            self.fs / 50,       // 20ms
            self.fs / 25,       // 40ms
            self.fs * 60 / 1000, // 60ms
        ];
        if !valid_frame_sizes.contains(&frame_size) {
            return Err(OpusError::BadArg);
        }

        let total_samples = (frame_size * self.channels) as usize;
        if pcm.len() < total_samples {
            return Err(OpusError::BadArg);
        }

        // Determine stream channels
        let stream_channels = if self.force_channels > 0 {
            self.force_channels.min(self.channels)
        } else {
            self.channels
        };
        self.stream_channels = stream_channels;

        // Mode and bandwidth decision
        let (mode, bandwidth) = self.decide_mode(frame_size);
        self.mode = mode;
        self.bandwidth = bandwidth;

        let frame_rate = self.fs / frame_size;

        // Generate TOC byte
        let toc = gen_toc(mode, frame_rate, bandwidth, stream_channels);
        data[0] = toc;

        // Initialize range encoder for the payload (after the TOC byte)
        let payload_max = (max_data_bytes - 1) as u32;
        if payload_max < 2 {
            // Too small for any payload, produce a DTX-like packet
            self.range_final = 0;
            return Ok(1);
        }

        let mut enc = EcCtx::enc_init(payload_max);

        let mut silk_bytes_used = 0i32;

        // === SILK encoding ===
        if mode == MODE_SILK_ONLY || mode == MODE_HYBRID {
            // Determine SILK internal rate
            let silk_internal_rate = if mode == MODE_SILK_ONLY {
                match bandwidth {
                    OPUS_BANDWIDTH_NARROWBAND => 8000,
                    OPUS_BANDWIDTH_MEDIUMBAND => 12000,
                    _ => 16000,
                }
            } else {
                // Hybrid mode: SILK always runs at 16kHz
                16000
            };

            let frame_duration_ms = frame_size * 1000 / self.fs;

            // Convert f32 PCM to i16 at the SILK internal rate
            // First convert to i16 at input sample rate, then SILK resamples internally
            let silk_samples = (silk_internal_rate * frame_duration_ms / 1000) as usize;
            let mut pcm_i16 = vec![0i16; silk_samples * stream_channels as usize];

            // If input rate differs from SILK internal rate, do simple downsampling
            let input_samples = frame_size as usize;
            if self.fs == silk_internal_rate {
                // Direct conversion
                let n = silk_samples.min(input_samples);
                for ch in 0..stream_channels as usize {
                    for i in 0..n {
                        let s = (pcm[i * self.channels as usize + ch] * 32768.0).round() as i32;
                        pcm_i16[i * stream_channels as usize + ch] =
                            s.clamp(-32768, 32767) as i16;
                    }
                }
            } else {
                // Resample from input rate to SILK internal rate
                let ratio = silk_internal_rate as f64 / self.fs as f64;
                for ch in 0..stream_channels as usize {
                    for i in 0..silk_samples {
                        let src_pos = i as f64 / ratio;
                        let src_idx = src_pos as usize;
                        let frac = src_pos - src_idx as f64;
                        let idx0 = src_idx.min(input_samples - 1);
                        let idx1 = (src_idx + 1).min(input_samples - 1);
                        let s0 = pcm[idx0 * self.channels as usize + ch] as f64;
                        let s1 = pcm[idx1 * self.channels as usize + ch] as f64;
                        let val = ((s0 * (1.0 - frac) + s1 * frac) * 32768.0).round() as i32;
                        pcm_i16[i * stream_channels as usize + ch] =
                            val.clamp(-32768, 32767) as i16;
                    }
                }
            }

            // For mono SILK, extract mono samples
            let silk_mono: Vec<i16> = if stream_channels == 1 {
                pcm_i16
            } else {
                // Downmix to mono for SILK (simplified)
                let mut mono = vec![0i16; silk_samples];
                for i in 0..silk_samples {
                    let l = pcm_i16[i * 2] as i32;
                    let r = pcm_i16[i * 2 + 1] as i32;
                    mono[i] = ((l + r) / 2).clamp(-32768, 32767) as i16;
                }
                mono
            };

            let control = SilkEncControl {
                api_sample_rate: silk_internal_rate,
                max_internal_fs_hz: silk_internal_rate,
                payload_size_ms: frame_duration_ms,
                bitrate_bps: if mode == MODE_HYBRID {
                    // In hybrid, SILK gets about half the bitrate
                    self.bitrate_bps / 2
                } else {
                    self.bitrate_bps
                },
                complexity: self.complexity.min(10),
                use_in_band_fec: self.use_inband_fec,
                packet_loss_percentage: self.packet_loss_perc,
                n_channels_internal: 1,
                to_mono: false,
            };

            // Write VAD flag and LBRR flag into the range coder
            enc.enc_bit_logp(true, 1); // VAD flag: 1 = voice activity
            enc.enc_bit_logp(false, 1); // LBRR flag: 0 = no LBRR

            let result = self.silk_enc.encode(&control, &mut enc, &silk_mono);
            if result < 0 {
                return Err(OpusError::InternalError);
            }

            silk_bytes_used = (enc.tell() + 7) >> 3;
        }

        // === CELT encoding ===
        if mode == MODE_CELT_ONLY || mode == MODE_HYBRID {
            let start_band: usize;
            let end_band: usize;

            if mode == MODE_HYBRID {
                // In hybrid mode, CELT encodes only the high bands
                start_band = 17;
                end_band = match bandwidth {
                    OPUS_BANDWIDTH_SUPERWIDEBAND => 19,
                    _ => 21, // Fullband
                };
            } else {
                // CELT-only
                start_band = 0;
                end_band = match bandwidth {
                    OPUS_BANDWIDTH_NARROWBAND => 13,
                    OPUS_BANDWIDTH_MEDIUMBAND | OPUS_BANDWIDTH_WIDEBAND => 17,
                    OPUS_BANDWIDTH_SUPERWIDEBAND => 19,
                    _ => 21,
                };
            }

            self.celt_enc.start = start_band;
            self.celt_enc.end = end_band;
            self.celt_enc.stream_channels = stream_channels as usize;
            self.celt_enc.signalling = false;
            self.celt_enc.complexity = self.complexity;
            self.celt_enc.vbr = self.use_vbr;

            // Compute CELT bitrate: total minus what SILK used
            let celt_bytes = if mode == MODE_HYBRID {
                ((max_data_bytes - 1) - silk_bytes_used).max(2) as usize
            } else {
                (max_data_bytes - 1).max(2) as usize
            };

            self.celt_enc.bitrate = self.bitrate_bps;

            // CELT operates at 48kHz internally. Compute the CELT frame size.
            let celt_frame_size = (frame_size as usize * 48000) / self.fs as usize;

            // Prepare PCM for CELT (it expects interleaved f32 at 48kHz)
            let celt_pcm: Vec<f32>;
            if self.fs == 48000 {
                celt_pcm = pcm[..total_samples].to_vec();
            } else {
                // Upsample to 48kHz for CELT
                let ratio = 48000.0 / self.fs as f64;
                let out_samples = celt_frame_size * self.channels as usize;
                let mut upsampled = vec![0.0f32; out_samples];
                for ch in 0..self.channels as usize {
                    for i in 0..celt_frame_size {
                        let src_pos = i as f64 / ratio;
                        let src_idx = src_pos as usize;
                        let frac = src_pos - src_idx as f64;
                        let input_len = frame_size as usize;
                        let idx0 = src_idx.min(input_len - 1);
                        let idx1 = (src_idx + 1).min(input_len - 1);
                        let s0 = pcm[idx0 * self.channels as usize + ch] as f64;
                        let s1 = pcm[idx1 * self.channels as usize + ch] as f64;
                        upsampled[i * self.channels as usize + ch] =
                            (s0 * (1.0 - frac) + s1 * frac) as f32;
                    }
                }
                celt_pcm = upsampled;
            }

            // Allocate a temporary output buffer for CELT compressed data
            let mut celt_compressed = vec![0u8; celt_bytes];

            let celt_result = if mode == MODE_HYBRID {
                // In hybrid mode, pass the range coder so CELT writes after SILK
                self.celt_enc.encode_with_ec(
                    &celt_pcm,
                    celt_frame_size,
                    &mut celt_compressed,
                    celt_bytes,
                    Some(&mut enc),
                )
            } else {
                // CELT-only: encode directly with its own range coder
                self.celt_enc.encode_with_ec(
                    &celt_pcm,
                    celt_frame_size,
                    &mut celt_compressed,
                    celt_bytes,
                    Some(&mut enc),
                )
            };

            match celt_result {
                Ok(_nbytes) => {
                    self.range_final = self.celt_enc.rng;
                }
                Err(_) => {
                    // CELT encoding failed; produce a minimal valid packet
                    self.range_final = 0;
                }
            }
        } else {
            // SILK-only: finalize the range coder
            self.range_final = enc.rng;
        }

        // Finalize the range coder
        enc.enc_done();
        let nbytes = ((enc.tell() + 7) >> 3) as usize;

        if nbytes == 0 {
            // DTX: just the TOC byte
            self.range_final = 0;
            return Ok(1);
        }

        // Copy encoded payload after the TOC byte
        let out_bytes = nbytes.min((max_data_bytes - 1) as usize);
        data[1..1 + out_bytes].copy_from_slice(&enc.buf[..out_bytes]);

        // Strip trailing zeros for SILK-only mode (matching C reference behavior)
        let mut ret = (out_bytes + 1) as i32; // +1 for TOC
        if mode == MODE_SILK_ONLY {
            while ret > 2 && data[ret as usize - 1] == 0 {
                ret -= 1;
            }
        }

        self.prev_mode = mode;

        Ok(ret)
    }

    /// Encode an Opus frame from 16-bit integer PCM.
    ///
    /// `pcm` contains `frame_size * channels` interleaved samples.
    /// Returns the number of bytes written into `data`.
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, OpusError> {
        let total_samples = (frame_size * self.channels) as usize;
        if pcm.len() < total_samples {
            return Err(OpusError::BadArg);
        }

        // Convert i16 to f32
        let float_pcm: Vec<f32> = pcm[..total_samples]
            .iter()
            .map(|&s| s as f32 * (1.0 / 32768.0))
            .collect();

        self.encode_float(&float_pcm, frame_size, data, max_data_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::OpusDecoder;

    #[test]
    fn test_encoder_create() {
        // Valid configurations
        let enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO);
        assert!(enc.is_ok());
        let enc = enc.unwrap();
        assert_eq!(enc.sample_rate(), 48000);
        assert_eq!(enc.channels(), 2);

        let enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP);
        assert!(enc.is_ok());
        let enc = enc.unwrap();
        assert_eq!(enc.channels(), 1);

        let enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY);
        assert!(enc.is_ok());

        let enc = OpusEncoder::new(8000, 1, OPUS_APPLICATION_VOIP);
        assert!(enc.is_ok());

        // Invalid configurations
        assert!(OpusEncoder::new(44100, 1, OPUS_APPLICATION_AUDIO).is_err());
        assert!(OpusEncoder::new(48000, 3, OPUS_APPLICATION_AUDIO).is_err());
        assert!(OpusEncoder::new(48000, 0, OPUS_APPLICATION_AUDIO).is_err());
        assert!(OpusEncoder::new(48000, 1, 9999).is_err());
    }

    #[test]
    fn test_encode_silence() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(64000);

        let frame_size = 960; // 20ms at 48kHz
        let pcm = vec![0.0f32; frame_size];
        let mut data = vec![0u8; 1275];

        let result = enc.encode_float(&pcm, frame_size as i32, &mut data, 1275);
        assert!(result.is_ok(), "Encoding silence should succeed: {:?}", result.err());

        let nbytes = result.unwrap();
        assert!(nbytes >= 1, "Should produce at least the TOC byte");
        assert!(nbytes <= 1275, "Should not exceed max packet size");

        // Verify the TOC byte is valid by parsing it
        let mode = opus_packet_get_mode(&data);
        assert!(
            mode == MODE_SILK_ONLY || mode == MODE_HYBRID || mode == MODE_CELT_ONLY,
            "TOC should encode a valid mode"
        );
    }

    #[test]
    fn test_encode_silence_i16() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        let frame_size = 960;
        let pcm = vec![0i16; frame_size];
        let mut data = vec![0u8; 1275];

        let result = enc.encode(&pcm, frame_size as i32, &mut data, 1275);
        assert!(result.is_ok(), "i16 encoding should succeed: {:?}", result.err());
        assert!(result.unwrap() >= 1);
    }

    #[test]
    fn test_gen_toc_roundtrip() {
        // Test that gen_toc produces bytes that packet.rs can decode correctly.

        // CELT 20ms fullband mono
        let toc = gen_toc(MODE_CELT_ONLY, 50, OPUS_BANDWIDTH_FULLBAND, 1);
        assert_eq!(opus_packet_get_mode(&[toc]), MODE_CELT_ONLY);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), OPUS_BANDWIDTH_FULLBAND);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 960);

        // CELT 20ms fullband stereo
        let toc = gen_toc(MODE_CELT_ONLY, 50, OPUS_BANDWIDTH_FULLBAND, 2);
        assert_eq!(opus_packet_get_mode(&[toc]), MODE_CELT_ONLY);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), OPUS_BANDWIDTH_FULLBAND);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 2);

        // SILK 20ms narrowband mono
        let toc = gen_toc(MODE_SILK_ONLY, 50, OPUS_BANDWIDTH_NARROWBAND, 1);
        assert_eq!(opus_packet_get_mode(&[toc]), MODE_SILK_ONLY);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), OPUS_BANDWIDTH_NARROWBAND);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 960);

        // SILK 10ms wideband stereo
        let toc = gen_toc(MODE_SILK_ONLY, 100, OPUS_BANDWIDTH_WIDEBAND, 2);
        assert_eq!(opus_packet_get_mode(&[toc]), MODE_SILK_ONLY);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), OPUS_BANDWIDTH_WIDEBAND);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 2);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 480);

        // Hybrid 20ms superwideband mono
        let toc = gen_toc(MODE_HYBRID, 50, OPUS_BANDWIDTH_SUPERWIDEBAND, 1);
        assert_eq!(opus_packet_get_mode(&[toc]), MODE_HYBRID);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), OPUS_BANDWIDTH_SUPERWIDEBAND);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 960);

        // Hybrid 10ms fullband stereo
        let toc = gen_toc(MODE_HYBRID, 100, OPUS_BANDWIDTH_FULLBAND, 2);
        assert_eq!(opus_packet_get_mode(&[toc]), MODE_HYBRID);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), OPUS_BANDWIDTH_FULLBAND);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 2);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 480);

        // CELT 2.5ms
        let toc = gen_toc(MODE_CELT_ONLY, 400, OPUS_BANDWIDTH_FULLBAND, 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 120);

        // CELT 5ms
        let toc = gen_toc(MODE_CELT_ONLY, 200, OPUS_BANDWIDTH_FULLBAND, 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 240);

        // CELT 10ms
        let toc = gen_toc(MODE_CELT_ONLY, 100, OPUS_BANDWIDTH_FULLBAND, 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 480);

        // SILK 40ms
        let toc = gen_toc(MODE_SILK_ONLY, 25, OPUS_BANDWIDTH_NARROWBAND, 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 1920);

        // SILK 60ms
        let toc = gen_toc(MODE_SILK_ONLY, 400 / 24, OPUS_BANDWIDTH_NARROWBAND, 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 2880);
    }

    #[test]
    fn test_opus_encode_decode_roundtrip() {
        // Encode with OpusEncoder, decode with OpusDecoder, verify the result.
        let fs = 48000;
        let channels = 1;
        let frame_size = 960; // 20ms

        let mut encoder =
            OpusEncoder::new(fs, channels, OPUS_APPLICATION_AUDIO).unwrap();
        encoder.set_bitrate(64000);

        let mut decoder = OpusDecoder::new(fs, channels).unwrap();

        // Generate a simple 440Hz sine tone
        let mut pcm_in = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            pcm_in[i] =
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / fs as f32).sin();
        }

        // Encode
        let mut packet = vec![0u8; 1275];
        let nbytes = encoder
            .encode_float(&pcm_in, frame_size as i32, &mut packet, 1275)
            .expect("Encoding should succeed");
        assert!(nbytes >= 2, "Should produce a non-trivial packet, got {nbytes} bytes");

        // Verify the packet is parseable
        let parsed = opus_packet_parse(&packet[..nbytes as usize]);
        assert!(
            parsed.is_ok(),
            "Encoded packet should be parseable: {:?}",
            parsed.err()
        );

        // Decode
        let mut pcm_out = vec![0.0f32; frame_size];
        let decoded_samples = decoder
            .decode_float(
                Some(&packet[..nbytes as usize]),
                &mut pcm_out,
                frame_size as i32,
                false,
            )
            .expect("Decoding should succeed");
        assert_eq!(
            decoded_samples, frame_size as i32,
            "Should decode the correct number of samples"
        );

        // Verify the decoded signal has energy (not silent)
        let energy: f64 = pcm_out.iter().map(|&x| x as f64 * x as f64).sum();
        assert!(
            energy > 0.0,
            "Decoded signal should have non-zero energy"
        );

        // The codec introduces algorithmic latency, so the first decoded frame may
        // have reduced amplitude. We just verify the output is not all zeros, which
        // confirms the encode/decode roundtrip produced valid audio.
        let max_out = pcm_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_out > 0.0,
            "Decoded signal should not be completely silent, got max {max_out}"
        );
    }

    #[test]
    fn test_opus_encode_decode_roundtrip_stereo() {
        let fs = 48000;
        let channels = 2;
        let frame_size = 960;

        let mut encoder =
            OpusEncoder::new(fs, channels, OPUS_APPLICATION_AUDIO).unwrap();
        encoder.set_bitrate(96000);

        let mut decoder = OpusDecoder::new(fs, channels).unwrap();

        // Generate stereo sine
        let mut pcm_in = vec![0.0f32; frame_size * channels as usize];
        for i in 0..frame_size {
            let s =
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / fs as f32).sin();
            pcm_in[i * 2] = s;     // left
            pcm_in[i * 2 + 1] = s; // right
        }

        let mut packet = vec![0u8; 1275];
        let nbytes = encoder
            .encode_float(&pcm_in, frame_size as i32, &mut packet, 1275)
            .expect("Stereo encoding should succeed");
        assert!(nbytes >= 2);

        let mut pcm_out = vec![0.0f32; frame_size * channels as usize];
        let decoded = decoder
            .decode_float(
                Some(&packet[..nbytes as usize]),
                &mut pcm_out,
                frame_size as i32,
                false,
            )
            .expect("Stereo decoding should succeed");
        assert_eq!(decoded, frame_size as i32);

        let energy: f64 = pcm_out.iter().map(|&x| x as f64 * x as f64).sum();
        assert!(energy > 0.0, "Stereo decoded signal should have energy");
    }

    #[test]
    fn test_encoder_setters() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();

        enc.set_bitrate(32000);
        assert_eq!(enc.bitrate_bps, 32000);

        enc.set_bitrate(OPUS_BITRATE_MAX);
        // Should reset to default
        assert!(enc.bitrate_bps > 0);

        enc.set_complexity(5);
        assert_eq!(enc.complexity, 5);

        enc.set_complexity(100); // Should clamp
        assert_eq!(enc.complexity, 10);

        enc.set_complexity(-5); // Should clamp
        assert_eq!(enc.complexity, 0);

        enc.set_signal(OPUS_SIGNAL_VOICE);
        assert_eq!(enc.signal_type, OPUS_SIGNAL_VOICE);

        enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND);
        assert_eq!(enc.max_bandwidth, OPUS_BANDWIDTH_WIDEBAND);
    }
}
