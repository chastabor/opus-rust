// Port of silk/dec_API.c, silk/decode_frame.c - SILK Decoder

use crate::tables::*;
use crate::*;
use opus_range_coder::EcCtx;

/// Decoder control structure (passed from the Opus layer)
#[derive(Clone)]
pub struct SilkDecControl {
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub api_sample_rate: i32,
    pub internal_sample_rate: i32,
    pub payload_size_ms: i32,
    pub prev_pitch_lag: i32,
}

impl Default for SilkDecControl {
    fn default() -> Self {
        Self {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: 48000,
            internal_sample_rate: 16000,
            payload_size_ms: 20,
            prev_pitch_lag: 0,
        }
    }
}

/// Post-filter hook called after decode_core, before PLC/CNG.
/// Allows the opus crate to inject OSCE enhancement without
/// opus-silk depending on opus-dnn.
pub trait SilkPostFilter {
    fn enhance_frame(
        &mut self,
        p_out: &mut [i16],
        channel: &ChannelState,
        control: &DecoderControl,
        num_bits: i32,
    );
}

/// No-op post-filter (used when DNN/OSCE is not active).
pub struct NoPostFilter;

impl SilkPostFilter for NoPostFilter {
    #[inline]
    fn enhance_frame(&mut self, _: &mut [i16], _: &ChannelState, _: &DecoderControl, _: i32) {}
}

/// SILK Decoder super struct
pub struct SilkDecoder {
    pub channel_state: [ChannelState; DECODER_NUM_CHANNELS],
    pub s_stereo: StereoDecState,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub prev_decode_only_middle: bool,
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SilkDecoder {
    pub fn new() -> Self {
        Self {
            channel_state: [ChannelState::new(), ChannelState::new()],
            s_stereo: StereoDecState::default(),
            n_channels_api: 0,
            n_channels_internal: 0,
            prev_decode_only_middle: false,
        }
    }

    pub fn reset(&mut self) {
        for ch in &mut self.channel_state {
            ch.reset();
        }
        self.s_stereo = StereoDecState::default();
        self.prev_decode_only_middle = false;
    }

    /// Decode a frame of SILK audio
    pub fn decode(
        &mut self,
        dec_control: &mut SilkDecControl,
        lost_flag: i32,
        new_packet_flag: bool,
        ps_range_dec: &mut EcCtx,
        samples_out: &mut [i16],
        n_samples_out: &mut i32,
        post_filter: &mut impl SilkPostFilter,
    ) -> i32 {
        let mut ret = 0i32;
        let mut decode_only_middle = false;

        let n_ch_internal = dec_control.n_channels_internal;
        let n_ch_api = dec_control.n_channels_api;

        // Test if first frame in payload
        if new_packet_flag {
            for n in 0..n_ch_internal as usize {
                self.channel_state[n].n_frames_decoded = 0;
            }
        }

        // If Mono -> Stereo transition
        if n_ch_internal > self.n_channels_internal {
            self.channel_state[1] = ChannelState::new();
        }

        let _stereo_to_mono = n_ch_internal == 1
            && self.n_channels_internal == 2
            && dec_control.internal_sample_rate == 1000 * self.channel_state[0].fs_khz;

        if self.channel_state[0].n_frames_decoded == 0 {
            for n in 0..n_ch_internal as usize {
                match dec_control.payload_size_ms {
                    0 | 10 => {
                        self.channel_state[n].n_frames_per_packet = 1;
                        self.channel_state[n].nb_subfr = 2;
                    }
                    20 => {
                        self.channel_state[n].n_frames_per_packet = 1;
                        self.channel_state[n].nb_subfr = 4;
                    }
                    40 => {
                        self.channel_state[n].n_frames_per_packet = 2;
                        self.channel_state[n].nb_subfr = 4;
                    }
                    60 => {
                        self.channel_state[n].n_frames_per_packet = 3;
                        self.channel_state[n].nb_subfr = 4;
                    }
                    _ => return -1,
                }
                let fs_khz_dec = (dec_control.internal_sample_rate >> 10) + 1;
                if fs_khz_dec != 8 && fs_khz_dec != 12 && fs_khz_dec != 16 {
                    return -1;
                }
                self.channel_state[n].set_fs(fs_khz_dec, dec_control.api_sample_rate);
            }
        }

        if n_ch_api == 2
            && n_ch_internal == 2
            && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.s_stereo.pred_prev_q13 = [0; 2];
            self.s_stereo.s_side = [0; 2];
            self.channel_state[1].resampler_state = self.channel_state[0].resampler_state.clone();
        }
        self.n_channels_api = n_ch_api;
        self.n_channels_internal = n_ch_internal;

        if dec_control.api_sample_rate > MAX_API_FS_KHZ as i32 * 1000
            || dec_control.api_sample_rate < 8000
        {
            return -1;
        }

        let mut ms_pred_q13 = [0i32; 2];

        if lost_flag != FLAG_PACKET_LOST && self.channel_state[0].n_frames_decoded == 0 {
            // First decoder call: decode VAD flags and LBRR flag
            for n in 0..n_ch_internal as usize {
                for i in 0..self.channel_state[n].n_frames_per_packet as usize {
                    self.channel_state[n].vad_flags[i] =
                        if ps_range_dec.dec_bit_logp(1) { 1 } else { 0 };
                }
                self.channel_state[n].lbrr_flag = if ps_range_dec.dec_bit_logp(1) { 1 } else { 0 };
            }

            // Decode LBRR flags
            for n in 0..n_ch_internal as usize {
                self.channel_state[n].lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
                if self.channel_state[n].lbrr_flag != 0 {
                    let nfpp = self.channel_state[n].n_frames_per_packet;
                    if nfpp == 1 {
                        self.channel_state[n].lbrr_flags[0] = 1;
                    } else {
                        let icdf = if nfpp == 2 {
                            &SILK_LBRR_FLAGS_2_ICDF[..]
                        } else {
                            &SILK_LBRR_FLAGS_3_ICDF[..]
                        };
                        let lbrr_symbol = ps_range_dec.dec_icdf(icdf, 8) as i32 + 1;
                        for i in 0..nfpp as usize {
                            self.channel_state[n].lbrr_flags[i] = (lbrr_symbol >> i) & 1;
                        }
                    }
                }
            }

            if lost_flag == FLAG_DECODE_NORMAL {
                // Skip all LBRR data
                let nfpp = self.channel_state[0].n_frames_per_packet;
                for i in 0..nfpp as usize {
                    for n in 0..n_ch_internal as usize {
                        if self.channel_state[n].lbrr_flags[i] != 0 {
                            if n_ch_internal == 2 && n == 0 {
                                stereo::silk_stereo_decode_pred(ps_range_dec, &mut ms_pred_q13);
                                if self.channel_state[1].lbrr_flags[i] == 0 {
                                    let _ = stereo::silk_stereo_decode_mid_only(ps_range_dec);
                                }
                            }
                            let cond = if i > 0 && self.channel_state[n].lbrr_flags[i - 1] != 0 {
                                CODE_CONDITIONALLY
                            } else {
                                CODE_INDEPENDENTLY
                            };
                            decode_indices::silk_decode_indices(
                                &mut self.channel_state[n],
                                ps_range_dec,
                                i as i32,
                                true,
                                cond,
                            );
                            let fl = self.channel_state[n].frame_length;
                            let sig_type = self.channel_state[n].indices.signal_type as i32;
                            let qot = self.channel_state[n].indices.quant_offset_type as i32;
                            let mut dummy_pulses = [0i16; MAX_FRAME_LENGTH];
                            decode_pulses::silk_decode_pulses(
                                ps_range_dec,
                                &mut dummy_pulses,
                                sig_type,
                                qot,
                                fl,
                            );
                        }
                    }
                }
            }
        }

        // Get MS predictor index
        if n_ch_internal == 2 {
            let nfd = self.channel_state[0].n_frames_decoded;
            if lost_flag == FLAG_DECODE_NORMAL
                || (lost_flag == FLAG_DECODE_LBRR
                    && self.channel_state[0].lbrr_flags[nfd as usize] == 1)
            {
                stereo::silk_stereo_decode_pred(ps_range_dec, &mut ms_pred_q13);
                if (lost_flag == FLAG_DECODE_NORMAL
                    && self.channel_state[1].vad_flags[nfd as usize] == 0)
                    || (lost_flag == FLAG_DECODE_LBRR
                        && self.channel_state[1].lbrr_flags[nfd as usize] == 0)
                {
                    decode_only_middle = stereo::silk_stereo_decode_mid_only(ps_range_dec) != 0;
                }
            } else {
                ms_pred_q13[0] = self.s_stereo.pred_prev_q13[0] as i32;
                ms_pred_q13[1] = self.s_stereo.pred_prev_q13[1] as i32;
            }
        }

        // Reset side channel if needed
        if n_ch_internal == 2 && !decode_only_middle && self.prev_decode_only_middle {
            self.channel_state[1].out_buf.fill(0);
            self.channel_state[1].s_lpc_q14_buf = [0; MAX_LPC_ORDER];
            self.channel_state[1].lag_prev = 100;
            self.channel_state[1].last_gain_index = 10;
            self.channel_state[1].prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            self.channel_state[1].first_frame_after_reset = true;
        }

        let mut n_samples_out_dec = 0i32;

        // Temp buffers for each channel (stack-allocated, max 2 channels)
        let mut samples_out1 = [[0i16; MAX_FRAME_LENGTH + 2]; 2];

        let has_side = if lost_flag == FLAG_DECODE_NORMAL {
            !decode_only_middle
        } else {
            !self.prev_decode_only_middle
        };

        // Decode each channel
        for (n, samples_out1_n) in samples_out1
            .iter_mut()
            .enumerate()
            .take(n_ch_internal as usize)
        {
            if n == 0 || has_side {
                let frame_idx = self.channel_state[0].n_frames_decoded - n as i32;
                let cond_coding = if frame_idx <= 0 {
                    CODE_INDEPENDENTLY
                } else if lost_flag == FLAG_DECODE_LBRR {
                    if self.channel_state[n].lbrr_flags[(frame_idx - 1) as usize] != 0 {
                        CODE_CONDITIONALLY
                    } else {
                        CODE_INDEPENDENTLY
                    }
                } else if n > 0 && self.prev_decode_only_middle {
                    CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };

                ret += silk_decode_frame(
                    &mut self.channel_state[n],
                    ps_range_dec,
                    &mut samples_out1_n[2..],
                    &mut n_samples_out_dec,
                    lost_flag,
                    cond_coding,
                    post_filter,
                );
            } else {
                samples_out1_n[2..2 + n_samples_out_dec as usize].fill(0);
            }
            self.channel_state[n].n_frames_decoded += 1;
        }

        if n_ch_api == 2 && n_ch_internal == 2 {
            // Convert Mid/Side to Left/Right
            // Split the vec to get two mutable references
            let (first, rest) = samples_out1.split_at_mut(1);
            stereo::silk_stereo_ms_to_lr(
                &mut self.s_stereo,
                &mut first[0],
                &mut rest[0],
                &ms_pred_q13,
                self.channel_state[0].fs_khz,
                n_samples_out_dec as usize,
            );
        } else {
            // Buffering for mono
            let buf = &mut samples_out1[0];
            buf[0] = self.s_stereo.s_mid[0];
            buf[1] = self.s_stereo.s_mid[1];
            self.s_stereo.s_mid[0] = buf[n_samples_out_dec as usize];
            self.s_stereo.s_mid[1] = buf[n_samples_out_dec as usize + 1];
        }

        // Number of output samples
        let internal_rate = self.channel_state[0].fs_khz * 1000;
        *n_samples_out = if internal_rate > 0 {
            (n_samples_out_dec as i64 * dec_control.api_sample_rate as i64 / internal_rate as i64)
                as i32
        } else {
            n_samples_out_dec
        };

        let n_out = *n_samples_out as usize;

        // Resample and interleave
        let n_ch_min = (n_ch_api as usize).min(n_ch_internal as usize);
        const MAX_RESAMPLE_OUT: usize = MAX_FRAME_LENGTH_MS * 48; // 20ms at 48kHz
        let mut resample_buf = [0i16; MAX_RESAMPLE_OUT];

        for n in 0..n_ch_min {
            resampler::silk_resampler(
                &mut self.channel_state[n].resampler_state,
                &mut resample_buf,
                &samples_out1[n][1..],
                n_samples_out_dec as usize,
            );

            if n_ch_api == 2 {
                for i in 0..n_out {
                    if n + 2 * i < samples_out.len() {
                        samples_out[n + 2 * i] = resample_buf[i];
                    }
                }
            } else {
                let copy_len = n_out.min(samples_out.len());
                samples_out[..copy_len].copy_from_slice(&resample_buf[..copy_len]);
            }
        }

        // Create stereo from mono if needed
        if n_ch_api == 2 && n_ch_internal == 1 {
            for i in 0..n_out {
                if 1 + 2 * i < samples_out.len() && 2 * i < samples_out.len() {
                    samples_out[1 + 2 * i] = samples_out[2 * i];
                }
            }
        }

        // Export pitch lag
        if self.channel_state[0].prev_signal_type == TYPE_VOICED {
            let mult_tab = [6, 4, 3];
            let idx = ((self.channel_state[0].fs_khz - 8) >> 2) as usize;
            dec_control.prev_pitch_lag = self.channel_state[0].lag_prev * mult_tab[idx.min(2)];
        } else {
            dec_control.prev_pitch_lag = 0;
        }

        if lost_flag == FLAG_PACKET_LOST {
            for i in 0..self.n_channels_internal as usize {
                self.channel_state[i].last_gain_index = 10;
            }
        } else {
            self.prev_decode_only_middle = decode_only_middle;
        }

        ret
    }
}

/// Decode a single frame
fn silk_decode_frame(
    ps_dec: &mut ChannelState,
    ps_range_dec: &mut EcCtx,
    p_out: &mut [i16],
    p_n: &mut i32,
    lost_flag: i32,
    cond_coding: i32,
    post_filter: &mut impl SilkPostFilter,
) -> i32 {
    let l = ps_dec.frame_length as usize;
    let mut ps_dec_ctrl = DecoderControl::default();

    if lost_flag == FLAG_DECODE_NORMAL
        || (lost_flag == FLAG_DECODE_LBRR
            && ps_dec.lbrr_flags[ps_dec.n_frames_decoded as usize] == 1)
    {
        let ec_start = ps_range_dec.tell();

        // Decode indices
        decode_indices::silk_decode_indices(
            ps_dec,
            ps_range_dec,
            ps_dec.n_frames_decoded,
            lost_flag != 0,
            cond_coding,
        );

        // Decode pulses
        let mut pulses = [0i16; MAX_FRAME_LENGTH];
        decode_pulses::silk_decode_pulses(
            ps_range_dec,
            &mut pulses,
            ps_dec.indices.signal_type as i32,
            ps_dec.indices.quant_offset_type as i32,
            ps_dec.frame_length,
        );

        // Decode parameters
        decode_params::silk_decode_parameters(ps_dec, &mut ps_dec_ctrl, cond_coding);

        // Run inverse NSQ
        decode_core::silk_decode_core(ps_dec, &ps_dec_ctrl, p_out, &pulses);

        // OSCE post-filter: enhance decoded output before PLC/CNG
        // Matches C: osce_enhance_frame() call in silk/decode_frame.c:113
        let num_bits = ps_range_dec.tell() - ec_start;
        post_filter.enhance_frame(p_out, ps_dec, &ps_dec_ctrl, num_bits);

        // Update output buffer
        let mv_len = (ps_dec.ltp_mem_length - ps_dec.frame_length) as usize;
        let fl = ps_dec.frame_length as usize;
        // memmove outBuf left
        ps_dec.out_buf.copy_within(fl..fl + mv_len, 0);
        // copy decoded output to end of outBuf
        ps_dec.out_buf[mv_len..mv_len + fl].copy_from_slice(&p_out[..fl]);

        // Update PLC state (good frame)
        plc::silk_plc(ps_dec, &mut ps_dec_ctrl, p_out, false);

        ps_dec.loss_cnt = 0;
        ps_dec.prev_signal_type = ps_dec.indices.signal_type as i32;
        ps_dec.first_frame_after_reset = false;
    } else {
        // Packet loss: use PLC
        plc::silk_plc(ps_dec, &mut ps_dec_ctrl, p_out, true);

        // Update output buffer
        let mv_len = (ps_dec.ltp_mem_length - ps_dec.frame_length) as usize;
        let fl = ps_dec.frame_length as usize;
        ps_dec.out_buf.copy_within(fl..fl + mv_len, 0);
        ps_dec.out_buf[mv_len..mv_len + fl].copy_from_slice(&p_out[..fl]);
    }

    // CNG
    cng::silk_cng(ps_dec, &ps_dec_ctrl, p_out, l);

    // PLC glue frames
    plc::silk_plc_glue_frames(ps_dec, p_out, l);

    // Update lag
    ps_dec.lag_prev = ps_dec_ctrl.pitch_l[ps_dec.nb_subfr as usize - 1];

    *p_n = l as i32;
    0
}
