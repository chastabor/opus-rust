//! Multistream Opus decoder for decoding packets with more than 2 channels.
//!
//! Port of C opus_multistream_decoder.c and opus_multistream.c.

use crate::decoder::OpusDecoder;
use crate::error::OpusError;
use crate::packet::{opus_packet_get_nb_samples, opus_packet_parse_self_delimited};
use crate::types::*;

/// Channel layout for multistream packets.
#[derive(Clone)]
pub struct ChannelLayout {
    pub nb_channels: usize,
    pub nb_streams: usize,
    pub nb_coupled_streams: usize,
    pub mapping: [u8; 256],
}

impl ChannelLayout {
    /// Validate that the channel layout is consistent.
    pub(crate) fn validate(&self) -> bool {
        let max_channel = self.nb_streams + self.nb_coupled_streams;
        if max_channel > 255 {
            return false;
        }
        for i in 0..self.nb_channels {
            if self.mapping[i] as usize >= max_channel && self.mapping[i] != 255 {
                return false;
            }
        }
        true
    }
}

/// Find the output channel index for the left channel of a coupled stream.
fn get_left_channel(layout: &ChannelLayout, stream_id: usize, prev: i32) -> i32 {
    let start = if prev < 0 { 0 } else { prev as usize + 1 };
    for i in start..layout.nb_channels {
        if layout.mapping[i] as usize == stream_id * 2 {
            return i as i32;
        }
    }
    -1
}

/// Find the output channel index for the right channel of a coupled stream.
fn get_right_channel(layout: &ChannelLayout, stream_id: usize, prev: i32) -> i32 {
    let start = if prev < 0 { 0 } else { prev as usize + 1 };
    for i in start..layout.nb_channels {
        if layout.mapping[i] as usize == stream_id * 2 + 1 {
            return i as i32;
        }
    }
    -1
}

/// Find the output channel index for a mono (uncoupled) stream.
fn get_mono_channel(layout: &ChannelLayout, stream_id: usize, prev: i32) -> i32 {
    let start = if prev < 0 { 0 } else { prev as usize + 1 };
    for i in start..layout.nb_channels {
        if layout.mapping[i] as usize == stream_id + layout.nb_coupled_streams {
            return i as i32;
        }
    }
    -1
}

/// Multistream Opus decoder.
///
/// Decodes Opus multistream packets containing multiple independent streams
/// mapped to output channels. Supports up to 255 channels.
pub struct OpusMSDecoder {
    layout: ChannelLayout,
    fs: i32,
    decoders: Vec<OpusDecoder>,
}

impl OpusMSDecoder {
    /// Create a new multistream decoder.
    ///
    /// - `fs`: Sample rate (8000, 12000, 16000, 24000, or 48000 Hz)
    /// - `channels`: Total number of output channels (1-255)
    /// - `streams`: Number of independent streams in the packet
    /// - `coupled_streams`: Number of stereo (2-channel) streams
    /// - `mapping`: Channel mapping table (`channels` entries)
    ///
    /// The first `coupled_streams` streams are decoded as stereo pairs,
    /// the remaining `streams - coupled_streams` as mono.
    pub fn new(
        sample_rate: SampleRate,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
    ) -> Result<Self, OpusError> {
        let fs = i32::from(sample_rate);
        if channels == 0 || channels > 255 {
            return Err(OpusError::BadArg);
        }
        if streams == 0 || streams > 255 {
            return Err(OpusError::BadArg);
        }
        if coupled_streams > streams {
            return Err(OpusError::BadArg);
        }
        if streams > 255 - coupled_streams {
            return Err(OpusError::BadArg);
        }
        if mapping.len() < channels {
            return Err(OpusError::BadArg);
        }

        let mut layout = ChannelLayout {
            nb_channels: channels,
            nb_streams: streams,
            nb_coupled_streams: coupled_streams,
            mapping: [0u8; 256],
        };
        layout.mapping[..channels].copy_from_slice(&mapping[..channels]);

        if !layout.validate() {
            return Err(OpusError::BadArg);
        }

        // Create per-stream decoders
        let mut decoders = Vec::with_capacity(streams);
        for s in 0..streams {
            let ch = if s < coupled_streams {
                Channels::Stereo
            } else {
                Channels::Mono
            };
            let dec = OpusDecoder::new(sample_rate, ch)?;
            decoders.push(dec);
        }

        Ok(OpusMSDecoder {
            layout,
            fs,
            decoders,
        })
    }

    /// Validate a multistream packet: ensure all streams have the same duration.
    fn validate_packet(&self, data: &[u8]) -> Result<i32, OpusError> {
        let mut offset = 0usize;
        let mut remaining = data.len();
        let mut samples = 0i32;

        for s in 0..self.layout.nb_streams {
            if remaining == 0 {
                return Err(OpusError::InvalidPacket);
            }
            let is_not_last = s != self.layout.nb_streams - 1;
            let stream_data = &data[offset..offset + remaining];

            let parsed = if is_not_last {
                opus_packet_parse_self_delimited(stream_data)?
            } else {
                crate::packet::opus_packet_parse(stream_data)?
            };

            let pkt_offset = parsed.packet_offset;
            let tmp_samples = opus_packet_get_nb_samples(
                &data[offset..offset + pkt_offset.min(remaining)],
                self.fs,
            )?;

            if s != 0 && samples != tmp_samples {
                return Err(OpusError::InvalidPacket);
            }
            samples = tmp_samples;

            offset += pkt_offset;
            remaining = remaining.saturating_sub(pkt_offset);
        }

        Ok(samples)
    }

    /// Decode a multistream Opus packet to float PCM.
    ///
    /// - `data`: The packet data, or `None` for packet loss concealment
    /// - `pcm`: Output buffer, interleaved float samples (`frame_size * channels`)
    /// - `frame_size`: Maximum number of samples per channel to decode
    ///
    /// Returns the number of decoded samples per channel.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
    ) -> Result<i32, OpusError> {
        if frame_size <= 0 {
            return Err(OpusError::BadArg);
        }

        // Limit frame_size to avoid excessive allocations (max 120ms)
        let frame_size = frame_size.min(self.fs / 25 * 3);
        let nb_channels = self.layout.nb_channels;

        // Temporary decode buffer (max 2 channels per stream)
        let mut buf = vec![0.0f32; 2 * frame_size as usize];

        let do_plc = data.is_none() || data.is_none_or(|d| d.is_empty());

        if let Some(d) = data
            && !do_plc
        {
            if d.len() < 2 * self.layout.nb_streams - 1 {
                return Err(OpusError::InvalidPacket);
            }
            // Validate all streams have same duration
            let validated_samples = self.validate_packet(d)?;
            if validated_samples > frame_size {
                return Err(OpusError::BufferTooSmall);
            }
        }

        let mut offset = 0usize;
        let mut remaining = data.map_or(0, |d| d.len());
        let mut actual_frame_size = frame_size;

        for s in 0..self.layout.nb_streams {
            let dec = &mut self.decoders[s];

            let stream_channels = if s < self.layout.nb_coupled_streams {
                2
            } else {
                1
            };

            let ret = if do_plc {
                // Packet loss concealment
                dec.decode_float(None, &mut buf, frame_size, false)
            } else {
                if remaining == 0 {
                    return Err(OpusError::InternalError);
                }
                let is_not_last = s != self.layout.nb_streams - 1;
                // Safety: do_plc is false here, meaning data is Some (checked on line 198)
                let stream_data = &data.expect("data guaranteed Some by do_plc check")
                    [offset..offset + remaining];

                // Parse to get packet_offset
                let parsed = if is_not_last {
                    opus_packet_parse_self_delimited(stream_data)?
                } else {
                    crate::packet::opus_packet_parse(stream_data)?
                };
                let pkt_offset = parsed.packet_offset;

                // Decode the stream's data
                let ret = dec.decode_float(
                    Some(&stream_data[..pkt_offset]),
                    &mut buf,
                    frame_size,
                    false,
                );

                offset += pkt_offset;
                remaining = remaining.saturating_sub(pkt_offset);

                ret
            };

            let ret = ret?;
            if ret <= 0 {
                return Err(OpusError::InternalError);
            }
            actual_frame_size = ret;

            // Scatter decoded samples to output channels
            if s < self.layout.nb_coupled_streams {
                // Coupled (stereo) stream: scatter left and right channels
                let mut prev = -1i32;
                loop {
                    let chan = get_left_channel(&self.layout, s, prev);
                    if chan == -1 {
                        break;
                    }
                    copy_channel_out_float(
                        pcm,
                        nb_channels,
                        chan as usize,
                        &buf,
                        stream_channels,
                        0,
                        actual_frame_size as usize,
                    );
                    prev = chan;
                }
                prev = -1;
                loop {
                    let chan = get_right_channel(&self.layout, s, prev);
                    if chan == -1 {
                        break;
                    }
                    copy_channel_out_float(
                        pcm,
                        nb_channels,
                        chan as usize,
                        &buf,
                        stream_channels,
                        1,
                        actual_frame_size as usize,
                    );
                    prev = chan;
                }
            } else {
                // Uncoupled (mono) stream
                let mut prev = -1i32;
                loop {
                    let chan = get_mono_channel(&self.layout, s, prev);
                    if chan == -1 {
                        break;
                    }
                    copy_channel_out_float(
                        pcm,
                        nb_channels,
                        chan as usize,
                        &buf,
                        1,
                        0,
                        actual_frame_size as usize,
                    );
                    prev = chan;
                }
            }
        }

        // Handle muted channels (mapping == 255)
        for c in 0..nb_channels {
            if self.layout.mapping[c] == 255 {
                for i in 0..actual_frame_size as usize {
                    pcm[i * nb_channels + c] = 0.0;
                }
            }
        }

        Ok(actual_frame_size)
    }

    /// Get the sample rate of this decoder.
    pub fn sample_rate(&self) -> i32 {
        self.fs
    }

    /// Get the number of output channels.
    pub fn channels(&self) -> usize {
        self.layout.nb_channels
    }

    /// Get the number of streams.
    pub fn streams(&self) -> usize {
        self.layout.nb_streams
    }

    /// Get the number of coupled streams.
    pub fn coupled_streams(&self) -> usize {
        self.layout.nb_coupled_streams
    }
}

/// Copy one channel from a decoded buffer to an interleaved output buffer.
fn copy_channel_out_float(
    dst: &mut [f32],
    dst_stride: usize,
    dst_channel: usize,
    src: &[f32],
    src_stride: usize,
    src_channel: usize,
    frame_size: usize,
) {
    for i in 0..frame_size {
        dst[i * dst_stride + dst_channel] = src[i * src_stride + src_channel];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ms_decoder_create_stereo() {
        // Simple stereo: 1 coupled stream, mapping [0, 1]
        let dec = OpusMSDecoder::new(SampleRate::Hz48000, 2, 1, 1, &[0, 1]).unwrap();
        assert_eq!(dec.channels(), 2);
        assert_eq!(dec.streams(), 1);
        assert_eq!(dec.coupled_streams(), 1);
    }

    #[test]
    fn test_ms_decoder_create_surround51() {
        // 5.1 surround: 6 channels, 4 streams (2 coupled + 2 mono)
        // Vorbis channel order: FL, FC, FR, RL, RR, LFE
        // mapping: [0, 4, 1, 2, 3, 5] (example)
        let dec = OpusMSDecoder::new(SampleRate::Hz48000, 6, 4, 2, &[0, 4, 1, 2, 3, 5]).unwrap();
        assert_eq!(dec.channels(), 6);
        assert_eq!(dec.streams(), 4);
        assert_eq!(dec.coupled_streams(), 2);
    }

    #[test]
    fn test_ms_decoder_invalid_args() {
        // 0 channels
        assert!(OpusMSDecoder::new(SampleRate::Hz48000, 0, 1, 0, &[]).is_err());
        // coupled > streams
        assert!(OpusMSDecoder::new(SampleRate::Hz48000, 2, 1, 2, &[0, 1]).is_err());
        // 0 streams
        assert!(OpusMSDecoder::new(SampleRate::Hz48000, 1, 0, 0, &[0]).is_err());
    }

    #[test]
    fn test_channel_layout_validate() {
        let layout = ChannelLayout {
            nb_channels: 2,
            nb_streams: 1,
            nb_coupled_streams: 1,
            mapping: {
                let mut m = [0u8; 256];
                m[0] = 0;
                m[1] = 1;
                m
            },
        };
        assert!(layout.validate());

        // Invalid: mapping[1] = 5 >= max_channel (1+1=2)
        let mut bad = layout.clone();
        bad.mapping[1] = 5;
        assert!(!bad.validate());

        // Valid: mapping[1] = 255 (muted channel)
        let mut muted = layout.clone();
        muted.mapping[1] = 255;
        assert!(muted.validate());
    }

    #[test]
    fn test_get_channels() {
        let layout = ChannelLayout {
            nb_channels: 6,
            nb_streams: 4,
            nb_coupled_streams: 2,
            mapping: {
                let mut m = [0u8; 256];
                // stream 0 coupled: left=0, right=1 -> channels mapped to 0,1
                m[0] = 0; // ch0 -> stream 0 left
                m[1] = 1; // ch1 -> stream 0 right
                // stream 1 coupled: left=2, right=3
                m[2] = 2; // ch2 -> stream 1 left
                m[3] = 3; // ch3 -> stream 1 right
                // stream 2 mono: mapping = 2+2=4
                m[4] = 4; // ch4 -> stream 2 mono
                // stream 3 mono: mapping = 3+2=5
                m[5] = 5; // ch5 -> stream 3 mono
                m
            },
        };

        assert_eq!(get_left_channel(&layout, 0, -1), 0);
        assert_eq!(get_right_channel(&layout, 0, -1), 1);
        assert_eq!(get_left_channel(&layout, 1, -1), 2);
        assert_eq!(get_right_channel(&layout, 1, -1), 3);
        assert_eq!(get_mono_channel(&layout, 2, -1), 4);
        assert_eq!(get_mono_channel(&layout, 3, -1), 5);
    }
}
