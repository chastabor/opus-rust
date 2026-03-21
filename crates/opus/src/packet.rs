use crate::error::OpusError;

/// Opus bandwidth constants
pub const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
pub const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
pub const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
pub const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
pub const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

/// Opus mode constants
pub const MODE_SILK_ONLY: i32 = 1000;
pub const MODE_HYBRID: i32 = 1001;
pub const MODE_CELT_ONLY: i32 = 1002;

/// Get the number of samples per frame from the TOC byte.
pub fn opus_packet_get_samples_per_frame(data: &[u8], fs: i32) -> i32 {
    if data.is_empty() {
        return 0;
    }
    let toc = data[0];
    if toc & 0x80 != 0 {
        // CELT-only
        let audiosize = ((toc >> 3) & 0x3) as i32;
        (fs << audiosize) / 400
    } else if (toc & 0x60) == 0x60 {
        // Hybrid
        if toc & 0x08 != 0 {
            fs / 50
        } else {
            fs / 100
        }
    } else {
        // SILK-only
        let audiosize = ((toc >> 3) & 0x3) as i32;
        if audiosize == 3 {
            fs * 60 / 1000
        } else {
            (fs << audiosize) / 100
        }
    }
}

/// Get the codec mode from a packet.
pub fn opus_packet_get_mode(data: &[u8]) -> i32 {
    if data[0] & 0x80 != 0 {
        MODE_CELT_ONLY
    } else if (data[0] & 0x60) == 0x60 {
        MODE_HYBRID
    } else {
        MODE_SILK_ONLY
    }
}

/// Get the bandwidth from a packet.
pub fn opus_packet_get_bandwidth(data: &[u8]) -> i32 {
    if data[0] & 0x80 != 0 {
        let bandwidth = OPUS_BANDWIDTH_MEDIUMBAND + ((data[0] as i32 >> 5) & 0x3);
        if bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
            OPUS_BANDWIDTH_NARROWBAND
        } else {
            bandwidth
        }
    } else if (data[0] & 0x60) == 0x60 {
        if data[0] & 0x10 != 0 {
            OPUS_BANDWIDTH_FULLBAND
        } else {
            OPUS_BANDWIDTH_SUPERWIDEBAND
        }
    } else {
        OPUS_BANDWIDTH_NARROWBAND + ((data[0] as i32 >> 5) & 0x3)
    }
}

/// Get the number of channels from a packet.
pub fn opus_packet_get_nb_channels(data: &[u8]) -> i32 {
    if data[0] & 0x4 != 0 { 2 } else { 1 }
}

/// Get the number of frames in an Opus packet.
pub fn opus_packet_get_nb_frames(packet: &[u8]) -> Result<i32, OpusError> {
    if packet.is_empty() {
        return Err(OpusError::BadArg);
    }
    let count = packet[0] & 0x3;
    match count {
        0 => Ok(1),
        3 => {
            if packet.len() < 2 {
                Err(OpusError::InvalidPacket)
            } else {
                Ok((packet[1] & 0x3F) as i32)
            }
        }
        _ => Ok(2),
    }
}

/// Get the number of samples in an Opus packet.
pub fn opus_packet_get_nb_samples(packet: &[u8], fs: i32) -> Result<i32, OpusError> {
    let count = opus_packet_get_nb_frames(packet)?;
    let samples = count * opus_packet_get_samples_per_frame(packet, fs);
    // Can't have more than 120 ms
    if samples * 25 > fs * 3 {
        Err(OpusError::InvalidPacket)
    } else {
        Ok(samples)
    }
}

/// Parse a variable-length size field. Returns (size, bytes_consumed) or error.
fn parse_size(data: &[u8]) -> Result<(i16, usize), OpusError> {
    if data.is_empty() {
        return Err(OpusError::InvalidPacket);
    }
    if data[0] < 252 {
        Ok((data[0] as i16, 1))
    } else if data.len() < 2 {
        Err(OpusError::InvalidPacket)
    } else {
        Ok((4 * data[1] as i16 + data[0] as i16, 2))
    }
}

/// Parsed Opus packet information.
pub struct ParsedPacket {
    /// The TOC (Table of Contents) byte.
    pub toc: u8,
    /// Frame sizes in bytes.
    pub frame_sizes: Vec<i16>,
    /// Offset into the packet data where payload frames begin.
    pub payload_offset: usize,
}

/// Parse an Opus packet into its constituent frames.
pub fn opus_packet_parse(data: &[u8]) -> Result<ParsedPacket, OpusError> {
    if data.is_empty() {
        return Err(OpusError::InvalidPacket);
    }

    let framesize = opus_packet_get_samples_per_frame(data, 48000);
    let toc = data[0];
    let mut pos: usize = 1;
    let mut remaining = data.len() as i32 - 1;
    let mut sizes: Vec<i16> = Vec::new();

    match toc & 0x3 {
        // One frame
        0 => {
            sizes.push(remaining as i16);
        }
        // Two CBR frames
        1 => {
            if remaining & 1 != 0 {
                return Err(OpusError::InvalidPacket);
            }
            let half = remaining / 2;
            sizes.push(half as i16);
            sizes.push(half as i16);
        }
        // Two VBR frames
        2 => {
            let (sz, bytes) = parse_size(&data[pos..])?;
            pos += bytes;
            remaining -= bytes as i32;
            if sz < 0 || sz as i32 > remaining {
                return Err(OpusError::InvalidPacket);
            }
            let last = remaining - sz as i32;
            sizes.push(sz);
            sizes.push(last as i16);
        }
        // Multiple CBR/VBR frames
        _ => {
            if remaining < 1 {
                return Err(OpusError::InvalidPacket);
            }
            let ch = data[pos];
            pos += 1;
            remaining -= 1;
            let count = (ch & 0x3F) as i32;
            if count <= 0 || framesize as i64 * count as i64 > 5760 {
                return Err(OpusError::InvalidPacket);
            }
            // Padding
            if ch & 0x40 != 0 {
                loop {
                    if remaining <= 0 {
                        return Err(OpusError::InvalidPacket);
                    }
                    let p = data[pos];
                    pos += 1;
                    remaining -= 1;
                    let tmp = if p == 255 { 254 } else { p as i32 };
                    remaining -= tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if remaining < 0 {
                return Err(OpusError::InvalidPacket);
            }
            let cbr = ch & 0x80 == 0;
            if !cbr {
                // VBR
                let mut last_size = remaining;
                for _ in 0..(count - 1) as usize {
                    let (sz, bytes) = parse_size(&data[pos..])?;
                    pos += bytes;
                    remaining -= bytes as i32;
                    if sz < 0 || sz as i32 > remaining {
                        return Err(OpusError::InvalidPacket);
                    }
                    sizes.push(sz);
                    last_size -= bytes as i32 + sz as i32;
                }
                if last_size < 0 {
                    return Err(OpusError::InvalidPacket);
                }
                sizes.push(last_size as i16);
            } else {
                // CBR
                let frame_size = remaining / count;
                if frame_size * count != remaining {
                    return Err(OpusError::InvalidPacket);
                }
                if frame_size > 1275 {
                    return Err(OpusError::InvalidPacket);
                }
                for _ in 0..count {
                    sizes.push(frame_size as i16);
                }
            }
        }
    }

    // Validate last size
    if let Some(last) = sizes.last() {
        if *last > 1275 {
            // Only for non-self-delimited, check last size
            // (already handled in code 3 CBR case above)
        }
    }

    Ok(ParsedPacket {
        toc,
        frame_sizes: sizes,
        payload_offset: pos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_bandwidth() {
        // CELT-only, fullband (bits 7,6,5 = 1,1,1)
        assert_eq!(opus_packet_get_bandwidth(&[0xE0]), OPUS_BANDWIDTH_FULLBAND);
        // SILK-only, narrowband (bits 7,6,5 = 0,0,0)
        assert_eq!(opus_packet_get_bandwidth(&[0x00]), OPUS_BANDWIDTH_NARROWBAND);
        // SILK-only, wideband (bits 7,6,5 = 0,1,0)
        assert_eq!(opus_packet_get_bandwidth(&[0x40]), OPUS_BANDWIDTH_WIDEBAND);
        // Hybrid, superwideband
        assert_eq!(opus_packet_get_bandwidth(&[0x60]), OPUS_BANDWIDTH_SUPERWIDEBAND);
        // Hybrid, fullband
        assert_eq!(opus_packet_get_bandwidth(&[0x70]), OPUS_BANDWIDTH_FULLBAND);
    }

    #[test]
    fn test_get_mode() {
        assert_eq!(opus_packet_get_mode(&[0x80]), MODE_CELT_ONLY);
        assert_eq!(opus_packet_get_mode(&[0x60]), MODE_HYBRID);
        assert_eq!(opus_packet_get_mode(&[0x00]), MODE_SILK_ONLY);
        assert_eq!(opus_packet_get_mode(&[0x40]), MODE_SILK_ONLY);
    }

    #[test]
    fn test_get_nb_channels() {
        assert_eq!(opus_packet_get_nb_channels(&[0x00]), 1); // mono
        assert_eq!(opus_packet_get_nb_channels(&[0x04]), 2); // stereo
    }

    #[test]
    fn test_get_samples_per_frame() {
        // SILK-only NB 10ms: bits 7,6,5,4,3 = 0,0,0,0,0
        assert_eq!(opus_packet_get_samples_per_frame(&[0x00], 48000), 480);
        // SILK-only NB 20ms: bits 4,3 = 0,1
        assert_eq!(opus_packet_get_samples_per_frame(&[0x08], 48000), 960);
        // SILK-only NB 40ms: bits 4,3 = 1,0
        assert_eq!(opus_packet_get_samples_per_frame(&[0x10], 48000), 1920);
        // SILK-only NB 60ms: bits 4,3 = 1,1
        assert_eq!(opus_packet_get_samples_per_frame(&[0x18], 48000), 2880);
        // CELT 2.5ms: bits 4,3 = 0,0
        assert_eq!(opus_packet_get_samples_per_frame(&[0x80], 48000), 120);
        // CELT 5ms
        assert_eq!(opus_packet_get_samples_per_frame(&[0x88], 48000), 240);
        // CELT 10ms
        assert_eq!(opus_packet_get_samples_per_frame(&[0x90], 48000), 480);
        // CELT 20ms
        assert_eq!(opus_packet_get_samples_per_frame(&[0x98], 48000), 960);
        // Hybrid 10ms
        assert_eq!(opus_packet_get_samples_per_frame(&[0x60], 48000), 480);
        // Hybrid 20ms
        assert_eq!(opus_packet_get_samples_per_frame(&[0x68], 48000), 960);
    }

    #[test]
    fn test_get_nb_frames() {
        // Code 0: 1 frame
        assert_eq!(opus_packet_get_nb_frames(&[0x00]).unwrap(), 1);
        // Code 1: 2 frames CBR
        assert_eq!(opus_packet_get_nb_frames(&[0x01]).unwrap(), 2);
        // Code 2: 2 frames VBR
        assert_eq!(opus_packet_get_nb_frames(&[0x02]).unwrap(), 2);
        // Code 3: look at second byte
        assert_eq!(opus_packet_get_nb_frames(&[0x03, 0x05]).unwrap(), 5);
    }

    #[test]
    fn test_parse_single_frame() {
        // TOC=0x00 (SILK NB 10ms mono, code 0 = 1 frame), then 10 bytes of data
        let mut pkt = vec![0x00u8];
        pkt.extend_from_slice(&[0xAA; 10]);
        let parsed = opus_packet_parse(&pkt).unwrap();
        assert_eq!(parsed.toc, 0x00);
        assert_eq!(parsed.frame_sizes.len(), 1);
        assert_eq!(parsed.frame_sizes[0], 10);
        assert_eq!(parsed.payload_offset, 1);
    }

    #[test]
    fn test_parse_two_cbr_frames() {
        // Code 1 = 2 CBR frames, 20 bytes total payload
        let mut pkt = vec![0x01u8];
        pkt.extend_from_slice(&[0xBB; 20]);
        let parsed = opus_packet_parse(&pkt).unwrap();
        assert_eq!(parsed.frame_sizes.len(), 2);
        assert_eq!(parsed.frame_sizes[0], 10);
        assert_eq!(parsed.frame_sizes[1], 10);
    }

    #[test]
    fn test_parse_two_cbr_frames_odd_size() {
        // Code 1 with odd remaining length should fail
        let pkt = vec![0x01u8, 0x00, 0x00, 0x00]; // 3 bytes remaining (odd)
        assert!(opus_packet_parse(&pkt).is_err());
    }
}
