use crate::error::OpusError;
use crate::packet::*;

/// Maximum number of frames that can be stored in a repacketizer.
const MAX_FRAMES: usize = 48;

/// Opus repacketizer: combines or splits Opus packets.
pub struct OpusRepacketizer {
    toc: u8,
    nb_frames: usize,
    frame_data: Vec<Vec<u8>>,
}

impl OpusRepacketizer {
    /// Create a new repacketizer.
    pub fn new() -> Self {
        OpusRepacketizer {
            toc: 0,
            nb_frames: 0,
            frame_data: Vec::new(),
        }
    }

    /// Reset the repacketizer state.
    pub fn reset(&mut self) {
        self.nb_frames = 0;
        self.frame_data.clear();
    }

    /// Get the number of frames currently stored.
    pub fn get_nb_frames(&self) -> usize {
        self.nb_frames
    }

    /// Add a packet to the repacketizer.
    pub fn cat(&mut self, data: &[u8]) -> Result<(), OpusError> {
        let parsed = opus_packet_parse(data)?;

        // Check TOC compatibility (same config, same channels)
        if self.nb_frames > 0 && (parsed.toc & 0xFC) != (self.toc & 0xFC) {
            return Err(OpusError::InvalidPacket);
        }

        let new_count = self.nb_frames + parsed.frame_sizes.len();
        if new_count > MAX_FRAMES {
            return Err(OpusError::InvalidPacket);
        }

        // Check 120ms limit
        let samples_per_frame = opus_packet_get_samples_per_frame(data, 48000);
        if samples_per_frame as i64 * new_count as i64 > 5760 {
            return Err(OpusError::InvalidPacket);
        }

        if self.nb_frames == 0 {
            self.toc = parsed.toc;
        }

        let mut offset = parsed.payload_offset;
        for &sz in &parsed.frame_sizes {
            let frame = data[offset..offset + sz as usize].to_vec();
            self.frame_data.push(frame);
            offset += sz as usize;
        }
        self.nb_frames = new_count;

        Ok(())
    }

    /// Output a repacketized packet containing frames [begin, end).
    pub fn out_range(&self, begin: usize, end: usize, out: &mut Vec<u8>) -> Result<usize, OpusError> {
        if begin >= end || end > self.nb_frames {
            return Err(OpusError::BadArg);
        }

        let count = end - begin;
        out.clear();

        if count == 1 {
            // Code 0: single frame
            out.push(self.toc & 0xFC); // code 0
            out.extend_from_slice(&self.frame_data[begin]);
        } else if count == 2 {
            let s0 = self.frame_data[begin].len();
            let s1 = self.frame_data[begin + 1].len();
            if s0 == s1 {
                // Code 1: two CBR frames
                out.push((self.toc & 0xFC) | 0x01);
                out.extend_from_slice(&self.frame_data[begin]);
                out.extend_from_slice(&self.frame_data[begin + 1]);
            } else {
                // Code 2: two VBR frames
                out.push((self.toc & 0xFC) | 0x02);
                encode_size(s0, out);
                out.extend_from_slice(&self.frame_data[begin]);
                out.extend_from_slice(&self.frame_data[begin + 1]);
            }
        } else {
            // Code 3: multiple frames
            // Check if CBR
            let first_size = self.frame_data[begin].len();
            let cbr = (begin..end).all(|i| self.frame_data[i].len() == first_size);

            out.push((self.toc & 0xFC) | 0x03);
            let mut ch: u8 = count as u8;
            if !cbr {
                ch |= 0x80; // VBR flag
            }
            out.push(ch);

            if cbr {
                for i in begin..end {
                    out.extend_from_slice(&self.frame_data[i]);
                }
            } else {
                // VBR: encode sizes for all but last frame
                for i in begin..end - 1 {
                    encode_size(self.frame_data[i].len(), out);
                }
                for i in begin..end {
                    out.extend_from_slice(&self.frame_data[i]);
                }
            }
        }

        Ok(out.len())
    }

    /// Output all frames as a single repacketized packet.
    pub fn out(&self, out: &mut Vec<u8>) -> Result<usize, OpusError> {
        self.out_range(0, self.nb_frames, out)
    }
}

impl Default for OpusRepacketizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode a frame size in variable-length format.
fn encode_size(size: usize, out: &mut Vec<u8>) {
    if size < 252 {
        out.push(size as u8);
    } else {
        out.push(252 + (size & 0x3) as u8);
        out.push(((size - (252 + (size & 0x3))) >> 2) as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repacketizer_single_frame() {
        let mut repacker = OpusRepacketizer::new();
        // Code 0, single frame: TOC + 5 bytes data
        let pkt = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
        repacker.cat(&pkt).unwrap();
        assert_eq!(repacker.get_nb_frames(), 1);

        let mut out = Vec::new();
        repacker.out(&mut out).unwrap();
        assert!(!out.is_empty());
    }

    #[test]
    fn test_repacketizer_reset() {
        let mut repacker = OpusRepacketizer::new();
        let pkt = vec![0x00, 0x01, 0x02, 0x03];
        repacker.cat(&pkt).unwrap();
        assert_eq!(repacker.get_nb_frames(), 1);
        repacker.reset();
        assert_eq!(repacker.get_nb_frames(), 0);
    }
}
