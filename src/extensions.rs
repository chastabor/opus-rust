//! Opus packet extension parsing and generation.
//!
//! Port of C extensions.c. Extensions are embedded in the padding area of
//! Opus packets and carry auxiliary data such as gain adjustments,
//! DRED (deep redundancy), and other metadata.
//!
//! Extension format overview:
//! - Each extension starts with an ID/L byte: bits 1-7 = extension ID, bit 0 = L flag
//! - ID 0: padding byte (L=0) or next-frame separator (L=1)
//! - ID 1: increment current frame index (L=0: +1, L=1: read next byte as increment)
//! - IDs 2: reserved (repeat last extension, not implemented here)
//! - IDs 3-31: short extensions (L=0: no data, L=1: 1 byte of data)
//! - IDs 32-127: long extensions
//!   - L=0: no additional data
//!   - L=1: followed by a varint length, then that many data bytes

use crate::error::OpusError;

/// Maximum number of extensions that can be parsed from a single packet.
pub const OPUS_MAX_EXTENSIONS: usize = 128;

/// Minimum extension ID for "long" extensions.
const LONG_EXT_THRESHOLD: i32 = 32;

/// Extension data entry.
#[derive(Clone, Debug)]
pub struct OpusExtensionData {
    /// Extension ID (1-127).
    pub id: i32,
    /// Frame index this extension applies to.
    pub frame: i32,
    /// Extension payload data.
    pub data: Vec<u8>,
}

/// Parse extensions from Opus packet padding data.
///
/// Reads through `data` and extracts all extension entries. Returns the
/// list of parsed extensions. The `data` slice should contain only the
/// extension/padding area of the packet, not the Opus packet header.
///
/// # Errors
///
/// Returns `OpusError::InvalidPacket` if the data is malformed.
/// Returns `OpusError::BadArg` if inputs are invalid.
pub fn opus_packet_extensions_parse(
    data: &[u8],
    nb_extensions: &mut i32,
    extensions: &mut Vec<OpusExtensionData>,
) -> Result<(), OpusError> {
    extensions.clear();
    *nb_extensions = 0;

    if data.is_empty() {
        return Ok(());
    }

    let mut pos = 0usize;
    let len = data.len();
    let mut current_frame: i32 = 0;

    while pos < len {
        if extensions.len() >= OPUS_MAX_EXTENSIONS {
            break;
        }

        let byte = data[pos];
        pos += 1;

        let id = (byte >> 1) as i32;
        let l_flag = byte & 1;

        match id {
            0 => {
                if l_flag == 0 {
                    // Padding byte -- skip (just a no-op filler)
                    continue;
                } else {
                    // Frame separator: advance to next frame
                    current_frame += 1;
                    continue;
                }
            }
            1 => {
                // Frame increment
                if l_flag == 0 {
                    current_frame += 1;
                } else {
                    // Read next byte as increment value
                    if pos >= len {
                        return Err(OpusError::InvalidPacket);
                    }
                    let inc = data[pos] as i32;
                    pos += 1;
                    current_frame += inc;
                }
                continue;
            }
            2 => {
                // Reserved (repeat extension) -- skip for now
                // In a full implementation this would repeat the last extension
                // with optional modifications. We skip any associated data.
                if l_flag == 1 {
                    // Has a repeat count byte
                    if pos >= len {
                        return Err(OpusError::InvalidPacket);
                    }
                    pos += 1; // skip the count byte
                }
                continue;
            }
            3..=31 => {
                // Short extension
                let ext_data = if l_flag == 1 {
                    // 1 byte of data
                    if pos >= len {
                        return Err(OpusError::InvalidPacket);
                    }
                    let d = vec![data[pos]];
                    pos += 1;
                    d
                } else {
                    Vec::new()
                };

                extensions.push(OpusExtensionData {
                    id,
                    frame: current_frame,
                    data: ext_data,
                });
            }
            _ => {
                // Long extension (IDs 32-127)
                if l_flag == 0 {
                    // No data
                    extensions.push(OpusExtensionData {
                        id,
                        frame: current_frame,
                        data: Vec::new(),
                    });
                } else {
                    // Read varint length
                    let (ext_len, bytes_consumed) = read_varint(&data[pos..])?;
                    pos += bytes_consumed;

                    if ext_len as usize > len - pos {
                        return Err(OpusError::InvalidPacket);
                    }

                    let ext_data = data[pos..pos + ext_len as usize].to_vec();
                    pos += ext_len as usize;

                    extensions.push(OpusExtensionData {
                        id,
                        frame: current_frame,
                        data: ext_data,
                    });
                }
            }
        }
    }

    *nb_extensions = extensions.len() as i32;
    Ok(())
}

/// Generate extension data for embedding in Opus packet padding.
///
/// Takes a list of extensions and encodes them into the provided buffer.
/// Extensions should be sorted by frame index for proper encoding.
///
/// - `data`: Output buffer for the extension payload
/// - `extensions`: List of extensions to encode
/// - `nb_frames`: Total number of frames in the packet (for validation)
///
/// Returns the number of bytes written to `data`.
///
/// # Errors
///
/// Returns `OpusError::BufferTooSmall` if the buffer is too small.
/// Returns `OpusError::BadArg` if an extension has an invalid ID.
pub fn opus_packet_extensions_generate(
    data: &mut [u8],
    extensions: &[OpusExtensionData],
    _nb_frames: i32,
) -> Result<i32, OpusError> {
    if extensions.is_empty() {
        return Ok(0);
    }

    let mut pos = 0usize;
    let max_len = data.len();
    let mut current_frame: i32 = 0;

    for ext in extensions {
        if ext.id < 1 || ext.id > 127 {
            return Err(OpusError::BadArg);
        }

        // Emit frame advancement if needed
        while current_frame < ext.frame {
            let delta = ext.frame - current_frame;
            if delta == 1 {
                // Use frame separator (ID=0, L=1)
                if pos >= max_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[pos] = 0x01; // ID=0, L=1
                pos += 1;
                current_frame += 1;
            } else if delta <= 255 {
                // Use frame increment with value (ID=1, L=1, value)
                if pos + 2 > max_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[pos] = 0x03; // ID=1, L=1
                data[pos + 1] = delta as u8;
                pos += 2;
                current_frame += delta;
            } else {
                // Large delta: emit multiple increments
                if pos + 2 > max_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[pos] = 0x03; // ID=1, L=1
                data[pos + 1] = 255;
                pos += 2;
                current_frame += 255;
            }
        }

        // Encode the extension
        let id = ext.id;
        let has_data = !ext.data.is_empty();

        if id >= LONG_EXT_THRESHOLD {
            // Long extension
            if has_data {
                let l_flag = 1u8;
                if pos >= max_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[pos] = ((id as u8) << 1) | l_flag;
                pos += 1;

                // Write varint length
                let len_bytes = write_varint(ext.data.len() as u32, &mut data[pos..])?;
                pos += len_bytes;

                // Write data
                if pos + ext.data.len() > max_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[pos..pos + ext.data.len()].copy_from_slice(&ext.data);
                pos += ext.data.len();
            } else {
                if pos >= max_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[pos] = (id as u8) << 1; // L=0
                pos += 1;
            }
        } else {
            // Short extension (IDs 3-31)
            if has_data {
                // L=1, followed by 1 byte of data
                if pos + 2 > max_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[pos] = ((id as u8) << 1) | 1;
                pos += 1;
                // Short extensions with L=1 carry exactly 1 byte
                data[pos] = ext.data[0];
                pos += 1;
            } else {
                // L=0, no data
                if pos >= max_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[pos] = (id as u8) << 1;
                pos += 1;
            }
        }
    }

    Ok(pos as i32)
}

/// Read a variable-length integer (varint) from a byte slice.
///
/// Uses LEB128-like encoding: each byte contributes 7 bits, with the
/// high bit indicating continuation (1 = more bytes follow, 0 = last byte).
///
/// Returns (value, bytes_consumed).
fn read_varint(data: &[u8]) -> Result<(u32, usize), OpusError> {
    if data.is_empty() {
        return Err(OpusError::InvalidPacket);
    }

    let mut value: u32 = 0;
    let mut shift: u32 = 0;
    let mut pos = 0usize;

    loop {
        if pos >= data.len() {
            return Err(OpusError::InvalidPacket);
        }
        if shift >= 28 && (data[pos] & 0x80 != 0) {
            // Overflow protection
            return Err(OpusError::InvalidPacket);
        }

        let byte = data[pos];
        value |= ((byte & 0x7F) as u32) << shift;
        pos += 1;
        shift += 7;

        if byte & 0x80 == 0 {
            break;
        }
    }

    Ok((value, pos))
}

/// Write a variable-length integer (varint) to a byte slice.
///
/// Returns the number of bytes written.
fn write_varint(mut value: u32, data: &mut [u8]) -> Result<usize, OpusError> {
    let mut pos = 0usize;

    loop {
        if pos >= data.len() {
            return Err(OpusError::BufferTooSmall);
        }

        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80; // continuation bit
        }

        data[pos] = byte;
        pos += 1;

        if value == 0 {
            break;
        }
    }

    Ok(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extensions_roundtrip() {
        // Generate extensions, then parse them back
        let extensions_in = vec![
            OpusExtensionData {
                id: 5,
                frame: 0,
                data: vec![0x42],
            },
            OpusExtensionData {
                id: 10,
                frame: 0,
                data: Vec::new(),
            },
            OpusExtensionData {
                id: 50,
                frame: 1,
                data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            },
        ];

        let mut buf = vec![0u8; 256];
        let nbytes = opus_packet_extensions_generate(&mut buf, &extensions_in, 4).unwrap();
        assert!(nbytes > 0, "Should produce extension bytes");

        let mut nb_ext = 0i32;
        let mut extensions_out = Vec::new();
        opus_packet_extensions_parse(&buf[..nbytes as usize], &mut nb_ext, &mut extensions_out)
            .unwrap();

        assert_eq!(nb_ext, 3, "Should parse 3 extensions, got {}", nb_ext);

        // Verify first extension
        assert_eq!(extensions_out[0].id, 5);
        assert_eq!(extensions_out[0].frame, 0);
        assert_eq!(extensions_out[0].data, vec![0x42]);

        // Verify second extension
        assert_eq!(extensions_out[1].id, 10);
        assert_eq!(extensions_out[1].frame, 0);
        assert!(extensions_out[1].data.is_empty());

        // Verify third extension
        assert_eq!(extensions_out[2].id, 50);
        assert_eq!(extensions_out[2].frame, 1);
        assert_eq!(extensions_out[2].data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_extensions_empty() {
        let mut nb_ext = 0i32;
        let mut extensions = Vec::new();
        opus_packet_extensions_parse(&[], &mut nb_ext, &mut extensions).unwrap();
        assert_eq!(nb_ext, 0);
        assert!(extensions.is_empty());
    }

    #[test]
    fn test_extensions_generate_empty() {
        let mut buf = vec![0u8; 256];
        let nbytes = opus_packet_extensions_generate(&mut buf, &[], 1).unwrap();
        assert_eq!(nbytes, 0);
    }

    #[test]
    fn test_extensions_padding_only() {
        // A buffer of all zeros is all padding bytes (ID=0, L=0)
        let data = vec![0u8; 10];
        let mut nb_ext = 0i32;
        let mut extensions = Vec::new();
        opus_packet_extensions_parse(&data, &mut nb_ext, &mut extensions).unwrap();
        assert_eq!(nb_ext, 0, "Padding-only data should produce no extensions");
    }

    #[test]
    fn test_extensions_frame_separator() {
        // Build: ext(id=5, frame=0), frame-sep, ext(id=5, frame=1)
        let extensions_in = vec![
            OpusExtensionData {
                id: 5,
                frame: 0,
                data: Vec::new(),
            },
            OpusExtensionData {
                id: 5,
                frame: 1,
                data: Vec::new(),
            },
            OpusExtensionData {
                id: 5,
                frame: 2,
                data: Vec::new(),
            },
        ];

        let mut buf = vec![0u8; 256];
        let nbytes = opus_packet_extensions_generate(&mut buf, &extensions_in, 4).unwrap();

        let mut nb_ext = 0i32;
        let mut extensions_out = Vec::new();
        opus_packet_extensions_parse(&buf[..nbytes as usize], &mut nb_ext, &mut extensions_out)
            .unwrap();

        assert_eq!(nb_ext, 3);
        assert_eq!(extensions_out[0].frame, 0);
        assert_eq!(extensions_out[1].frame, 1);
        assert_eq!(extensions_out[2].frame, 2);
    }

    #[test]
    fn test_extensions_large_frame_jump() {
        // Extension that jumps to frame 10
        let extensions_in = vec![OpusExtensionData {
            id: 5,
            frame: 10,
            data: Vec::new(),
        }];

        let mut buf = vec![0u8; 256];
        let nbytes = opus_packet_extensions_generate(&mut buf, &extensions_in, 20).unwrap();

        let mut nb_ext = 0i32;
        let mut extensions_out = Vec::new();
        opus_packet_extensions_parse(&buf[..nbytes as usize], &mut nb_ext, &mut extensions_out)
            .unwrap();

        assert_eq!(nb_ext, 1);
        assert_eq!(extensions_out[0].id, 5);
        assert_eq!(extensions_out[0].frame, 10);
    }

    #[test]
    fn test_extensions_long_with_data() {
        // Long extension with substantial data
        let payload = vec![0x11u8; 300];
        let extensions_in = vec![OpusExtensionData {
            id: 64,
            frame: 0,
            data: payload.clone(),
        }];

        let mut buf = vec![0u8; 512];
        let nbytes = opus_packet_extensions_generate(&mut buf, &extensions_in, 1).unwrap();

        let mut nb_ext = 0i32;
        let mut extensions_out = Vec::new();
        opus_packet_extensions_parse(&buf[..nbytes as usize], &mut nb_ext, &mut extensions_out)
            .unwrap();

        assert_eq!(nb_ext, 1);
        assert_eq!(extensions_out[0].id, 64);
        assert_eq!(extensions_out[0].data.len(), 300);
        assert_eq!(extensions_out[0].data, payload);
    }

    #[test]
    fn test_extensions_invalid_id() {
        // ID 0 is reserved for padding/frame-sep
        let ext = vec![OpusExtensionData {
            id: 0,
            frame: 0,
            data: Vec::new(),
        }];
        let mut buf = vec![0u8; 256];
        let result = opus_packet_extensions_generate(&mut buf, &ext, 1);
        assert!(result.is_err(), "ID 0 should be rejected");

        // ID > 127 is invalid
        let ext = vec![OpusExtensionData {
            id: 128,
            frame: 0,
            data: Vec::new(),
        }];
        let result = opus_packet_extensions_generate(&mut buf, &ext, 1);
        assert!(result.is_err(), "ID 128 should be rejected");
    }

    #[test]
    fn test_extensions_buffer_too_small() {
        let ext = vec![OpusExtensionData {
            id: 64,
            frame: 0,
            data: vec![0; 100],
        }];
        let mut buf = vec![0u8; 5]; // Too small
        let result = opus_packet_extensions_generate(&mut buf, &ext, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_varint_roundtrip() {
        // Test various varint values
        for &value in &[
            0u32, 1, 127, 128, 255, 256, 16383, 16384, 100000, 0x0FFFFFFF,
        ] {
            let mut buf = vec![0u8; 8];
            let written = write_varint(value, &mut buf).unwrap();
            let (parsed, consumed) = read_varint(&buf[..written]).unwrap();
            assert_eq!(
                parsed, value,
                "Varint roundtrip failed for value {}: got {}",
                value, parsed
            );
            assert_eq!(written, consumed);
        }
    }

    #[test]
    fn test_extensions_multiple_short() {
        let extensions_in = vec![
            OpusExtensionData {
                id: 3,
                frame: 0,
                data: Vec::new(),
            },
            OpusExtensionData {
                id: 4,
                frame: 0,
                data: vec![0xFF],
            },
            OpusExtensionData {
                id: 31,
                frame: 0,
                data: vec![0x01],
            },
        ];

        let mut buf = vec![0u8; 256];
        let nbytes = opus_packet_extensions_generate(&mut buf, &extensions_in, 1).unwrap();

        let mut nb_ext = 0i32;
        let mut extensions_out = Vec::new();
        opus_packet_extensions_parse(&buf[..nbytes as usize], &mut nb_ext, &mut extensions_out)
            .unwrap();

        assert_eq!(nb_ext, 3);
        assert_eq!(extensions_out[0].id, 3);
        assert!(extensions_out[0].data.is_empty());
        assert_eq!(extensions_out[1].id, 4);
        assert_eq!(extensions_out[1].data, vec![0xFF]);
        assert_eq!(extensions_out[2].id, 31);
        assert_eq!(extensions_out[2].data, vec![0x01]);
    }
}
