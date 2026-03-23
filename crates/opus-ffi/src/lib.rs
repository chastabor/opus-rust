//! FFI bindings to C libopus for cross-validation and benchmarking.
//!
//! Provides safe Rust wrappers (`COpusEncoder`, `COpusDecoder`) around the
//! C reference Opus encoder/decoder, allowing side-by-side comparison with
//! the pure-Rust implementation.

use std::ptr;

// ── Raw FFI declarations ──

// Opus constants (matching opus_defines.h)
pub const OPUS_APPLICATION_VOIP: i32 = 2048;
pub const OPUS_APPLICATION_AUDIO: i32 = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: i32 = 2051;

pub const OPUS_AUTO: i32 = -1000;
pub const OPUS_BITRATE_MAX: i32 = -1;

pub const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
pub const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
pub const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
pub const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
pub const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

pub const OPUS_SIGNAL_VOICE: i32 = 3001;
pub const OPUS_SIGNAL_MUSIC: i32 = 3002;

// Opaque C types
#[repr(C)]
pub struct OpusEncoderC {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OpusDecoderC {
    _private: [u8; 0],
}

unsafe extern "C" {
    // Core encoder API
    fn opus_encoder_create(
        fs: i32,
        channels: i32,
        application: i32,
        error: *mut i32,
    ) -> *mut OpusEncoderC;
    fn opus_encoder_destroy(enc: *mut OpusEncoderC);
    fn opus_encode_float(
        enc: *mut OpusEncoderC,
        pcm: *const f32,
        frame_size: i32,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;

    // Core decoder API
    fn opus_decoder_create(fs: i32, channels: i32, error: *mut i32) -> *mut OpusDecoderC;
    fn opus_decoder_destroy(dec: *mut OpusDecoderC);
    fn opus_decode_float(
        dec: *mut OpusDecoderC,
        data: *const u8,
        len: i32,
        pcm: *mut f32,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;

    // Non-variadic CTL wrappers (from wrapper.c)
    fn opus_enc_set_bitrate(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_complexity(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_max_bandwidth(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_bandwidth(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_vbr(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_signal(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_force_channels(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_inband_fec(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_packet_loss_perc(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_get_final_range(enc: *mut OpusEncoderC, val: *mut u32) -> i32;
    fn opus_enc_reset(enc: *mut OpusEncoderC) -> i32;
    fn opus_dec_get_final_range(dec: *mut OpusDecoderC, val: *mut u32) -> i32;
    fn opus_dec_reset(dec: *mut OpusDecoderC) -> i32;
}

// ── Error handling ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct COpusError(pub i32);

impl std::fmt::Display for COpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "C opus error code {}", self.0)
    }
}

impl std::error::Error for COpusError {}

fn check(code: i32) -> Result<(), COpusError> {
    if code < 0 {
        Err(COpusError(code))
    } else {
        Ok(())
    }
}

// ── Safe Encoder Wrapper ──

pub struct COpusEncoder {
    raw: *mut OpusEncoderC,
}

// SAFETY: The C encoder is single-threaded but we never share it across threads.
unsafe impl Send for COpusEncoder {}

impl COpusEncoder {
    pub fn new(fs: i32, channels: i32, application: i32) -> Result<Self, COpusError> {
        let mut error = 0i32;
        let raw = unsafe { opus_encoder_create(fs, channels, application, &mut error) };
        if raw.is_null() || error < 0 {
            return Err(COpusError(error));
        }
        Ok(Self { raw })
    }

    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        output: &mut [u8],
    ) -> Result<i32, COpusError> {
        let ret = unsafe {
            opus_encode_float(
                self.raw,
                pcm.as_ptr(),
                frame_size,
                output.as_mut_ptr(),
                output.len() as i32,
            )
        };
        if ret < 0 {
            Err(COpusError(ret))
        } else {
            Ok(ret)
        }
    }

    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_bitrate(self.raw, bitrate) })
    }

    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_complexity(self.raw, complexity) })
    }

    pub fn set_max_bandwidth(&mut self, bw: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_max_bandwidth(self.raw, bw) })
    }

    pub fn set_bandwidth(&mut self, bw: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_bandwidth(self.raw, bw) })
    }

    pub fn set_vbr(&mut self, enabled: bool) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_vbr(self.raw, enabled as i32) })
    }

    pub fn set_signal(&mut self, signal: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_signal(self.raw, signal) })
    }

    pub fn set_force_channels(&mut self, channels: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_force_channels(self.raw, channels) })
    }

    pub fn set_inband_fec(&mut self, enabled: bool) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_inband_fec(self.raw, enabled as i32) })
    }

    pub fn set_packet_loss_perc(&mut self, perc: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_packet_loss_perc(self.raw, perc) })
    }

    pub fn final_range(&mut self) -> u32 {
        let mut val = 0u32;
        unsafe { opus_enc_get_final_range(self.raw, &mut val) };
        val
    }

    pub fn reset(&mut self) -> Result<(), COpusError> {
        check(unsafe { opus_enc_reset(self.raw) })
    }
}

impl Drop for COpusEncoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { opus_encoder_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

// ── Safe Decoder Wrapper ──

pub struct COpusDecoder {
    raw: *mut OpusDecoderC,
}

unsafe impl Send for COpusDecoder {}

impl COpusDecoder {
    pub fn new(fs: i32, channels: i32) -> Result<Self, COpusError> {
        let mut error = 0i32;
        let raw = unsafe { opus_decoder_create(fs, channels, &mut error) };
        if raw.is_null() || error < 0 {
            return Err(COpusError(error));
        }
        Ok(Self { raw })
    }

    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, COpusError> {
        let (data_ptr, data_len) = match data {
            Some(d) => (d.as_ptr(), d.len() as i32),
            None => (ptr::null(), 0),
        };
        let ret = unsafe {
            opus_decode_float(
                self.raw,
                data_ptr,
                data_len,
                pcm.as_mut_ptr(),
                frame_size,
                decode_fec as i32,
            )
        };
        if ret < 0 {
            Err(COpusError(ret))
        } else {
            Ok(ret)
        }
    }

    pub fn final_range(&mut self) -> u32 {
        let mut val = 0u32;
        unsafe { opus_dec_get_final_range(self.raw, &mut val) };
        val
    }

    pub fn reset(&mut self) -> Result<(), COpusError> {
        check(unsafe { opus_dec_reset(self.raw) })
    }
}

impl Drop for COpusDecoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { opus_decoder_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_creates_and_encodes() {
        let mut enc = COpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(64000).unwrap();
        let pcm = vec![0.0f32; 960];
        let mut output = vec![0u8; 4000];
        let len = enc.encode_float(&pcm, 960, &mut output).unwrap();
        assert!(len > 0);
    }

    #[test]
    fn decoder_creates_and_decodes() {
        // Encode a frame, then decode it.
        let mut enc = COpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(64000).unwrap();
        let pcm_in = vec![0.0f32; 960];
        let mut packet = vec![0u8; 4000];
        let pkt_len = enc.encode_float(&pcm_in, 960, &mut packet).unwrap();

        let mut dec = COpusDecoder::new(48000, 1).unwrap();
        let mut pcm_out = vec![0.0f32; 960];
        let samples = dec
            .decode_float(Some(&packet[..pkt_len as usize]), &mut pcm_out, 960, false)
            .unwrap();
        assert_eq!(samples, 960);
    }
}
