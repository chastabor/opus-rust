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
    // SILK low-level functions (for cross-validation)
    fn silk_A2NLSF(
        nlsf: *mut i16,    // O: NLSFs in Q15 [d]
        a_q16: *mut i32,   // I/O: LPC coefficients in Q16 [d]
        d: i32,            // I: filter order (must be even)
    );

    // silk_burg_modified_c: only available with OPUS_FIXED_POINT=ON
    // Verified identical to Rust with fixed-point build.

    // Float DSP leaf functions (Layer 0)
    fn silk_energy_FLP(data: *const f32, data_length: i32) -> f64;
    // Note: the C float build dispatches inner_product via arch detection.
    // The generic C fallback is silk_inner_product_FLP_c.
    #[link_name = "silk_inner_product_FLP_c"]
    fn silk_inner_product_FLP(data1: *const f32, data2: *const f32, data_length: i32) -> f64;
    fn silk_schur_FLP(refl_coef: *mut f32, auto_corr: *const f32, order: i32) -> f32;
    fn silk_k2a_FLP(a: *mut f32, rc: *const f32, order: i32);
    fn silk_bwexpander_FLP(ar: *mut f32, d: i32, chirp: f32);
    fn silk_apply_sine_window_FLP(px_win: *mut f32, px: *const f32, win_type: i32, length: i32);
    fn silk_scale_copy_vector_FLP(data_out: *mut f32, data_in: *const f32, gain: f32, data_size: i32);
    fn silk_LPC_analysis_filter_FLP(r_lpc: *mut f32, pred_coef: *const f32, s: *const f32, length: i32, order: i32);
    fn silk_LPC_inverse_pred_gain_FLP(a: *const f32, order: i32) -> f32;
    fn silk_autocorrelation_FLP(results: *mut f32, input: *const f32, input_size: i32, corr_count: i32, arch: i32);

    fn silk_NLSF_VQ_weights_laroia(
        pNLSFW_Q_OUT: *mut i16,  // O: NLSF weights [order]
        pNLSF_Q15: *const i16,   // I: NLSFs [order]
        order: i32,              // I: filter order
    );

    // Codebook struct is opaque from Rust side; we access it via pointer
    static silk_NLSF_CB_WB: u8; // address-only — we pass &silk_NLSF_CB_WB as *const c_void

    fn silk_NLSF_encode(
        nlsf_indices: *mut i8,         // O: codebook path [order+1]
        pNLSF_Q15: *mut i16,          // I/O: quantized NLSFs [order]
        psNLSF_CB: *const u8,         // I: codebook struct pointer
        pW_QW: *const i16,            // I: NLSF weights [order]
        NLSF_mu_Q20: i32,             // I: rate weight
        nSurvivors: i32,              // I: max survivors
        signalType: i32,              // I: signal type 0/1/2
    ) -> i32;

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

// ── SILK low-level safe wrappers ──

/// Call the C reference silk_A2NLSF to convert LPC coefficients to NLSFs.
/// Both `nlsf_q15` and `a_q16` must have length >= `order`.
/// Note: `a_q16` is modified in place (bandwidth expansion during root search).
pub fn c_silk_a2nlsf(nlsf_q15: &mut [i16], a_q16: &mut [i32], order: usize) {
    assert!(nlsf_q15.len() >= order && a_q16.len() >= order);
    unsafe {
        silk_A2NLSF(nlsf_q15.as_mut_ptr(), a_q16.as_mut_ptr(), order as i32);
    }
}

// c_silk_burg_modified: only available with OPUS_FIXED_POINT=ON.
// Verified identical to Rust silk_burg_modified.

pub fn c_silk_nlsf_vq_weights_laroia(weights: &mut [i16], nlsf_q15: &[i16], order: usize) {
    assert!(weights.len() >= order && nlsf_q15.len() >= order);
    unsafe {
        silk_NLSF_VQ_weights_laroia(weights.as_mut_ptr(), nlsf_q15.as_ptr(), order as i32);
    }
}

/// Call the C reference NLSF encoder (VQ + trellis) for the WB codebook.
/// Returns RD in Q25. `nlsf_indices` must be [order+1], `nlsf_q15` and `w_q2` must be [order].
pub fn c_silk_nlsf_encode_wb(
    nlsf_indices: &mut [i8],
    nlsf_q15: &mut [i16],
    w_q2: &[i16],
    mu_q20: i32,
    n_survivors: i32,
    signal_type: i32,
) -> i32 {
    unsafe {
        silk_NLSF_encode(
            nlsf_indices.as_mut_ptr(),
            nlsf_q15.as_mut_ptr(),
            &silk_NLSF_CB_WB as *const u8,
            w_q2.as_ptr(),
            mu_q20,
            n_survivors,
            signal_type,
        )
    }
}

// ── Float DSP leaf function wrappers (Layer 0) ──

pub fn c_silk_energy_flp(data: &[f32]) -> f64 {
    unsafe { silk_energy_FLP(data.as_ptr(), data.len() as i32) }
}

pub fn c_silk_inner_product_flp(data1: &[f32], data2: &[f32]) -> f64 {
    let n = data1.len().min(data2.len());
    unsafe { silk_inner_product_FLP(data1.as_ptr(), data2.as_ptr(), n as i32) }
}

pub fn c_silk_schur_flp(refl_coef: &mut [f32], auto_corr: &[f32], order: usize) -> f32 {
    unsafe { silk_schur_FLP(refl_coef.as_mut_ptr(), auto_corr.as_ptr(), order as i32) }
}

pub fn c_silk_k2a_flp(a: &mut [f32], rc: &[f32], order: usize) {
    unsafe { silk_k2a_FLP(a.as_mut_ptr(), rc.as_ptr(), order as i32) }
}

pub fn c_silk_bwexpander_flp(ar: &mut [f32], d: usize, chirp: f32) {
    unsafe { silk_bwexpander_FLP(ar.as_mut_ptr(), d as i32, chirp) }
}

pub fn c_silk_apply_sine_window_flp(px_win: &mut [f32], px: &[f32], win_type: i32, length: usize) {
    unsafe { silk_apply_sine_window_FLP(px_win.as_mut_ptr(), px.as_ptr(), win_type, length as i32) }
}

pub fn c_silk_scale_copy_vector_flp(data_out: &mut [f32], data_in: &[f32], gain: f32, len: usize) {
    unsafe { silk_scale_copy_vector_FLP(data_out.as_mut_ptr(), data_in.as_ptr(), gain, len as i32) }
}

pub fn c_silk_lpc_analysis_filter_flp(r_lpc: &mut [f32], pred_coef: &[f32], s: &[f32], length: usize, order: usize) {
    unsafe { silk_LPC_analysis_filter_FLP(r_lpc.as_mut_ptr(), pred_coef.as_ptr(), s.as_ptr(), length as i32, order as i32) }
}

pub fn c_silk_lpc_inverse_pred_gain_flp(a: &[f32], order: usize) -> f32 {
    unsafe { silk_LPC_inverse_pred_gain_FLP(a.as_ptr(), order as i32) }
}

pub fn c_silk_autocorrelation_flp(results: &mut [f32], input: &[f32], corr_count: usize) {
    unsafe { silk_autocorrelation_FLP(results.as_mut_ptr(), input.as_ptr(), input.len() as i32, corr_count as i32, 0) }
}
