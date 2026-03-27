use super::*;

/// DRED decoded state. Matches C `OpusDRED` from dred_decoder.h.
pub struct OpusDred {
    pub fec_features: Vec<f32>,
    pub state: Vec<f32>,
    pub latents: Vec<f32>,
    pub nb_latents: usize,
    pub process_stage: i32,
    pub dred_offset: i32,
    pub latent_dim: usize,
    pub state_dim: usize,
}

impl OpusDred {
    pub fn new(latent_dim: usize, state_dim: usize) -> Self {
        OpusDred {
            fec_features: vec![0.0; 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES],
            state: vec![0.0; state_dim],
            latents: vec![0.0; (DRED_NUM_REDUNDANCY_FRAMES / 2) * (latent_dim + 1)],
            nb_latents: 0,
            process_stage: 0,
            dred_offset: 0,
            latent_dim,
            state_dim,
        }
    }
}

/// Decode DRED latents from entropy-coded bytes.
/// Matches C `dred_ec_decode` from dred_decoder.c.
///
/// This function requires the range coder (ec_dec) and laplace decoder
/// from opus-range-coder, plus the quantization statistics tables
/// (quant_scales, r_q8, p0_q8) from the model weight data.
///
/// Returns the number of decoded latent pairs, or 0 on failure.
///
/// NOTE: Full implementation requires integration with the opus-range-coder
/// crate for entropy decoding (ec_dec_init, ec_dec_uint, ec_decode,
/// ec_laplace_decode_p0). This will be completed in Phase 7 (integration).
/// For now, the structure and state management are in place.
pub fn dred_ec_decode(
    dec: &mut OpusDred,
    _bytes: &[u8],
    _num_bytes: usize,
    _min_feature_frames: usize,
    _dred_frame_offset: i32,
    _state_quant_scales: &[u8],
    _state_r: &[u8],
    _state_p0: &[u8],
    _latent_quant_scales: &[u8],
    _latent_r: &[u8],
    _latent_p0: &[u8],
) -> usize {
    // TODO: Implement entropy decoding using opus-range-coder.
    // The C implementation uses:
    //   ec_dec_init(&ec, bytes, num_bytes)
    //   q0 = ec_dec_uint(&ec, 16)
    //   dQ = ec_dec_uint(&ec, 8)
    //   ... then loops decoding latents via ec_laplace_decode_p0
    //
    // This requires:
    //   1. opus-range-coder dependency
    //   2. Laplace decoder (ec_laplace_decode_p0)
    //   3. Stats data from dred_rdovae_stats_data.h (quant_scales, r, p0)
    //
    // For now, set state to indicate no latents decoded.
    dec.nb_latents = 0;
    dec.process_stage = 0;
    0
}

/// Decode latent vector from range coder using quantization stats.
/// Matches C `dred_decode_latents` from dred_decoder.c.
///
/// NOTE: Requires ec_laplace_decode_p0 from opus-range-coder.
/// Stub implementation for now.
pub fn dred_decode_latents(
    x: &mut [f32],
    _scale: &[u8],
    _r: &[u8],
    _p0: &[u8],
    dim: usize,
) {
    // TODO: Implement using ec_laplace_decode_p0.
    // For each element:
    //   if r[i]==0 || p0[i]==255: q = 0
    //   else: q = ec_laplace_decode_p0(dec, p0[i]<<7, r[i]<<7)
    //   x[i] = q * 256.0 / max(1, scale[i])
    for i in 0..dim {
        x[i] = 0.0;
    }
}
