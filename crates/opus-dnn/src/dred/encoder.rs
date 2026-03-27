use crate::freq::PREEMPHASIS;
use crate::lpcnet::enc::{LpcnetEncState, preemphasis, compute_frame_features, NB_TOTAL_FEATURES};

use super::*;
use super::rdovae_enc::*;

/// DRED encoder state. Matches C `DREDEnc` from dred_encoder.h.
pub struct DredEnc {
    pub model: RdovaeEnc,
    pub lpcnet_enc_state: LpcnetEncState,
    pub rdovae_enc: RdovaeEncState,
    pub loaded: bool,

    pub input_buffer: [f32; 2 * DRED_DFRAME_SIZE],
    pub input_buffer_fill: usize,
    pub dred_offset: i32,
    pub latent_offset: i32,
    pub latents_buffer: Vec<f32>,
    pub latents_buffer_fill: usize,
    pub state_buffer: Vec<f32>,
    pub resample_mem: [f32; 9],
}

/// Initialize DRED encoder state.
pub fn dred_encoder_init(
    model: RdovaeEnc,
    lpcnet_enc_state: LpcnetEncState,
) -> DredEnc {
    let rdovae_enc = rdovae_enc_state_init(&model);
    let latent_dim = model.latent_dim;
    let state_dim = model.state_dim;
    DredEnc {
        model,
        lpcnet_enc_state,
        rdovae_enc,
        loaded: true,
        input_buffer: [0.0; 2 * DRED_DFRAME_SIZE],
        input_buffer_fill: 0,
        dred_offset: 0,
        latent_offset: 0,
        latents_buffer: vec![0.0; DRED_MAX_FRAMES * latent_dim],
        latents_buffer_fill: 0,
        state_buffer: vec![0.0; DRED_MAX_FRAMES * state_dim],
        resample_mem: [0.0; 9],
    }
}

/// Reset encoder state for a new stream.
pub fn dred_encoder_reset(enc: &mut DredEnc) {
    enc.input_buffer = [0.0; 2 * DRED_DFRAME_SIZE];
    enc.input_buffer_fill = 0;
    enc.dred_offset = 0;
    enc.latent_offset = 0;
    enc.latents_buffer_fill = 0;
    enc.resample_mem = [0.0; 9];
    enc.rdovae_enc = rdovae_enc_state_init(&enc.model);
}

/// Compute DRED latents from PCM input.
/// Runs LPCNet feature extraction on each frame, accumulates features
/// into double-frames, and encodes them through the RDOVAE encoder.
/// Matches C `dred_compute_latents` from dred_encoder.c.
pub fn dred_compute_latents(
    enc: &mut DredEnc,
    pcm: &[f32],
    frame_size: usize,
    _extra_delay: usize,
) {
    let latent_dim = enc.model.latent_dim;
    let state_dim = enc.model.state_dim;
    let mut remaining = frame_size;
    let mut pcm_offset = 0;

    while remaining > 0 {
        let copy_len = remaining.min(DRED_DFRAME_SIZE * 2 - enc.input_buffer_fill);
        enc.input_buffer[enc.input_buffer_fill..enc.input_buffer_fill + copy_len]
            .copy_from_slice(&pcm[pcm_offset..pcm_offset + copy_len]);
        enc.input_buffer_fill += copy_len;
        pcm_offset += copy_len;
        remaining -= copy_len;

        if enc.input_buffer_fill >= DRED_DFRAME_SIZE {
            let mut dframe_features = [0.0f32; 2 * NB_TOTAL_FEATURES];

            // First frame
            let mut x = [0.0f32; DRED_FRAME_SIZE];
            x.copy_from_slice(&enc.input_buffer[..DRED_FRAME_SIZE]);
            let x_copy = x;
            preemphasis(&mut x, &mut enc.lpcnet_enc_state.mem_preemph, &x_copy, PREEMPHASIS, DRED_FRAME_SIZE);
            compute_frame_features(&mut enc.lpcnet_enc_state, &x);
            dframe_features[..NB_TOTAL_FEATURES].copy_from_slice(&enc.lpcnet_enc_state.features);

            // Second frame
            x.copy_from_slice(&enc.input_buffer[DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE]);
            let x_copy = x;
            preemphasis(&mut x, &mut enc.lpcnet_enc_state.mem_preemph, &x_copy, PREEMPHASIS, DRED_FRAME_SIZE);
            compute_frame_features(&mut enc.lpcnet_enc_state, &x);
            dframe_features[NB_TOTAL_FEATURES..].copy_from_slice(&enc.lpcnet_enc_state.features);

            // Encode through RDOVAE
            let lat_start = enc.latents_buffer_fill * latent_dim;
            let state_start = enc.latents_buffer_fill * state_dim;
            if lat_start + latent_dim <= enc.latents_buffer.len()
                && state_start + state_dim <= enc.state_buffer.len()
            {
                dred_rdovae_encode_dframe(
                    &mut enc.rdovae_enc,
                    &enc.model,
                    &mut enc.latents_buffer[lat_start..lat_start + latent_dim],
                    &mut enc.state_buffer[state_start..state_start + state_dim],
                    &dframe_features,
                );
                enc.latents_buffer_fill += 1;
            }

            // Shift input buffer
            let dframe_samples = DRED_DFRAME_SIZE;
            enc.input_buffer.copy_within(dframe_samples.., 0);
            enc.input_buffer_fill -= dframe_samples;
        }
    }
}
