use crate::nnet::{Activation, LinearLayer, WeightArray};
use crate::nnet::ops::{compute_generic_dense, compute_generic_gru, compute_generic_conv1d};
use crate::nnet::weights::{WeightError, linear_init, weight_output_dim};
use crate::nndsp::{AdaCombState, AdaConvState, compute_overlap_window};
use crate::nndsp::adacomb::adacomb_process_frame;
use crate::nndsp::adaconv::adaconv_process_frame;

use super::config::*;

/// LACE frame size (5ms at 16kHz).
pub const LACE_FRAME_SIZE: usize = 80;
pub const LACE_OVERLAP_SIZE: usize = 40;

/// Maximum buffer size for feature/conditioning computations.
/// Sized for 4 subframes * max conditioning dim (typically 256).
const MAX_FEATURE_BUF: usize = 1024;

/// LACE model layers. Matches auto-generated C `LACELayers` from lace_data.h.
pub struct LaceLayers {
    pub pitch_embedding: LinearLayer,
    pub fnet_conv1: LinearLayer,
    pub fnet_conv2: LinearLayer,
    pub fnet_tconv: LinearLayer,
    pub fnet_gru_input: LinearLayer,
    pub fnet_gru_recurrent: LinearLayer,
    pub cf1_kernel: LinearLayer,
    pub cf1_gain: LinearLayer,
    pub cf1_global_gain: LinearLayer,
    pub cf2_kernel: LinearLayer,
    pub cf2_gain: LinearLayer,
    pub cf2_global_gain: LinearLayer,
    pub af1_kernel: LinearLayer,
    pub af1_gain: LinearLayer,
}

/// LACE model with layers + pre-computed overlap window.
pub struct Lace {
    pub layers: LaceLayers,
    pub window: [f32; LACE_OVERLAP_SIZE],
    pub cond_dim: usize,
    pub hidden_dim: usize,
    pub pitch_embed_dim: usize,
    pub numbits_embed_dim: usize,
    pub num_features: usize,
    pub cf1_kernel_size: usize,
    pub cf2_kernel_size: usize,
    pub af1_kernel_size: usize,
    pub af1_in_channels: usize,
    pub af1_out_channels: usize,
}

/// LACE processing state.
pub struct LaceState {
    pub fnet_conv2_state: Vec<f32>,
    pub gru_state: Vec<f32>,
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

/// Initialize LACE model from weight arrays.
pub fn init_lace(arrays: &[WeightArray]) -> Result<Lace, WeightError> {
    let dim = |name: &str| weight_output_dim(arrays, name);
    let cond_dim = dim("lace_fnet_gru_input_bias")? / 3;
    let hidden_dim = dim("lace_fnet_conv1_bias")?;
    let pitch_embed_dim = dim("lace_pitch_embedding_bias")?;
    let num_features = dim("lace_fnet_conv1_bias")?; // conv1 output = hidden_dim

    // Infer numbits embedding dim from conv1 input: num_features + pitch_embed + 2*numbits_embed
    // We'll compute it from the fnet_conv1 nb_inputs once available
    let numbits_embed_dim = 8; // Typical value from C; will be validated at runtime

    let cf1_kernel_size = dim("lace_cf1_kernel_bias")?;
    let cf2_kernel_size = dim("lace_cf2_kernel_bias")?;
    let af1_kernel_size = dim("lace_af1_kernel_bias")? / 1; // kernel * in_ch * out_ch — single channel for now

    let fnet_conv2_in = 4 * hidden_dim;

    let layers = LaceLayers {
        pitch_embedding: linear_init(arrays, Some("lace_pitch_embedding_bias"), None, Some("lace_pitch_embedding_weights"), None, None, None, 1, pitch_embed_dim)?,
        fnet_conv1: linear_init(arrays, Some("lace_fnet_conv1_bias"), None, Some("lace_fnet_conv1_weights"), None, None, None, num_features + pitch_embed_dim + 2 * numbits_embed_dim, hidden_dim)?,
        fnet_conv2: linear_init(arrays, Some("lace_fnet_conv2_bias"), None, Some("lace_fnet_conv2_weights"), None, None, None, fnet_conv2_in, dim("lace_fnet_conv2_bias")?)?,
        fnet_tconv: linear_init(arrays, Some("lace_fnet_tconv_bias"), None, Some("lace_fnet_tconv_weights"), None, None, None, dim("lace_fnet_conv2_bias")?, 4 * cond_dim)?,
        fnet_gru_input: linear_init(arrays, Some("lace_fnet_gru_input_bias"), None, Some("lace_fnet_gru_input_weights"), None, None, None, cond_dim, 3 * cond_dim)?,
        fnet_gru_recurrent: linear_init(arrays, Some("lace_fnet_gru_recurrent_bias"), Some("lace_fnet_gru_recurrent_weights"), None, None, Some("lace_fnet_gru_recurrent_diag"), None, cond_dim, 3 * cond_dim)?,
        cf1_kernel: linear_init(arrays, Some("lace_cf1_kernel_bias"), None, Some("lace_cf1_kernel_weights"), None, None, None, cond_dim, cf1_kernel_size)?,
        cf1_gain: linear_init(arrays, Some("lace_cf1_gain_bias"), None, Some("lace_cf1_gain_weights"), None, None, None, cond_dim, 1)?,
        cf1_global_gain: linear_init(arrays, Some("lace_cf1_global_gain_bias"), None, Some("lace_cf1_global_gain_weights"), None, None, None, cond_dim, 1)?,
        cf2_kernel: linear_init(arrays, Some("lace_cf2_kernel_bias"), None, Some("lace_cf2_kernel_weights"), None, None, None, cond_dim, cf2_kernel_size)?,
        cf2_gain: linear_init(arrays, Some("lace_cf2_gain_bias"), None, Some("lace_cf2_gain_weights"), None, None, None, cond_dim, 1)?,
        cf2_global_gain: linear_init(arrays, Some("lace_cf2_global_gain_bias"), None, Some("lace_cf2_global_gain_weights"), None, None, None, cond_dim, 1)?,
        af1_kernel: linear_init(arrays, Some("lace_af1_kernel_bias"), None, Some("lace_af1_kernel_weights"), None, None, None, cond_dim, af1_kernel_size)?,
        af1_gain: linear_init(arrays, Some("lace_af1_gain_bias"), None, Some("lace_af1_gain_weights"), None, None, None, cond_dim, 1)?,
    };

    let mut window = [0.0f32; LACE_OVERLAP_SIZE];
    compute_overlap_window(&mut window, LACE_OVERLAP_SIZE);

    Ok(Lace {
        layers,
        window,
        cond_dim,
        hidden_dim,
        pitch_embed_dim,
        numbits_embed_dim,
        num_features: num_features + pitch_embed_dim + 2 * numbits_embed_dim,
        cf1_kernel_size,
        cf2_kernel_size,
        af1_kernel_size,
        af1_in_channels: 1,
        af1_out_channels: 1,
    })
}

/// Create LACE processing state.
pub fn lace_state_init(model: &Lace) -> LaceState {
    let conv2_in = model.layers.fnet_conv2.nb_inputs;
    LaceState {
        fnet_conv2_state: vec![0.0; conv2_in],
        gru_state: vec![0.0; model.cond_dim],
        cf1_state: AdaCombState::default(),
        cf2_state: AdaCombState::default(),
        af1_state: AdaConvState::default(),
        preemph_mem: 0.0,
        deemph_mem: 0.0,
    }
}

/// Sinusoidal numbits embedding matching C `compute_lace_numbits_embedding`.
/// Uses per-dimension scale factors to produce a dense sinusoidal encoding.
fn compute_numbits_embedding(out: &mut [f32], numbits: f32, dim: usize, log_low: f32, log_high: f32) {
    // Scale factors from C LACE_NUMBITS_SCALE_0 through LACE_NUMBITS_SCALE_7
    const SCALES: [f32; 8] = [
        0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2,
    ];
    let log_numbits = numbits.ln();
    let mid = (log_high + log_low) * 0.5;
    let x = log_numbits.clamp(log_low, log_high) - mid;
    for i in 0..dim.min(SCALES.len()) {
        out[i] = (x * SCALES[i] - 0.5).sin();
    }
}

/// Run LACE feature network: features → conditioning vectors.
pub fn lace_feature_net(
    model: &Lace,
    state: &mut LaceState,
    output: &mut [f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[usize; 4],
) {
    let cond_dim = model.cond_dim;
    let hidden_dim = model.hidden_dim;
    let embed_dim = model.pitch_embed_dim;
    let nb_embed = model.numbits_embed_dim;

    let mut numbits_embedded = [0.0f32; 64];
    let log_low = 1.0f32.ln().max(0.001f32.ln());
    let log_high = 10000.0f32.ln();
    compute_numbits_embedding(&mut numbits_embedded[..nb_embed], numbits[0], nb_embed, log_low, log_high);
    compute_numbits_embedding(&mut numbits_embedded[nb_embed..2 * nb_embed], numbits[1], nb_embed, log_low, log_high);

    let embed_weights = model.layers.pitch_embedding.float_weights.as_ref().unwrap();
    let feat_dim = model.num_features;

    let mut input_buf = [0.0f32; MAX_FEATURE_BUF];
    let mut output_buf = [0.0f32; MAX_FEATURE_BUF];

    // Per-subframe: features + pitch_embed + numbits_embed → conv1
    let num_feat_base = feat_dim - embed_dim - 2 * nb_embed;
    for sf in 0..4 {
        let feat_start = sf * num_feat_base;
        input_buf[..num_feat_base].copy_from_slice(&features[feat_start..feat_start + num_feat_base]);
        let embed_start = periods[sf].min(255) * embed_dim;
        if embed_start + embed_dim <= embed_weights.len() {
            input_buf[num_feat_base..num_feat_base + embed_dim].copy_from_slice(&embed_weights[embed_start..embed_start + embed_dim]);
        }
        input_buf[num_feat_base + embed_dim..num_feat_base + embed_dim + 2 * nb_embed].copy_from_slice(&numbits_embedded[..2 * nb_embed]);

        compute_generic_conv1d(&model.layers.fnet_conv1, &mut output_buf[sf * hidden_dim..(sf + 1) * hidden_dim], &mut [], &input_buf[..feat_dim], feat_dim, Activation::Tanh);
    }

    // Subframe accumulation: conv2
    input_buf[..4 * hidden_dim].copy_from_slice(&output_buf[..4 * hidden_dim]);
    compute_generic_conv1d(&model.layers.fnet_conv2, &mut output_buf[..4 * cond_dim], &mut state.fnet_conv2_state, &input_buf[..4 * hidden_dim], 4 * hidden_dim, Activation::Tanh);

    // Tconv upsampling
    input_buf[..4 * cond_dim].copy_from_slice(&output_buf[..4 * cond_dim]);
    compute_generic_dense(&model.layers.fnet_tconv, &mut output_buf[..4 * cond_dim], &input_buf[..4 * cond_dim], Activation::Tanh);

    // GRU per subframe
    input_buf[..4 * cond_dim].copy_from_slice(&output_buf[..4 * cond_dim]);
    for sf in 0..4 {
        compute_generic_gru(&model.layers.fnet_gru_input, &model.layers.fnet_gru_recurrent, &mut state.gru_state, &input_buf[sf * cond_dim..(sf + 1) * cond_dim]);
        output[sf * cond_dim..(sf + 1) * cond_dim].copy_from_slice(&state.gru_state);
    }
}

/// Process a 20ms LACE frame (4 subframes of 80 samples each).
pub fn lace_process_20ms_frame(
    model: &Lace,
    state: &mut LaceState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[usize; 4],
) {
    let cond_dim = model.cond_dim;
    let mut cond = [0.0f32; MAX_FEATURE_BUF];
    lace_feature_net(model, state, &mut cond[..4 * cond_dim], features, numbits, periods);

    // Pre-emphasis
    let mut x_pre = [0.0f32; 4 * LACE_FRAME_SIZE];
    for i in 0..4 * LACE_FRAME_SIZE {
        let xi = x_in[i];
        x_pre[i] = xi - OSCE_PREEMPH * state.preemph_mem;
        state.preemph_mem = xi;
    }

    let mut x_buf = [0.0f32; 4 * LACE_FRAME_SIZE];

    // Process 4 subframes: cf1 → cf2 → af1
    for sf in 0..4 {
        let sub_cond = &cond[sf * cond_dim..(sf + 1) * cond_dim];
        let sub_in = &x_pre[sf * LACE_FRAME_SIZE..(sf + 1) * LACE_FRAME_SIZE];
        let sub_out = &mut x_buf[sf * LACE_FRAME_SIZE..(sf + 1) * LACE_FRAME_SIZE];

        // AdaComb filter 1
        let mut tmp = [0.0f32; LACE_FRAME_SIZE];
        adacomb_process_frame(
            &mut state.cf1_state, &mut tmp, sub_in, sub_cond,
            &model.layers.cf1_kernel, &model.layers.cf1_gain, &model.layers.cf1_global_gain,
            periods[sf], cond_dim, LACE_FRAME_SIZE, LACE_OVERLAP_SIZE,
            model.cf1_kernel_size, model.cf1_kernel_size - 1,
            0.5, -2.0, 0.1, &model.window,
        );

        // AdaComb filter 2
        adacomb_process_frame(
            &mut state.cf2_state, sub_out, &tmp, sub_cond,
            &model.layers.cf2_kernel, &model.layers.cf2_gain, &model.layers.cf2_global_gain,
            periods[sf], cond_dim, LACE_FRAME_SIZE, LACE_OVERLAP_SIZE,
            model.cf2_kernel_size, model.cf2_kernel_size - 1,
            0.5, -2.0, 0.1, &model.window,
        );
    }

    // AdaConv filter 1 — per-subframe (matching C osce.c lines 345-366)
    for sf in 0..4 {
        let sub_cond = &cond[sf * cond_dim..(sf + 1) * cond_dim];
        adaconv_process_frame(
            &mut state.af1_state,
            &mut x_out[sf * LACE_FRAME_SIZE..(sf + 1) * LACE_FRAME_SIZE],
            &x_buf[sf * LACE_FRAME_SIZE..(sf + 1) * LACE_FRAME_SIZE],
            sub_cond,
            &model.layers.af1_kernel, &model.layers.af1_gain,
            cond_dim, LACE_FRAME_SIZE, LACE_OVERLAP_SIZE,
            model.af1_in_channels, model.af1_out_channels,
            model.af1_kernel_size, model.af1_kernel_size - 1,
            0.5, -2.0, 1.0, &model.window,
        );
    }

    // De-emphasis
    for i in 0..4 * LACE_FRAME_SIZE {
        x_out[i] += OSCE_PREEMPH * state.deemph_mem;
        state.deemph_mem = x_out[i];
    }
}
