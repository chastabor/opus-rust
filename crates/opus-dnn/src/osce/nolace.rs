use crate::nnet::{Activation, LinearLayer, WeightArray};
use crate::nnet::ops::compute_generic_conv1d;
use crate::nnet::weights::{WeightError, linear_init, weight_output_dim};
use crate::nndsp::{AdaCombState, AdaConvState, AdaShapeState, compute_overlap_window};
use crate::nndsp::adacomb::adacomb_process_frame;
use crate::nndsp::adaconv::adaconv_process_frame;
use crate::nndsp::adashape::adashape_process_frame;

use super::config::*;
use super::common::*;

pub const NOLACE_FRAME_SIZE: usize = 80;
const NOLACE_OVERLAP_SIZE: usize = 40;

// Filter gain constants from nolace_data.h.
const CF_FILTER_GAIN_A: f32 = 0.690776;
const CF_FILTER_GAIN_B: f32 = 0.0;
const CF_LOG_GAIN_LIMIT: f32 = 1.151293;
const AF_FILTER_GAIN_A: f32 = 1.381551;
const AF_FILTER_GAIN_B: f32 = 0.0;

/// NoLACE model layers (34 total). Matches C `NOLACELayers` from nolace_data.h.
pub struct NoLaceLayers {
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
    pub tdshape1_alpha1_f: LinearLayer,
    pub tdshape1_alpha1_t: LinearLayer,
    pub tdshape1_alpha2: LinearLayer,
    pub tdshape2_alpha1_f: LinearLayer,
    pub tdshape2_alpha1_t: LinearLayer,
    pub tdshape2_alpha2: LinearLayer,
    pub tdshape3_alpha1_f: LinearLayer,
    pub tdshape3_alpha1_t: LinearLayer,
    pub tdshape3_alpha2: LinearLayer,
    pub af2_kernel: LinearLayer,
    pub af2_gain: LinearLayer,
    pub af3_kernel: LinearLayer,
    pub af3_gain: LinearLayer,
    pub af4_kernel: LinearLayer,
    pub af4_gain: LinearLayer,
    pub post_cf1: LinearLayer,
    pub post_cf2: LinearLayer,
    pub post_af1: LinearLayer,
    pub post_af2: LinearLayer,
    pub post_af3: LinearLayer,
}

impl FeatureNetLayers for NoLaceLayers {
    fn pitch_embedding(&self) -> &LinearLayer { &self.pitch_embedding }
    fn fnet_conv1(&self) -> &LinearLayer { &self.fnet_conv1 }
    fn fnet_conv2(&self) -> &LinearLayer { &self.fnet_conv2 }
    fn fnet_tconv(&self) -> &LinearLayer { &self.fnet_tconv }
    fn fnet_gru_input(&self) -> &LinearLayer { &self.fnet_gru_input }
    fn fnet_gru_recurrent(&self) -> &LinearLayer { &self.fnet_gru_recurrent }
}

/// NoLACE model with layers + window + shared params.
pub struct NoLace {
    pub layers: NoLaceLayers,
    pub window: [f32; NOLACE_OVERLAP_SIZE],
    pub params: FeatureNetParams,
}

/// NoLACE processing state.
pub struct NoLaceState {
    pub fnet_conv2_state: Vec<f32>,
    pub gru_state: Vec<f32>,
    pub post_cf1_state: Vec<f32>,
    pub post_cf2_state: Vec<f32>,
    pub post_af1_state: Vec<f32>,
    pub post_af2_state: Vec<f32>,
    pub post_af3_state: Vec<f32>,
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub af2_state: AdaConvState,
    pub af3_state: AdaConvState,
    pub af4_state: AdaConvState,
    pub tdshape1_state: AdaShapeState,
    pub tdshape2_state: AdaShapeState,
    pub tdshape3_state: AdaShapeState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl FeatureNetState for NoLaceState {
    fn fnet_conv2_state(&mut self) -> &mut [f32] { &mut self.fnet_conv2_state }
    fn gru_state(&mut self) -> &mut [f32] { &mut self.gru_state }
}

/// Initialize NoLACE model from weight arrays.
pub fn init_nolace(arrays: &[WeightArray]) -> Result<NoLace, WeightError> {
    let dim = |name: &str| weight_output_dim(arrays, name);
    let cond_dim = dim("nolace_fnet_gru_input_bias")? / 3;
    let hidden_dim = dim("nolace_fnet_conv1_bias")?;
    let pitch_embed_dim = dim("nolace_pitch_embedding_bias")?;
    let numbits_embed_dim = 8;
    let feat_in = OSCE_FEATURE_DIM + pitch_embed_dim + 2 * numbits_embed_dim;
    let fnet_conv2_in = 4 * hidden_dim;

    let l = |bias: &str, weights: &str, ni: usize, no: usize| -> Result<LinearLayer, WeightError> {
        linear_init(arrays, Some(bias), None, Some(weights), None, None, None, ni, no)
    };
    let lg = |bias: &str, weights: &str, diag: &str, ni: usize, no: usize| -> Result<LinearLayer, WeightError> {
        linear_init(arrays, Some(bias), Some(weights), None, None, Some(diag), None, ni, no)
    };

    let cf1_ks = dim("nolace_cf1_kernel_bias")?;
    let cf2_ks = dim("nolace_cf2_kernel_bias")?;
    let af1_ks_total = dim("nolace_af1_kernel_bias")?;
    let af2_ks_total = dim("nolace_af2_kernel_bias")?;
    let af3_ks_total = dim("nolace_af3_kernel_bias")?;
    let af4_ks_total = dim("nolace_af4_kernel_bias")?;
    let post_dim = dim("nolace_post_cf1_bias")?;

    let layers = NoLaceLayers {
        pitch_embedding: l("nolace_pitch_embedding_bias", "nolace_pitch_embedding_weights", 1, pitch_embed_dim)?,
        fnet_conv1: l("nolace_fnet_conv1_bias", "nolace_fnet_conv1_weights", feat_in, hidden_dim)?,
        fnet_conv2: l("nolace_fnet_conv2_bias", "nolace_fnet_conv2_weights", fnet_conv2_in, dim("nolace_fnet_conv2_bias")?)?,
        fnet_tconv: l("nolace_fnet_tconv_bias", "nolace_fnet_tconv_weights", dim("nolace_fnet_conv2_bias")?, 4 * cond_dim)?,
        fnet_gru_input: l("nolace_fnet_gru_input_bias", "nolace_fnet_gru_input_weights", cond_dim, 3 * cond_dim)?,
        fnet_gru_recurrent: lg("nolace_fnet_gru_recurrent_bias", "nolace_fnet_gru_recurrent_weights", "nolace_fnet_gru_recurrent_diag", cond_dim, 3 * cond_dim)?,
        cf1_kernel: l("nolace_cf1_kernel_bias", "nolace_cf1_kernel_weights", cond_dim, cf1_ks)?,
        cf1_gain: l("nolace_cf1_gain_bias", "nolace_cf1_gain_weights", cond_dim, 1)?,
        cf1_global_gain: l("nolace_cf1_global_gain_bias", "nolace_cf1_global_gain_weights", cond_dim, 1)?,
        cf2_kernel: l("nolace_cf2_kernel_bias", "nolace_cf2_kernel_weights", cond_dim, cf2_ks)?,
        cf2_gain: l("nolace_cf2_gain_bias", "nolace_cf2_gain_weights", cond_dim, 1)?,
        cf2_global_gain: l("nolace_cf2_global_gain_bias", "nolace_cf2_global_gain_weights", cond_dim, 1)?,
        af1_kernel: l("nolace_af1_kernel_bias", "nolace_af1_kernel_weights", cond_dim, af1_ks_total)?,
        af1_gain: l("nolace_af1_gain_bias", "nolace_af1_gain_weights", cond_dim, 2)?,
        tdshape1_alpha1_f: l("nolace_tdshape1_alpha1_f_bias", "nolace_tdshape1_alpha1_f_weights", cond_dim, dim("nolace_tdshape1_alpha1_f_bias")?)?,
        tdshape1_alpha1_t: l("nolace_tdshape1_alpha1_t_bias", "nolace_tdshape1_alpha1_t_weights", 21, dim("nolace_tdshape1_alpha1_t_bias")?)?,
        tdshape1_alpha2: l("nolace_tdshape1_alpha2_bias", "nolace_tdshape1_alpha2_weights", dim("nolace_tdshape1_alpha2_bias")?, dim("nolace_tdshape1_alpha2_bias")?)?,
        tdshape2_alpha1_f: l("nolace_tdshape2_alpha1_f_bias", "nolace_tdshape2_alpha1_f_weights", cond_dim, dim("nolace_tdshape2_alpha1_f_bias")?)?,
        tdshape2_alpha1_t: l("nolace_tdshape2_alpha1_t_bias", "nolace_tdshape2_alpha1_t_weights", 21, dim("nolace_tdshape2_alpha1_t_bias")?)?,
        tdshape2_alpha2: l("nolace_tdshape2_alpha2_bias", "nolace_tdshape2_alpha2_weights", dim("nolace_tdshape2_alpha2_bias")?, dim("nolace_tdshape2_alpha2_bias")?)?,
        tdshape3_alpha1_f: l("nolace_tdshape3_alpha1_f_bias", "nolace_tdshape3_alpha1_f_weights", cond_dim, dim("nolace_tdshape3_alpha1_f_bias")?)?,
        tdshape3_alpha1_t: l("nolace_tdshape3_alpha1_t_bias", "nolace_tdshape3_alpha1_t_weights", 21, dim("nolace_tdshape3_alpha1_t_bias")?)?,
        tdshape3_alpha2: l("nolace_tdshape3_alpha2_bias", "nolace_tdshape3_alpha2_weights", dim("nolace_tdshape3_alpha2_bias")?, dim("nolace_tdshape3_alpha2_bias")?)?,
        af2_kernel: l("nolace_af2_kernel_bias", "nolace_af2_kernel_weights", cond_dim, af2_ks_total)?,
        af2_gain: l("nolace_af2_gain_bias", "nolace_af2_gain_weights", cond_dim, 2)?,
        af3_kernel: l("nolace_af3_kernel_bias", "nolace_af3_kernel_weights", cond_dim, af3_ks_total)?,
        af3_gain: l("nolace_af3_gain_bias", "nolace_af3_gain_weights", cond_dim, 2)?,
        af4_kernel: l("nolace_af4_kernel_bias", "nolace_af4_kernel_weights", cond_dim, af4_ks_total)?,
        af4_gain: l("nolace_af4_gain_bias", "nolace_af4_gain_weights", cond_dim, 1)?,
        post_cf1: l("nolace_post_cf1_bias", "nolace_post_cf1_weights", cond_dim, post_dim)?,
        post_cf2: l("nolace_post_cf2_bias", "nolace_post_cf2_weights", cond_dim, post_dim)?,
        post_af1: l("nolace_post_af1_bias", "nolace_post_af1_weights", cond_dim, post_dim)?,
        post_af2: l("nolace_post_af2_bias", "nolace_post_af2_weights", cond_dim, post_dim)?,
        post_af3: l("nolace_post_af3_bias", "nolace_post_af3_weights", cond_dim, post_dim)?,
    };

    let mut window = [0.0f32; NOLACE_OVERLAP_SIZE];
    compute_overlap_window(&mut window, NOLACE_OVERLAP_SIZE);

    Ok(NoLace {
        layers,
        window,
        params: FeatureNetParams {
            cond_dim,
            hidden_dim,
            pitch_embed_dim,
            numbits_embed_dim,
            pitch_max: 299,
            numbits_range_low: 50.0,
            numbits_range_high: 650.0,
            numbits_scales: [
                1.0357312, 1.7355591, 3.6004558, 4.5524783,
                5.9325595, 7.1769705, 8.1149988, 8.7706327,
            ],
        },
    })
}

pub fn nolace_state_init(model: &NoLace) -> NoLaceState {
    let cd = model.params.cond_dim;
    NoLaceState {
        fnet_conv2_state: vec![0.0; model.layers.fnet_conv2.nb_inputs],
        gru_state: vec![0.0; cd],
        post_cf1_state: vec![0.0; cd],
        post_cf2_state: vec![0.0; cd],
        post_af1_state: vec![0.0; cd],
        post_af2_state: vec![0.0; cd],
        post_af3_state: vec![0.0; cd],
        cf1_state: AdaCombState::default(),
        cf2_state: AdaCombState::default(),
        af1_state: AdaConvState::default(),
        af2_state: AdaConvState::default(),
        af3_state: AdaConvState::default(),
        af4_state: AdaConvState::default(),
        tdshape1_state: AdaShapeState::default(),
        tdshape2_state: AdaShapeState::default(),
        tdshape3_state: AdaShapeState::default(),
        preemph_mem: 0.0,
        deemph_mem: 0.0,
    }
}

/// Process a 20ms NoLACE frame.
/// Chain: feature net → cf1 → post_cf1 → cf2 → post_cf2 →
///        af1 → post_af1 → tdshape1 → af2 → post_af2 → tdshape2 →
///        af3 → post_af3 → tdshape3 → af4 → de-emphasis
pub fn nolace_process_20ms_frame(
    model: &NoLace,
    state: &mut NoLaceState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[usize; 4],
) {
    let cd = model.params.cond_dim;
    let mut cond = [0.0f32; 1024];
    osce_feature_net(&model.layers, state, &model.params, &mut cond[..4 * cd], features, numbits, periods);

    let total = 4 * NOLACE_FRAME_SIZE;
    let mut x = [0.0f32; 4 * 80];
    apply_preemphasis(&mut x[..total], &x_in[..total], &mut state.preemph_mem);

    let mut buf1 = [0.0f32; 4 * 80];
    let mut buf2 = [0.0f32; 4 * 80 * 2];

    for sf in 0..4 {
        let sc = &cond[sf * cd..(sf + 1) * cd];
        let si = sf * NOLACE_FRAME_SIZE;
        let ei = si + NOLACE_FRAME_SIZE;
        let si2 = sf * NOLACE_FRAME_SIZE * 2;
        let ch2 = si2 + NOLACE_FRAME_SIZE; // second channel offset

        let mut post_out = [0.0f32; 256];

        // cf1
        adacomb_process_frame(&mut state.cf1_state, &mut buf1[si..ei], &x[si..ei], sc,
            &model.layers.cf1_kernel, &model.layers.cf1_gain, &model.layers.cf1_global_gain,
            periods[sf], cd, NOLACE_FRAME_SIZE, NOLACE_OVERLAP_SIZE, 16, 8,
            CF_FILTER_GAIN_A, CF_FILTER_GAIN_B, CF_LOG_GAIN_LIMIT, &model.window);

        compute_generic_conv1d(&model.layers.post_cf1, &mut post_out[..cd], &mut state.post_cf1_state, sc, cd, Activation::Tanh);

        // cf2
        adacomb_process_frame(&mut state.cf2_state, &mut x[si..ei], &buf1[si..ei], &post_out[..cd],
            &model.layers.cf2_kernel, &model.layers.cf2_gain, &model.layers.cf2_global_gain,
            periods[sf], cd, NOLACE_FRAME_SIZE, NOLACE_OVERLAP_SIZE, 16, 8,
            CF_FILTER_GAIN_A, CF_FILTER_GAIN_B, CF_LOG_GAIN_LIMIT, &model.window);

        compute_generic_conv1d(&model.layers.post_cf2, &mut post_out[..cd], &mut state.post_cf2_state, sc, cd, Activation::Tanh);

        // af1 (1 → 2 channels)
        adaconv_process_frame(&mut state.af1_state, &mut buf2[si2..si2 + NOLACE_FRAME_SIZE * 2],
            &x[si..ei], &post_out[..cd],
            &model.layers.af1_kernel, &model.layers.af1_gain,
            cd, NOLACE_FRAME_SIZE, NOLACE_OVERLAP_SIZE, 1, 2, 16, 15,
            AF_FILTER_GAIN_A, AF_FILTER_GAIN_B, 1.0, &model.window);

        compute_generic_conv1d(&model.layers.post_af1, &mut post_out[..cd], &mut state.post_af1_state, sc, cd, Activation::Tanh);

        // tdshape1 — second channel
        let mut ts_in = [0.0f32; NOLACE_FRAME_SIZE];
        ts_in.copy_from_slice(&buf2[ch2..ch2 + NOLACE_FRAME_SIZE]);
        adashape_process_frame(&mut state.tdshape1_state,
            &mut buf2[ch2..ch2 + NOLACE_FRAME_SIZE], &ts_in,
            &post_out[..cd], &model.layers.tdshape1_alpha1_f, &model.layers.tdshape1_alpha1_t, &model.layers.tdshape1_alpha2,
            cd, NOLACE_FRAME_SIZE, 4, 1);

        // af2 (2 → 2)
        let mut af2_buf = [0.0f32; NOLACE_FRAME_SIZE * 2];
        adaconv_process_frame(&mut state.af2_state, &mut af2_buf,
            &buf2[si2..si2 + NOLACE_FRAME_SIZE * 2],
            &post_out[..cd], &model.layers.af2_kernel, &model.layers.af2_gain,
            cd, NOLACE_FRAME_SIZE, NOLACE_OVERLAP_SIZE, 2, 2, 16, 15,
            AF_FILTER_GAIN_A, AF_FILTER_GAIN_B, 1.0, &model.window);

        compute_generic_conv1d(&model.layers.post_af2, &mut post_out[..cd], &mut state.post_af2_state, sc, cd, Activation::Tanh);

        // tdshape2 — second channel
        ts_in.copy_from_slice(&af2_buf[NOLACE_FRAME_SIZE..NOLACE_FRAME_SIZE * 2]);
        adashape_process_frame(&mut state.tdshape2_state,
            &mut af2_buf[NOLACE_FRAME_SIZE..NOLACE_FRAME_SIZE * 2], &ts_in,
            &post_out[..cd], &model.layers.tdshape2_alpha1_f, &model.layers.tdshape2_alpha1_t, &model.layers.tdshape2_alpha2,
            cd, NOLACE_FRAME_SIZE, 4, 1);

        // af3 (2 → 2)
        let mut af3_buf = [0.0f32; NOLACE_FRAME_SIZE * 2];
        adaconv_process_frame(&mut state.af3_state, &mut af3_buf,
            &af2_buf,
            &post_out[..cd], &model.layers.af3_kernel, &model.layers.af3_gain,
            cd, NOLACE_FRAME_SIZE, NOLACE_OVERLAP_SIZE, 2, 2, 16, 15,
            AF_FILTER_GAIN_A, AF_FILTER_GAIN_B, 1.0, &model.window);

        compute_generic_conv1d(&model.layers.post_af3, &mut post_out[..cd], &mut state.post_af3_state, sc, cd, Activation::Tanh);

        // tdshape3 — second channel
        ts_in.copy_from_slice(&af3_buf[NOLACE_FRAME_SIZE..NOLACE_FRAME_SIZE * 2]);
        adashape_process_frame(&mut state.tdshape3_state,
            &mut af3_buf[NOLACE_FRAME_SIZE..NOLACE_FRAME_SIZE * 2], &ts_in,
            &post_out[..cd], &model.layers.tdshape3_alpha1_f, &model.layers.tdshape3_alpha1_t, &model.layers.tdshape3_alpha2,
            cd, NOLACE_FRAME_SIZE, 4, 1);

        // af4 (2 → 1)
        adaconv_process_frame(&mut state.af4_state, &mut x_out[si..ei],
            &af3_buf,
            &post_out[..cd], &model.layers.af4_kernel, &model.layers.af4_gain,
            cd, NOLACE_FRAME_SIZE, NOLACE_OVERLAP_SIZE, 2, 1, 16, 15,
            AF_FILTER_GAIN_A, AF_FILTER_GAIN_B, 1.0, &model.window);
    }

    apply_deemphasis(&mut x_out[..total], &mut state.deemph_mem);
}
