use crate::nndsp::{AdaConvState, AdaCombState, AdaShapeState};
use super::config::*;

/// OSCE feature extraction state. Matches C `OSCEFeatureState`.
pub struct OsceFeatureState {
    pub numbits_smooth: f32,
    pub pitch_hangover_count: i32,
    pub last_lag: i32,
    pub last_type: i32,
    pub signal_history: [f32; OSCE_FEATURES_MAX_HISTORY],
    pub reset: bool,
}

impl Default for OsceFeatureState {
    fn default() -> Self {
        OsceFeatureState {
            numbits_smooth: 0.0,
            pitch_hangover_count: 0,
            last_lag: 0,
            last_type: 0,
            signal_history: [0.0; OSCE_FEATURES_MAX_HISTORY],
            reset: true,
        }
    }
}

/// BWE feature state. Matches C `OSCEBWEFeatureState`.
pub struct OsceBweFeatureState {
    pub signal_history: [f32; OSCE_BWE_HALF_WINDOW_SIZE],
    pub last_spec: [f32; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2],
}

impl Default for OsceBweFeatureState {
    fn default() -> Self {
        OsceBweFeatureState {
            signal_history: [0.0; OSCE_BWE_HALF_WINDOW_SIZE],
            last_spec: [0.0; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2],
        }
    }
}

/// LACE processing state. Matches C `LACEState`.
pub struct LaceState {
    pub feature_net_conv2_state: Vec<f32>,
    pub feature_net_gru_state: Vec<f32>,
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl LaceState {
    pub fn new(conv2_state_size: usize, cond_dim: usize) -> Self {
        LaceState {
            feature_net_conv2_state: vec![0.0; conv2_state_size],
            feature_net_gru_state: vec![0.0; cond_dim],
            cf1_state: AdaCombState::default(),
            cf2_state: AdaCombState::default(),
            af1_state: AdaConvState::default(),
            preemph_mem: 0.0,
            deemph_mem: 0.0,
        }
    }
}

/// NoLACE processing state. Matches C `NoLACEState`.
pub struct NoLaceState {
    pub feature_net_conv2_state: Vec<f32>,
    pub feature_net_gru_state: Vec<f32>,
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

impl NoLaceState {
    pub fn new(conv2_state_size: usize, cond_dim: usize) -> Self {
        NoLaceState {
            feature_net_conv2_state: vec![0.0; conv2_state_size],
            feature_net_gru_state: vec![0.0; cond_dim],
            post_cf1_state: vec![0.0; cond_dim],
            post_cf2_state: vec![0.0; cond_dim],
            post_af1_state: vec![0.0; cond_dim],
            post_af2_state: vec![0.0; cond_dim],
            post_af3_state: vec![0.0; cond_dim],
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
}

/// OSCE model container. Matches C `OSCEModel`.
/// Model layer definitions (LACE/NoLACE/BBWENet layers) are loaded from weight data.
/// The actual layer structs are auto-generated — here we store them as opaque loaded state.
pub struct OsceModel {
    pub loaded: bool,
    pub method: i32,
}

impl Default for OsceModel {
    fn default() -> Self {
        OsceModel {
            loaded: false,
            method: OSCE_METHOD_NONE,
        }
    }
}
