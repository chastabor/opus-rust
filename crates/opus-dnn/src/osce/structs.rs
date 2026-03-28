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

/// OSCE model container. Matches C `OSCEModel`.
pub struct OsceModel {
    pub loaded: bool,
    pub method: i32,
    pub lace: Option<super::lace::Lace>,
    pub nolace: Option<super::nolace::NoLace>,
}

impl Default for OsceModel {
    fn default() -> Self {
        OsceModel {
            loaded: false,
            method: OSCE_METHOD_NONE,
            lace: None,
            nolace: None,
        }
    }
}
