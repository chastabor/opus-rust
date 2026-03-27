// OSCE configuration constants from osce_config.h.

pub const OSCE_FEATURES_MAX_HISTORY: usize = 350;
pub const OSCE_FEATURE_DIM: usize = 93;
pub const OSCE_MAX_FEATURE_FRAMES: usize = 4;

pub const OSCE_CLEAN_SPEC_NUM_BANDS: usize = 64;
pub const OSCE_NOISY_SPEC_NUM_BANDS: usize = 18;

pub const OSCE_NO_PITCH_VALUE: usize = 7;
pub const OSCE_PREEMPH: f32 = 0.85;
pub const OSCE_PITCH_HANGOVER: usize = 0;

pub const OSCE_CLEAN_SPEC_START: usize = 0;
pub const OSCE_CLEAN_SPEC_LENGTH: usize = 64;
pub const OSCE_NOISY_CEPSTRUM_START: usize = 64;
pub const OSCE_NOISY_CEPSTRUM_LENGTH: usize = 18;
pub const OSCE_ACORR_START: usize = 82;
pub const OSCE_ACORR_LENGTH: usize = 5;
pub const OSCE_LTP_START: usize = 87;
pub const OSCE_LTP_LENGTH: usize = 5;
pub const OSCE_LOG_GAIN_START: usize = 92;
pub const OSCE_LOG_GAIN_LENGTH: usize = 1;

pub const OSCE_BWE_MAX_INSTAFREQ_BIN: usize = 40;
pub const OSCE_BWE_HALF_WINDOW_SIZE: usize = 160;
pub const OSCE_BWE_WINDOW_SIZE: usize = 2 * OSCE_BWE_HALF_WINDOW_SIZE;
pub const OSCE_BWE_NUM_BANDS: usize = 32;
pub const OSCE_BWE_FEATURE_DIM: usize = 114;
pub const OSCE_BWE_OUTPUT_DELAY: usize = 21;

/// OSCE operating modes.
pub const OSCE_MODE_SILK_ONLY: i32 = 1000;
pub const OSCE_MODE_HYBRID: i32 = 1001;
pub const OSCE_MODE_CELT_ONLY: i32 = 1002;
pub const OSCE_MODE_SILK_BBWE: i32 = 1003;

/// OSCE method selection.
pub const OSCE_METHOD_NONE: i32 = 0;
pub const OSCE_METHOD_LACE: i32 = 1;
pub const OSCE_METHOD_NOLACE: i32 = 2;
