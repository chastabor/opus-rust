use crate::tables::*;

/// Pulse cache data.
pub struct PulseCache {
    pub size: usize,
    pub index: &'static [i16],
    pub bits: &'static [u8],
    pub caps: &'static [u8],
}

/// CELT mode definition (static, 48kHz/960 only).
pub struct CeltMode {
    pub fs: i32,
    pub overlap: usize,
    pub nb_ebands: usize,
    pub eff_ebands: usize,
    pub preemph: [f32; 4],
    pub ebands: &'static [i16],
    pub max_lm: usize,
    pub nb_short_mdcts: usize,
    pub short_mdct_size: usize,
    pub nb_alloc_vectors: usize,
    pub alloc_vectors: &'static [[u8; NB_EBANDS]; NB_ALLOC_VECTORS],
    pub log_n: &'static [i16],
    pub window: &'static [f32],
    pub cache: PulseCache,
}

/// Static 48kHz/960 mode.
pub static MODE_48000_960: CeltMode = CeltMode {
    fs: 48000,
    overlap: 120,
    nb_ebands: NB_EBANDS,
    eff_ebands: EFF_EBANDS,
    preemph: PREEMPH,
    ebands: &EBANDS_48000_960,
    max_lm: MAX_LM,
    nb_short_mdcts: 1,
    short_mdct_size: SHORT_MDCT_SIZE,
    nb_alloc_vectors: NB_ALLOC_VECTORS,
    alloc_vectors: &BAND_ALLOCATION,
    log_n: &LOG_N_400,
    window: &WINDOW_120,
    cache: PulseCache {
        size: 392,
        index: &CACHE_INDEX_50,
        bits: &CACHE_BITS_50,
        caps: &CACHE_CAPS_50,
    },
};

impl CeltMode {
    /// Get the static 48kHz/960 mode.
    pub fn get_mode() -> &'static CeltMode {
        &MODE_48000_960
    }
}
