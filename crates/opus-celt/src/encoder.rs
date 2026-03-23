use crate::mode::CeltMode;
use crate::tables::*;
use crate::mdct::{MdctLookup, clt_mdct_forward};
use crate::rate::{init_caps, clt_compute_allocation_enc};
use crate::quant_energy::{quant_coarse_energy, quant_fine_energy, quant_energy_finalise_enc};
use crate::bands::{compute_band_energies, normalise_bands, amp2_log2, quant_all_bands_enc};
use opus_range_coder::EcCtx;

/// CELT encoder state.
pub struct CeltEncoder {
    pub channels: usize,
    pub stream_channels: usize,
    pub upsample: usize,
    pub start: usize,
    pub end: usize,
    pub signalling: bool,
    pub disable_inv: bool,
    pub force_intra: bool,
    pub clip: bool,
    pub disable_pf: bool,
    pub complexity: i32,
    pub bitrate: i32,
    pub vbr: bool,
    pub constrained_vbr: bool,
    pub loss_rate: i32,
    pub lsb_depth: i32,
    pub lfe: bool,
    // State (gets reset)
    pub rng: u32,
    pub spread_decision: i32,
    pub delayed_intra: f32,
    pub tonal_average: i32,
    pub last_coded_bands: i32,
    pub hf_average: i32,
    pub tapset_decision: i32,
    pub prefilter_period: usize,
    pub prefilter_gain: f32,
    pub prefilter_tapset: usize,
    pub consec_transient: i32,
    pub preemph_mem_e: [f32; 2],
    pub preemph_mem_d: [f32; 2],
    // VBR state
    pub vbr_reservoir: i32,
    pub vbr_drift: i32,
    pub vbr_offset: i32,
    pub vbr_count: i32,
    pub overlap_max: f32,
    pub stereo_saving: f32,
    pub intensity: i32,
    pub spec_avg: f32,
    // Memory buffers
    pub in_mem: Vec<f32>,
    pub prefilter_mem: Vec<f32>,
    pub old_band_e: Vec<f32>,
    pub old_log_e: Vec<f32>,
    pub old_log_e2: Vec<f32>,
    pub energy_error: Vec<f32>,
    // MDCT lookup
    pub mdct: MdctLookup,
}

/// Pre-emphasis filter (float path).
/// Matches C `celt_preemphasis()` for the float case.
///
/// For the common 48 kHz path (`coef[1] == 0`, `upsample == 1`):
///   `out[i] = coef[2] * pcm[i * cc] - mem;  mem = coef[0] * coef[2] * pcm[i * cc]`
/// but since `coef[2] == 1.0` for 48 kHz this simplifies to:
///   `out[i] = pcm[i * cc] - mem;  mem = coef[0] * pcm[i * cc]`
/// If `clip`, clamp the signal value to `[-65536, 65536]` before filtering.
pub fn celt_preemphasis(
    pcm: &[f32],
    out: &mut [f32],
    n: usize,
    cc: usize,
    upsample: usize,
    coef: &[f32; 4],
    mem: &mut f32,
    clip: bool,
) {
    let coef0 = coef[0];
    let mut m = *mem;

    // Fast path: common 48 kHz case (coef[1]==0, upsample==1, no clip)
    if coef[1] == 0.0 && upsample == 1 && !clip {
        for i in 0..n {
            // RES2SIG for float is just *1.0 (identity)
            let x = pcm[cc * i];
            out[i] = x - m;
            m = coef0 * x;
        }
        *mem = m;
        return;
    }

    let nu = n / upsample;
    if upsample != 1 {
        for i in 0..n {
            out[i] = 0.0;
        }
    }
    for i in 0..nu {
        out[i * upsample] = pcm[cc * i];
    }

    if clip {
        for i in 0..nu {
            out[i * upsample] = out[i * upsample].clamp(-65536.0, 65536.0);
        }
    }

    // Apply pre-emphasis using coef[0] only (coef[1]==0 for standard modes)
    for i in 0..n {
        let x = out[i];
        out[i] = x - m;
        m = coef0 * x;
    }
    *mem = m;
}

/// Compute MDCTs for the encoder.
/// Matches C `compute_mdcts()` for the float case.
///
/// `input` is `CC * (B*N + overlap)` samples, where each channel occupies
/// a contiguous block of `B*N + overlap` samples.
/// `freq` is `CC * B * N` (or `C * B * N` after downmix) frequency-domain output.
fn compute_mdcts(
    mode: &CeltMode,
    short_blocks: i32,
    input: &[f32],
    freq: &mut [f32],
    c: usize,
    cc: usize,
    lm: i32,
    upsample: usize,
    mdct: &MdctLookup,
) {
    let overlap = mode.overlap;
    let (b, nb, shift) = if short_blocks != 0 {
        let m = 1usize << lm;
        (m, mode.short_mdct_size, mode.max_lm)
    } else {
        (1usize, mode.short_mdct_size << lm as usize, mode.max_lm - lm as usize)
    };
    let n = b * nb;

    for ch in 0..cc {
        for blk in 0..b {
            // Input: channel ch starts at ch * (n + overlap), block starts at blk * nb
            let in_start = ch * (n + overlap) + blk * nb;
            // Output: interleaved with stride b, starting at blk + ch * n
            clt_mdct_forward(
                mdct,
                &input[in_start..],
                &mut freq[blk + ch * n..],
                mode.window,
                overlap,
                shift,
                b,
            );
        }
    }

    // If stereo input but mono stream, downmix
    if cc == 2 && c == 1 {
        for i in 0..(b * nb) {
            freq[i] = 0.5 * freq[i] + 0.5 * freq[b * nb + i];
        }
    }

    // Handle upsampling: scale and zero-fill
    if upsample != 1 {
        for ch in 0..c {
            let bound = b * nb / upsample;
            let base = ch * b * nb;
            for i in 0..bound {
                freq[base + i] *= upsample as f32;
            }
            for i in bound..(b * nb) {
                freq[base + i] = 0.0;
            }
        }
    }
}

/// Encode TF (time-frequency) resolution decisions.
/// Mirror of the decoder's `tf_decode`, writing bits instead of reading.
fn tf_encode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    tf_select: i32,
    enc: &mut EcCtx,
) {
    let budget = enc.storage as i32 * 8;
    let mut tell = enc.tell();
    let mut logp = if is_transient { 2u32 } else { 4u32 };
    let tf_select_rsv = lm > 0 && tell + logp as i32 + 1 <= budget;
    let budget = budget - if tf_select_rsv { 1 } else { 0 };
    let mut curr = 0i32;
    let mut tf_changed = 0i32;

    for i in start..end {
        if tell + logp as i32 <= budget {
            enc.enc_bit_logp(tf_res[i] ^ curr != 0, logp);
            tell = enc.tell();
            curr = tf_res[i];
            tf_changed |= curr;
        } else {
            tf_res[i] = curr;
        }
        logp = if is_transient { 4 } else { 5 };
    }

    // Only code tf_select if it would actually make a difference
    let tf_select = if tf_select_rsv
        && TF_SELECT_TABLE[lm as usize][(4 * is_transient as usize) + tf_changed as usize]
            != TF_SELECT_TABLE[lm as usize][(4 * is_transient as usize) + 2 + tf_changed as usize]
    {
        enc.enc_bit_logp(tf_select != 0, 1);
        tf_select
    } else {
        0
    };

    for i in start..end {
        tf_res[i] = TF_SELECT_TABLE[lm as usize]
            [4 * is_transient as usize + 2 * tf_select as usize + tf_res[i] as usize]
            as i32;
    }
}

/// Simplified spreading decision.
/// For complexity < 3 or short blocks, return SPREAD_NORMAL.
/// Otherwise, analyze spectral shape (simplified version).
#[allow(clippy::too_many_arguments)]
fn spreading_decision(
    _m: &CeltMode,
    _x: &[f32],
    _tonal_average: &mut i32,
    _spread_decision: i32,
    _hf_average: &mut i32,
    _tapset_decision: &mut i32,
    _pf_on: bool,
    _eff_end: usize,
    _c: usize,
    _mm: usize,
) -> i32 {
    // Simplified: always return SPREAD_NORMAL.
    // A full implementation would analyze spectral shape.
    SPREAD_NORMAL
}

/// Simplified alloc trim analysis. Returns default trim = 5.
fn alloc_trim_analysis() -> i32 {
    5
}

/// Simplified dynamic allocation analysis.
/// Sets offsets to 0 and returns 0 for maxDepth.
fn dynalloc_analysis(offsets: &mut [i32], _start: usize, _end: usize) -> f32 {
    for v in offsets.iter_mut() {
        *v = 0;
    }
    0.0
}

/// Simplified stereo analysis. Returns 0 (not dual stereo).
fn stereo_analysis(_m: &CeltMode, _x: &[f32], _lm: i32, _n: usize) -> i32 {
    0
}

impl CeltEncoder {
    /// Create a new CELT encoder for the given sample rate and channel count.
    pub fn new(sample_rate: i32, channels: usize) -> Result<Self, i32> {
        if channels < 1 || channels > 2 {
            return Err(-1);
        }
        let mode = CeltMode::get_mode();
        let overlap = mode.overlap;
        let nb_ebands = mode.nb_ebands;

        let upsample = match sample_rate {
            48000 => 1,
            24000 => 2,
            16000 => 3,
            12000 => 4,
            8000 => 6,
            _ => return Err(-1),
        };

        // Allocate memory buffers.
        // in_mem: channels * overlap (holds the overlap portion from previous frame)
        let in_mem = vec![0.0f32; channels * overlap];

        // prefilter_mem: channels * COMBFILTER_MAXPERIOD
        let prefilter_mem = vec![0.0f32; channels * COMBFILTER_MAXPERIOD];

        // Band energy arrays: always 2 * nb_ebands for mono->stereo compatibility
        let old_band_e = vec![0.0f32; 2 * nb_ebands];
        let old_log_e = vec![-28.0f32; 2 * nb_ebands];
        let old_log_e2 = vec![-28.0f32; 2 * nb_ebands];
        let energy_error = vec![0.0f32; 2 * nb_ebands];

        // Create MDCT: N = 2 * shortMdctSize * nbShortMdcts
        let nb_short_mdcts = 1usize << mode.max_lm;
        let mdct_n = 2 * mode.short_mdct_size * nb_short_mdcts;
        let mdct = MdctLookup::new(mdct_n, mode.max_lm);

        let enc = CeltEncoder {
            channels,
            stream_channels: channels,
            upsample,
            start: 0,
            end: mode.eff_ebands,
            signalling: false,
            disable_inv: channels == 1,
            force_intra: false,
            clip: true,
            disable_pf: false,
            complexity: 5,
            bitrate: 510000, // OPUS_BITRATE_MAX equivalent
            vbr: false,
            constrained_vbr: true,
            loss_rate: 0,
            lsb_depth: 24,
            lfe: false,
            rng: 0,
            spread_decision: SPREAD_NORMAL,
            delayed_intra: 0.0,
            tonal_average: 256,
            last_coded_bands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            consec_transient: 0,
            preemph_mem_e: [0.0; 2],
            preemph_mem_d: [0.0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0.0,
            stereo_saving: 0.0,
            intensity: 0,
            spec_avg: 0.0,
            in_mem,
            prefilter_mem,
            old_band_e,
            old_log_e,
            old_log_e2,
            energy_error,
            mdct,
        };

        Ok(enc)
    }

    /// Encode a CELT frame.
    ///
    /// The main encoding pipeline:
    ///  1. Validate inputs, compute LM from frame_size
    ///  2. Initialize range encoder if ec is None
    ///  3. Pre-emphasis
    ///  4. Silence detection
    ///  5. Skip prefilter (encode pf_on=0)
    ///  6. Transient flag (simplified: always false)
    ///  7. Compute MDCTs
    ///  8. Compute band energies and normalize
    ///  9. Energy error bias
    /// 10. Coarse energy quantization
    /// 11. TF encode (all zeros)
    /// 12. Spread decision and encode
    /// 13. Dynamic allocation encode (simplified)
    /// 14. Alloc trim encode
    /// 15. Bit allocation (clt_compute_allocation_enc)
    /// 16. Fine energy quantization
    /// 17. Band quantization (quant_all_bands_enc)
    /// 18. Anti-collapse
    /// 19. Energy finalization
    /// 20. State update (oldBandE, oldLogE, oldLogE2, energyError)
    /// 21. enc_done() and return compressed size
    pub fn encode_with_ec(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        compressed: &mut [u8],
        nb_compressed_bytes: usize,
        ec: Option<&mut EcCtx>,
    ) -> Result<usize, i32> {
        let mode = CeltMode::get_mode();
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;
        let cc = self.channels;
        let c = self.stream_channels;

        if nb_compressed_bytes < 2 {
            return Err(-1);
        }

        let frame_size = frame_size * self.upsample;

        // Find LM from frame_size
        let mut lm = 0i32;
        while lm <= mode.max_lm as i32 {
            if mode.short_mdct_size << lm as usize == frame_size {
                break;
            }
            lm += 1;
        }
        if lm > mode.max_lm as i32 {
            return Err(-1);
        }
        let mm = 1usize << lm;
        let n = mm * mode.short_mdct_size;

        let mut nb_compressed_bytes = nb_compressed_bytes.min(1275);

        let start = self.start;
        let end = self.end;
        let mut eff_end = end;
        if eff_end > mode.eff_ebands {
            eff_end = mode.eff_ebands;
        }

        // -----------------------------------------------------------------
        // 2. Initialize range encoder
        // -----------------------------------------------------------------
        let mut local_enc;
        let mut tell;
        let nb_filled_bytes;

        if ec.is_none() {
            tell = 1;
            nb_filled_bytes = 0;
        } else {
            // If an external encoder was provided, compute initial tell
            let ext = ec.as_ref().unwrap();
            tell = ext.tell();
            nb_filled_bytes = ((tell + 4) >> 3) as usize;
        }
        let _ = tell; // used later after enc init

        // For CBR, compute the actual packet size from bitrate
        if !self.vbr || self.bitrate == 510000 {
            if self.bitrate != 510000 {
                let tmp = (self.bitrate as i64 * frame_size as i64) as i64;
                let new_bytes = 2i32.max(nb_compressed_bytes as i32).min(
                    ((tmp + 4 * mode.fs as i64) / (8 * mode.fs as i64)) as i32,
                );
                nb_compressed_bytes = new_bytes as usize;
            }
        }

        let nb_available_bytes = nb_compressed_bytes.saturating_sub(nb_filled_bytes);

        local_enc = EcCtx::enc_init(nb_compressed_bytes as u32);
        // Use either the provided encoder or our local one
        let enc_is_external = ec.is_some();
        let enc = if let Some(ext) = ec {
            ext
        } else {
            &mut local_enc
        };

        let total_bits = nb_compressed_bytes as i32 * 8;
        tell = enc.tell();

        // -----------------------------------------------------------------
        // 3. Pre-emphasis
        // -----------------------------------------------------------------
        // Allocate the working buffer: CC * (N + overlap)
        let buf_size = cc * (n + overlap);
        let mut inp = vec![0.0f32; buf_size];

        // Compute sample_max for silence detection
        let sample_max = {
            let mut smax = self.overlap_max;
            let non_overlap_len = cc * (n - overlap) / self.upsample;
            for i in 0..non_overlap_len.min(pcm.len()) {
                smax = smax.max(pcm[i].abs());
            }
            let overlap_start = non_overlap_len;
            let overlap_pcm_len = cc * overlap / self.upsample;
            let mut overlap_max = 0.0f32;
            for i in overlap_start..(overlap_start + overlap_pcm_len).min(pcm.len()) {
                overlap_max = overlap_max.max(pcm[i].abs());
            }
            self.overlap_max = overlap_max;
            smax.max(overlap_max)
        };

        // Apply pre-emphasis per channel
        for ch in 0..cc {
            let need_clip = self.clip && sample_max > 65536.0;
            let out_start = ch * (n + overlap) + overlap;
            celt_preemphasis(
                &pcm[ch..],
                &mut inp[out_start..out_start + n],
                n,
                cc,
                self.upsample,
                &mode.preemph,
                &mut self.preemph_mem_e[ch],
                need_clip,
            );
            // Copy overlap from prefilter memory (tail of previous frame)
            // prefilter_mem layout: ch * COMBFILTER_MAXPERIOD .. (ch+1)*COMBFILTER_MAXPERIOD
            let mem_start = (ch + 1) * COMBFILTER_MAXPERIOD - overlap;
            let mem_end = (ch + 1) * COMBFILTER_MAXPERIOD;
            let in_base = ch * (n + overlap);
            inp[in_base..in_base + overlap]
                .copy_from_slice(&self.prefilter_mem[mem_start..mem_end]);
        }

        // -----------------------------------------------------------------
        // 4. Silence detection
        // -----------------------------------------------------------------
        let silence = sample_max <= 1.0 / (1i32 << self.lsb_depth) as f32;
        if tell == 1 {
            enc.enc_bit_logp(silence, 15);
        }
        if silence {
            // Pretend we have filled all remaining bits with zeros
            let tell_now = enc.tell();
            enc.nbits_total += total_bits - tell_now;
        }

        // -----------------------------------------------------------------
        // 5. Prefilter (skip for now: encode pf_on=0)
        // -----------------------------------------------------------------
        let is_transient = false;
        let short_blocks = 0i32;

        // Encode prefilter off (if we have bits for it and start==0)
        tell = enc.tell();
        if start == 0 && tell + 16 <= total_bits {
            enc.enc_bit_logp(false, 1); // pf_on = 0
        }

        // -----------------------------------------------------------------
        // 6. Transient flag (simplified: always non-transient)
        // -----------------------------------------------------------------
        tell = enc.tell();
        if lm > 0 && tell + 3 <= total_bits {
            enc.enc_bit_logp(is_transient, 3);
        }

        // -----------------------------------------------------------------
        // 7. Compute MDCTs
        // -----------------------------------------------------------------
        let freq_size = c * n;
        let mut freq = vec![0.0f32; freq_size.max(1)];

        compute_mdcts(
            mode,
            short_blocks,
            &inp,
            &mut freq,
            c,
            cc,
            lm,
            self.upsample,
            &self.mdct,
        );

        // -----------------------------------------------------------------
        // 8. Compute band energies and normalize
        // -----------------------------------------------------------------
        let mut band_e = vec![0.0f32; c * nb_ebands];
        let mut band_log_e = vec![0.0f32; c * nb_ebands];

        compute_band_energies(mode, &freq, &mut band_e, eff_end, c, lm);

        if self.lfe {
            for i in 2..end {
                band_e[i] = band_e[i].min(1e-4 * band_e[0]).max(1e-15);
                if c == 2 {
                    band_e[nb_ebands + i] =
                        band_e[nb_ebands + i].min(1e-4 * band_e[nb_ebands]).max(1e-15);
                }
            }
        }

        amp2_log2(mode, eff_end, end, &band_e, &mut band_log_e, c);

        // Normalize bands (creates normalized MDCTs in x_norm)
        let mut x_norm = vec![0.0f32; c * n];
        normalise_bands(mode, &freq, &mut x_norm, &band_e, eff_end, c, mm);

        // -----------------------------------------------------------------
        // 9. Energy error bias
        // -----------------------------------------------------------------
        for ch in 0..c {
            for i in start..end {
                let idx = i + ch * nb_ebands;
                if (band_log_e[idx] - self.old_band_e[idx]).abs() < 2.0 {
                    band_log_e[idx] -= 0.25 * self.energy_error[idx];
                }
            }
        }

        // -----------------------------------------------------------------
        // 10. Coarse energy quantization
        // -----------------------------------------------------------------
        let mut error = vec![0.0f32; c * nb_ebands];
        quant_coarse_energy(
            mode,
            start,
            end,
            eff_end,
            &band_log_e,
            &mut self.old_band_e,
            total_bits,
            &mut error,
            enc,
            c,
            lm as usize,
            nb_available_bytes as i32,
            self.force_intra,
            &mut self.delayed_intra,
            self.complexity >= 4,
            self.loss_rate,
            self.lfe,
        );

        // -----------------------------------------------------------------
        // 11. TF encode (all zeros for simplified encoder)
        // -----------------------------------------------------------------
        let mut tf_res = vec![0i32; nb_ebands];
        for i in 0..nb_ebands {
            tf_res[i] = if is_transient { 1 } else { 0 };
        }
        let tf_select = 0i32;
        tf_encode(start, end, is_transient, &mut tf_res, lm, tf_select, enc);

        // -----------------------------------------------------------------
        // 12. Spread decision and encode
        // -----------------------------------------------------------------
        tell = enc.tell();
        if tell + 4 <= total_bits {
            if self.lfe {
                self.spread_decision = SPREAD_NORMAL;
            } else if short_blocks != 0 || self.complexity < 3 || nb_available_bytes < 10 * c {
                if self.complexity == 0 {
                    self.spread_decision = SPREAD_NONE;
                } else {
                    self.spread_decision = SPREAD_NORMAL;
                }
            } else {
                self.spread_decision = spreading_decision(
                    mode,
                    &x_norm,
                    &mut self.tonal_average,
                    self.spread_decision,
                    &mut self.hf_average,
                    &mut self.tapset_decision,
                    false, // pf_on
                    eff_end,
                    c,
                    mm,
                );
            }
            enc.enc_icdf(self.spread_decision as usize, &SPREAD_ICDF, 5);
        } else {
            self.spread_decision = SPREAD_NORMAL;
        }

        // -----------------------------------------------------------------
        // 13. Dynamic allocation
        // -----------------------------------------------------------------
        let mut cap = vec![0i32; nb_ebands];
        init_caps(mode, &mut cap, lm, c as i32);

        let mut offsets = vec![0i32; nb_ebands];
        let _max_depth = dynalloc_analysis(&mut offsets, start, end);

        // Encode dynamic allocation (all zeros for simplified encoder)
        let mut dynalloc_logp = 6i32;
        let total_bits_shifted = nb_compressed_bytes as i32 * 8 << BITRES;
        let mut total_boost = 0i32;
        tell = enc.tell_frac() as i32;
        for i in start..end {
            let width = c as i32 * (mode.ebands[i + 1] - mode.ebands[i]) as i32 * mm as i32;
            let quanta = (width << BITRES).min((6i32 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            let mut j = 0;
            while (tell + (dynalloc_loop_logp << BITRES)) < total_bits_shifted - total_boost
                && boost < cap[i]
            {
                let flag = j < offsets[i];
                enc.enc_bit_logp(flag, dynalloc_loop_logp as u32);
                tell = enc.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_boost += quanta;
                dynalloc_loop_logp = 1;
                j += 1;
            }
            if j > 0 {
                dynalloc_logp = 2i32.max(dynalloc_logp - 1);
            }
            offsets[i] = boost;
        }

        // -----------------------------------------------------------------
        // 14. Alloc trim encode
        // -----------------------------------------------------------------
        let alloc_trim;
        tell = enc.tell_frac() as i32;
        if tell + (6 << BITRES) <= total_bits_shifted - total_boost {
            alloc_trim = alloc_trim_analysis();
            enc.enc_icdf(alloc_trim as usize, &TRIM_ICDF, 7);
            let _ = enc.tell_frac();
        } else {
            alloc_trim = 5;
        }

        // -----------------------------------------------------------------
        // 15. Bit allocation
        // -----------------------------------------------------------------
        let mut fine_quant = vec![0i32; nb_ebands];
        let mut pulses = vec![0i32; nb_ebands];
        let mut fine_priority = vec![0i32; nb_ebands];
        let mut balance = 0i32;

        let mut bits =
            ((nb_compressed_bytes as i32 * 8) << BITRES) - enc.tell_frac() as i32 - 1;
        let anti_collapse_rsv =
            if is_transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
                1 << BITRES
            } else {
                0
            };
        bits -= anti_collapse_rsv;

        // For stereo, determine intensity and dual_stereo
        let mut enc_intensity = if c == 2 {
            self.intensity.min(end as i32).max(start as i32)
        } else {
            0
        };
        let mut dual_stereo = if c == 2 {
            stereo_analysis(mode, &x_norm, lm, n)
        } else {
            0
        };

        let coded_bands = clt_compute_allocation_enc(
            mode,
            start,
            end,
            &offsets,
            &cap,
            alloc_trim,
            &mut enc_intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            c as i32,
            lm,
            enc,
            self.last_coded_bands,
            end as i32 - 1, // signalBandwidth
        );

        // Update last_coded_bands
        if self.last_coded_bands != 0 {
            self.last_coded_bands = self
                .last_coded_bands
                .min(coded_bands as i32 + 1)
                .max((coded_bands as i32).saturating_sub(1));
        } else {
            self.last_coded_bands = coded_bands as i32;
        }
        self.intensity = enc_intensity;

        // -----------------------------------------------------------------
        // 16. Fine energy quantization
        // -----------------------------------------------------------------
        quant_fine_energy(
            mode,
            start,
            end,
            &mut self.old_band_e,
            &mut error,
            &fine_quant,
            enc,
            c,
        );

        // Clear energy error (will be repopulated at the end)
        for v in self.energy_error.iter_mut() {
            *v = 0.0;
        }

        // -----------------------------------------------------------------
        // 17. Band quantization (quant_all_bands_enc)
        // -----------------------------------------------------------------
        let mut collapse_masks = vec![0u8; c * nb_ebands];

        // Split x_norm into X and Y for stereo (safe disjoint borrows via split_at_mut)
        let (x_ref, y_ref): (&mut [f32], Option<&mut [f32]>) = if c == 2 {
            let (x_part, y_part) = x_norm.split_at_mut(n);
            (x_part, Some(y_part))
        } else {
            (&mut x_norm[..], None)
        };

        quant_all_bands_enc(
            mode,
            start,
            end,
            x_ref,
            y_ref,
            &mut collapse_masks,
            &band_e,
            &mut pulses,
            short_blocks,
            self.spread_decision,
            dual_stereo,
            enc_intensity,
            &tf_res,
            nb_compressed_bytes as i32 * (8 << BITRES) - anti_collapse_rsv,
            balance,
            enc,
            lm,
            coded_bands,
            &mut self.rng,
            self.disable_inv,
        );

        // -----------------------------------------------------------------
        // 18. Anti-collapse
        // -----------------------------------------------------------------
        let _anti_collapse_on = if anti_collapse_rsv > 0 {
            let on = self.consec_transient < 2;
            enc.enc_bits(if on { 1 } else { 0 }, 1);
            on
        } else {
            false
        };

        // -----------------------------------------------------------------
        // 19. Energy finalization
        // -----------------------------------------------------------------
        quant_energy_finalise_enc(
            mode,
            start,
            end,
            Some(&mut self.old_band_e),
            &mut error,
            &fine_quant,
            &fine_priority,
            nb_compressed_bytes as i32 * 8 - enc.tell(),
            enc,
            c,
        );

        // -----------------------------------------------------------------
        // 20. State update
        // -----------------------------------------------------------------
        // Update energy error (clamped to [-0.5, 0.5])
        for ch in 0..c {
            for i in start..end {
                self.energy_error[i + ch * nb_ebands] =
                    error[i + ch * nb_ebands].clamp(-0.5, 0.5);
            }
        }

        if silence {
            for i in 0..(c * nb_ebands) {
                self.old_band_e[i] = -28.0;
            }
        }

        // If mono stream but stereo channels, copy
        if cc == 2 && c == 1 {
            self.old_band_e.copy_within(0..nb_ebands, nb_ebands);
        }

        if !is_transient {
            self.old_log_e2[..cc * nb_ebands]
                .copy_from_slice(&self.old_log_e[..cc * nb_ebands]);
            self.old_log_e[..cc * nb_ebands]
                .copy_from_slice(&self.old_band_e[..cc * nb_ebands]);
        } else {
            for i in 0..(cc * nb_ebands) {
                self.old_log_e[i] = self.old_log_e[i].min(self.old_band_e[i]);
            }
        }

        // Clear out-of-range bands
        for ch in 0..cc {
            for i in 0..start {
                self.old_band_e[ch * nb_ebands + i] = 0.0;
                self.old_log_e[ch * nb_ebands + i] = -28.0;
                self.old_log_e2[ch * nb_ebands + i] = -28.0;
            }
            for i in end..nb_ebands {
                self.old_band_e[ch * nb_ebands + i] = 0.0;
                self.old_log_e[ch * nb_ebands + i] = -28.0;
                self.old_log_e2[ch * nb_ebands + i] = -28.0;
            }
        }

        if is_transient {
            self.consec_transient += 1;
        } else {
            self.consec_transient = 0;
        }

        self.rng = enc.rng;

        // Update prefilter memory: shift and copy the new frame tail
        for ch in 0..cc {
            let mem_base = ch * COMBFILTER_MAXPERIOD;
            let in_base = ch * (n + overlap);
            // Shift prefilter memory: keep the tail, prepend new data
            // The prefilter_mem stores the last COMBFILTER_MAXPERIOD samples per channel.
            // After encoding, the new samples are in inp[in_base + overlap .. in_base + overlap + n]
            // We want to keep the last COMBFILTER_MAXPERIOD of those.
            if n >= COMBFILTER_MAXPERIOD {
                // Copy the last COMBFILTER_MAXPERIOD samples from the input buffer
                let src_start = in_base + overlap + n - COMBFILTER_MAXPERIOD;
                self.prefilter_mem[mem_base..mem_base + COMBFILTER_MAXPERIOD]
                    .copy_from_slice(&inp[src_start..src_start + COMBFILTER_MAXPERIOD]);
            } else {
                // Shift existing memory left by n, append new n samples
                let keep = COMBFILTER_MAXPERIOD - n;
                self.prefilter_mem
                    .copy_within(mem_base + n..mem_base + COMBFILTER_MAXPERIOD, mem_base);
                let src_start = in_base + overlap;
                self.prefilter_mem[mem_base + keep..mem_base + COMBFILTER_MAXPERIOD]
                    .copy_from_slice(&inp[src_start..src_start + n]);
            }
        }

        // -----------------------------------------------------------------
        // 21. Finalize and return
        // -----------------------------------------------------------------
        enc.enc_done();

        if enc.get_error() {
            return Err(-3); // OPUS_INTERNAL_ERROR
        }

        // Copy the encoded data to the output buffer
        if !enc_is_external {
            let encoded_len = nb_compressed_bytes.min(compressed.len());
            compressed[..encoded_len]
                .copy_from_slice(&enc.buf[..encoded_len]);
        }

        Ok(nb_compressed_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::CeltDecoder;

    #[test]
    fn test_encoder_create() {
        let enc = CeltEncoder::new(48000, 1);
        assert!(enc.is_ok());
        let enc = enc.unwrap();
        assert_eq!(enc.channels, 1);
        assert_eq!(enc.upsample, 1);

        let enc = CeltEncoder::new(48000, 2);
        assert!(enc.is_ok());

        let enc = CeltEncoder::new(48000, 3);
        assert!(enc.is_err());
    }

    #[test]
    fn test_encode_silence() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        let pcm = vec![0.0f32; 960]; // 20ms at 48kHz
        let mut compressed = vec![0u8; 128];
        let result = enc.encode_with_ec(&pcm, 960, &mut compressed, 128, None);
        assert!(result.is_ok());
        let nbytes = result.unwrap();
        assert!(nbytes > 0);
        assert!(nbytes <= 128);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Encode a sine wave with the CELT encoder, then decode it
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        let mut dec = CeltDecoder::new(48000, 1).unwrap();

        // Generate a 440 Hz sine wave
        let n = 960; // 20ms frame
        let mut pcm = vec![0.0f32; n];
        for i in 0..n {
            pcm[i] = 10000.0 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin();
        }

        // Encode
        let mut compressed = vec![0u8; 128];
        let nbytes = enc
            .encode_with_ec(&pcm, n, &mut compressed, 128, None)
            .unwrap();
        assert!(nbytes > 2, "Encoded packet should be more than 2 bytes");

        // Decode
        let mut decoded = vec![0.0f32; n];
        let result = dec.decode_with_ec(&compressed[..nbytes], &mut decoded, n, None);
        assert!(result.is_ok(), "Decoder should accept encoder output");

        // Verify the decoded signal has some energy (not silence)
        let energy: f32 = decoded.iter().map(|x| x * x).sum();
        assert!(energy > 0.0, "Decoded signal should have non-zero energy");
    }

    #[test]
    fn test_encode_multiple_frames() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        let mut dec = CeltDecoder::new(48000, 1).unwrap();

        for frame in 0..5 {
            let n = 960;
            let mut pcm = vec![0.0f32; n];
            let freq = 440.0 + frame as f32 * 100.0;
            for i in 0..n {
                pcm[i] = 5000.0
                    * (2.0 * std::f32::consts::PI * freq * (frame * n + i) as f32 / 48000.0)
                        .sin();
            }

            let mut compressed = vec![0u8; 128];
            let nbytes = enc
                .encode_with_ec(&pcm, n, &mut compressed, 128, None)
                .unwrap();

            let mut decoded = vec![0.0f32; n];
            let result = dec.decode_with_ec(&compressed[..nbytes], &mut decoded, n, None);
            assert!(
                result.is_ok(),
                "Frame {frame}: decoder should accept encoder output"
            );
        }
    }
}
