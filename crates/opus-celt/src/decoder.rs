use crate::bands::{anti_collapse, denormalise_bands, quant_all_bands};
use crate::mdct::{MdctLookup, clt_mdct_backward};
use crate::mode::CeltMode;
use crate::pitch::comb_filter_inplace;
use crate::quant_energy::{unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy};
use crate::rate::{clt_compute_allocation, init_caps};
use crate::tables::*;
use opus_range_coder::EcCtx;

const DECODE_BUFFER_SIZE: usize = DEC_PITCH_BUF_SIZE;

/// CELT decoder state.
pub struct CeltDecoder {
    pub channels: usize,
    pub stream_channels: usize,
    pub downsample: usize,
    pub start: usize,
    pub end: usize,
    pub signalling: bool,
    pub disable_inv: bool,

    // State that gets reset
    pub rng: u32,
    pub error: bool,
    pub last_pitch_index: usize,
    pub loss_duration: usize,
    pub last_frame_type: i32,
    pub skip_plc: bool,
    pub postfilter_period: usize,
    pub postfilter_period_old: usize,
    pub postfilter_gain: f32,
    pub postfilter_gain_old: f32,
    pub postfilter_tapset: usize,
    pub postfilter_tapset_old: usize,
    pub prefilter_and_fold: bool,

    pub preemph_mem: [f32; 2],

    // Decode memory buffer (channels * (DECODE_BUFFER_SIZE + overlap))
    pub decode_mem: Vec<f32>,
    // Band energies: 2 * nb_ebands
    pub old_band_e: Vec<f32>,
    pub old_log_e: Vec<f32>,
    pub old_log_e2: Vec<f32>,
    pub background_log_e: Vec<f32>,
    // LPC coefficients: channels * CELT_LPC_ORDER
    pub lpc: Vec<f32>,

    // MDCT lookup
    pub mdct: MdctLookup,
}

impl CeltDecoder {
    /// Create a new CELT decoder for the given sample rate and channel count.
    pub fn new(sample_rate: i32, channels: usize) -> Result<Self, i32> {
        if !(1..=2).contains(&channels) {
            return Err(-1);
        }
        let mode = CeltMode::get_mode();
        let overlap = mode.overlap;
        let nb_ebands = mode.nb_ebands;

        let downsample = match sample_rate {
            48000 => 1,
            24000 => 2,
            16000 => 3,
            12000 => 4,
            8000 => 6,
            _ => return Err(-1),
        };

        let mem_size = channels * (DECODE_BUFFER_SIZE + overlap);
        let decode_mem = vec![0.0f32; mem_size];
        let old_band_e = vec![0.0f32; 2 * nb_ebands];
        let old_log_e = vec![-28.0f32; 2 * nb_ebands];
        let old_log_e2 = vec![-28.0f32; 2 * nb_ebands];
        let background_log_e = vec![-28.0f32; 2 * nb_ebands];
        let lpc = vec![0.0f32; channels * CELT_LPC_ORDER];

        // Create MDCT: N = 2 * shortMdctSize * nbShortMdcts
        // For 48kHz: 2 * 120 * 8 = 1920
        let nb_short_mdcts = 1usize << mode.max_lm;
        let mdct_n = 2 * mode.short_mdct_size * nb_short_mdcts;
        let mdct = MdctLookup::new(mdct_n, mode.max_lm);

        let dec = CeltDecoder {
            channels,
            stream_channels: channels,
            downsample,
            start: 0,
            end: mode.eff_ebands,
            signalling: true,
            disable_inv: channels == 1,
            rng: 0,
            error: false,
            last_pitch_index: 0,
            loss_duration: 0,
            last_frame_type: FRAME_NONE,
            skip_plc: false,
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0.0,
            postfilter_gain_old: 0.0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            prefilter_and_fold: false,
            preemph_mem: [0.0; 2],
            decode_mem,
            old_band_e,
            old_log_e,
            old_log_e2,
            background_log_e,
            lpc,
            mdct,
        };

        Ok(dec)
    }

    /// Decode a CELT frame.
    pub fn decode_with_ec(
        &mut self,
        data: &[u8],
        pcm: &mut [f32],
        frame_size: usize,
        ec: Option<&mut EcCtx>,
    ) -> Result<(), i32> {
        let mode = CeltMode::get_mode();
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;
        let cc = self.channels;
        let frame_size = frame_size * self.downsample;

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

        let len = data.len();
        if len > 1275 {
            return Err(-1);
        }

        let start = self.start;
        let end = self.end;
        let mut eff_end = end;
        if eff_end > mode.eff_ebands {
            eff_end = mode.eff_ebands;
        }

        let c = self.stream_channels;

        // Packet loss concealment: if no data, output silence/noise
        if data.is_empty() || len <= 1 {
            self.decode_lost(n, lm, cc);
            self.deemphasis(pcm, n, cc);
            return Ok(());
        }

        if self.loss_duration == 0 {
            self.skip_plc = false;
        }

        // Initialize range decoder
        let mut local_dec = EcCtx::dec_init(data);
        let dec = ec.unwrap_or(&mut local_dec);

        // If mono, merge stereo band energies
        if c == 1 {
            for i in 0..nb_ebands {
                self.old_band_e[i] = self.old_band_e[i].max(self.old_band_e[nb_ebands + i]);
            }
        }

        let total_bits = len as i32 * 8;
        let mut tell = dec.tell();

        // Silence flag
        let silence = if tell >= total_bits {
            true
        } else if tell == 1 {
            dec.dec_bit_logp(15)
        } else {
            false
        };

        if silence {
            let tell_now = dec.tell();
            dec.nbits_total += len as i32 * 8 - tell_now;
        }

        // Postfilter
        let mut postfilter_gain = 0.0f32;
        let mut postfilter_pitch = 0usize;
        let mut postfilter_tapset = 0usize;
        tell = dec.tell();
        if start == 0 && tell + 16 <= total_bits {
            if dec.dec_bit_logp(1) {
                let octave = dec.dec_uint(6) as usize;
                postfilter_pitch = (16 << octave) + dec.dec_bits((4 + octave) as u32) as usize - 1;
                let qg = dec.dec_bits(3);
                if dec.tell() + 2 <= total_bits {
                    postfilter_tapset = dec.dec_icdf(&TAPSET_ICDF, 2);
                }
                postfilter_gain = 0.09375 * (qg as f32 + 1.0);
            }
            tell = dec.tell();
        }

        // Transient flag
        let is_transient = if lm > 0 && tell + 3 <= total_bits {
            dec.dec_bit_logp(3)
        } else {
            false
        };
        let short_blocks = if is_transient { mm as i32 } else { 0 };

        // Intra energy
        tell = dec.tell();
        let intra_ener = if tell + 3 <= total_bits {
            dec.dec_bit_logp(3)
        } else {
            false
        };

        // Coarse energy decode
        unquant_coarse_energy(
            mode,
            start,
            end,
            &mut self.old_band_e,
            intra_ener,
            dec,
            c,
            lm as usize,
        );

        // TF decode
        let mut tf_res = vec![0i32; nb_ebands];
        tf_decode(start, end, is_transient, &mut tf_res, lm, dec);

        // Spread decision
        tell = dec.tell();
        let spread_decision = if tell + 4 <= total_bits {
            dec.dec_icdf(&SPREAD_ICDF, 5) as i32
        } else {
            SPREAD_NORMAL
        };

        // Init caps
        let mut cap = vec![0i32; nb_ebands];
        init_caps(mode, &mut cap, lm, c as i32);

        // Dynamic allocation
        let mut offsets = vec![0i32; nb_ebands];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_shifted = (len as i32 * 8) << BITRES;
        tell = dec.tell_frac() as i32;
        for i in start..end {
            let width = c as i32 * (mode.ebands[i + 1] - mode.ebands[i]) as i32 * mm as i32;
            let quanta = (width << BITRES).min((6i32 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while (tell + (dynalloc_loop_logp << BITRES)) < total_bits_shifted && boost < cap[i] {
                let flag = dec.dec_bit_logp(dynalloc_loop_logp as u32);
                tell = dec.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits_shifted -= quanta;
                dynalloc_loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = 2i32.max(dynalloc_logp - 1);
            }
        }

        // Fine energy quantization
        let mut fine_quant = vec![0i32; nb_ebands];
        let alloc_trim = if (dec.tell_frac() as i32) + (6 << BITRES) <= total_bits_shifted {
            dec.dec_icdf(&TRIM_ICDF, 7) as i32
        } else {
            5
        };

        let mut bits = ((len as i32 * 8) << BITRES) - dec.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;

        let mut pulses_vec = vec![0i32; nb_ebands];
        let mut fine_priority = vec![0i32; nb_ebands];
        let mut balance = 0i32;
        let mut intensity = 0i32;
        let mut dual_stereo = 0i32;

        let coded_bands = clt_compute_allocation(
            mode,
            start,
            end,
            &offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses_vec,
            &mut fine_quant,
            &mut fine_priority,
            c as i32,
            lm,
            dec,
        );

        unquant_fine_energy(mode, start, end, &mut self.old_band_e, &fine_quant, dec, c);

        // Allocate normalized MDCTs
        let mut x_norm = vec![0.0f32; c * n];

        // Shift decode memory
        for ch in 0..cc {
            let base = ch * (DECODE_BUFFER_SIZE + overlap);
            self.decode_mem
                .copy_within((base + n)..(base + DECODE_BUFFER_SIZE + overlap), base);
        }

        // Decode fixed codebook
        let mut collapse_masks = vec![0u8; c * nb_ebands];

        // Split x_norm into X and Y for stereo (safe disjoint borrows via split_at_mut)
        let (x_ref, y_ref): (&mut [f32], Option<&mut [f32]>) = if c == 2 {
            let (x_part, y_part) = x_norm.split_at_mut(n);
            (x_part, Some(y_part))
        } else {
            (&mut x_norm[..], None)
        };

        quant_all_bands(
            mode,
            start,
            end,
            x_ref,
            y_ref,
            &mut collapse_masks,
            &mut pulses_vec,
            short_blocks,
            spread_decision,
            dual_stereo,
            intensity,
            &tf_res,
            len as i32 * (8 << BITRES) - anti_collapse_rsv,
            balance,
            dec,
            lm,
            coded_bands,
            &mut self.rng,
            self.disable_inv,
        );

        // Anti-collapse
        let anti_collapse_on = if anti_collapse_rsv > 0 {
            dec.dec_bits(1) != 0
        } else {
            false
        };

        unquant_energy_finalise(
            mode,
            start,
            end,
            Some(&mut self.old_band_e),
            &fine_quant,
            &fine_priority,
            len as i32 * 8 - dec.tell(),
            dec,
            c,
        );

        if anti_collapse_on {
            anti_collapse(
                mode,
                &mut x_norm,
                &collapse_masks,
                lm,
                c,
                n,
                start,
                end,
                &self.old_band_e,
                &self.old_log_e,
                &self.old_log_e2,
                &pulses_vec,
                self.rng,
            );
        }

        if silence {
            for i in 0..(c * nb_ebands) {
                self.old_band_e[i] = -28.0;
            }
        }

        // Synthesis
        self.celt_synthesis(
            mode,
            &x_norm,
            start,
            eff_end,
            c,
            cc,
            is_transient,
            lm,
            n,
            silence,
        );

        // Comb filter (postfilter)
        for ch in 0..cc {
            let buf_base = ch * (DECODE_BUFFER_SIZE + overlap);
            let out_start = buf_base + DECODE_BUFFER_SIZE - n;

            let pf_period = self.postfilter_period.max(COMBFILTER_MINPERIOD);
            let pf_period_old = self.postfilter_period_old.max(COMBFILTER_MINPERIOD);

            // Apply comb filter: old parameters transition to new
            // We need to work on the output buffer in-place.
            // The comb filter reads from x[offset - period..] so we need history.
            // The decode_mem buffer has the full history.
            //
            // First segment: overlap transition from old to new parameters
            // Second segment: new parameters only
            let nb = mode.short_mdct_size;

            // Apply comb filter in-place: transition from old to current state parameters
            comb_filter_inplace(
                &mut self.decode_mem,
                out_start,
                pf_period_old,
                pf_period,
                nb,
                self.postfilter_gain_old,
                self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
                mode.window,
                overlap,
            );

            // Apply transition from current state to new bitstream parameters
            if lm != 0 {
                comb_filter_inplace(
                    &mut self.decode_mem,
                    out_start + nb,
                    pf_period,
                    postfilter_pitch.max(COMBFILTER_MINPERIOD),
                    n - nb,
                    self.postfilter_gain,
                    postfilter_gain,
                    self.postfilter_tapset,
                    postfilter_tapset,
                    mode.window,
                    overlap,
                );
            }
        }

        // Update postfilter state
        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.postfilter_period = postfilter_pitch;
        self.postfilter_gain = postfilter_gain;
        self.postfilter_tapset = postfilter_tapset;
        if lm != 0 {
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
        }

        // Copy mono to stereo if needed
        if c == 1 {
            let (first, second) = self.old_band_e.split_at_mut(nb_ebands);
            second[..nb_ebands].copy_from_slice(first);
        }

        // Update log energy history
        if !is_transient {
            self.old_log_e2[..2 * nb_ebands].copy_from_slice(&self.old_log_e[..2 * nb_ebands]);
            self.old_log_e[..2 * nb_ebands].copy_from_slice(&self.old_band_e[..2 * nb_ebands]);
        } else {
            for i in 0..(2 * nb_ebands) {
                self.old_log_e[i] = self.old_log_e[i].min(self.old_band_e[i]);
            }
        }

        // Update background noise estimate
        let max_bg_increase = (self.loss_duration as f32 + mm as f32).min(160.0) * 0.001;
        for i in 0..(2 * nb_ebands) {
            self.background_log_e[i] =
                (self.background_log_e[i] + max_bg_increase).min(self.old_band_e[i]);
        }

        // Clear out-of-range bands
        for ch in 0..2 {
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

        self.rng = dec.rng;

        // De-emphasis and output
        self.deemphasis(pcm, n, cc);

        self.loss_duration = 0;
        self.last_frame_type = FRAME_NORMAL;
        self.prefilter_and_fold = false;

        if dec.tell() > 8 * len as i32 {
            return Err(-3); // OPUS_INTERNAL_ERROR
        }
        if dec.get_error() {
            self.error = true;
        }
        Ok(())
    }

    /// Synthesis: denormalize + IMDCT + overlap-add.
    /// Matches C celt_synthesis() for the normal mono/stereo case.
    fn celt_synthesis(
        &mut self,
        mode: &CeltMode,
        x: &[f32],
        start: usize,
        eff_end: usize,
        c: usize,
        cc: usize,
        is_transient: bool,
        lm: i32,
        n: usize,
        silence: bool,
    ) {
        let overlap = mode.overlap;
        let mm = 1usize << lm;
        let nb_ebands = mode.nb_ebands;

        let (b, nb) = if is_transient {
            (mm, mode.short_mdct_size)
        } else {
            (1usize, mode.short_mdct_size << lm as usize)
        };
        let shift = if is_transient {
            mode.max_lm
        } else {
            mode.max_lm - lm as usize
        };

        let mut freq = vec![0.0f32; n];

        for ch in 0..cc.min(c) {
            // Denormalize
            denormalise_bands(
                mode,
                &x[ch * n..],
                &mut freq,
                &self.old_band_e[ch * nb_ebands..],
                start,
                eff_end,
                mm,
                self.downsample,
                silence,
            );

            // IMDCT for each short block
            // C: clt_mdct_backward(&mode->mdct, &freq[blk], out_syn[c]+NB*blk, window, overlap, shift, B)
            // freq[blk] means starting at index blk, with stride B
            // out_syn[c]+NB*blk means output at position NB*blk within the channel output
            let buf_base = ch * (DECODE_BUFFER_SIZE + overlap);
            let out_start = buf_base + DECODE_BUFFER_SIZE - n;
            for blk in 0..b {
                clt_mdct_backward(
                    &mut self.mdct,
                    &freq[blk..],
                    &mut self.decode_mem[out_start + nb * blk..],
                    mode.window,
                    overlap,
                    shift,
                    b, // stride = B
                );
            }
        }

        // If mono input -> stereo output, copy
        if cc == 2 && c == 1 {
            let base0 = DECODE_BUFFER_SIZE - n;
            let base1 = (DECODE_BUFFER_SIZE + overlap) + DECODE_BUFFER_SIZE - n;
            for i in 0..(n + overlap) {
                self.decode_mem[base1 + i] = self.decode_mem[base0 + i];
            }
        }

        // Saturate
        for ch in 0..cc {
            let buf_base = ch * (DECODE_BUFFER_SIZE + overlap);
            let out_start = buf_base + DECODE_BUFFER_SIZE - n;
            for i in 0..n {
                self.decode_mem[out_start + i] =
                    self.decode_mem[out_start + i].clamp(-SIG_SAT, SIG_SAT);
            }
        }
    }

    /// De-emphasis filter and write to PCM output.
    /// Matches C deemphasis() for float: applies IIR filter and SIG2RES scaling.
    fn deemphasis(&mut self, pcm: &mut [f32], n: usize, cc: usize) {
        let mode = CeltMode::get_mode();
        let coef0 = mode.preemph[0];
        let nd = n / self.downsample;
        let overlap = mode.overlap;
        // C float uses SIG2RES(a) = (1/CELT_SIG_SCALE)*a = a/32768
        const CELT_SIG_SCALE: f32 = 32768.0;
        let sig2res = 1.0 / CELT_SIG_SCALE;

        for ch in 0..cc {
            let buf_base = ch * (DECODE_BUFFER_SIZE + overlap);
            let out_start = buf_base + DECODE_BUFFER_SIZE - n;
            let mut m = self.preemph_mem[ch];

            if self.downsample > 1 {
                let mut scratch = vec![0.0f32; n];
                for (j, item) in scratch.iter_mut().enumerate().take(n) {
                    let tmp =
                        (self.decode_mem[out_start + j] + VERY_SMALL + m).clamp(-SIG_SAT, SIG_SAT);
                    m = coef0 * tmp;
                    *item = sig2res * tmp;
                }
                for j in 0..nd {
                    pcm[j * cc + ch] = scratch[j * self.downsample];
                }
            } else {
                for j in 0..n {
                    let tmp =
                        (self.decode_mem[out_start + j] + VERY_SMALL + m).clamp(-SIG_SAT, SIG_SAT);
                    m = coef0 * tmp;
                    pcm[j * cc + ch] = sig2res * tmp;
                }
            }
            self.preemph_mem[ch] = m;
        }
    }

    /// Handle packet loss concealment (simplified: output silence).
    fn decode_lost(&mut self, n: usize, lm: i32, cc: usize) {
        let mode = CeltMode::get_mode();
        let overlap = mode.overlap;

        for ch in 0..cc {
            let buf_base = ch * (DECODE_BUFFER_SIZE + overlap);
            // Shift memory
            self.decode_mem.copy_within(
                (buf_base + n)..(buf_base + DECODE_BUFFER_SIZE + overlap),
                buf_base,
            );
            // Fill with zeros (silence-based PLC)
            let out_start = buf_base + DECODE_BUFFER_SIZE - n;
            for i in 0..n {
                self.decode_mem[out_start + i] = 0.0;
            }
        }

        self.loss_duration = (self.loss_duration + (1 << lm as usize)).min(10000);
        self.last_frame_type = FRAME_PLC_NOISE;
    }
}

/// TF (time-frequency) resolution decode.
fn tf_decode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    dec: &mut EcCtx,
) {
    let budget = dec.storage as i32 * 8;
    let mut tell = dec.tell();
    let mut logp = if is_transient { 2u32 } else { 4u32 };
    let tf_select_rsv = lm > 0 && tell + (logp as i32) < budget;
    let budget = budget - if tf_select_rsv { 1 } else { 0 };
    let mut tf_changed = false;
    let mut curr = 0i32;

    for item in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp as i32 <= budget {
            curr ^= if dec.dec_bit_logp(logp) { 1 } else { 0 };
            tell = dec.tell();
            tf_changed = tf_changed || curr != 0;
        }
        *item = curr;
        logp = if is_transient { 4 } else { 5 };
    }

    let tf_select = if tf_select_rsv
        && TF_SELECT_TABLE[lm as usize][(4 * is_transient as usize) + tf_changed as usize]
            != TF_SELECT_TABLE[lm as usize][(4 * is_transient as usize) + 2 + tf_changed as usize]
    {
        if dec.dec_bit_logp(1) { 1 } else { 0 }
    } else {
        0
    };

    for item in tf_res.iter_mut().take(end).skip(start) {
        *item = TF_SELECT_TABLE[lm as usize]
            [4 * is_transient as usize + 2 * tf_select + *item as usize] as i32;
    }
}
