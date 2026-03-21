use crate::mode::CeltMode;
use crate::tables::*;
use crate::mathops::*;
use crate::rate::{bits2pulses, pulses2bits};
use opus_range_coder::EcCtx;

// =========================================================================
// denormalise_bands - matches C bands.c denormalise_bands() for float
// =========================================================================
pub fn denormalise_bands(
    m: &CeltMode,
    x: &[f32],
    freq: &mut [f32],
    band_log_e: &[f32],
    start: usize,
    end: usize,
    mm: usize,
    downsample: usize,
    silence: bool,
) {
    let n = mm * m.short_mdct_size;
    let mut bound = mm * m.ebands[end] as usize;
    if downsample != 1 {
        bound = bound.min(n / downsample);
    }
    let (start, end) = if silence { (0, 0) } else { (start, end) };
    if silence {
        for i in 0..n {
            freq[i] = 0.0;
        }
        return;
    }

    let mut x_idx = mm * m.ebands[start] as usize;

    // Zero up to the start band
    if start != 0 {
        let start_pos = mm * m.ebands[start] as usize;
        for j in 0..start_pos {
            freq[j] = 0.0;
        }
    }

    for i in start..end {
        let j_start = mm * m.ebands[i] as usize;
        let band_end = mm * m.ebands[i + 1] as usize;
        let lg = band_log_e[i] + E_MEANS[i];
        let g = celt_exp2(lg.min(32.0));
        let mut j = j_start;
        while j < band_end {
            freq[j] = x[x_idx] * g;
            x_idx += 1;
            j += 1;
        }
    }

    // Zero from bound to end
    for i in bound..n {
        freq[i] = 0.0;
    }
}

// =========================================================================
// anti_collapse - matches C bands.c anti_collapse() for float decode path
// =========================================================================
pub fn anti_collapse(
    m: &CeltMode,
    x: &mut [f32],
    collapse_masks: &[u8],
    lm: i32,
    c: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1_log_e: &[f32],
    prev2_log_e: &[f32],
    pulses: &[i32],
    mut seed: u32,
) {
    for i in start..end {
        let n0 = (m.ebands[i + 1] - m.ebands[i]) as usize;
        let depth = (celt_udiv(1 + pulses[i], (m.ebands[i + 1] - m.ebands[i]) as i32) >> lm) as i32;
        let thresh = 0.5 * celt_exp2(-0.125 * depth as f32);
        let sqrt_1 = celt_rsqrt((n0 << lm) as f32);

        for ch in 0..c {
            let mut prev1 = prev1_log_e[ch * m.nb_ebands + i];
            let mut prev2 = prev2_log_e[ch * m.nb_ebands + i];
            if c == 1 {
                prev1 = prev1.max(prev1_log_e[m.nb_ebands + i]);
                prev2 = prev2.max(prev2_log_e[m.nb_ebands + i]);
            }
            let ediff = (log_e[ch * m.nb_ebands + i] - prev1.min(prev2)).max(0.0);
            let mut r = 2.0 * celt_exp2(-ediff);
            if lm == 3 {
                r *= 1.41421356;
            }
            let r = thresh.min(r) * sqrt_1;

            let x_off = ch * size + (m.ebands[i] as usize) * (1usize << lm);
            let mut renormalize = false;
            for k in 0..(1usize << lm) {
                if (collapse_masks[i * c + ch] & (1u8 << k)) == 0 {
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        let idx = x_off + (j << lm) + k;
                        if idx < x.len() {
                            x[idx] = if seed & 0x8000 != 0 { r } else { -r };
                        }
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                let band_len = n0 << lm;
                if x_off + band_len <= x.len() {
                    renormalise_vector(&mut x[x_off..x_off + band_len], band_len, Q31ONE);
                }
            }
        }
    }
}

// =========================================================================
// Haar wavelet transform
// =========================================================================
pub fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    let n0 = n0 >> 1;
    let inv_sqrt2 = 0.70710678f32;
    for i in 0..stride {
        for j in 0..n0 {
            let tmp1 = inv_sqrt2 * x[stride * 2 * j + i];
            let tmp2 = inv_sqrt2 * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = tmp1 + tmp2;
            x[stride * (2 * j + 1) + i] = tmp1 - tmp2;
        }
    }
}

// =========================================================================
// Interleave / Deinterleave Hadamard
// =========================================================================
fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ordery[i] * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ordery[i] * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

// =========================================================================
// Stereo merge
// =========================================================================
// compute_qn for theta quantization
// =========================================================================
fn compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    let n2 = (2 * n as i32 - 1) - if stereo && n == 2 { 1 } else { 0 };
    let mut qb = celt_sudiv(b + n2 * offset, n2);
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let exp2_table8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
        let qn = exp2_table8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        ((qn + 1) >> 1 << 1).min(256)
    }
}

// =========================================================================
// exp_rotation - spreading function from vq.c
// =========================================================================
fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    let ms = -s;
    if len <= stride {
        return;
    }
    // Forward pass
    for i in 0..(len - stride) {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }
    // Backward pass
    if len >= 2 * stride + 1 {
        for i in (0..=(len - 2 * stride - 1)).rev() {
            let x1 = x[i];
            let x2 = x[i + stride];
            x[i + stride] = c * x2 + s * x1;
            x[i] = c * x1 + ms * x2;
        }
    }
}

fn exp_rotation(x: &mut [f32], len: usize, dir: i32, stride: usize, k: usize, spread: i32) {
    if 2 * k >= len || spread == SPREAD_NONE {
        return;
    }
    let spread_factor: [i32; 3] = [15, 10, 5];
    let factor = spread_factor[(spread - 1) as usize];

    let gain = (len as f32) / (len as f32 + factor as f32 * k as f32);
    let theta = 0.5 * gain * gain;

    // cos/sin approximation matching C (for float, just use trig)
    let c = (theta * std::f32::consts::PI * 0.5).cos();
    let s = (theta * std::f32::consts::PI * 0.5).sin();

    let mut stride2 = 0usize;
    if len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }

    let sub_len = celt_udiv(len as i32, stride as i32) as usize;
    for i in 0..stride {
        let x_sub = &mut x[i * sub_len..(i + 1) * sub_len];
        if dir < 0 {
            if stride2 > 0 {
                exp_rotation1(x_sub, sub_len, stride2, s, c);
            }
            exp_rotation1(x_sub, sub_len, 1, c, s);
        } else {
            exp_rotation1(x_sub, sub_len, 1, c, -s);
            if stride2 > 0 {
                exp_rotation1(x_sub, sub_len, stride2, s, -c);
            }
        }
    }
}

// =========================================================================
// extract_collapse_mask - from vq.c
// =========================================================================
fn extract_collapse_mask(iy: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = celt_udiv(n as i32, b as i32) as usize;
    let mut collapse_mask = 0u32;
    for i in 0..b {
        let mut tmp = 0u32;
        for j in 0..n0 {
            tmp |= iy[i * n0 + j] as u32;
        }
        if tmp != 0 {
            collapse_mask |= 1 << i;
        }
    }
    collapse_mask
}

// =========================================================================
// normalise_residual - from vq.c, float path
// =========================================================================
fn normalise_residual(iy: &[i32], x: &mut [f32], n: usize, ryy: f32, gain: f32) {
    if ryy < 1e-30 {
        for i in 0..n {
            x[i] = 0.0;
        }
        return;
    }
    let g = gain / ryy.sqrt();
    for i in 0..n {
        x[i] = iy[i] as f32 * g;
    }
}

// =========================================================================
// alg_unquant - matches C vq.c alg_unquant() for float decode
// =========================================================================
fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: usize,
    spread: i32,
    b: usize,
    ec: &mut EcCtx,
    gain: f32,
) -> u32 {
    debug_assert!(k > 0);
    debug_assert!(n > 1);
    let mut iy = vec![0i32; n];
    let ryy = ec.decode_pulses(&mut iy, n, k);
    normalise_residual(&iy, x, n, ryy, gain);
    exp_rotation(x, n, -1, b, k, spread);
    extract_collapse_mask(&iy, n, b)
}

// =========================================================================
// Band context
// =========================================================================
struct BandCtx {
    spread: i32,
    seed: u32,
    remaining_bits: i32,
    avoid_split_noise: bool,
    tf_change: i32,
}

// =========================================================================
// quant_partition - matches C bands.c quant_partition() decode path
// =========================================================================
fn quant_partition(
    m: &CeltMode,
    band_idx: usize,
    ec: &mut EcCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b0: usize,  // B
    lm: i32,
    ctx: &mut BandCtx,
    gain: f32,
    fill: u32,
    lowband: Option<&[f32]>,
) -> u32 {
    let cache_idx = m.cache.index[((lm + 1) as usize) * m.nb_ebands + band_idx];
    let cache = if cache_idx >= 0 {
        &m.cache.bits[cache_idx as usize..]
    } else {
        &[] as &[u8]
    };

    // Check if we should split
    if lm != -1 && !cache.is_empty() && b > cache[cache[0] as usize] as i32 + 12 && n > 2 {
        // Recursive split
        let half_n = n >> 1;
        let new_lm = lm - 1;
        let new_b0 = (b0 + 1) >> 1;
        let mut new_fill = fill;
        if b0 == 1 {
            new_fill = (fill & 1) | (fill << 1);
        }
        let new_b = new_b0;  // B after split (used for fill masking)

        // Compute theta (use new_lm, matching C's LM which is decremented before compute_theta)
        let pulse_cap = m.log_n[band_idx] as i32 + new_lm * (1 << BITRES);
        let offset = (pulse_cap >> 1) - QTHETA_OFFSET;
        let qn = compute_qn(half_n, b, offset, pulse_cap, false);

        let mut itheta;
        let tell = ec.tell_frac();
        if qn != 1 {
            if b0 > 1 {
                itheta = ec.dec_uint((qn + 1) as u32) as i32;
            } else {
                // Triangular PDF
                let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
                let fm = ec.decode(ft as u32);
                if (fm as i32) < ((qn >> 1) * ((qn >> 1) + 1) >> 1) {
                    itheta = ((isqrt32(8 * fm + 1) as i32 - 1) >> 1) as i32;
                    let fs = itheta + 1;
                    let fl = itheta * (itheta + 1) >> 1;
                    ec.dec_update(fl as u32, (fl + fs) as u32, ft as u32);
                } else {
                    itheta = (2 * (qn + 1) - isqrt32(8 * (ft as u32 - fm - 1) + 1) as i32) >> 1;
                    let fs = qn + 1 - itheta;
                    let fl = ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
                    ec.dec_update(fl as u32, (fl + fs) as u32, ft as u32);
                }
            }
            itheta = celt_udiv(itheta * 16384, qn);
        } else {
            itheta = 0;
        }
        let qalloc = ec.tell_frac() as i32 - tell as i32;

        let imid;
        let iside;
        let delta;
        let b_left = b - qalloc;
        if itheta == 0 {
            imid = 32767i32;
            iside = 0i32;
            new_fill &= (1u32 << new_b) - 1;
            delta = -16384i32;
        } else if itheta == 16384 {
            imid = 0i32;
            iside = 32767i32;
            new_fill &= ((1u32 << new_b) - 1) << new_b;
            delta = 16384i32;
        } else {
            imid = bitexact_cos(itheta as i16) as i32;
            iside = bitexact_cos((16384 - itheta) as i16) as i32;
            delta = frac_mul16((half_n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
        }

        let mid = imid as f32 / 32768.0;
        let side = iside as f32 / 32768.0;

        // Give more bits to low-energy MDCTs
        let mut delta_adj = delta;
        if b0 > 1 && (itheta & 0x3fff) != 0 {
            if itheta > 8192 {
                delta_adj -= delta_adj >> (4 - new_lm);
            } else {
                delta_adj = 0i32.min(delta_adj + ((half_n as i32) << BITRES >> (5 - new_lm)));
            }
        }

        let mbits = 0i32.max(b_left.min((b_left - delta_adj) / 2));
        let sbits = b_left - mbits;
        ctx.remaining_bits -= qalloc;

        let next_lowband2 = lowband.map(|lb| {
            if half_n <= lb.len() { Some(&lb[half_n..]) } else { None }
        }).flatten();

        // Rebalancing
        let rebalance = ctx.remaining_bits;
        let mut final_sbits = sbits;
        let mut final_mbits = mbits;

        if mbits >= sbits {
            let cm_mid = quant_partition(
                m, band_idx, ec, &mut x[..half_n], half_n, mbits, new_b0, new_lm,
                ctx, gain * mid, new_fill, lowband,
            );
            let rebalance_amount = mbits - (rebalance - ctx.remaining_bits);
            if rebalance_amount > 3 << BITRES && itheta != 0 {
                final_sbits = sbits + rebalance_amount - (3 << BITRES);
            }
            let cm_side = quant_partition(
                m, band_idx, ec, &mut x[half_n..], half_n, final_sbits, new_b0, new_lm,
                ctx, gain * side, new_fill >> new_b as u32, next_lowband2,
            );
            let result = cm_mid | (cm_side << (b0 >> 1) as u32);
            result
        } else {
            let cm_side = quant_partition(
                m, band_idx, ec, &mut x[half_n..], half_n, sbits, new_b0, new_lm,
                ctx, gain * side, new_fill >> new_b as u32, next_lowband2,
            );
            let rebalance_amount = sbits - (rebalance - ctx.remaining_bits);
            if rebalance_amount > 3 << BITRES && itheta != 16384 {
                final_mbits = mbits + rebalance_amount - (3 << BITRES);
            }
            let cm_mid = quant_partition(
                m, band_idx, ec, &mut x[..half_n], half_n, final_mbits, new_b0, new_lm,
                ctx, gain * mid, new_fill, lowband,
            );
            let result = cm_mid | (cm_side << (b0 >> 1) as u32);
            result
        }
    } else {
        // Base case: PVQ decode
        let q = bits2pulses(m, band_idx, lm, b);
        let curr_bits = pulses2bits(m, band_idx, lm, q);
        ctx.remaining_bits -= curr_bits;
        let mut q = q;
        let mut curr_bits = curr_bits;

        // Ensure we can never bust the budget
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(m, band_idx, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q) as usize;
            let cm = alg_unquant(x, n, k, ctx.spread, b0, ec, gain);
            return cm;
        } else {
            // No pulses: fill with noise or folding
            let cm_mask = if b0 < 32 { (1u32 << b0) - 1 } else { 0xFFFFFFFF };
            let fill = fill & cm_mask;
            if fill == 0 {
                for j in 0..n {
                    x[j] = 0.0;
                }
            } else {
                if let Some(lb) = lowband {
                    // Folded spectrum
                    for j in 0..n {
                        ctx.seed = celt_lcg_rand(ctx.seed);
                        let tmp: f32 = 1.0 / 256.0;
                        let tmp = if ctx.seed & 0x8000 != 0 { tmp } else { -tmp };
                        x[j] = lb[j.min(lb.len() - 1)] + tmp;
                    }
                } else {
                    // Noise
                    for j in 0..n {
                        ctx.seed = celt_lcg_rand(ctx.seed);
                        x[j] = (ctx.seed as i32 >> 20) as f32;
                    }
                }
                renormalise_vector(&mut x[..n], n, gain);
                return if lowband.is_some() { fill } else { cm_mask };
            }
            return 0;
        }
    }
}

// =========================================================================
// quant_band - matches C bands.c quant_band() decode path
// =========================================================================
fn quant_band(
    m: &CeltMode,
    band_idx: usize,
    ec: &mut EcCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: usize,  // B
    lm: i32,
    ctx: &mut BandCtx,
    gain: f32,
    lowband: Option<&[f32]>,
    lowband_out: Option<&mut [f32]>,
    _lowband_scratch: Option<&mut Vec<f32>>,
    fill: u32,
) -> u32 {
    let n0 = n;
    let mut n_b = celt_udiv(n as i32, b_blocks as i32) as usize;
    let mut b_cur = b_blocks; // B
    let b0 = b_blocks;
    let mut time_divide = 0usize;
    let mut recombine = 0usize;
    let long_blocks = b0 == 1;
    let tf_change = ctx.tf_change;
    let mut fill = fill;

    // Special case: N=1
    if n == 1 {
        let mut sign = 0u32;
        if ctx.remaining_bits >= 1 << BITRES {
            sign = ec.dec_bits(1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        x[0] = if sign != 0 { -NORM_SCALING } else { NORM_SCALING };
        if let Some(lo) = lowband_out {
            lo[0] = x[0];
        }
        return 1;
    }

    // Band recombining to increase frequency resolution
    if tf_change > 0 {
        recombine = tf_change as usize;
    }

    // Copy lowband to scratch if needed for modifications
    let mut lowband_copy: Option<Vec<f32>> = None;
    if let Some(lb) = lowband {
        if recombine > 0 || ((n_b & 1) == 0 && tf_change < 0) || b0 > 1 {
            lowband_copy = Some(lb[..n.min(lb.len())].to_vec());
        }
    }

    // Apply recombine haar transforms (decode: only on lowband)
    for k in 0..recombine {
        if let Some(ref mut lb) = lowband_copy {
            if n >> k >= 2 {
                haar1(lb, n >> k, 1 << k);
            }
        }
        fill = BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize] as u32
            | (BIT_INTERLEAVE_TABLE[((fill >> 4) & 0xF) as usize] as u32) << 2;
    }
    b_cur >>= recombine;
    n_b <<= recombine;

    // Increasing the time resolution
    let mut tc = tf_change;
    while (n_b & 1) == 0 && tc < 0 {
        if let Some(ref mut lb) = lowband_copy {
            haar1(lb, n_b, b_cur);
        }
        fill |= fill << b_cur;
        b_cur <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tc += 1;
    }
    let b0_final = b_cur;
    let n_b0 = n_b;

    // Reorganize samples in time order instead of frequency order
    if b0_final > 1 {
        if let Some(ref mut lb) = lowband_copy {
            deinterleave_hadamard(lb, n_b >> recombine, b0_final << recombine, long_blocks);
        }
    }

    // Do the actual quantization through quant_partition
    let lb_ref = lowband_copy.as_deref();
    let mut cm = quant_partition(
        m, band_idx, ec, x, n, b, b0_final, lm, ctx, gain, fill, lb_ref,
    );

    // Undo the sample reorganization going from time order to frequency order
    if b0_final > 1 {
        interleave_hadamard(x, n_b >> recombine, b0_final << recombine, long_blocks);
    }

    // Undo time-freq changes
    let mut n_b_undo = n_b0;
    let mut b_undo = b0_final;
    for _ in 0..time_divide {
        b_undo >>= 1;
        n_b_undo <<= 1;
        cm |= cm >> b_undo;
        haar1(x, n_b_undo, b_undo);
    }

    for k in 0..recombine {
        cm = BIT_DEINTERLEAVE_TABLE[(cm & 0xF) as usize] as u32;
        haar1(x, n0 >> k, 1 << k);
    }
    // B after undo = (B_orig >> recombine) << recombine (matches C's B<<=recombine after undo)
    let b_final = (b_blocks >> recombine) << recombine;

    // Scale output for later folding
    // In float path: SHL32(EXTEND32(N0),22) is just N0 (SHL32 is identity in float)
    // and MULT16_32_Q15(n, X[j]) is just n*X[j]
    if let Some(lo) = lowband_out {
        let n_scale = celt_sqrt(n0 as f32);
        for j in 0..n0.min(lo.len()) {
            lo[j] = n_scale * x[j];
        }
    }
    cm & ((1u32 << b_final) - 1)
}

// =========================================================================
// quant_all_bands - matches C bands.c quant_all_bands() decode path
// =========================================================================
pub fn quant_all_bands(
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    collapse_masks: &mut [u8],
    pulses: &mut [i32],
    short_blocks: i32,
    spread: i32,
    tf_res: &[i32],
    total_bits: i32,
    balance: i32,
    ec: &mut EcCtx,
    lm: i32,
    coded_bands: usize,
    seed: &mut u32,
) {
    let mm = 1usize << lm;
    let b_short = if short_blocks != 0 { mm } else { 1usize };
    let c = 1usize; // mono for now
    let norm_offset = mm * m.ebands[start] as usize;

    // Allocate norm buffers for folding
    let norm_end = mm * m.ebands[m.nb_ebands - 1] as usize;
    let norm_size = norm_end - norm_offset;
    let mut norm = vec![0.0f32; norm_size.max(1)];

    let mut lowband_offset = 0usize;
    let mut update_lowband = true;

    let mut ctx = BandCtx {
        spread,
        seed: *seed,
        remaining_bits: total_bits,
        avoid_split_noise: b_short > 1,
        tf_change: 0,
    };

    let mut bal = balance;

    for i in start..end {
        let tell = ec.tell_frac() as i32;
        let n = (mm * (m.ebands[i + 1] - m.ebands[i]) as usize) as usize;
        let x_off = mm * m.ebands[i] as usize;
        let last = i == end - 1;

        // Compute how many bits to allocate
        if i != start {
            bal -= tell;
        }
        ctx.remaining_bits = total_bits - tell - 1;

        let b;
        if i <= coded_bands.saturating_sub(1) {
            let curr_balance = celt_sudiv(bal, (coded_bands - i).min(3) as i32);
            b = 0i32.max(16383i32.min((ctx.remaining_bits + 1).min(pulses[i] + curr_balance)));
        } else {
            b = 0;
        }
        // Update lowband offset
        if i > start {
            let eff_low = mm * m.ebands[i] as usize;
            if (eff_low >= mm * m.ebands[start] as usize + n || i == start + 1)
                && (update_lowband || lowband_offset == 0)
            {
                lowband_offset = i;
            }
        }

        // Special hybrid folding for band start+1
        // C: special_hybrid_folding: OPUS_COPY(&norm[n1], &norm[2*n1 - n2], n2-n1);
        if i == start + 1 {
            let n1 = mm * (m.ebands[start + 1] - m.ebands[start]) as usize;
            if start + 2 <= m.nb_ebands {
                let n2 = mm * (m.ebands[start + 2] - m.ebands[start + 1]) as usize;
                let copy_len = n2 - n1;
                if copy_len > 0 && 2 * n1 >= n2 {
                    let src_start = 2 * n1 - n2;
                    for j in 0..copy_len {
                        if src_start + j < norm.len() && n1 + j < norm.len() {
                            norm[n1 + j] = norm[src_start + j];
                        }
                    }
                }
            }
        }

        let tf_change = tf_res[i] as i32;
        ctx.tf_change = tf_change;

        // Get effective lowband for folding
        let effective_lowband = if lowband_offset != 0
            && (spread != SPREAD_AGGRESSIVE || b_short > 1 || tf_change < 0)
        {
            let eff = (mm * m.ebands[lowband_offset] as usize).saturating_sub(norm_offset + n);
            eff as i32
        } else {
            -1i32
        };

        // Compute fill mask from collapse masks of folding bands
        let (mut x_cm, mut y_cm);
        if effective_lowband >= 0 {
            let fold_start_bound = effective_lowband as usize + norm_offset;
            // C: fold_start = lowband_offset; while(M*eBands[--fold_start] > eff+norm_offset);
            // Pre-decrement: always decrements at least once before checking
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if !((mm * m.ebands[fold_start] as usize) > fold_start_bound) {
                    break;
                }
            }
            // C: fold_end = lowband_offset-1; while(++fold_end < i && M*eBands[fold_end] < eff+norm_offset+N);
            let mut fold_end = lowband_offset;
            while fold_end < i && (mm * m.ebands[fold_end] as usize) < fold_start_bound + n {
                fold_end += 1;
            }
            x_cm = 0u32;
            y_cm = 0u32;
            // C: do { ... } while (++fold_i < fold_end); -- always executes at least once
            let mut fold_i = fold_start;
            loop {
                x_cm |= collapse_masks[fold_i * c] as u32;
                y_cm |= collapse_masks[fold_i * c + c - 1] as u32;
                fold_i += 1;
                if fold_i >= fold_end {
                    break;
                }
            }
        } else {
            x_cm = (1u32 << b_short) - 1;
            y_cm = x_cm;
        }

        // Prepare lowband for folding
        let lowband_slice = if effective_lowband >= 0 && (effective_lowband as usize) < norm.len() {
            Some(&norm[effective_lowband as usize..])
        } else {
            None
        };

        // Save remaining_bits for tf_change

        // Decode band (mono path)
        let norm_off_band = (mm * m.ebands[i] as usize).saturating_sub(norm_offset);
        let lowband_out_valid = !last && norm_off_band + n <= norm.len();

        let cm;
        // We need a temporary for lowband_out since we can't borrow norm mutably while also reading it
        let mut lowband_out_tmp = vec![0.0f32; if lowband_out_valid { n } else { 0 }];

        cm = quant_band(
            m, i, ec,
            &mut x[x_off..x_off + n], n, b, b_short, lm,
            &mut ctx, Q31ONE,
            lowband_slice,
            if lowband_out_valid { Some(&mut lowband_out_tmp) } else { None },
            None,
            x_cm | y_cm,
        );

        // Copy lowband_out back to norm
        if lowband_out_valid && norm_off_band + n <= norm.len() {
            let copy_n = n.min(lowband_out_tmp.len());
            norm[norm_off_band..norm_off_band + copy_n].copy_from_slice(&lowband_out_tmp[..copy_n]);
        }

        collapse_masks[i * c] = cm as u8;
        if c == 2 {
            collapse_masks[i * c + 1] = cm as u8;
        }

        bal += pulses[i] + tell;
        update_lowband = b > (n as i32) << BITRES;
        ctx.avoid_split_noise = false;
    }

    *seed = ctx.seed;
}
