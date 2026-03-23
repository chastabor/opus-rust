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
// Stereo split / merge (float path)
// =========================================================================
#[allow(dead_code)]
fn stereo_split(x: &mut [f32], y: &mut [f32], n: usize) {
    let inv_sqrt2: f32 = 0.70710678;
    for j in 0..n {
        let l = inv_sqrt2 * x[j];
        let r = inv_sqrt2 * y[j];
        x[j] = l + r;
        y[j] = r - l;
    }
}

fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    // Compute the norm of X+Y and X-Y as |X|^2 + |Y|^2 +/- sum(xy)
    let xp_raw = celt_inner_prod(y, x, n);
    let side = celt_inner_prod(y, y, n);
    // Compensating for the mid normalization
    let xp = mid * xp_raw;
    let el = mid * mid * 0.125 + side - 2.0 * xp;
    let er = mid * mid * 0.125 + side + 2.0 * xp;
    if el < 6e-4 || er < 6e-4 {
        x[..n].copy_from_slice(&y[..n]);
        return;
    }
    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();
    for j in 0..n {
        let l = mid * x[j];
        let r = y[j];
        x[j] = lgain * (l - r);
        y[j] = rgain * (l + r);
    }
}

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
// compute_theta - decode stereo/split theta parameter
// Matches C bands.c compute_theta() decode path
// =========================================================================
fn compute_theta(
    m: &CeltMode,
    band_idx: usize,
    ec: &mut EcCtx,
    n: usize,
    b: &mut i32,
    b0: usize,
    lm: i32,
    ctx: &mut BandCtx,
    stereo: bool,
    fill: &mut u32,
) -> SplitCtx {
    let i = band_idx;
    let intensity = ctx.intensity;

    // Decide on the resolution to give to the split parameter theta
    let pulse_cap = m.log_n[i] as i32 + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1) - if stereo && n == 2 { QTHETA_OFFSET_TWOPHASE } else { QTHETA_OFFSET };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if stereo && (i as i32) >= intensity {
        qn = 1;
    }

    let tell = ec.tell_frac();
    let mut itheta;
    let mut inv = false;

    if qn != 1 {
        // Entropy coding of the angle
        if stereo && n > 2 {
            // Step PDF for stereo N>2
            let p0: i32 = 3;
            let x0 = qn / 2;
            let ft = (p0 * (x0 + 1) + x0) as u32;
            let fs = ec.decode(ft) as i32;
            let x;
            if fs < (x0 + 1) * p0 {
                x = fs / p0;
            } else {
                x = x0 + 1 + (fs - (x0 + 1) * p0);
            }
            let fl = if x <= x0 { p0 * x } else { (x - 1 - x0) + (x0 + 1) * p0 };
            let fh = if x <= x0 { p0 * (x + 1) } else { (x - x0) + (x0 + 1) * p0 };
            ec.dec_update(fl as u32, fh as u32, ft);
            itheta = x;
        } else if b0 > 1 || stereo {
            // Uniform PDF
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
    } else if stereo {
        // qn==1 stereo path: decode inv bit
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            inv = ec.dec_bit_logp(2);
        } else {
            inv = false;
        }
        // inv flag override to avoid problems with downmixing
        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
    } else {
        itheta = 0;
    }

    let qalloc = ec.tell_frac() as i32 - tell as i32;
    *b -= qalloc;

    let imid;
    let iside;
    let delta;
    let b_val = b0; // B for fill masking
    if itheta == 0 {
        imid = 32767i32;
        iside = 0i32;
        *fill &= (1u32 << b_val) - 1;
        delta = -16384i32;
    } else if itheta == 16384 {
        imid = 0i32;
        iside = 32767i32;
        *fill &= ((1u32 << b_val) - 1) << b_val;
        delta = 16384i32;
    } else {
        imid = bitexact_cos(itheta as i16) as i32;
        iside = bitexact_cos((16384 - itheta) as i16) as i32;
        delta = frac_mul16((n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
    }

    SplitCtx {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
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
    intensity: i32,
    disable_inv: bool,
}

struct SplitCtx {
    inv: bool,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    qalloc: i32,
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
// quant_band_stereo - matches C bands.c quant_band_stereo() decode path
// =========================================================================
fn quant_band_stereo(
    m: &CeltMode,
    band_idx: usize,
    ec: &mut EcCtx,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    mut b: i32,
    b_blocks: usize, // B
    lm: i32,
    ctx: &mut BandCtx,
    lowband: Option<&[f32]>,
    lowband_out: Option<&mut [f32]>,
    fill: u32,
) -> u32 {
    let mut cm: u32;

    // Special case for N=1: decode sign bits independently
    if n == 1 {
        // Decode X sign
        let mut sign = 0u32;
        if ctx.remaining_bits >= 1 << BITRES {
            sign = ec.dec_bits(1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        x[0] = if sign != 0 { -NORM_SCALING } else { NORM_SCALING };
        // Decode Y sign
        sign = 0;
        if ctx.remaining_bits >= 1 << BITRES {
            sign = ec.dec_bits(1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        y[0] = if sign != 0 { -NORM_SCALING } else { NORM_SCALING };
        if let Some(lo) = lowband_out {
            lo[0] = x[0] * 0.0625; // SHR32(X[0], 4) in float = X[0] / 16
        }
        return 1;
    }

    let orig_fill = fill;
    let mut fill = fill;

    let sctx = compute_theta(
        m, band_idx, ec, n, &mut b, b_blocks, lm, ctx, true, &mut fill,
    );
    let inv = sctx.inv;
    let imid = sctx.imid;
    let iside = sctx.iside;
    let delta = sctx.delta;
    let itheta = sctx.itheta;
    let qalloc = sctx.qalloc;

    let mid = imid as f32 / 32768.0;
    let side = iside as f32 / 32768.0;

    // N==2 special case: encode side with just one bit
    if n == 2 {
        let mbits;
        let mut sbits = 0i32;
        // Only need one bit for the side
        if itheta != 0 && itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits = b - sbits;
        let c = itheta > 8192;
        ctx.remaining_bits -= qalloc + sbits;

        let mut sign = 0i32;
        if sbits != 0 {
            sign = ec.dec_bits(1) as i32;
        }
        sign = 1 - 2 * sign;

        // x2/y2 pointers: if c, swap X and Y for decoding
        // We need to work with temporaries since we can't have two mutable refs
        let mut x2 = [x[0], x[1]];
        let mut y2 = [y[0], y[1]];

        if c {
            // x2 = Y, y2 = X
            x2 = [y[0], y[1]];
            y2 = [x[0], x[1]];
        }

        // Decode x2 with quant_band using orig_fill
        let cm_val = quant_band(
            m, band_idx, ec,
            &mut x2, n, mbits, b_blocks, lm,
            ctx, Q31ONE,
            lowband,
            None,
            None,
            orig_fill,
        );
        cm = cm_val;

        // Construct y2 from rotation
        y2[0] = -(sign as f32) * x2[1];
        y2[1] = (sign as f32) * x2[0];

        // Write back, undoing the swap
        if c {
            y[0] = x2[0]; y[1] = x2[1];
            x[0] = y2[0]; x[1] = y2[1];
        } else {
            x[0] = x2[0]; x[1] = x2[1];
            y[0] = y2[0]; y[1] = y2[1];
        }

        // Apply mid/side scaling and reconstruct L/R
        let mx0 = mid * x[0];
        let mx1 = mid * x[1];
        let sy0 = side * y[0];
        let sy1 = side * y[1];
        x[0] = mx0 - sy0;
        y[0] = mx0 + sy0;
        x[1] = mx1 - sy1;
        y[1] = mx1 + sy1;
    } else {
        // Normal split code (N>2)
        let mbits = 0i32.max(b.min((b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        let rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            // Decode mid (X) first - in stereo mode no scaling applied (Q31ONE)
            let mut lowband_out_tmp: Option<Vec<f32>> = lowband_out.as_ref().map(|_| vec![0.0f32; n]);
            cm = quant_band(
                m, band_idx, ec,
                x, n, mbits, b_blocks, lm,
                ctx, Q31ONE,
                lowband,
                lowband_out_tmp.as_deref_mut(),
                None,
                fill,
            );
            // Copy lowband_out_tmp back if needed
            if let (Some(tmp), Some(lo)) = (&lowband_out_tmp, lowband_out) {
                let copy_n = n.min(tmp.len()).min(lo.len());
                lo[..copy_n].copy_from_slice(&tmp[..copy_n]);
            }

            let rebalance_amount = mbits - (rebalance - ctx.remaining_bits);
            if rebalance_amount > 3 << BITRES && itheta != 0 {
                sbits += rebalance_amount - (3 << BITRES);
            }
            // For stereo split, high bits of fill are always zero, so no folding for side
            cm |= quant_band(
                m, band_idx, ec,
                y, n, sbits, b_blocks, lm,
                ctx, side,
                None,
                None,
                None,
                fill >> b_blocks,
            );
        } else {
            // Decode side (Y) first
            cm = quant_band(
                m, band_idx, ec,
                y, n, sbits, b_blocks, lm,
                ctx, side,
                None,
                None,
                None,
                fill >> b_blocks,
            );
            let rebalance_amount = sbits - (rebalance - ctx.remaining_bits);
            let mut final_mbits = mbits;
            if rebalance_amount > 3 << BITRES && itheta != 16384 {
                final_mbits = mbits + rebalance_amount - (3 << BITRES);
            }
            // Decode mid (X)
            let mut lowband_out_tmp: Option<Vec<f32>> = lowband_out.as_ref().map(|_| vec![0.0f32; n]);
            cm |= quant_band(
                m, band_idx, ec,
                x, n, final_mbits, b_blocks, lm,
                ctx, Q31ONE,
                lowband,
                lowband_out_tmp.as_deref_mut(),
                None,
                fill,
            );
            // Copy lowband_out_tmp back if needed
            if let (Some(tmp), Some(lo)) = (&lowband_out_tmp, lowband_out) {
                let copy_n = n.min(tmp.len()).min(lo.len());
                lo[..copy_n].copy_from_slice(&tmp[..copy_n]);
            }
        }

        // stereo_merge for N!=2
        stereo_merge(x, y, mid, n);

        // Apply inv flag
        if inv {
            for j in 0..n {
                y[j] = -y[j];
            }
        }
    }

    cm
}

// =========================================================================
// quant_all_bands - matches C bands.c quant_all_bands() decode path
// =========================================================================
pub fn quant_all_bands(
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    collapse_masks: &mut [u8],
    pulses: &mut [i32],
    short_blocks: i32,
    spread: i32,
    dual_stereo: i32,
    intensity: i32,
    tf_res: &[i32],
    total_bits: i32,
    balance: i32,
    ec: &mut EcCtx,
    lm: i32,
    coded_bands: usize,
    seed: &mut u32,
    disable_inv: bool,
) {
    let mm = 1usize << lm;
    let b_short = if short_blocks != 0 { mm } else { 1usize };
    let c = if y.is_some() { 2usize } else { 1usize };
    let norm_offset = mm * m.ebands[start] as usize;

    // Allocate norm buffers for folding
    let norm_end = mm * m.ebands[m.nb_ebands - 1] as usize;
    let norm_size = norm_end - norm_offset;
    let mut norm = vec![0.0f32; (c * norm_size).max(1)];
    // norm2 is the second half when C==2
    let norm2_offset = norm_size;

    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut dual_stereo = dual_stereo != 0;

    let mut ctx = BandCtx {
        spread,
        seed: *seed,
        remaining_bits: total_bits,
        avoid_split_noise: b_short > 1,
        tf_change: 0,
        intensity,
        disable_inv,
    };

    let mut bal = balance;

    // We take y out of the Option to work with it mutably in the loop
    // If y is None, we'll use a dummy reference that's never accessed
    let have_y = y.is_some();
    let y_storage: &mut [f32] = match y {
        Some(y_ref) => y_ref,
        None => &mut [],
    };

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
        if (mm * m.ebands[i] as usize >= mm * m.ebands[start] as usize + n || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }

        // Special hybrid folding for band start+1
        if i == start + 1 {
            let n1 = mm * (m.ebands[start + 1] - m.ebands[start]) as usize;
            if start + 2 <= m.nb_ebands {
                let n2 = mm * (m.ebands[start + 2] - m.ebands[start + 1]) as usize;
                let copy_len = n2 - n1;
                if copy_len > 0 && 2 * n1 >= n2 {
                    let src_start = 2 * n1 - n2;
                    for j in 0..copy_len {
                        if src_start + j < norm_size && n1 + j < norm_size {
                            norm[n1 + j] = norm[src_start + j];
                        }
                    }
                    // Also copy norm2 when dual_stereo
                    if dual_stereo {
                        for j in 0..copy_len {
                            if src_start + j < norm_size && n1 + j < norm_size {
                                norm[norm2_offset + n1 + j] = norm[norm2_offset + src_start + j];
                            }
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
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if !((mm * m.ebands[fold_start] as usize) > fold_start_bound) {
                    break;
                }
            }
            let mut fold_end = lowband_offset;
            while fold_end < i && (mm * m.ebands[fold_end] as usize) < fold_start_bound + n {
                fold_end += 1;
            }
            x_cm = 0u32;
            y_cm = 0u32;
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

        // Switch off dual stereo at intensity boundary
        if dual_stereo && i as i32 == intensity {
            dual_stereo = false;
            // Merge norms: norm[j] = 0.5*(norm[j]+norm2[j])
            let merge_len = (mm * m.ebands[i] as usize).saturating_sub(norm_offset);
            for j in 0..merge_len.min(norm_size) {
                norm[j] = 0.5 * (norm[j] + norm[norm2_offset + j]);
            }
        }

        // Prepare lowband for folding
        let lowband_slice = if effective_lowband >= 0 && (effective_lowband as usize) < norm_size {
            Some(&norm[effective_lowband as usize..norm_size])
        } else {
            None
        };

        let lowband2_slice_start = if effective_lowband >= 0 && (effective_lowband as usize) < norm_size {
            Some(norm2_offset + effective_lowband as usize)
        } else {
            None
        };

        let norm_off_band = (mm * m.ebands[i] as usize).saturating_sub(norm_offset);
        let lowband_out_valid = !last && norm_off_band + n <= norm_size;

        if dual_stereo {
            // Dual stereo: decode X and Y independently with b/2 bits each
            let mut lowband_out_tmp = vec![0.0f32; if lowband_out_valid { n } else { 0 }];

            x_cm = quant_band(
                m, i, ec,
                &mut x[x_off..x_off + n], n, b / 2, b_short, lm,
                &mut ctx, Q31ONE,
                lowband_slice,
                if lowband_out_valid { Some(&mut lowband_out_tmp) } else { None },
                None,
                x_cm,
            );

            // Copy lowband_out to norm
            if lowband_out_valid && norm_off_band + n <= norm_size {
                let copy_n = n.min(lowband_out_tmp.len());
                norm[norm_off_band..norm_off_band + copy_n].copy_from_slice(&lowband_out_tmp[..copy_n]);
            }

            // For Y: use norm2 for folding
            let lowband2_slice = if let Some(start_idx) = lowband2_slice_start {
                if start_idx < norm.len() {
                    // We need to copy to a temp to avoid aliasing issues
                    Some(norm[start_idx..norm.len().min(start_idx + n * 4)].to_vec())
                } else {
                    None
                }
            } else {
                None
            };

            let mut lowband_out_tmp2 = vec![0.0f32; if lowband_out_valid { n } else { 0 }];

            y_cm = quant_band(
                m, i, ec,
                &mut y_storage[x_off..x_off + n], n, b / 2, b_short, lm,
                &mut ctx, Q31ONE,
                lowband2_slice.as_deref(),
                if lowband_out_valid { Some(&mut lowband_out_tmp2) } else { None },
                None,
                y_cm,
            );

            // Copy lowband_out to norm2
            if lowband_out_valid && norm_off_band + n <= norm_size {
                let copy_n = n.min(lowband_out_tmp2.len());
                norm[norm2_offset + norm_off_band..norm2_offset + norm_off_band + copy_n]
                    .copy_from_slice(&lowband_out_tmp2[..copy_n]);
            }
        } else if have_y {
            // Joint stereo: quant_band_stereo
            let mut lowband_out_tmp = vec![0.0f32; if lowband_out_valid { n } else { 0 }];

            x_cm = quant_band_stereo(
                m, i, ec,
                &mut x[x_off..x_off + n],
                &mut y_storage[x_off..x_off + n],
                n, b, b_short, lm,
                &mut ctx,
                lowband_slice,
                if lowband_out_valid { Some(&mut lowband_out_tmp) } else { None },
                x_cm | y_cm,
            );
            y_cm = x_cm;

            // Copy lowband_out to norm
            if lowband_out_valid && norm_off_band + n <= norm_size {
                let copy_n = n.min(lowband_out_tmp.len());
                norm[norm_off_band..norm_off_band + copy_n].copy_from_slice(&lowband_out_tmp[..copy_n]);
            }
        } else {
            // Mono path
            let mut lowband_out_tmp = vec![0.0f32; if lowband_out_valid { n } else { 0 }];

            x_cm = quant_band(
                m, i, ec,
                &mut x[x_off..x_off + n], n, b, b_short, lm,
                &mut ctx, Q31ONE,
                lowband_slice,
                if lowband_out_valid { Some(&mut lowband_out_tmp) } else { None },
                None,
                x_cm | y_cm,
            );
            y_cm = x_cm;

            // Copy lowband_out back to norm
            if lowband_out_valid && norm_off_band + n <= norm_size {
                let copy_n = n.min(lowband_out_tmp.len());
                norm[norm_off_band..norm_off_band + copy_n].copy_from_slice(&lowband_out_tmp[..copy_n]);
            }
        }

        collapse_masks[i * c] = x_cm as u8;
        collapse_masks[i * c + c - 1] = y_cm as u8;

        bal += pulses[i] + tell;
        update_lowband = b > (n as i32) << BITRES;
        ctx.avoid_split_noise = false;
    }

    *seed = ctx.seed;
}

// =========================================================================
// ENCODER-SIDE FUNCTIONS
// =========================================================================

/// Compute band energies (amplitudes) from frequency-domain signal.
/// Matches C compute_band_energies() float path.
pub fn compute_band_energies(
    m: &CeltMode,
    x: &[f32],
    band_e: &mut [f32],
    eff_end: usize,
    c: usize,
    lm: i32,
) {
    let n = m.short_mdct_size << lm as usize;
    for ch in 0..c {
        for i in 0..eff_end {
            let start = (m.ebands[i] as usize) << lm as usize;
            let end = (m.ebands[i + 1] as usize) << lm as usize;
            let mut sum = 1e-27f32;
            for j in start..end {
                let idx = ch * n + j;
                if idx < x.len() {
                    sum += x[idx] * x[idx];
                }
            }
            band_e[i + ch * m.nb_ebands] = celt_sqrt(sum);
        }
    }
}

/// Normalize each band so that energy is one.
/// Matches C normalise_bands() float path.
pub fn normalise_bands(
    m: &CeltMode,
    freq: &[f32],
    x: &mut [f32],
    band_e: &[f32],
    eff_end: usize,
    c: usize,
    mm: usize,
) {
    let n = mm * m.short_mdct_size;
    for ch in 0..c {
        for i in 0..eff_end {
            let g = 1.0 / (1e-27 + band_e[i + ch * m.nb_ebands]);
            let start = mm * m.ebands[i] as usize;
            let end = mm * m.ebands[i + 1] as usize;
            for j in start..end {
                x[j + ch * n] = freq[j + ch * n] * g;
            }
        }
    }
}

/// Convert band amplitudes to log2 energy representation.
/// Matches C amp2Log2() float path.
pub fn amp2_log2(
    m: &CeltMode,
    eff_end: usize,
    end: usize,
    band_e: &[f32],
    band_log_e: &mut [f32],
    c: usize,
) {
    for ch in 0..c {
        for i in 0..eff_end {
            band_log_e[i + ch * m.nb_ebands] =
                celt_log2(band_e[i + ch * m.nb_ebands].max(1e-30)) - E_MEANS[i];
        }
        for i in eff_end..end {
            band_log_e[ch * m.nb_ebands + i] = -14.0;
        }
    }
}

/// PVQ search: find the best pulse vector for encoding.
/// Matches C op_pvq_search_c() in vq.c (float path).
fn op_pvq_search(x: &[f32], iy: &mut [i32], n: usize, k: usize) -> f32 {
    let mut sum: f32 = 0.0;
    let mut xy: f32 = 0.0;
    let mut yy: f32 = 0.0;

    // Initial quantization: round to nearest integer
    let mut k_left = k as i32;
    for j in 0..n {
        iy[j] = 0;
    }

    // Greedy search: assign pulses one at a time
    // First pass: absolute value rounding
    for j in 0..n {
        sum += x[j].abs();
    }

    if sum < 1e-10 {
        // Zero signal - put all pulses at position 0
        iy[0] = k as i32;
        return k as f32;
    }

    let rcp = (k as f32) / sum;
    for j in 0..n {
        iy[j] = (x[j].abs() * rcp).floor() as i32;
        k_left -= iy[j];
    }

    // Distribute remaining pulses
    while k_left > 0 {
        let mut best_idx = 0usize;
        let mut best_gain = -1e30f32;
        for j in 0..n {
            let x_abs = x[j].abs();
            let curr = iy[j] as f32;
            // Maximize correlation gain: (xy + x_abs) / sqrt(yy + 2*curr + 1) vs xy / sqrt(yy)
            // Simplified: check (xy + x_abs)^2 * yy vs xy^2 * (yy + 2*curr + 1)
            let gain = (2.0 * xy * x_abs + x_abs * x_abs) * yy
                - (2.0 * curr + 1.0) * xy * xy;
            if gain > best_gain || (gain == best_gain && j == 0) {
                best_gain = gain;
                best_idx = j;
            }
        }
        iy[best_idx] += 1;
        xy += x[best_idx].abs();
        yy += 2.0 * (iy[best_idx] - 1) as f32 + 1.0;
        k_left -= 1;
    }

    // Compute ryy and apply signs
    let mut ryy = 0.0f32;
    for j in 0..n {
        if x[j] < 0.0 {
            iy[j] = -iy[j];
        }
        ryy += (iy[j] * iy[j]) as f32;
    }
    ryy
}

/// PVQ encode: quantize and encode a normalized vector.
/// Matches C alg_quant() in vq.c for float encode path.
fn alg_quant(
    x: &mut [f32],
    n: usize,
    k: usize,
    spread: i32,
    b: usize,
    enc: &mut EcCtx,
    gain: f32,
) -> u32 {
    debug_assert!(k > 0);
    debug_assert!(n > 1);

    // Apply spreading rotation (forward direction)
    exp_rotation(x, n, 1, b, k, spread);

    // Search for best pulse vector
    let mut iy = vec![0i32; n];
    let ryy = op_pvq_search(x, &mut iy, n, k);

    // Encode the pulse vector
    enc.encode_pulses(&iy, n, k);

    // Reconstruct the quantized signal
    normalise_residual(&iy, x, n, ryy, gain);

    // Undo the spreading rotation
    exp_rotation(x, n, -1, b, k, spread);

    extract_collapse_mask(&iy, n, b)
}

/// Compute stereo angle itheta from actual X and Y signals.
/// Matches C stereo_itheta() in bands.c.
fn stereo_itheta(x: &[f32], y: &[f32], n: usize, _stereo: bool) -> i32 {
    let mut itheta;
    let e_l = 1e-27 + celt_inner_prod(x, x, n);
    let e_r = 1e-27 + celt_inner_prod(y, y, n);
    let mid = celt_sqrt(e_l);
    let side = celt_sqrt(e_r);
    itheta = (0.5 + 16384.0 * side.atan2(mid) / std::f32::consts::FRAC_PI_2).floor() as i32;
    itheta = itheta.max(0).min(16384);
    itheta
}

/// Encode-side compute_theta: compute and encode the split angle.
fn compute_theta_enc(
    m: &CeltMode,
    band_idx: usize,
    ec: &mut EcCtx,
    x: &[f32],
    y: &[f32],
    n: usize,
    b: &mut i32,
    b0: usize,
    lm: i32,
    ctx: &mut BandCtx,
    stereo: bool,
    fill: &mut u32,
) -> SplitCtx {
    let i = band_idx;
    let intensity = ctx.intensity;

    let pulse_cap = m.log_n[i] as i32 + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1) - if stereo && n == 2 { QTHETA_OFFSET_TWOPHASE } else { QTHETA_OFFSET };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if stereo && (i as i32) >= intensity {
        qn = 1;
    }

    let tell = ec.tell_frac();
    let mut itheta;
    let mut inv = false;

    // Compute itheta from actual signals
    itheta = stereo_itheta(x, y, n, stereo);

    if qn != 1 {
        // Quantize itheta
        itheta = (itheta * qn + 8192) / 16384;

        // Entropy-code the angle
        if stereo && n > 2 {
            let p0: i32 = 3;
            let x0 = qn / 2;
            let ft = (p0 * (x0 + 1) + x0) as u32;
            let fl;
            if itheta <= x0 {
                fl = p0 * itheta;
            } else {
                fl = (itheta - 1 - x0) + (x0 + 1) * p0;
            }
            let fh = if itheta <= x0 { p0 * (itheta + 1) } else { (itheta - x0) + (x0 + 1) * p0 };
            ec.encode(fl as u32, fh as u32, ft);
        } else if b0 > 1 || stereo {
            ec.enc_uint(itheta as u32, (qn + 1) as u32);
        } else {
            // Triangular PDF
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let fl;
            let fh;
            if itheta <= qn >> 1 {
                fl = itheta * (itheta + 1) >> 1;
                fh = fl + itheta + 1;
            } else {
                fl = ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
                fh = fl + (qn + 1 - itheta);
            }
            ec.encode(fl as u32, fh as u32, ft as u32);
        }
        itheta = celt_udiv(itheta * 16384, qn);
    } else if stereo {
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            // Determine inv from actual signal correlation
            let e_l2 = celt_inner_prod(x, x, n);
            let e_r2 = celt_inner_prod(y, y, n);
            inv = e_l2 > 2.0 * e_r2 || e_r2 > 2.0 * e_l2;
            if ctx.disable_inv {
                inv = false;
            }
            ec.enc_bit_logp(inv, 2);
        } else {
            inv = false;
        }
        itheta = 0;
    } else {
        itheta = 0;
    }

    let qalloc = ec.tell_frac() as i32 - tell as i32;
    *b -= qalloc;

    let imid;
    let iside;
    let delta;
    let b_val = b0;
    if itheta == 0 {
        imid = 32767i32;
        iside = 0i32;
        *fill &= (1u32 << b_val) - 1;
        delta = -16384i32;
    } else if itheta == 16384 {
        imid = 0i32;
        iside = 32767i32;
        *fill &= ((1u32 << b_val) - 1) << b_val;
        delta = 16384i32;
    } else {
        imid = bitexact_cos(itheta as i16) as i32;
        iside = bitexact_cos((16384 - itheta) as i16) as i32;
        delta = frac_mul16((n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
    }

    SplitCtx {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
    }
}

/// Encode-side quant_partition.
fn quant_partition_enc(
    m: &CeltMode,
    band_idx: usize,
    ec: &mut EcCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b0: usize,
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

    if lm != -1 && !cache.is_empty() && b > cache[cache[0] as usize] as i32 + 12 && n > 2 {
        // Recursive split (same structure as decoder)
        let half_n = n >> 1;
        let new_lm = lm - 1;
        let new_b0 = (b0 + 1) >> 1;
        let mut new_fill = fill;
        if b0 == 1 {
            new_fill = (fill & 1) | (fill << 1);
        }
        let new_b = new_b0;

        let pulse_cap = m.log_n[band_idx] as i32 + new_lm * (1 << BITRES);
        let offset = (pulse_cap >> 1) - QTHETA_OFFSET;
        let qn = compute_qn(half_n, b, offset, pulse_cap, false);

        // Compute itheta from signal energy split
        let mut e_l = 1e-27f32;
        let mut e_r = 1e-27f32;
        for j in 0..half_n {
            e_l += x[j] * x[j];
            e_r += x[half_n + j] * x[half_n + j];
        }
        let mid_val = celt_sqrt(e_l);
        let side_val = celt_sqrt(e_r);

        let mut itheta;
        let tell = ec.tell_frac();
        if qn != 1 {
            itheta = (0.5 + 16384.0 * side_val.atan2(mid_val) / std::f32::consts::FRAC_PI_2).floor() as i32;
            itheta = itheta.max(0).min(16384);
            itheta = (itheta * qn + 8192) / 16384;

            if b0 > 1 {
                ec.enc_uint(itheta as u32, (qn + 1) as u32);
            } else {
                let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
                let fl;
                let fh;
                if itheta <= qn >> 1 {
                    fl = itheta * (itheta + 1) >> 1;
                    fh = fl + itheta + 1;
                } else {
                    fl = ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
                    fh = fl + (qn + 1 - itheta);
                }
                ec.encode(fl as u32, fh as u32, ft as u32);
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

        let next_lowband2 = lowband.and_then(|lb| {
            if half_n <= lb.len() { Some(&lb[half_n..]) } else { None }
        });

        let rebalance = ctx.remaining_bits;
        let mut final_sbits = sbits;

        if mbits >= sbits {
            let cm_mid = quant_partition_enc(
                m, band_idx, ec, &mut x[..half_n], half_n, mbits, new_b0, new_lm,
                ctx, gain * mid, new_fill, lowband,
            );
            let rebalance_amount = mbits - (rebalance - ctx.remaining_bits);
            if rebalance_amount > 3 << BITRES && itheta != 0 {
                final_sbits = sbits + rebalance_amount - (3 << BITRES);
            }
            let cm_side = quant_partition_enc(
                m, band_idx, ec, &mut x[half_n..], half_n, final_sbits, new_b0, new_lm,
                ctx, gain * side, new_fill >> new_b as u32, next_lowband2,
            );
            cm_mid | (cm_side << (b0 >> 1) as u32)
        } else {
            let cm_side = quant_partition_enc(
                m, band_idx, ec, &mut x[half_n..], half_n, sbits, new_b0, new_lm,
                ctx, gain * side, new_fill >> new_b as u32, next_lowband2,
            );
            let rebalance_amount = sbits - (rebalance - ctx.remaining_bits);
            let mut final_mbits = mbits;
            if rebalance_amount > 3 << BITRES && itheta != 16384 {
                final_mbits = mbits + rebalance_amount - (3 << BITRES);
            }
            let cm_mid = quant_partition_enc(
                m, band_idx, ec, &mut x[..half_n], half_n, final_mbits, new_b0, new_lm,
                ctx, gain * mid, new_fill, lowband,
            );
            cm_mid | (cm_side << (b0 >> 1) as u32)
        }
    } else {
        // Base case: PVQ encode
        let q = bits2pulses(m, band_idx, lm, b);
        let curr_bits = pulses2bits(m, band_idx, lm, q);
        ctx.remaining_bits -= curr_bits;
        let mut q = q;
        let mut curr_bits = curr_bits;

        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(m, band_idx, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q) as usize;
            alg_quant(x, n, k, ctx.spread, b0, ec, gain)
        } else {
            // No pulses: encode noise/folding
            let cm_mask = if b0 < 32 { (1u32 << b0) - 1 } else { 0xFFFFFFFF };
            let fill = fill & cm_mask;
            if fill == 0 {
                for j in 0..n {
                    x[j] = 0.0;
                }
            } else {
                if let Some(lb) = lowband {
                    for j in 0..n {
                        ctx.seed = celt_lcg_rand(ctx.seed);
                        let tmp: f32 = 1.0 / 256.0;
                        let tmp = if ctx.seed & 0x8000 != 0 { tmp } else { -tmp };
                        x[j] = lb[j.min(lb.len() - 1)] + tmp;
                    }
                } else {
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

/// Encode-side quant_band.
fn quant_band_enc(
    m: &CeltMode,
    band_idx: usize,
    ec: &mut EcCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: usize,
    lm: i32,
    ctx: &mut BandCtx,
    gain: f32,
    lowband: Option<&[f32]>,
    lowband_out: Option<&mut [f32]>,
    fill: u32,
) -> u32 {
    let n0 = n;
    let mut n_b = celt_udiv(n as i32, b_blocks as i32) as usize;
    let mut b_cur = b_blocks;
    let b0 = b_blocks;
    let mut time_divide = 0usize;
    let mut recombine = 0usize;
    let long_blocks = b0 == 1;
    let tf_change = ctx.tf_change;
    let mut fill = fill;

    if n == 1 {
        let sign = if x[0] < 0.0 { 1u32 } else { 0u32 };
        if ctx.remaining_bits >= 1 << BITRES {
            ec.enc_bits(sign, 1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        x[0] = if sign != 0 { -NORM_SCALING } else { NORM_SCALING };
        if let Some(lo) = lowband_out {
            lo[0] = x[0];
        }
        return 1;
    }

    if tf_change > 0 {
        recombine = tf_change as usize;
    }

    let mut lowband_copy: Option<Vec<f32>> = None;
    if let Some(lb) = lowband {
        if recombine > 0 || ((n_b & 1) == 0 && tf_change < 0) || b0 > 1 {
            lowband_copy = Some(lb[..n.min(lb.len())].to_vec());
        }
    }

    for k in 0..recombine {
        // Encoder: apply haar on X too (before encoding)
        if n >> k >= 2 {
            haar1(x, n >> k, 1 << k);
        }
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

    let mut tc = tf_change;
    while (n_b & 1) == 0 && tc < 0 {
        // Encoder: apply haar on X
        haar1(x, n_b, b_cur);
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

    if b0_final > 1 {
        // Encoder: deinterleave X
        deinterleave_hadamard(x, n_b >> recombine, b0_final << recombine, long_blocks);
        if let Some(ref mut lb) = lowband_copy {
            deinterleave_hadamard(lb, n_b >> recombine, b0_final << recombine, long_blocks);
        }
    }

    let lb_ref = lowband_copy.as_deref();
    let mut cm = quant_partition_enc(
        m, band_idx, ec, x, n, b, b0_final, lm, ctx, gain, fill, lb_ref,
    );

    if b0_final > 1 {
        interleave_hadamard(x, n_b >> recombine, b0_final << recombine, long_blocks);
    }

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
    let b_final = (b_blocks >> recombine) << recombine;

    if let Some(lo) = lowband_out {
        let n_scale = celt_sqrt(n0 as f32);
        for j in 0..n0.min(lo.len()) {
            lo[j] = n_scale * x[j];
        }
    }
    cm & ((1u32 << b_final) - 1)
}

/// Encode-side quant_band_stereo.
fn quant_band_stereo_enc(
    m: &CeltMode,
    band_idx: usize,
    ec: &mut EcCtx,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    mut b: i32,
    b_blocks: usize,
    lm: i32,
    ctx: &mut BandCtx,
    lowband: Option<&[f32]>,
    lowband_out: Option<&mut [f32]>,
    fill: u32,
) -> u32 {
    let mut cm: u32;

    if n == 1 {
        let sign_x = if x[0] < 0.0 { 1u32 } else { 0u32 };
        if ctx.remaining_bits >= 1 << BITRES {
            ec.enc_bits(sign_x, 1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        x[0] = if sign_x != 0 { -NORM_SCALING } else { NORM_SCALING };

        let sign_y = if y[0] < 0.0 { 1u32 } else { 0u32 };
        if ctx.remaining_bits >= 1 << BITRES {
            ec.enc_bits(sign_y, 1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        y[0] = if sign_y != 0 { -NORM_SCALING } else { NORM_SCALING };
        if let Some(lo) = lowband_out {
            lo[0] = x[0] * 0.0625;
        }
        return 1;
    }

    let orig_fill = fill;
    let mut fill = fill;

    // Apply stereo split before computing theta
    stereo_split(x, y, n);

    let sctx = compute_theta_enc(
        m, band_idx, ec, x, y, n, &mut b, b_blocks, lm, ctx, true, &mut fill,
    );
    let inv = sctx.inv;
    let imid = sctx.imid;
    let iside = sctx.iside;
    let delta = sctx.delta;
    let itheta = sctx.itheta;
    let qalloc = sctx.qalloc;

    let mid = imid as f32 / 32768.0;
    let side = iside as f32 / 32768.0;

    if n == 2 {
        let mbits;
        let mut sbits = 0i32;
        if itheta != 0 && itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits = b - sbits;
        let c = itheta > 8192;
        ctx.remaining_bits -= qalloc + sbits;

        // Determine sign from original signal
        let mut x2;
        let mut y2;
        if c {
            x2 = [y[0], y[1]];
            y2 = [x[0], x[1]];
        } else {
            x2 = [x[0], x[1]];
            y2 = [y[0], y[1]];
        }

        // Encode x2
        cm = quant_band_enc(
            m, band_idx, ec,
            &mut x2, n, mbits, b_blocks, lm,
            ctx, Q31ONE,
            lowband,
            None,
            orig_fill,
        );

        // Determine sign for y2
        let cross = x2[0] * y2[1] - x2[1] * y2[0];
        let sign = if cross < 0.0 { 1i32 } else { 0i32 };
        if sbits != 0 {
            ec.enc_bits(sign as u32, 1);
        }
        let sign_val = 1 - 2 * sign;
        y2[0] = -(sign_val as f32) * x2[1];
        y2[1] = (sign_val as f32) * x2[0];

        if c {
            y[0] = x2[0]; y[1] = x2[1];
            x[0] = y2[0]; x[1] = y2[1];
        } else {
            x[0] = x2[0]; x[1] = x2[1];
            y[0] = y2[0]; y[1] = y2[1];
        }

        let mx0 = mid * x[0];
        let mx1 = mid * x[1];
        let sy0 = side * y[0];
        let sy1 = side * y[1];
        x[0] = mx0 - sy0;
        y[0] = mx0 + sy0;
        x[1] = mx1 - sy1;
        y[1] = mx1 + sy1;
    } else {
        let mbits = 0i32.max(b.min((b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        let rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            let mut lowband_out_tmp: Option<Vec<f32>> = lowband_out.as_ref().map(|_| vec![0.0f32; n]);
            cm = quant_band_enc(
                m, band_idx, ec,
                x, n, mbits, b_blocks, lm,
                ctx, Q31ONE,
                lowband,
                lowband_out_tmp.as_deref_mut(),
                fill,
            );
            if let (Some(tmp), Some(lo)) = (&lowband_out_tmp, lowband_out) {
                let copy_n = n.min(tmp.len()).min(lo.len());
                lo[..copy_n].copy_from_slice(&tmp[..copy_n]);
            }

            let rebalance_amount = mbits - (rebalance - ctx.remaining_bits);
            if rebalance_amount > 3 << BITRES && itheta != 0 {
                sbits += rebalance_amount - (3 << BITRES);
            }
            cm |= quant_band_enc(
                m, band_idx, ec,
                y, n, sbits, b_blocks, lm,
                ctx, side,
                None,
                None,
                fill >> b_blocks,
            );
        } else {
            cm = quant_band_enc(
                m, band_idx, ec,
                y, n, sbits, b_blocks, lm,
                ctx, side,
                None,
                None,
                fill >> b_blocks,
            );
            let rebalance_amount = sbits - (rebalance - ctx.remaining_bits);
            let mut final_mbits = mbits;
            if rebalance_amount > 3 << BITRES && itheta != 16384 {
                final_mbits = mbits + rebalance_amount - (3 << BITRES);
            }
            let mut lowband_out_tmp: Option<Vec<f32>> = lowband_out.as_ref().map(|_| vec![0.0f32; n]);
            cm |= quant_band_enc(
                m, band_idx, ec,
                x, n, final_mbits, b_blocks, lm,
                ctx, Q31ONE,
                lowband,
                lowband_out_tmp.as_deref_mut(),
                fill,
            );
            if let (Some(tmp), Some(lo)) = (&lowband_out_tmp, lowband_out) {
                let copy_n = n.min(tmp.len()).min(lo.len());
                lo[..copy_n].copy_from_slice(&tmp[..copy_n]);
            }
        }

        stereo_merge(x, y, mid, n);

        if inv {
            for j in 0..n {
                y[j] = -y[j];
            }
        }
    }

    cm
}

/// Encode-side quant_all_bands.
pub fn quant_all_bands_enc(
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    collapse_masks: &mut [u8],
    _band_e: &[f32],
    pulses: &mut [i32],
    short_blocks: i32,
    spread: i32,
    dual_stereo: i32,
    intensity: i32,
    tf_res: &[i32],
    total_bits: i32,
    balance: i32,
    ec: &mut EcCtx,
    lm: i32,
    coded_bands: usize,
    seed: &mut u32,
    disable_inv: bool,
) {
    let mm = 1usize << lm;
    let b_short = if short_blocks != 0 { mm } else { 1usize };
    let c = if y.is_some() { 2usize } else { 1usize };
    let norm_offset = mm * m.ebands[start] as usize;

    let norm_end = mm * m.ebands[m.nb_ebands - 1] as usize;
    let norm_size = norm_end - norm_offset;
    let mut norm = vec![0.0f32; (c * norm_size).max(1)];
    let norm2_offset = norm_size;

    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut dual_stereo = dual_stereo != 0;

    let mut ctx = BandCtx {
        spread,
        seed: *seed,
        remaining_bits: total_bits,
        avoid_split_noise: b_short > 1,
        tf_change: 0,
        intensity,
        disable_inv,
    };

    let mut bal = balance;

    let have_y = y.is_some();
    let y_storage: &mut [f32] = match y {
        Some(y_ref) => y_ref,
        None => &mut [],
    };

    for i in start..end {
        let tell = ec.tell_frac() as i32;
        let n = mm * (m.ebands[i + 1] - m.ebands[i]) as usize;
        let x_off = mm * m.ebands[i] as usize;
        let last = i == end - 1;

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

        if (mm * m.ebands[i] as usize >= mm * m.ebands[start] as usize + n || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }

        if i == start + 1 {
            let n1 = mm * (m.ebands[start + 1] - m.ebands[start]) as usize;
            if start + 2 <= m.nb_ebands {
                let n2 = mm * (m.ebands[start + 2] - m.ebands[start + 1]) as usize;
                let copy_len = n2 - n1;
                if copy_len > 0 && 2 * n1 >= n2 {
                    let src_start = 2 * n1 - n2;
                    for j in 0..copy_len {
                        if src_start + j < norm_size && n1 + j < norm_size {
                            norm[n1 + j] = norm[src_start + j];
                        }
                    }
                    if dual_stereo {
                        for j in 0..copy_len {
                            if src_start + j < norm_size && n1 + j < norm_size {
                                norm[norm2_offset + n1 + j] = norm[norm2_offset + src_start + j];
                            }
                        }
                    }
                }
            }
        }

        let tf_change = tf_res[i] as i32;
        ctx.tf_change = tf_change;

        let effective_lowband = if lowband_offset != 0
            && (spread != SPREAD_AGGRESSIVE || b_short > 1 || tf_change < 0)
        {
            let eff = (mm * m.ebands[lowband_offset] as usize).saturating_sub(norm_offset + n);
            eff as i32
        } else {
            -1i32
        };

        let (mut x_cm, mut y_cm);
        if effective_lowband >= 0 {
            let fold_start_bound = effective_lowband as usize + norm_offset;
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if !((mm * m.ebands[fold_start] as usize) > fold_start_bound) {
                    break;
                }
            }
            let mut fold_end = lowband_offset;
            while fold_end < i && (mm * m.ebands[fold_end] as usize) < fold_start_bound + n {
                fold_end += 1;
            }
            x_cm = 0u32;
            y_cm = 0u32;
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

        if dual_stereo && i as i32 == intensity {
            dual_stereo = false;
            let merge_len = (mm * m.ebands[i] as usize).saturating_sub(norm_offset);
            for j in 0..merge_len.min(norm_size) {
                norm[j] = 0.5 * (norm[j] + norm[norm2_offset + j]);
            }
        }

        let lowband_slice = if effective_lowband >= 0 && (effective_lowband as usize) < norm_size {
            Some(&norm[effective_lowband as usize..norm_size])
        } else {
            None
        };

        let lowband2_slice_start = if effective_lowband >= 0 && (effective_lowband as usize) < norm_size {
            Some(norm2_offset + effective_lowband as usize)
        } else {
            None
        };

        let norm_off_band = (mm * m.ebands[i] as usize).saturating_sub(norm_offset);
        let lowband_out_valid = !last && norm_off_band + n <= norm_size;

        if dual_stereo {
            let mut lowband_out_tmp = vec![0.0f32; if lowband_out_valid { n } else { 0 }];

            x_cm = quant_band_enc(
                m, i, ec,
                &mut x[x_off..x_off + n], n, b / 2, b_short, lm,
                &mut ctx, Q31ONE,
                lowband_slice,
                if lowband_out_valid { Some(&mut lowband_out_tmp) } else { None },
                x_cm,
            );

            if lowband_out_valid && norm_off_band + n <= norm_size {
                let copy_n = n.min(lowband_out_tmp.len());
                norm[norm_off_band..norm_off_band + copy_n].copy_from_slice(&lowband_out_tmp[..copy_n]);
            }

            let lowband2_slice = if let Some(start_idx) = lowband2_slice_start {
                if start_idx < norm.len() {
                    Some(norm[start_idx..norm.len().min(start_idx + n * 4)].to_vec())
                } else {
                    None
                }
            } else {
                None
            };

            let mut lowband_out_tmp2 = vec![0.0f32; if lowband_out_valid { n } else { 0 }];

            y_cm = quant_band_enc(
                m, i, ec,
                &mut y_storage[x_off..x_off + n], n, b / 2, b_short, lm,
                &mut ctx, Q31ONE,
                lowband2_slice.as_deref(),
                if lowband_out_valid { Some(&mut lowband_out_tmp2) } else { None },
                y_cm,
            );

            if lowband_out_valid && norm_off_band + n <= norm_size {
                let copy_n = n.min(lowband_out_tmp2.len());
                norm[norm2_offset + norm_off_band..norm2_offset + norm_off_band + copy_n]
                    .copy_from_slice(&lowband_out_tmp2[..copy_n]);
            }
        } else if have_y {
            let mut lowband_out_tmp = vec![0.0f32; if lowband_out_valid { n } else { 0 }];

            x_cm = quant_band_stereo_enc(
                m, i, ec,
                &mut x[x_off..x_off + n],
                &mut y_storage[x_off..x_off + n],
                n, b, b_short, lm,
                &mut ctx,
                lowband_slice,
                if lowband_out_valid { Some(&mut lowband_out_tmp) } else { None },
                x_cm | y_cm,
            );
            y_cm = x_cm;

            if lowband_out_valid && norm_off_band + n <= norm_size {
                let copy_n = n.min(lowband_out_tmp.len());
                norm[norm_off_band..norm_off_band + copy_n].copy_from_slice(&lowband_out_tmp[..copy_n]);
            }
        } else {
            let mut lowband_out_tmp = vec![0.0f32; if lowband_out_valid { n } else { 0 }];

            x_cm = quant_band_enc(
                m, i, ec,
                &mut x[x_off..x_off + n], n, b, b_short, lm,
                &mut ctx, Q31ONE,
                lowband_slice,
                if lowband_out_valid { Some(&mut lowband_out_tmp) } else { None },
                x_cm | y_cm,
            );
            y_cm = x_cm;

            if lowband_out_valid && norm_off_band + n <= norm_size {
                let copy_n = n.min(lowband_out_tmp.len());
                norm[norm_off_band..norm_off_band + copy_n].copy_from_slice(&lowband_out_tmp[..copy_n]);
            }
        }

        collapse_masks[i * c] = x_cm as u8;
        collapse_masks[i * c + c - 1] = y_cm as u8;

        bal += pulses[i] + tell;
        update_lowband = b > (n as i32) << BITRES;
        ctx.avoid_split_noise = false;
    }

    *seed = ctx.seed;
}
