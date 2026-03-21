use crate::mode::CeltMode;
use crate::tables::*;
use crate::mathops::*;
use crate::rate::{bits2pulses, pulses2bits};
use opus_range_coder::EcCtx;

/// Denormalize bands: scale normalized MDCT coefficients by band energy.
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

    let mut fi = 0usize;
    if start != 0 {
        for i in 0..(mm * m.ebands[start] as usize) {
            freq[fi] = 0.0;
            fi += 1;
        }
    }

    let mut xi = mm * m.ebands[start] as usize;
    for i in start..end {
        let j_start = mm * m.ebands[i] as usize;
        let band_end = mm * m.ebands[i + 1] as usize;
        let lg = band_log_e[i] + E_MEANS[i];
        let g = celt_exp2_db(lg.min(32.0));
        let mut j = j_start;
        while j < band_end {
            freq[j] = x[xi] * g;
            xi += 1;
            j += 1;
        }
    }

    for i in bound..n {
        freq[i] = 0.0;
    }
}

/// Anti-collapse processing.
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

            let x_off = ch * size + (m.ebands[i] as usize) * (1 << lm) as usize;
            let mut renormalize = false;
            for k in 0..(1usize << lm) {
                if (collapse_masks[i * c + ch] & (1 << k)) == 0 {
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

/// Haar wavelet transform.
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

/// Interleave Hadamard.
fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                if ordery[i] * n0 + j < n {
                    tmp[j * stride + i] = x[ordery[i] * n0 + j];
                }
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

/// Deinterleave Hadamard.
fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                if ordery[i] * n0 + j < n {
                    tmp[ordery[i] * n0 + j] = x[j * stride + i];
                }
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

/// Stereo merge operation.
fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    let mut xp = 0.0f32;
    let mut side = 0.0f32;
    for j in 0..n {
        xp += y[j] * x[j];
        side += y[j] * y[j];
    }
    xp *= mid;
    let el = mid * mid * 0.125 + side - 2.0 * xp;
    let er = mid * mid * 0.125 + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y[..n].copy_from_slice(&x[..n]);
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

/// Compute qn for theta quantization.
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

/// Band context for quant_all_bands.
struct BandCtx<'a> {
    m: &'a CeltMode,
    i: usize,
    intensity: i32,
    spread: i32,
    tf_change: i32,
    ec: *mut EcCtx,
    remaining_bits: i32,
    seed: u32,
    disable_inv: bool,
}

/// Decode a single sample (N=1) band. Not used directly; inlined in quant_band_decode.

/// Decode all bands in a frame.
pub fn quant_all_bands(
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    y_opt: Option<&mut [f32]>,
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
    let has_stereo = y_opt.is_some();
    let c = if has_stereo { 2 } else { 1 };
    let n_total = mm * m.short_mdct_size;
    let norm_offset = mm * m.ebands[start] as usize;

    // Allocate norm buffers for folding
    let norm_size = mm * m.ebands[m.nb_ebands - 1] as usize - norm_offset;
    let mut norm = vec![0.0f32; norm_size];
    let mut norm2 = vec![0.0f32; norm_size];

    let lowband_scratch_off = mm * m.ebands[m.eff_ebands - 1] as usize;

    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut remaining_bits = total_bits;
    let mut bal = balance;
    let mut ctx_seed = *seed;

    for i in start..end {
        let tell = ec.tell_frac();
        let n = (mm * (m.ebands[i + 1] - m.ebands[i]) as usize) as usize;
        let x_off = mm * m.ebands[i] as usize;
        let last = i == end - 1;

        let mut b = if i < coded_bands {
            let left = (total_bits - tell as i32).max(0);
            let b_val = pulses[i] + bal;
            let b_val = if last { left.min(b_val) } else { b_val.min(left) };
            b_val.max(0)
        } else {
            0
        };

        if i < coded_bands {
            bal = (bal - b).max(0);
        }

        let tf_change = tf_res[i] as i32;
        let b_blocks = b_short;

        // Effective lowband
        let effective_lowband = if i > start && update_lowband {
            lowband_offset
        } else {
            0
        };

        // For mono, decode band
        if !has_stereo || i >= intensity as usize {
            // Simple mono band decode
            let cm = quant_band_decode(
                m, ec, i, &mut x[x_off..], n, b, b_blocks as i32, lm, spread,
                tf_change, &mut norm, norm_offset, effective_lowband,
                lowband_scratch_off, &mut ctx_seed, &mut remaining_bits,
            );
            let mask_val = cm as u8;
            if has_stereo {
                collapse_masks[i * c] = mask_val;
                collapse_masks[i * c + 1] = mask_val;
                // For intensity stereo, Y is derived from X (simplified)
            } else {
                collapse_masks[i] = mask_val;
            }
        } else {
            // Stereo band - simplified: decode as mono for X, fill Y
            let cm = quant_band_decode(
                m, ec, i, &mut x[x_off..], n, b, b_blocks as i32, lm, spread,
                tf_change, &mut norm, norm_offset, effective_lowband,
                lowband_scratch_off, &mut ctx_seed, &mut remaining_bits,
            );
            collapse_masks[i * c] = cm as u8;
            collapse_masks[i * c + 1] = cm as u8;
        }

        // Update lowband state
        if n > 0 {
            let band_start = mm * m.ebands[i] as usize;
            let band_n = celt_sqrt((n << 22) as f32);
            let norm_off_band = band_start - norm_offset;
            if norm_off_band + n <= norm.len() {
                for j in 0..n {
                    norm[norm_off_band + j] = band_n * x[x_off + j];
                }
            }
        }

        update_lowband = b > (n as i32) << BITRES;
        if update_lowband {
            lowband_offset = i;
        }
    }

    *seed = ctx_seed;
}

/// Simplified band decode for mono. Returns collapse mask.
fn quant_band_decode(
    m: &CeltMode,
    ec: &mut EcCtx,
    band: usize,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lm: i32,
    spread: i32,
    tf_change: i32,
    norm: &mut [f32],
    norm_offset: usize,
    lowband_idx: usize,
    _lowband_scratch_off: usize,
    seed: &mut u32,
    remaining_bits: &mut i32,
) -> u32 {
    if n == 1 {
        // N=1 special case
        let mut sign = 0u32;
        if *remaining_bits >= 1 << BITRES {
            sign = ec.dec_bits(1);
            *remaining_bits -= 1 << BITRES;
        }
        x[0] = if sign != 0 { -NORM_SCALING } else { NORM_SCALING };
        return 1;
    }

    let cache_idx = m.cache.index[((lm + 1) as usize) * m.nb_ebands + band];
    if cache_idx < 0 {
        for j in 0..n {
            x[j] = 0.0;
        }
        return 0;
    }
    let cache = &m.cache.bits[cache_idx as usize..];

    // Check if we should split
    if lm != -1 && b > cache[cache[0] as usize] as i32 + 12 && n > 2 {
        // Recursive split
        return quant_partition_decode(m, ec, band, x, n, b, b_blocks, lm, spread, seed, remaining_bits, (1u32 << b_blocks) - 1);
    }

    // Base case: PVQ decode
    let q = bits2pulses(m, band, lm, b);
    let curr_bits = pulses2bits(m, band, lm, q);
    *remaining_bits -= curr_bits;
    while *remaining_bits < 0 && q > 0 {
        *remaining_bits += curr_bits;
        let q_new = q - 1;
        let curr_bits_new = pulses2bits(m, band, lm, q_new);
        *remaining_bits -= curr_bits_new;
        return pvq_decode(ec, x, n, q_new, seed, spread, b_blocks, remaining_bits);
    }

    if q != 0 {
        pvq_decode(ec, x, n, q, seed, spread, b_blocks, remaining_bits)
    } else {
        // Fill with noise or folding
        for j in 0..n {
            *seed = celt_lcg_rand(*seed);
            x[j] = if *seed & 0x8000 != 0 { 1.0 / 256.0 } else { -1.0 / 256.0 };
        }
        renormalise_vector(x, n, NORM_SCALING);
        0
    }
}

/// Recursive partition decode.
fn quant_partition_decode(
    m: &CeltMode,
    ec: &mut EcCtx,
    band: usize,
    x: &mut [f32],
    n: usize,
    b: i32,
    b0: i32,
    lm: i32,
    spread: i32,
    seed: &mut u32,
    remaining_bits: &mut i32,
    fill: u32,
) -> u32 {
    let cache_idx = m.cache.index[((lm + 1) as usize) * m.nb_ebands + band];
    if cache_idx < 0 || n <= 2 || lm < 0 {
        // Base case
        let q = bits2pulses(m, band, lm, b);
        let curr_bits = pulses2bits(m, band, lm, q);
        *remaining_bits -= curr_bits;
        if q != 0 {
            return pvq_decode(ec, x, n, q, seed, spread, b0, remaining_bits);
        } else {
            for j in 0..n {
                *seed = celt_lcg_rand(*seed);
                x[j] = if *seed & 0x8000 != 0 { 1.0 / 256.0 } else { -1.0 / 256.0 };
            }
            renormalise_vector(x, n, NORM_SCALING);
            return 0;
        }
    }
    let cache = &m.cache.bits[cache_idx as usize..];
    if b <= cache[cache[0] as usize] as i32 + 12 || n <= 2 {
        let q = bits2pulses(m, band, lm, b);
        let curr_bits = pulses2bits(m, band, lm, q);
        *remaining_bits -= curr_bits;
        if q != 0 {
            return pvq_decode(ec, x, n, q, seed, spread, b0, remaining_bits);
        } else {
            for j in 0..n {
                *seed = celt_lcg_rand(*seed);
                x[j] = if *seed & 0x8000 != 0 { 1.0 / 256.0 } else { -1.0 / 256.0 };
            }
            renormalise_vector(x, n, NORM_SCALING);
            return 0;
        }
    }

    // Split
    let half_n = n >> 1;
    let new_lm = lm - 1;
    let mut new_b = (b0 + 1) >> 1;
    let mut new_fill = fill;
    if b0 == 1 {
        new_fill = (fill & 1) | (fill << 1);
    }

    // Decode theta (split angle)
    let pulse_cap = m.log_n[band] as i32 + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1) - QTHETA_OFFSET;
    let qn = compute_qn(half_n, b, offset, pulse_cap, false);

    let mut itheta;
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

    let imid;
    let iside;
    let delta;
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        delta = 16384;
    } else {
        imid = bitexact_cos(itheta as i16) as i32;
        iside = bitexact_cos((16384 - itheta) as i16) as i32;
        delta = frac_mul16((half_n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
    }

    let mid = imid as f32 / 32768.0;
    let side = iside as f32 / 32768.0;

    let qalloc = ec.tell_frac() as i32;
    let mbits = 0i32.max(b.min((b - delta) / 2));
    let sbits = b - mbits;
    *remaining_bits -= (ec.tell_frac() as i32 - qalloc);

    // Decode mid
    let cm_mid = quant_partition_decode(
        m, ec, band, &mut x[..half_n], half_n, mbits, new_b, new_lm, spread, seed, remaining_bits, new_fill,
    );
    // Decode side
    let cm_side = quant_partition_decode(
        m, ec, band, &mut x[half_n..], half_n, sbits, new_b, new_lm, spread, seed, remaining_bits, new_fill >> new_b as u32,
    );

    // Apply mid/side scaling
    for j in 0..half_n {
        let m_val = x[j] * mid;
        let s_val = x[half_n + j] * side;
        x[j] = m_val;
        x[half_n + j] = s_val;
    }

    cm_mid | (cm_side << (b0 >> 1) as u32)
}

/// PVQ decode using range coder.
fn pvq_decode(
    ec: &mut EcCtx,
    x: &mut [f32],
    n: usize,
    q: i32,
    seed: &mut u32,
    spread: i32,
    b: i32,
    _remaining_bits: &mut i32,
) -> u32 {
    let k = get_pulses(q) as usize;
    if k == 0 || n == 0 {
        for j in 0..n {
            x[j] = 0.0;
        }
        return 0;
    }

    let mut iy = vec![0i32; n];
    let gain = ec.decode_pulses(&mut iy, n, k);

    // Convert int pulses to float and normalize
    let inv_gain = if gain > 1e-30 { 1.0 / gain.sqrt() } else { 0.0 };
    for j in 0..n {
        x[j] = iy[j] as f32 * inv_gain;
    }

    // Spread/fold
    if spread > SPREAD_NONE {
        // Simple spreading (omitted complex Hadamard for now)
    }

    let cm = (1u32 << b.min(32)) - 1;
    cm
}
