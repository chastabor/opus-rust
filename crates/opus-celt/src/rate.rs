use crate::mode::CeltMode;
use crate::tables::*;
use crate::mathops::celt_udiv;
use opus_range_coder::EcCtx;

/// Look up the number of pulses for a given number of bits.
/// Matches C rate.h bits2pulses() exactly.
pub fn bits2pulses(m: &CeltMode, band: usize, lm: i32, b: i32) -> i32 {
    let cache_idx = m.cache.index[((lm + 1) as usize) * m.nb_ebands + band];
    if cache_idx < 0 {
        return 0;
    }
    let cache = &m.cache.bits[cache_idx as usize..];
    let mut lo = 0i32;
    let mut hi = cache[0] as i32;
    let bits = b - 1; // C does bits-- before the search
    for _ in 0..6 {
        let mid = (lo + hi + 1) >> 1; // C uses (lo+hi+1)>>1
        if cache[mid as usize] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // C: bits- (lo == 0 ? -1 : (int)cache[lo]) <= (int)cache[hi]-bits
    let lo_val = if lo == 0 { -1i32 } else { cache[lo as usize] as i32 };
    if bits - lo_val <= cache[hi as usize] as i32 - bits {
        lo
    } else {
        hi
    }
}

/// Get the number of bits used for a given number of pulses.
pub fn pulses2bits(m: &CeltMode, band: usize, lm: i32, k: i32) -> i32 {
    let cache_idx = m.cache.index[((lm + 1) as usize) * m.nb_ebands + band];
    if cache_idx < 0 {
        return 0;
    }
    let cache = &m.cache.bits[cache_idx as usize..];
    if k == 0 {
        0
    } else {
        (cache[k as usize] as i32) + 1
    }
}

/// Initialize band caps.
pub fn init_caps(m: &CeltMode, cap: &mut [i32], lm: i32, c: i32) {
    for i in 0..m.nb_ebands {
        let n = ((m.ebands[i + 1] - m.ebands[i]) as i32) << lm;
        cap[i] = ((m.cache.caps[m.nb_ebands * (2 * lm as usize + c as usize - 1) + i] as i32)
            + 64)
            * c
            * n
            >> 2;
    }
}

/// Compute bit allocation, returning the number of coded bands.
pub fn clt_compute_allocation(
    m: &CeltMode,
    start: usize,
    end: usize,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut i32,
    total: i32,
    balance: &mut i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    ec: &mut EcCtx,
) -> usize {
    let mut total = total.max(0);
    let len = m.nb_ebands;
    let skip_start = {
        let mut ss = start;
        for j in start..end {
            if offsets[j] > 0 {
                ss = j;
            }
        }
        ss
    };

    // Reserve bits for skip signal
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;

    // Reserve bits for intensity and dual stereo parameters
    let mut intensity_rsv = 0i32;
    let mut dual_stereo_rsv = 0i32;
    if c == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[(end - start).min(23)] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }

    let mut bits1 = vec![0i32; len];
    let mut bits2 = vec![0i32; len];
    let mut thresh = vec![0i32; len];
    let mut trim_offset = vec![0i32; len];

    let log_m = lm << BITRES;
    let stereo = if c > 1 { 1i32 } else { 0i32 };
    let alloc_floor = c << BITRES;

    for j in start..end {
        thresh[j] = (c << BITRES)
            .max((3 * ((m.ebands[j + 1] - m.ebands[j]) as i32) << lm << BITRES) >> 4);
        trim_offset[j] = c
            * (m.ebands[j + 1] - m.ebands[j]) as i32
            * (alloc_trim - 5 - lm)
            * (end as i32 - j as i32 - 1)
            * (1 << (lm + BITRES))
            >> 6;
        if ((m.ebands[j + 1] - m.ebands[j]) as i32) << lm == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }

    // Binary search for allocation vectors
    let mut lo = 1i32;
    let mut hi = m.nb_alloc_vectors as i32 - 1;
    while lo <= hi {
        let mid = (lo + hi) >> 1;
        let mut done = false;
        let mut psum = 0i32;
        for j in (start..end).rev() {
            let n = (m.ebands[j + 1] - m.ebands[j]) as i32;
            let mut bitsj = c * n * (m.alloc_vectors[mid as usize][j] as i32) << lm >> 2;
            if bitsj > 0 {
                bitsj = (bitsj + trim_offset[j]).max(0);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += bitsj.min(cap[j]);
            } else if bitsj >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    hi = lo;
    lo -= 1;

    for j in start..end {
        let n = (m.ebands[j + 1] - m.ebands[j]) as i32;
        let mut bits1j = c * n * (m.alloc_vectors[lo as usize][j] as i32) << lm >> 2;
        let mut bits2j = if hi >= m.nb_alloc_vectors as i32 {
            cap[j]
        } else {
            c * n * (m.alloc_vectors[hi as usize][j] as i32) << lm >> 2
        };
        if bits1j > 0 {
            bits1j = (bits1j + trim_offset[j]).max(0);
        }
        if bits2j > 0 {
            bits2j = (bits2j + trim_offset[j]).max(0);
        }
        if lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        bits2j = (bits2j - bits1j).max(0);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    // Interpolate bits
    let coded_bands = interp_bits2pulses(
        m,
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        balance,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        c,
        lm,
        ec,
    );
    coded_bands
}

fn interp_bits2pulses(
    m: &CeltMode,
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    mut total: i32,
    balance_out: &mut i32,
    skip_rsv: i32,
    intensity: &mut i32,
    mut intensity_rsv: i32,
    dual_stereo: &mut i32,
    dual_stereo_rsv: i32,
    bits: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    ec: &mut EcCtx,
) -> usize {
    let stereo = if c > 1 { 1i32 } else { 0i32 };
    let alloc_floor = c << BITRES;
    let log_m = lm << BITRES;

    // Binary search for interpolation factor
    let mut lo = 0i32;
    let mut hi = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid as i64 * bits2[j] as i64) >> ALLOC_STEPS) as i32;
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let mut psum = 0i32;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo as i64 * bits2[j] as i64) >> ALLOC_STEPS) as i32;
        if tmp < thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // Skip band decision (decoder)
    let mut coded_bands = end;
    loop {
        if coded_bands <= start + 1 {
            total += skip_rsv;
            break;
        }
        let j = coded_bands - 1;
        if j <= skip_start {
            total += skip_rsv;
            break;
        }
        let left = total - psum;
        let band_width = (m.ebands[coded_bands] - m.ebands[start]) as i32;
        let percoeff = if band_width > 0 { celt_udiv(left, band_width) } else { 0 };
        let left2 = left - band_width * percoeff;
        let rem = (left2 - (m.ebands[j] - m.ebands[start]) as i32).max(0);
        let bw = (m.ebands[coded_bands] - m.ebands[j]) as i32;
        let band_bits = bits[j] + percoeff * bw + rem;


        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            let flag = ec.dec_bit_logp(1);
            if flag {
                break;
            }
            psum += 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC_TABLE[(j - start).min(23)] as i32;
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }

    // Code intensity and dual stereo parameters
    if intensity_rsv > 0 {
        *intensity = start as i32 + ec.dec_uint((coded_bands + 1 - start) as u32) as i32;
    } else {
        *intensity = 0;
    }
    if *intensity <= start as i32 {
        total += dual_stereo_rsv;
    }
    let dual_stereo_rsv = if *intensity <= start as i32 { 0 } else { dual_stereo_rsv };
    if dual_stereo_rsv > 0 {
        *dual_stereo = if ec.dec_bit_logp(1) { 1 } else { 0 };
    } else {
        *dual_stereo = 0;
    }

    // Allocate remaining bits
    let left = total - psum;
    let band_width = (m.ebands[coded_bands] - m.ebands[start]) as i32;
    let percoeff = if band_width > 0 { celt_udiv(left, band_width) } else { 0 };
    let mut left = left - band_width * percoeff;
    for j in start..coded_bands {
        bits[j] += percoeff * (m.ebands[j + 1] - m.ebands[j]) as i32;
    }
    for j in start..coded_bands {
        let tmp = left.min((m.ebands[j + 1] - m.ebands[j]) as i32);
        bits[j] += tmp;
        left -= tmp;
    }

    // Compute fine energy bits
    let mut bal = 0i32;
    for j in start..coded_bands {
        let n0 = (m.ebands[j + 1] - m.ebands[j]) as i32;
        let n = n0 << lm;
        let bit = bits[j] + bal;

        if n > 1 {
            let excess = (bit - cap[j]).max(0);
            bits[j] = bit - excess;
            let den = c * n
                + if c == 2 && n > 2 && *dual_stereo == 0 && j < *intensity as usize {
                    1
                } else {
                    0
                };
            let nc_log_n = den * (m.log_n[j] as i32 + log_m);
            let mut offset = (nc_log_n >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << BITRES >> 2;
            }
            if bits[j] + offset < den * 2 << BITRES {
                offset += nc_log_n >> 2;
            } else if bits[j] + offset < den * 3 << BITRES {
                offset += nc_log_n >> 3;
            }

            ebits[j] = ((bits[j] + offset + (den << (BITRES - 1))).max(0) / den) >> BITRES;
            if c * ebits[j] > (bits[j] >> BITRES) {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            fine_priority[j] = if ebits[j] * (den << BITRES) >= bits[j] + offset {
                1
            } else {
                0
            };
            bits[j] -= c * ebits[j] << BITRES;
        } else {
            let excess = (bit - (c << BITRES)).max(0);
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }

        // Re-balance excess
        let mut excess = if n > 1 {
            (bit - cap[j]).max(0)
        } else {
            (bit - (c << BITRES)).max(0)
        };
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = extra_fine * c << BITRES;
            fine_priority[j] = if extra_bits >= excess - bal { 1 } else { 0 };
            excess -= extra_bits;
        }
        bal = excess;
    }
    *balance_out = bal;

    // Skipped bands use all bits for fine energy
    for j in coded_bands..end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        fine_priority[j] = if ebits[j] < 1 { 1 } else { 0 };
    }

    coded_bands
}
