use crate::lpc::{celt_autocorr, celt_lpc};
use crate::mathops::*;
use crate::tables::*;

/// Comb filter for postfilter.
/// Matches C celt.c comb_filter() for the float case.
/// Operates in-place: y and x point to the same buffer at the same offset.
pub fn comb_filter_inplace(
    buf: &mut [f32],
    offset: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: usize,
    tapset1: usize,
    window: &[f32],
    overlap: usize,
) {
    if g0 == 0.0 && g1 == 0.0 {
        // In-place: nothing to do (y[i] = x[i] and they're the same)
        return;
    }

    let t0 = t0.max(COMBFILTER_MINPERIOD);
    let t1 = t1.max(COMBFILTER_MINPERIOD);

    let g00 = g0 * COMB_FILTER_GAINS[tapset0][0];
    let g01 = g0 * COMB_FILTER_GAINS[tapset0][1];
    let g02 = g0 * COMB_FILTER_GAINS[tapset0][2];
    let g10 = g1 * COMB_FILTER_GAINS[tapset1][0];
    let g11 = g1 * COMB_FILTER_GAINS[tapset1][1];
    let g12 = g1 * COMB_FILTER_GAINS[tapset1][2];

    // Pre-load sliding window for the T1-period filter, matching C:
    // x1 = x[-T1+1], x2 = x[-T1], x3 = x[-T1-1], x4 = x[-T1-2]
    let mut x1 = buf[offset.wrapping_sub(t1) + 1];
    let mut x2 = buf[offset.wrapping_sub(t1)];
    let mut x3 = buf[offset.wrapping_sub(t1).wrapping_sub(1)];
    let mut x4 = buf[offset.wrapping_sub(t1).wrapping_sub(2)];

    let actual_overlap = if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        0
    } else {
        overlap
    };

    for i in 0..actual_overlap {
        let x0 = buf[offset + i - t1 + 2];
        let f = window[i] * window[i];
        let xi = buf[offset + i];
        let val = xi
            + (1.0 - f) * g00 * buf[offset + i - t0]
            + (1.0 - f) * g01 * (buf[offset + i - t0 + 1] + buf[offset + i - t0 - 1])
            + (1.0 - f) * g02 * (buf[offset + i - t0 + 2] + buf[offset + i - t0 - 2])
            + f * g10 * x2
            + f * g11 * (x1 + x3)
            + f * g12 * (x0 + x4);
        buf[offset + i] = val.clamp(-SIG_SAT, SIG_SAT);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }

    if g1 == 0.0 {
        // In-place: nothing to do
        return;
    }

    // Constant filter part (after overlap)
    // C: comb_filter_const(y+overlap, x+overlap, T1, N-overlap, g10, g11, g12)
    for i in actual_overlap..n {
        let xi = buf[offset + i];
        let val = xi
            + g10 * buf[offset + i - t1]
            + g11 * (buf[offset + i - t1 + 1] + buf[offset + i - t1 - 1])
            + g12 * (buf[offset + i - t1 + 2] + buf[offset + i - t1 - 2]);
        buf[offset + i] = val.clamp(-SIG_SAT, SIG_SAT);
    }
}

/// Comb filter with separate input and output buffers, using (buffer, offset) addressing.
/// In C, this is called as `comb_filter(y_ptr, x_ptr, ...)` where x_ptr has history before it.
/// In Rust, we use `(x_buf, x_off)` so that `x_buf[x_off - T1]` is valid (no negative indices).
/// Similarly `(y_buf, y_off)` for the output.
pub fn comb_filter(
    y_buf: &mut [f32],
    y_off: usize,
    x_buf: &[f32],
    x_off: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: usize,
    tapset1: usize,
    window: &[f32],
    overlap: usize,
) {
    let t0 = t0.max(COMBFILTER_MINPERIOD);
    let t1 = t1.max(COMBFILTER_MINPERIOD);

    let g00 = g0 * COMB_FILTER_GAINS[tapset0][0];
    let g01 = g0 * COMB_FILTER_GAINS[tapset0][1];
    let g02 = g0 * COMB_FILTER_GAINS[tapset0][2];
    let g10 = g1 * COMB_FILTER_GAINS[tapset1][0];
    let g11 = g1 * COMB_FILTER_GAINS[tapset1][1];
    let g12 = g1 * COMB_FILTER_GAINS[tapset1][2];

    // Pre-load sliding window: x[off - T1 + 1], x[off - T1], x[off - T1 - 1], x[off - T1 - 2]
    let mut x1 = x_buf[x_off - t1 + 1];
    let mut x2 = x_buf[x_off - t1];
    let mut x3 = x_buf[x_off - t1 - 1];
    let mut x4 = x_buf[x_off - t1 - 2];

    let actual_overlap = if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        0
    } else {
        overlap
    };

    for i in 0..actual_overlap {
        let x0 = x_buf[x_off + i - t1 + 2];
        let f = window[i] * window[i];
        let xi = x_buf[x_off + i];
        let val = xi
            + (1.0 - f) * g00 * x_buf[x_off + i - t0]
            + (1.0 - f) * g01 * (x_buf[x_off + i - t0 + 1] + x_buf[x_off + i - t0 - 1])
            + (1.0 - f) * g02 * (x_buf[x_off + i - t0 + 2] + x_buf[x_off + i - t0 - 2])
            + f * g10 * x2
            + f * g11 * (x1 + x3)
            + f * g12 * (x0 + x4);
        y_buf[y_off + i] = val.clamp(-SIG_SAT, SIG_SAT);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }

    if g1 == 0.0 {
        for i in actual_overlap..n {
            y_buf[y_off + i] = x_buf[x_off + i];
        }
        return;
    }

    for i in actual_overlap..n {
        let xi = x_buf[x_off + i];
        let val = xi
            + g10 * x_buf[x_off + i - t1]
            + g11 * (x_buf[x_off + i - t1 + 1] + x_buf[x_off + i - t1 - 1])
            + g12 * (x_buf[x_off + i - t1 + 2] + x_buf[x_off + i - t1 - 2]);
        y_buf[y_off + i] = val.clamp(-SIG_SAT, SIG_SAT);
    }
}

/// 5-tap FIR filter applied in-place.
fn celt_fir5(x: &mut [f32], num: &[f32; 5], len: usize) {
    let mut mem = [0.0f32; 5];
    for i in 0..len {
        let sum = x[i]
            + num[0] * mem[0]
            + num[1] * mem[1]
            + num[2] * mem[2]
            + num[3] * mem[3]
            + num[4] * mem[4];
        mem[4] = mem[3];
        mem[3] = mem[2];
        mem[2] = mem[1];
        mem[1] = mem[0];
        mem[0] = x[i];
        x[i] = sum;
    }
}

/// Downsample and LPC-filter for pitch analysis.
/// Port of pitch.c pitch_downsample (float path, factor=2).
pub fn pitch_downsample(x: &[&[f32]], x_lp: &mut [f32], len: usize, c: usize) {
    // Downsample by 2 with 3-tap filter
    for i in 1..len {
        x_lp[i] = 0.25 * x[0][2 * i - 1] + 0.5 * x[0][2 * i] + 0.25 * x[0][2 * i + 1];
    }
    x_lp[0] = 0.25 * x[0][1] + 0.5 * x[0][0];
    if c == 2 {
        for i in 1..len {
            x_lp[i] +=
                0.25 * x[1][2 * i - 1] + 0.5 * x[1][2 * i] + 0.25 * x[1][2 * i + 1];
        }
        x_lp[0] += 0.25 * x[1][1] + 0.5 * x[1][0];
    }

    // Compute order-4 autocorrelation
    let mut ac = [0.0f32; 5];
    celt_autocorr(x_lp, &mut ac, &[], 0, 4, len);

    // Noise floor -40 dB
    ac[0] *= 1.0001;

    // Lag windowing
    for i in 1..=4 {
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }

    // LPC analysis (order 4)
    let mut lpc = [0.0f32; 4];
    celt_lpc(&mut lpc, &ac, 4);

    // Bandwidth expansion: multiply each coef by 0.9^(i+1)
    let mut tmp = 1.0f32;
    for i in 0..4 {
        tmp *= 0.9;
        lpc[i] *= tmp;
    }

    // Add a zero to make 5-tap
    let mut lpc2 = [0.0f32; 5];
    lpc2[0] = lpc[0] + 0.8;
    lpc2[1] = lpc[1] + 0.8 * lpc[0];
    lpc2[2] = lpc[2] + 0.8 * lpc[1];
    lpc2[3] = lpc[3] + 0.8 * lpc[2];
    lpc2[4] = 0.8 * lpc[3];

    // Apply FIR filter
    celt_fir5(x_lp, &lpc2, len);
}

/// Cross-correlation for pitch search.
/// Each xcorr[i] = inner_product(x, y[i..], len).
pub fn celt_pitch_xcorr(
    x: &[f32],
    y: &[f32],
    xcorr: &mut [f32],
    len: usize,
    max_pitch: usize,
) {
    for i in 0..max_pitch {
        xcorr[i] = celt_inner_prod(x, &y[i..], len);
    }
}

/// Find top-2 pitch candidates by normalized correlation.
fn find_best_pitch(
    xcorr: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [usize; 2],
) {
    let mut best_num = [-1.0f32; 2];
    let mut best_den = [0.0f32; 2];
    best_pitch[0] = 0;
    best_pitch[1] = 1;

    let mut syy = 1.0f32;
    for j in 0..len {
        syy += y[j] * y[j];
    }

    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let xcorr16 = xcorr[i] * 1e-12;
            let num = xcorr16 * xcorr16;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        syy = syy.max(1.0);
    }
}

/// Multi-resolution pitch search.
/// Finds the best pitch period in the downsampled signal.
pub fn pitch_search(
    x_lp: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    pitch: &mut usize,
) {
    let lag = len + max_pitch;

    // Downsample by 2 again to get 4x decimated signals
    let len4 = len >> 2;
    let lag4 = lag >> 2;
    let max_pitch4 = max_pitch >> 2;
    let len2 = len >> 1;
    let max_pitch2 = max_pitch >> 1;

    let mut x_lp4 = vec![0.0f32; len4];
    let mut y_lp4 = vec![0.0f32; lag4];
    let mut xcorr = vec![0.0f32; max_pitch2];

    for j in 0..len4 {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..lag4 {
        y_lp4[j] = y[2 * j];
    }

    // Coarse search with 4x decimation
    celt_pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr, len4, max_pitch4);

    let mut best_pitch = [0usize; 2];
    find_best_pitch(&xcorr, &y_lp4, len4, max_pitch4, &mut best_pitch);

    // Finer search with 2x decimation
    for i in 0..max_pitch2 {
        xcorr[i] = 0.0;
        if (i as isize - 2 * best_pitch[0] as isize).unsigned_abs() > 2
            && (i as isize - 2 * best_pitch[1] as isize).unsigned_abs() > 2
        {
            continue;
        }
        let sum = celt_inner_prod(&x_lp[..len2], &y[i..i + len2], len2);
        xcorr[i] = sum.max(-1.0);
    }
    find_best_pitch(&xcorr, y, len2, max_pitch2, &mut best_pitch);

    // Refine by pseudo-interpolation
    let offset;
    if best_pitch[0] > 0 && best_pitch[0] < max_pitch2 - 1 {
        let a = xcorr[best_pitch[0] - 1];
        let b = xcorr[best_pitch[0]];
        let c = xcorr[best_pitch[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            offset = 1isize;
        } else if (a - c) > 0.7 * (b - c) {
            offset = -1isize;
        } else {
            offset = 0;
        }
    } else {
        offset = 0;
    }
    *pitch = (2 * best_pitch[0] as isize - offset) as usize;
}

/// Compute pitch gain from cross-correlation and energies.
fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / (1.0 + xx * yy).sqrt()
}

/// Table used by remove_doubling for secondary sub-harmonic checks.
const SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// Remove pitch period doubling errors.
/// Returns the pitch gain.
///
/// `x` is the full buffer; the C code does `x += maxperiod` so that negative
/// indices reach into the past.  We use `base = maxperiod/2` as the origin.
pub fn remove_doubling(
    x: &[f32],
    maxperiod: usize,
    minperiod: usize,
    n: usize,
    t0: &mut usize,
    prev_period: usize,
    prev_gain: f32,
) -> f32 {
    let minperiod0 = minperiod;
    let maxperiod = maxperiod / 2;
    let minperiod = minperiod / 2;
    *t0 /= 2;
    let prev_period = prev_period / 2;
    let n = n / 2;

    // `base` is the index into `x` that corresponds to C's `x` pointer after
    // `x += maxperiod`.  C code accesses x[0..N-1] as x[base..base+N-1] and
    // x[-T] as x[base-T].
    let base = maxperiod;

    if *t0 >= maxperiod {
        *t0 = maxperiod - 1;
    }

    let t0_val = *t0;
    let mut t = t0_val;

    // dual_inner_prod(x, x, x-T0, N, &xx, &xy)
    let mut xx = 0.0f32;
    let mut xy = 0.0f32;
    for j in 0..n {
        xx += x[base + j] * x[base + j];
        xy += x[base + j] * x[base + j - t0_val];
    }

    // Build yy_lookup table
    let mut yy_lookup = vec![0.0f32; maxperiod + 1];
    yy_lookup[0] = xx;
    let mut yy = xx;
    for i in 1..=maxperiod {
        yy = yy + x[base - i] * x[base - i] - x[base + n - i] * x[base + n - i];
        yy_lookup[i] = yy.max(0.0);
    }
    yy = yy_lookup[t0_val];
    let mut best_xy = xy;
    let mut best_yy = yy;
    let g0 = compute_pitch_gain(xy, xx, yy);
    let mut g = g0;

    // Look for any pitch at T/k
    for k in 2..=15 {
        let t1 = (2 * t0_val + k) / (2 * k);
        if t1 < minperiod {
            break;
        }
        // Look for another strong correlation at T1b
        let t1b = if k == 2 {
            if t1 + t0_val > maxperiod {
                t0_val
            } else {
                t0_val + t1
            }
        } else {
            (2 * SECOND_CHECK[k] * t0_val + k) / (2 * k)
        };

        // dual_inner_prod(x, x-T1, x-T1b, N, &xy, &xy2)
        let mut xy_val = 0.0f32;
        let mut xy2 = 0.0f32;
        for j in 0..n {
            xy_val += x[base + j] * x[base + j - t1];
            xy2 += x[base + j] * x[base + j - t1b];
        }
        xy_val = 0.5 * (xy_val + xy2);
        let yy_val = 0.5 * (yy_lookup[t1] + yy_lookup[t1b]);
        let g1 = compute_pitch_gain(xy_val, xx, yy_val);

        let cont;
        if (t1 as isize - prev_period as isize).unsigned_abs() <= 1 {
            cont = prev_gain;
        } else if (t1 as isize - prev_period as isize).unsigned_abs() <= 2
            && 5 * k * k < t0_val
        {
            cont = 0.5 * prev_gain;
        } else {
            cont = 0.0;
        }

        let mut thresh = (0.3f32).max(0.7 * g0 - cont);
        // Bias against very high pitch (very short period)
        if t1 < 3 * minperiod {
            thresh = (0.4f32).max(0.85 * g0 - cont);
        } else if t1 < 2 * minperiod {
            thresh = (0.5f32).max(0.9 * g0 - cont);
        }
        if g1 > thresh {
            best_xy = xy_val;
            best_yy = yy_val;
            t = t1;
            g = g1;
        }
    }

    best_xy = best_xy.max(0.0);
    let mut pg = if best_yy <= best_xy {
        1.0f32
    } else {
        best_xy / (best_yy + 1.0)
    };

    // Refine with interpolation
    let mut xcorr = [0.0f32; 3];
    for k in 0..3usize {
        let lag = t + k - 1;
        xcorr[k] = celt_inner_prod(
            &x[base..base + n],
            &x[base - lag..base - lag + n],
            n,
        );
    }
    let offset;
    if (xcorr[2] - xcorr[0]) > 0.7 * (xcorr[1] - xcorr[0]) {
        offset = 1isize;
    } else if (xcorr[0] - xcorr[2]) > 0.7 * (xcorr[1] - xcorr[2]) {
        offset = -1isize;
    } else {
        offset = 0;
    }

    if pg > g {
        pg = g;
    }
    *t0 = (2 * t as isize + offset) as usize;

    if *t0 < minperiod0 {
        *t0 = minperiod0;
    }
    pg
}

