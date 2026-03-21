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

