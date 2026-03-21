use crate::tables::*;

/// Comb filter for postfilter.
pub fn comb_filter(
    y: &mut [f32],
    x: &[f32],
    y_offset: usize,
    x_offset: usize,
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
        for i in 0..n {
            y[y_offset + i] = x[x_offset + i];
        }
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

    let actual_overlap = if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        0
    } else {
        overlap
    };

    for i in 0..actual_overlap {
        let f = window[i] * window[i];
        let xi = x[x_offset + i];
        let mut val = xi;
        // Old filter contribution (1-f)*g0
        if x_offset + i >= t0 {
            val += (1.0 - f) * g00 * x[x_offset + i - t0];
            if x_offset + i >= t0 + 1 && x_offset + i + 1 >= t0 {
                val += (1.0 - f) * g01 * (safe_idx(x, x_offset + i - t0 + 1) + safe_idx(x, x_offset + i - t0 - 1));
                val += (1.0 - f) * g02 * (safe_idx(x, x_offset + i - t0 + 2) + safe_idx(x, x_offset + i - t0 - 2));
            }
        }
        // New filter contribution f*g1
        if x_offset + i >= t1 {
            val += f * g10 * x[x_offset + i - t1];
            if x_offset + i >= t1 + 1 {
                val += f * g11 * (safe_idx(x, x_offset + i - t1 + 1) + safe_idx(x, x_offset + i - t1 - 1));
                val += f * g12 * (safe_idx(x, x_offset + i - t1 + 2) + safe_idx(x, x_offset + i - t1 - 2));
            }
        }
        y[y_offset + i] = val.clamp(-SIG_SAT, SIG_SAT);
    }

    if g1 == 0.0 {
        for i in actual_overlap..n {
            y[y_offset + i] = x[x_offset + i];
        }
        return;
    }

    // Constant filter part
    for i in actual_overlap..n {
        let xi = x[x_offset + i];
        let mut val = xi;
        if x_offset + i >= t1 {
            val += g10 * x[x_offset + i - t1];
            val += g11 * (safe_idx(x, x_offset + i - t1 + 1) + safe_idx(x, x_offset + i - t1 - 1));
            val += g12 * (safe_idx(x, x_offset + i - t1 + 2) + safe_idx(x, x_offset + i - t1 - 2));
        }
        y[y_offset + i] = val.clamp(-SIG_SAT, SIG_SAT);
    }
}

#[inline]
fn safe_idx(x: &[f32], idx: usize) -> f32 {
    if idx < x.len() {
        x[idx]
    } else {
        0.0
    }
}
