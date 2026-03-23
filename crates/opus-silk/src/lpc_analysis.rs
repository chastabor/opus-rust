// Port of SILK LPC analysis functions for the encoder:
// - Autocorrelation
// - Levinson-Durbin recursion
// - A2NLSF (LPC to NLSF conversion)
// - NLSF VQ weights (Laroia method)
//
// Ported from silk/A2NLSF.c, silk/NLSF_VQ_weights_laroia.c

use crate::*;
use crate::tables::*;

// Constants from silk/define.h
const SILK_MAX_ORDER_LPC: usize = 16;
const BIN_DIV_STEPS_A2NLSF_FIX: i32 = 3;
const MAX_ITERATIONS_A2NLSF_FIX: i32 = 16;

// NLSF_W_Q = 2 (from silk/define.h)
const NLSF_W_Q: i32 = 2;

/// Compute autocorrelation of input signal.
///
/// `results[k] = sum_i(input[i] * input[i-k])` for each lag k = 0..correlation_count.
/// Uses i64 accumulator, then right-shifts to fit i32.
pub fn silk_autocorrelation(
    results: &mut [i32],
    input: &[i16],
    input_len: usize,
    correlation_count: usize,
) {
    // Compute energy (lag 0) first to determine shift
    let mut nrg: i64 = 0;
    for i in 0..input_len {
        nrg += (input[i] as i64) * (input[i] as i64);
    }

    // Determine right-shift needed to fit in i32
    let shift = if nrg > i32::MAX as i64 {
        let mut s = 0;
        let mut tmp = nrg;
        while tmp > i32::MAX as i64 {
            tmp >>= 1;
            s += 1;
        }
        s
    } else {
        0i32
    };

    results[0] = (nrg >> shift) as i32;

    for k in 1..=correlation_count {
        let mut acc: i64 = 0;
        for i in k..input_len {
            acc += (input[i] as i64) * (input[i - k] as i64);
        }
        results[k] = (acc >> shift) as i32;
    }
}

/// Levinson-Durbin recursion: convert autocorrelation to LPC coefficients.
/// Returns prediction gain in Q24.
///
/// Standard Levinson-Durbin with Q24 internal precision for reflection coefficients.
pub fn silk_levinson_durbin(
    a_q16: &mut [i32],
    corr: &[i32],
    order: usize,
) -> i32 {
    // Early exit if energy is zero
    if corr[0] == 0 {
        for i in 0..order {
            a_q16[i] = 0;
        }
        return 0;
    }

    // Internal buffers
    let mut a_new = [0i64; SILK_MAX_ORDER_LPC];
    let mut a_old = [0i64; SILK_MAX_ORDER_LPC];

    // Q24 scaling factor
    let mut error_q24: i64 = (corr[0] as i64) << 24;

    for m in 0..order {
        // Compute reflection coefficient: k = -sum(a[j]*corr[m+1-j]) / error
        let mut acc: i64 = (corr[m + 1] as i64) << 24;
        for j in 0..m {
            acc += a_old[j] * (corr[m - j] as i64);
        }

        if error_q24 == 0 {
            for i in 0..order {
                a_q16[i] = 0;
            }
            return 0;
        }

        let rc_q24 = -(acc / (error_q24 >> 24)).clamp(-(1i64 << 24) + 1, (1i64 << 24) - 1);

        // Update LPC coefficients
        for j in 0..m {
            a_new[j] = a_old[j] + ((rc_q24 * a_old[m - 1 - j]) >> 24);
        }
        a_new[m] = rc_q24;

        // Copy new to old
        a_old[..=m].copy_from_slice(&a_new[..=m]);

        // Update error: error *= (1 - k^2)
        // Use i128 to avoid overflow in intermediate products
        let rc_sq = ((rc_q24 as i128 * rc_q24 as i128) >> 24) as i64;
        error_q24 -= ((error_q24 as i128 * rc_sq as i128) >> 24) as i64;

        if error_q24 <= 0 {
            for i in 0..order {
                a_q16[i] = 0;
            }
            return 0;
        }
    }

    // Convert from Q24 to Q16
    for i in 0..order {
        a_q16[i] = (a_old[i] >> 8) as i32;
    }

    // Compute prediction gain: corr[0] / error (in Q24)
    let gain_q24 = ((corr[0] as i64) << 24) / (error_q24 >> 24).max(1);
    gain_q24.clamp(0, i32::MAX as i64) as i32
}

// ---- A2NLSF: Convert LPC coefficients to NLSF representation ----

/// Transform polynomials from cos(n*f) to cos(f)^n (matching silk_A2NLSF_trans_poly)
fn silk_a2nlsf_trans_poly(p: &mut [i32], dd: usize) {
    for k in 2..=dd {
        for n in (k + 1..=dd).rev() {
            p[n - 2] -= p[n];
        }
        p[k - 2] -= p[k] << 1;
    }
}

/// Polynomial evaluation using Horner's method (matching silk_A2NLSF_eval_poly).
/// Returns the polynomial evaluation in Q16.
fn silk_a2nlsf_eval_poly(p: &[i32], x: i32, dd: usize) -> i32 {
    let mut y32 = p[dd]; // Q16
    let x_q16 = x << 4;  // Q12 -> Q16

    for n in (0..dd).rev() {
        // silk_SMLAWW(p[n], y32, x_Q16) = p[n] + ((y32 as i64 * x_Q16 as i64) >> 16)
        y32 = (p[n] as i64 + ((y32 as i64 * x_q16 as i64) >> 16)) as i32;
    }
    y32
}

/// Initialize P and Q polynomials from LPC coefficients (matching silk_A2NLSF_init)
fn silk_a2nlsf_init(a_q16: &[i32], p: &mut [i32], q: &mut [i32], dd: usize) {
    p[dd] = 1 << 16;
    q[dd] = 1 << 16;
    for k in 0..dd {
        p[k] = -a_q16[dd - k - 1] - a_q16[dd + k]; // Q16
        q[k] = -a_q16[dd - k - 1] + a_q16[dd + k]; // Q16
    }

    // Divide out zeros
    for k in (1..=dd).rev() {
        p[k - 1] -= p[k];
        q[k - 1] += q[k];
    }

    // Transform polynomials from cos(n*f) to cos(f)^n
    silk_a2nlsf_trans_poly(p, dd);
    silk_a2nlsf_trans_poly(q, dd);
}

/// Convert LPC coefficients (Q16) to NLSF representation (Q15).
///
/// Port of silk_A2NLSF from silk/A2NLSF.c. The algorithm:
/// 1. Create symmetric and antisymmetric polynomials P and Q from the LPC polynomial
/// 2. Evaluate P and Q at frequencies along the unit circle using Chebyshev recursion
/// 3. Find zero crossings (sign changes) -- these are the NLSF frequencies
/// 4. Refine each root with bisection
pub fn silk_a2nlsf(
    nlsf_q15: &mut [i16],
    a_q16: &mut [i32],
    order: usize,
) {
    let dd = order >> 1;
    let mut p = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];
    let mut q = [0i32; SILK_MAX_ORDER_LPC / 2 + 1];

    silk_a2nlsf_init(a_q16, &mut p, &mut q, dd);

    // Find roots, alternating between P and Q
    let mut xlo = SILK_LSF_COS_TAB_FIX_Q12[0] as i32; // Q12
    let mut ylo = silk_a2nlsf_eval_poly(&p, xlo, dd);

    let mut root_ix: usize;
    if ylo < 0 {
        nlsf_q15[0] = 0;
        ylo = silk_a2nlsf_eval_poly(&q, xlo, dd);
        root_ix = 1;
    } else {
        root_ix = 0;
    }

    let mut k: usize = 1;
    let mut i: i32 = 0;
    let mut thr: i32 = 0;

    loop {
        // Evaluate polynomial
        let xhi = SILK_LSF_COS_TAB_FIX_Q12[k] as i32; // Q12
        let yhi = if (root_ix & 1) != 0 {
            silk_a2nlsf_eval_poly(&q, xhi, dd)
        } else {
            silk_a2nlsf_eval_poly(&p, xhi, dd)
        };

        // Detect zero crossing
        if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
            if yhi == 0 {
                thr = 1;
            } else {
                thr = 0;
            }

            // Binary division
            let mut ffrac: i32 = -256;
            let mut xlo_bisect = xlo;
            let mut ylo_bisect = ylo;
            let mut xhi_bisect = xhi;
            let mut yhi_bisect = yhi;

            for m in 0..BIN_DIV_STEPS_A2NLSF_FIX {
                let xmid = silk_rshift_round(xlo_bisect + xhi_bisect, 1);
                let ymid = if (root_ix & 1) != 0 {
                    silk_a2nlsf_eval_poly(&q, xmid, dd)
                } else {
                    silk_a2nlsf_eval_poly(&p, xmid, dd)
                };

                if (ylo_bisect <= 0 && ymid >= 0) || (ylo_bisect >= 0 && ymid <= 0) {
                    xhi_bisect = xmid;
                    yhi_bisect = ymid;
                } else {
                    xlo_bisect = xmid;
                    ylo_bisect = ymid;
                    ffrac += 128 >> m;
                }
            }

            // Interpolate
            if ylo_bisect.abs() < 65536 {
                let den = ylo_bisect - yhi_bisect;
                let nom = (ylo_bisect << (8 - BIN_DIV_STEPS_A2NLSF_FIX)) + (den >> 1);
                if den != 0 {
                    ffrac += silk_div32(nom, den);
                }
            } else {
                // No risk of dividing by zero because abs(ylo - yhi) >= abs(ylo) >= 65536
                let denom = (ylo_bisect - yhi_bisect) >> (8 - BIN_DIV_STEPS_A2NLSF_FIX);
                if denom != 0 {
                    ffrac += silk_div32(ylo_bisect, denom);
                }
            }

            let nlsf_val = ((k as i32) << 8) + ffrac;
            nlsf_q15[root_ix] = nlsf_val.min(i16::MAX as i32) as i16;

            root_ix += 1;
            if root_ix >= order {
                break;
            }

            xlo = SILK_LSF_COS_TAB_FIX_Q12[k - 1] as i32;
            ylo = (1i32 - (root_ix as i32 & 2)) << 12;
        } else {
            k += 1;
            xlo = xhi;
            ylo = yhi;
            thr = 0;

            if k > LSF_COS_TAB_SZ_FIX {
                i += 1;
                if i > MAX_ITERATIONS_A2NLSF_FIX {
                    let step = (1i32 << 15) / (order as i32 + 1);
                    nlsf_q15[0] = step as i16;
                    for idx in 1..order {
                        nlsf_q15[idx] = (nlsf_q15[idx - 1] as i32 + step) as i16;
                    }
                    return;
                }

                nlsf::silk_bwexpander_32(a_q16, order, 65536 - (1 << i));

                silk_a2nlsf_init(a_q16, &mut p, &mut q, dd);

                xlo = SILK_LSF_COS_TAB_FIX_Q12[0] as i32;
                ylo = silk_a2nlsf_eval_poly(&p, xlo, dd);

                if ylo < 0 {
                    nlsf_q15[0] = 0;
                    ylo = silk_a2nlsf_eval_poly(&q, xlo, dd);
                    root_ix = 1;
                } else {
                    root_ix = 0;
                }
                k = 1;
            }
        }
    }
}

/// Compute NLSF weights for quantization (Laroia method).
///
/// Port of silk_NLSF_VQ_weights_laroia from silk/NLSF_VQ_weights_laroia.c.
pub fn silk_nlsf_vq_weights_laroia(
    p_nlsf_w_q_out: &mut [i16],
    p_nlsf_q15: &[i16],
    order: usize,
) {
    debug_assert!(order > 0);
    debug_assert!(order & 1 == 0);

    // First value
    let tmp1_int = {
        let delta = (p_nlsf_q15[0] as i32).max(1);
        (1i32 << (15 + NLSF_W_Q)) / delta
    };
    let mut tmp2_int = {
        let delta = ((p_nlsf_q15[1] as i32) - (p_nlsf_q15[0] as i32)).max(1);
        (1i32 << (15 + NLSF_W_Q)) / delta
    };
    p_nlsf_w_q_out[0] = (tmp1_int + tmp2_int).min(i16::MAX as i32) as i16;

    // Main loop (step by 2)
    let mut k = 1usize;
    while k < order - 1 {
        let tmp1 = {
            let delta = ((p_nlsf_q15[k + 1] as i32) - (p_nlsf_q15[k] as i32)).max(1);
            (1i32 << (15 + NLSF_W_Q)) / delta
        };
        p_nlsf_w_q_out[k] = (tmp1 + tmp2_int).min(i16::MAX as i32) as i16;

        let tmp2 = {
            let delta = ((p_nlsf_q15[k + 2] as i32) - (p_nlsf_q15[k + 1] as i32)).max(1);
            (1i32 << (15 + NLSF_W_Q)) / delta
        };
        p_nlsf_w_q_out[k + 1] = (tmp1 + tmp2).min(i16::MAX as i32) as i16;

        tmp2_int = tmp2;
        k += 2;
    }

    // Last value
    let tmp1_last = {
        let delta = ((1i32 << 15) - (p_nlsf_q15[order - 1] as i32)).max(1);
        (1i32 << (15 + NLSF_W_Q)) / delta
    };
    p_nlsf_w_q_out[order - 1] = (tmp1_last + tmp2_int).min(i16::MAX as i32) as i16;
}
