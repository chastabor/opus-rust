// Port of SILK LPC analysis functions for the encoder:
// - Autocorrelation
// - Levinson-Durbin recursion
// - A2NLSF (LPC to NLSF conversion)
// - NLSF VQ weights (Laroia method)
//
// Ported from silk/A2NLSF.c, silk/NLSF_VQ_weights_laroia.c

use crate::tables::*;
use crate::*;

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
    for item in input.iter().take(input_len) {
        nrg += (*item as i64) * (*item as i64);
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
pub fn silk_levinson_durbin(a_q16: &mut [i32], corr: &[i32], order: usize) -> i32 {
    silk_levinson_durbin_constrained(a_q16, corr, order, 0)
}

/// Levinson-Durbin with prediction gain constraint.
/// `min_inv_gain_q30`: minimum inverse gain (0 = no constraint).
/// When the prediction gain would exceed `1/min_inv_gain`, the recursion
/// stops and remaining coefficients are zeroed. This matches the C reference's
/// Burg modified algorithm's gain limiting behavior.
pub fn silk_levinson_durbin_constrained(
    a_q16: &mut [i32],
    corr: &[i32],
    order: usize,
    min_inv_gain_q30: i32,
) -> i32 {
    // Early exit if energy is zero
    if corr[0] == 0 {
        for item in a_q16.iter_mut().take(order) {
            *item = 0;
        }
        return 0;
    }

    // Internal buffers
    let mut a_new = [0i64; SILK_MAX_ORDER_LPC];
    let mut a_old = [0i64; SILK_MAX_ORDER_LPC];

    // Q24 scaling factor
    let mut error_q24: i64 = (corr[0] as i64) << 24;
    let corr0_q24 = error_q24;
    let mut final_order = order;

    for m in 0..order {
        // Compute reflection coefficient: k = -sum(a[j]*corr[m+1-j]) / error
        let mut acc: i64 = (corr[m + 1] as i64) << 24;
        for j in 0..m {
            acc += a_old[j] * (corr[m - j] as i64);
        }

        if error_q24 == 0 {
            for item in a_q16.iter_mut().take(order) {
                *item = 0;
            }
            return 0;
        }

        let rc_q24 = -(acc / (error_q24 >> 24)).clamp(-(1i64 << 24) + 1, (1i64 << 24) - 1);

        // Update error: error *= (1 - k^2)
        let rc_sq = ((rc_q24 as i128 * rc_q24 as i128) >> 24) as i64;
        let new_error = error_q24 - ((error_q24 as i128 * rc_sq as i128) >> 24) as i64;

        // Check prediction gain constraint (C: burg_modified_FIX.c line 202)
        // inv_gain = error / corr[0]. If inv_gain < min_inv_gain, stop.
        if min_inv_gain_q30 > 0 && new_error > 0 && corr0_q24 > 0 {
            // inv_gain_q30 ≈ (new_error / corr0) << 30
            // To avoid overflow: inv_gain_q30 = new_error / (corr0 >> 30)
            let inv_gain_approx = if corr0_q24 >> 30 > 0 {
                (new_error / (corr0_q24 >> 30)) as i32
            } else {
                i32::MAX
            };
            if inv_gain_approx <= min_inv_gain_q30 {
                // Prediction gain limit reached. Zero remaining coefficients.
                final_order = m + 1;
                // Still apply this coefficient but with no further recursion
                for j in 0..m {
                    a_new[j] = a_old[j] + ((rc_q24 * a_old[m - 1 - j]) >> 24);
                }
                a_new[m] = rc_q24;
                a_old[..=m].copy_from_slice(&a_new[..=m]);
                error_q24 = new_error;
                break;
            }
        }

        // Update LPC coefficients
        for j in 0..m {
            a_new[j] = a_old[j] + ((rc_q24 * a_old[m - 1 - j]) >> 24);
        }
        a_new[m] = rc_q24;

        // Copy new to old
        a_old[..=m].copy_from_slice(&a_new[..=m]);

        error_q24 = new_error;

        if error_q24 <= 0 {
            for item in a_q16.iter_mut().take(order) {
                *item = 0;
            }
            return 0;
        }
    }

    // Convert from Q24 to Q16
    for i in 0..final_order {
        a_q16[i] = (a_old[i] >> 8) as i32;
    }
    // Zero remaining coefficients
    for item in a_q16.iter_mut().take(order).skip(final_order) {
        *item = 0;
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
    let x_q16 = x << 4; // Q12 -> Q16

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
pub fn silk_a2nlsf(nlsf_q15: &mut [i16], a_q16: &mut [i32], order: usize) {
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
pub fn silk_nlsf_vq_weights_laroia(p_nlsf_w_q_out: &mut [i16], p_nlsf_q15: &[i16], order: usize) {
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

// ---- Burg's Modified Algorithm (port of silk/fixed/burg_modified_FIX.c) ----

const QA: i32 = 25;
const N_BITS_HEAD_ROOM: i32 = 3;
const MIN_RSHIFTS: i32 = -16;
const MAX_RSHIFTS: i32 = 32 - QA; // = 7
// FIND_LPC_COND_FAC = 1e-5f → in Q32: (1e-5 * 2^32) ≈ 42949
const FIND_LPC_COND_FAC_Q32: i32 = 42949;

/// Burg's modified algorithm for LPC analysis with prediction gain constraint.
/// Port of silk/fixed/burg_modified_FIX.c.
///
/// Computes LPC coefficients and residual energy from input signal,
/// enforcing a maximum prediction gain via `min_inv_gain_q30`.
pub fn silk_burg_modified(
    res_nrg: &mut i32,     // O: Residual energy
    res_nrg_q: &mut i32,   // O: Residual energy Q value
    a_q16: &mut [i32],     // O: LPC coefficients (length D)
    x: &[i16],             // I: Input signal, length nb_subfr * subfr_length
    min_inv_gain_q30: i32, // I: Inverse of max prediction gain
    subfr_length: usize,   // I: Subframe length (including D preceding samples)
    nb_subfr: usize,       // I: Number of subframes
    d: usize,              // I: LPC order
) {
    let mut c_first_row = [0i32; SILK_MAX_ORDER_LPC];
    let mut c_last_row = [0i32; SILK_MAX_ORDER_LPC];
    let mut af_qa = [0i32; SILK_MAX_ORDER_LPC];
    let mut caf = [0i32; SILK_MAX_ORDER_LPC + 1];
    let mut cab = [0i32; SILK_MAX_ORDER_LPC + 1];

    // Compute total energy C0
    let mut c0_64: i64 = 0;
    let total_len = subfr_length * nb_subfr;
    for item in x.iter().take(total_len.min(x.len())) {
        c0_64 += *item as i64 * *item as i64;
    }

    let lz = if c0_64 == 0 {
        64
    } else {
        c0_64.leading_zeros() as i32
    };
    let rshifts = (32 + 1 + N_BITS_HEAD_ROOM - lz).clamp(MIN_RSHIFTS, MAX_RSHIFTS);

    let c0 = if rshifts > 0 {
        (c0_64 >> rshifts) as i32
    } else {
        (c0_64 as i32) << (-rshifts)
    };

    // Initialize CAf[0], CAb[0] with conditioning factor
    let cond_add = silk_smmul(FIND_LPC_COND_FAC_Q32, c0) + 1;
    caf[0] = c0 + cond_add;
    cab[0] = caf[0];

    // Compute initial cross-correlations
    for s in 0..nb_subfr {
        let x_ptr = &x[s * subfr_length..];
        for n in 1..=d {
            let mut acc: i64 = 0;
            let len = subfr_length
                .saturating_sub(n)
                .min(x_ptr.len().saturating_sub(n));
            for i in 0..len {
                acc += x_ptr[i] as i64 * x_ptr[i + n] as i64;
            }
            if rshifts > 0 {
                c_first_row[n - 1] += (acc >> rshifts) as i32;
            } else {
                c_first_row[n - 1] += (acc as i32) << (-rshifts);
            }
        }
    }
    c_last_row[..d].copy_from_slice(&c_first_row[..d]);

    // Re-initialize (same as above — C reference does this twice)
    caf[0] = c0 + cond_add;
    cab[0] = caf[0];

    let mut inv_gain_q30: i32 = 1 << 30;
    let mut reached_max_gain = false;

    for n in 0..d {
        // Update correlation rows and C*Af/C*Ab
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..];
            if n < x_ptr.len() && subfr_length > n {
                let x1 = -(x_ptr[n] as i32) << (16 - rshifts).clamp(0, 16);
                let x2 = if subfr_length - n - 1 < x_ptr.len() {
                    -(x_ptr[subfr_length - n - 1] as i32) << (16 - rshifts).clamp(0, 16)
                } else {
                    0
                };
                let mut tmp1 = (x_ptr[n] as i32) << (QA - 16).clamp(0, 16);
                let mut tmp2 = if subfr_length - n - 1 < x_ptr.len() {
                    (x_ptr[subfr_length - n - 1] as i32) << (QA - 16).clamp(0, 16)
                } else {
                    0
                };

                for k in 0..n {
                    if n > k && n - k - 1 < x_ptr.len() {
                        c_first_row[k] = silk_smlawb(c_first_row[k], x1, x_ptr[n - k - 1] as i32);
                    }
                    if subfr_length > n && subfr_length - n + k < x_ptr.len() {
                        c_last_row[k] =
                            silk_smlawb(c_last_row[k], x2, x_ptr[subfr_length - n + k] as i32);
                    }
                    let atmp_qa = af_qa[k];
                    if n > k && n - k - 1 < x_ptr.len() {
                        tmp1 = silk_smlawb(tmp1, atmp_qa, x_ptr[n - k - 1] as i32);
                    }
                    if subfr_length > n && subfr_length - n + k < x_ptr.len() {
                        tmp2 = silk_smlawb(tmp2, atmp_qa, x_ptr[subfr_length - n + k] as i32);
                    }
                }
                tmp1 = (-tmp1) << (32 - QA - rshifts).clamp(0, 30);
                tmp2 = (-tmp2) << (32 - QA - rshifts).clamp(0, 30);
                for k in 0..=n {
                    if n >= k && n - k < x_ptr.len() {
                        caf[k] = silk_smlawb(caf[k], tmp1, x_ptr[n - k] as i32);
                    }
                    if subfr_length > n
                        && subfr_length - n + k >= 1
                        && subfr_length - n + k - 1 < x_ptr.len()
                    {
                        cab[k] = silk_smlawb(cab[k], tmp2, x_ptr[subfr_length - n + k - 1] as i32);
                    }
                }
            }
        }

        // Calculate reflection coefficient (C: burg_modified_FIX.c lines 171-197)
        // Use local accumulators — do NOT modify c_first_row/c_last_row arrays
        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        let mut num: i32 = 0;
        let mut nrg: i32 = cab[0].wrapping_add(caf[0]);
        for k in 0..n {
            let atmp_qa = af_qa[k];
            let lz = (silk_clz32(atmp_qa.abs()) - 1).clamp(0, 32 - QA);
            let atmp1 = (atmp_qa as u32).wrapping_shl(lz as u32) as i32;
            let shift = (32 - QA - lz).clamp(0, 31);

            tmp1 = tmp1.wrapping_add(silk_smmul(c_last_row[n - k - 1], atmp1) << shift);
            tmp2 = tmp2.wrapping_add(silk_smmul(c_first_row[n - k - 1], atmp1) << shift);
            num = num.wrapping_add(silk_smmul(cab[n - k], atmp1) << shift);
            nrg = nrg.wrapping_add(silk_smmul(cab[k + 1].wrapping_add(caf[k + 1]), atmp1) << shift);
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;
        num = num.wrapping_add(tmp2);
        num = (-num) << 1;

        // Compute reflection coefficient
        let rc_q31 = if num.abs() < nrg {
            silk_div32_varq(num, nrg, 31)
        } else if num > 0 {
            i32::MAX
        } else {
            i32::MIN
        };

        // Update inverse prediction gain
        let tmp1_gain = (1i32 << 30) - silk_smmul(rc_q31, rc_q31);
        let new_inv_gain = silk_smmul(inv_gain_q30, tmp1_gain) << 2;

        let rc_q31 = if new_inv_gain <= min_inv_gain_q30 {
            // Max prediction gain exceeded — adjust rc to exactly hit the limit
            let tmp2_adj = (1i32 << 30) - silk_div32_varq(min_inv_gain_q30, inv_gain_q30, 30);
            let mut rc_adj = silk_sqrt_approx(tmp2_adj); // Q15
            if rc_adj > 0 {
                rc_adj = (rc_adj + tmp2_adj / rc_adj) >> 1; // Newton-Raphson
                rc_adj <<= 16; // Q15 → Q31
                if num < 0 {
                    rc_adj = -rc_adj;
                }
            }
            inv_gain_q30 = min_inv_gain_q30;
            reached_max_gain = true;
            rc_adj
        } else {
            inv_gain_q30 = new_inv_gain;
            rc_q31
        };

        // Update AR coefficients
        for k in 0..((n + 1) >> 1) {
            let t1 = af_qa[k];
            let t2 = af_qa[n - k - 1];
            af_qa[k] = t1.wrapping_add(silk_smmul(t2, rc_q31) << 1);
            af_qa[n - k - 1] = t2.wrapping_add(silk_smmul(t1, rc_q31) << 1);
        }
        af_qa[n] = rc_q31 >> (31 - QA);

        if reached_max_gain {
            for item in af_qa.iter_mut().take(d).skip(n + 1) {
                *item = 0;
            }
            break;
        }

        // Update C*Af and C*Ab
        for k in 0..=(n + 1) {
            let t1 = caf[k];
            let t2 = cab[n + 1 - k];
            caf[k] = t1.wrapping_add(silk_smmul(t2, rc_q31) << 1);
            cab[n + 1 - k] = t2.wrapping_add(silk_smmul(t1, rc_q31) << 1);
        }
    }

    // Output LPC coefficients and residual energy
    if reached_max_gain {
        for k in 0..d {
            a_q16[k] = -silk_rshift_round(af_qa[k], QA - 16);
        }
        // Subtract energy of preceding D samples from C0
        let mut c0_adj = c0;
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..];
            let mut e: i64 = 0;
            for item in x_ptr.iter().take(d.min(x_ptr.len())) {
                e += *item as i64 * *item as i64;
            }
            if rshifts > 0 {
                c0_adj -= (e >> rshifts) as i32;
            } else {
                c0_adj -= (e as i32) << (-rshifts);
            }
        }
        *res_nrg = silk_smmul(inv_gain_q30, c0_adj) << 2;
        *res_nrg_q = -rshifts;
    } else {
        let mut nrg = caf[0];
        let mut tmp1_sum = 1i32 << 16;
        for k in 0..d {
            let atmp1 = silk_rshift_round(af_qa[k], QA - 16);
            nrg = nrg.wrapping_add(((caf[k + 1] as i64 * atmp1 as i64) >> 16) as i32);
            tmp1_sum = tmp1_sum.wrapping_add(((atmp1 as i64 * atmp1 as i64) >> 16) as i32);
            a_q16[k] = -atmp1;
        }
        *res_nrg = nrg.wrapping_add(
            ((silk_smmul(FIND_LPC_COND_FAC_Q32, c0) as i64 * (-tmp1_sum) as i64) >> 16) as i32,
        );
        *res_nrg_q = -rshifts;
    }
}

// ---- Float Burg's Modified Algorithm (port of silk/float/burg_modified_FLP.c) ----

const FIND_LPC_COND_FAC: f64 = 1e-5;

/// Float-point Burg's modified algorithm for LPC analysis.
/// Port of silk/float/burg_modified_FLP.c — matches the C reference's
/// default (non-FIXED_POINT) encoder path.
///
/// Returns residual energy as f32. Output coefficients `a` are f32.
/// Internally uses f64 for all accumulation (matching C's `double`).
pub fn silk_burg_modified_flp(
    a: &mut [f32],       // O: prediction coefficients [D]
    x: &[f32],           // I: input signal [nb_subfr * subfr_length]
    min_inv_gain: f32,   // I: minimum inverse prediction gain
    subfr_length: usize, // I: subframe length (incl. D preceding samples)
    nb_subfr: usize,     // I: number of subframes
    d: usize,            // I: LPC order
) -> f32 {
    let mut c_first_row = [0.0f64; SILK_MAX_ORDER_LPC];
    let mut c_last_row = [0.0f64; SILK_MAX_ORDER_LPC];
    let mut af = [0.0f64; SILK_MAX_ORDER_LPC];
    let mut caf = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    let mut cab = [0.0f64; SILK_MAX_ORDER_LPC + 1];

    // Compute total energy C0
    let mut c0: f64 = 0.0;
    let total_len = nb_subfr * subfr_length;
    for item in x.iter().take(total_len.min(x.len())) {
        c0 += (*item as f64) * (*item as f64);
    }

    // Compute initial cross-correlations
    for s in 0..nb_subfr {
        let x_ptr = &x[s * subfr_length..];
        for n in 1..=d {
            let len = subfr_length.saturating_sub(n);
            let mut acc: f64 = 0.0;
            for i in 0..len.min(x_ptr.len().saturating_sub(n)) {
                acc += (x_ptr[i] as f64) * (x_ptr[i + n] as f64);
            }
            c_first_row[n - 1] += acc;
        }
    }
    c_last_row[..d].copy_from_slice(&c_first_row[..d]);

    // Initialize CAf[0], CAb[0] with conditioning factor
    caf[0] = c0 + FIND_LPC_COND_FAC * c0 + 1e-9;
    cab[0] = caf[0];

    let mut inv_gain: f64 = 1.0;
    let min_inv_gain_f64 = min_inv_gain as f64;
    let mut reached_max_gain = false;

    for n in 0..d {
        // Update correlation rows and CAf/CAb
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..];
            if n < x_ptr.len() && subfr_length > n {
                let mut tmp1 = x_ptr[n] as f64;
                let mut tmp2 = if subfr_length - n - 1 < x_ptr.len() {
                    x_ptr[subfr_length - n - 1] as f64
                } else {
                    0.0
                };

                for k in 0..n {
                    if n > k && n - k - 1 < x_ptr.len() {
                        c_first_row[k] -= (x_ptr[n] as f64) * (x_ptr[n - k - 1] as f64);
                    }
                    if subfr_length > n && subfr_length - n + k < x_ptr.len() {
                        c_last_row[k] -= (x_ptr[subfr_length - n - 1] as f64)
                            * (x_ptr[subfr_length - n + k] as f64);
                    }
                    let atmp = af[k];
                    if n > k && n - k - 1 < x_ptr.len() {
                        tmp1 += (x_ptr[n - k - 1] as f64) * atmp;
                    }
                    if subfr_length > n && subfr_length - n + k < x_ptr.len() {
                        tmp2 += (x_ptr[subfr_length - n + k] as f64) * atmp;
                    }
                }
                for k in 0..=n {
                    if n >= k && n - k < x_ptr.len() {
                        caf[k] -= tmp1 * (x_ptr[n - k] as f64);
                    }
                    if subfr_length > n
                        && subfr_length - n + k >= 1
                        && subfr_length - n + k - 1 < x_ptr.len()
                    {
                        cab[k] -= tmp2 * (x_ptr[subfr_length - n + k - 1] as f64);
                    }
                }
            }
        }

        // Calculate reflection coefficient
        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        let mut num: f64 = 0.0;
        let mut nrg_b: f64 = cab[0];
        let mut nrg_f: f64 = caf[0];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        let mut rc = -2.0 * (num + cab[n + 1]) / (nrg_f + nrg_b);

        // Update inverse prediction gain
        let tmp1_gain = inv_gain * (1.0 - rc * rc);
        if tmp1_gain <= min_inv_gain_f64 {
            // Max prediction gain exceeded
            rc = (1.0 - min_inv_gain_f64 / inv_gain).sqrt();
            if num + cab[n + 1] > 0.0 {
                rc = -rc;
            }
            inv_gain = min_inv_gain_f64;
            reached_max_gain = true;
        } else {
            inv_gain = tmp1_gain;
        }

        // Update AR coefficients
        for k in 0..((n + 1) >> 1) {
            let t1 = af[k];
            let t2 = af[n - k - 1];
            af[k] = t1 + rc * t2;
            af[n - k - 1] = t2 + rc * t1;
        }
        af[n] = rc;

        if reached_max_gain {
            for item in af.iter_mut().take(d).skip(n + 1) {
                *item = 0.0;
            }
            break;
        }

        // Update CAf and CAb
        for k in 0..=(n + 1) {
            let t1 = caf[k];
            caf[k] += rc * cab[n + 1 - k];
            cab[n + 1 - k] += rc * t1;
        }
    }

    // Output coefficients and residual energy
    let nrg_f;
    if reached_max_gain {
        for k in 0..d {
            a[k] = -af[k] as f32;
        }
        let mut c0_adj = c0;
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..];
            let mut e: f64 = 0.0;
            for item in x_ptr.iter().take(d.min(x_ptr.len())) {
                e += (*item as f64) * (*item as f64);
            }
            c0_adj -= e;
        }
        nrg_f = c0_adj * inv_gain;
    } else {
        let mut nrg = caf[0];
        let mut tmp1_sum = 1.0f64;
        for k in 0..d {
            let atmp = af[k];
            nrg += caf[k + 1] * atmp;
            tmp1_sum += atmp * atmp;
            a[k] = -atmp as f32;
        }
        nrg_f = nrg - FIND_LPC_COND_FAC * c0 * tmp1_sum;
    }

    nrg_f as f32
}
