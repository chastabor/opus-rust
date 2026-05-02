/// Quantizer step table for DRED latent coding.
const DQ_TABLE: [i32; 8] = [0, 2, 3, 4, 6, 8, 12, 16];

/// Compute the quantizer level for a given latent index.
/// Matches C `compute_quantizer` from dred_coding.c.
///
/// `q0`: base quantizer, `dq`: quantizer step index (into DQ_TABLE),
/// `qmax`: maximum quantizer, `i`: latent pair index.
pub fn compute_quantizer(q0: i32, dq: i32, qmax: i32, i: i32) -> i32 {
    let quant = q0 + (DQ_TABLE[dq as usize] * i + 8) / 16;
    quant.min(qmax)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_quantizer_base() {
        assert_eq!(compute_quantizer(6, 0, 15, 0), 6);
        assert_eq!(compute_quantizer(6, 0, 15, 10), 6);
    }

    #[test]
    fn test_compute_quantizer_increasing() {
        let q0 = compute_quantizer(6, 4, 15, 0);
        let q1 = compute_quantizer(6, 4, 15, 5);
        let q2 = compute_quantizer(6, 4, 15, 10);
        assert!(q0 <= q1);
        assert!(q1 <= q2);
    }

    #[test]
    fn test_compute_quantizer_clamped() {
        assert_eq!(compute_quantizer(14, 7, 15, 100), 15);
    }
}
