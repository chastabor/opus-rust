/// Compute LPC coefficients from autocorrelation using Levinson-Durbin.
/// Float path: SHR32/SHL32/MULT32_32_Q31/frac_div32 are all identity in float mode.
pub fn celt_lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    for i in 0..p {
        lpc[i] = 0.0;
    }
    if ac[0] <= 1e-10 {
        return;
    }
    let mut error = ac[0];
    for i in 0..p {
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -(rr / error);
        lpc[i] = r;
        let half = (i + 1) >> 1;
        for j in 0..half {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }
        error -= r * r * error;
        if error <= 0.001 * ac[0] {
            break;
        }
    }
}

/// FIR filter.
pub fn celt_fir(x: &[f32], lpc: &[f32], y: &mut [f32], n: usize, ord: usize) {
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..ord {
            if i > j {
                sum += lpc[j] * x[i - j - 1];
            }
        }
        y[i] = sum;
    }
}

/// IIR filter.
pub fn celt_iir(x: &[f32], lpc: &[f32], y: &mut [f32], n: usize, ord: usize, mem: &mut [f32]) {
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..ord {
            sum -= lpc[j] * mem[j];
        }
        // Shift memory
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        if ord > 0 {
            mem[0] = sum;
        }
        y[i] = sum;
    }
}

/// Autocorrelation.
pub fn celt_autocorr(x: &[f32], ac: &mut [f32], _window: &[f32], _overlap: usize, p: usize, n: usize) {
    for k in 0..=p {
        let mut sum = 0.0f32;
        for i in k..n {
            sum += x[i] * x[i - k];
        }
        ac[k] = sum;
    }
}
