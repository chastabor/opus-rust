use crate::fft::{KissFftCpx, KissFftState, opus_fft_impl};

/// MDCT lookup structure.
pub struct MdctLookup {
    pub n: usize,
    pub max_shift: usize,
    pub kfft: Vec<KissFftState>,
    pub trig: Vec<f32>,
}

impl MdctLookup {
    /// Create a new MDCT lookup for size N with maxshift levels.
    /// Trig table: trig[i] = cos(2*PI*(i+0.125)/N) for i=0..N/2-1
    /// at each shift level.
    pub fn new(n: usize, max_shift: usize) -> Self {
        let mut kfft = Vec::new();
        let mut trig = Vec::new();
        let mut cur_n = n;
        for _s in 0..=max_shift {
            kfft.push(KissFftState::new(cur_n >> 2));
            let n2 = cur_n >> 1;
            let pi: f64 = std::f64::consts::PI;
            for i in 0..n2 {
                let phase = 2.0 * pi * (i as f64 + 0.125) / cur_n as f64;
                trig.push(phase.cos() as f32);
            }
            cur_n >>= 1;
        }
        MdctLookup {
            n,
            max_shift,
            kfft,
            trig,
        }
    }
}

/// MDCT backward (inverse) transform.
/// Matches clt_mdct_backward_c in celt/mdct.c for the float case.
///
/// `input`: frequency-domain data (read with stride)
/// `output`: time-domain output buffer
/// `window`: TDAC window of length `overlap`
/// `overlap`: overlap length
/// `shift`: shift level (0 = full size, 1 = half, etc.)
/// `stride`: interleave stride for short blocks (B)
pub fn clt_mdct_backward(
    l: &MdctLookup,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let mut n = l.n;
    let mut trig_offset = 0;
    for _ in 0..shift {
        trig_offset += n >> 1;
        n >>= 1;
    }
    let n2 = n >> 1;
    let n4 = n >> 2;

    let trig = &l.trig[trig_offset..];
    let st = &l.kfft[shift];

    // === Pre-rotation ===
    // C code writes directly to out+(overlap>>1) at bitrev positions.
    let yp_base = overlap >> 1;
    {
        let mut xp1_idx = 0usize;
        let mut xp2_idx = stride * (n2 - 1);
        let bitrev = &st.bitrev;
        for i in 0..n4 {
            let rev = bitrev[i];
            let x1 = if xp1_idx < input.len() { input[xp1_idx] } else { 0.0 };
            let x2 = if xp2_idx < input.len() { input[xp2_idx] } else { 0.0 };
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            // We swap real and imag because we use an FFT instead of an IFFT.
            output[yp_base + 2 * rev + 1] = yr;
            output[yp_base + 2 * rev] = yi;
            xp1_idx += 2 * stride;
            xp2_idx = xp2_idx.wrapping_sub(2 * stride);
        }
    }

    // === N/4-point complex FFT in-place ===
    {
        let mut f2 = vec![KissFftCpx::default(); n4];
        for i in 0..n4 {
            f2[i].r = output[yp_base + 2 * i];
            f2[i].i = output[yp_base + 2 * i + 1];
        }
        opus_fft_impl(st, &mut f2);
        for i in 0..n4 {
            output[yp_base + 2 * i] = f2[i].r;
            output[yp_base + 2 * i + 1] = f2[i].i;
        }
    }

    // === Post-rotate and de-shuffle from both ends simultaneously (in-place) ===
    {
        let half = (n4 + 1) >> 1;
        for i in 0..half {
            let yp0 = yp_base + 2 * i;
            let yp1 = yp_base + n2 - 2 - 2 * i;

            let re0 = output[yp0 + 1];
            let im0 = output[yp0];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr0 = re0 * t0 + im0 * t1;
            let yi0 = re0 * t1 - im0 * t0;

            let re1 = output[yp1 + 1];
            let im1 = output[yp1];

            output[yp0] = yr0;
            output[yp1 + 1] = yi0;

            let t0b = trig[n4 - i - 1];
            let t1b = trig[n2 - i - 1];
            let yr1 = re1 * t0b + im1 * t1b;
            let yi1 = re1 * t1b - im1 * t0b;

            output[yp1] = yr1;
            output[yp0 + 1] = yi1;
        }
    }

    // === Mirror for TDAC windowing ===
    for i in 0..(overlap / 2) {
        let xp1_idx = overlap - 1 - i;
        let yp1_idx = i;
        let x1 = output[xp1_idx];
        let x2 = output[yp1_idx];
        let wp1 = window[i];
        let wp2 = window[overlap - 1 - i];
        output[yp1_idx] = x2 * wp2 - x1 * wp1;
        output[xp1_idx] = x2 * wp1 + x1 * wp2;
    }
}
