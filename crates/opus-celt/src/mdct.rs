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

/// MDCT forward (analysis) transform.
/// Matches clt_mdct_forward_c in celt/mdct.c for the float case.
///
/// `input`: time-domain input buffer (windowed)
/// `output`: frequency-domain output buffer (written with stride)
/// `window`: TDAC window of length `overlap`
/// `overlap`: overlap length
/// `shift`: shift level (0 = full size, 1 = half, etc.)
/// `stride`: interleave stride for short blocks (B)
pub fn clt_mdct_forward(
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
    let scale = st.scale;

    // === Phase 1: Window, shuffle, fold → f[0..N2-1] ===
    // Matches C clt_mdct_forward_c exactly: three regions with windowed folding.
    debug_assert!(overlap > 0 && overlap <= n2);
    debug_assert!(input.len() >= n2 + overlap);
    debug_assert!(window.len() >= overlap);

    let mut f_buf = vec![0.0f32; n2];
    {
        // Use isize for pointers that decrement past zero (matching C pointer arithmetic)
        let mut xp1 = (overlap >> 1) as isize;
        let mut xp2 = (n2 - 1 + (overlap >> 1)) as isize;
        let mut yp = 0usize;
        let mut wp1 = (overlap >> 1) as isize;
        let mut wp2 = (overlap >> 1) as isize - 1;
        let n2i = n2 as isize;

        // Loop 1: overlap beginning region
        let loop1_end = (overlap + 3) >> 2;
        for _i in 0..loop1_end {
            let w1 = window[wp1 as usize];
            let w2 = window[wp2 as usize];
            f_buf[yp] = w2 * input[(xp1 + n2i) as usize] + w1 * input[xp2 as usize];
            f_buf[yp + 1] = w1 * input[xp1 as usize] - w2 * input[(xp2 - n2i) as usize];
            yp += 2;
            xp1 += 2;
            xp2 -= 2;
            wp1 += 2;
            wp2 -= 2;
        }

        // Loop 2: middle region (no windowing, window = 1.0)
        let loop2_end = n4 - ((overlap + 3) >> 2);
        for _i in loop1_end..loop2_end {
            f_buf[yp] = input[xp2 as usize];
            f_buf[yp + 1] = input[xp1 as usize];
            yp += 2;
            xp1 += 2;
            xp2 -= 2;
        }

        // Loop 3: overlap end region
        wp1 = 0;
        wp2 = overlap as isize - 1;
        for _i in loop2_end..n4 {
            let w1 = window[wp1 as usize];
            let w2 = window[wp2 as usize];
            f_buf[yp] = -w1 * input[(xp1 - n2i) as usize] + w2 * input[xp2 as usize];
            f_buf[yp + 1] = w2 * input[xp1 as usize] + w1 * input[(xp2 + n2i) as usize];
            yp += 2;
            xp1 += 2;
            xp2 -= 2;
            wp1 += 2;
            wp2 -= 2;
        }
    }

    // === Phase 2: Pre-rotation with scale → f2[bitrev[i]] ===
    let mut f2 = vec![KissFftCpx::default(); n4];
    {
        let mut yp = 0usize;
        for i in 0..n4 {
            let re = f_buf[yp];
            let im = f_buf[yp + 1];
            yp += 2;
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = re * t0 - im * t1;
            let yi = im * t0 + re * t1;
            f2[st.bitrev[i]].r = yr * scale;
            f2[st.bitrev[i]].i = yi * scale;
        }
    }

    // === Phase 3: N/4-point complex FFT in-place ===
    opus_fft_impl(st, &mut f2);

    // === Phase 4: Post-rotation → output with stride ===
    // C: yr = fp->i*t1 - fp->r*t0;  yi = fp->r*t1 + fp->i*t0
    // where t0 = t[i], t1 = t[N4+i]
    {
        let mut yp1 = 0usize; // C: yp1 = out
        let mut yp2 = stride * (n2 - 1); // C: yp2 = out + stride*(N2-1)
        for i in 0..n4 {
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = f2[i].i * t1 - f2[i].r * t0;
            let yi = f2[i].r * t1 + f2[i].i * t0;
            if yp1 < output.len() {
                output[yp1] = yr;
            }
            if yp2 < output.len() {
                output[yp2] = yi;
            }
            yp1 += 2 * stride;
            yp2 = yp2.wrapping_sub(2 * stride);
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
