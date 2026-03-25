use crate::fft::{KissFftCpx, KissFftState, opus_fft_impl};

/// MDCT lookup structure with pre-allocated scratch buffers.
pub struct MdctLookup {
    pub n: usize,
    pub max_shift: usize,
    pub kfft: Vec<KissFftState>,
    pub trig: Vec<f32>,
    /// Scratch buffer for forward MDCT fold output (max N/2 floats).
    fwd_buf: Vec<f32>,
    /// Scratch buffer for FFT complex data (max N/4 complex values).
    fft_buf: Vec<KissFftCpx>,
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
            fwd_buf: vec![0.0f32; n >> 1],
            fft_buf: vec![KissFftCpx::default(); n >> 2],
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
    l: &mut MdctLookup,
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

    // Copy immutable data we need before taking mutable scratch borrows.
    let scale = l.kfft[shift].scale;
    let bitrev = &l.kfft[shift].bitrev;

    // === Phase 1: Window, shuffle, fold → f[0..N2-1] ===
    // Matches C clt_mdct_forward_c exactly: three regions with windowed folding.
    debug_assert!(overlap > 0 && overlap <= n2);
    debug_assert!(input.len() >= n2 + overlap);
    debug_assert!(window.len() >= overlap);

    let f_buf = &mut l.fwd_buf[..n2];
    {
        let xp1_base = overlap >> 1;
        let xp2_base = n2 - 1 + (overlap >> 1);
        let wp_base = overlap >> 1;

        // Loop 1: overlap beginning region
        let loop1_end = (overlap + 3) >> 2;
        for i in 0..loop1_end {
            let xp1 = xp1_base + 2 * i;
            let xp2 = xp2_base - 2 * i;
            let w1 = window[wp_base + 2 * i];
            let w2 = window[wp_base - 1 - 2 * i];
            f_buf[2 * i] = w2 * input[xp1 + n2] + w1 * input[xp2];
            f_buf[2 * i + 1] = w1 * input[xp1] - w2 * input[xp2 - n2];
        }

        // Loop 2: middle region (no windowing, window = 1.0)
        let loop2_end = n4 - ((overlap + 3) >> 2);
        for i in loop1_end..loop2_end {
            let xp1 = xp1_base + 2 * i;
            let xp2 = xp2_base - 2 * i;
            f_buf[2 * i] = input[xp2];
            f_buf[2 * i + 1] = input[xp1];
        }

        // Loop 3: overlap end region
        for i in loop2_end..n4 {
            let xp1 = xp1_base + 2 * i;
            let xp2 = xp2_base - 2 * i;
            let k = i - loop2_end;
            let w1 = window[2 * k];
            let w2 = window[overlap - 1 - 2 * k];
            f_buf[2 * i] = -w1 * input[xp1 - n2] + w2 * input[xp2];
            f_buf[2 * i + 1] = w2 * input[xp1] + w1 * input[xp2 + n2];
        }
    }

    // Phases 2-4 use trig and fft_buf as disjoint borrows from l.
    let (trig, fft_buf, kfft) = (&l.trig[trig_offset..], &mut l.fft_buf[..n4], &l.kfft[shift]);

    // === Phase 2: Pre-rotation with scale → fft_buf[bitrev[i]] ===
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
            fft_buf[bitrev[i]].r = yr * scale;
            fft_buf[bitrev[i]].i = yi * scale;
        }
    }

    // === Phase 3: N/4-point complex FFT in-place ===
    opus_fft_impl(kfft, fft_buf);

    // === Phase 4: Post-rotation → output with stride ===
    {
        let out_len = output.len();
        for i in 0..n4 {
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = fft_buf[i].i * t1 - fft_buf[i].r * t0;
            let yi = fft_buf[i].r * t1 + fft_buf[i].i * t0;
            let yp1 = 2 * i * stride;
            let yp2 = stride * (n2 - 1 - 2 * i);
            if yp1 < out_len {
                output[yp1] = yr;
            }
            if yp2 < out_len {
                output[yp2] = yi;
            }
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
    l: &mut MdctLookup,
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

    // === Pre-rotation ===
    let yp_base = overlap >> 1;
    {
        let trig = &l.trig[trig_offset..];
        let bitrev = &l.kfft[shift].bitrev;
        for i in 0..n4 {
            let rev = bitrev[i];
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            output[yp_base + 2 * rev + 1] = yr;
            output[yp_base + 2 * rev] = yi;
        }
    }

    // === N/4-point complex FFT in-place ===
    {
        let f2 = &mut l.fft_buf[..n4];
        for i in 0..n4 {
            f2[i].r = output[yp_base + 2 * i];
            f2[i].i = output[yp_base + 2 * i + 1];
        }
        opus_fft_impl(&l.kfft[shift], f2);
        for i in 0..n4 {
            output[yp_base + 2 * i] = f2[i].r;
            output[yp_base + 2 * i + 1] = f2[i].i;
        }
    }

    // === Post-rotate and de-shuffle from both ends simultaneously (in-place) ===
    {
        let trig = &l.trig[trig_offset..];
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
