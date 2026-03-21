use crate::fft::{KissFftCpx, KissFftState, opus_fft};

/// MDCT lookup structure.
pub struct MdctLookup {
    pub n: usize,
    pub max_shift: usize,
    pub kfft: Vec<KissFftState>,
    pub trig: Vec<f32>,
}

impl MdctLookup {
    /// Create a new MDCT lookup for size N with maxshift levels.
    pub fn new(n: usize, max_shift: usize) -> Self {
        let mut kfft = Vec::new();
        let mut trig = Vec::new();
        let mut cur_n = n;
        for _s in 0..=max_shift {
            kfft.push(KissFftState::new(cur_n >> 2));
            let n2 = cur_n >> 1;
            for i in 0..n2 {
                let phase = 2.0 * std::f32::consts::PI * (i as f32 + 0.125) / cur_n as f32;
                trig.push(phase.cos());
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
/// Reads from `input` with given stride, adds overlap to `output`.
/// `window` is the TDAC window, `overlap` is the window length.
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

    // Pre-rotate: interleave input into complex pairs
    let mut f2 = vec![KissFftCpx::default(); n4];
    {
        let mut xp1_idx = 0;
        let mut xp2_idx = n2 - 1;
        for i in 0..n4 {
            let x1 = input[xp1_idx * stride];
            let x2 = input[xp2_idx * stride];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            // Swap real/imag for FFT instead of IFFT
            f2[i] = KissFftCpx { r: yi, i: yr };
            xp1_idx += 2;
            xp2_idx = xp2_idx.wrapping_sub(2);
        }
    }

    // N/4 complex FFT
    let mut f2_out = vec![KissFftCpx::default(); n4];
    opus_fft(st, &f2, &mut f2_out);

    // Post-rotate and de-shuffle
    {
        let mut yp0 = vec![0.0f32; n2];
        for i in 0..n4 {
            let re = f2_out[i].i; // swapped
            let im = f2_out[i].r;
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            yp0[2 * i] = re * t0 + im * t1;
            yp0[2 * i + 1] = re * t1 - im * t0;
        }
        // Second pass: mirror from both ends
        let half = (n4 + 1) >> 1;
        for i in 0..half {
            let a = yp0[2 * i + 1];
            let b = yp0[2 * i];
            let c_idx = n2 - 2 - 2 * i;
            let c = yp0[c_idx + 1];
            let d = yp0[c_idx];

            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr1 = a * t0 + b * t1;
            let yi1 = a * t1 - b * t0;

            let t0b = trig[n4 - i - 1];
            let t1b = trig[n2 - i - 1];
            let yr2 = c * t0b + d * t1b;
            let yi2 = c * t1b - d * t0b;

            yp0[2 * i] = yr1;
            yp0[2 * i + 1] = yi1;
            yp0[c_idx] = yr2;
            yp0[c_idx + 1] = yi2;
        }

        // Write to output with TDAC window overlap
        let out_base = overlap >> 1;
        // Write the middle part
        for i in 0..n2 {
            output[out_base + i] = 2.0 * yp0[i];
        }
    }

    // Mirror on both sides for TDAC
    {
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
}
