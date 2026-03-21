/// Complex number for FFT.
#[derive(Clone, Copy, Default)]
pub struct KissFftCpx {
    pub r: f32,
    pub i: f32,
}

/// FFT state containing twiddle factors and bit-reversal table.
pub struct KissFftState {
    pub nfft: usize,
    pub scale: f32,
    pub shift: i32,
    pub factors: Vec<usize>,
    pub bitrev: Vec<usize>,
    pub twiddles: Vec<KissFftCpx>,
}

impl KissFftState {
    /// Create a new FFT state for a given size.
    pub fn new(nfft: usize) -> Self {
        let scale = 1.0 / nfft as f32;
        let twiddles = compute_twiddles(nfft);
        let factors = compute_factors(nfft);
        let bitrev = compute_bitrev(nfft, &factors);
        KissFftState {
            nfft,
            scale,
            shift: -1,
            factors,
            bitrev,
            twiddles,
        }
    }
}

fn compute_twiddles(nfft: usize) -> Vec<KissFftCpx> {
    let mut tw = Vec::with_capacity(nfft);
    for i in 0..nfft {
        let phase = -2.0 * std::f32::consts::PI * i as f32 / nfft as f32;
        tw.push(KissFftCpx {
            r: phase.cos(),
            i: phase.sin(),
        });
    }
    tw
}

fn compute_factors(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let radices = [5, 4, 3, 2];
    for &r in &radices {
        while n % r == 0 {
            factors.push(r);
            n /= r;
        }
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

fn compute_bitrev(nfft: usize, factors: &[usize]) -> Vec<usize> {
    let mut bitrev = vec![0usize; nfft];
    let mut stage_strides = Vec::new();
    let mut stride = nfft;
    for &f in factors {
        stride /= f;
        stage_strides.push(stride);
    }
    for i in 0..nfft {
        let mut idx = 0;
        let mut remainder = i;
        for (fi, &f) in factors.iter().enumerate() {
            let digit = remainder / stage_strides[fi];
            remainder %= stage_strides[fi];
            idx += digit * (nfft / (factors[..=fi].iter().product::<usize>()));
        }
        bitrev[i] = idx % nfft;
    }
    // For a simple approach, just use identity bitrev (the actual MDCT
    // writes directly to bitrev positions).
    for i in 0..nfft {
        bitrev[i] = i;
    }
    bitrev
}

/// Perform in-place complex FFT.
pub fn opus_fft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    let n = st.nfft;
    assert!(fin.len() >= n);
    assert!(fout.len() >= n);
    // Copy input to output
    fout[..n].copy_from_slice(&fin[..n]);
    // Cooley-Tukey mixed-radix DIT FFT
    let mut stride = n;
    for &radix in &st.factors {
        stride /= radix;
        match radix {
            2 => fft_pass2(fout, n, stride, &st.twiddles),
            3 => fft_pass3(fout, n, stride, &st.twiddles),
            4 => fft_pass4(fout, n, stride, &st.twiddles),
            5 => fft_pass5(fout, n, stride, &st.twiddles),
            _ => {}
        }
    }
}

fn fft_pass2(f: &mut [KissFftCpx], n: usize, stride: usize, tw: &[KissFftCpx]) {
    let m = stride;
    for k in (0..n).step_by(2 * m) {
        for j in 0..m {
            let tw_idx = j * (n / (2 * m));
            let t = KissFftCpx {
                r: f[k + j + m].r * tw[tw_idx].r - f[k + j + m].i * tw[tw_idx].i,
                i: f[k + j + m].r * tw[tw_idx].i + f[k + j + m].i * tw[tw_idx].r,
            };
            f[k + j + m] = KissFftCpx {
                r: f[k + j].r - t.r,
                i: f[k + j].i - t.i,
            };
            f[k + j].r += t.r;
            f[k + j].i += t.i;
        }
    }
}

fn fft_pass3(f: &mut [KissFftCpx], n: usize, stride: usize, tw: &[KissFftCpx]) {
    let m = stride;
    let tw_step = n / (3 * m);
    for k in (0..n).step_by(3 * m) {
        for j in 0..m {
            let tw1 = tw[j * tw_step];
            let tw2 = tw[2 * j * tw_step % n];
            let a0 = f[k + j];
            let a1 = cmul(f[k + j + m], tw1);
            let a2 = cmul(f[k + j + 2 * m], tw2);
            let t1 = KissFftCpx {
                r: a1.r + a2.r,
                i: a1.i + a2.i,
            };
            let t2 = KissFftCpx {
                r: a0.r - 0.5 * t1.r,
                i: a0.i - 0.5 * t1.i,
            };
            let t3r = 0.86602540378 * (a1.i - a2.i);
            let t3i = 0.86602540378 * (a2.r - a1.r);
            f[k + j] = KissFftCpx {
                r: a0.r + t1.r,
                i: a0.i + t1.i,
            };
            f[k + j + m] = KissFftCpx {
                r: t2.r + t3r as f32,
                i: t2.i + t3i as f32,
            };
            f[k + j + 2 * m] = KissFftCpx {
                r: t2.r - t3r as f32,
                i: t2.i - t3i as f32,
            };
        }
    }
}

fn fft_pass4(f: &mut [KissFftCpx], n: usize, stride: usize, tw: &[KissFftCpx]) {
    let m = stride;
    let tw_step = n / (4 * m);
    for k in (0..n).step_by(4 * m) {
        for j in 0..m {
            let tw1 = tw[j * tw_step];
            let tw2 = tw[(2 * j * tw_step) % n];
            let tw3 = tw[(3 * j * tw_step) % n];
            let a0 = f[k + j];
            let a1 = cmul(f[k + j + m], tw1);
            let a2 = cmul(f[k + j + 2 * m], tw2);
            let a3 = cmul(f[k + j + 3 * m], tw3);
            let t0 = KissFftCpx {
                r: a0.r + a2.r,
                i: a0.i + a2.i,
            };
            let t1 = KissFftCpx {
                r: a0.r - a2.r,
                i: a0.i - a2.i,
            };
            let t2 = KissFftCpx {
                r: a1.r + a3.r,
                i: a1.i + a3.i,
            };
            let t3 = KissFftCpx {
                r: a1.r - a3.r,
                i: a1.i - a3.i,
            };
            // Note: this is a forward FFT (negative sign convention)
            f[k + j] = KissFftCpx {
                r: t0.r + t2.r,
                i: t0.i + t2.i,
            };
            f[k + j + m] = KissFftCpx {
                r: t1.r + t3.i,
                i: t1.i - t3.r,
            };
            f[k + j + 2 * m] = KissFftCpx {
                r: t0.r - t2.r,
                i: t0.i - t2.i,
            };
            f[k + j + 3 * m] = KissFftCpx {
                r: t1.r - t3.i,
                i: t1.i + t3.r,
            };
        }
    }
}

fn fft_pass5(f: &mut [KissFftCpx], n: usize, stride: usize, tw: &[KissFftCpx]) {
    let m = stride;
    let tw_step = n / (5 * m);
    let ya_r: f32 = 0.30901699;
    let ya_i: f32 = -0.95105652;
    let yb_r: f32 = -0.80901699;
    let yb_i: f32 = -0.58778525;

    for k in (0..n).step_by(5 * m) {
        for j in 0..m {
            let tw1 = tw[(j * tw_step) % n];
            let tw2 = tw[(2 * j * tw_step) % n];
            let tw3 = tw[(3 * j * tw_step) % n];
            let tw4 = tw[(4 * j * tw_step) % n];
            let a0 = f[k + j];
            let a1 = cmul(f[k + j + m], tw1);
            let a2 = cmul(f[k + j + 2 * m], tw2);
            let a3 = cmul(f[k + j + 3 * m], tw3);
            let a4 = cmul(f[k + j + 4 * m], tw4);

            let s12 = cadd(a1, a4);
            let d12 = csub(a1, a4);
            let s34 = cadd(a2, a3);
            let d34 = csub(a2, a3);

            f[k + j].r = a0.r + s12.r + s34.r;
            f[k + j].i = a0.i + s12.i + s34.i;

            let t1r = a0.r + s12.r * ya_r + s34.r * yb_r;
            let t1i = a0.i + s12.i * ya_r + s34.i * yb_r;
            let t2r = d12.i * ya_i + d34.i * yb_i;
            let t2i = -(d12.r * ya_i + d34.r * yb_i);

            f[k + j + m] = KissFftCpx { r: t1r - t2r, i: t1i - t2i };
            f[k + j + 4 * m] = KissFftCpx { r: t1r + t2r, i: t1i + t2i };

            let t3r = a0.r + s12.r * yb_r + s34.r * ya_r;
            let t3i = a0.i + s12.i * yb_r + s34.i * ya_r;
            let t4r = d12.i * yb_i - d34.i * ya_i;
            let t4i = -(d12.r * yb_i - d34.r * ya_i);

            f[k + j + 2 * m] = KissFftCpx { r: t3r - t4r, i: t3i - t4i };
            f[k + j + 3 * m] = KissFftCpx { r: t3r + t4r, i: t3i + t4i };
        }
    }
}

#[inline]
fn cmul(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}

#[inline]
fn cadd(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}

#[inline]
fn csub(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r - b.r,
        i: a.i - b.i,
    }
}
