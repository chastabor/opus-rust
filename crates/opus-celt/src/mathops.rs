/// Float exp2 approximation: 2^x
/// Matches C celt mathops.h celt_exp2() exactly.
#[inline]
pub fn celt_exp2(x: f32) -> f32 {
    let integer = x.floor() as i32;
    if integer < -50 {
        return 0.0;
    }
    let frac = x - integer as f32;
    const A0: f32 = 9.999999403953552246093750000000e-01;
    const A1: f32 = 6.931530833244323730468750000000e-01;
    const A2: f32 = 2.401536107063293457031250000000e-01;
    const A3: f32 = 5.582631751894950866699218750000e-02;
    const A4: f32 = 8.989339694380760192871093750000e-03;
    const A5: f32 = 1.877576694823801517486572265625e-03;
    let poly = A0 + frac * (A1 + frac * (A2 + frac * (A3 + frac * (A4 + frac * A5))));
    let bits = poly.to_bits();
    let result_bits = ((bits as i32).wrapping_add(integer << 23)) as u32 & 0x7fffffff;
    f32::from_bits(result_bits)
}

/// Float log2 approximation: log2(x)
#[inline]
pub fn celt_log2(x: f32) -> f32 {
    x.log2()
}

/// Float sqrt.
#[inline]
pub fn celt_sqrt(x: f32) -> f32 {
    x.sqrt()
}

/// Float reciprocal sqrt.
#[inline]
pub fn celt_rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// Bit-exact cos approximation for bit allocation.
/// Input: x in Q14 [0, 16384], output in Q15.
pub fn bitexact_cos(x: i16) -> i16 {
    let x = x as i32;
    let tmp = (4096 + x * x) >> 13;
    let x2 = tmp as i16;
    let x2i = x2 as i32;
    let result = (32767 - x2i)
        + frac_mul16(x2i, -7651 + frac_mul16(x2i, 8277 + frac_mul16(-626, x2i)));
    (1 + result) as i16
}

/// Bit-exact log2(tan) approximation for bit allocation.
pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ec_ilog(icos as u32);
    let ls = ec_ilog(isin as u32);
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls as i32 - lc as i32) * (1 << 11)
        + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// FRAC_MUL16: (a * b) >> 15 with rounding.
#[inline]
pub fn frac_mul16(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 15) as i32
}

/// Integer log2 (number of bits needed to represent val).
#[inline]
pub fn ec_ilog(val: u32) -> i32 {
    if val == 0 {
        0
    } else {
        32 - val.leading_zeros() as i32
    }
}

/// Integer sqrt (floor).
pub fn isqrt32(mut val: u32) -> u32 {
    if val == 0 {
        return 0;
    }
    let mut g: u32 = 0;
    let mut bshift = (ec_ilog(val) as i32 - 1) >> 1;
    let mut b = 1u32 << bshift;
    loop {
        let t = (((g as u64) << 1) + b as u64) << bshift as u64;
        if t <= val as u64 {
            g += b;
            val -= t as u32;
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

/// LCG random number generator.
#[inline]
pub fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// Unsigned integer division (with rounding toward zero).
#[inline]
pub fn celt_udiv(n: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    (n as u32 / d as u32) as i32
}

/// Signed integer division with positive denominator.
#[inline]
pub fn celt_sudiv(n: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    n / d
}

/// Inner product of two float slices.
pub fn celt_inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += x[i] * y[i];
    }
    sum
}

/// Renormalize a vector to have the given target gain.
pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    let mut e = 1e-27f32;
    for i in 0..n {
        e += x[i] * x[i];
    }
    let g = gain / e.sqrt();
    for i in 0..n {
        x[i] *= g;
    }
}
