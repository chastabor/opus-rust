use crate::mode::CeltMode;
use crate::tables::*;
use opus_range_coder::EcCtx;

/// Unquantize coarse energy.
pub fn unquant_coarse_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_band_e: &mut [f32],
    intra: bool,
    dec: &mut EcCtx,
    c: usize,
    lm: usize,
) {
    let prob_model = &E_PROB_MODEL[lm][if intra { 1 } else { 0 }];
    let mut prev = [0.0f64; 2];
    let coef: f32;
    let beta: f32;

    if intra {
        coef = 0.0;
        beta = BETA_INTRA;
    } else {
        beta = BETA_COEF[lm];
        coef = PRED_COEF[lm];
    }

    let budget = dec.storage as i32 * 8;

    for i in start..end {
        for ch in 0..c {
            let tell = dec.tell();
            let qi: i32;
            if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                qi = dec.laplace_decode(
                    (prob_model[pi] as u32) << 7,
                    (prob_model[pi + 1] as i32) << 6,
                );
            } else if budget - tell >= 2 {
                let raw = dec.dec_icdf(&SMALL_ENERGY_ICDF, 2);
                qi = (raw as i32 >> 1) ^ -(raw as i32 & 1);
            } else if budget - tell >= 1 {
                qi = -(dec.dec_bit_logp(1) as i32);
            } else {
                qi = -1;
            }
            let q = qi as f32;

            old_band_e[i + ch * m.nb_ebands] =
                old_band_e[i + ch * m.nb_ebands].max(-9.0);
            let tmp = coef * old_band_e[i + ch * m.nb_ebands] + prev[ch] as f32 + q;
            old_band_e[i + ch * m.nb_ebands] = tmp;
            prev[ch] = prev[ch] + q as f64 - (beta * q) as f64;
        }
    }
}

/// Unquantize fine energy.
pub fn unquant_fine_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_band_e: &mut [f32],
    fine_quant: &[i32],
    dec: &mut EcCtx,
    c: usize,
) {
    for i in start..end {
        if fine_quant[i] <= 0 {
            continue;
        }
        if dec.tell() + (c as i32) * fine_quant[i] > dec.storage as i32 * 8 {
            continue;
        }
        for ch in 0..c {
            let q2 = dec.dec_bits(fine_quant[i] as u32) as i32;
            let offset =
                (q2 as f32 + 0.5) * ((1 << (14 - fine_quant[i])) as f32) * (1.0 / 16384.0)
                    - 0.5;
            old_band_e[i + ch * m.nb_ebands] += offset;
        }
    }
}

/// Unquantize energy finalise (use remaining bits).
pub fn unquant_energy_finalise(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_band_e: Option<&mut [f32]>,
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut EcCtx,
    c: usize,
) {
    let old_band_e = match old_band_e {
        Some(e) => e,
        None => return,
    };
    for prio in 0..2 {
        for i in start..end {
            if bits_left < c as i32 {
                return;
            }
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                continue;
            }
            for ch in 0..c {
                let q2 = dec.dec_bits(1) as i32;
                let offset = (q2 as f32 - 0.5)
                    * ((1 << (14 - fine_quant[i] - 1)) as f32)
                    * (1.0 / 16384.0);
                old_band_e[i + ch * m.nb_ebands] += offset;
                bits_left -= 1;
            }
        }
    }
}
