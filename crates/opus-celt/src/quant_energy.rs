use crate::mode::CeltMode;
use crate::tables::*;
use opus_range_coder::EcCtx;

/// Compute loss distortion between current and old band energies.
fn loss_distortion(e_bands: &[f32], old_e_bands: &[f32], start: usize, end: usize, len: usize, c: usize) -> f32 {
    let mut dist = 0.0f32;
    for ch in 0..c {
        for i in start..end {
            let d = e_bands[i + ch * len] - old_e_bands[i + ch * len];
            dist += d * d;
        }
    }
    dist.min(200.0)
}

/// Encode coarse energy (single pass implementation).
fn quant_coarse_energy_impl(
    m: &CeltMode,
    start: usize,
    end: usize,
    e_bands: &[f32],
    old_band_e: &mut [f32],
    budget: i32,
    tell: i32,
    prob_model: &[u8],
    error: &mut [f32],
    enc: &mut EcCtx,
    c: usize,
    lm: usize,
    intra: bool,
    max_decay: f32,
    lfe: bool,
) -> i32 {
    let mut badness = 0i32;
    let mut prev = [0.0f64; 2];

    let coef: f32;
    let beta: f32;

    if tell + 3 <= budget {
        enc.enc_bit_logp(intra, 3);
    }

    if intra {
        coef = 0.0;
        beta = BETA_INTRA;
    } else {
        beta = BETA_COEF[lm];
        coef = PRED_COEF[lm];
    }

    for i in start..end {
        for ch in 0..c {
            let x = e_bands[i + ch * m.nb_ebands];
            let old_e = old_band_e[i + ch * m.nb_ebands].max(-9.0);
            let f = x - coef * old_e - prev[ch] as f32;
            let mut qi = (0.5 + f).floor() as i32;
            let decay_bound = old_band_e[i + ch * m.nb_ebands].max(-28.0) - max_decay;
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x) as i32;
                if qi > 0 {
                    qi = 0;
                }
            }
            let qi0 = qi;

            let tell = enc.tell();
            let bits_left = budget - tell - 3 * c as i32 * (end as i32 - i as i32);
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
            }
            if lfe && i >= 2 {
                qi = qi.min(0);
            }
            let tell = enc.tell();
            if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                enc.laplace_encode(
                    &mut qi,
                    (prob_model[pi] as u32) << 7,
                    (prob_model[pi + 1] as i32) << 6,
                );
            } else if budget - tell >= 2 {
                qi = qi.max(-1).min(1);
                let sym = (2 * qi ^ -(if qi < 0 { 1 } else { 0 })) as usize;
                enc.enc_icdf(sym, &SMALL_ENERGY_ICDF, 2);
            } else if budget - tell >= 1 {
                qi = qi.min(0);
                enc.enc_bit_logp(-qi != 0, 1);
            } else {
                qi = -1;
            }
            error[i + ch * m.nb_ebands] = f - qi as f32;
            badness += (qi0 - qi).abs();
            let q = qi as f32;
            let tmp = coef * old_e + prev[ch] as f32 + q;
            old_band_e[i + ch * m.nb_ebands] = tmp;
            prev[ch] = prev[ch] + q as f64 - (beta * q) as f64;
        }
    }
    if lfe { 0 } else { badness }
}

/// Encode coarse band energies (two-pass intra/inter).
pub fn quant_coarse_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    eff_end: usize,
    e_bands: &[f32],
    old_band_e: &mut [f32],
    total_bits: i32,
    error: &mut [f32],
    enc: &mut EcCtx,
    c: usize,
    lm: usize,
    nb_available_bytes: i32,
    force_intra: bool,
    delayed_intra: &mut f32,
    two_pass: bool,
    loss_rate: i32,
    lfe: bool,
) {
    let budget = total_bits;
    let tell = enc.tell();

    let mut intra = force_intra
        || (!two_pass && *delayed_intra > 2.0 * c as f32 * (end - start) as f32
            && nb_available_bytes > (end - start) as i32 * c as i32);
    let intra_bias = ((budget as f32 * *delayed_intra * loss_rate as f32) / (c as f32 * 512.0)) as i32;
    let new_distortion = loss_distortion(e_bands, old_band_e, start, eff_end, m.nb_ebands, c);

    if tell + 3 > budget {
        intra = false;
    }
    let two_pass = if tell + 3 > budget { false } else { two_pass };

    let mut max_decay = 16.0f32;
    if end - start > 10 {
        max_decay = max_decay.min(0.125 * nb_available_bytes as f32);
    }
    if lfe {
        max_decay = 3.0;
    }

    let enc_start_state = enc.save_state();

    let mut old_band_e_intra = old_band_e[..c * m.nb_ebands].to_vec();
    let mut error_intra = vec![0.0f32; c * m.nb_ebands];
    let mut badness1 = 0i32;

    if two_pass || intra {
        badness1 = quant_coarse_energy_impl(
            m, start, end, e_bands, &mut old_band_e_intra, budget,
            tell, &E_PROB_MODEL[lm][1], &mut error_intra, enc,
            c, lm, true, max_decay, lfe,
        );
    }

    if !intra {
        let tell_intra = enc.tell_frac() as i32;
        let enc_intra_state = enc.save_state();

        enc.restore_state(&enc_start_state);

        let badness2 = quant_coarse_energy_impl(
            m, start, end, e_bands, old_band_e, budget,
            tell, &E_PROB_MODEL[lm][if intra { 1 } else { 0 }], error, enc,
            c, lm, false, max_decay, lfe,
        );

        if two_pass && (badness1 < badness2 || (badness1 == badness2 && enc.tell_frac() as i32 + intra_bias > tell_intra)) {
            enc.restore_state(&enc_intra_state);
            old_band_e[..c * m.nb_ebands].copy_from_slice(&old_band_e_intra);
            error[..c * m.nb_ebands].copy_from_slice(&error_intra);
            intra = true;
        }
    } else {
        old_band_e[..c * m.nb_ebands].copy_from_slice(&old_band_e_intra);
        error[..c * m.nb_ebands].copy_from_slice(&error_intra);
    }

    if intra {
        *delayed_intra = new_distortion;
    } else {
        *delayed_intra = PRED_COEF[lm] * PRED_COEF[lm] * *delayed_intra + new_distortion;
    }
}

/// Encode fine energy bits.
pub fn quant_fine_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_band_e: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    enc: &mut EcCtx,
    c: usize,
) {
    for i in start..end {
        if fine_quant[i] <= 0 {
            continue;
        }
        let extra = 1i32 << fine_quant[i];
        if enc.tell() + c as i32 * fine_quant[i] > enc.storage as i32 * 8 {
            continue;
        }
        for ch in 0..c {
            let mut q2 = ((error[i + ch * m.nb_ebands] + 0.5) * extra as f32).floor() as i32;
            if q2 > extra - 1 {
                q2 = extra - 1;
            }
            if q2 < 0 {
                q2 = 0;
            }
            enc.enc_bits(q2 as u32, fine_quant[i] as u32);
            let offset = (q2 as f32 + 0.5) * ((1 << (14 - fine_quant[i])) as f32) * (1.0 / 16384.0) - 0.5;
            old_band_e[i + ch * m.nb_ebands] += offset;
            error[i + ch * m.nb_ebands] -= offset;
        }
    }
}

/// Encode energy finalise (use remaining bits).
pub fn quant_energy_finalise_enc(
    m: &CeltMode,
    start: usize,
    end: usize,
    mut old_band_e: Option<&mut [f32]>,
    error: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    enc: &mut EcCtx,
    c: usize,
) {
    for prio in 0..2 {
        for i in start..end {
            if bits_left < c as i32 {
                return;
            }
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                continue;
            }
            for ch in 0..c {
                let q2 = if error[i + ch * m.nb_ebands] < 0.0 { 0i32 } else { 1i32 };
                enc.enc_bits(q2 as u32, 1);
                let offset = (q2 as f32 - 0.5) * ((1 << (14 - fine_quant[i] - 1)) as f32) * (1.0 / 16384.0);
                if let Some(ref mut obe) = old_band_e {
                    obe[i + ch * m.nb_ebands] += offset;
                }
                error[i + ch * m.nb_ebands] -= offset;
                bits_left -= 1;
            }
        }
    }
}

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
