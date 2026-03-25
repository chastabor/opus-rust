// Port of silk/encode_pulses.c, silk/shell_coder.c (encoder), silk/code_signs.c (encoder)
// This is the mirror of decode_pulses.rs.

use crate::tables::*;
use crate::*;
use opus_range_coder::EcCtx;

/// Maximum pulses per block at each combining stage
const SILK_MAX_PULSES_TABLE: [i32; 4] = [8, 10, 12, 16];

/// Rate level bit costs (Q5) per signal type for choosing the best rate level
const SILK_RATE_LEVELS_BITS_Q5: [[i32; 9]; 2] = [
    [131, 74, 141, 79, 80, 138, 95, 104, 134],
    [95, 99, 91, 125, 93, 76, 123, 115, 123],
];

/// Bit costs (Q5) for pulses per block at each rate level
const SILK_PULSES_PER_BLOCK_BITS_Q5: [[i32; 18]; 9] = [
    [
        31, 57, 107, 160, 205, 205, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ],
    [
        69, 47, 67, 111, 166, 205, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ],
    [
        82, 74, 79, 95, 109, 128, 145, 160, 173, 205, 205, 205, 224, 255, 255, 224, 255, 224,
    ],
    [
        125, 74, 59, 69, 97, 141, 182, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ],
    [
        173, 115, 85, 73, 76, 92, 115, 145, 173, 205, 224, 224, 255, 255, 255, 255, 255, 255,
    ],
    [
        166, 134, 113, 102, 101, 102, 107, 118, 125, 138, 145, 155, 166, 182, 192, 192, 205, 150,
    ],
    [
        224, 182, 134, 101, 83, 79, 85, 97, 120, 145, 173, 205, 224, 255, 255, 255, 255, 255,
    ],
    [
        255, 224, 192, 150, 120, 101, 92, 89, 93, 102, 118, 134, 160, 182, 192, 224, 224, 224,
    ],
    [
        255, 224, 224, 182, 155, 134, 118, 109, 104, 102, 106, 111, 118, 131, 145, 160, 173, 131,
    ],
];

/// Check if combining two child pulse vectors exceeds max_pulses.
/// Returns true if any sum exceeds max_pulses (overflow).
fn combine_and_check(
    pulses_comb: &mut [i32],
    pulses_in: &[i32],
    max_pulses: i32,
    len: usize,
) -> bool {
    let mut overflow = false;
    for k in 0..len {
        let sum = pulses_in[2 * k] + pulses_in[2 * k + 1];
        if sum > max_pulses {
            overflow = true;
        }
        pulses_comb[k] = sum;
    }
    overflow
}

/// Combine pulses without overflow check
fn combine_pulses(out: &mut [i32], inp: &[i32], len: usize) {
    for k in 0..len {
        out[k] = inp[2 * k] + inp[2 * k + 1];
    }
}

/// Shell encoder - encode one shell code frame of 16 non-negative pulse amplitudes
fn silk_shell_encoder(enc: &mut EcCtx, pulses0: &[i32]) {
    let mut pulses1 = [0i32; 8];
    let mut pulses2 = [0i32; 4];
    let mut pulses3 = [0i32; 2];
    let mut pulses4 = [0i32; 1];

    // Build the tree from leaves to root
    combine_pulses(&mut pulses1, pulses0, 8);
    combine_pulses(&mut pulses2, &pulses1, 4);
    combine_pulses(&mut pulses3, &pulses2, 2);
    combine_pulses(&mut pulses4, &pulses3, 1);

    // Encode from root to leaves (same order as C reference)
    encode_split(enc, pulses3[0], pulses4[0], &SILK_SHELL_CODE_TABLE3);

    encode_split(enc, pulses2[0], pulses3[0], &SILK_SHELL_CODE_TABLE2);

    encode_split(enc, pulses1[0], pulses2[0], &SILK_SHELL_CODE_TABLE1);
    encode_split(enc, pulses0[0], pulses1[0], &SILK_SHELL_CODE_TABLE0);
    encode_split(enc, pulses0[2], pulses1[1], &SILK_SHELL_CODE_TABLE0);

    encode_split(enc, pulses1[2], pulses2[1], &SILK_SHELL_CODE_TABLE1);
    encode_split(enc, pulses0[4], pulses1[2], &SILK_SHELL_CODE_TABLE0);
    encode_split(enc, pulses0[6], pulses1[3], &SILK_SHELL_CODE_TABLE0);

    encode_split(enc, pulses2[2], pulses3[1], &SILK_SHELL_CODE_TABLE2);

    encode_split(enc, pulses1[4], pulses2[2], &SILK_SHELL_CODE_TABLE1);
    encode_split(enc, pulses0[8], pulses1[4], &SILK_SHELL_CODE_TABLE0);
    encode_split(enc, pulses0[10], pulses1[5], &SILK_SHELL_CODE_TABLE0);

    encode_split(enc, pulses1[6], pulses2[3], &SILK_SHELL_CODE_TABLE1);
    encode_split(enc, pulses0[12], pulses1[6], &SILK_SHELL_CODE_TABLE0);
    encode_split(enc, pulses0[14], pulses1[7], &SILK_SHELL_CODE_TABLE0);
}

/// Encode a parent-to-child split
fn encode_split(enc: &mut EcCtx, p_child1: i32, p: i32, shell_table: &[u8]) {
    if p > 0 {
        let offset = SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
        enc.enc_icdf(p_child1 as usize, &shell_table[offset..], 8);
    }
}

/// Encode signs of excitation pulses.
/// silk_enc_map(a) = (a >> 15) + 1: maps negative to 0, positive to 1.
fn silk_encode_signs(
    enc: &mut EcCtx,
    pulses: &[i8],
    length: i32,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    let mut icdf = [0u8; 2];
    icdf[1] = 0;

    let icdf_offset = 7 * (quant_offset_type + (signal_type << 1)) as usize;
    let n_blocks =
        ((length as usize) + SHELL_CODEC_FRAME_LENGTH / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH;

    for (i, sum_p) in sum_pulses.iter().enumerate().take(n_blocks) {
        let p = *sum_p;
        if p > 0 {
            icdf[0] = SILK_SIGN_ICDF[icdf_offset + ((p & 0x1F) as usize).min(6)];
            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                let idx = i * SHELL_CODEC_FRAME_LENGTH + j;
                if idx < pulses.len() && pulses[idx] != 0 {
                    // silk_enc_map: (a >> 15) + 1 for i8 means: negative->0, positive->1
                    let sign = if pulses[idx] > 0 { 1usize } else { 0usize };
                    enc.enc_icdf(sign, &icdf, 8);
                }
            }
        }
    }
}

/// Encode quantization indices of excitation.
/// Port of silk_encode_pulses from silk/encode_pulses.c.
pub fn silk_encode_pulses(
    enc: &mut EcCtx,
    pulses: &[i8],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: i32,
) {
    // Calculate number of shell blocks
    let mut iter = (frame_length as usize) >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length as usize {
        iter += 1;
    }

    // Take absolute value of pulses, pad to full shell blocks
    let total_len = iter * SHELL_CODEC_FRAME_LENGTH;
    let mut abs_pulses = vec![0i32; total_len];
    for i in 0..frame_length as usize {
        abs_pulses[i] = (pulses[i] as i32).abs();
    }

    // Compute sum pulses per shell code frame, with right-shift if needed
    let mut sum_pulses = vec![0i32; iter];
    let mut n_rshifts = vec![0i32; iter];
    let mut pulses_comb = [0i32; 8];

    for i in 0..iter {
        n_rshifts[i] = 0;
        let base = i * SHELL_CODEC_FRAME_LENGTH;

        loop {
            let ap = &abs_pulses[base..base + SHELL_CODEC_FRAME_LENGTH];

            // 1+1 -> 2
            let mut scale_down =
                combine_and_check(&mut pulses_comb, ap, SILK_MAX_PULSES_TABLE[0], 8) as i32;
            // 2+2 -> 4
            let mut tmp4 = [0i32; 4];
            scale_down +=
                combine_and_check(&mut tmp4, &pulses_comb[..8], SILK_MAX_PULSES_TABLE[1], 4) as i32;
            // 4+4 -> 8
            let mut tmp2 = [0i32; 2];
            scale_down += combine_and_check(&mut tmp2, &tmp4, SILK_MAX_PULSES_TABLE[2], 2) as i32;
            // 8+8 -> 16
            let mut tmp1 = [0i32; 1];
            scale_down += combine_and_check(&mut tmp1, &tmp2, SILK_MAX_PULSES_TABLE[3], 1) as i32;
            sum_pulses[i] = tmp1[0];

            if scale_down != 0 {
                // Downscale
                n_rshifts[i] += 1;
                for k in 0..SHELL_CODEC_FRAME_LENGTH {
                    abs_pulses[base + k] >>= 1;
                }
            } else {
                break;
            }
        }
    }

    // Find best rate level
    let mut min_sum_bits_q5 = i32::MAX;
    let mut rate_level_index = 0usize;
    let sig_half = (signal_type >> 1) as usize;

    for (k, n_bits_ptr) in SILK_PULSES_PER_BLOCK_BITS_Q5
        .iter()
        .enumerate()
        .take(N_RATE_LEVELS - 1)
    {
        let mut sum_bits_q5 = SILK_RATE_LEVELS_BITS_Q5[sig_half][k];
        for i in 0..iter {
            if n_rshifts[i] > 0 {
                sum_bits_q5 += n_bits_ptr[SILK_MAX_PULSES + 1];
            } else {
                sum_bits_q5 += n_bits_ptr[sum_pulses[i] as usize];
            }
        }
        if sum_bits_q5 < min_sum_bits_q5 {
            min_sum_bits_q5 = sum_bits_q5;
            rate_level_index = k;
        }
    }

    // Encode rate level
    enc.enc_icdf(rate_level_index, &SILK_RATE_LEVELS_ICDF[sig_half], 8);

    // Sum-Weighted-Pulses Encoding
    let cdf = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    for i in 0..iter {
        if n_rshifts[i] == 0 {
            enc.enc_icdf(sum_pulses[i] as usize, cdf, 8);
        } else {
            enc.enc_icdf(SILK_MAX_PULSES + 1, cdf, 8);
            for _k in 0..n_rshifts[i] - 1 {
                enc.enc_icdf(
                    SILK_MAX_PULSES + 1,
                    &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                    8,
                );
            }
            enc.enc_icdf(
                sum_pulses[i] as usize,
                &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                8,
            );
        }
    }

    // Shell Encoding
    for (i, sum_p) in sum_pulses.iter().enumerate().take(iter) {
        if *sum_p > 0 {
            let base = i * SHELL_CODEC_FRAME_LENGTH;
            silk_shell_encoder(enc, &abs_pulses[base..base + SHELL_CODEC_FRAME_LENGTH]);
        }
    }

    // LSB Encoding
    for (i, n_rs) in n_rshifts.iter().enumerate().take(iter) {
        if *n_rs > 0 {
            let base = i * SHELL_CODEC_FRAME_LENGTH;
            let n_ls = *n_rs - 1;
            for k in 0..SHELL_CODEC_FRAME_LENGTH {
                let idx = base + k;
                let abs_q = if idx < pulses.len() {
                    (pulses[idx] as i32).abs()
                } else {
                    0
                };
                // Encode LSBs from most significant to least significant
                for j in (0..=n_ls).rev() {
                    let bit = (abs_q >> j) & 1;
                    enc.enc_icdf(bit as usize, &SILK_LSB_ICDF, 8);
                }
            }
        }
    }

    // Encode signs
    silk_encode_signs(
        enc,
        pulses,
        frame_length,
        signal_type,
        quant_offset_type,
        &sum_pulses,
    );
}
