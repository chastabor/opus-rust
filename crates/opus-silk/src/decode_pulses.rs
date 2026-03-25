// Port of silk/decode_pulses.c, silk/shell_coder.c, silk/code_signs.c

use crate::tables::*;
use crate::*;
use opus_range_coder::EcCtx;

/// Decode quantization indices of excitation
pub fn silk_decode_pulses(
    ps_range_dec: &mut EcCtx,
    pulses: &mut [i16],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: i32,
) {
    // Decode rate level
    let rate_level_index =
        ps_range_dec.dec_icdf(&SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize], 8);

    // Calculate number of shell blocks
    let mut iter = frame_length as usize >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length as usize {
        iter += 1;
    }

    // Sum-Weighted-Pulses Decoding
    let mut sum_pulses = [0i32; MAX_NB_SHELL_BLOCKS];
    let mut n_lshifts = [0i32; MAX_NB_SHELL_BLOCKS];

    let cdf = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    for i in 0..iter {
        n_lshifts[i] = 0;
        sum_pulses[i] = ps_range_dec.dec_icdf(cdf, 8) as i32;

        // LSB indication
        while sum_pulses[i] == SILK_MAX_PULSES as i32 + 1 {
            n_lshifts[i] += 1;
            // When we've already got 10 LSBs, shift the table
            let offset = if n_lshifts[i] == 10 { 1 } else { 0 };
            sum_pulses[i] = ps_range_dec
                .dec_icdf(&SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1][offset..], 8)
                as i32;
        }
    }

    // Shell decoding
    for i in 0..iter {
        let offset = i * SHELL_CODEC_FRAME_LENGTH;
        if sum_pulses[i] > 0 {
            silk_shell_decoder(&mut pulses[offset..], ps_range_dec, sum_pulses[i]);
        } else {
            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                pulses[offset + j] = 0;
            }
        }
    }

    // LSB Decoding
    for i in 0..iter {
        if n_lshifts[i] > 0 {
            let n_ls = n_lshifts[i];
            let offset = i * SHELL_CODEC_FRAME_LENGTH;
            for k in 0..SHELL_CODEC_FRAME_LENGTH {
                let mut abs_q = pulses[offset + k] as i32;
                for _j in 0..n_ls {
                    abs_q = (abs_q << 1) + ps_range_dec.dec_icdf(&SILK_LSB_ICDF, 8) as i32;
                }
                pulses[offset + k] = abs_q as i16;
            }
            sum_pulses[i] |= n_ls << 5;
        }
    }

    // Decode and add signs
    silk_decode_signs(
        ps_range_dec,
        pulses,
        frame_length,
        signal_type,
        quant_offset_type,
        &sum_pulses,
    );
}

/// Shell decoder - operates on one shell code frame of 16 pulses
fn silk_shell_decoder(pulses0: &mut [i16], ps_range_dec: &mut EcCtx, pulses4: i32) {
    let mut pulses3 = [0i16; 2];
    let mut pulses2 = [0i16; 4];
    let mut pulses1 = [0i16; 8];

    let (a, b) = decode_split(ps_range_dec, pulses4, &SILK_SHELL_CODE_TABLE3);
    pulses3[0] = a;
    pulses3[1] = b;

    let (a, b) = decode_split(ps_range_dec, pulses3[0] as i32, &SILK_SHELL_CODE_TABLE2);
    pulses2[0] = a;
    pulses2[1] = b;

    let (a, b) = decode_split(ps_range_dec, pulses2[0] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[0] = a;
    pulses1[1] = b;
    let (a, b) = decode_split(ps_range_dec, pulses1[0] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[0] = a;
    pulses0[1] = b;
    let (a, b) = decode_split(ps_range_dec, pulses1[1] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[2] = a;
    pulses0[3] = b;

    let (a, b) = decode_split(ps_range_dec, pulses2[1] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[2] = a;
    pulses1[3] = b;
    let (a, b) = decode_split(ps_range_dec, pulses1[2] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[4] = a;
    pulses0[5] = b;
    let (a, b) = decode_split(ps_range_dec, pulses1[3] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[6] = a;
    pulses0[7] = b;

    let (a, b) = decode_split(ps_range_dec, pulses3[1] as i32, &SILK_SHELL_CODE_TABLE2);
    pulses2[2] = a;
    pulses2[3] = b;

    let (a, b) = decode_split(ps_range_dec, pulses2[2] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[4] = a;
    pulses1[5] = b;
    let (a, b) = decode_split(ps_range_dec, pulses1[4] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[8] = a;
    pulses0[9] = b;
    let (a, b) = decode_split(ps_range_dec, pulses1[5] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[10] = a;
    pulses0[11] = b;

    let (a, b) = decode_split(ps_range_dec, pulses2[3] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[6] = a;
    pulses1[7] = b;
    let (a, b) = decode_split(ps_range_dec, pulses1[6] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[12] = a;
    pulses0[13] = b;
    let (a, b) = decode_split(ps_range_dec, pulses1[7] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[14] = a;
    pulses0[15] = b;
}

fn decode_split(ps_range_dec: &mut EcCtx, p: i32, shell_table: &[u8]) -> (i16, i16) {
    if p > 0 {
        let offset = SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
        let child1 = ps_range_dec.dec_icdf(&shell_table[offset..], 8) as i16;
        let child2 = p as i16 - child1;
        (child1, child2)
    } else {
        (0, 0)
    }
}

/// Decode signs of excitation
fn silk_decode_signs(
    ps_range_dec: &mut EcCtx,
    pulses: &mut [i16],
    length: i32,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    let mut icdf = [0u8; 2];
    icdf[1] = 0;

    let icdf_offset = 7 * (quant_offset_type + signal_type * 2) as usize;
    let n_blocks =
        (length as usize + SHELL_CODEC_FRAME_LENGTH / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH;

    for i in 0..n_blocks {
        let p = sum_pulses[i];
        if p > 0 {
            icdf[0] = SILK_SIGN_ICDF[icdf_offset + ((p & 0x1F) as usize).min(6)];
            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                let idx = i * SHELL_CODEC_FRAME_LENGTH + j;
                if idx < pulses.len() && pulses[idx] > 0 {
                    let sign = ps_range_dec.dec_icdf(&icdf, 8);
                    // silk_dec_map: (sign << 1) - 1
                    let multiplier = (sign as i16) * 2 - 1;
                    pulses[idx] *= multiplier;
                }
            }
        }
    }
}
