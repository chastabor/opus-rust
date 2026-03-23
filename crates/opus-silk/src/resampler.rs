// Port of silk/resampler.c, silk/resampler_private_up2_HQ.c,
// silk/resampler_private_IIR_FIR.c - SILK resampler

use crate::{silk_smulwb, silk_smlawb, silk_smlabb, silk_smulbb, silk_sat16, silk_rshift_round, MAX_FS_KHZ};

/// Resampler state
#[derive(Clone)]
pub struct ResamplerState {
    pub s_iir: [i32; 6],
    pub s_fir_i16: [i16; 16], // RESAMPLER_ORDER_FIR_12 = 8, but allow more
    pub delay_buf: Vec<i16>,
    pub resampler_function: ResamplerFunc,
    pub batch_size: i32,
    pub inv_ratio_q16: i32,
    pub fir_order: i32,
    pub fir_fracs: i32,
    pub fs_in_khz: i32,
    pub fs_out_khz: i32,
    pub input_delay: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ResamplerFunc {
    Copy,
    Up2Hq,
    IirFir,
    DownFir,
}

impl ResamplerState {
    pub fn new() -> Self {
        Self {
            s_iir: [0; 6],
            s_fir_i16: [0; 16],
            delay_buf: vec![0i16; 48],
            resampler_function: ResamplerFunc::Copy,
            batch_size: 0,
            inv_ratio_q16: 0,
            fir_order: 0,
            fir_fracs: 0,
            fs_in_khz: 0,
            fs_out_khz: 0,
            input_delay: 0,
        }
    }
}

// Delay matrix for decoder
const DELAY_MATRIX_DEC: [[i8; 6]; 3] = [
    // out: 8  12  16  24  48  96
    [4, 0, 2, 0, 0, 0],  // in: 8
    [0, 9, 4, 7, 4, 4],  // in: 12
    [0, 3, 12, 7, 7, 7], // in: 16
];

fn rate_id(r: i32) -> usize {
    // Simple way to make [8000, 12000, 16000, 24000, 48000] to [0, 1, 2, 3, 4]
    match r {
        8000 => 0,
        12000 => 1,
        16000 => 2,
        24000 => 3,
        48000 => 4,
        _ => 5,
    }
}

// Allpass filter coefficients for 2x upsampling (from resampler_rom.h)
// C: { 1746, 14986, 39083 - 65536 } = { 1746, 14986, -26453 }
const SILK_RESAMPLER_UP2_HQ_0: [i16; 3] = [1746, 14986, -26453];
// C: { 6854, 25769, 55542 - 65536 } = { 6854, 25769, -9994 }
const SILK_RESAMPLER_UP2_HQ_1: [i16; 3] = [6854, 25769, -9994];

// FIR interpolation table (from resampler_rom.c)
const RESAMPLER_ORDER_FIR_12: usize = 8;
const SILK_RESAMPLER_FRAC_FIR_12: [[i16; 4]; 12] = [
    [189, -600, 617, 30567],
    [117, -159, -1070, 29704],
    [52, 221, -2392, 28276],
    [-4, 529, -3350, 26341],
    [-48, 758, -3956, 23973],
    [-80, 905, -4235, 21254],
    [-99, 972, -4222, 18278],
    [-107, 967, -3957, 15143],
    [-103, 896, -3487, 11950],
    [-91, 773, -2865, 8798],
    [-71, 611, -2143, 5784],
    [-46, 425, -1375, 2996],
];

/// Initialize resampler for given input/output rates
pub fn resampler_init(s: &mut ResamplerState, fs_hz_in: i32, fs_hz_out: i32, _for_enc: bool) {
    *s = ResamplerState::new();

    let in_id = rate_id(fs_hz_in);
    let out_id = rate_id(fs_hz_out);

    if in_id < 3 && out_id < 6 {
        s.input_delay = DELAY_MATRIX_DEC[in_id][out_id] as i32;
    }

    s.fs_in_khz = fs_hz_in / 1000;
    s.fs_out_khz = fs_hz_out / 1000;
    s.batch_size = s.fs_in_khz * 10; // RESAMPLER_MAX_BATCH_SIZE_MS = 10

    let mut up2x = 0i32;
    if fs_hz_out > fs_hz_in {
        if fs_hz_out == fs_hz_in * 2 {
            s.resampler_function = ResamplerFunc::Up2Hq;
        } else {
            s.resampler_function = ResamplerFunc::IirFir;
            up2x = 1;
        }
    } else if fs_hz_out < fs_hz_in {
        s.resampler_function = ResamplerFunc::DownFir;
        if fs_hz_out * 4 == fs_hz_in * 3 {
            s.fir_fracs = 3;
            s.fir_order = 12;
        } else if fs_hz_out * 3 == fs_hz_in * 2 {
            s.fir_fracs = 2;
            s.fir_order = 12;
        } else if fs_hz_out * 2 == fs_hz_in {
            s.fir_fracs = 1;
            s.fir_order = 24;
        } else if fs_hz_out * 3 == fs_hz_in {
            s.fir_fracs = 1;
            s.fir_order = 36;
        } else if fs_hz_out * 4 == fs_hz_in {
            s.fir_fracs = 1;
            s.fir_order = 36;
        } else if fs_hz_out * 6 == fs_hz_in {
            s.fir_fracs = 1;
            s.fir_order = 36;
        }
    } else {
        s.resampler_function = ResamplerFunc::Copy;
    }

    // Compute invRatio_Q16 matching the C code exactly:
    // S->invRatio_Q16 = silk_LSHIFT32(silk_DIV32(silk_LSHIFT32(Fs_Hz_in, 14 + up2x), Fs_Hz_out), 2);
    let fs_shifted = fs_hz_in << (14 + up2x);
    s.inv_ratio_q16 = (fs_shifted / fs_hz_out) << 2;

    // Round up: while silk_SMULWW(invRatio_Q16, Fs_Hz_out) < (Fs_Hz_in << up2x)
    let target = fs_hz_in << up2x;
    while crate::silk_smulww_correct(s.inv_ratio_q16, fs_hz_out) < target {
        s.inv_ratio_q16 += 1;
    }

    s.delay_buf.resize(s.fs_in_khz.max(1) as usize, 0);
}

/// Resampler: convert from one sampling rate to another
pub fn silk_resampler(
    s: &mut ResamplerState,
    out: &mut [i16],
    input: &[i16],
    in_len: usize,
) {
    let fs_in = s.fs_in_khz.max(1) as usize;
    let fs_out = s.fs_out_khz.max(1) as usize;
    let input_delay = s.input_delay as usize;

    let n_samples = fs_in - input_delay;

    // Copy to delay buffer
    let copy_len = n_samples.min(input.len()).min(s.delay_buf.len().saturating_sub(input_delay));
    if input_delay + copy_len <= s.delay_buf.len() {
        s.delay_buf[input_delay..input_delay + copy_len]
            .copy_from_slice(&input[..copy_len]);
    }

    // Copy delay_buf to stack once to avoid borrow conflict with other s fields
    let mut delay_tmp = [0i16; 48];
    let delay_len = s.delay_buf.len().min(delay_tmp.len());
    delay_tmp[..delay_len].copy_from_slice(&s.delay_buf[..delay_len]);

    match s.resampler_function {
        ResamplerFunc::Copy => {
            let copy1 = fs_in.min(out.len()).min(delay_len);
            out[..copy1].copy_from_slice(&delay_tmp[..copy1]);

            let remaining_in = in_len.saturating_sub(fs_in);
            let remaining_out = out.len().saturating_sub(fs_out);
            let copy2 = remaining_in.min(remaining_out);
            if copy2 > 0 && n_samples < input.len() {
                out[fs_out..fs_out + copy2]
                    .copy_from_slice(&input[n_samples..n_samples + copy2]);
            }
        }
        ResamplerFunc::Up2Hq => {
            silk_resampler_private_up2_hq(&mut s.s_iir, out, &delay_tmp[..delay_len], fs_in as i32);
            if n_samples < input.len() && fs_out < out.len() {
                let remain_len = (in_len as i32 - fs_in as i32).max(0);
                silk_resampler_private_up2_hq(
                    &mut s.s_iir,
                    &mut out[fs_out..],
                    &input[n_samples..],
                    remain_len,
                );
            }
        }
        ResamplerFunc::IirFir => {
            silk_resampler_private_iir_fir(s, out, &delay_tmp[..delay_len], fs_in as i32);
            if n_samples < input.len() && fs_out < out.len() {
                let remain_len = (in_len as i32 - fs_in as i32).max(0);
                silk_resampler_private_iir_fir_inner(s, &mut out[fs_out..], &input[n_samples..], remain_len);
            }
        }
        ResamplerFunc::DownFir => {
            resample_linear(out, input, in_len, fs_in, fs_out);
        }
    }

    // Copy to delay buffer for next call
    if in_len >= input_delay && input_delay <= s.delay_buf.len() && in_len <= input.len() {
        s.delay_buf[..input_delay].copy_from_slice(&input[in_len - input_delay..in_len]);
    }
}

/// Upsample by a factor 2, high quality
/// Uses 2nd order allpass filters for the 2x upsampling
fn silk_resampler_private_up2_hq(
    s: &mut [i32; 6],
    out: &mut [i16],
    input: &[i16],
    len: i32,
) {
    let len = len as usize;
    for k in 0..len.min(input.len()) {
        // Convert to Q10
        let in32 = (input[k] as i32) << 10;

        // First all-pass section for even output sample
        let y = in32 - s[0];
        let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_0[0] as i32);
        let out32_1 = s[0] + x;
        s[0] = in32 + x;

        // Second all-pass section for even output sample
        let y = out32_1 - s[1];
        let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_0[1] as i32);
        let out32_2 = s[1] + x;
        s[1] = out32_1 + x;

        // Third all-pass section for even output sample
        let y = out32_2 - s[2];
        let x = silk_smlawb(y, y, SILK_RESAMPLER_UP2_HQ_0[2] as i32);
        let out32_1 = s[2] + x;
        s[2] = out32_2 + x;

        // Apply gain in Q15, convert back to int16 and store to output
        if 2 * k < out.len() {
            out[2 * k] = silk_sat16(silk_rshift_round(out32_1, 10));
        }

        // First all-pass section for odd output sample
        let y = in32 - s[3];
        let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_1[0] as i32);
        let out32_1 = s[3] + x;
        s[3] = in32 + x;

        // Second all-pass section for odd output sample
        let y = out32_1 - s[4];
        let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_1[1] as i32);
        let out32_2 = s[4] + x;
        s[4] = out32_1 + x;

        // Third all-pass section for odd output sample
        let y = out32_2 - s[5];
        let x = silk_smlawb(y, y, SILK_RESAMPLER_UP2_HQ_1[2] as i32);
        let out32_1 = s[5] + x;
        s[5] = out32_2 + x;

        // Apply gain in Q15, convert back to int16 and store to output
        if 2 * k + 1 < out.len() {
            out[2 * k + 1] = silk_sat16(silk_rshift_round(out32_1, 10));
        }
    }
}

/// IIR/FIR polyphase resampler - interpolation stage
fn silk_resampler_private_iir_fir_interpol(
    out: &mut [i16],
    buf: &[i16],
    max_index_q16: i32,
    index_increment_q16: i32,
) -> usize {
    let mut out_idx = 0usize;
    let mut index_q16 = 0i32;

    while index_q16 < max_index_q16 {
        let table_index = silk_smulwb(index_q16 & 0xFFFF, 12) as usize;
        let buf_offset = (index_q16 >> 16) as usize;

        if buf_offset + 7 < buf.len() && out_idx < out.len() && table_index < 12 {
            let buf_ptr = &buf[buf_offset..];
            let fir_fwd = &SILK_RESAMPLER_FRAC_FIR_12[table_index];
            let fir_rev = &SILK_RESAMPLER_FRAC_FIR_12[11 - table_index];

            let mut res_q15 = silk_smulbb(buf_ptr[0] as i32, fir_fwd[0] as i32);
            res_q15 = silk_smlabb(res_q15, buf_ptr[1] as i32, fir_fwd[1] as i32);
            res_q15 = silk_smlabb(res_q15, buf_ptr[2] as i32, fir_fwd[2] as i32);
            res_q15 = silk_smlabb(res_q15, buf_ptr[3] as i32, fir_fwd[3] as i32);
            res_q15 = silk_smlabb(res_q15, buf_ptr[4] as i32, fir_rev[3] as i32);
            res_q15 = silk_smlabb(res_q15, buf_ptr[5] as i32, fir_rev[2] as i32);
            res_q15 = silk_smlabb(res_q15, buf_ptr[6] as i32, fir_rev[1] as i32);
            res_q15 = silk_smlabb(res_q15, buf_ptr[7] as i32, fir_rev[0] as i32);

            out[out_idx] = silk_sat16(silk_rshift_round(res_q15, 15));
            out_idx += 1;
        }

        index_q16 += index_increment_q16;
    }
    out_idx
}

/// IIR/FIR resampler - wrapper that handles state properly
fn silk_resampler_private_iir_fir(
    s: &mut ResamplerState,
    out: &mut [i16],
    input: &[i16],
    in_len: i32,
) {
    silk_resampler_private_iir_fir_inner(s, out, input, in_len);
}

fn silk_resampler_private_iir_fir_inner(
    s: &mut ResamplerState,
    out: &mut [i16],
    input: &[i16],
    in_len: i32,
) {
    let index_increment_q16 = s.inv_ratio_q16;
    let mut in_offset = 0usize;
    let mut out_offset = 0usize;
    let mut remaining = in_len;

    while remaining > 0 {
        let n_samples_in = remaining.min(s.batch_size) as usize;

        // Buffer for up2x output + FIR filter state. Max batch = 160, so max = 2*160+8 = 328.
        const MAX_RESAMPLER_BUF: usize = 2 * 10 * MAX_FS_KHZ + RESAMPLER_ORDER_FIR_12;
        let mut buf = [0i16; MAX_RESAMPLER_BUF];

        // Copy buffered FIR samples to start of buffer
        let fir_copy = RESAMPLER_ORDER_FIR_12.min(s.s_fir_i16.len());
        buf[..fir_copy].copy_from_slice(&s.s_fir_i16[..fir_copy]);

        // Upsample 2x
        let available_in = input.len().saturating_sub(in_offset);
        let actual_in = n_samples_in.min(available_in);
        silk_resampler_private_up2_hq(
            &mut s.s_iir,
            &mut buf[RESAMPLER_ORDER_FIR_12..],
            &input[in_offset..in_offset + actual_in],
            actual_in as i32,
        );

        let max_index_q16 = (n_samples_in as i32) << (16 + 1); // + 1 because 2x upsampling

        let n_written = silk_resampler_private_iir_fir_interpol(
            &mut out[out_offset..],
            &buf,
            max_index_q16,
            index_increment_q16,
        );
        out_offset += n_written;

        in_offset += n_samples_in;
        remaining -= n_samples_in as i32;

        if remaining > 0 {
            // Copy last part of filtered signal to beginning of buffer for next iteration
            let src_start = n_samples_in << 1;
            let copy_len = RESAMPLER_ORDER_FIR_12.min(buf.len().saturating_sub(src_start));
            // Save to s_fir_i16 temporarily
            s.s_fir_i16[..copy_len].copy_from_slice(&buf[src_start..src_start + copy_len]);
        } else {
            // Save state for next call
            let src_start = n_samples_in << 1;
            let copy_len = RESAMPLER_ORDER_FIR_12.min(buf.len().saturating_sub(src_start));
            s.s_fir_i16[..copy_len].copy_from_slice(&buf[src_start..src_start + copy_len]);
        }
    }
}

/// Simple linear interpolation resampler (fallback for down-FIR)
fn resample_linear(
    out: &mut [i16],
    input: &[i16],
    in_len: usize,
    _fs_in: usize,
    _fs_out: usize,
) {
    if in_len == 0 || out.is_empty() || input.is_empty() {
        return;
    }

    let out_len = out.len();
    let ratio = ((in_len as u64) << 16) / out_len.max(1) as u64;

    let mut in_pos_q16: u64 = 0;
    for i in 0..out_len {
        let in_idx = (in_pos_q16 >> 16) as usize;
        let frac = (in_pos_q16 & 0xFFFF) as i32;

        if in_idx + 1 < input.len() {
            out[i] = ((input[in_idx] as i32 * (65536 - frac) + input[in_idx + 1] as i32 * frac) >> 16) as i16;
        } else if in_idx < input.len() {
            out[i] = input[in_idx];
        } else {
            out[i] = 0;
        }

        in_pos_q16 += ratio;
    }
}
