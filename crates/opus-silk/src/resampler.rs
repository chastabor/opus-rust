// Port of silk/resampler.c - SILK resampler (simplified for decode path)

/// Resampler state
#[derive(Clone)]
pub struct ResamplerState {
    pub s_iir: [i32; 6],
    pub s_fir_i32: [i32; 36],
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
            s_fir_i32: [0; 36],
            delay_buf: vec![0i16; 96],
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
    match r {
        8000 => 0,
        12000 => 1,
        16000 => 2,
        24000 => 3,
        48000 => 4,
        _ => 5,
    }
}

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

    let mut up2x = false;
    if fs_hz_out > fs_hz_in {
        if fs_hz_out == fs_hz_in * 2 {
            s.resampler_function = ResamplerFunc::Up2Hq;
        } else {
            s.resampler_function = ResamplerFunc::IirFir;
            up2x = true;
        }
    } else if fs_hz_out < fs_hz_in {
        s.resampler_function = ResamplerFunc::DownFir;
        // Set FIR parameters based on ratio
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

    let shift = if up2x { 1 } else { 0 };
    let fs_in_shifted = fs_hz_in << shift;
    s.inv_ratio_q16 = ((fs_in_shifted as i64) << 16) as i32 / fs_hz_out;

    s.delay_buf.resize(s.fs_in_khz.max(1) as usize, 0);
}

/// Resampler: convert from one sampling rate to another
/// Simplified version that handles the common cases for decode
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
    let copy_len = n_samples.min(input.len()).min(s.delay_buf.len() - input_delay);
    s.delay_buf[input_delay..input_delay + copy_len]
        .copy_from_slice(&input[..copy_len]);

    match s.resampler_function {
        ResamplerFunc::Copy => {
            // Same rate: just copy
            let copy_from_delay = fs_in.min(out.len()).min(s.delay_buf.len());
            out[..copy_from_delay].copy_from_slice(&s.delay_buf[..copy_from_delay]);

            let remaining = in_len.saturating_sub(fs_in);
            let out_remaining = out.len().saturating_sub(fs_out);
            let copy_len2 = remaining.min(out_remaining);
            if copy_len2 > 0 && n_samples < input.len() {
                out[fs_out..fs_out + copy_len2].copy_from_slice(&input[n_samples..n_samples + copy_len2]);
            }
        }
        ResamplerFunc::Up2Hq => {
            // 2x upsampling (simplified)
            resampler_up2_simple(s, out, &s.delay_buf.clone(), fs_in);
            if n_samples < input.len() {
                let out_offset = fs_out;
                let remain_in = &input[n_samples..];
                let remain_len = in_len - fs_in;
                if out_offset < out.len() {
                    resampler_up2_simple(s, &mut out[out_offset..], remain_in, remain_len);
                }
            }
        }
        _ => {
            // For other modes, do simple linear interpolation as fallback
            resample_linear(out, input, in_len, fs_in, fs_out);
        }
    }

    // Copy to delay buffer for next call
    if in_len >= input_delay && input_delay <= s.delay_buf.len() {
        s.delay_buf[..input_delay].copy_from_slice(&input[in_len - input_delay..in_len]);
    }
}

/// Simple 2x upsampling (zero-order hold + interpolation)
fn resampler_up2_simple(
    _s: &mut ResamplerState,
    out: &mut [i16],
    input: &[i16],
    in_len: usize,
) {
    let out_len = in_len * 2;
    for i in 0..in_len.min(input.len()) {
        let idx = i * 2;
        if idx < out.len() {
            out[idx] = input[i];
        }
        if idx + 1 < out.len() {
            // Simple interpolation with next sample
            let next = if i + 1 < input.len() { input[i + 1] } else { input[i] };
            out[idx + 1] = ((input[i] as i32 + next as i32) >> 1) as i16;
        }
    }
    // Zero remaining output
    if out_len < out.len() {
        out[out_len..].fill(0);
    }
}

/// Simple linear interpolation resampler
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
