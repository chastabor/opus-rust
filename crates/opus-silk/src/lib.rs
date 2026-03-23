// SILK codec crate for Opus audio codec
// Ported from the C reference implementation

pub mod tables;
pub mod decode_indices;
pub mod decode_pulses;
pub mod nlsf;
pub mod decode_params;
pub mod decode_core;
pub mod plc;
pub mod cng;
pub mod resampler;
pub mod stereo;
pub mod decoder;

// Encoder modules
pub mod gain_quant;
pub mod encode_indices;
pub mod encode_pulses;
pub mod nsq;
pub mod nsq_del_dec;
pub mod nlsf_encode;
pub mod lpc_analysis;
pub mod pitch_analysis;
pub mod noise_shape_analysis;
pub mod encoder;
pub mod signal_processing;
pub mod vad;

// Re-export the main decoder
pub use decoder::{SilkDecoder, SilkDecControl};

// Re-export the main encoder
pub use encoder::SilkEncoder;

// ---- Constants (from silk/define.h) ----

pub const DECODER_NUM_CHANNELS: usize = 2;
pub const MAX_FRAMES_PER_PACKET: usize = 3;

pub const MAX_FS_KHZ: usize = 16;
pub const MAX_API_FS_KHZ: usize = 48;

// Signal types
pub const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
pub const TYPE_UNVOICED: i32 = 1;
pub const TYPE_VOICED: i32 = 2;

// Conditional coding types
pub const CODE_INDEPENDENTLY: i32 = 0;
pub const CODE_INDEPENDENTLY_NO_LTP_SCALING: i32 = 1;
pub const CODE_CONDITIONALLY: i32 = 2;

// Frame dimensions
pub const MAX_NB_SUBFR: usize = 4;
pub const LTP_MEM_LENGTH_MS: usize = 20;
pub const SUB_FRAME_LENGTH_MS: usize = 5;
pub const MAX_SUB_FRAME_LENGTH: usize = SUB_FRAME_LENGTH_MS * MAX_FS_KHZ;
pub const MAX_FRAME_LENGTH_MS: usize = SUB_FRAME_LENGTH_MS * MAX_NB_SUBFR;
pub const MAX_FRAME_LENGTH: usize = MAX_FRAME_LENGTH_MS * MAX_FS_KHZ;

// LPC
pub const MAX_LPC_ORDER: usize = 16;
pub const MIN_LPC_ORDER: usize = 10;
pub const LTP_ORDER: usize = 5;
pub const NB_LTP_CBKS: usize = 3;
pub const MAX_LPC_STABILIZE_ITERATIONS: usize = 16;

// Shell codec
pub const SHELL_CODEC_FRAME_LENGTH: usize = 16;
pub const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
pub const MAX_NB_SHELL_BLOCKS: usize = MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH;

// Pulse coding
pub const N_RATE_LEVELS: usize = 10;
pub const SILK_MAX_PULSES: usize = 16;

// Gain quantization
pub const MIN_QGAIN_DB: i32 = 2;
pub const MAX_QGAIN_DB: i32 = 88;
pub const N_LEVELS_QGAIN: i32 = 64;
pub const MAX_DELTA_GAIN_QUANT: i32 = 36;
pub const MIN_DELTA_GAIN_QUANT: i32 = -4;

// Quantization offsets
pub const OFFSET_VL_Q10: i32 = 32;
pub const OFFSET_VH_Q10: i32 = 100;
pub const OFFSET_UVL_Q10: i32 = 100;
pub const OFFSET_UVH_Q10: i32 = 240;
pub const QUANT_LEVEL_ADJUST_Q10: i32 = 80;

// NLSF
pub const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
// NLSF_QUANT_LEVEL_ADJ = 0.1 in Q10 = 102
pub const NLSF_QUANT_LEVEL_ADJ_Q10: i32 = 102;

// Stereo
pub const STEREO_QUANT_TAB_SIZE: usize = 16;
pub const STEREO_QUANT_SUB_STEPS: i32 = 5;
pub const STEREO_INTERP_LEN_MS: i32 = 8;

// BWE after loss
pub const BWE_AFTER_LOSS_Q16: i32 = 63570;

// CNG
pub const CNG_BUF_MASK_MAX: usize = 255;
pub const CNG_GAIN_SMTH_Q16: i32 = 4634;
pub const CNG_GAIN_SMTH_THRESHOLD_Q16: i32 = 46396;
pub const CNG_NLSF_SMTH_Q16: i32 = 16348;

// Pitch estimation
pub const PE_MIN_LAG_MS: i32 = 2;
pub const PE_MAX_LAG_MS: i32 = 18;
pub const PE_MAX_NB_SUBFR: usize = 4;
pub const PE_NB_CBKS_STAGE2_EXT: usize = 11;
pub const PE_NB_CBKS_STAGE3_MAX: usize = 34;
pub const PE_NB_CBKS_STAGE2_10MS: usize = 3;
pub const PE_NB_CBKS_STAGE3_10MS: usize = 12;
pub const PE_SUBFR_LENGTH_MS: i32 = 5;
pub const PE_LTP_MEM_LENGTH_MS: i32 = 20;
pub const PE_D_SRCH_LENGTH: usize = 24;
pub const PE_NB_STAGE3_LAGS: usize = 5;
pub const PE_NB_CBKS_STAGE2: usize = 3;
pub const PE_NB_CBKS_STAGE3_MID: usize = 24;
pub const PE_NB_CBKS_STAGE3_MIN: usize = 16;
pub const PE_SHORTLAG_BIAS: i32 = 1638;       // 0.2 in Q13 = SILK_FIX_CONST(0.2, 13)
pub const PE_PREVLAG_BIAS: i32 = 1638;        // 0.2 in Q13 = SILK_FIX_CONST(0.2, 13)
pub const PE_FLATCONTOUR_BIAS: i32 = 1638;    // 0.05 in Q15 = SILK_FIX_CONST(0.05, 15)

// LSF cosine table size
pub const LSF_COS_TAB_SZ_FIX: usize = 128;

// Packet loss flags
pub const FLAG_DECODE_NORMAL: i32 = 0;
pub const FLAG_PACKET_LOST: i32 = 1;
pub const FLAG_DECODE_LBRR: i32 = 2;

// Max prediction power gain
pub const MAX_PREDICTION_POWER_GAIN: f32 = 1e4;

// ---- Fixed-point macros as inline functions ----

/// Signed multiply, keep top bits: (a * (i16)b) >> 16
/// Matches C: silk_SMULWB(a32, b32) = (a32 * (opus_int64)(opus_int16)(b32)) >> 16
/// The b argument is truncated to 16 bits (sign-extended).
#[inline(always)]
pub fn silk_smulwb(a: i32, b: i32) -> i32 {
    ((a as i64 * (b as i16) as i64) >> 16) as i32
}

/// Signed multiply-accumulate, keep top: a + (b * (i16)c) >> 16
/// Matches C: silk_SMLAWB(a32, b32, c32) = a32 + ((b32 * (opus_int64)(opus_int16)(c32)) >> 16)
/// The c argument is truncated to 16 bits (sign-extended).
#[inline(always)]
pub fn silk_smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((b as i64 * (c as i16) as i64) >> 16) as i32)
}

/// silk_SMULWW: multiply two 32-bit numbers, return bits [47:16]
#[inline(always)]
pub fn silk_smulww_correct(a: i32, b: i32) -> i32 {
    // The C code for silk_SMULWW computes (a*b)>>16 but handles the full 32x32 case
    // It's: SMULWB(a,b) + a * RSHIFT_ROUND(b, 16)
    // Which equals (a*(b&0xFFFF))>>16 + a*(b>>16)  approximately
    // Simplest correct approach:
    ((a as i64 * b as i64) >> 16) as i32
}

/// Signed multiply, both 16-bit: (i16)a * (i16)b
/// Matches C: silk_SMULBB(a32, b32) = (opus_int16)(a32) * (opus_int16)(b32)
/// Both arguments truncated to 16 bits (sign-extended).
#[inline(always)]
pub fn silk_smulbb(a: i32, b: i32) -> i32 {
    (a as i16 as i32).wrapping_mul(b as i16 as i32)
}

/// Multiply-accumulate with 16-bit operands: a + (i16)b * (i16)c
/// Matches C: silk_SMLABB(a32, b32, c32) = a32 + (opus_int16)(b32) * (opus_int16)(c32)
/// The b and c arguments are truncated to 16 bits (sign-extended).
#[inline(always)]
pub fn silk_smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add((b as i16 as i32).wrapping_mul(c as i16 as i32))
}

/// Signed multiply high: (a * b) >> 32
#[inline(always)]
pub fn silk_smmul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 32) as i32
}

/// Right shift with rounding
#[inline(always)]
pub fn silk_rshift_round(a: i32, shift: i32) -> i32 {
    if shift == 0 {
        a
    } else if shift == 1 {
        (a >> 1).wrapping_add(a & 1)
    } else {
        (a.wrapping_add(1 << (shift - 1))) >> shift
    }
}

/// Right shift with rounding for 64-bit
#[inline(always)]
pub fn silk_rshift_round64(a: i64, shift: i32) -> i64 {
    if shift == 0 {
        a
    } else if shift <= 0 {
        a
    } else {
        (a.wrapping_add(1i64 << (shift - 1))) >> shift
    }
}

/// Saturating addition for i32
#[inline(always)]
pub fn silk_add_sat32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// Saturating left shift
#[inline(always)]
pub fn silk_lshift_sat32(a: i32, shift: i32) -> i32 {
    let r = (a as i64) << shift;
    if r > i32::MAX as i64 {
        i32::MAX
    } else if r < i32::MIN as i64 {
        i32::MIN
    } else {
        r as i32
    }
}

/// Saturate to 16-bit range
#[inline(always)]
pub fn silk_sat16(a: i32) -> i16 {
    a.clamp(-32768, 32767) as i16
}

/// Add with left shift: a + (b << shift)
#[inline(always)]
pub fn silk_add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b << shift)
}

/// Subtract with left shift
#[inline(always)]
pub fn silk_sub_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_sub(b << shift)
}

/// silk_ADD_LSHIFT(a, b, shift) (for use as combined add+lshift for smaller values)
#[inline(always)]
pub fn silk_add_lshift(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b << shift)
}

/// Count leading zeros
#[inline(always)]
pub fn silk_clz32(x: i32) -> i32 {
    if x == 0 { 32 } else { (x as u32).leading_zeros() as i32 }
}

/// Integer RAND: 907633515 + (seed * 196314165)
#[inline(always)]
pub fn silk_rand(seed: i32) -> i32 {
    (seed as u32).wrapping_mul(196314165).wrapping_add(907633515) as i32
}

/// silk_DIV32_16: integer division of 32-bit by 16-bit
#[inline(always)]
pub fn silk_div32_16(a: i32, b: i16) -> i32 {
    a / (b as i32)
}

/// silk_DIV32: integer division
#[inline(always)]
pub fn silk_div32(a: i32, b: i32) -> i32 {
    if b == 0 { if a >= 0 { i32::MAX } else { i32::MIN } } else { a / b }
}

/// silk_INVERSE32_varQ: compute 1/a in variable Q domain
#[inline(always)]
pub fn silk_inverse32_varq(b: i32, q_res: i32) -> i32 {
    if b == 0 {
        return i32::MAX;
    }
    let b_abs = b.unsigned_abs();
    let lz = b_abs.leading_zeros() as i32;
    let b_norm = (b_abs as i32) << (lz - 1); // Q30
    // result = (1 << q_res) / b
    // = (1 << (q_res - 30)) * (1<<30 / b_norm) (approx)
    let result_shift = q_res - (30 - lz + 1);
    // Use 64-bit division
    let one = 1i64 << (result_shift + 30);
    let result = (one / b_norm as i64) as i32;
    if b < 0 { -result } else { result }
}

/// silk_DIV32_varQ: compute a/b with variable Q
#[inline(always)]
pub fn silk_div32_varq(a: i32, b: i32, q_res: i32) -> i32 {
    if b == 0 {
        return if a >= 0 { i32::MAX } else { i32::MIN };
    }
    let a64 = (a as i64) << q_res;
    (a64 / b as i64) as i32
}

/// silk_CLZ_FRAC: get leading zeros and 7-bit fraction
#[inline(always)]
fn silk_clz_frac(input: i32) -> (i32, i32) {
    let lzeros = silk_clz32(input);
    // silk_ROR32(in, 24 - lzeros) & 0x7f
    let shift = 24 - lzeros;
    let rotated = if shift >= 0 && shift < 32 {
        ((input as u32).wrapping_shr(shift as u32) | (input as u32).wrapping_shl((32 - shift) as u32)) as i32
    } else if shift < 0 {
        let neg_shift = (-shift) as u32;
        ((input as u32).wrapping_shl(neg_shift) | (input as u32).wrapping_shr((32 - neg_shift) as u32)) as i32
    } else {
        input
    };
    (lzeros, rotated & 0x7f)
}

/// silk_lin2log: Approximation of 128 * log2() (very close inverse of silk_log2lin())
#[inline(always)]
pub fn silk_lin2log(in_lin: i32) -> i32 {
    if in_lin <= 0 {
        return 0;
    }
    let (lz, frac_q7) = silk_clz_frac(in_lin);
    // silk_ADD_LSHIFT32(silk_SMLAWB(frac_Q7, silk_MUL(frac_Q7, 128 - frac_Q7), 179), 31 - lz, 7)
    let parabolic = silk_smlawb(frac_q7, frac_q7.wrapping_mul(128 - frac_q7), 179);
    parabolic + ((31 - lz) << 7)
}

/// silk_log2lin: Approximation of 2^() (very close inverse of silk_lin2log())
#[inline(always)]
pub fn silk_log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    }
    if in_log_q7 >= 3967 {
        return i32::MAX;
    }

    let int_part = in_log_q7 >> 7;
    let frac_q7 = in_log_q7 & 0x7F;
    let mut out = 1i32 << int_part;

    // Piece-wise parabolic approximation
    // frac_correction = silk_SMLAWB(frac_Q7, silk_SMULBB(frac_Q7, 128 - frac_Q7), -174)
    let frac_correction = silk_smlawb(frac_q7, frac_q7.wrapping_mul(128 - frac_q7), -174);

    if in_log_q7 < 2048 {
        // out = silk_ADD_RSHIFT32(out, silk_MUL(out, frac_correction), 7)
        // silk_ADD_RSHIFT32(a, b, c) = a + (b >> c)
        out = out + (out.wrapping_mul(frac_correction) >> 7);
    } else {
        // out = silk_MLA(out, silk_RSHIFT(out, 7), frac_correction)
        // silk_MLA(a, b, c) = a + b * c
        out = out + (out >> 7).wrapping_mul(frac_correction);
    }
    out
}

/// silk_SQRT_APPROX: approximate integer square root (matching C reference)
#[inline(always)]
pub fn silk_sqrt_approx(x: i32) -> i32 {
    if x <= 0 { return 0; }
    let (lz, frac_q7) = silk_clz_frac(x);

    let mut y = if lz & 1 != 0 { 32768i32 } else { 46214i32 };

    // get scaling right
    y >>= lz >> 1;

    // increment using fractional part of input
    // silk_SMLAWB(y, y, silk_SMULBB(213, frac_Q7))
    y = silk_smlawb(y, y, 213i32.wrapping_mul(frac_q7));

    y
}

/// Saturating add for i16
#[inline(always)]
pub fn silk_add_sat16(a: i16, b: i16) -> i16 {
    a.saturating_add(b)
}

/// silk_sum_sqr_shift: compute sum of squares with adaptive shift
pub fn silk_sum_sqr_shift(energy: &mut i32, shift: &mut i32, data: &[i16], len: usize) {
    let mut nrg = 0i32;
    let mut shft = 0i32;

    for i in 0..len {
        let val = data[i] as i32;
        nrg = nrg.wrapping_add(((val * val) >> shft) as i32);
        if nrg < 0 {
            // overflow, increase shift
            nrg = ((nrg as u32) >> 2) as i32;
            shft += 2;
        }
    }

    // Make sure we don't have negative
    if nrg < 0 {
        nrg = i32::MAX;
    }

    *energy = nrg;
    *shift = shft;
}

// ---- Struct definitions (from silk/structs.h) ----

/// Side information indices
#[derive(Clone, Default)]
pub struct SideInfoIndices {
    pub gains_indices: [i8; MAX_NB_SUBFR],
    pub ltp_index: [i8; MAX_NB_SUBFR],
    pub nlsf_indices: [i8; MAX_LPC_ORDER + 1],
    pub lag_index: i16,
    pub contour_index: i8,
    pub signal_type: i8,
    pub quant_offset_type: i8,
    pub nlsf_interp_coef_q2: i8,
    pub per_index: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,
}

/// Decoder control parameters (per-frame)
#[derive(Clone)]
pub struct DecoderControl {
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub gains_q16: [i32; MAX_NB_SUBFR],
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    pub ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub ltp_scale_q14: i32,
}

impl Default for DecoderControl {
    fn default() -> Self {
        Self {
            pitch_l: [0; MAX_NB_SUBFR],
            gains_q16: [0; MAX_NB_SUBFR],
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; LTP_ORDER * MAX_NB_SUBFR],
            ltp_scale_q14: 0,
        }
    }
}

/// PLC state
#[derive(Clone)]
pub struct PlcState {
    pub pitch_l_q8: i32,
    pub ltp_coef_q14: [i16; LTP_ORDER],
    pub prev_lpc_q12: [i16; MAX_LPC_ORDER],
    pub last_frame_lost: bool,
    pub rand_seed: i32,
    pub rand_scale_q14: i16,
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prev_ltp_scale_q14: i16,
    pub prev_gain_q16: [i32; 2],
    pub fs_khz: i32,
    pub nb_subfr: i32,
    pub subfr_length: i32,
}

impl Default for PlcState {
    fn default() -> Self {
        Self {
            pitch_l_q8: 0,
            ltp_coef_q14: [0; LTP_ORDER],
            prev_lpc_q12: [0; MAX_LPC_ORDER],
            last_frame_lost: false,
            rand_seed: 0,
            rand_scale_q14: 0,
            conc_energy: 0,
            conc_energy_shift: 0,
            prev_ltp_scale_q14: 0,
            prev_gain_q16: [0; 2],
            fs_khz: 0,
            nb_subfr: 0,
            subfr_length: 0,
        }
    }
}

/// CNG state
#[derive(Clone)]
pub struct CngState {
    pub cng_exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    pub cng_smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub cng_synth_state: [i32; MAX_LPC_ORDER],
    pub cng_smth_gain_q16: i32,
    pub rand_seed: i32,
    pub fs_khz: i32,
}

impl Default for CngState {
    fn default() -> Self {
        Self {
            cng_exc_buf_q14: [0; MAX_FRAME_LENGTH],
            cng_smth_nlsf_q15: [0; MAX_LPC_ORDER],
            cng_synth_state: [0; MAX_LPC_ORDER],
            cng_smth_gain_q16: 0,
            rand_seed: 0,
            fs_khz: 0,
        }
    }
}

/// Stereo decoder state
#[derive(Clone, Default)]
pub struct StereoDecState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
}

/// Per-channel decoder state
#[derive(Clone)]
pub struct ChannelState {
    pub prev_gain_q16: i32,
    pub exc_q14: Vec<i32>,
    pub s_lpc_q14_buf: [i32; MAX_LPC_ORDER],
    pub out_buf: Vec<i16>,
    pub lag_prev: i32,
    pub last_gain_index: i8,
    pub fs_khz: i32,
    pub fs_api_hz: i32,
    pub nb_subfr: i32,
    pub frame_length: i32,
    pub subfr_length: i32,
    pub ltp_mem_length: i32,
    pub lpc_order: i32,
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub first_frame_after_reset: bool,

    // NLSF codebook selection (stored as enum/index)
    pub nlsf_cb: NlsfCbSel,

    // Pitch contour iCDF selection
    pub pitch_contour_icdf: PitchContourSel,
    // Pitch lag low bits iCDF selection
    pub pitch_lag_low_bits_icdf: PitchLagLowBitsSel,

    pub n_frames_decoded: i32,
    pub n_frames_per_packet: i32,

    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,

    pub vad_flags: [i32; MAX_FRAMES_PER_PACKET],
    pub lbrr_flag: i32,
    pub lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],

    pub resampler_state: resampler::ResamplerState,

    pub indices: SideInfoIndices,
    pub s_cng: CngState,
    pub loss_cnt: i32,
    pub prev_signal_type: i32,
    pub s_plc: PlcState,
}

/// Enum for selecting the NLSF codebook (NB/MB vs WB)
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NlsfCbSel {
    NbMb,
    Wb,
}

impl Default for NlsfCbSel {
    fn default() -> Self { NlsfCbSel::NbMb }
}

/// Enum for selecting pitch contour iCDF table
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PitchContourSel {
    Nb,         // silk_pitch_contour_NB_iCDF
    Wb,         // silk_pitch_contour_iCDF
    Nb10ms,     // silk_pitch_contour_10_ms_NB_iCDF
    Wb10ms,     // silk_pitch_contour_10_ms_iCDF
}

impl Default for PitchContourSel {
    fn default() -> Self { PitchContourSel::Nb }
}

/// Enum for selecting pitch lag low bits iCDF table
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PitchLagLowBitsSel {
    Uniform4,
    Uniform6,
    Uniform8,
}

impl Default for PitchLagLowBitsSel {
    fn default() -> Self { PitchLagLowBitsSel::Uniform4 }
}

impl ChannelState {
    pub fn new() -> Self {
        Self {
            prev_gain_q16: 65536,
            exc_q14: vec![0i32; MAX_FRAME_LENGTH],
            s_lpc_q14_buf: [0; MAX_LPC_ORDER],
            out_buf: vec![0i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
            lag_prev: 100,
            last_gain_index: 10,
            fs_khz: 0,
            fs_api_hz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: 0,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            first_frame_after_reset: true,
            nlsf_cb: NlsfCbSel::NbMb,
            pitch_contour_icdf: PitchContourSel::Nb,
            pitch_lag_low_bits_icdf: PitchLagLowBitsSel::Uniform4,
            n_frames_decoded: 0,
            n_frames_per_packet: 0,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            vad_flags: [0; MAX_FRAMES_PER_PACKET],
            lbrr_flag: 0,
            lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            resampler_state: resampler::ResamplerState::new(),
            indices: SideInfoIndices::default(),
            s_cng: CngState::default(),
            loss_cnt: 0,
            prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            s_plc: PlcState::default(),
        }
    }

    pub fn reset(&mut self) {
        self.prev_gain_q16 = 65536;
        self.exc_q14.fill(0);
        self.s_lpc_q14_buf = [0; MAX_LPC_ORDER];
        self.out_buf.fill(0);
        self.lag_prev = 100;
        self.last_gain_index = 10;
        self.prev_nlsf_q15 = [0; MAX_LPC_ORDER];
        self.first_frame_after_reset = true;
        self.ec_prev_signal_type = 0;
        self.ec_prev_lag_index = 0;
        self.vad_flags = [0; MAX_FRAMES_PER_PACKET];
        self.lbrr_flag = 0;
        self.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
        self.indices = SideInfoIndices::default();
        self.loss_cnt = 0;
        self.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;

        // Reset CNG
        cng::cng_reset(self);
        // Reset PLC
        plc::plc_reset(self);
    }

    /// Set decoder sampling rate
    pub fn set_fs(&mut self, fs_khz: i32, fs_api_hz: i32) {
        assert!(fs_khz == 8 || fs_khz == 12 || fs_khz == 16);

        let subfr_length = SUB_FRAME_LENGTH_MS as i32 * fs_khz;
        let frame_length = self.nb_subfr * subfr_length;

        // Initialize resampler when switching
        if self.fs_khz != fs_khz || self.fs_api_hz != fs_api_hz {
            self.resampler_state = resampler::ResamplerState::new();
            resampler::resampler_init(&mut self.resampler_state, fs_khz * 1000, fs_api_hz, false);
            self.fs_api_hz = fs_api_hz;
        }

        if self.fs_khz != fs_khz || frame_length != self.frame_length {
            // Set pitch contour iCDF
            if fs_khz == 8 {
                self.pitch_contour_icdf = if self.nb_subfr == MAX_NB_SUBFR as i32 {
                    PitchContourSel::Nb
                } else {
                    PitchContourSel::Nb10ms
                };
            } else {
                self.pitch_contour_icdf = if self.nb_subfr == MAX_NB_SUBFR as i32 {
                    PitchContourSel::Wb
                } else {
                    PitchContourSel::Wb10ms
                };
            }

            if self.fs_khz != fs_khz {
                self.ltp_mem_length = LTP_MEM_LENGTH_MS as i32 * fs_khz;
                if fs_khz == 8 || fs_khz == 12 {
                    self.lpc_order = MIN_LPC_ORDER as i32;
                    self.nlsf_cb = NlsfCbSel::NbMb;
                } else {
                    self.lpc_order = MAX_LPC_ORDER as i32;
                    self.nlsf_cb = NlsfCbSel::Wb;
                }
                self.pitch_lag_low_bits_icdf = match fs_khz {
                    16 => PitchLagLowBitsSel::Uniform8,
                    12 => PitchLagLowBitsSel::Uniform6,
                    _ => PitchLagLowBitsSel::Uniform4,
                };
                self.first_frame_after_reset = true;
                self.lag_prev = 100;
                self.last_gain_index = 10;
                self.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
                self.out_buf.fill(0);
                self.s_lpc_q14_buf = [0; MAX_LPC_ORDER];
            }

            self.fs_khz = fs_khz;
            self.frame_length = frame_length;
        }

        self.subfr_length = subfr_length;
    }

    /// Get the pitch contour iCDF table for this channel
    pub fn get_pitch_contour_icdf(&self) -> &'static [u8] {
        match self.pitch_contour_icdf {
            PitchContourSel::Nb => &tables::SILK_PITCH_CONTOUR_NB_ICDF,
            PitchContourSel::Wb => &tables::SILK_PITCH_CONTOUR_ICDF,
            PitchContourSel::Nb10ms => &tables::SILK_PITCH_CONTOUR_10_MS_NB_ICDF,
            PitchContourSel::Wb10ms => &tables::SILK_PITCH_CONTOUR_10_MS_ICDF,
        }
    }

    /// Get the pitch lag low bits iCDF table for this channel
    pub fn get_pitch_lag_low_bits_icdf(&self) -> &'static [u8] {
        match self.pitch_lag_low_bits_icdf {
            PitchLagLowBitsSel::Uniform4 => &tables::SILK_UNIFORM4_ICDF,
            PitchLagLowBitsSel::Uniform6 => &tables::SILK_UNIFORM6_ICDF,
            PitchLagLowBitsSel::Uniform8 => &tables::SILK_UNIFORM8_ICDF,
        }
    }
}

/// NLSF codebook struct (references static table data)
pub struct NlsfCbStruct {
    pub n_vectors: i16,
    pub order: i16,
    pub quant_step_size_q16: i16,
    pub inv_quant_step_size_q6: i16,
    pub cb1_nlsf_q8: &'static [u8],
    pub cb1_wght_q9: &'static [i16],
    pub cb1_icdf: &'static [u8],
    pub pred_q8: &'static [u8],
    pub ec_sel: &'static [u8],
    pub ec_icdf: &'static [u8],
    pub ec_rates_q5: &'static [u8],
    pub delta_min_q15: &'static [i16],
}

/// Get the NLSF codebook struct for a given selection
pub fn get_nlsf_cb(sel: NlsfCbSel) -> &'static NlsfCbStruct {
    match sel {
        NlsfCbSel::NbMb => &tables::SILK_NLSF_CB_NB_MB,
        NlsfCbSel::Wb => &tables::SILK_NLSF_CB_WB,
    }
}
