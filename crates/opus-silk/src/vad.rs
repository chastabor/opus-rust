// Voice Activity Detection (VAD) for the SILK encoder
// Ported from silk/VAD.c in the C reference implementation
//
// Copyright (c) 2006-2011, Skype Limited. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// - Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
// - Neither the name of Internet Society, IETF or IETF Trust, nor the
// names of specific contributors, may be used to endorse or promote
// products derived from this software without specific prior written
// permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

use crate::signal_processing::*;
use crate::*;

// ---- VAD Constants (from silk/define.h) ----

/// Number of frequency bands for VAD
pub const VAD_N_BANDS: usize = 4;

/// log2 of number of internal subframes
pub const VAD_INTERNAL_SUBFRAMES_LOG2: i32 = 2;

/// Number of internal subframes (1 << VAD_INTERNAL_SUBFRAMES_LOG2)
pub const VAD_INTERNAL_SUBFRAMES: i32 = 1 << VAD_INTERNAL_SUBFRAMES_LOG2;

/// Noise level smoothing coefficient (Q16), must be < 4096
pub const VAD_NOISE_LEVEL_SMOOTH_COEF_Q16: i32 = 1024;

/// Initial noise level bias
pub const VAD_NOISE_LEVELS_BIAS: i32 = 50;

/// Negative offset for sigmoid (sigmoid is 0 at -128 in Q5)
pub const VAD_NEGATIVE_OFFSET_Q5: i32 = 128;

/// SNR factor for speech activity estimation (Q16)
pub const VAD_SNR_FACTOR_Q16: i32 = 45000;

/// Smoothing coefficient for SNR measurement (Q18)
pub const VAD_SNR_SMOOTH_COEF_Q18: i32 = 4096;

/// Weighting factors for tilt measure, one per band
static TILT_WEIGHTS: [i32; VAD_N_BANDS] = [30000, 6000, -12000, -12000];

// ---- VAD State ----

/// VAD state (matches silk_VAD_state from C reference)
///
/// Fields correspond to the C struct fields:
/// - `ana_state`/`ana_state1`/`ana_state2`: Three stages of analysis filterbank state
/// - `xnrg_subfr`: Subframe energies carried over between frames
/// - `nrg_ratio_smth_q8`: Smoothed energy-to-noise ratio per band (Q8)
/// - `hp_state`: Differentiator state for lowest band HP filter
/// - `nl`/`inv_nl`: Noise level estimates and their inverses
/// - `noise_level_bias`: Per-band noise level bias (approx pink noise)
/// - `counter`: Frame counter for initial faster adaptation
#[derive(Clone)]
pub struct VadState {
    /// Analysis filterbank state: 0-8 kHz split
    pub ana_state: [i32; 2],
    /// Analysis filterbank state: 0-4 kHz split
    pub ana_state1: [i32; 2],
    /// Analysis filterbank state: 0-2 kHz split
    pub ana_state2: [i32; 2],
    /// Subframe energies per band
    pub xnrg_subfr: [i32; VAD_N_BANDS],
    /// Smoothed energy-to-noise ratio per band (Q8)
    pub nrg_ratio_smth_q8: [i32; VAD_N_BANDS],
    /// State of differentiator (HP filter) in the lowest band
    pub hp_state: i16,
    /// Noise energy level in each band
    pub nl: [i32; VAD_N_BANDS],
    /// Inverse noise energy level in each band
    pub inv_nl: [i32; VAD_N_BANDS],
    /// Noise level estimator bias/offset per band
    pub noise_level_bias: [i32; VAD_N_BANDS],
    /// Frame counter used in the initial phase
    pub counter: i32,
}

impl Default for VadState {
    fn default() -> Self {
        let mut state = Self {
            ana_state: [0; 2],
            ana_state1: [0; 2],
            ana_state2: [0; 2],
            xnrg_subfr: [0; VAD_N_BANDS],
            nrg_ratio_smth_q8: [0; VAD_N_BANDS],
            hp_state: 0,
            nl: [0; VAD_N_BANDS],
            inv_nl: [0; VAD_N_BANDS],
            noise_level_bias: [0; VAD_N_BANDS],
            counter: 0,
        };
        silk_vad_init(&mut state);
        state
    }
}

// ---- Helper: saturating add of two positive values ----

/// Add two positive i32 values with saturation to i32::MAX.
/// Matches silk_ADD_POS_SAT32: if the unsigned sum overflows (bit 31 set), return i32::MAX.
#[inline(always)]
fn add_pos_sat32(a: i32, b: i32) -> i32 {
    let sum = (a as u32).wrapping_add(b as u32);
    if sum & 0x8000_0000 != 0 {
        i32::MAX
    } else {
        sum as i32
    }
}

// ---- VAD Init ----

/// Initialize/reset VAD state.
///
/// Sets noise_level_bias to initial noise estimate (approx pink noise: psd proportional
/// to inverse of frequency), zeros filter states, sets initial counter and smoothed
/// energy-to-noise ratios.
///
/// Matches silk_VAD_Init from the C reference.
pub fn silk_vad_init(state: &mut VadState) {
    // Reset all state memory to zero
    state.ana_state = [0; 2];
    state.ana_state1 = [0; 2];
    state.ana_state2 = [0; 2];
    state.xnrg_subfr = [0; VAD_N_BANDS];
    state.nrg_ratio_smth_q8 = [0; VAD_N_BANDS];
    state.hp_state = 0;
    state.nl = [0; VAD_N_BANDS];
    state.inv_nl = [0; VAD_N_BANDS];
    state.noise_level_bias = [0; VAD_N_BANDS];
    state.counter = 0;

    // Initialize noise level bias with approx pink noise levels
    // (psd proportional to inverse of frequency)
    for b in 0..VAD_N_BANDS {
        state.noise_level_bias[b] = (VAD_NOISE_LEVELS_BIAS / (b as i32 + 1)).max(1);
    }

    // Initialize noise levels: NL = 100 * bias, inv_NL = MAX / NL
    for b in 0..VAD_N_BANDS {
        state.nl[b] = 100 * state.noise_level_bias[b];
        state.inv_nl[b] = silk_div32(i32::MAX, state.nl[b]);
    }

    // Start counter at 15
    state.counter = 15;

    // Init smoothed energy-to-noise ratio: 100 * 256 = 25600 --> ~20 dB SNR
    for b in 0..VAD_N_BANDS {
        state.nrg_ratio_smth_q8[b] = 100 * 256;
    }
}

// ---- Noise Level Estimation ----

/// Update noise level estimates based on subband energies.
///
/// Uses an adaptive smoothing approach: faster initial adaptation (first ~20 sec),
/// then slower tracking. Updates are reduced when subband energy is high relative
/// to the noise estimate (i.e., speech is present).
///
/// Matches silk_VAD_GetNoiseLevels from the C reference.
fn silk_vad_get_noise_levels(xnrg: &[i32; VAD_N_BANDS], state: &mut VadState) {
    // Initially faster smoothing
    let min_coef: i32 = if state.counter < 1000 {
        // 1000 frames = ~20 sec
        let mc = i16::MAX as i32 / ((state.counter >> 4) + 1);
        // Increment frame counter
        state.counter += 1;
        mc
    } else {
        0
    };

    for k in 0..VAD_N_BANDS {
        // Get old noise level estimate for current band
        let nl = state.nl[k];
        debug_assert!(nl >= 0);

        // Add bias
        let nrg = add_pos_sat32(xnrg[k], state.noise_level_bias[k]);
        debug_assert!(nrg > 0);

        // Invert energies
        let inv_nrg = silk_div32(i32::MAX, nrg);
        debug_assert!(inv_nrg >= 0);

        // Less update when subband energy is high
        let coef = if nrg > nl << 3 {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3
        } else if nrg < nl {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16
        } else {
            silk_smulwb(
                silk_smulww_correct(inv_nrg, nl),
                VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 << 1,
            )
        };

        // Initially faster smoothing
        let coef = coef.max(min_coef);

        // Smooth inverse energies
        state.inv_nl[k] = silk_smlawb(state.inv_nl[k], inv_nrg - state.inv_nl[k], coef);
        debug_assert!(state.inv_nl[k] >= 0);

        // Compute noise level by inverting again
        let mut nl = silk_div32(i32::MAX, state.inv_nl[k]);
        debug_assert!(nl >= 0);

        // Limit noise levels (guarantee 7 bits of head room)
        nl = nl.min(0x00FF_FFFF);

        // Store as part of state
        state.nl[k] = nl;
    }
}

// ---- Filter Bank Decimation Helper ----

/// Perform 3-stage cascaded analysis filter bank decimation, splitting the input
/// signal into 4 frequency bands.
///
/// Uses `silk_ana_filt_bank_1` which writes low-pass and high-pass outputs into
/// a single buffer at specified offsets.
///
/// Returns the working buffer and the per-band offsets into it.
fn decimate_into_bands(
    input: &[i16],
    state: &mut VadState,
    frame_length: i32,
    x_out: &mut [i16],
) -> [usize; VAD_N_BANDS] {
    let decimated_framelength1 = (frame_length >> 1) as usize; // frame_length / 2
    let decimated_framelength2 = (frame_length >> 2) as usize; // frame_length / 4
    let decimated_framelength = (frame_length >> 3) as usize; // frame_length / 8

    // Band offsets in the working buffer (matches C layout):
    //   Band 0 (0-1 kHz):  offset 0, length = decimated_framelength
    //   Band 1 (1-2 kHz):  offset = decimated_framelength + decimated_framelength2
    //   Band 2 (2-4 kHz):  offset = band1_offset + decimated_framelength
    //   Band 3 (4-8 kHz):  offset = band2_offset + decimated_framelength2
    let x_offset: [usize; VAD_N_BANDS] = [
        0,
        decimated_framelength + decimated_framelength2,
        decimated_framelength + decimated_framelength2 + decimated_framelength,
        decimated_framelength
            + decimated_framelength2
            + decimated_framelength
            + decimated_framelength2,
    ];
    let x_len = x_offset[3] + decimated_framelength1;
    let mut x = &mut x_out[..x_len];

    // Stage 1: 0-8 kHz -> 0-4 kHz (low, outL) and 4-8 kHz (high, outH)
    // C: silk_ana_filt_bank_1(pIn, &AnaState[0], X, &X[X_offset[3]], frame_length)
    //   outL = X[0..], outH = X[X_offset[3]..]
    silk_ana_filt_bank_1(
        input,
        &mut state.ana_state,
        &mut x,
        0,           // lp_offset: low band (0-4 kHz) at start of buffer
        x_offset[3], // hp_offset: high band (4-8 kHz) at X_offset[3]
        frame_length,
    );

    // Stage 2: 0-4 kHz -> 0-2 kHz (low) and 2-4 kHz (high)
    // C: silk_ana_filt_bank_1(X, &AnaState1[0], X, &X[X_offset[2]], decimated_framelength1)
    //   Input and outL both start at X[0], so we need a copy of the input.
    let mut input_stage2_buf = [0i16; 200]; // max decimated_framelength1 = 160
    input_stage2_buf[..decimated_framelength1].copy_from_slice(&x[..decimated_framelength1]);
    let input_stage2 = &input_stage2_buf[..decimated_framelength1];
    silk_ana_filt_bank_1(
        &input_stage2,
        &mut state.ana_state1,
        &mut x,
        0,           // lp_offset: low band (0-2 kHz) at start
        x_offset[2], // hp_offset: high band (2-4 kHz)
        decimated_framelength1 as i32,
    );

    // Stage 3: 0-2 kHz -> 0-1 kHz (low) and 1-2 kHz (high)
    // C: silk_ana_filt_bank_1(X, &AnaState2[0], X, &X[X_offset[1]], decimated_framelength2)
    let mut input_stage3_buf = [0i16; 100]; // max decimated_framelength2 = 80
    input_stage3_buf[..decimated_framelength2].copy_from_slice(&x[..decimated_framelength2]);
    let input_stage3 = &input_stage3_buf[..decimated_framelength2];
    silk_ana_filt_bank_1(
        &input_stage3,
        &mut state.ana_state2,
        &mut x,
        0,           // lp_offset: low band (0-1 kHz)
        x_offset[1], // hp_offset: high band (1-2 kHz)
        decimated_framelength2 as i32,
    );

    x_offset
}

// ---- Core VAD computation (shared by both entry points) ----

/// Internal implementation of the VAD speech activity computation.
///
/// `is_20ms_frame` and `is_10ms_frame` flags control frame-length-dependent
/// adjustments that depend on knowledge of `fs_kHz`.
fn silk_vad_compute(
    state: &mut VadState,
    sa_q8: &mut i32,
    snr_db_q7: &mut i32,
    quality_bands_q15: &mut [i32; VAD_N_BANDS],
    tilt_q15: &mut i32,
    input: &[i16],
    frame_length: i32,
    is_20ms_frame: bool,
    is_10ms_frame: bool,
) -> i32 {
    debug_assert!(frame_length <= 512);
    debug_assert!(frame_length == 8 * (frame_length >> 3));

    let decimated_framelength = (frame_length >> 3) as usize;

    // ---- Filter and Decimate into 4 bands ----
    // Stack buffer for decimated bands (max ~800 i16 = 1600 bytes)
    let mut x_buf = [0i16; 800];
    let x_offset = decimate_into_bands(input, state, frame_length, &mut x_buf);
    let x = &mut x_buf[..];

    // ---- HP filter on lowest band (differentiator) ----
    // This removes DC and very low frequencies from band 0.
    // Process in reverse: y[i] = x[i]/2 - x[i-1]/2, with state for continuity.
    let df = decimated_framelength;
    x[df - 1] = (x[df - 1] as i32 >> 1) as i16;
    let hp_state_tmp = x[df - 1];
    for i in (1..df).rev() {
        x[i - 1] = (x[i - 1] as i32 >> 1) as i16;
        x[i] = (x[i] as i32 - x[i - 1] as i32) as i16;
    }
    x[0] = (x[0] as i32 - state.hp_state as i32) as i16;
    state.hp_state = hp_state_tmp;

    // ---- Calculate energy in each band ----
    let mut xnrg = [0i32; VAD_N_BANDS];

    for b in 0..VAD_N_BANDS {
        // Find the decimated framelength in the non-uniformly divided bands
        // Band 0: frame_length >> 3  (0-1 kHz, most decimated)
        // Band 1: frame_length >> 3  (1-2 kHz)
        // Band 2: frame_length >> 2  (2-4 kHz)
        // Band 3: frame_length >> 1  (4-8 kHz, least decimated)
        let shift = (VAD_N_BANDS as i32 - b as i32).min(VAD_N_BANDS as i32 - 1);
        let band_framelength = frame_length >> shift;

        // Split length into subframe lengths
        let dec_subframe_length = band_framelength >> VAD_INTERNAL_SUBFRAMES_LOG2;
        let mut dec_subframe_offset: i32 = 0;

        // Initialize with summed energy of last subframe (look-ahead from previous frame)
        xnrg[b] = state.xnrg_subfr[b];

        let mut sum_squared = 0i32;
        for s in 0..VAD_INTERNAL_SUBFRAMES {
            sum_squared = 0;
            for i in 0..dec_subframe_length {
                // Right shift by 3 to prevent overflow in accumulation.
                // Energy will be < dec_subframe_length * (i16::MIN / 8)^2.
                // Safe as long as dec_subframe_length <= 128.
                let x_tmp = x[x_offset[b] + (i + dec_subframe_offset) as usize] as i32 >> 3;
                sum_squared = silk_smlabb(sum_squared, x_tmp, x_tmp);
                debug_assert!(sum_squared >= 0);
            }

            // Add/saturate summed energy of current subframe
            if s < VAD_INTERNAL_SUBFRAMES - 1 {
                xnrg[b] = add_pos_sat32(xnrg[b], sum_squared);
            } else {
                // Look-ahead subframe: half weight
                xnrg[b] = add_pos_sat32(xnrg[b], sum_squared >> 1);
            }

            dec_subframe_offset += dec_subframe_length;
        }
        // Store last subframe energy for next frame's look-ahead
        state.xnrg_subfr[b] = sum_squared;
    }

    // ---- Noise estimation ----
    silk_vad_get_noise_levels(&xnrg, state);

    // ---- Signal-plus-noise to noise ratio estimation ----
    let mut sum_squared = 0i32;
    let mut input_tilt = 0i32;
    let mut nrg_to_noise_ratio_q8 = [0i32; VAD_N_BANDS];

    for b in 0..VAD_N_BANDS {
        let speech_nrg = xnrg[b] - state.nl[b];
        if speech_nrg > 0 {
            // Divide, with sufficient resolution
            if (xnrg[b] & 0xFF80_0000u32 as i32) == 0 {
                // xnrg fits in 23 bits, so we can shift left by 8 safely
                nrg_to_noise_ratio_q8[b] = silk_div32(xnrg[b] << 8, state.nl[b] + 1);
            } else {
                nrg_to_noise_ratio_q8[b] = silk_div32(xnrg[b], (state.nl[b] >> 8) + 1);
            }

            // Convert to log domain
            let mut snr_q7 = silk_lin2log(nrg_to_noise_ratio_q8[b]) - 8 * 128;

            // Sum-of-squares (Q14)
            sum_squared = silk_smlabb(sum_squared, snr_q7, snr_q7);

            // Tilt measure: scale down SNR for small subband speech energies
            if speech_nrg < (1 << 20) {
                snr_q7 = silk_smulwb(silk_sqrt_approx(speech_nrg) << 6, snr_q7);
            }
            input_tilt = silk_smlawb(input_tilt, TILT_WEIGHTS[b], snr_q7);
        } else {
            nrg_to_noise_ratio_q8[b] = 256;
        }
    }

    // Mean-of-squares (Q14)
    sum_squared /= VAD_N_BANDS as i32;

    // Root-mean-square approximation, scale to dBs, write to output pointer
    // Cast to i16 to match C: pSNR_dB_Q7 = (opus_int16)(3 * silk_SQRT_APPROX(...))
    let p_snr_db_q7 = (3 * silk_sqrt_approx(sum_squared)) as i16 as i32;

    // ---- Speech Probability Estimation ----
    let mut sa_q15 =
        silk_sigm_q15(silk_smulwb(VAD_SNR_FACTOR_Q16, p_snr_db_q7) - VAD_NEGATIVE_OFFSET_Q5);

    // ---- Frequency Tilt Measure ----
    *tilt_q15 = (silk_sigm_q15(input_tilt) - 16384) << 1;

    // ---- Scale the sigmoid output based on power levels ----
    let mut speech_nrg_total = 0i32;
    for b in 0..VAD_N_BANDS {
        // Accumulate signal-without-noise energies, higher frequency bands have more weight
        speech_nrg_total += (b as i32 + 1) * ((xnrg[b] - state.nl[b]) >> 4);
    }

    if is_20ms_frame {
        speech_nrg_total >>= 1;
    }

    // Power scaling
    if speech_nrg_total <= 0 {
        sa_q15 >>= 1;
    } else if speech_nrg_total < 16384 {
        let scaled_nrg = speech_nrg_total << 16;
        let sqrt_nrg = silk_sqrt_approx(scaled_nrg);
        sa_q15 = silk_smulwb(32768 + sqrt_nrg, sa_q15);
    }

    // Copy the resulting speech activity in Q8 (clamp to 0..255)
    *sa_q8 = (sa_q15 >> 7).min(255);

    // ---- Energy Level and SNR estimation ----
    // Smoothing coefficient
    let mut smooth_coef_q16 = silk_smulwb(VAD_SNR_SMOOTH_COEF_Q18, silk_smulwb(sa_q15, sa_q15));

    if is_10ms_frame {
        smooth_coef_q16 >>= 1;
    }

    for b in 0..VAD_N_BANDS {
        // Compute smoothed energy-to-noise ratio per band
        state.nrg_ratio_smth_q8[b] = silk_smlawb(
            state.nrg_ratio_smth_q8[b],
            nrg_to_noise_ratio_q8[b] - state.nrg_ratio_smth_q8[b],
            smooth_coef_q16,
        );

        // Signal to noise ratio in dB per band
        let snr_q7 = 3 * (silk_lin2log(state.nrg_ratio_smth_q8[b]) - 8 * 128);

        // quality = sigmoid( 0.25 * ( SNR_dB - 16 ) )
        quality_bands_q15[b] = silk_sigm_q15((snr_q7 - 16 * 128) >> 4);
    }

    // Output SNR
    *snr_db_q7 = p_snr_db_q7;

    0
}

// ---- Public API ----

/// Compute speech activity level (Q8), SNR estimate, per-band quality, and spectral tilt.
///
/// This is the main VAD entry point, matching silk_VAD_GetSA_Q8_c from the C reference.
/// Since `fs_kHz` is not available, the 10ms/20ms frame distinction uses a heuristic:
/// frame lengths of 160, 240, or 320 are treated as 20ms frames.
///
/// For precise behavior matching the C reference, use [`silk_vad_get_sa_q8_ex`] which
/// accepts `fs_kHz` as an additional parameter.
///
/// # Arguments
/// * `state` - VAD state (updated in-place)
/// * `sa_q8` - Output: speech activity level, 0..255 (Q8)
/// * `snr_db_q7` - Output: estimated SNR in dB (Q7)
/// * `quality_bands_q15` - Output: per-band input quality (Q15), 4 bands
/// * `tilt_q15` - Output: spectral tilt measure (Q15)
/// * `input` - Input PCM signal (16-bit)
/// * `frame_length` - Frame length in samples (must be multiple of 8, max 512)
///
/// # Returns
/// 0 on success
pub fn silk_vad_get_sa_q8(
    state: &mut VadState,
    sa_q8: &mut i32,
    snr_db_q7: &mut i32,
    quality_bands_q15: &mut [i32; VAD_N_BANDS],
    tilt_q15: &mut i32,
    input: &[i16],
    frame_length: i32,
) -> i32 {
    // Heuristic: standard 20ms frame lengths for SILK (8/12/16 kHz)
    let is_20ms_frame = matches!(frame_length, 160 | 240 | 320);

    silk_vad_compute(
        state,
        sa_q8,
        snr_db_q7,
        quality_bands_q15,
        tilt_q15,
        input,
        frame_length,
        is_20ms_frame,
        false, // Cannot determine 10ms without fs_kHz
    )
}

/// Extended version that also accepts `fs_kHz` for accurate 10ms/20ms frame detection.
///
/// This matches the C reference more closely since the C code accesses `psEncC->fs_kHz`
/// to distinguish 10ms from 20ms frames for power scaling and smoothing adjustments.
///
/// # Arguments
/// * `state` - VAD state (updated in-place)
/// * `sa_q8` - Output: speech activity level, 0..255 (Q8)
/// * `snr_db_q7` - Output: estimated SNR in dB (Q7)
/// * `quality_bands_q15` - Output: per-band input quality (Q15), 4 bands
/// * `tilt_q15` - Output: spectral tilt measure (Q15)
/// * `input` - Input PCM signal (16-bit)
/// * `frame_length` - Frame length in samples (must be multiple of 8, max 512)
/// * `fs_khz` - Internal sampling rate in kHz (8, 12, or 16)
///
/// # Returns
/// 0 on success
pub fn silk_vad_get_sa_q8_ex(
    state: &mut VadState,
    sa_q8: &mut i32,
    snr_db_q7: &mut i32,
    quality_bands_q15: &mut [i32; VAD_N_BANDS],
    tilt_q15: &mut i32,
    input: &[i16],
    frame_length: i32,
    fs_khz: i32,
) -> i32 {
    let is_20ms_frame = frame_length == 20 * fs_khz;
    let is_10ms_frame = frame_length == 10 * fs_khz;

    silk_vad_compute(
        state,
        sa_q8,
        snr_db_q7,
        quality_bands_q15,
        tilt_q15,
        input,
        frame_length,
        is_20ms_frame,
        is_10ms_frame,
    )
}
