# SILK Encoder Redesign: Faithful C Float Path Port

## Motivation

The current Rust SILK encoder was built incrementally with a fixed-point-first approach,
then patched with float components piecemeal. This produced a mixed-arithmetic encoder
that doesn't match any single C reference path. The result: individual functions (Burg,
A2NLSF, NLSF encode) are byte-identical to the C reference, but the full pipeline
produces 345-byte packets vs the C reference's 51 bytes.

This document describes a clean rebuild that faithfully mirrors the C float encoder
(`silk/float/`) component by component, with FFI cross-validation at each stage.

## Architecture: C Float Encoder Component Map

The C float SILK encoder has a strict pipeline executed per frame:

```
silk_Encode (enc_API.c)
  └─ silk_encode_frame_FLP (encode_frame_FLP.c)
       ├─ 1. silk_LP_variable_cutoff       -- bandwidth smoothing (fixed-point biquad)
       ├─ 2. silk_short2float_array        -- int16 → float conversion
       ├─ 3. silk_find_pitch_lags_FLP      -- pitch analysis → pitch lags, signal type
       ├─ 4. silk_noise_shape_analysis_FLP -- spectral shaping → gains, AR, tilt, lambda
       ├─ 5. silk_find_pred_coefs_FLP      -- LPC/LTP analysis → NLSFs, residual energy
       ├─ 6. silk_process_gains_FLP        -- gain flooring + quantization → gain indices
       ├─ 7. silk_LBRR_encode_FLP          -- redundancy for packet loss
       └─ 8. [Bitrate control loop]:
            ├─ silk_NSQ_wrapper_FLP        -- float→fixed conversion + NSQ
            ├─ silk_encode_indices          -- entropy code side info
            └─ silk_encode_pulses           -- entropy code excitation
```

## Component Inventory

### Layer 0: Leaf DSP Functions (float, stateless, ~10-50 lines each)

| # | Function | C File | Description | Test Strategy |
|---|----------|--------|-------------|---------------|
| 0a | `silk_energy_FLP` | `energy_FLP.c` | Sum of squares | FFI: random vectors |
| 0b | `silk_inner_product_FLP` | `inner_product_FLP.c` | Dot product | FFI: random vectors |
| 0c | `silk_autocorrelation_FLP` | `autocorrelation_FLP.c` | Autocorrelation | FFI: sine + noise |
| 0d | `silk_schur_FLP` | `schur_FLP.c` | Schur recursion → reflection coeffs | FFI: from autocorr |
| 0e | `silk_k2a_FLP` | `k2a_FLP.c` | Reflection → LPC | FFI: from schur |
| 0f | `silk_bwexpander_FLP` | `bwexpander_FLP.c` | Bandwidth expansion | FFI: from k2a |
| 0g | `silk_apply_sine_window_FLP` | `apply_sine_window_FLP.c` | Sine window | FFI: known signal |
| 0h | `silk_LPC_analysis_filter_FLP` | `LPC_analysis_filter_FLP.c` | FIR analysis filter | FFI: from LPC + signal |
| 0i | `silk_scale_copy_vector_FLP` | `scale_copy_vector_FLP.c` | Scale and copy | FFI: trivial |
| 0j | `silk_warped_autocorrelation_FLP` | `warped_autocorrelation_FLP.c` | Warped autocorrelation | FFI: from signal |
| 0k | `silk_corrMatrix_FLP` | `corrMatrix_FLP.c` | Correlation matrix for LTP | FFI: from signal |
| 0l | `silk_LPC_inv_pred_gain_FLP` | `LPC_inv_pred_gain_FLP.c` | LPC stability check | FFI: from LPC |

### Layer 1: Fixed-Point Core (already verified byte-identical)

| # | Function | C File | Status |
|---|----------|--------|--------|
| 1a | `silk_A2NLSF` | `A2NLSF.c` | **Verified identical** via FFI test |
| 1b | `silk_NLSF2A` | `NLSF2A.c` | **Verified identical** (implicit from NLSF encode test) |
| 1c | `silk_NLSF_encode` | `NLSF_encode.c` | **Verified identical** via FFI test |
| 1d | `silk_NLSF_VQ_weights_laroia` | `NLSF_VQ_weights_laroia.c` | **Verified identical** via FFI test |
| 1e | `silk_gains_quant` / `dequant` | `gain_quant.c` | Working, needs FFI verification |
| 1f | `silk_encode_indices` | `encode_indices.c` | Working |
| 1g | `silk_encode_pulses` | `encode_pulses.c` | Working |
| 1h | `silk_VAD_GetSA_Q8` | `VAD.c` | Working |
| 1i | `silk_NSQ_c` | `NSQ.c` | Working |
| 1j | `silk_NSQ_del_dec_c` | `NSQ_del_dec.c` | Working |
| 1k | `silk_interpolate` | `interpolate.c` | Working |

### Layer 2: Float Wrappers (thin conversion layers)

| # | Function | C File | Converts |
|---|----------|--------|----------|
| 2a | `silk_A2NLSF_FLP` | `wrappers_FLP.c` | float LPC → Q16 → `silk_A2NLSF` |
| 2b | `silk_NLSF2A_FLP` | `wrappers_FLP.c` | Q15 NLSF → `silk_NLSF2A` → float LPC |
| 2c | `silk_process_NLSFs_FLP` | `wrappers_FLP.c` | NLSF encode + interpolation |
| 2d | `silk_NSQ_wrapper_FLP` | `wrappers_FLP.c` | All float params → Qxx → NSQ |
| 2e | `silk_quant_LTP_gains_FLP` | `wrappers_FLP.c` | float LTP → Q14 → `silk_quant_LTP_gains` |

### Layer 3: Float Analysis Components

| # | Function | C File | Key Outputs |
|---|----------|--------|-------------|
| 3a | `silk_burg_modified_FLP` | `burg_modified_FLP.c` | LPC coefficients + residual energy |
| 3b | `silk_find_pitch_lags_FLP` | `find_pitch_lags_FLP.c` | Pitch lags, signal type, LTP correlation |
| 3c | `silk_noise_shape_analysis_FLP` | `noise_shape_analysis_FLP.c` | Gains, AR shaping, tilt, lambda |
| 3d | `silk_find_LPC_FLP` | `find_LPC_FLP.c` | NLSFs with interpolation search |
| 3e | `silk_find_pred_coefs_FLP` | `find_pred_coefs_FLP.c` | LPC/LTP, NLSF encode, residual energy |
| 3f | `silk_residual_energy_FLP` | `residual_energy_FLP.c` | Per-subframe residual with gain scaling |
| 3g | `silk_process_gains_FLP` | `process_gains_FLP.c` | Gain flooring, quantization, lambda |

### Layer 4: Frame Encoder

| # | Function | C File |
|---|----------|--------|
| 4a | `silk_encode_frame_FLP` | `encode_frame_FLP.c` |
| 4b | `silk_LBRR_encode_FLP` | `encode_frame_FLP.c` (tail) |

### Layer 5: API

| # | Function | C File |
|---|----------|--------|
| 5a | `silk_Encode` | `enc_API.c` |

## Porting Strategy

### Phase 1: Leaf DSP (Layer 0) — ~500 lines total

Port all 12 leaf functions as standalone Rust functions in a new `silk_flp.rs` module.
Each function:
- Takes `&[f32]` inputs, returns `f32` or fills `&mut [f32]` output
- Has no state, no side effects
- Gets an FFI test comparing output against the C function for 3+ test vectors

**FFI test pattern:**
```rust
#[test]
fn energy_flp_matches_c() {
    let signal = [1.0f32, 2.0, 3.0, -1.0, 0.5];
    let rust_result = silk_energy_flp(&signal);
    let c_result = c_silk_energy_flp(&signal); // via FFI
    assert!((rust_result - c_result).abs() < 1e-6);
}
```

### Phase 2: Float Wrappers (Layer 2) — ~200 lines total

Port the 5 wrapper functions. Each converts between float and fixed-point
and calls an existing Layer 1 function. Test by comparing the Q-format
outputs against the C wrapper's Q-format outputs.

### Phase 3: Float Analysis (Layer 3) — ~1500 lines total

Port the 7 analysis components. This is the bulk of the work. Each function
calls Layer 0 DSP functions and Layer 2 wrappers. Test by:
1. Creating a synthetic test signal (440Hz sine at 16kHz)
2. Running the C function via FFI on the signal
3. Running the Rust function on the same signal
4. Comparing all outputs (gains, LPC, NLSFs, etc.)

### Phase 4: Frame Encoder (Layer 4) — ~300 lines

Assemble the frame encoder that calls Layers 0-3 in the correct order.
The frame encoder IS the pipeline — getting the call order and data flow
right is the critical test.

### Phase 5: API (Layer 5) — ~200 lines

Port the outer encode loop with resampler, VAD, and multi-frame assembly.

## State Structures

### `SilkEncoderState` (maps to `silk_encoder_state`)
```rust
pub struct SilkEncoderState {
    // Configuration
    pub fs_khz: i32,
    pub nb_subfr: i32,
    pub frame_length: i32,
    pub subfr_length: i32,
    pub ltp_mem_length: i32,
    pub la_pitch: i32,
    pub la_shape: i32,
    pub predict_lpc_order: i32,
    pub shaping_lpc_order: i32,

    // Submodule states
    pub nsq: NsqState,
    pub vad: VadState,
    pub lp: LpState,
    pub resampler: ResamplerState,

    // Previous-frame memory
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub prev_signal_type: i32,
    pub prev_lag: i32,
    pub first_frame_after_reset: bool,

    // Indices for entropy coding
    pub indices: SideInfoIndices,
    pub pulses: [i8; MAX_FRAME_LENGTH],

    // ...etc
}
```

### `SilkEncoderStateFLP` (maps to `silk_encoder_state_FLP`)
```rust
pub struct SilkEncoderStateFLP {
    pub cmn: SilkEncoderState,         // fixed-point base
    pub shape: ShapeStateFLP,           // float shaping state
    pub x_buf: Vec<f32>,               // float analysis buffer
    pub ltp_corr: f32,                 // pitch correlation
}
```

### `SilkEncoderControlFLP` (maps to `silk_encoder_control_FLP`)
```rust
pub struct SilkEncoderControlFLP {
    pub gains: [f32; MAX_NB_SUBFR],
    pub pred_coef: [[f32; MAX_LPC_ORDER]; 2],
    pub ltp_coef: [f32; MAX_NB_SUBFR * LTP_ORDER],
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub ar: [f32; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    pub lf_ma_shp: [f32; MAX_NB_SUBFR],
    pub lf_ar_shp: [f32; MAX_NB_SUBFR],
    pub tilt: [f32; MAX_NB_SUBFR],
    pub harm_shape_gain: [f32; MAX_NB_SUBFR],
    pub lambda: f32,
    pub input_quality: f32,
    pub coding_quality: f32,
    pub pred_gain: f32,
    pub ltp_pred_cod_gain: f32,
    pub res_nrg: [f32; MAX_NB_SUBFR],
    pub gains_unq_q16: [i32; MAX_NB_SUBFR],
    pub last_gain_index_prev: i8,
}
```

## FFI Infrastructure

The `opus-ffi` crate already has bindings for `silk_A2NLSF`, `silk_NLSF_encode`,
and `silk_NLSF_VQ_weights_laroia`. For the float functions, add:

```rust
unsafe extern "C" {
    fn silk_energy_FLP(data: *const f32, data_length: i32) -> f64;
    fn silk_schur_FLP(refl_coef: *mut f32, auto_corr: *const f32, order: i32) -> f32;
    fn silk_burg_modified_FLP(
        A: *mut f32, x: *const f32, min_inv_gain: f32,
        subfr_length: i32, nb_subfr: i32, D: i32, arch: i32
    ) -> f32;
    // ... etc for each Layer 0 function
}
```

Build `opus-ffi` with `OPUS_FIXED_POINT=OFF` (the default) so these float
symbols are available.

## File Organization

```
crates/opus-silk/src/
  encoder_flp/
    mod.rs              -- SilkEncoderStateFLP, encode_frame_flp()
    dsp.rs              -- Layer 0 leaf DSP functions
    wrappers.rs         -- Layer 2 float↔fixed wrappers
    find_pitch_lags.rs  -- Layer 3a: pitch analysis
    noise_shape.rs      -- Layer 3b: noise shaping
    find_pred_coefs.rs  -- Layer 3c: LPC/LTP prediction
    find_lpc.rs         -- Layer 3d: Burg + NLSF interpolation
    process_gains.rs    -- Layer 3e: gain processing
    residual_energy.rs  -- Layer 3f: per-subframe residual
    lbrr.rs             -- Layer 4b: LBRR encoding
  encoder.rs            -- Layer 5: silk_Encode API (reuses encoder_flp/)
  nsq.rs                -- Layer 2a: NSQ (fixed-point, unchanged)
  nsq_del_dec.rs        -- Layer 2b: NSQ delayed decision (unchanged)
  nlsf.rs               -- Layer 1: NLSF functions (unchanged)
  nlsf_encode.rs        -- Layer 1: NLSF VQ encoding (unchanged)
  gain_quant.rs         -- Layer 1: gain quantization (unchanged)
  encode_indices.rs     -- Layer 1: index encoding (unchanged)
  encode_pulses.rs      -- Layer 1: pulse encoding (unchanged)
  vad.rs                -- Layer 1: VAD (unchanged)
  lpc_analysis.rs       -- Burg + A2NLSF (keep existing, add float)
  tables.rs             -- Codebook tables (unchanged)
```

## Test Hierarchy

```
tests/
  flp_dsp_tests.rs      -- FFI tests for all Layer 0 functions
  flp_wrapper_tests.rs   -- FFI tests for Layer 2 wrappers
  flp_analysis_tests.rs  -- FFI tests for Layer 3 (Burg, noise_shape, etc.)
  flp_frame_tests.rs     -- Integration test: full frame encode comparison
  a2nlsf_comparison.rs   -- Existing FFI cross-validation (keep)
```

## Success Criteria

1. **Every Layer 0 function**: Rust output matches C output within 1e-6 for all test vectors
2. **Every Layer 1 function**: Byte-identical output (already verified for key functions)
3. **Layer 3 analysis**: Per-subframe gains, LPC coefficients, NLSFs match C within 1 LSB
4. **Full frame**: Gain indices match C reference (gain_index=28 for 440Hz sine at 16kbps WB)
5. **Packet size**: Within 10% of C reference (target: <56 bytes for the test signal)
