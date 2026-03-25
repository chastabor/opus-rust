# CELT Testing Framework: C Reference Cross-Validation Plan

## Status: Phases 1-5 Complete (56 new cross-validation tests, 2 bugs fixed)

### Bugs Found and Fixed
1. **`frac_mul16` missing rounding bias** — Rust was `((a*b) >> 15)` but C is `((16384 + (i16)a * (i16)b) >> 15)`. Fixed to match C exactly. This also fixed `bitexact_cos`.
2. **`celt_lpc` leftover fixed-point scaling** — Rust had `/64.0` and `*64.0` factors from a fixed-point port that should be identity in float mode. LPC coefficients were completely wrong. Fixed by removing all scaling.

## Context

The Rust SILK implementation has comprehensive cross-validation tests comparing each function against the C reference via FFI. The CELT crate (`opus-celt`) has only basic roundtrip tests with no C reference comparison. **Zero CELT functions are currently exposed via FFI** — only SILK low-level functions and the high-level opus encoder/decoder API exist in `opus-ffi`.

The goal is to expose CELT internal functions via FFI and build 1-to-1 test cases comparing every Rust CELT function against its C counterpart, following the exact SILK testing pattern.

The existing `correctness_vs_c.rs` already has 8 CELT full-pipeline configs (e.g., `celt_silence`, `celt_sine440`, `celt_stereo_sine`) but with loose thresholds (2.0). This plan focuses on low-level function validation that will tighten those tolerances.

---

## Feature Mapping: Rust CELT vs C Reference

### Implemented in Rust (needs C cross-validation)

| Module | Rust Functions | C Equivalent | Status |
|---|---|---|---|
| **mathops.rs** | `celt_exp2`, `celt_log2`, `celt_sqrt`, `celt_rsqrt`, `celt_rcp` | mathops.h (static inline) | No FFI |
| **mathops.rs** | `bitexact_cos`, `bitexact_log2tan`, `isqrt32`, `frac_mul16` | bands.c / mathops.c | No FFI |
| **mathops.rs** | `celt_lcg_rand`, `celt_inner_prod`, `celt_maxabs`, `renormalise_vector` | bands.c / vq.c | No FFI |
| **lpc.rs** | `celt_lpc`, `celt_fir`, `celt_iir`, `celt_autocorr` | celt_lpc.c | No FFI |
| **fft.rs** | `opus_fft`, `opus_fft_impl`, `KissFftState::new` | kiss_fft.c | No FFI |
| **mdct.rs** | `clt_mdct_forward`, `clt_mdct_backward`, `MdctLookup::new` | mdct.c | No FFI |
| **pitch.rs** | `pitch_downsample`, `pitch_search`, `celt_pitch_xcorr`, `remove_doubling` | pitch.c | No FFI |
| **pitch.rs** | `comb_filter`, `comb_filter_inplace` | celt.h / pitch.h | No FFI |
| **bands.rs** | `compute_band_energies`, `normalise_bands`, `denormalise_bands` | bands.c | No FFI |
| **bands.rs** | `amp2_log2`, `anti_collapse`, `quant_all_bands` / `quant_all_bands_enc` | bands.c | No FFI |
| **quant_energy.rs** | `quant_coarse_energy`, `quant_fine_energy`, `quant_energy_finalise_enc` | quant_bands.c | No FFI |
| **quant_energy.rs** | `unquant_coarse_energy`, `unquant_fine_energy`, `unquant_energy_finalise` | quant_bands.c | No FFI |
| **rate.rs** | `bits2pulses`, `pulses2bits`, `init_caps`, `clt_compute_allocation` | rate.c | No FFI |
| **encoder.rs** | `CeltEncoder::encode_with_ec`, `celt_preemphasis` | celt_encoder.c | No FFI |
| **decoder.rs** | `CeltDecoder::decode_with_ec`, `celt_synthesis`, `deemphasis` | celt_decoder.c | No FFI |
| **tables.rs** | Static lookup tables (EBANDS, WINDOW, BAND_ALLOCATION, etc.) | modes.c / static_modes_float.h | No FFI |

### Not Implemented in Rust (C-only features)

| Feature | C Source | Notes |
|---|---|---|
| Alternative sample rates (8/12/16/24 kHz) | modes.c | Rust is 48kHz only |
| Fixed-point arithmetic paths | Throughout | Rust is float-only |
| VQ search (`alg_quant`, `alg_unquant`, `op_pvq_search`) | vq.c | In Rust `quant_all_bands` |
| CWRS pulse coding (`encode_pulses`, `decode_pulses`) | cwrs.c | In Rust `quant_all_bands` |
| SIMD optimizations (SSE/NEON/AVX) | x86/, arm/ | Generic C only for testing |
| Laplace coding (`ec_laplace_encode/decode`) | laplace.c | In Rust range coder |
| `spreading_decision`, `hysteresis_decision` | bands.c | Internal to encoder |

---

## Step 1: Build System & C Wrapper Infrastructure

- [x] **Files:**
  - `crates/opus-ffi/build.rs` — add CELT include paths and compile new wrapper
  - `crates/opus-ffi/src/celt_wrapper.c` — **new file** with C shims

### build.rs changes
Add include paths for CELT headers and compile the new wrapper file:
```rust
cc::Build::new()
    .file(manifest_dir.join("src/wrapper.c"))
    .file(manifest_dir.join("src/celt_wrapper.c"))  // NEW
    .include(&include_dir)
    .include(manifest_dir.join("opus-c/include"))
    .include(manifest_dir.join("opus-c/celt"))       // NEW: for celt/*.h
    .include(manifest_dir.join("opus-c"))             // NEW: for config.h if needed
    .compile("opus_wrapper");
```

### celt_wrapper.c
C shims needed because many CELT functions are `static inline` (not linkable) or require opaque types (`CELTMode*`, `kiss_fft_state*`, `mdct_lookup*`).

**Wrapper categories:**
1. **Static-inline math wrappers**: `wrap_celt_exp2`, `wrap_celt_log2`, `wrap_celt_inner_prod`, `wrap_celt_maxabs16`, `wrap_celt_rcp`, `wrap_frac_mul16`
2. **CELTMode-dependent wrappers**: Functions that need the 48kHz/960 mode — use a `get_celt_mode_48000()` helper that lazily creates the mode once
3. **FFT/MDCT wrappers**: Allocate internal state, run transform, return results via flat arrays
4. **Pitch wrappers**: Handle `celt_sig *x[]` array-of-pointers pattern

---

## Step 2: FFI Declarations & Safe Wrappers in lib.rs

- [x] **File:** `crates/opus-ffi/src/lib.rs` — add a `// ── CELT low-level FFI ──` section

Following the SILK pattern: raw `extern "C"` block, then `pub fn c_*()` safe wrappers.

### Group A — Direct extern (externally defined, no wrapper needed)

| C function | Safe wrapper | Source |
|---|---|---|
| `isqrt32(u32) -> u32` | `c_isqrt32` | mathops.c |
| `bitexact_cos(i16) -> i16` | `c_bitexact_cos` | bands.c |
| `bitexact_log2tan(i32, i32) -> i32` | `c_bitexact_log2tan` | bands.c |
| `celt_lcg_rand(u32) -> u32` | `c_celt_lcg_rand` | bands.c |
| `_celt_lpc(*f32, *f32, i32)` | `c_celt_lpc` | celt_lpc.c |
| `celt_fir_c(*f32, *f32, *f32, N, ord, arch)` | `c_celt_fir` | celt_lpc.c |
| `celt_iir(*f32, *f32, *f32, N, ord, *f32, arch)` | `c_celt_iir` | celt_lpc.c |
| `_celt_autocorr(*f32, *f32, *f32, overlap, lag, n, arch)` | `c_celt_autocorr` | celt_lpc.c |
| `celt_pitch_xcorr_c(*f32, *f32, *f32, len, max_pitch, arch)` | `c_celt_pitch_xcorr` | pitch.c |
| `renormalise_vector(*f32, N, f32, arch)` | `c_renormalise_vector` | vq.c |
| `exp_rotation(*f32, len, dir, stride, K, spread)` | `c_exp_rotation` | vq.c |

### Group B — Via celt_wrapper.c shims (static inline or require opaque types)

| C wrapper | Safe wrapper | Why wrapper needed |
|---|---|---|
| `wrap_celt_exp2(f32) -> f32` | `c_celt_exp2` | static inline |
| `wrap_celt_log2(f32) -> f32` | `c_celt_log2` | static inline / macro |
| `wrap_celt_inner_prod(*f32, *f32, N) -> f32` | `c_celt_inner_prod` | static inline |
| `wrap_celt_maxabs16(*f32, N) -> f32` | `c_celt_maxabs16` | static inline |
| `wrap_celt_rcp(f32) -> f32` | `c_celt_rcp` | static inline |
| `wrap_frac_mul16(i32, i32) -> i32` | `c_frac_mul16` | macro `FRAC_MUL16` |
| `wrap_opus_fft(nfft, *fin_r, *fin_i, *fout_r, *fout_i)` | `c_opus_fft` | requires kiss_fft_state* |
| `wrap_clt_mdct_forward(...)` | `c_clt_mdct_forward` | requires mdct_lookup* |
| `wrap_clt_mdct_backward(...)` | `c_clt_mdct_backward` | requires mdct_lookup* |
| `wrap_pitch_downsample_mono(*f32, *f32, len)` | `c_pitch_downsample_mono` | array-of-pointers arg |
| `wrap_pitch_search(*f32, *f32, len, max, *pitch)` | `c_pitch_search` | arch param |
| `wrap_remove_doubling(*f32, max, min, N, *T0, prev, gain)` | `c_remove_doubling` | arch param |
| `wrap_comb_filter(...)` | `c_comb_filter` | needs window ptr from CELTMode |
| `wrap_compute_band_energies(...)` | `c_compute_band_energies` | needs CELTMode* |
| `wrap_normalise_bands(...)` | `c_normalise_bands` | needs CELTMode* |
| `wrap_denormalise_bands(...)` | `c_denormalise_bands` | needs CELTMode* |
| `wrap_bits2pulses(band, LM, bits) -> i32` | `c_bits2pulses` | needs CELTMode* |
| `wrap_pulses2bits(band, LM, k) -> i32` | `c_pulses2bits` | needs CELTMode* |
| `wrap_init_caps(*i32, LM, C)` | `c_init_caps` | needs CELTMode* |

---

## Step 3: Test Files

All test files go in `crates/opus/tests/`, following the SILK naming pattern. Each imports `opus_ffi::*` and `opus_celt::*`.

### 3A. `celt_mathops_tests.rs` (Priority: first — validates FFI plumbing)

- [x] Pattern: `flp_dsp_tests.rs`

**Integer tests (exact match):**
- `bitexact_cos`: sweep 0..16384 in steps of 100, assert `==`
- `bitexact_log2tan`: pairs like (16384,16384)→0, (32767,1)→large, sweep
- `isqrt32`: values 0, 1, 4, 9, 100, 65536, u32::MAX, assert `==`
- `celt_lcg_rand`: chain 10 iterations from seed=42, assert `==`
- `frac_mul16`: various (a, b) pairs, assert `==`

**Float tests (tolerance 1e-6):**
- `celt_exp2`: x = 0.0, 1.0, -1.0, 10.0, -50.0; sweep -20..20
- `celt_log2`: same sweep for positive values
- `celt_inner_prod`: sine×sine, noise×noise, orthogonal signals
- `celt_maxabs`: signal with known maximum
- `celt_rcp`: various positive values
- `renormalise_vector`: random vector with gain=1.0, gain=0.5

### 3B. `celt_lpc_tests.rs`

- [x] Pattern: `ffi_layer1_tests.rs`

- `celt_lpc`: autocorrelation from sine (order 24) and noise (order 10), tolerance 1e-4
- `celt_fir`: sine input + random coefficients, orders 4/8/24, tolerance 1e-5
- `celt_iir`: sine input + LPC coefficients, verify output + memory state, tolerance 1e-4
- `celt_autocorr`: sine and noise signals, lags 0..24, tolerance 1e-3

### 3C. `celt_fft_tests.rs`

- [x] FFT forward (N=120, 240, 480): delta input, known sinusoid, roundtrip FFT→IFFT
- Tolerance 1e-5

### 3D. `celt_mdct_tests.rs`

- [x] `clt_mdct_forward` (N=960, overlap=120): sine input, compare frequency-domain output
- `clt_mdct_backward`: known spectrum, compare time-domain output
- Forward→backward roundtrip, tolerance 1e-4

### 3E. `celt_pitch_tests.rs`

- [x] `pitch_downsample`: mono sine 440Hz at 48kHz, len=480, tolerance 1e-4
- `celt_pitch_xcorr`: known signals with predictable correlation, tolerance 1e-4
- `pitch_search`: sine at known period, verify detected pitch ±1 sample
- `remove_doubling`: signal with known pitch, various prev_period values

### 3F. `celt_bands_tests.rs`

- [x] `compute_band_energies`: sine signal, compare per-band energy, tolerance 1e-4
- `normalise_bands` / `denormalise_bands`: roundtrip, tolerance 1e-4
- `anti_collapse`: known collapse masks and energies

### 3G. `celt_rate_tests.rs`

- [x] `bits2pulses` / `pulses2bits`: sweep all 21 bands × LM 0..3, exact match
- `init_caps`: LM 0..3, C=1 and C=2, exact match

### 3H. `celt_quant_energy_tests.rs` (most complex — requires ec_ctx wrappers)

- [ ] `quant_coarse_energy` + `unquant_coarse_energy` roundtrip via ec buffer
- `quant_fine_energy` + `unquant_fine_energy` roundtrip
- Encode with C → decode energy with Rust and vice versa

---

## Step 4: Benchmarks

- [x] **File:** `crates/opus/benches/celt_internal.rs` (new) + update `Cargo.toml`

Criterion-based benchmarks for hot-path CELT internals:
- `bench_celt_fft_240` / `bench_celt_fft_480`: Rust vs C FFT
- `bench_celt_mdct_fwd` / `bench_celt_mdct_bwd`: MDCT transforms
- `bench_celt_pitch_xcorr`: Pitch cross-correlation
- `bench_celt_lpc`: Levinson-Durbin (order 24)

---

## Implementation Order

| Phase | What | Files Modified/Created | Depends On |
|---|---|---|---|
| **1** | Build infra + math wrappers | `build.rs`, `celt_wrapper.c` (new), `lib.rs` | — |
| **2** | Math + LPC tests | `celt_mathops_tests.rs` (new), `celt_lpc_tests.rs` (new) | Phase 1 |
| **3** | FFT/MDCT wrappers + tests | `celt_wrapper.c`, `lib.rs`, `celt_fft_tests.rs` (new), `celt_mdct_tests.rs` (new) | Phase 1 |
| **4** | Pitch wrappers + tests | `celt_wrapper.c`, `lib.rs`, `celt_pitch_tests.rs` (new) | Phase 1 |
| **5** | CELTMode-dependent wrappers + tests | `celt_wrapper.c`, `lib.rs`, `celt_bands_tests.rs` (new), `celt_rate_tests.rs` (new) | Phase 1 |
| **6** | Energy quant wrappers + tests | `celt_wrapper.c`, `lib.rs`, `celt_quant_energy_tests.rs` (new) | Phase 5 |
| **7** | Benchmarks | `celt_internal.rs` (new), `Cargo.toml` | Phases 1-5 |

---

## Key Risk: FLOAT_APPROX

The Rust `celt_exp2` uses a polynomial approximation (matching C's `FLOAT_APPROX` path). The Rust `celt_log2` uses `x.log2()` (standard math). The C build may or may not have `FLOAT_APPROX` defined.

**Resolution**: Check at Phase 2 (mathops tests). If results diverge, add `.define("FLOAT_APPROX", "1")` to the cmake config in `build.rs`, or accept tolerance differences for `celt_exp2` only.

---

## Verification

After each phase:
```bash
cargo test -p opus --test celt_mathops_tests     # Phase 2
cargo test -p opus --test celt_lpc_tests          # Phase 2
cargo test -p opus --test celt_fft_tests          # Phase 3
cargo test -p opus --test celt_mdct_tests         # Phase 3
cargo test -p opus --test celt_pitch_tests        # Phase 4
cargo test -p opus --test celt_bands_tests        # Phase 5
cargo test -p opus --test celt_rate_tests         # Phase 5
cargo test -p opus --test celt_quant_energy_tests # Phase 6
cargo bench -p opus --bench celt_internal         # Phase 7
```

Full regression: `cargo test --workspace`
