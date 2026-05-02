# CELT Testing Framework: C Reference Cross-Validation

## Status: All Phases Complete

**55 cross-validation tests** across 8 test files, all passing.
**27 C wrapper functions** in `celt_wrapper.c`, **34 safe Rust wrappers** in `opus-ffi/src/lib.rs`.
**8 Criterion benchmarks** (Rust vs C) for FFT, MDCT, pitch xcorr, and LPC.

### Bugs Found and Fixed
1. **`frac_mul16` missing rounding bias** (`mathops.rs`) — Rust was `((a*b) >> 15)` but C is `((16384 + (i16)a * (i16)b) >> 15)`. Fixed to match C exactly. This also fixed `bitexact_cos`.
2. **`celt_lpc` leftover fixed-point scaling** (`lpc.rs`) — Rust had `/64.0` and `*64.0` factors from a fixed-point port that should be identity in float mode. LPC coefficients were completely wrong. Fixed by removing all scaling.

### Key Finding: FLOAT_APPROX
The C build uses standard math (no `FLOAT_APPROX`). The Rust `celt_exp2` uses the polynomial approximation while `celt_log2` uses standard `x.log2()`. Tests confirm `celt_log2` matches C exactly; `celt_exp2` matches within 1e-5 relative tolerance.

### Key Finding: Range Coder Bitstream Parity
The Rust and C energy quantization encoders produce **identical band energy states** but **different byte-level bitstream representations**. Cross-decoding (C-encoded bytes → Rust decoder) works perfectly. The full pipeline `correctness_vs_c.rs` tests pass within tolerance.

---

## Cross-Validated Functions

| Module | Functions Tested | Test File | Tests |
|---|---|---|---|
| **mathops.rs** | `celt_exp2`, `celt_log2`, `celt_rcp`, `bitexact_cos`, `bitexact_log2tan`, `isqrt32`, `frac_mul16`, `celt_lcg_rand`, `celt_inner_prod`, `celt_maxabs`, `renormalise_vector` | `celt_mathops_tests.rs` | 16 |
| **lpc.rs** | `celt_lpc`, `celt_fir`, `celt_iir`, `celt_autocorr` | `celt_lpc_tests.rs` | 9 |
| **fft.rs** | `opus_fft` (N=120, 240, 480) | `celt_fft_tests.rs` | 4 |
| **mdct.rs** | `clt_mdct_forward`, `clt_mdct_backward` (shift=0, shift=1) | `celt_mdct_tests.rs` | 4 |
| **pitch.rs** | `celt_pitch_xcorr`, `pitch_downsample`, `pitch_search`, `remove_doubling`, `comb_filter` | `celt_pitch_tests.rs` | 8 |
| **bands.rs** | `compute_band_energies`, `normalise_bands`, `denormalise_bands`, `anti_collapse` | `celt_bands_tests.rs` | 4 |
| **rate.rs** | `bits2pulses`, `pulses2bits`, `init_caps` | `celt_rate_tests.rs` | 5 |
| **quant_energy.rs** | `quant_coarse_energy`, `unquant_coarse_energy`, `quant_fine_energy`, `unquant_fine_energy`, `quant_energy_finalise`, `unquant_energy_finalise` | `celt_quant_energy_tests.rs` | 5 |

### Not Yet Cross-Validated (tested indirectly via full pipeline)

| Module | Functions | Notes |
|---|---|---|
| **bands.rs** | `quant_all_bands` / `quant_all_bands_enc` | Full band quantization; tested via `correctness_vs_c.rs` |
| **rate.rs** | `clt_compute_allocation` | Requires full ec_ctx + many allocation parameters |
| **encoder.rs** | `CeltEncoder::encode_with_ec`, `celt_preemphasis` | Tested via `correctness_vs_c.rs` full pipeline |
| **decoder.rs** | `CeltDecoder::decode_with_ec`, `celt_synthesis`, `deemphasis` | Tested via `correctness_vs_c.rs` full pipeline |

---

## Implementation Summary

| Phase | What | Status | Files |
|---|---|---|---|
| **1** | Build infra + C wrappers | Done | `build.rs`, `celt_wrapper.c`, `lib.rs` |
| **2** | Math + LPC tests (25 tests) | Done | `celt_mathops_tests.rs`, `celt_lpc_tests.rs` |
| **3** | FFT + MDCT tests (8 tests) | Done | `celt_fft_tests.rs`, `celt_mdct_tests.rs` |
| **4** | Pitch tests (8 tests) | Done | `celt_pitch_tests.rs` |
| **5** | Band + rate tests (9 tests) | Done | `celt_bands_tests.rs`, `celt_rate_tests.rs` |
| **6** | Energy quant tests (5 tests) | Done | `celt_quant_energy_tests.rs` |
| **7** | Benchmarks (8 benchmarks) | Done | `celt_internal.rs`, `Cargo.toml` |

---

## Verification

```bash
# All CELT cross-validation tests
cargo test -p opus --test celt_mathops_tests      # 16 tests
cargo test -p opus --test celt_lpc_tests           # 9 tests
cargo test -p opus --test celt_fft_tests           # 4 tests
cargo test -p opus --test celt_mdct_tests          # 4 tests
cargo test -p opus --test celt_pitch_tests         # 8 tests
cargo test -p opus --test celt_bands_tests         # 4 tests
cargo test -p opus --test celt_rate_tests          # 5 tests
cargo test -p opus --test celt_quant_energy_tests  # 5 tests

# Benchmarks (Rust vs C)
cargo bench -p opus --bench celt_internal

# Full regression
cargo test --workspace
```
