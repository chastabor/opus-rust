# Remaining DNN Work — Prioritized by Dependencies

## Dependency Graph

```
build.rs weight pipeline ──┬──> FFI comparison tests
                           ├──> NoLACE model wiring
                           └──> BBWENet model wiring

SILK decoder type exposure ───> OSCE feature extraction ───> OSCE end-to-end

DRED encoder framing (independent — uses existing opus-range-coder encoder)

Stale doc comment fix (trivial, no deps)
```

---

## Step 1: Stale doc + minor cleanup (trivial)

**Files:** `crates/opus/src/dnn_decoder.rs:13`

Remove the outdated "stub" note — the DRED entropy decoder is now fully implemented.

---

## Step 2: DRED encoder bitstream framing

**Why first:** Independent of everything else. Uses existing `opus-range-coder` encoder (`enc_init`, `enc_uint`, `laplace_encode_p0`, `enc_done`). Completes the DRED encode path so encoded packets can be tested.

**What:** Port `dred_encode_silk_frame` from C `dnn/dred_encoder.c:94-195`.
- Quantize latents using `compute_quantizer` + stats data
- Encode header (q0, dQ, offset, qmax) via `ec_enc_uint`
- Encode state via `dred_encode_latents` (laplace encoding per element)
- Encode latent pairs in a loop until budget exhausted
- Return packed bytes as Opus extension 126 payload

**Files:**
- `crates/opus-dnn/src/dred/encoder.rs` — add `dred_encode_silk_frame`
- `crates/opus-dnn/src/dred/decoder.rs` — add `DredStats` to encoder model (reuse same stats for encode/decode), or add a matching `dred_encode_latents` function

**C reference:** `dnn/dred_encoder.c:94-195`

**Dependencies:** opus-range-coder encoder (exists), dred/coding.rs (exists)

---

## Step 3: OSCE feature extraction (SILK bridge)

**Why next:** This unlocks OSCE end-to-end. The feature extraction reads from SILK decoder internal state to build the 93-dim feature vector. Without it, LACE/NoLACE can't receive correct inputs.

**What:** Port `osce_calculate_features` from C `dnn/osce_features.c`.

The function needs these SILK decoder fields per subframe:
- `psDecCtrl->PredCoef_Q12` — LPC prediction coefficients (for clean spectrum)
- `psDecCtrl->pitchL[k]` — pitch lag per subframe
- `psDecCtrl->LTPCoef_Q14` — LTP coefficients
- `psDecCtrl->Gains_Q16[k]` — gains per subframe
- `psDec->LPC_order` — LPC order
- `psDec->indices.signalType` — voiced/unvoiced flag
- `psDec->nb_subfr` — number of subframes
- `psDec->osce.features` — OSCE feature state (signal_history, numbits_smooth, etc.)

**Approach:** Define an `OsceInput` struct in opus-dnn that contains the needed values as plain types (not SILK types). The caller (in `opus/src/decoder.rs` or a new `opus/src/dnn_silk_bridge.rs`) extracts these from the SILK `ChannelState` and `DecoderControl` into `OsceInput`, then passes to `osce_calculate_features`. This avoids opus-dnn depending on opus-silk.

```rust
// In opus-dnn/src/osce/features.rs
pub struct OsceInput {
    pub pred_coef_q12: [[i16; 16]; 2],  // LPC coefs per half-frame
    pub pitch_lags: [i32; 4],            // pitch per subframe
    pub ltp_coef_q14: [[i16; 5]; 4],    // LTP coefs per subframe
    pub gains_q16: [i32; 4],             // gain per subframe
    pub lpc_order: usize,
    pub signal_type: i32,
    pub nb_subfr: usize,
    pub num_bits: i32,
}
```

**Files:**
- `crates/opus-dnn/src/osce/features.rs` — implement `osce_calculate_features` with `OsceInput`
- `crates/opus/src/dnn_decoder.rs` or new `dnn_silk_bridge.rs` — extract SILK state → `OsceInput`

**C reference:** `dnn/osce_features.c:368-467`

**Dependencies:** SILK decoder types are already public in `opus-silk/src/lib.rs` (`ChannelState`, `DecoderControl`)

---

## Step 4: build.rs weight download pipeline

**Why now:** Steps 2-3 make the code functionally complete. The build.rs makes it operationally complete — models can be embedded at compile time rather than requiring runtime blob loading.

**What:**
1. `crates/opus-dnn/build.rs` — download `opus_data-{MODEL_HASH}.tar.gz` from `media.xiph.org`, extract to `model-data/`, verify SHA256
2. Parse the C `*_data.c` files and generate Rust `const` arrays in `OUT_DIR`
3. Emit `cargo:rerun-if-changed` for the model data directory
4. `crates/opus-ffi/build.rs` — enable `OPUS_DRED=ON`, `OPUS_OSCE=ON` in the cmake build (requires the same weight data for the C build)

**Files:**
- `crates/opus-dnn/build.rs` (new)
- `crates/opus-ffi/build.rs` (modify cmake flags)
- `crates/opus-dnn/src/data/mod.rs` (add `include!` for generated arrays)

**Dependencies:** Network access for download; `tar` and `sha2` crate deps for extraction/verification

---

## Step 5: NoLACE model wiring

**Why after build.rs:** With weights available, can validate the model struct matches the actual layer names. Follows the exact same pattern as LACE.

**What:** Port NoLACE processing from C `osce.c:406-750`.
- `NoLaceLayers` struct with 34 LinearLayer fields (matching field names from C)
- `init_nolace` — initialize from weights (same `weight_output_dim` + `linear_init` pattern)
- `nolace_feature_net` — same structure as LACE feature net
- `nolace_process_20ms_frame` — chain: feature net → 2x AdaComb → 4x AdaConv (with post-conv processing) → 3x AdaShape → de-emphasis

**Files:**
- `crates/opus-dnn/src/osce/nolace.rs` (new)
- `crates/opus-dnn/src/osce/mod.rs` (add NoLACE dispatch in `osce_enhance_frame`)
- `crates/opus-dnn/src/osce/structs.rs` (verify `NoLaceState` matches)

**C reference:** `dnn/osce.c:406-750`

**Dependencies:** Weight data (for validation), nndsp primitives (done)

---

## Step 6: BBWENet model wiring

**Why last among models:** BBWENet is bandwidth extension (16→48kHz), a nice-to-have beyond the core SILK enhancement. Requires a resampler that doesn't exist yet.

**What:**
- `BbwenetLayers` struct (~5 feature net layers + 3 AdaConv + 2 AdaShape)
- `init_bbwenet` from weights
- `osce_bwe` processing: 16kHz input → feature extraction → adaptive upsampling → 48kHz output
- Polyphase resampler port (C `resamp_state` from `osce_structs.h`)

**Files:**
- `crates/opus-dnn/src/osce/bbwenet.rs` (new)
- `crates/opus-dnn/src/osce/mod.rs` (add BWE dispatch)

**C reference:** `dnn/osce.c:1361-1600` (BWE processing)

**Dependencies:** Weight data, resampler (new code), nndsp (done)

---

## Step 7: FFI comparison tests

**Why last:** Requires both weight data (build.rs) and C DNN functions exposed via opus-ffi. This is the validation gate.

**What:**
1. `crates/opus-ffi/src/dnn_wrapper.c` — C shims exposing DNN internals (activations, linear, GRU, conv, PitchDNN, FARGAN, RDOVAE, OSCE)
2. `crates/opus-ffi/build.rs` — compile `dnn_wrapper.c`, link with DNN-enabled C libopus
3. `crates/opus-ffi/src/lib.rs` — Rust FFI declarations for DNN wrappers
4. `crates/opus/tests/dnn_*.rs` — comparison tests:
   - `dnn_nnet_tests.rs` — Layer 0-2 (activations, linear, GRU, conv) with handcrafted small layers
   - `dnn_pitchdnn_tests.rs` — PitchDNN feature extraction comparison
   - `dnn_fargan_tests.rs` — FARGAN synthesis comparison
   - `dnn_dred_tests.rs` — DRED encode/decode roundtrip comparison
   - `dnn_osce_tests.rs` — OSCE LACE/NoLACE enhancement comparison
   - `correctness_vs_c_dnn.rs` — Full encode/decode with DNN features

**Dependencies:** build.rs weight pipeline (Step 4), all model implementations (Steps 1-6)

---

## Summary

| Step | What | Blocked by | Effort |
|------|------|------------|--------|
| 1 | Stale doc fix | Nothing | Trivial |
| 2 | DRED encoder framing | Nothing | Medium |
| 3 | OSCE feature extraction | Nothing (uses bridge struct) | Medium |
| 4 | build.rs weight pipeline | Nothing (but needs network) | Medium-High |
| 5 | NoLACE model | Weights (Step 4) for validation | Medium |
| 6 | BBWENet model + resampler | Weights + new resampler code | High |
| 7 | FFI comparison tests | Steps 4 + all models | High |

Steps 1-3 can proceed immediately in parallel. Step 4 unblocks Steps 5-7.
