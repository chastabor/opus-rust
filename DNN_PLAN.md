# DNN Feature Implementation Plan for opus-rust

## Context

C libopus has added neural network features: DRED (Deep Redundancy), OSCE (Opus Speech Enhancement), LPCNet-based deep PLC (Packet Loss Concealment), FARGAN vocoder, and PitchDNN. These features share a common NN inference runtime (`dnn/nnet.*`, `dnn/nndsp.*`) and are conditionally compiled via `OPUS_DRED`, `OPUS_OSCE`, `OPUS_DEEP_PLC` cmake flags.

This plan ports those features to Rust, structured for incremental comparison testing against C at every layer.

---

## New Crate: `opus-dnn`

Single new workspace member at `crates/opus-dnn/`. All DNN code lives here, feature-gated by Cargo features. Follows the project's `unsafe-code = "forbid"` policy.

```toml
[features]
default = ["dred", "osce", "deep-plc"]
deep-plc = []       # LPCNet PLC + FARGAN + PitchDNN
dred = ["deep-plc"] # DRED requires LPCNet feature extraction
osce = []           # OSCE (LACE, NoLACE, BBWENet) - decoder only
```

**Dependencies:** `opus-range-coder` (DRED entropy coding), `opus-celt` (pitch_xcorr, celt_fir reuse).

### Module Layout

```
crates/opus-dnn/src/
  lib.rs

  # Core NN runtime (always built)
  nnet/
    mod.rs            # LinearLayer, Conv2dLayer, WeightArray types
    activations.rs    # vec_sigmoid (tansig table), vec_tanh, relu, swish, softmax
    linear.rs         # compute_linear: sgemv, cgemv8x4, sparse variants
    conv2d.rs         # compute_conv2d, conv2d_float, conv2d_3x3_float
    ops.rs            # compute_generic_dense, compute_generic_gru, compute_glu,
                      #   compute_generic_conv1d, compute_generic_conv1d_dilation
    weights.rs        # parse_weights, linear_init, conv2d_init

  # Neural DSP (needed by OSCE)
  nndsp/
    mod.rs            # AdaConvState, AdaCombState, AdaShapeState
    adaconv.rs        # adaconv_process_frame
    adacomb.rs        # adacomb_process_frame
    adashape.rs       # adashape_process_frame

  # Shared utilities
  freq.rs             # Band energy, burg cepstral analysis, DCT, LPC from cepstrum
  burg.rs             # Burg's method (dnn/burg.c)

  # PitchDNN (feature: deep-plc)
  pitchdnn.rs         # PitchDNNState, compute_pitchdnn

  # FARGAN vocoder (feature: deep-plc)
  fargan/
    mod.rs            # FARGANState, fargan_init/cont/synthesize

  # LPCNet feature extraction + PLC (feature: deep-plc)
  lpcnet/
    mod.rs            # LPCNetEncState, LPCNetPLCState
    enc.rs            # compute_frame_features
    plc.rs            # lpcnet_plc_init/update/conceal/fec_add

  # DRED (feature: dred)
  dred/
    mod.rs            # DREDEnc, OpusDRED, constants
    coding.rs         # compute_quantizer, dred_decode_latents
    rdovae_enc.rs     # RDOVAEEncState, dred_rdovae_encode_dframe
    rdovae_dec.rs     # RDOVAEDecState, dred_rdovae_decode_qframe
    encoder.rs        # dred_encoder_init, dred_compute_latents
    decoder.rs        # dred_ec_decode

  # OSCE (feature: osce)
  osce/
    mod.rs            # OSCEModel, osce_enhance_frame, osce_load_models
    config.rs         # Constants from osce_config.h
    structs.rs        # LACEState, NoLACEState, BBWENetState
    features.rs       # OSCE feature extraction
    lace.rs           # LACE processing
    nolace.rs         # NoLACE processing
    bbwenet.rs        # BBWENet bandwidth extension

  # Weight data
  data/
    mod.rs            # Embedded weight arrays, model init functions
```

---

## Component Layers for Comparison Testing

Each layer is independently testable against C via FFI wrappers. Tests go in `crates/opus/tests/dnn_*.rs` (cross-crate, needs FFI).

### Layer 0: Activations
**C files:** `nnet_arch.h` (compute_activation_), `vec.h` (vec_sigmoid, vec_tanh, tansig_table.h)
**Rust:** `nnet/activations.rs`
**Test:** Random float vectors through each activation type, exact match vs C

### Layer 1: Linear Algebra
**C files:** `nnet_arch.h` (compute_linear_), `vec.h` (sgemv, cgemv8x4, sparse variants)
**Rust:** `nnet/linear.rs`
**Test:** Handcrafted small LinearLayers (8x4 float, 16x8 int8, sparse), compare output vs C

### Layer 2: Composite NN Ops
**C files:** `nnet.c` (compute_generic_dense, compute_generic_gru, compute_glu, compute_generic_conv1d, compute_generic_conv1d_dilation), `nnet_arch.h` (compute_conv2d_)
**Rust:** `nnet/ops.rs`, `nnet/conv2d.rs`
**Test:** Handcrafted layers + state buffers, compare outputs including GRU state evolution

### Layer 3: Weight Parsing
**C files:** `nnet.c` (parse_weights), `nnet.h` (linear_init, conv2d_init, WeightArray/WeightHead)
**Rust:** `nnet/weights.rs`
**Test:** Parse same binary blob in C and Rust, verify array counts/names/sizes/dimensions match

### Layer 4: Neural DSP
**C files:** `nndsp.c` (adaconv_process_frame, adacomb_process_frame, adashape_process_frame)
**Rust:** `nndsp/*.rs`
**Test:** Handcrafted features + layers + pitch, compare frame output

### Layer 5: Feature-Level Components
**C files:** `pitchdnn.c`, `lpcnet_enc.c`, `fargan.c`, `dred_rdovae_enc.c`, `dred_rdovae_dec.c`, `osce_features.c`
**Rust:** `pitchdnn.rs`, `lpcnet/enc.rs`, `fargan/mod.rs`, `dred/rdovae_*.rs`, `osce/features.rs`
**Test:** Full model weights loaded, compare per-frame outputs on test signals

### Layer 6: High-Level Operations
**C files:** `lpcnet_plc.c`, `dred_encoder.c`, `dred_decoder.c`, `dred_coding.c`, `osce.c`
**Rust:** `lpcnet/plc.rs`, `dred/encoder.rs`, `dred/decoder.rs`, `osce/mod.rs`
**Test:** Multi-frame sequences, PLC concealment, DRED encode/decode roundtrip, OSCE enhancement

### Layer 7: Opus Integration
**Rust:** `opus/src/encoder.rs`, `opus/src/decoder.rs`, `opus-silk/src/decoder.rs`
**Test:** Full encode/decode with DNN features enabled, C-vs-Rust PCM comparison

---

## FFI Extension Plan

### Step 1: Enable DNN in C build (`opus-ffi/build.rs`)

Change cmake flags:
```rust
.define("OPUS_DRED", "ON")
.define("OPUS_OSCE", "ON")
```

The C build requires weight data files. The `opus-c/dnn/download_model.sh` script fetches them; `build.rs` needs to run this or bundle them.

### Step 2: Create `opus-ffi/src/dnn_wrapper.c`

Following the existing `celt_wrapper.c` pattern, expose DNN internals:

**Layer 0-2 (synthetic small layers, no weight files needed):**
- `wrap_compute_activation_c(float *out, const float *in, int N, int activation)`
- `wrap_compute_linear_c(const LinearLayer *linear, float *out, const float *in)`
- `wrap_compute_generic_dense(...)`, `wrap_compute_generic_gru(...)`, `wrap_compute_glu(...)`
- `wrap_compute_generic_conv1d(...)`, `wrap_compute_conv2d_c(...)`

**Layer 3+ (requires initialized models):**
- `wrap_init_pitchdnn(PitchDNNState *st)` / `wrap_compute_pitchdnn(...)`
- `wrap_init_fargan(FARGANState *st)` / `wrap_fargan_synthesize(...)` / `wrap_fargan_cont(...)`
- `wrap_compute_frame_features(LPCNetEncState *st, const float *in)`
- `wrap_dred_rdovae_encode_dframe(...)` / `wrap_dred_rdovae_decode_qframe(...)`
- `wrap_adaconv_process_frame(...)` / `wrap_adacomb_process_frame(...)` / `wrap_adashape_process_frame(...)`
- `wrap_osce_enhance_frame(...)`

### Step 3: Rust FFI bindings in `opus-ffi/src/lib.rs`

Corresponding `unsafe extern "C"` declarations + safe wrappers. For Layer 0-2 tests, expose a way to construct small `LinearLayer` structs in C from Rust-provided data.

---

## Weight Data Strategy

### How C libopus handles weights

Weight data is **not** checked into the C repo. The `autogen.sh` script calls
`dnn/download_model.sh` with a SHA256 hash that doubles as the filename:

```
https://media.xiph.org/opus/models/opus_data-<sha256>.tar.gz
```

Current hash: `a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e`

The tarball extracts into `dnn/` and contains:
- **Pre-generated C source files** (the actual build inputs):
  `fargan_data.c/h`, `plc_data.c/h`, `pitchdnn_data.c/h`, `lace_data.c/h`,
  `nolace_data.c/h`, `bbwenet_data.c/h`, `dred_rdovae_enc_data.c/h`,
  `dred_rdovae_dec_data.c/h`, `dred_rdovae_stats_data.c/h`,
  `dred_rdovae_constants.h`, `lossgen_data.c/h`
- **PyTorch model files** (`.pth`, for retraining only — not needed at build):
  `dnn/models/lace_v2.pth`, `fargan_sq1Ab_adv_50.pth`, etc.

### Download at build time, not checked in

Both `opus-ffi` (C build) and `opus-dnn` (Rust build) must download the weight
data at build time. Weights are stored under a gitignored directory.

**Download location:** `crates/opus-dnn/model-data/`
- Downloaded once by `build.rs`, cached across rebuilds
- Contains the extracted tarball contents (C data files + .pth models)
- Added to `.gitignore` so binary data is never committed

**`opus-ffi/build.rs` changes:**
- Before cmake, download and extract the tarball into the C `dnn/` directory
  (where cmake expects the `*_data.c` files to exist)
- Use the same SHA256 hash and URL as `autogen.sh`
- Skip download if files already exist (cache check)

**`opus-dnn/build.rs`:**
- Downloads the same tarball to `crates/opus-dnn/model-data/`
- Parses the C `*_data.c` files and generates Rust `const` arrays via codegen
  (writes to `OUT_DIR` so generated code is not committed)
- Alternatively, reads the `.pth` files directly (more maintainable long-term
  but requires understanding the PyTorch checkpoint format)
- Emits `cargo:rerun-if-changed` for the model data directory

**`.gitignore` additions:**
```
# DNN model weights (downloaded at build time)
crates/opus-dnn/model-data/
```

### Weight data constants

The model hash and URL are defined as constants in `build.rs`:
```rust
const MODEL_HASH: &str = "a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e";
const MODEL_URL: &str = "https://media.xiph.org/opus/models";
// Download: {MODEL_URL}/opus_data-{MODEL_HASH}.tar.gz
```

When the upstream model is updated, only this hash needs to change.

### In Rust library code (`opus-dnn`)

- `LinearLayer` stores owned `Vec<f32>` / `Vec<i8>` for weights (not raw pointers)
- Embedded weights: `build.rs` codegen from downloaded C data files into `OUT_DIR`
- External loading: port `parse_weights` for `OPUS_SET_DNN_BLOB` CTL (runtime override)
- One-time copy from static data at model init is acceptable; weights are loaded once

```rust
pub struct LinearLayer {
    pub bias: Option<Vec<f32>>,
    pub subias: Option<Vec<f32>>,
    pub weights: Option<Vec<i8>>,       // int8 quantized
    pub float_weights: Option<Vec<f32>>, // float dense
    pub weights_idx: Option<Vec<i32>>,   // sparse indices
    pub diag: Option<Vec<f32>>,          // GRU diagonal
    pub scale: Option<Vec<f32>>,         // int8 scale
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}
```

---

## Implementation Phases

### Phase 1: NN Infrastructure (Layers 0-2)
**Build:** `opus-dnn` crate skeleton + `nnet/` module (activations, linear, conv2d, ops)
**FFI:** `dnn_wrapper.c` with Layer 0-2 wrappers, enable DNN in C build
**Test:** `dnn_nnet_tests.rs` - activations, linear, GRU, conv1d, conv2d vs C with handcrafted small layers
**Milestone:** All NN primitives match C output

### Phase 2: Weight Parsing + Model Init (Layer 3)
**Build:** `nnet/weights.rs` (parse_weights, linear_init, conv2d_init), model struct definitions in `data/`
**FFI:** Weight blob parsing comparison
**Test:** `dnn_weights_tests.rs` - parse blob, verify array names/sizes; init each model, verify dimensions
**Milestone:** All models parse and initialize from weight data

### Phase 3: PitchDNN + LPCNet Features (Layer 5a)
**Build:** `freq.rs`, `burg.rs`, `pitchdnn.rs`, `lpcnet/enc.rs`
**FFI:** `wrap_compute_pitchdnn`, `wrap_compute_frame_features`, `wrap_burg_cepstral_analysis`
**Test:** `dnn_pitchdnn_tests.rs`, `dnn_lpcnet_features_tests.rs` - sine/noise signals, compare feature vectors
**Milestone:** Feature extraction pipeline matches C

### Phase 4: FARGAN + LPCNet PLC (Layers 5b-6a)
**Build:** `fargan/mod.rs`, `lpcnet/plc.rs`
**FFI:** `wrap_fargan_cont/synthesize`, `wrap_lpcnet_plc_update/conceal/fec_add`
**Test:** `dnn_fargan_tests.rs`, `dnn_plc_tests.rs` - multi-frame synthesis, loss concealment
**Milestone:** PLC concealment matches C

### Phase 5: DRED Encoder + Decoder (Layers 5c-6b)
**Build:** `dred/` module (coding, rdovae_enc, rdovae_dec, encoder, decoder)
**FFI:** `wrap_dred_rdovae_encode/decode_dframe`, `wrap_dred_ec_decode`, `wrap_dred_compute_latents`
**Test:** `dnn_dred_tests.rs` - RDOVAE roundtrip, extension parsing, latent computation
**Milestone:** DRED encode/decode is bitstream-compatible with C

### Phase 6: OSCE (Layers 4-6c)
**Build:** `nndsp/` module, `osce/` module (features, lace, nolace, bbwenet)
**FFI:** `wrap_adaconv/adacomb/adashape_process_frame`, `wrap_osce_enhance_frame`
**Test:** `dnn_nndsp_tests.rs`, `dnn_osce_tests.rs` - adaptive layer outputs, SILK enhancement
**Milestone:** OSCE enhancement matches C

### Phase 7: Opus Integration (Layer 7)
**Build:** Extend `OpusEncoder` with `DREDEnc`, `OpusDecoder` with `LPCNetPLCState`, `SilkDecoder` with `OSCEModel`
**New CTLs:** `OPUS_SET_DRED_DURATION`, `OPUS_SET_DNN_BLOB`, `OPUS_SET_OSCE_BWE`
**Test:** `correctness_vs_c_dnn.rs` - full encode/decode with DNN, packet loss simulation
**Milestone:** End-to-end DNN encode/decode matches C reference

---

## Key C Source Files Reference

| Component | C Files | Rust Module |
|-----------|---------|-------------|
| NN types | `dnn/nnet.h` | `nnet/mod.rs` |
| Activations | `dnn/nnet_arch.h`, `dnn/vec.h`, `dnn/tansig_table.h` | `nnet/activations.rs` |
| Linear/sgemv | `dnn/nnet_arch.h`, `dnn/vec.h` | `nnet/linear.rs` |
| Composite ops | `dnn/nnet.c` | `nnet/ops.rs` |
| Conv2D | `dnn/nnet_arch.h` | `nnet/conv2d.rs` |
| Weight parsing | `dnn/nnet.c` (parse_weights) | `nnet/weights.rs` |
| Neural DSP | `dnn/nndsp.c`, `dnn/nndsp.h` | `nndsp/*.rs` |
| PitchDNN | `dnn/pitchdnn.c`, `dnn/pitchdnn.h` | `pitchdnn.rs` |
| Freq utils | `dnn/freq.c`, `dnn/freq.h` | `freq.rs` |
| Burg analysis | `dnn/burg.c`, `dnn/burg.h` | `burg.rs` |
| FARGAN | `dnn/fargan.c`, `dnn/fargan.h` | `fargan/mod.rs` |
| LPCNet enc | `dnn/lpcnet_enc.c`, `dnn/lpcnet_private.h` | `lpcnet/enc.rs` |
| LPCNet PLC | `dnn/lpcnet_plc.c` | `lpcnet/plc.rs` |
| DRED config | `dnn/dred_config.h` | `dred/mod.rs` |
| DRED coding | `dnn/dred_coding.c` | `dred/coding.rs` |
| RDOVAE enc | `dnn/dred_rdovae_enc.c` | `dred/rdovae_enc.rs` |
| RDOVAE dec | `dnn/dred_rdovae_dec.c` | `dred/rdovae_dec.rs` |
| DRED encoder | `dnn/dred_encoder.c` | `dred/encoder.rs` |
| DRED decoder | `dnn/dred_decoder.c` | `dred/decoder.rs` |
| OSCE config | `dnn/osce_config.h` | `osce/config.rs` |
| OSCE structs | `dnn/osce_structs.h` | `osce/structs.rs` |
| OSCE features | `dnn/osce_features.c` | `osce/features.rs` |
| OSCE main | `dnn/osce.c` | `osce/mod.rs` |
| Encoder integ. | `src/opus_encoder.c` | `opus/src/encoder.rs` |
| Decoder integ. | `src/opus_decoder.c` | `opus/src/decoder.rs` |
| SILK dec integ. | `silk/decode_frame.c` | `opus-silk/src/decoder.rs` |

---

## Verification

At each phase, run:
```bash
# Phase-specific tests
cargo test --package opus --test dnn_nnet_tests -- --nocapture
cargo test --package opus --test dnn_weights_tests -- --nocapture
# ... etc per phase

# Full suite (should not regress existing tests)
cargo test --package opus

# Clippy
cargo clippy --workspace --tests --benches
```

Final integration test:
```bash
cargo test --package opus --test correctness_vs_c_dnn -- --nocapture
```
