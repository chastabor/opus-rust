# Opus Rust

This is a Rust port of [opus v1.6](https://github.com/xiph/opus) C repository,
implementing the CELT (voice), SILK (audio) and Opus hybrid audio streaming
library. Only safe rust code has been utlized and as of version 2 the API
enforces valid configurations through Rust enums.

As of version 2.1 the optional `dnn` feature enables DNN-based enhancements
ported from libopus 1.6: **DRED** (Deep REDundancy) for resilient encoding,
**deep PLC** (FARGAN + PitchDNN) for packet loss concealment, and **OSCE**
(LACE/NoLACE) for speech enhancement. DNN weights are loaded at runtime via
`load_dnn()` — see the DNN section below.

Opening this up to the public and royalty free [Licensing](/COPYING)
is included here and is based off the original xiph opus repository.


## Encoder/Decoder API Enums

| Enum | Variants | Replaces |
| ---- | -------- | -------- |
| Application | Voip, Audio, RestrictedLowDelay | OPUS_APPLICATION_* (2048/2049/2051) |
| Bandwidth | Narrowband, Mediumband, Wideband, Superwideband, Fullband | OPUS_BANDWIDTH_* (1101–1105) |
| Mode | SilkOnly, Hybrid, CeltOnly | MODE_* (1000–1002) |
| Signal | Auto, Voice, Music | OPUS_SIGNAL_* (-1000/3001/3002)/ OPUS_AUTO |
| Bitrate | Auto, Max, BitsPerSecond(i32) | OPUS_AUTO / OPUS_BITRATE_MAX / raw i32 |
| SampleRate | Hz8000, Hz12000, Hz16000, Hz24000, Hz48000 | raw i32 fs param |
| Channels | Mono, Stereo | raw i32 channels param |
| ForceChannels | Auto, Mono, Stereo | raw i32 (-1/1/2) |


## DNN Features (optional)

Enable the `dnn` feature to access DRED, deep PLC, and OSCE:

```toml
[dependencies]
opus = { path = "crates/opus", features = ["dnn"] }
```

Load DNN weights at runtime from the binary blob (same format as C libopus
`OPUS_SET_DNN_BLOB`):

```rust
use opus::{OpusEncoder, OpusDecoder, SampleRate, Channels, Application};

// Encoder: load weights and enable DRED
let mut enc = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip)?;
enc.load_dnn(&weight_blob)?;
enc.set_dred_duration(10); // 10 frames of deep redundancy

// Decoder: load weights for deep PLC + OSCE
let mut dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono)?;
dec.load_dnn(&weight_blob)?;
```

| Method | Available on | Description |
| ------ | ------------ | ----------- |
| `load_dnn(&[u8])` | Encoder, Decoder | Load DNN model weights from binary blob |
| `set_dred_duration(i32)` | Encoder | Set DRED redundancy frames (0 = disabled) |
| `dred_duration()` | Encoder | Query current DRED duration |
| `dnn_loaded()` | Encoder, Decoder | Check if DNN models are loaded |


## Tests and Benchmarks

Test for correctness to C version:

`cargo test -p opus --test correctness_vs_c`


Test overall opus features

`cargo test -p opus`


Benchmark both Rust implementation and C Reference

`cargo bench -p opus`

