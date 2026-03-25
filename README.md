# Opus Rust

This is a Rust port of [opus v1.6](https://github.com/xiph/opus) C repository,
implementing the CELT (voice), SILK (audio) and Opus hybrid audio streaming
library. Only safe rust code has been utlized and as of version 2 the API
enforces valid configurations through Rust enums.

At this time the DNN (for noice reduction) has not been implemented. At this
time the stable 2.0.2 is being utilized for personal projects and may see
more optimizations. Opening this up to the public and royalty free [Licensing](/COPYING)
is included here and is based off the original xiph opus repository.


## Encoder/Decoder API Enums

| Enum | Variants | Replaces |
| Application | Voip, Audio, RestrictedLowDelay | OPUS_APPLICATION_* (2048/2049/2051) |
| Bandwidth | Narrowband, Mediumband, Wideband, Superwideband, Fullband | OPUS_BANDWIDTH_* (1101–1105) |
| Mode | SilkOnly, Hybrid, CeltOnly | MODE_* (1000–1002) |
| Signal | Auto, Voice, Music | OPUS_SIGNAL_* (-1000/3001/3002)/ OPUS_AUTO |
| Bitrate | Auto, Max, BitsPerSecond(i32) | OPUS_AUTO / OPUS_BITRATE_MAX / raw i32 |
| SampleRate | Hz8000, Hz12000, Hz16000, Hz24000, Hz48000 | raw i32 fs param |
| Channels | Mono, Stereo | raw i32 channels param |
| ForceChannels | Auto, Mono, Stereo | raw i32 (-1/1/2) |


## Tests and Benchmarks

Test for correctness to C version:

`cargo test -p opus --test correctness_vs_c`


Test overall opus features

`cargo test -p opus`


Benchmark both Rust implementation and C Reference

`cargo bench -p opus`

