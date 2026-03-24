//! Compare gain indices between C and Rust encoded SILK packets.
mod common;
use common::rms;
use opus::decoder::OpusDecoder;
use opus::encoder::{OpusEncoder, OPUS_APPLICATION_VOIP};
use opus::packet::*;
use opus_ffi::{COpusDecoder, COpusEncoder};

#[test]
fn compare_silk_gain_indices() {
    let fs = 16000i32;
    let frame = 320i32;
    let bitrate = 16000i32;

    let mut c_enc = COpusEncoder::new(fs, 1, opus_ffi::OPUS_APPLICATION_VOIP).unwrap();
    c_enc.set_bitrate(bitrate).unwrap();
    let mut rust_enc = OpusEncoder::new(fs, 1, OPUS_APPLICATION_VOIP).unwrap();
    rust_enc.set_bitrate(bitrate);

    let mut pcm = vec![0.0f32; frame as usize];
    let mut c_pkt = vec![0u8; 4000];
    let mut rust_pkt = vec![0u8; 4000];

    for f in 0..=15 {
        for i in 0..frame as usize {
            let t = (f * frame as usize + i) as f32 / fs as f32;
            pcm[i] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }
        let c_len = c_enc.encode_float(&pcm, frame, &mut c_pkt).unwrap();
        let rust_len = rust_enc.encode_float(&pcm, frame, &mut rust_pkt, 4000).unwrap();

        if f == 15 {
            eprintln!("=== Frame 15 ===");
            eprintln!("C pkt: {} bytes", c_len);
            eprintln!("Rust pkt: {} bytes", rust_len);

            // Decode C packet with Rust decoder to see what gain indices C used
            let mut rust_dec = OpusDecoder::new(fs, 1).unwrap();
            let mut out = vec![0.0f32; frame as usize];
            eprintln!("--- Decoding C packet with Rust decoder ---");
            rust_dec.decode_float(Some(&c_pkt[..c_len as usize]), &mut out, frame, false).unwrap();
            eprintln!("--- Decoding Rust packet with Rust decoder ---");
            let mut rust_dec2 = OpusDecoder::new(fs, 1).unwrap();
            rust_dec2.decode_float(Some(&rust_pkt[..rust_len as usize]), &mut out, frame, false).unwrap();
        }
    }
}
