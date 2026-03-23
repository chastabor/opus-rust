//! Test SILK mode encode→decode gain against C reference.
//! Uses 16kHz mono VOIP at 16kbps to force SILK-only mode.

mod common;

use common::rms;
use opus::decoder::OpusDecoder;
use opus::encoder::{OpusEncoder, OPUS_APPLICATION_VOIP};
use opus_ffi::{COpusDecoder, COpusEncoder};

const FRAME_SIZE: i32 = 320; // 20ms at 16kHz
const MAX_PACKET: usize = 4000;

#[test]
fn silk_16k_gain_analysis() {
    let fs = 16000i32;
    let bitrate = 16000i32;

    let mut c_enc = COpusEncoder::new(fs, 1, opus_ffi::OPUS_APPLICATION_VOIP).unwrap();
    c_enc.set_bitrate(bitrate).unwrap();

    let mut rust_enc = OpusEncoder::new(fs, 1, OPUS_APPLICATION_VOIP).unwrap();
    rust_enc.set_bitrate(bitrate);

    let mut c_dec = COpusDecoder::new(fs, 1).unwrap();
    let mut rust_dec = OpusDecoder::new(fs, 1).unwrap();
    let mut rust_dec_for_c = OpusDecoder::new(fs, 1).unwrap();
    let mut c_dec_for_rust = COpusDecoder::new(fs, 1).unwrap();

    let mut pcm_in = vec![0.0f32; FRAME_SIZE as usize];
    let mut c_pkt = vec![0u8; MAX_PACKET];
    let mut rust_pkt = vec![0u8; MAX_PACKET];
    let mut c_c_out = vec![0.0f32; FRAME_SIZE as usize];
    let mut rust_rust_out = vec![0.0f32; FRAME_SIZE as usize];
    let mut c_rust_out = vec![0.0f32; FRAME_SIZE as usize];
    let mut rust_c_out = vec![0.0f32; FRAME_SIZE as usize];

    let n_warmup = 15;
    for frame in 0..=n_warmup {
        for i in 0..FRAME_SIZE as usize {
            let t = (frame * FRAME_SIZE as usize + i) as f32 / fs as f32;
            pcm_in[i] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }

        let c_len = c_enc.encode_float(&pcm_in, FRAME_SIZE, &mut c_pkt).unwrap();
        let rust_len = rust_enc
            .encode_float(&pcm_in, FRAME_SIZE, &mut rust_pkt, MAX_PACKET as i32)
            .unwrap();

        c_dec
            .decode_float(Some(&c_pkt[..c_len as usize]), &mut c_c_out, FRAME_SIZE, false)
            .unwrap();
        rust_dec
            .decode_float(
                Some(&rust_pkt[..rust_len as usize]),
                &mut rust_rust_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();
        c_dec_for_rust
            .decode_float(
                Some(&rust_pkt[..rust_len as usize]),
                &mut c_rust_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();
        rust_dec_for_c
            .decode_float(
                Some(&c_pkt[..c_len as usize]),
                &mut rust_c_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();

        if frame == n_warmup {
            let in_rms = rms(&pcm_in);
            let cc_rms = rms(&c_c_out);
            let rr_rms = rms(&rust_rust_out);
            let cr_rms = rms(&c_rust_out);
            let rc_rms = rms(&rust_c_out);

            let c_toc = c_pkt[0];
            let rust_toc = rust_pkt[0];

            eprintln!("=== SILK 16kbps gain analysis (frame {frame}) ===");
            eprintln!("  Input RMS:                {in_rms:.6}");
            eprintln!("  C enc → C dec:            {cc_rms:.6} (ratio {:.4})", cc_rms / in_rms);
            eprintln!("  Rust enc → Rust dec:      {rr_rms:.6} (ratio {:.4})", rr_rms / in_rms);
            eprintln!("  Rust enc → C dec:         {cr_rms:.6} (ratio {:.4})", cr_rms / in_rms);
            eprintln!("  C enc → Rust dec:         {rc_rms:.6} (ratio {:.4})", rc_rms / in_rms);
            eprintln!(
                "  C pkt: size={c_len}, TOC=0x{c_toc:02x} (config={})",
                (c_toc >> 3) & 0x1F
            );
            eprintln!(
                "  Rust pkt: size={rust_len}, TOC=0x{rust_toc:02x} (config={})",
                (rust_toc >> 3) & 0x1F
            );

            let gain_ratio = rr_rms / cc_rms.max(1e-10);
            eprintln!("  Rust/C gain ratio:        {gain_ratio:.4}");
        }
    }
}
