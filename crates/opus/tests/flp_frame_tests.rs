//! Integration tests for the float frame encoder (Layer 4).
//! Tests that silk_encode_frame_flp produces valid output for test signals.

use opus_range_coder::EcCtx;
use opus_silk::encoder_flp::encode_frame::silk_encode_frame_flp;
use opus_silk::nsq::NsqState;
use opus_silk::*;

const FS_KHZ: i32 = 16;
const NB_SUBFR: i32 = 4;
const SUBFR_LENGTH: i32 = 80;
const FRAME_LENGTH: i32 = 320;
const LTP_MEM_LENGTH: i32 = 320; // 20ms * 16kHz
const LPC_ORDER: i32 = 16;
const SHAPING_LPC_ORDER: i32 = 16;
// C: shapeWinLength = SUB_FRAME_LENGTH_MS * fs_kHz + 2 * la_shape
const SHAPE_WIN_LENGTH: i32 = 5 * FS_KHZ + 2 * 5 * FS_KHZ; // subfr_len + 2*la_shape = 80 + 160 = 240
const LA_SHAPE: usize = 5 * FS_KHZ as usize; // 80

fn gen_sine_i16(len: usize, freq: f32, fs: f32, amplitude: f32) -> Vec<i16> {
    (0..len)
        .map(|i| {
            let s = amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin();
            (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

/// Test that the float frame encoder produces a valid packet for a sine wave.
#[test]
fn encode_frame_flp_produces_output() {
    let nlsf_cb = get_nlsf_cb(NlsfCbSel::Wb);
    let ltp_mem = LTP_MEM_LENGTH as usize;
    let x_buf_len = ltp_mem + LA_SHAPE + FRAME_LENGTH as usize;
    let mut x_buf = vec![0.0f32; x_buf_len];
    let mut nsq_state = NsqState::new();
    let mut indices = SideInfoIndices::default();
    let mut prev_nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    let mut prev_lag = 0i32;
    let mut first_frame_after_reset = true;
    let mut last_gain_index = 10i8;
    let mut prev_harm_smth = 0.0f32;
    let mut prev_tilt_smth = 0.0f32;
    let input_quality_bands = [16384i32, 16384, 16384, 16384]; // moderate quality
    let snr_db_q7 = 2415; // ~19 dB, typical for 16kbps WB

    let max_packet = 1275;
    let mut scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];

    // Encode multiple frames of a 440Hz sine
    let mut total_bytes = 0;
    for frame in 0..16 {
        let input = gen_sine_i16(
            FRAME_LENGTH as usize,
            440.0,
            16000.0,
            0.5,
        );

        let mut enc = EcCtx::enc_init(max_packet as u32);
        // Write VAD + LBRR flags (2 bits)
        enc.enc_bit_logp(true, 1);
        enc.enc_bit_logp(false, 1);

        let bytes = silk_encode_frame_flp(
            &mut x_buf,
            &mut nsq_state,
            &mut indices,
            &mut prev_nlsf_q15,
            &mut prev_signal_type,
            &mut prev_lag,
            &mut first_frame_after_reset,
            &mut last_gain_index,
            &mut prev_harm_smth,
            &mut prev_tilt_smth,
            255, // speech_activity_q8
            &input_quality_bands,
            0, // input_tilt_q15
            snr_db_q7,
            &input,
            FS_KHZ,
            NB_SUBFR,
            SUBFR_LENGTH,
            FRAME_LENGTH,
            LTP_MEM_LENGTH,
            LPC_ORDER,
            SHAPING_LPC_ORDER, SHAPE_WIN_LENGTH,
            0, // warping_q16
            10, // complexity
            nlsf_cb,
            (max_packet - 1) * 8,
            &mut enc,
            &mut scratch_s_ltp_q15,
            &mut scratch_s_ltp,
            &mut scratch_x_sc_q10,
            &mut scratch_xq_tmp,
        );

        if frame == 15 {
            total_bytes = bytes;
            eprintln!("[frame {}] bytes={} gain_idx={:?} interp={}",
                frame, bytes,
                &indices.gains_indices[..NB_SUBFR as usize],
                indices.nlsf_interp_coef_q2);
        }
    }

    eprintln!("Float frame encoder: {} bytes on frame 15", total_bytes);
    assert!(total_bytes > 0, "Frame encoder produced no output");
    assert!(total_bytes < 1275, "Frame encoder produced oversized output");
}

/// Test that the encoder handles the first-frame-after-reset case.
#[test]
fn encode_frame_flp_first_frame() {
    let nlsf_cb = get_nlsf_cb(NlsfCbSel::Wb);
    let ltp_mem = LTP_MEM_LENGTH as usize;
    let x_buf_len = ltp_mem + LA_SHAPE + FRAME_LENGTH as usize;
    let mut x_buf = vec![0.0f32; x_buf_len];
    let mut nsq_state = NsqState::new();
    let mut indices = SideInfoIndices::default();
    let mut prev_nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    let mut prev_lag = 0i32;
    let mut first_frame_after_reset = true;
    let mut last_gain_index = 10i8;
    let mut prev_harm_smth = 0.0f32;
    let mut prev_tilt_smth = 0.0f32;

    let input = gen_sine_i16(FRAME_LENGTH as usize, 440.0, 16000.0, 0.5);

    let mut enc = EcCtx::enc_init(1275);
    enc.enc_bit_logp(true, 1);
    enc.enc_bit_logp(false, 1);

    let mut scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];

    let bytes = silk_encode_frame_flp(
        &mut x_buf, &mut nsq_state, &mut indices,
        &mut prev_nlsf_q15, &mut prev_signal_type, &mut prev_lag,
        &mut first_frame_after_reset, &mut last_gain_index,
        &mut prev_harm_smth, &mut prev_tilt_smth,
        255, &[16384; 4], 0, 2415, &input,
        FS_KHZ, NB_SUBFR, SUBFR_LENGTH, FRAME_LENGTH,
        LTP_MEM_LENGTH, LPC_ORDER, SHAPING_LPC_ORDER, SHAPE_WIN_LENGTH,
        0, 10, get_nlsf_cb(NlsfCbSel::Wb), 1275 * 8,
        &mut enc,
        &mut scratch_s_ltp_q15, &mut scratch_s_ltp,
        &mut scratch_x_sc_q10, &mut scratch_xq_tmp,
    );

    eprintln!("First frame: {} bytes, first_frame_after_reset now = {}", bytes, first_frame_after_reset);
    assert!(bytes > 0, "First frame produced no output");
    assert!(!first_frame_after_reset, "first_frame_after_reset should be cleared");
}
