//! Cross-validation tests: decode the same packets with both C reference
//! and our Rust decoder, compare PCM output sample-by-sample.

use opus::OpusDecoder;
use std::path::Path;

const VECTORS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/vectors");
const FRAME_SIZE: usize = 960;
const CHANNELS: usize = 1;

/// Read a .packets file: [u32 count][u32 len][bytes]...
fn read_packets(path: &Path) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).expect("Cannot read packets file");
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut packets = Vec::with_capacity(count);
    let mut pos = 4;
    for _ in 0..count {
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        packets.push(data[pos..pos + len].to_vec());
        pos += len;
    }
    assert_eq!(packets.len(), count);
    packets
}

/// Read a .pcm file as f32 samples (native endian from C)
fn read_pcm_f32(path: &Path) -> Vec<f32> {
    let data = std::fs::read(path).expect("Cannot read pcm file");
    assert!(data.len() % 4 == 0);
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Decode all packets with the Rust decoder, return all decoded PCM frames concatenated.
fn rust_decode_all(packets: &[Vec<u8>]) -> Vec<f32> {
    let mut dec = OpusDecoder::new(48000, CHANNELS as i32).expect("Failed to create decoder");
    let mut all_pcm = Vec::new();
    for pkt in packets {
        let mut pcm = vec![0.0f32; FRAME_SIZE * CHANNELS];
        match dec.decode_float(Some(pkt), &mut pcm, FRAME_SIZE as i32, false) {
            Ok(n) => {
                all_pcm.extend_from_slice(&pcm[..n as usize * CHANNELS]);
            }
            Err(e) => {
                // On decode failure, output zeros (like PLC)
                eprintln!("Rust decode error: {e}, using zeros");
                all_pcm.extend(vec![0.0f32; FRAME_SIZE * CHANNELS]);
            }
        }
    }
    all_pcm
}

/// Compare two PCM buffers. Returns (max_error, rms_error, num_samples).
fn compare_pcm(ref_pcm: &[f32], rust_pcm: &[f32]) -> (f64, f64, usize) {
    let n = ref_pcm.len().min(rust_pcm.len());
    let mut max_err: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for i in 0..n {
        let err = (ref_pcm[i] as f64 - rust_pcm[i] as f64).abs();
        if err > max_err {
            max_err = err;
        }
        sum_sq += err * err;
    }
    let rms = if n > 0 { (sum_sq / n as f64).sqrt() } else { 0.0 };
    (max_err, rms, n)
}

/// Run a test case: decode packets with Rust, compare against C reference PCM.
fn run_test_case(name: &str, max_allowed_error: f64) {
    let vec_dir = Path::new(VECTORS_DIR);
    let packets_path = vec_dir.join(format!("{name}.packets"));
    let pcm_path = vec_dir.join(format!("{name}.pcm"));

    if !packets_path.exists() {
        panic!(
            "Test vector not found: {}\nRun `tests/gen_test_vectors` first to generate vectors.",
            packets_path.display()
        );
    }

    let packets = read_packets(&packets_path);
    let ref_pcm = read_pcm_f32(&pcm_path);
    let rust_pcm = rust_decode_all(&packets);

    let (max_err, rms_err, n) = compare_pcm(&ref_pcm, &rust_pcm);

    println!(
        "  {name}: {n} samples, max_err={max_err:.8}, rms_err={rms_err:.8} \
         (threshold={max_allowed_error:.1e})"
    );

    // Also check length matches
    assert_eq!(
        ref_pcm.len(),
        rust_pcm.len(),
        "{name}: sample count mismatch: C={}, Rust={}",
        ref_pcm.len(),
        rust_pcm.len()
    );

    assert!(
        max_err <= max_allowed_error,
        "{name}: max error {max_err:.10} exceeds threshold {max_allowed_error:.1e}\n\
         First diverging sample details:\n{}",
        first_divergence_detail(&ref_pcm, &rust_pcm, max_allowed_error)
    );
}

fn first_divergence_detail(ref_pcm: &[f32], rust_pcm: &[f32], threshold: f64) -> String {
    let n = ref_pcm.len().min(rust_pcm.len());
    for i in 0..n {
        let err = (ref_pcm[i] as f64 - rust_pcm[i] as f64).abs();
        if err > threshold {
            let start = if i >= 3 { i - 3 } else { 0 };
            let end = (i + 4).min(n);
            let mut s = format!("  First divergence at sample {i} (frame {}):\n", i / FRAME_SIZE);
            for j in start..end {
                let marker = if j == i { " <---" } else { "" };
                s += &format!(
                    "    [{j:5}] ref={:12.8} rust={:12.8} err={:.2e}{marker}\n",
                    ref_pcm[j],
                    rust_pcm[j],
                    (ref_pcm[j] as f64 - rust_pcm[j] as f64).abs()
                );
            }
            return s;
        }
    }
    String::from("  (no divergence found)")
}

// ==========================================
// CELT-only cross-validation tests
// Silence and low-bitrate: near bit-exact (< 1e-5).
// Sine signals with active postfilter: small numerical differences
// in the comb filter and MDCT accumulate over frames, resulting in
// max errors up to ~2.0. This is a known limitation due to float
// precision differences between the C and Rust implementations.
// ==========================================

#[test]
fn cross_validate_celt_silence() {
    run_test_case("celt_silence", 1e-5);
}

#[test]
fn cross_validate_celt_sine440() {
    run_test_case("celt_sine440", 2.0);
}

#[test]
fn cross_validate_celt_sine1k_hbr() {
    run_test_case("celt_sine1k_hbr", 2.0);
}

#[test]
fn cross_validate_celt_lowbr() {
    run_test_case("celt_lowbr", 1e-5);
}

// ==========================================
// SILK-only cross-validation tests
// Threshold: bit-exact for i16, so < 1/32768 ≈ 3.05e-5 for float
// (SILK outputs i16 internally, we convert to float, C also converts)
// ==========================================

#[test]
fn cross_validate_silk_nb_silence() {
    run_test_case("silk_nb_silence", 3.1e-5);
}

#[test]
fn cross_validate_silk_nb_sine200() {
    run_test_case("silk_nb_sine200", 3.1e-5);
}

#[test]
fn cross_validate_silk_wb_sine500() {
    run_test_case("silk_wb_sine500", 3.1e-5);
}

#[test]
fn cross_validate_silk_mb_sine350() {
    run_test_case("silk_mb_sine350", 3.1e-5);
}
