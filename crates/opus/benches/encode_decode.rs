//! Criterion benchmarks for the Rust Opus encoder and decoder.

mod common;

use common::*;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use opus::decoder::OpusDecoder;
use opus::encoder::OpusEncoder;

fn bench_rust_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_encode");
    for cfg in &bench_configs() {
        let input_frames = generate_input_frames(cfg);
        group.throughput(Throughput::Elements(FRAMES_PER_ITER as u64));
        group.bench_function(cfg.name, |b| {
            b.iter_batched(
                || {
                    let mut enc =
                        OpusEncoder::new(SAMPLE_RATE, cfg.channels, cfg.application).unwrap();
                    enc.set_bandwidth(cfg.max_bandwidth);
                    enc.set_complexity(cfg.complexity);
                    enc.set_bitrate(cfg.bitrate);
                    enc
                },
                |mut enc| {
                    let mut packet = vec![0u8; 4000];
                    for frame in &input_frames {
                        let _ = enc.encode_float(frame, FRAME_SIZE, &mut packet, 4000);
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_rust_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_decode");
    for cfg in &bench_configs() {
        // Pre-encode packets using a C encoder (authoritative packets).
        let packets = pre_encode_with_c(cfg);
        if packets.is_empty() {
            continue;
        }
        group.throughput(Throughput::Elements(packets.len() as u64));
        group.bench_function(cfg.name, |b| {
            b.iter_batched(
                || OpusDecoder::new(SAMPLE_RATE, cfg.channels).unwrap(),
                |mut dec| {
                    let mut pcm = vec![0.0f32; FRAME_SIZE as usize * cfg.channels as usize];
                    for pkt in &packets {
                        let _ = dec.decode_float(Some(pkt), &mut pcm, FRAME_SIZE, false);
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Pre-encode packets using the C encoder for decode benchmarks.
fn pre_encode_with_c(cfg: &BenchConfig) -> Vec<Vec<u8>> {
    use opus_ffi::COpusEncoder;
    let mut enc = COpusEncoder::new(SAMPLE_RATE, cfg.channels, cfg.application).unwrap();
    enc.set_max_bandwidth(cfg.max_bandwidth).unwrap();
    enc.set_complexity(cfg.complexity).unwrap();
    enc.set_bitrate(cfg.bitrate).unwrap();

    let input_frames = generate_input_frames(cfg);
    let mut packets = Vec::with_capacity(input_frames.len());
    for frame in &input_frames {
        let mut packet = vec![0u8; 4000];
        match enc.encode_float(frame, FRAME_SIZE, &mut packet) {
            Ok(len) => packets.push(packet[..len as usize].to_vec()),
            Err(_) => return Vec::new(),
        }
    }
    packets
}

criterion_group!(benches, bench_rust_encode, bench_rust_decode);
criterion_main!(benches);
