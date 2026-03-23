//! Criterion benchmarks for the C libopus encoder and decoder (via FFI).

mod common;

use common::*;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use opus_ffi::{COpusDecoder, COpusEncoder};

fn bench_c_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("c_encode");
    for cfg in &bench_configs() {
        let input_frames = generate_input_frames(cfg);
        group.throughput(Throughput::Elements(FRAMES_PER_ITER as u64));
        group.bench_function(cfg.name, |b| {
            b.iter_batched(
                || {
                    let mut enc =
                        COpusEncoder::new(SAMPLE_RATE, cfg.channels, cfg.application).unwrap();
                    enc.set_max_bandwidth(cfg.max_bandwidth).unwrap();
                    enc.set_complexity(cfg.complexity).unwrap();
                    enc.set_bitrate(cfg.bitrate).unwrap();
                    enc
                },
                |mut enc| {
                    let mut packet = vec![0u8; 4000];
                    for frame in &input_frames {
                        let _ = enc.encode_float(frame, FRAME_SIZE, &mut packet);
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_c_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("c_decode");
    for cfg in &bench_configs() {
        // Pre-encode packets using C encoder
        let packets = pre_encode(cfg);
        if packets.is_empty() {
            continue;
        }
        group.throughput(Throughput::Elements(packets.len() as u64));
        group.bench_function(cfg.name, |b| {
            b.iter_batched(
                || COpusDecoder::new(SAMPLE_RATE, cfg.channels).unwrap(),
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

fn pre_encode(cfg: &BenchConfig) -> Vec<Vec<u8>> {
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

criterion_group!(benches, bench_c_encode, bench_c_decode);
criterion_main!(benches);
