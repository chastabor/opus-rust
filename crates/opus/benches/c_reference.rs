//! Criterion benchmarks for the C libopus encoder and decoder (via FFI).

mod common;

use common::*;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use opus_ffi::{COpusDecoder, COpusEncoder};

fn bench_c_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("c_encode");
    for cfg in &bench_configs() {
        let input_frames = generate_input_frames(cfg);
        group.throughput(Throughput::Elements(FRAMES_PER_ITER as u64));
        group.bench_function(cfg.name, |b| {
            b.iter_batched(
                || {
                    let mut enc = COpusEncoder::new(
                        i32::from(SAMPLE_RATE),
                        i32::from(cfg.channels),
                        i32::from(cfg.application),
                    )
                    .unwrap();
                    enc.set_max_bandwidth(i32::from(cfg.max_bandwidth)).unwrap();
                    enc.set_complexity(cfg.complexity).unwrap();
                    enc.set_bitrate(cfg.bitrate).unwrap();
                    (enc, vec![0u8; MAX_PACKET])
                },
                |(mut enc, mut packet)| {
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
        let packets = pre_encode_with_c(cfg);
        if packets.is_empty() {
            continue;
        }
        group.throughput(Throughput::Elements(packets.len() as u64));
        group.bench_function(cfg.name, |b| {
            b.iter_batched(
                || {
                    let dec =
                        COpusDecoder::new(i32::from(SAMPLE_RATE), i32::from(cfg.channels)).unwrap();
                    let pcm = vec![0.0f32; FRAME_SIZE as usize * i32::from(cfg.channels) as usize];
                    (dec, pcm)
                },
                |(mut dec, mut pcm)| {
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

criterion_group!(benches, bench_c_encode, bench_c_decode);
criterion_main!(benches);
