//! Benchmarks for data integrity validation and compression auto-selection.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tenrso_ooc::data_integrity::{
    ChecksumAlgorithm, ChunkIntegrityMetadata, IntegrityChecker, ValidationPolicy,
};

#[cfg(feature = "compression")]
use tenrso_ooc::compression_auto::{AutoSelectConfig, CompressionAutoSelector, SelectionPolicy};

// Data Integrity Benchmarks

fn bench_checksum_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("checksum_algorithms");

    let sizes = [1024, 10_240, 102_400, 1_048_576]; // 1KB, 10KB, 100KB, 1MB

    for size in &sizes {
        let data = vec![0u8; *size];
        group.throughput(Throughput::Bytes(*size as u64));

        for algo in &[
            ChecksumAlgorithm::Crc32,
            ChecksumAlgorithm::XxHash64,
            ChecksumAlgorithm::Blake3,
        ] {
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", algo), size),
                &data,
                |b, data| {
                    b.iter(|| {
                        ChunkIntegrityMetadata::new(
                            "bench".to_string(),
                            data.len(),
                            vec![*size],
                            *algo,
                            std::hint::black_box(data),
                        )
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_validation_policies(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_policies");

    let data = vec![42u8; 10_240]; // 10KB
    group.throughput(Throughput::Bytes(data.len() as u64));

    let metadata = ChunkIntegrityMetadata::new(
        "bench".to_string(),
        data.len(),
        vec![10_240],
        ChecksumAlgorithm::XxHash64,
        &data,
    );

    for policy in &[
        ValidationPolicy::None,
        ValidationPolicy::Opportunistic,
        ValidationPolicy::Strict,
    ] {
        let mut checker = IntegrityChecker::new(ChecksumAlgorithm::XxHash64, *policy);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", policy)),
            &data,
            |b, data| {
                b.iter(|| {
                    checker
                        .validate(std::hint::black_box(&metadata), std::hint::black_box(data))
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_validation");

    let chunk_counts = [10, 100, 1000];

    for count in &chunk_counts {
        let chunks: Vec<Vec<u8>> = (0..*count).map(|_| vec![42u8; 1024]).collect();
        let metadata: Vec<ChunkIntegrityMetadata> = chunks
            .iter()
            .enumerate()
            .map(|(i, data)| {
                ChunkIntegrityMetadata::new(
                    format!("chunk_{}", i),
                    data.len(),
                    vec![1024],
                    ChecksumAlgorithm::XxHash64,
                    data,
                )
            })
            .collect();

        group.throughput(Throughput::Elements(*count as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &(chunks, metadata),
            |b, (chunks, metadata)| {
                b.iter(|| {
                    let mut checker = IntegrityChecker::new(
                        ChecksumAlgorithm::XxHash64,
                        ValidationPolicy::Strict,
                    );
                    for (chunk, meta) in chunks.iter().zip(metadata.iter()) {
                        checker
                            .validate(std::hint::black_box(meta), std::hint::black_box(chunk))
                            .unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// Compression Auto-Selection Benchmarks

#[cfg(feature = "compression")]
fn bench_entropy_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_calculation");

    let sizes = [1024, 10_240, 102_400]; // 1KB, 10KB, 100KB

    for size in &sizes {
        // Different data patterns
        let uniform = vec![42u8; *size];
        let sequential: Vec<u8> = (0..*size).map(|i| (i % 256) as u8).collect();
        let random: Vec<u8> = (0..*size).map(|i| ((i * 37) % 256) as u8).collect();

        group.throughput(Throughput::Bytes(*size as u64));

        for (name, data) in &[
            ("uniform", &uniform),
            ("sequential", &sequential),
            ("random", &random),
        ] {
            group.bench_with_input(BenchmarkId::new(*name, size), data, |b, data| {
                let selector = CompressionAutoSelector::new();
                b.iter(|| selector.analyze_data(std::hint::black_box(data)));
            });
        }
    }

    group.finish();
}

#[cfg(feature = "compression")]
fn bench_codec_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_selection");

    let data = vec![42u8; 10_240]; // 10KB uniform data

    for policy in &[
        SelectionPolicy::MaxCompression,
        SelectionPolicy::MaxSpeed,
        SelectionPolicy::Balanced,
        SelectionPolicy::Adaptive,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", policy)),
            &data,
            |b, data| {
                let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
                    policy: *policy,
                    ..Default::default()
                });
                b.iter(|| selector.select_codec(std::hint::black_box(data)));
            },
        );
    }

    group.finish();
}

#[cfg(feature = "compression")]
fn bench_selection_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("selection_overhead");

    let sizes = [1024, 10_240, 102_400, 1_048_576]; // 1KB to 1MB

    for size in &sizes {
        let data = vec![42u8; *size];
        group.throughput(Throughput::Bytes(*size as u64));

        // With sampling (default 8KB)
        group.bench_with_input(BenchmarkId::new("with_sampling", size), &data, |b, data| {
            let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
                sample_size: 8192,
                ..Default::default()
            });
            b.iter(|| selector.select_codec(std::hint::black_box(data)));
        });

        // Full analysis (no sampling)
        group.bench_with_input(BenchmarkId::new("full_analysis", size), &data, |b, data| {
            let mut selector = CompressionAutoSelector::with_config(AutoSelectConfig {
                sample_size: 0, // Disable sampling
                ..Default::default()
            });
            b.iter(|| selector.select_codec(std::hint::black_box(data)));
        });
    }

    group.finish();
}

#[cfg(feature = "compression")]
fn bench_pattern_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_detection");

    let size = 10_240;
    group.throughput(Throughput::Bytes(size as u64));

    let uniform = vec![0u8; size];
    let sequential: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    let sparse = {
        let mut v = vec![0u8; size];
        for i in (0..size).step_by(100) {
            v[i] = (i / 100) as u8;
        }
        v
    };
    let random: Vec<u8> = (0..size).map(|i| ((i * 37 + 17) % 256) as u8).collect();

    for (name, data) in &[
        ("uniform", &uniform),
        ("sequential", &sequential),
        ("sparse", &sparse),
        ("random", &random),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(*name), data, |b, data| {
            let selector = CompressionAutoSelector::new();
            b.iter(|| selector.analyze_data(std::hint::black_box(data)));
        });
    }

    group.finish();
}

criterion_group!(
    integrity_benches,
    bench_checksum_algorithms,
    bench_validation_policies,
    bench_batch_validation
);

#[cfg(feature = "compression")]
criterion_group!(
    autoselect_benches,
    bench_entropy_calculation,
    bench_codec_selection,
    bench_selection_overhead,
    bench_pattern_detection
);

#[cfg(feature = "compression")]
criterion_main!(integrity_benches, autoselect_benches);

#[cfg(not(feature = "compression"))]
criterion_main!(integrity_benches);
