//! Benchmarks for advanced features: batch I/O, tiered memory, working set prediction

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use std::hint::black_box;
use tenrso_core::DenseND;
use tenrso_ooc::{
    BatchConfig, BatchReader, BatchWriter, PredictionMode, TierAccessPattern, TierConfig,
    TieredMemoryManager, WorkingSetPredictor,
};

fn bench_batch_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_io");

    // Create test chunks
    let temp_dir = std::env::temp_dir().join("tenrso_bench_batch_io");
    std::fs::create_dir_all(&temp_dir).unwrap();

    for num_chunks in [10, 50, 100] {
        let mut chunks = HashMap::new();
        let mut shapes = HashMap::new();

        for i in 0..num_chunks {
            let id = format!("bench_{}", i);
            let data = vec![i as f64; 1000]; // 8KB per chunk
            let tensor = DenseND::from_vec(data, &[10, 100]).unwrap();
            shapes.insert(id.clone(), vec![10, 100]);
            chunks.insert(id, tensor);
        }

        let total_bytes = num_chunks * 1000 * 8;
        group.throughput(Throughput::Bytes(total_bytes as u64));

        // Benchmark sequential write
        group.bench_with_input(
            BenchmarkId::new("write_sequential", num_chunks),
            &num_chunks,
            |b, _| {
                b.iter(|| {
                    let writer = BatchWriter::new(BatchConfig::default().parallel(false));
                    writer.write_batch(&chunks, &temp_dir).unwrap()
                });
            },
        );

        // Benchmark parallel write
        group.bench_with_input(
            BenchmarkId::new("write_parallel", num_chunks),
            &num_chunks,
            |b, _| {
                b.iter(|| {
                    let writer = BatchWriter::new(BatchConfig::default().parallel(true));
                    writer.write_batch(&chunks, &temp_dir).unwrap()
                });
            },
        );

        // Write once for read benchmarks
        let writer = BatchWriter::new(BatchConfig::default());
        writer.write_batch(&chunks, &temp_dir).unwrap();

        let chunk_ids: Vec<String> = chunks.keys().cloned().collect();

        // Benchmark sequential read
        group.bench_with_input(
            BenchmarkId::new("read_sequential", num_chunks),
            &num_chunks,
            |b, _| {
                b.iter(|| {
                    let reader = BatchReader::new(BatchConfig::default().parallel(false));
                    reader.read_batch(&chunk_ids, &shapes, &temp_dir).unwrap()
                });
            },
        );

        // Benchmark parallel read
        group.bench_with_input(
            BenchmarkId::new("read_parallel", num_chunks),
            &num_chunks,
            |b, _| {
                b.iter(|| {
                    let reader = BatchReader::new(BatchConfig::default().parallel(true));
                    reader.read_batch(&chunk_ids, &shapes, &temp_dir).unwrap()
                });
            },
        );
    }

    group.finish();

    // Cleanup
    std::fs::remove_dir_all(&temp_dir).unwrap();
}

fn bench_tiered_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered_memory");

    // Benchmark chunk registration and tier management
    for num_chunks in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("register_chunks", num_chunks),
            &num_chunks,
            |b, &n| {
                b.iter(|| {
                    let mut manager = TieredMemoryManager::new()
                        .tier_config(TierConfig {
                            ram_mb: 10,
                            ssd_mb: 50,
                            disk_mb: 200,
                        })
                        .auto_migration(false); // Disable for consistent benchmarks

                    for i in 0..n {
                        let data = vec![i as f64; 1000];
                        let tensor = DenseND::from_vec(data, &[10, 100]).unwrap();
                        manager
                            .register_chunk(
                                &format!("chunk_{}", i),
                                tensor,
                                TierAccessPattern::Random,
                            )
                            .unwrap();
                    }

                    black_box(manager);
                });
            },
        );

        // Benchmark chunk access patterns
        group.bench_with_input(
            BenchmarkId::new("access_chunks", num_chunks),
            &num_chunks,
            |b, &n| {
                // Setup
                let mut manager = TieredMemoryManager::new()
                    .tier_config(TierConfig {
                        ram_mb: 10,
                        ssd_mb: 50,
                        disk_mb: 200,
                    })
                    .auto_migration(true);

                for i in 0..n {
                    let data = vec![i as f64; 1000];
                    let tensor = DenseND::from_vec(data, &[10, 100]).unwrap();
                    manager
                        .register_chunk(&format!("chunk_{}", i), tensor, TierAccessPattern::Random)
                        .unwrap();
                }

                // Benchmark
                b.iter(|| {
                    // Access pattern: hot chunks accessed frequently
                    for i in 0..(n / 5) {
                        let _ = manager.get_chunk(&format!("chunk_{}", i)).unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_working_set_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("working_set_prediction");

    // Benchmark different prediction modes
    for mode in [
        PredictionMode::Frequency,
        PredictionMode::Recency,
        PredictionMode::Hybrid,
        PredictionMode::Adaptive,
        PredictionMode::Sequential,
    ] {
        let mode_name = format!("{:?}", mode);

        group.bench_with_input(BenchmarkId::new("predict", &mode_name), &mode, |b, &m| {
            // Setup: record access history
            let mut predictor = WorkingSetPredictor::new()
                .prediction_mode(m)
                .window_size(100);

            // Simulate access pattern
            for round in 0..10 {
                for i in 0..20 {
                    predictor.record_access(&format!("chunk_{}", i), 8000);
                }
                // Hot chunks accessed more
                if round % 2 == 0 {
                    for i in 0..5 {
                        predictor.record_access(&format!("chunk_{}", i), 8000);
                    }
                }
            }

            // Benchmark prediction
            b.iter(|| {
                let predictions = predictor.predict_working_set(10).unwrap();
                black_box(predictions);
            });
        });
    }

    group.finish();
}

fn bench_integrated_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("integrated_workflow");

    // Benchmark complete workflow: tiered memory + working set prediction
    group.bench_function("complete_workflow", |b| {
        b.iter(|| {
            let mut tier_manager = TieredMemoryManager::new()
                .tier_config(TierConfig {
                    ram_mb: 5,
                    ssd_mb: 20,
                    disk_mb: 100,
                })
                .auto_migration(true);

            let mut ws_predictor = WorkingSetPredictor::new()
                .prediction_mode(PredictionMode::Adaptive)
                .window_size(50);

            // Create and register chunks
            for i in 0..30 {
                let data = vec![i as f64; 500];
                let tensor = DenseND::from_vec(data, &[10, 50]).unwrap();
                let pattern = if i < 10 {
                    TierAccessPattern::Temporal
                } else {
                    TierAccessPattern::Random
                };
                tier_manager
                    .register_chunk(&format!("chunk_{}", i), tensor, pattern)
                    .unwrap();
            }

            // Simulate workload with prediction
            for _ in 0..5 {
                // Access hot chunks
                for i in 0..10 {
                    let _ = tier_manager.get_chunk(&format!("chunk_{}", i)).unwrap();
                    ws_predictor.record_access(&format!("chunk_{}", i), 4000);
                }

                // Predict and prefetch
                let predictions = ws_predictor.predict_working_set(5).unwrap();
                for pred in predictions {
                    let _ = tier_manager.get_chunk(&pred.chunk_id);
                }
            }

            black_box((tier_manager, ws_predictor));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_io,
    bench_tiered_memory,
    bench_working_set_prediction,
    bench_integrated_workflow
);
criterion_main!(benches);
