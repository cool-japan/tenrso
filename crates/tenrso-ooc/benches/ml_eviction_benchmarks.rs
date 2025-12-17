//! Benchmarks for ML-based eviction policies
//!
//! This benchmark suite measures the performance of the ML eviction policy
//! compared to simple heuristics and evaluates the overhead of online learning.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenrso_ooc::ml_eviction::{MLConfig, MLEvictionPolicy};

/// Benchmark recording accesses
fn bench_record_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_eviction_record_access");

    for num_chunks in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*num_chunks as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_chunks),
            num_chunks,
            |b, &num_chunks| {
                let mut policy = MLEvictionPolicy::new(MLConfig::default());

                b.iter(|| {
                    for chunk_id in 0..num_chunks {
                        policy.record_access(black_box(chunk_id), black_box(100.0));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark getting eviction candidates
fn bench_get_eviction_candidates(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_eviction_get_candidates");

    for num_chunks in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*num_chunks as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_chunks),
            num_chunks,
            |b, &num_chunks| {
                let mut policy = MLEvictionPolicy::new(MLConfig::default());

                // Populate with some access history
                for chunk_id in 0..num_chunks {
                    for t in 0..10 {
                        policy.record_access(chunk_id, t as f64 * 10.0);
                    }
                }

                let chunk_ids: Vec<usize> = (0..num_chunks).collect();

                b.iter(|| {
                    let candidates =
                        policy.get_eviction_candidates(black_box(&chunk_ids), black_box(200.0));
                    black_box(candidates);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark feature extraction overhead
fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_eviction_feature_extraction");

    for num_accesses in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*num_accesses as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_accesses),
            num_accesses,
            |b, &num_accesses| {
                let mut policy = MLEvictionPolicy::new(MLConfig::default());

                // Record access history
                for t in 0..num_accesses {
                    policy.record_access(0, t as f64 * 10.0);
                }

                b.iter(|| {
                    let candidates =
                        policy.get_eviction_candidates(black_box(&[0]), black_box(1000.0));
                    black_box(candidates);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark online learning overhead
fn bench_online_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_eviction_online_learning");

    group.bench_function("record_with_learning", |b| {
        let mut policy = MLEvictionPolicy::new(MLConfig::default());

        // Pre-populate so learning kicks in
        for chunk_id in 0..10 {
            for t in 0..20 {
                policy.record_access(chunk_id, t as f64 * 10.0);
            }
        }

        let mut time = 200.0;

        b.iter(|| {
            policy.record_access(black_box(0), black_box(time));
            time += 1.0;
        });
    });

    group.finish();
}

/// Benchmark ensemble vs single model
fn bench_ensemble_vs_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_eviction_ensemble_comparison");

    for use_ensemble in [true, false].iter() {
        group.bench_with_input(
            BenchmarkId::new("ensemble", use_ensemble),
            use_ensemble,
            |b, &use_ensemble| {
                let config = MLConfig {
                    use_ensemble,
                    ..Default::default()
                };
                let mut policy = MLEvictionPolicy::new(config);

                // Populate with access history
                for chunk_id in 0..100 {
                    for t in 0..10 {
                        policy.record_access(chunk_id, t as f64 * 10.0);
                    }
                }

                let chunk_ids: Vec<usize> = (0..100).collect();

                b.iter(|| {
                    let candidates =
                        policy.get_eviction_candidates(black_box(&chunk_ids), black_box(200.0));
                    black_box(candidates);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark realistic workload simulation
fn bench_realistic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_eviction_realistic_workload");

    group.bench_function("mixed_access_patterns", |b| {
        b.iter(|| {
            let mut policy = MLEvictionPolicy::new(MLConfig::default());

            // Hot chunks (frequently accessed)
            for t in 0..50 {
                policy.record_access(0, t as f64 * 5.0);
                if t % 2 == 0 {
                    policy.record_access(1, t as f64 * 5.0);
                }
            }

            // Cold chunks (rarely accessed)
            policy.record_access(2, 10.0);
            policy.record_access(3, 20.0);

            // Sequential chunks
            for t in 0..30 {
                policy.record_access(4, 300.0 + t as f64 * 10.0);
            }

            // Random access
            for t in 0..20 {
                let chunk_id = (t * 7) % 10;
                policy.record_access(chunk_id, 500.0 + t as f64 * 8.0);
            }

            // Get eviction candidates
            let chunk_ids: Vec<usize> = (0..10).collect();
            let candidates =
                policy.get_eviction_candidates(black_box(&chunk_ids), black_box(1000.0));

            black_box(candidates);
        });
    });

    group.finish();
}

/// Benchmark memory overhead
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_eviction_memory");

    for num_chunks in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_chunks),
            num_chunks,
            |b, &num_chunks| {
                b.iter(|| {
                    let mut policy = MLEvictionPolicy::new(MLConfig::default());

                    // Track many chunks
                    for chunk_id in 0..num_chunks {
                        for t in 0..5 {
                            policy.record_access(chunk_id, t as f64 * 10.0);
                        }
                        policy.set_chunk_metadata(chunk_id, 1024 * 1024, 0);
                    }

                    black_box(policy);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark prediction accuracy improvement over time
fn bench_learning_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_eviction_learning_convergence");

    group.bench_function("training_iterations", |b| {
        b.iter(|| {
            let mut policy = MLEvictionPolicy::new(MLConfig::default());

            // Simulate predictable access pattern
            for iteration in 0..100 {
                for chunk_id in 0..10 {
                    // Chunks 0-4: accessed every 10 time units
                    // Chunks 5-9: accessed every 50 time units
                    let interval = if chunk_id < 5 { 10.0 } else { 50.0 };

                    if iteration as f64 % interval == 0.0 {
                        policy.record_access(chunk_id, iteration as f64);
                    }
                }
            }

            let stats = policy.get_stats();
            black_box(stats);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_record_access,
    bench_get_eviction_candidates,
    bench_feature_extraction,
    bench_online_learning,
    bench_ensemble_vs_single,
    bench_realistic_workload,
    bench_memory_overhead,
    bench_learning_convergence,
);

criterion_main!(benches);
