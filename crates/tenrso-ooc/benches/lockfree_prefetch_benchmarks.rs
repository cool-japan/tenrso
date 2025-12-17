//! Benchmarks comparing lock-free prefetcher with mutex-based prefetcher
//!
//! Run with: cargo bench --bench lockfree_prefetch_benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;
use std::thread;
use tenrso_core::DenseND;

#[cfg(feature = "lock-free")]
use tenrso_ooc::lockfree_prefetch::{LockFreePrefetcher, PrefetchStrategy as LFStrategy};

use parking_lot::Mutex;
use tenrso_ooc::prefetch::{PrefetchStrategy, Prefetcher};

/// Benchmark single-threaded schedule performance
fn bench_single_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_schedule");

    for num_chunks in [10, 100, 1000] {
        group.throughput(Throughput::Elements(num_chunks as u64));

        // Mutex-based prefetcher (wrapped in Arc<Mutex<>>)
        group.bench_with_input(
            BenchmarkId::new("mutex", num_chunks),
            &num_chunks,
            |b, &n| {
                let prefetcher = Arc::new(Mutex::new(
                    Prefetcher::new()
                        .strategy(PrefetchStrategy::Sequential)
                        .queue_size(n as usize),
                ));

                b.iter(|| {
                    let chunks: Vec<String> = (0..n).map(|i| format!("chunk_{}", i)).collect();
                    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();

                    let mut pf = prefetcher.lock();
                    let _ = pf.schedule_prefetch(chunk_refs);
                    drop(pf);

                    // Clear for next iteration
                    prefetcher.lock().clear();
                });
            },
        );

        // Lock-free prefetcher
        #[cfg(feature = "lock-free")]
        group.bench_with_input(
            BenchmarkId::new("lockfree", num_chunks),
            &num_chunks,
            |b, &n| {
                let prefetcher = Arc::new(
                    LockFreePrefetcher::new()
                        .strategy(LFStrategy::Sequential)
                        .queue_size(n as usize),
                );

                b.iter(|| {
                    let chunks: Vec<String> = (0..n).map(|i| format!("chunk_{}", i)).collect();
                    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();

                    let _ = prefetcher.schedule_prefetch(chunk_refs);

                    // Clear for next iteration
                    prefetcher.clear();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent schedule from multiple threads
#[cfg(feature = "lock-free")]
fn bench_concurrent_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_schedule");

    for num_threads in [2, 4, 8] {
        let chunks_per_thread = 100;
        group.throughput(Throughput::Elements(
            (num_threads * chunks_per_thread) as u64,
        ));

        // Mutex-based prefetcher
        group.bench_with_input(
            BenchmarkId::new("mutex", num_threads),
            &num_threads,
            |b, &threads| {
                let prefetcher = Arc::new(Mutex::new(
                    Prefetcher::new()
                        .strategy(PrefetchStrategy::Sequential)
                        .queue_size(threads * chunks_per_thread),
                ));

                b.iter(|| {
                    let mut handles = vec![];
                    for t in 0..threads {
                        let pf = prefetcher.clone();
                        let handle = thread::spawn(move || {
                            let chunks: Vec<String> = (0..chunks_per_thread)
                                .map(|i| format!("chunk_{}_{}", t, i))
                                .collect();
                            let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();

                            let mut pf_guard = pf.lock();
                            let _ = pf_guard.schedule_prefetch(chunk_refs);
                        });
                        handles.push(handle);
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    // Clear for next iteration
                    prefetcher.lock().clear();
                });
            },
        );

        // Lock-free prefetcher
        group.bench_with_input(
            BenchmarkId::new("lockfree", num_threads),
            &num_threads,
            |b, &threads| {
                let prefetcher = Arc::new(
                    LockFreePrefetcher::new()
                        .strategy(LFStrategy::Sequential)
                        .queue_size(threads * chunks_per_thread),
                );

                b.iter(|| {
                    let mut handles = vec![];
                    for t in 0..threads {
                        let pf = prefetcher.clone();
                        let handle = thread::spawn(move || {
                            let chunks: Vec<String> = (0..chunks_per_thread)
                                .map(|i| format!("chunk_{}_{}", t, i))
                                .collect();
                            let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();

                            let _ = pf.schedule_prefetch(chunk_refs);
                        });
                        handles.push(handle);
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    // Clear for next iteration
                    prefetcher.clear();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark add/get operations
fn bench_add_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_get");

    for num_ops in [10, 100, 1000] {
        group.throughput(Throughput::Elements(num_ops as u64));

        // Mutex-based
        group.bench_with_input(BenchmarkId::new("mutex", num_ops), &num_ops, |b, &n| {
            let prefetcher = Arc::new(Mutex::new(Prefetcher::new()));

            b.iter(|| {
                // Add tensors
                for i in 0..n {
                    let tensor = DenseND::<f64>::from_vec(vec![i as f64; 4], &[2, 2]).unwrap();
                    let chunk_id = format!("chunk_{}", i);
                    prefetcher.lock().add_prefetched(&chunk_id, tensor);
                }

                // Get tensors
                for i in 0..n {
                    let chunk_id = format!("chunk_{}", i);
                    let _ = black_box(prefetcher.lock().get(&chunk_id));
                }
            });
        });

        // Lock-free
        #[cfg(feature = "lock-free")]
        group.bench_with_input(BenchmarkId::new("lockfree", num_ops), &num_ops, |b, &n| {
            let prefetcher = Arc::new(LockFreePrefetcher::new());

            b.iter(|| {
                // Add tensors
                for i in 0..n {
                    let tensor = DenseND::<f64>::from_vec(vec![i as f64; 4], &[2, 2]).unwrap();
                    let chunk_id = format!("chunk_{}", i);
                    prefetcher.add_prefetched(&chunk_id, tensor);
                }

                // Get tensors
                for i in 0..n {
                    let chunk_id = format!("chunk_{}", i);
                    let _ = black_box(prefetcher.get(&chunk_id));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark concurrent add/get from multiple threads
#[cfg(feature = "lock-free")]
fn bench_concurrent_add_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_add_get");

    for num_threads in [2, 4, 8] {
        let ops_per_thread = 100;
        group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));

        // Mutex-based
        group.bench_with_input(
            BenchmarkId::new("mutex", num_threads),
            &num_threads,
            |b, &threads| {
                let prefetcher = Arc::new(Mutex::new(Prefetcher::new()));

                b.iter(|| {
                    // Writer threads
                    let mut handles = vec![];
                    for t in 0..threads {
                        let pf = prefetcher.clone();
                        let handle = thread::spawn(move || {
                            for i in 0..ops_per_thread {
                                let tensor =
                                    DenseND::<f64>::from_vec(vec![t as f64, i as f64], &[2])
                                        .unwrap();
                                let chunk_id = format!("chunk_{}_{}", t, i);
                                pf.lock().add_prefetched(&chunk_id, tensor);
                            }
                        });
                        handles.push(handle);
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    // Reader threads
                    let mut handles = vec![];
                    for t in 0..threads {
                        let pf = prefetcher.clone();
                        let handle = thread::spawn(move || {
                            for i in 0..ops_per_thread {
                                let chunk_id = format!("chunk_{}_{}", t, i);
                                let _ = black_box(pf.lock().get(&chunk_id));
                            }
                        });
                        handles.push(handle);
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    prefetcher.lock().clear();
                });
            },
        );

        // Lock-free
        group.bench_with_input(
            BenchmarkId::new("lockfree", num_threads),
            &num_threads,
            |b, &threads| {
                let prefetcher = Arc::new(LockFreePrefetcher::new());

                b.iter(|| {
                    // Writer threads
                    let mut handles = vec![];
                    for t in 0..threads {
                        let pf = prefetcher.clone();
                        let handle = thread::spawn(move || {
                            for i in 0..ops_per_thread {
                                let tensor =
                                    DenseND::<f64>::from_vec(vec![t as f64, i as f64], &[2])
                                        .unwrap();
                                let chunk_id = format!("chunk_{}_{}", t, i);
                                pf.add_prefetched(&chunk_id, tensor);
                            }
                        });
                        handles.push(handle);
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    // Reader threads
                    let mut handles = vec![];
                    for t in 0..threads {
                        let pf = prefetcher.clone();
                        let handle = thread::spawn(move || {
                            for i in 0..ops_per_thread {
                                let chunk_id = format!("chunk_{}_{}", t, i);
                                let _ = black_box(pf.get(&chunk_id));
                            }
                        });
                        handles.push(handle);
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    prefetcher.clear();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark record_access for adaptive prefetching
fn bench_record_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_access");

    for num_accesses in [10, 100, 1000] {
        group.throughput(Throughput::Elements(num_accesses as u64));

        // Mutex-based
        group.bench_with_input(
            BenchmarkId::new("mutex", num_accesses),
            &num_accesses,
            |b, &n| {
                let prefetcher = Arc::new(Mutex::new(
                    Prefetcher::new()
                        .strategy(PrefetchStrategy::Adaptive)
                        .history_window(100),
                ));

                b.iter(|| {
                    for i in 0..n {
                        let chunk_id = format!("chunk_{}", i);
                        prefetcher.lock().record_access(&chunk_id);
                    }
                });
            },
        );

        // Lock-free
        #[cfg(feature = "lock-free")]
        group.bench_with_input(
            BenchmarkId::new("lockfree", num_accesses),
            &num_accesses,
            |b, &n| {
                let prefetcher = Arc::new(
                    LockFreePrefetcher::new()
                        .strategy(LFStrategy::Adaptive)
                        .history_window(100),
                );

                b.iter(|| {
                    for i in 0..n {
                        let chunk_id = format!("chunk_{}", i);
                        prefetcher.record_access(&chunk_id);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark statistics gathering
fn bench_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats");

    // Mutex-based
    group.bench_function("mutex", |b| {
        let prefetcher = Arc::new(Mutex::new(Prefetcher::new()));

        // Add some data
        for i in 0..100 {
            let tensor = DenseND::<f64>::zeros(&[2, 2]);
            prefetcher
                .lock()
                .add_prefetched(&format!("chunk_{}", i), tensor);
        }

        b.iter(|| {
            let _ = black_box(prefetcher.lock().stats());
        });
    });

    // Lock-free
    #[cfg(feature = "lock-free")]
    group.bench_function("lockfree", |b| {
        let prefetcher = Arc::new(LockFreePrefetcher::new());

        // Add some data
        for i in 0..100 {
            let tensor = DenseND::<f64>::zeros(&[2, 2]);
            prefetcher.add_prefetched(&format!("chunk_{}", i), tensor);
        }

        b.iter(|| {
            let _ = black_box(prefetcher.stats_snapshot());
        });
    });

    group.finish();
}

#[cfg(feature = "lock-free")]
criterion_group!(
    benches,
    bench_single_schedule,
    bench_concurrent_schedule,
    bench_add_get,
    bench_concurrent_add_get,
    bench_record_access,
    bench_stats
);

#[cfg(not(feature = "lock-free"))]
criterion_group!(
    benches,
    bench_single_schedule,
    bench_add_get,
    bench_record_access,
    bench_stats
);

criterion_main!(benches);
