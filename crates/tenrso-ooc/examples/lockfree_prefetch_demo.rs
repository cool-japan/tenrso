//! Lock-free prefetcher demonstration
//!
//! This example demonstrates the high-performance lock-free prefetcher
//! with concurrent access from multiple threads.
//!
//! Run with: cargo run --example lockfree_prefetch_demo --features lock-free

#[cfg(feature = "lock-free")]
fn main() -> anyhow::Result<()> {
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;
    use tenrso_core::DenseND;
    use tenrso_ooc::lockfree_prefetch::{LockFreePrefetcher, PrefetchStrategy};

    println!("=== Lock-Free Prefetcher Demonstration ===\n");

    // Create a lock-free prefetcher
    let prefetcher = Arc::new(
        LockFreePrefetcher::new()
            .strategy(PrefetchStrategy::Sequential)
            .queue_size(100)
            .num_threads(4)
            .history_window(20),
    );

    println!("Created lock-free prefetcher with:");
    println!("  - Strategy: Sequential");
    println!("  - Queue size: 100");
    println!("  - Threads: 4");
    println!("  - History window: 20\n");

    // === Example 1: Concurrent Scheduling ===
    println!("Example 1: Concurrent Scheduling from Multiple Threads");
    println!("--------------------------------------------------------");

    let start = Instant::now();
    let num_threads = 4;
    let chunks_per_thread = 25;

    let mut handles = vec![];
    for thread_id in 0..num_threads {
        let pf = prefetcher.clone();
        let handle = thread::spawn(move || {
            let chunks: Vec<String> = (0..chunks_per_thread)
                .map(|i| format!("chunk_{}_{}", thread_id, i))
                .collect();
            let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();

            pf.schedule_prefetch(chunk_refs)
                .expect("Failed to schedule");

            println!(
                "  Thread {} scheduled {} chunks",
                thread_id, chunks_per_thread
            );
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    println!("  Total scheduled: {} chunks", prefetcher.queue_len());
    println!("  Time: {:?}", elapsed);
    println!();

    // === Example 2: Concurrent Add/Get Operations ===
    println!("Example 2: Concurrent Add and Get Operations");
    println!("----------------------------------------------");

    // Clear previous data
    prefetcher.clear();

    // Writer threads
    let start = Instant::now();
    let mut handles = vec![];
    for thread_id in 0..num_threads {
        let pf = prefetcher.clone();
        let handle = thread::spawn(move || {
            for i in 0..chunks_per_thread {
                let data: Vec<f64> = (0..16)
                    .map(|x| (thread_id * 100 + i * 10 + x) as f64)
                    .collect();
                let tensor =
                    DenseND::<f64>::from_vec(data, &[4, 4]).expect("Failed to create tensor");

                let chunk_id = format!("chunk_{}_{}", thread_id, i);
                pf.add_prefetched(&chunk_id, tensor);
            }
            println!("  Writer thread {} completed", thread_id);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let write_time = start.elapsed();
    println!("  Write time: {:?}", write_time);
    println!("  Prefetched chunks: {}", prefetcher.prefetch_count());

    // Reader threads
    let start = Instant::now();
    let mut handles = vec![];
    for thread_id in 0..num_threads {
        let pf = prefetcher.clone();
        let handle = thread::spawn(move || {
            let mut retrieved = 0;
            for i in 0..chunks_per_thread {
                let chunk_id = format!("chunk_{}_{}", thread_id, i);
                if let Some(tensor) = pf.get(&chunk_id) {
                    // Verify tensor shape
                    assert_eq!(tensor.shape(), &[4, 4]);
                    retrieved += 1;
                }
            }
            println!(
                "  Reader thread {} retrieved {} chunks",
                thread_id, retrieved
            );
            retrieved
        });
        handles.push(handle);
    }

    let total_retrieved: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    let read_time = start.elapsed();
    println!("  Read time: {:?}", read_time);
    println!("  Total retrieved: {}", total_retrieved);
    println!();

    // === Example 3: Adaptive Prefetching ===
    println!("Example 3: Adaptive Prefetching with Access Pattern");
    println!("----------------------------------------------------");

    // Create new prefetcher with adaptive strategy
    let adaptive_pf = Arc::new(
        LockFreePrefetcher::new()
            .strategy(PrefetchStrategy::Adaptive)
            .queue_size(10)
            .history_window(10),
    );

    // Simulate sequential access pattern
    println!("  Simulating sequential access pattern...");
    for i in 0..5 {
        adaptive_pf.record_access(&format!("chunk_{}", i));
    }

    // Adaptive strategy should predict chunk_5, chunk_6, etc.
    let stats = adaptive_pf.stats_snapshot();
    println!(
        "  Queue length after adaptive prediction: {}",
        stats.queue_len
    );
    println!("  (Predicted next chunks based on sequential pattern)");
    println!();

    // === Example 4: Statistics and Performance Monitoring ===
    println!("Example 4: Statistics and Performance Monitoring");
    println!("-------------------------------------------------");

    let stats = prefetcher.stats_snapshot();
    println!("  Strategy: {:?}", stats.strategy);
    println!("  Queue size: {}", stats.queue_size);
    println!("  Current queue length: {}", stats.queue_len);
    println!("  Prefetched count: {}", stats.prefetched_count);
    println!("  Access history length: {}", stats.access_history_len);
    println!("  Enabled: {}", stats.enabled);
    println!();

    println!("  Performance Statistics:");
    println!("    Total scheduled: {}", stats.performance.scheduled_total);
    println!(
        "    Total prefetched: {}",
        stats.performance.prefetched_total
    );
    println!("    Cache hits: {}", stats.performance.cache_hits);
    println!("    Cache misses: {}", stats.performance.cache_misses);
    println!("    Hit rate: {:.2}%", stats.performance.hit_rate() * 100.0);
    println!("    Evictions: {}", stats.performance.evictions);
    println!();

    // === Example 5: Mixed Read/Write Workload ===
    println!("Example 5: Mixed Read/Write Workload");
    println!("-------------------------------------");

    prefetcher.clear();

    let start = Instant::now();
    let mut handles = vec![];

    // Spawn mixed reader/writer threads
    for thread_id in 0..num_threads {
        let pf = prefetcher.clone();
        let handle = thread::spawn(move || {
            let mut reads = 0;
            let mut writes = 0;

            for i in 0..chunks_per_thread {
                // Write every other chunk
                if i % 2 == 0 {
                    let tensor =
                        DenseND::<f64>::from_vec(vec![thread_id as f64; 4], &[2, 2]).unwrap();
                    let chunk_id = format!("mixed_{}_{}", thread_id, i);
                    pf.add_prefetched(&chunk_id, tensor);
                    writes += 1;
                } else {
                    // Try to read
                    let chunk_id = format!("mixed_{}_{}", thread_id, i - 1);
                    if pf.get(&chunk_id).is_some() {
                        reads += 1;
                    }
                }
            }

            (reads, writes)
        });
        handles.push(handle);
    }

    let results: Vec<(usize, usize)> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let elapsed = start.elapsed();
    let total_reads: usize = results.iter().map(|(r, _)| r).sum();
    let total_writes: usize = results.iter().map(|(_, w)| w).sum();

    println!("  Total reads: {}", total_reads);
    println!("  Total writes: {}", total_writes);
    println!("  Time: {:?}", elapsed);
    println!(
        "  Throughput: {:.2} ops/sec",
        (total_reads + total_writes) as f64 / elapsed.as_secs_f64()
    );
    println!();

    // === Summary ===
    println!("=== Summary ===");
    println!("The lock-free prefetcher provides:");
    println!("  ✓ Thread-safe concurrent access without blocking");
    println!("  ✓ Lock-free scheduling, add, and get operations");
    println!("  ✓ Wait-free statistics updates");
    println!("  ✓ Scalable performance with multiple threads");
    println!("  ✓ Adaptive prefetching based on access patterns");
    println!();

    println!("Benefits over mutex-based approach:");
    println!("  - No lock contention in multi-threaded scenarios");
    println!("  - Better scalability with increasing thread count");
    println!("  - Lower latency for individual operations");
    println!("  - No risk of priority inversion or deadlocks");
    println!();

    println!("Run benchmarks to see performance comparison:");
    println!("  cargo bench --bench lockfree_prefetch_benchmarks");

    Ok(())
}

#[cfg(not(feature = "lock-free"))]
fn main() {
    println!("This example requires the 'lock-free' feature.");
    println!("Run with: cargo run --example lockfree_prefetch_demo --features lock-free");
}
