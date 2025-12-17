//! Thread-Local Memory Pooling Example
//!
//! Demonstrates the use of thread-local memory pools for zero-contention
//! parallel execution and smart pooling heuristics for automatic optimization.
//!
//! Run with: `cargo run --example thread_local_pooling --release`

use std::thread;
use std::time::Instant;
use tenrso_exec::executor::pool_heuristics::{PoolingPolicy, PoolingRecommender};
use tenrso_exec::executor::thread_local_pool::ThreadLocalPoolManager;

fn main() {
    println!("=== Thread-Local Memory Pooling Demo ===\n");

    // Example 1: Basic thread-local pooling
    example_1_basic_usage();

    println!("\n{}\n", "=".repeat(60));

    // Example 2: Parallel scalability
    example_2_parallel_scalability();

    println!("\n{}\n", "=".repeat(60));

    // Example 3: Smart pooling heuristics
    example_3_smart_heuristics();

    println!("\n{}\n", "=".repeat(60));

    // Example 4: Pooling recommendations
    example_4_pooling_recommendations();
}

/// Example 1: Basic thread-local pool usage
fn example_1_basic_usage() {
    println!("Example 1: Basic Thread-Local Pool Usage");
    println!("{}", "-".repeat(50));

    let manager = ThreadLocalPoolManager::new();

    // First acquisition - will be a miss
    let buf1 = manager.acquire_f64(&[1000]);
    println!("Acquired buffer of 1000 elements");

    // Release it back to the pool
    manager.release_f64(&[1000], buf1);
    println!("Released buffer to pool");

    // Second acquisition - will be a hit
    let buf2 = manager.acquire_f64(&[1000]);
    println!("Acquired buffer again (should be from pool)");

    let stats = manager.thread_stats_f64();
    println!("\nThread Statistics:");
    println!("  Hits: {}", stats.hits);
    println!("  Misses: {}", stats.misses);
    println!("  Hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("  Buffers pooled: {}", stats.total_buffers_pooled);

    manager.release_f64(&[1000], buf2);
}

/// Example 2: Parallel scalability with zero contention
fn example_2_parallel_scalability() {
    println!("Example 2: Parallel Scalability");
    println!("{}", "-".repeat(50));

    let manager = ThreadLocalPoolManager::new();
    let num_threads = 8;
    let iterations = 1000;

    println!(
        "Spawning {} threads, {} iterations each",
        num_threads, iterations
    );

    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let mgr = manager.clone();
            thread::spawn(move || {
                for _ in 0..iterations {
                    // Each thread works with its own pool
                    let buf = mgr.acquire_f64(&[1000]);
                    // Simulate work
                    let _sum: f64 = buf.iter().sum();
                    mgr.release_f64(&[1000], buf);
                }
                (thread_id, mgr.thread_stats_f64())
            })
        })
        .collect();

    let thread_results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let duration = start.elapsed();

    println!("\nExecution time: {:.3}ms", duration.as_secs_f64() * 1000.0);
    println!(
        "Operations/sec: {:.0}",
        (num_threads * iterations) as f64 / duration.as_secs_f64()
    );

    // Calculate aggregate statistics
    let total_hits: usize = thread_results.iter().map(|(_, s)| s.hits).sum();
    let total_misses: usize = thread_results.iter().map(|(_, s)| s.misses).sum();
    let total_allocs: usize = thread_results
        .iter()
        .map(|(_, s)| s.total_allocations)
        .sum();

    println!("\nAggregate Statistics:");
    println!("  Total threads: {}", num_threads);
    println!("  Total allocations: {}", total_allocs);
    println!("  Total hits: {}", total_hits);
    println!("  Total misses: {}", total_misses);
    println!(
        "  Overall hit rate: {:.1}%",
        (total_hits as f64 / total_allocs as f64) * 100.0
    );

    println!("\nPer-Thread Statistics:");
    for (thread_id, stats) in thread_results {
        println!(
            "  Thread {}: {:.1}% hit rate",
            thread_id,
            stats.hit_rate * 100.0
        );
    }
}

/// Example 3: Smart pooling heuristics
fn example_3_smart_heuristics() {
    println!("Example 3: Smart Pooling Heuristics");
    println!("{}", "-".repeat(50));

    let policies = [
        ("Default", PoolingPolicy::default()),
        ("Conservative", PoolingPolicy::conservative()),
        ("Aggressive", PoolingPolicy::aggressive()),
        ("Memory Constrained", PoolingPolicy::memory_constrained()),
    ];

    let test_shapes = [
        (&[100][..], "Small (800B)"),
        (&[1000][..], "Medium (8KB)"),
        (&[100_000][..], "Large (800KB)"),
        (&[10_000_000][..], "Very Large (80MB)"),
    ];

    println!("\nPooling decisions for different policies:\n");
    println!(
        "{:<20} {:<15} {:<15} {:<15} {:<20}",
        "Shape", "Default", "Conservative", "Aggressive", "Memory Constrained"
    );
    println!("{}", "-".repeat(85));

    for (shape, label) in &test_shapes {
        print!("{:<20}", label);
        for (_, policy) in &policies {
            let should_pool = policy.should_pool(shape, 8); // f64 = 8 bytes
            let marker = if should_pool { "✓ Pool" } else { "✗ Skip" };
            print!(" {:<15}", marker);
        }
        println!();
    }

    println!("\nMemory Pressure Adaptation:");
    println!("{}", "-".repeat(50));

    let default_policy = PoolingPolicy::default();

    let scenarios = [
        (0.05, "Critical (5% free)"),
        (0.15, "Low (15% free)"),
        (0.5, "Normal (50% free)"),
        (0.9, "Abundant (90% free)"),
    ];

    for (memory_ratio, label) in &scenarios {
        let adjusted = default_policy.with_memory_pressure(*memory_ratio);
        println!("\n{}", label);
        println!("  Min size: {} bytes", adjusted.min_size_bytes);
        println!("  Max size: {} MB", adjusted.max_size_bytes / (1024 * 1024));
        println!("  Min frequency: {}", adjusted.min_frequency);
    }
}

/// Example 4: Pooling recommendations based on access patterns
fn example_4_pooling_recommendations() {
    println!("Example 4: Pooling Recommendations");
    println!("{}", "-".repeat(50));

    let mut recommender = PoolingRecommender::new();

    println!("\nSimulating workload with mixed buffer sizes...");

    // Simulate a workload
    for _ in 0..100 {
        recommender.record_allocation(&[10, 100]); // 1000 elements, frequently used
    }
    for _ in 0..50 {
        recommender.record_allocation(&[50, 50]); // 2500 elements, moderately used
    }
    for _ in 0..10 {
        recommender.record_allocation(&[100, 100]); // 10000 elements, rarely used
    }
    for _ in 0..5 {
        recommender.record_allocation(&[5, 5]); // 25 elements, too small
    }

    let report = recommender.generate_report(8); // f64 = 8 bytes

    println!("\n");
    report.print();

    println!("\n\nRecommendations:");
    if report.recommended_shapes_count > 0 {
        println!(
            "  Enable pooling for {} shapes",
            report.recommended_shapes_count
        );
        println!(
            "  Expected hit rate: {:.1}%",
            report.potential_hit_rate * 100.0
        );
        println!(
            "  This would eliminate {:.1}% of allocations",
            report.potential_hit_rate * 100.0
        );
    } else {
        println!("  No pooling recommended for this workload");
        println!("  Buffers are either too small or accessed infrequently");
    }

    // Compare different policies
    println!("\n\nPolicy Comparison:");
    println!("{}", "-".repeat(50));

    let policies = [
        ("Default", PoolingPolicy::default()),
        ("Conservative", PoolingPolicy::conservative()),
        ("Aggressive", PoolingPolicy::aggressive()),
    ];

    for (name, policy) in &policies {
        let mut rec = PoolingRecommender::with_policy(policy.clone());

        // Same workload
        for _ in 0..100 {
            rec.record_allocation(&[10, 100]);
        }
        for _ in 0..50 {
            rec.record_allocation(&[50, 50]);
        }
        for _ in 0..10 {
            rec.record_allocation(&[100, 100]);
        }
        for _ in 0..5 {
            rec.record_allocation(&[5, 5]);
        }

        let report = rec.generate_report(8);
        println!("\n{} Policy:", name);
        println!("  Shapes recommended: {}", report.recommended_shapes_count);
        println!(
            "  Potential hit rate: {:.1}%",
            report.potential_hit_rate * 100.0
        );
    }
}
