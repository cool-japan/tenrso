//! Prefetch strategies example
//!
//! Demonstrates different prefetch strategies for improved performance.

use anyhow::Result;
use tenrso_core::DenseND;
use tenrso_ooc::{PrefetchStrategy, Prefetcher};

fn main() -> Result<()> {
    println!("=== Prefetch Strategies Example ===\n");

    // Test Sequential strategy
    println!("1. Sequential Prefetch Strategy");
    println!("   Prefetches chunks in sequential order\n");

    let mut prefetcher = Prefetcher::new()
        .strategy(PrefetchStrategy::Sequential)
        .queue_size(5);

    // Schedule chunks for prefetch
    let chunks = vec!["chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_4"];
    let scheduled = prefetcher.schedule_prefetch(chunks)?;

    println!("  Scheduled {} chunks for prefetch", scheduled);

    // Simulate adding prefetched data
    for i in 0..3 {
        let chunk_id = format!("chunk_{}", i);
        let tensor = DenseND::<f64>::ones(&[100, 100]);
        prefetcher.add_prefetched(&chunk_id, tensor);
    }

    let stats = prefetcher.stats();
    println!("  Prefetched: {}", stats.prefetched_count);
    println!("  Queue length: {}", stats.queue_len);

    // Test Adaptive strategy
    println!("\n2. Adaptive Prefetch Strategy");
    println!("   Learns access patterns and predicts next chunks\n");

    let mut adaptive_prefetcher = Prefetcher::new()
        .strategy(PrefetchStrategy::Adaptive)
        .queue_size(5)
        .history_window(10);

    // Simulate sequential access pattern
    println!("  Simulating sequential access pattern:");
    for i in 0..5 {
        let chunk_id = format!("chunk_{}", i);
        println!("    Accessing {}", chunk_id);
        adaptive_prefetcher.record_access(&chunk_id);
    }

    let stats = adaptive_prefetcher.stats();
    println!("\n  Access history: {} entries", stats.access_history_len);
    println!("  Predictions queued: {}", stats.queue_len);

    // Retrieve prefetched chunk
    println!("\n3. Retrieving Prefetched Chunks");

    let tensor = DenseND::<f64>::zeros(&[50, 50]);
    prefetcher.add_prefetched("test_chunk", tensor);

    if let Some(retrieved) = prefetcher.get("test_chunk") {
        println!("  Retrieved chunk: {:?}", retrieved.shape());
        println!(
            "  Prefetch cache now has {} chunks",
            prefetcher.prefetch_count()
        );
    }

    println!("\n✓ Prefetch strategies demonstration completed!");

    Ok(())
}
