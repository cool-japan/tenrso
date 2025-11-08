//! Performance profiling example
//!
//! Demonstrates how to use the profiler to monitor out-of-core operations.

use anyhow::Result;
use std::thread;
use std::time::Duration;
use tenrso_core::DenseND;
use tenrso_ooc::{Profiler, StreamConfig, StreamingExecutor};

fn main() -> Result<()> {
    println!("=== Performance Profiling Example ===\n");

    let mut profiler = Profiler::new();

    // Simulate various operations
    println!("Running profiled operations...\n");

    // 1. Simulate I/O operation
    {
        let mut scope = profiler.begin_scope("load_tensor");
        thread::sleep(Duration::from_millis(50));
        scope.add_bytes_read(1024 * 1024); // 1 MB read
    }

    // 2. Simulate computation
    {
        let _scope = profiler.begin_scope("compute");
        thread::sleep(Duration::from_millis(100));
    }

    // 3. Multiple small operations
    for i in 0..10 {
        let _scope = profiler.begin_scope("small_op");
        thread::sleep(Duration::from_millis(5));

        if i == 0 {
            println!("  Running 10 small operations...");
        }
    }

    // 4. Real tensor operation
    {
        let _scope = profiler.begin_scope("matmul");

        let config = StreamConfig::new().max_memory_mb(256);
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::ones(&[128, 128]);
        let b = DenseND::<f64>::ones(&[128, 128]);

        let _result = executor.matmul_chunked(&a, &b, Some(64))?;
    }

    // Print detailed statistics
    println!("\n=== Operation Statistics ===\n");

    for (name, stats) in profiler.get_all_stats() {
        println!("Operation: {}", name);
        println!("  Calls: {}", stats.call_count);
        println!("  Total time: {:?}", stats.total_time);
        println!("  Avg time: {:?}", stats.avg_time);
        println!("  Min time: {:?}", stats.min_time);
        println!("  Max time: {:?}", stats.max_time);

        if stats.bytes_read > 0 || stats.bytes_written > 0 {
            println!("  Bytes read: {} KB", stats.bytes_read / 1024);
            println!("  Bytes written: {} KB", stats.bytes_written / 1024);
        }

        println!();
    }

    // Print summary
    let summary = profiler.summary();
    println!("=== Summary ===\n");
    println!("Total operations: {}", summary.total_operations);
    println!("Operation types: {}", summary.num_operation_types);
    println!("Total time: {:?}", summary.total_time);
    println!("Elapsed time: {:?}", summary.elapsed_time);
    println!("Total I/O: {} MB", summary.total_io_bytes / (1024 * 1024));
    println!("I/O bandwidth: {:.2} MB/s", summary.io_bandwidth_mbps());
    println!("Ops/sec: {:.2}", summary.ops_per_second());

    // Find slowest operation
    if let Some((name, stats)) = profiler.slowest_operation() {
        println!("\nSlowest operation: {} ({:?})", name, stats.total_time);
    }

    // Find most called operation
    if let Some((name, stats)) = profiler.most_called_operation() {
        println!("Most called: {} ({} times)", name, stats.call_count);
    }

    println!("\n✓ Profiling demonstration completed!");

    Ok(())
}
