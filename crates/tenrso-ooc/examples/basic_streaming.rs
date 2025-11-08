//! Basic streaming execution example
//!
//! Demonstrates how to use the streaming executor for large matrix multiplication.

use anyhow::Result;
use tenrso_core::DenseND;
use tenrso_ooc::{StreamConfig, StreamingExecutor};

fn main() -> Result<()> {
    println!("=== Basic Streaming Execution Example ===\n");

    // Create streaming configuration with memory limit
    let config = StreamConfig::new()
        .max_memory_mb(256) // Limit to 256MB
        .chunk_size(vec![128]) // Process 128 rows at a time
        .enable_spill(true);

    let mut executor = StreamingExecutor::new(config);

    println!("Configuration:");
    println!("  Max memory: 256 MB");
    println!("  Chunk size: 128\n");

    // Create large matrices (larger than chunk size to demonstrate streaming)
    let m = 512;
    let n = 512;

    println!("Creating matrices:");
    println!("  A: {} x {}", m, n);
    println!("  B: {} x {}", n, n);

    let a = DenseND::<f64>::ones(&[m, n]);
    let b = DenseND::<f64>::ones(&[n, n]);

    println!("\nExecuting chunked matrix multiplication...");

    // Perform chunked matrix multiplication
    let result = executor.matmul_chunked(&a, &b, Some(128))?;

    println!("Result shape: {:?}", result.shape());
    println!("Current memory usage: {} bytes", executor.current_memory());

    // Verify result dimensions
    assert_eq!(result.shape(), &[m, n]);

    println!("\n✓ Streaming execution completed successfully!");

    Ok(())
}
