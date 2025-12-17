//! Sparse Gradient Example
//!
//! Demonstrates efficient handling of sparse gradients where most components are zero.
//! Sparse gradients are common in large networks, embedding layers, and dropout.

use anyhow::Result;
use scirs2_core::ndarray_ext::Array;
use tenrso_ad::sparse_grad::{SparseGradient, SparseGradientBatch, SparsityConfig};

fn main() -> Result<()> {
    println!("=== Sparse Gradient Handling Example ===\n");

    // Example 1: Basic sparse gradient conversion
    println!("Example 1: Dense to Sparse Conversion");
    println!("-------------------------------------");

    // Create a mostly-zero gradient (typical for large networks)
    let mut values = vec![0.0; 1000];
    values[10] = 1.5;
    values[100] = 2.3;
    values[500] = -1.2;
    values[999] = 0.8;

    let dense = Array::from_vec(values).into_dyn();
    println!("Dense gradient size: {} elements", dense.len());

    let threshold = 0.1;
    let sparse = SparseGradient::from_dense(&dense, threshold)?;

    println!("Sparse representation:");
    println!("  Non-zero elements: {}", sparse.nnz());
    println!("  Total elements: {}", sparse.total_elements());
    println!("  Sparsity: {:.1}%", sparse.sparsity() * 100.0);
    println!(
        "  Memory savings: {} bytes",
        dense.len() * 8 - sparse.memory_bytes()
    );

    // Example 2: Automatic format selection
    println!("\nExample 2: Automatic Format Selection");
    println!("-------------------------------------");

    let config = SparsityConfig {
        zero_threshold: 1e-8,
        auto_sparse_threshold: 0.5, // Use sparse if >50% zeros
        ..Default::default()
    };

    // Dense gradient (40% zeros)
    let mut dense_grad = vec![1.0; 100];
    for i in (0..100).step_by(5) {
        dense_grad[i] = 0.0; // 20% zeros
    }
    for i in (1..100).step_by(5) {
        dense_grad[i] = 0.0; // Another 20% zeros
    }

    let dense = Array::from_vec(dense_grad).into_dyn();
    let auto1 = SparseGradient::from_dense_auto(&dense, &config)?;

    println!("Gradient with 40% zeros:");
    match auto1 {
        SparseGradient::Dense(_) => println!("  → Stored as DENSE (< 50% sparsity)"),
        SparseGradient::Sparse { .. } => println!("  → Stored as SPARSE"),
    }

    // Sparse gradient (90% zeros)
    let mut sparse_grad = vec![0.0; 100];
    for elem in sparse_grad.iter_mut().take(10) {
        *elem = 1.0;
    }

    let dense = Array::from_vec(sparse_grad).into_dyn();
    let auto2 = SparseGradient::from_dense_auto(&dense, &config)?;

    println!("Gradient with 90% zeros:");
    match auto2 {
        SparseGradient::Dense(_) => println!("  → Stored as DENSE"),
        SparseGradient::Sparse { .. } => println!("  → Stored as SPARSE (≥ 50% sparsity)"),
    }

    // Example 3: Gradient accumulation
    println!("\nExample 3: Sparse Gradient Accumulation");
    println!("---------------------------------------");

    let grad1_vec = vec![1.0, 0.0, 0.0, 2.0, 0.0];
    let grad2_vec = vec![0.0, 3.0, 0.0, 0.0, 4.0];

    let grad1 = SparseGradient::from_dense(&Array::from_vec(grad1_vec.clone()).into_dyn(), 0.5)?;

    let grad2 = SparseGradient::from_dense(&Array::from_vec(grad2_vec.clone()).into_dyn(), 0.5)?;

    println!("Gradient 1 (nnz=2): {:?}", grad1_vec);
    println!("Gradient 2 (nnz=2): {:?}", grad2_vec);

    let mut accumulated = grad1.clone();
    accumulated.accumulate(&grad2)?;

    let result = accumulated.to_dense()?;
    println!("Accumulated: {:?}", result.as_slice().unwrap());

    // Example 4: Batch gradient averaging
    println!("\nExample 4: Batch Gradient Processing");
    println!("------------------------------------");

    let config = SparsityConfig::default();
    let mut batch = SparseGradientBatch::new(config);

    // Simulate gradients from different samples in a batch
    for i in 0..8 {
        let mut grad_vec = vec![0.0; 100];

        // Each sample has gradients in different positions
        let base = i * 10;
        for j in 0..10 {
            if (base + j) < 100 {
                grad_vec[base + j] = (i + 1) as f64;
            }
        }

        let grad = SparseGradient::from_dense_auto(
            &Array::from_vec(grad_vec).into_dyn(),
            &SparsityConfig::default(),
        )?;

        batch.add(grad);
    }

    println!("Batch of {} gradients", 8);

    let stats = batch.statistics();
    println!("\nBatch Statistics:");
    println!("  Total gradients: {}", stats.total_gradients);
    println!("  Sparse format: {}", stats.sparse_gradients);
    println!("  Dense format: {}", stats.dense_gradients);
    println!("  Average sparsity: {:.1}%", stats.average_sparsity * 100.0);
    println!(
        "  Total memory: {:.2} KB",
        stats.total_memory_bytes as f64 / 1024.0
    );

    let avg_grad = batch.average()?;
    println!("\nAveraged gradient:");
    println!("  NNZ: {}", avg_grad.nnz());
    println!("  Sparsity: {:.1}%", avg_grad.sparsity() * 100.0);

    // Example 5: Gradient compression
    println!("\nExample 5: Gradient Compression");
    println!("-------------------------------");

    // Full gradient
    let full_grad: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.001).collect();
    let mut grad = SparseGradient::Dense(Array::from_vec(full_grad).into_dyn());

    println!("Original gradient:");
    println!("  Elements: {}", grad.total_elements());
    println!("  Memory: {} bytes", grad.memory_bytes());

    // Compress to 90% sparsity (keep only top 10% values)
    let compress_config = SparsityConfig {
        target_sparsity: 0.9,
        ..Default::default()
    };

    grad.compress(&compress_config)?;

    println!("\nAfter compression to 90% sparsity:");
    println!("  Non-zeros: {}", grad.nnz());
    println!("  Memory: {} bytes", grad.memory_bytes());
    println!(
        "  Memory reduction: {:.1}%",
        (1.0 - grad.memory_bytes() as f64 / (1000 * 8) as f64) * 100.0
    );

    // Example 6: Scaling sparse gradients
    println!("\nExample 6: Gradient Scaling");
    println!("---------------------------");

    let sparse_vec = vec![0.0, 2.0, 0.0, 4.0, 0.0];
    let mut grad = SparseGradient::from_dense(&Array::from_vec(sparse_vec).into_dyn(), 0.5)?;

    println!(
        "Original sparse gradient (nnz={}): {:?}",
        grad.nnz(),
        grad.to_dense()?.as_slice().unwrap()
    );

    grad.scale(0.5)?;

    println!(
        "After scaling by 0.5: {:?}",
        grad.to_dense()?.as_slice().unwrap()
    );

    // Example 7: Memory comparison at scale
    println!("\nExample 7: Large-Scale Memory Comparison");
    println!("----------------------------------------");

    let sizes = vec![10_000, 100_000, 1_000_000];
    let sparsities = vec![0.9, 0.95, 0.99];

    for &size in &sizes {
        println!("\nGradient size: {} parameters", size);

        for &sparsity in &sparsities {
            let nnz = ((1.0 - sparsity) * size as f64) as usize;

            // Dense memory
            let dense_memory = size * 8; // f64 = 8 bytes

            // Sparse memory (indices + values)
            let sparse_memory = nnz * (8 + 8); // usize + f64

            let savings = dense_memory - sparse_memory;
            let savings_ratio = savings as f64 / dense_memory as f64;

            println!(
                "  Sparsity {:.0}% (nnz={}): {:.2} MB → {:.2} MB (saves {:.1}%)",
                sparsity * 100.0,
                nnz,
                dense_memory as f64 / 1_048_576.0,
                sparse_memory as f64 / 1_048_576.0,
                savings_ratio * 100.0
            );
        }
    }

    println!("\n=== Sparse Gradient Handling Complete ===");

    Ok(())
}
