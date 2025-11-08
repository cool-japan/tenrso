//! CP-ALS (Canonical Polyadic via Alternating Least Squares) Example
//!
//! This example demonstrates how to use CP-ALS decomposition to factorize
//! a tensor into a sum of rank-1 components.
//!
//! Run with:
//! ```bash
//! cargo run --example cp_als
//! ```

use tenrso_core::DenseND;
use tenrso_decomp::{cp_als, InitStrategy};

fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(80));
    println!("CP-ALS Decomposition Example");
    println!("{}", "=".repeat(80));
    println!();

    // ========================================================================
    // Example 1: Basic CP-ALS on a synthetic 3D tensor
    // ========================================================================
    println!("Example 1: Basic CP-ALS decomposition");
    println!("{}", "-".repeat(80));

    let shape = vec![50, 40, 30];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let rank = 10;
    let max_iters = 50;
    let tolerance = 1e-4;

    println!("Tensor shape: {:?}", shape);
    println!("Target rank: {}", rank);
    println!("Max iterations: {}", max_iters);
    println!("Tolerance: {}", tolerance);
    println!();

    let cp = cp_als(&tensor, rank, max_iters, tolerance, InitStrategy::Random)?;

    println!("Results:");
    println!("  - Iterations: {}", cp.iters);
    println!("  - Final fit: {:.6}", cp.fit);
    println!("  - Factor matrix shapes:");
    for (i, factor) in cp.factors.iter().enumerate() {
        println!("    Mode {}: {:?}", i, factor.shape());
    }
    println!();

    // Verify reconstruction
    let reconstructed = cp.reconstruct(&shape)?;
    let original_norm = tensor.frobenius_norm();
    let diff = &tensor - &reconstructed;
    let error = diff.frobenius_norm();
    let relative_error = error / original_norm;

    println!("Reconstruction:");
    println!("  - Original norm: {:.6}", original_norm);
    println!("  - Reconstruction error: {:.6}", error);
    println!("  - Relative error: {:.6}", relative_error);
    println!("  - Fit (1 - relative_error): {:.6}", 1.0 - relative_error);
    println!();

    // ========================================================================
    // Example 2: Compare different initialization strategies
    // ========================================================================
    println!("Example 2: Comparing initialization strategies");
    println!("{}", "-".repeat(80));

    let small_shape = vec![20, 20, 20];
    let small_tensor = DenseND::<f64>::random_uniform(&small_shape, 0.0, 1.0);
    let small_rank = 5;

    let strategies = vec![
        ("Random Uniform", InitStrategy::Random),
        ("Random Normal", InitStrategy::RandomNormal),
        ("SVD-based (HOSVD)", InitStrategy::Svd),
    ];

    for (name, strategy) in strategies {
        print!("  Testing {}... ", name);
        let start = std::time::Instant::now();
        let result = cp_als(&small_tensor, small_rank, 30, 1e-4, strategy)?;
        let elapsed = start.elapsed();

        println!(
            "fit={:.4}, iters={}, time={:.2}ms",
            result.fit,
            result.iters,
            elapsed.as_secs_f64() * 1000.0
        );
    }
    println!();

    // ========================================================================
    // Example 3: Low-rank approximation for compression
    // ========================================================================
    println!("Example 3: Low-rank approximation for compression");
    println!("{}", "-".repeat(80));

    let data_shape = vec![100, 100, 100];
    let data_tensor = DenseND::<f64>::random_uniform(&data_shape, 0.0, 1.0);
    let original_elements: usize = data_shape.iter().product();

    println!("Original tensor: {:?}", data_shape);
    println!("Original elements: {}", original_elements);
    println!();

    for &test_rank in &[10, 20, 40, 80] {
        let cp = cp_als(&data_tensor, test_rank, 50, 1e-5, InitStrategy::Random)?;

        // Calculate compressed size
        let compressed_elements: usize = data_shape.iter().map(|&dim| dim * test_rank).sum();
        let compression_ratio = original_elements as f64 / compressed_elements as f64;

        println!(
            "Rank {}: fit={:.4}, compression={:.2}x, elements={}",
            test_rank, cp.fit, compression_ratio, compressed_elements
        );
    }
    println!();

    // ========================================================================
    // Example 4: Analyzing factor sparsity and weights
    // ========================================================================
    println!("Example 4: Analyzing CP decomposition structure");
    println!("{}", "-".repeat(80));

    let analysis_tensor = DenseND::<f64>::random_uniform(&[30, 30, 30], 0.0, 1.0);
    let cp = cp_als(&analysis_tensor, 15, 50, 1e-4, InitStrategy::Svd)?;

    println!("Factor matrix statistics:");
    for (i, factor) in cp.factors.iter().enumerate() {
        let mean = factor.iter().sum::<f64>() / factor.len() as f64;
        let std =
            (factor.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / factor.len() as f64).sqrt();
        let min = factor.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = factor.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!(
            "  Mode {}: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            i, mean, std, min, max
        );
    }
    println!();

    if let Some(weights) = &cp.weights {
        println!("Component weights:");
        println!("  {:?}", weights);
        let total_weight: f64 = weights.iter().sum();
        println!("  Total weight: {:.6}", total_weight);
        println!("  Top 5 components:");
        let mut indexed_weights: Vec<_> = weights.iter().enumerate().collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (i, &weight) in indexed_weights.iter().take(5) {
            println!(
                "    Component {}: {:.6} ({:.2}%)",
                i,
                weight,
                weight / total_weight * 100.0
            );
        }
    }
    println!();

    // ========================================================================
    // Example 5: Convergence behavior
    // ========================================================================
    println!("Example 5: Convergence behavior with varying iterations");
    println!("{}", "-".repeat(80));

    let conv_tensor = DenseND::<f64>::random_uniform(&[40, 40, 40], 0.0, 1.0);
    let conv_rank = 10;

    println!("Iterations | Fit    | Time (ms)");
    println!("{}", "-".repeat(40));

    for &iters in &[1, 5, 10, 20, 50, 100] {
        let start = std::time::Instant::now();
        let cp = cp_als(&conv_tensor, conv_rank, iters, 1e-10, InitStrategy::Random)?;
        let elapsed = start.elapsed();

        println!(
            "{:10} | {:.4} | {:.2}",
            cp.iters,
            cp.fit,
            elapsed.as_secs_f64() * 1000.0
        );
    }
    println!();

    // ========================================================================
    // Example 6: Non-cubic tensors
    // ========================================================================
    println!("Example 6: Non-cubic tensor decomposition");
    println!("{}", "-".repeat(80));

    let asymmetric_shapes = vec![vec![100, 50, 25], vec![80, 80, 20], vec![60, 40, 20]];

    for shape in asymmetric_shapes {
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let rank = 10;

        let start = std::time::Instant::now();
        let cp = cp_als(&tensor, rank, 30, 1e-4, InitStrategy::Random)?;
        let elapsed = start.elapsed();

        let original_elements: usize = shape.iter().product();
        let compressed_elements: usize = shape.iter().map(|&dim| dim * rank).sum();
        let compression_ratio = original_elements as f64 / compressed_elements as f64;

        println!(
            "Shape {:?}: fit={:.4}, compression={:.2}x, time={:.2}ms",
            shape,
            cp.fit,
            compression_ratio,
            elapsed.as_secs_f64() * 1000.0
        );
    }
    println!();

    println!("{}", "=".repeat(80));
    println!("CP-ALS Example Complete!");
    println!("{}", "=".repeat(80));

    Ok(())
}
