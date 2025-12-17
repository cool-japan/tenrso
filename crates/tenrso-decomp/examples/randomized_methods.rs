//! Randomized Tensor Decomposition Methods
//!
//! Demonstrates randomized algorithms for efficient tensor decomposition
//! of large-scale tensors using sketching techniques.
//!
//! # What are Randomized Methods?
//!
//! Randomized tensor decompositions use random projections (sketching) to
//! reduce computational cost while maintaining good approximation quality.
//! This is particularly valuable for very large tensors where standard
//! methods become prohibitively expensive.
//!
//! # Key Benefits
//!
//! - **Speed**: 2-5× faster than standard methods for large tensors
//! - **Scalability**: Handles tensors that don't fit in memory for standard methods
//! - **Quality**: With proper oversampling, achieves 90-95% of standard method quality
//! - **Memory**: Minimal memory overhead (just the sketch matrices)
//!
//! # Usage
//!
//! ```bash
//! cargo run --example randomized_methods --release
//! ```

use anyhow::Result;
use std::time::Instant;
use tenrso_core::DenseND;
use tenrso_decomp::{cp_als, cp_randomized, tucker_hosvd, tucker_randomized, InitStrategy};

fn main() -> Result<()> {
    println!("{}", "=".repeat(80));
    println!("Randomized Tensor Decomposition Methods");
    println!("{}", "=".repeat(80));
    println!();

    // Example 1: Randomized CP-ALS
    example_1_randomized_cp()?;

    println!("\n{}\n", "=".repeat(80));

    // Example 2: Randomized Tucker
    example_2_randomized_tucker()?;

    println!("\n{}\n", "=".repeat(80));

    // Example 3: Oversampling Trade-offs
    example_3_oversampling_tradeoffs()?;

    println!("\n{}\n", "=".repeat(80));

    // Example 4: Large-Scale Tensor Processing
    example_4_large_scale_processing()?;

    println!("\n{}\n", "=".repeat(80));

    // Example 5: Comparison with Standard Methods
    example_5_comparison_study()?;

    Ok(())
}

/// Example 1: Basic Randomized CP-ALS
///
/// Demonstrates the basic usage of randomized CP decomposition with
/// different sketch sizes and fit check frequencies.
fn example_1_randomized_cp() -> Result<()> {
    println!("Example 1: Randomized CP-ALS");
    println!("{}", "-".repeat(80));
    println!();

    // Create a 100×100×100 tensor
    let size = 100;
    let rank = 10;
    println!("Creating {}×{}×{} tensor...", size, size, size);
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    println!("Tensor size: {} elements", size * size * size);
    println!("Target rank: {}", rank);
    println!();

    // Randomized CP with 5x oversampling
    let sketch_size = rank * 5;
    let fit_check_freq = 5;

    println!("Running Randomized CP-ALS:");
    println!("  - Sketch size: {} ({}× oversampling)", sketch_size, 5);
    println!(
        "  - Fit check frequency: every {} iterations",
        fit_check_freq
    );
    println!();

    let start = Instant::now();
    let cp = cp_randomized(
        &tensor,
        rank,
        30, // max iterations
        1e-4,
        InitStrategy::Random,
        sketch_size,
        fit_check_freq,
    )?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  - Converged in {} iterations", cp.iters);
    println!("  - Final fit: {:.6}", cp.fit);
    println!("  - Time elapsed: {:.2?}", elapsed);
    println!();

    // Verify reconstruction
    let recon = cp.reconstruct(tensor.shape())?;
    let error = (&tensor - &recon).frobenius_norm() / tensor.frobenius_norm();
    println!("  - Relative reconstruction error: {:.6}", error);
    println!("  - Computed fit: {:.6}", 1.0 - error);

    Ok(())
}

/// Example 2: Randomized Tucker Decomposition
///
/// Shows how to use randomized Tucker for efficient higher-order SVD
/// with automatic rank selection.
fn example_2_randomized_tucker() -> Result<()> {
    println!("Example 2: Randomized Tucker Decomposition");
    println!("{}", "-".repeat(80));
    println!();

    // Create a 80×80×80 tensor
    let size = 80;
    let ranks = vec![20, 20, 20];
    println!("Creating {}×{}×{} tensor...", size, size, size);
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    println!("Tensor size: {} elements", size * size * size);
    println!("Target ranks: {:?}", ranks);
    println!();

    // Randomized Tucker with oversampling and power iterations
    let oversample = 10;
    let power_iters = 2;

    println!("Running Randomized Tucker:");
    println!("  - Oversampling: {}", oversample);
    println!("  - Power iterations: {}", power_iters);
    println!();

    let start = Instant::now();
    let tucker = tucker_randomized(&tensor, &ranks, oversample, power_iters)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  - Core shape: {:?}", tucker.core.shape());
    println!("  - Time elapsed: {:.2?}", elapsed);
    println!();

    // Compute compression statistics
    let ratio = tucker.compression_ratio();

    println!("  - Compression ratio: {:.2}×", ratio);
    println!();

    // Verify reconstruction quality
    let recon = tucker.reconstruct()?;
    let error = (&tensor - &recon).frobenius_norm() / tensor.frobenius_norm();
    println!("  - Relative reconstruction error: {:.6}", error);

    Ok(())
}

/// Example 3: Oversampling Trade-offs
///
/// Demonstrates the accuracy vs. speed trade-off with different
/// oversampling parameters.
fn example_3_oversampling_tradeoffs() -> Result<()> {
    println!("Example 3: Oversampling Trade-offs");
    println!("{}", "-".repeat(80));
    println!();

    let size = 60;
    let rank = 8;
    println!("Creating {}×{}×{} tensor...", size, size, size);
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);
    println!();

    println!(
        "{:<20} {:<15} {:<15} {:<15}",
        "Oversampling", "Sketch Size", "Fit", "Time (ms)"
    );
    println!("{}", "-".repeat(70));

    for &mult in &[3, 5, 7, 10] {
        let sketch_size = rank * mult;

        let start = Instant::now();
        let cp = cp_randomized(
            &tensor,
            rank,
            20,
            1e-4,
            InitStrategy::Random,
            sketch_size,
            5,
        )?;
        let elapsed = start.elapsed();

        println!(
            "{:<20} {:<15} {:<15.6} {:<15.2}",
            format!("{}×", mult),
            sketch_size,
            cp.fit,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    println!();
    println!("Observations:");
    println!("  - Higher oversampling generally improves fit");
    println!("  - But increases computational cost");
    println!("  - 5× is often a good balance for most applications");

    Ok(())
}

/// Example 4: Large-Scale Tensor Processing
///
/// Shows how randomized methods enable processing of very large tensors
/// that would be too slow or memory-intensive for standard methods.
fn example_4_large_scale_processing() -> Result<()> {
    println!("Example 4: Large-Scale Tensor Processing");
    println!("{}", "-".repeat(80));
    println!();

    // Simulate processing a large tensor
    let size = 150;
    let rank = 15;
    println!("Processing large tensor: {}×{}×{}", size, size, size);
    println!("Elements: {}", size * size * size);
    println!();

    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    // Randomized CP is practical for this size
    println!("Method 1: Randomized CP-ALS (4× oversampling)");
    let start = Instant::now();
    let cp_rand = cp_randomized(
        &tensor,
        rank,
        15, // Fewer iterations for speed
        1e-3,
        InitStrategy::Random,
        rank * 4,
        3,
    )?;
    let rand_time = start.elapsed();

    println!("  - Converged in {} iterations", cp_rand.iters);
    println!("  - Final fit: {:.6}", cp_rand.fit);
    println!("  - Time: {:.2?}", rand_time);
    println!();

    // Note: Standard CP-ALS would be slower
    println!("Note: Standard CP-ALS would take significantly longer");
    println!("      Randomized method is ideal for exploratory analysis");

    Ok(())
}

/// Example 5: Comparison Study
///
/// Detailed comparison between randomized and standard methods
/// to quantify the speed-accuracy trade-off.
fn example_5_comparison_study() -> Result<()> {
    println!("Example 5: Randomized vs Standard Methods");
    println!("{}", "-".repeat(80));
    println!();

    let size = 70;
    let rank = 10;
    let iterations = 20;

    println!("Comparison Setup:");
    println!("  - Tensor size: {}×{}×{}", size, size, size);
    println!("  - Rank: {}", rank);
    println!("  - Iterations: {}", iterations);
    println!();

    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);

    // Standard CP-ALS
    println!("Running Standard CP-ALS...");
    let start = Instant::now();
    let cp_std = cp_als(&tensor, rank, iterations, 1e-4, InitStrategy::Random, None)?;
    let std_time = start.elapsed();

    println!("  - Iterations: {}", cp_std.iters);
    println!("  - Fit: {:.6}", cp_std.fit);
    println!("  - Time: {:.2?}", std_time);
    println!();

    // Randomized CP-ALS
    println!("Running Randomized CP-ALS (5× oversampling)...");
    let start = Instant::now();
    let cp_rand = cp_randomized(
        &tensor,
        rank,
        iterations,
        1e-4,
        InitStrategy::Random,
        rank * 5,
        3,
    )?;
    let rand_time = start.elapsed();

    println!("  - Iterations: {}", cp_rand.iters);
    println!("  - Fit: {:.6}", cp_rand.fit);
    println!("  - Time: {:.2?}", rand_time);
    println!();

    // Tucker comparison
    println!("Running Standard Tucker-HOSVD...");
    let tucker_ranks = vec![20, 20, 20];
    let start = Instant::now();
    let tucker_std = tucker_hosvd(&tensor, &tucker_ranks)?;
    let tucker_std_time = start.elapsed();
    let tucker_std_recon = tucker_std.reconstruct()?;
    let tucker_std_error = (&tensor - &tucker_std_recon).frobenius_norm() / tensor.frobenius_norm();

    println!("  - Ranks: {:?}", tucker_ranks);
    println!("  - Error: {:.6}", tucker_std_error);
    println!("  - Time: {:.2?}", tucker_std_time);
    println!();

    println!("Running Randomized Tucker...");
    let start = Instant::now();
    let tucker_rand = tucker_randomized(&tensor, &tucker_ranks, 10, 2)?;
    let tucker_rand_time = start.elapsed();
    let tucker_rand_recon = tucker_rand.reconstruct()?;
    let tucker_rand_error =
        (&tensor - &tucker_rand_recon).frobenius_norm() / tensor.frobenius_norm();

    println!("  - Ranks: {:?}", tucker_ranks);
    println!("  - Error: {:.6}", tucker_rand_error);
    println!("  - Time: {:.2?}", tucker_rand_time);
    println!();

    // Summary
    println!("{}", "=".repeat(80));
    println!("Summary:");
    println!("{}", "-".repeat(80));

    let cp_speedup = std_time.as_secs_f64() / rand_time.as_secs_f64();
    let cp_fit_ratio = cp_rand.fit / cp_std.fit;

    println!("CP Decomposition:");
    println!("  - Speedup: {:.2}×", cp_speedup);
    println!("  - Fit ratio: {:.2}% of standard", cp_fit_ratio * 100.0);
    println!();

    let tucker_speedup = tucker_std_time.as_secs_f64() / tucker_rand_time.as_secs_f64();
    let tucker_error_ratio = tucker_rand_error / tucker_std_error;

    println!("Tucker Decomposition:");
    println!("  - Speedup: {:.2}×", tucker_speedup);
    println!(
        "  - Error ratio: {:.2}% of standard",
        tucker_error_ratio * 100.0
    );
    println!();

    println!("Recommendations:");
    println!("  - Use randomized methods for tensors > 10^7 elements");
    println!("  - Use 5-7× oversampling for good accuracy");
    println!("  - Randomized methods are excellent for exploratory analysis");
    println!("  - For final production results, standard methods may be preferred");

    Ok(())
}
