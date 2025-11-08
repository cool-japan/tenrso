//! Tensor Reconstruction Example
//!
//! This example demonstrates how to reconstruct tensors from various
//! decomposition formats (CP, Tucker, TT) and analyze reconstruction quality.
//!
//! Run with:
//! ```bash
//! cargo run --example reconstruction
//! ```

use tenrso_core::DenseND;
use tenrso_decomp::{cp_als, tt_svd, tucker_hooi, tucker_hosvd, InitStrategy};

fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(80));
    println!("Tensor Reconstruction Example");
    println!("{}", "=".repeat(80));
    println!();

    // ========================================================================
    // Example 1: Basic reconstruction from CP decomposition
    // ========================================================================
    println!("Example 1: CP decomposition reconstruction");
    println!("{}", "-".repeat(80));

    let shape = vec![40, 40, 40];
    let original = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let rank = 15;

    println!("Original tensor shape: {:?}", shape);
    println!("CP rank: {}", rank);
    println!();

    // Perform CP-ALS
    let cp = cp_als(&original, rank, 50, 1e-4, InitStrategy::Random)?;
    println!(
        "CP-ALS converged in {} iterations (fit={:.4})",
        cp.iters, cp.fit
    );

    // Reconstruct
    let reconstructed = cp.reconstruct(&shape)?;
    println!("Reconstructed tensor shape: {:?}", reconstructed.shape());

    // Analyze error
    let diff = &original - &reconstructed;
    let original_norm = original.frobenius_norm();
    let error = diff.frobenius_norm();
    let relative_error = error / original_norm;

    println!("Reconstruction quality:");
    println!("  - Original norm: {:.6}", original_norm);
    println!("  - Error norm: {:.6}", error);
    println!("  - Relative error: {:.6}", relative_error);
    println!("  - Fit (1 - rel_error): {:.6}", 1.0 - relative_error);
    println!();

    // ========================================================================
    // Example 2: Tucker reconstruction (HOSVD)
    // ========================================================================
    println!("Example 2: Tucker-HOSVD reconstruction");
    println!("{}", "-".repeat(80));

    let tucker_shape = vec![50, 50, 50];
    let tucker_original = DenseND::<f64>::random_uniform(&tucker_shape, 0.0, 1.0);
    let tucker_ranks = vec![25, 25, 25];

    println!("Original tensor shape: {:?}", tucker_shape);
    println!("Tucker ranks: {:?}", tucker_ranks);
    println!();

    // Perform Tucker-HOSVD
    let tucker = tucker_hosvd(&tucker_original, &tucker_ranks)?;
    println!("Core tensor shape: {:?}", tucker.core.shape());
    println!("Compression ratio: {:.2}x", tucker.compression_ratio());

    // Reconstruct
    let tucker_reconstructed = tucker.reconstruct()?;
    println!(
        "Reconstructed tensor shape: {:?}",
        tucker_reconstructed.shape()
    );

    // Analyze error
    let tucker_diff = &tucker_original - &tucker_reconstructed;
    let tucker_error = tucker_diff.frobenius_norm() / tucker_original.frobenius_norm();

    println!("Reconstruction quality:");
    println!("  - Relative error: {:.6}", tucker_error);
    println!();

    // ========================================================================
    // Example 3: Tucker-HOOI iterative improvement
    // ========================================================================
    println!("Example 3: Tucker-HOOI iterative reconstruction");
    println!("{}", "-".repeat(80));

    let hooi_shape = vec![40, 40, 40];
    let hooi_original = DenseND::<f64>::random_uniform(&hooi_shape, 0.0, 1.0);
    let hooi_ranks = vec![20, 20, 20];

    println!("Original tensor shape: {:?}", hooi_shape);
    println!("Tucker ranks: {:?}", hooi_ranks);
    println!();

    // Compare HOSVD vs HOOI reconstruction
    let tucker_hosvd_result = tucker_hosvd(&hooi_original, &hooi_ranks)?;
    let hosvd_recon = tucker_hosvd_result.reconstruct()?;
    let hosvd_error =
        (&hooi_original - &hosvd_recon).frobenius_norm() / hooi_original.frobenius_norm();

    let tucker_hooi = tucker_hooi(&hooi_original, &hooi_ranks, 20, 1e-4)?;
    let hooi_recon = tucker_hooi.reconstruct()?;
    let hooi_error =
        (&hooi_original - &hooi_recon).frobenius_norm() / hooi_original.frobenius_norm();

    println!("HOSVD reconstruction error: {:.6}", hosvd_error);
    println!(
        "HOOI reconstruction error:  {:.6} ({} iterations)",
        hooi_error, tucker_hooi.iters
    );
    println!(
        "Improvement: {:.2}%",
        (hosvd_error - hooi_error) / hosvd_error * 100.0
    );
    println!();

    // ========================================================================
    // Example 4: TT-SVD reconstruction
    // ========================================================================
    println!("Example 4: TT-SVD reconstruction");
    println!("{}", "-".repeat(80));

    let tt_shape = vec![10, 10, 10, 10, 10];
    let tt_original = DenseND::<f64>::random_uniform(&tt_shape, 0.0, 1.0);
    let tt_max_ranks = vec![8, 8, 8, 8];
    let tt_tolerance = 1e-6;

    println!("Original tensor shape: {:?}", tt_shape);
    println!("Max TT-ranks: {:?}", tt_max_ranks);
    println!("Tolerance: {}", tt_tolerance);
    println!();

    // Perform TT-SVD
    let tt = tt_svd(&tt_original, &tt_max_ranks, tt_tolerance)?;
    println!("Actual TT-ranks: {:?}", tt.ranks);
    println!("Compression ratio: {:.2}x", tt.compression_ratio());

    // Reconstruct
    let tt_reconstructed = tt.reconstruct()?;
    println!("Reconstructed tensor shape: {:?}", tt_reconstructed.shape());

    // Analyze error
    let tt_diff = &tt_original - &tt_reconstructed;
    let tt_error = tt_diff.frobenius_norm() / tt_original.frobenius_norm();

    println!("Reconstruction quality:");
    println!("  - Relative error: {:.6}", tt_error);
    println!();

    // ========================================================================
    // Example 5: Comparing reconstruction accuracy across methods
    // ========================================================================
    println!("Example 5: Reconstruction accuracy comparison");
    println!("{}", "-".repeat(80));

    let comp_shape = vec![30, 30, 30, 30];
    let comp_original = DenseND::<f64>::random_uniform(&comp_shape, 0.0, 1.0);

    println!("Original tensor shape: {:?}", comp_shape);
    let comp_original_elements: usize = comp_shape.iter().product();
    println!("Original elements: {}", comp_original_elements);
    println!();

    // CP decomposition (various ranks)
    println!("CP Decomposition:");
    for &rank in &[5, 10, 20] {
        let cp = cp_als(&comp_original, rank, 50, 1e-4, InitStrategy::Random)?;
        let recon = cp.reconstruct(&comp_shape)?;
        let error = (&comp_original - &recon).frobenius_norm() / comp_original.frobenius_norm();

        let cp_elements: usize = comp_shape.iter().sum::<usize>() * rank;
        let compression = comp_original_elements as f64 / cp_elements as f64;

        println!(
            "  Rank {:2}: error={:.6}, compression={:5.2}x, elements={}",
            rank, error, compression, cp_elements
        );
    }
    println!();

    // Tucker decomposition (various ranks)
    println!("Tucker Decomposition:");
    for &r in &[10, 15, 20] {
        let ranks = vec![r; comp_shape.len()];
        let tucker = tucker_hosvd(&comp_original, &ranks)?;
        let recon = tucker.reconstruct()?;
        let error = (&comp_original - &recon).frobenius_norm() / comp_original.frobenius_norm();

        println!(
            "  Ranks {:?}: error={:.6}, compression={:5.2}x",
            ranks,
            error,
            tucker.compression_ratio()
        );
    }
    println!();

    // TT decomposition (various ranks)
    println!("TT Decomposition:");
    for &r in &[5, 10, 15] {
        let max_ranks = vec![r; comp_shape.len() - 1];
        let tt = tt_svd(&comp_original, &max_ranks, 1e-8)?;
        let recon = tt.reconstruct()?;
        let error = (&comp_original - &recon).frobenius_norm() / comp_original.frobenius_norm();

        println!(
            "  Max rank {:2} (actual {:?}): error={:.6}, compression={:5.2}x",
            r,
            tt.ranks,
            error,
            tt.compression_ratio()
        );
    }
    println!();

    // ========================================================================
    // Example 6: Element-wise reconstruction error analysis
    // ========================================================================
    println!("Example 6: Element-wise error distribution");
    println!("{}", "-".repeat(80));

    let error_shape = vec![20, 20, 20];
    let error_original = DenseND::<f64>::random_uniform(&error_shape, 0.0, 1.0);
    let error_rank = 10;

    let cp = cp_als(&error_original, error_rank, 50, 1e-4, InitStrategy::Random)?;
    let recon = cp.reconstruct(&error_shape)?;

    // Calculate element-wise errors
    let diff = &error_original - &recon;
    let diff_data = diff.as_slice();

    let abs_errors: Vec<f64> = diff_data.iter().map(|&x| x.abs()).collect();
    let mut sorted_errors = abs_errors.clone();
    sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_error = abs_errors.iter().sum::<f64>() / abs_errors.len() as f64;
    let max_error = sorted_errors.last().unwrap();
    let median_error = sorted_errors[sorted_errors.len() / 2];
    let p95_error = sorted_errors[(sorted_errors.len() as f64 * 0.95) as usize];

    println!("Element-wise absolute error statistics:");
    println!("  - Mean: {:.6}", mean_error);
    println!("  - Median: {:.6}", median_error);
    println!("  - 95th percentile: {:.6}", p95_error);
    println!("  - Max: {:.6}", max_error);
    println!();

    // ========================================================================
    // Example 7: Partial reconstruction (selected modes)
    // ========================================================================
    println!("Example 7: Reconstruction quality vs computation time");
    println!("{}", "-".repeat(80));

    let timing_shape = vec![50, 50, 50];
    let timing_original = DenseND::<f64>::random_uniform(&timing_shape, 0.0, 1.0);

    println!("Original tensor shape: {:?}", timing_shape);
    println!();

    println!("Method                      | Time (ms) | Rel. Error | Compression");
    println!("{}", "-".repeat(75));

    // CP-ALS with different ranks
    for &rank in &[5, 10, 20] {
        let start = std::time::Instant::now();
        let cp = cp_als(&timing_original, rank, 50, 1e-4, InitStrategy::Random)?;
        let recon = cp.reconstruct(&timing_shape)?;
        let elapsed = start.elapsed();

        let error = (&timing_original - &recon).frobenius_norm() / timing_original.frobenius_norm();
        let compression = timing_shape.iter().product::<usize>() as f64
            / (timing_shape.iter().sum::<usize>() * rank) as f64;

        println!(
            "CP-ALS (rank {:2})           | {:9.2} | {:10.6} | {:9.2}x",
            rank,
            elapsed.as_secs_f64() * 1000.0,
            error,
            compression
        );
    }

    // Tucker-HOSVD
    for &r in &[20, 30] {
        let ranks = vec![r, r, r];
        let start = std::time::Instant::now();
        let tucker = tucker_hosvd(&timing_original, &ranks)?;
        let recon = tucker.reconstruct()?;
        let elapsed = start.elapsed();

        let error = (&timing_original - &recon).frobenius_norm() / timing_original.frobenius_norm();

        println!(
            "Tucker-HOSVD (ranks {:?}) | {:9.2} | {:10.6} | {:9.2}x",
            ranks,
            elapsed.as_secs_f64() * 1000.0,
            error,
            tucker.compression_ratio()
        );
    }
    println!();

    // ========================================================================
    // Example 8: Reconstruction from modified factors
    // ========================================================================
    println!("Example 8: Effect of factor modification on reconstruction");
    println!("{}", "-".repeat(80));

    let mod_shape = vec![30, 30, 30];
    let mod_original = DenseND::<f64>::random_uniform(&mod_shape, 0.0, 1.0);
    let mod_rank = 10;

    let mut cp = cp_als(&mod_original, mod_rank, 50, 1e-4, InitStrategy::Random)?;
    let orig_recon = cp.reconstruct(&mod_shape)?;
    let orig_error = (&mod_original - &orig_recon).frobenius_norm() / mod_original.frobenius_norm();

    println!("Original CP reconstruction error: {:.6}", orig_error);
    println!();

    // Scale one factor
    cp.factors[0] = cp.factors[0].mapv(|x| x * 0.9);
    cp.factors[1] = cp.factors[1].mapv(|x| x * 1.0 / 0.9);

    let scaled_recon = cp.reconstruct(&mod_shape)?;
    let scaled_error =
        (&mod_original - &scaled_recon).frobenius_norm() / mod_original.frobenius_norm();

    println!(
        "After scaling factors (should be similar): {:.6}",
        scaled_error
    );
    println!(
        "Error difference: {:.2e}",
        (orig_error - scaled_error).abs()
    );
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("{}", "=".repeat(80));
    println!("Reconstruction Quality Summary");
    println!("{}", "-".repeat(80));
    println!();
    println!("Key observations:");
    println!("  1. CP decomposition provides good compression for low-rank tensors");
    println!("  2. Tucker (HOOI) improves reconstruction over HOSVD through iteration");
    println!("  3. TT decomposition excels for high-order tensors");
    println!("  4. Higher ranks/compression reduce reconstruction error at cost of storage");
    println!("  5. Factor scaling invariance allows flexible representations");
    println!();
    println!("{}", "=".repeat(80));
    println!("Reconstruction Example Complete!");
    println!("{}", "=".repeat(80));

    Ok(())
}
