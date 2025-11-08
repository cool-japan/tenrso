//! Tucker Decomposition Example
//!
//! This example demonstrates Tucker-HOSVD (Higher-Order SVD) and
//! Tucker-HOOI (Higher-Order Orthogonal Iteration) decompositions.
//!
//! Tucker decomposition factorizes a tensor into a core tensor
//! and orthogonal factor matrices along each mode.
//!
//! Run with:
//! ```bash
//! cargo run --example tucker
//! ```

use tenrso_core::DenseND;
use tenrso_decomp::{tucker_hooi, tucker_hosvd};

fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(80));
    println!("Tucker Decomposition Example");
    println!("{}", "=".repeat(80));
    println!();

    // ========================================================================
    // Example 1: Basic Tucker-HOSVD decomposition
    // ========================================================================
    println!("Example 1: Tucker-HOSVD (Higher-Order SVD)");
    println!("{}", "-".repeat(80));

    let shape = vec![60, 50, 40];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let ranks = vec![30, 25, 20];

    println!("Tensor shape: {:?}", shape);
    println!("Target ranks: {:?}", ranks);
    println!();

    let start = std::time::Instant::now();
    let tucker = tucker_hosvd(&tensor, &ranks)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  - Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  - Core shape: {:?}", tucker.core.shape());
    println!("  - Factor matrix shapes:");
    for (i, factor) in tucker.factors.iter().enumerate() {
        println!("    Mode {}: {:?}", i, factor.shape());
    }
    println!("  - Compression ratio: {:.2}x", tucker.compression_ratio());
    println!();

    // Verify reconstruction
    let reconstructed = tucker.reconstruct()?;
    let original_norm = tensor.frobenius_norm();
    let diff = &tensor - &reconstructed;
    let error = diff.frobenius_norm();
    let relative_error = error / original_norm;

    println!("Reconstruction:");
    println!("  - Original norm: {:.6}", original_norm);
    println!("  - Reconstruction error: {:.6}", error);
    println!("  - Relative error: {:.6}", relative_error);
    println!();

    // ========================================================================
    // Example 2: Tucker-HOOI (iterative refinement)
    // ========================================================================
    println!("Example 2: Tucker-HOOI (iterative refinement)");
    println!("{}", "-".repeat(80));

    let shape = vec![40, 40, 40];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let ranks = vec![20, 20, 20];
    let max_iters = 20;
    let tolerance = 1e-4;

    println!("Tensor shape: {:?}", shape);
    println!("Target ranks: {:?}", ranks);
    println!("Max iterations: {}", max_iters);
    println!("Tolerance: {}", tolerance);
    println!();

    let start = std::time::Instant::now();
    let tucker_result = tucker_hooi(&tensor, &ranks, max_iters, tolerance)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  - Iterations: {}", tucker_result.iters);
    println!("  - Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  - Core shape: {:?}", tucker_result.core.shape());
    println!(
        "  - Compression ratio: {:.2}x",
        tucker_result.compression_ratio()
    );
    println!();

    let reconstructed_hooi = tucker_result.reconstruct()?;
    let diff_hooi = &tensor - &reconstructed_hooi;
    let error_hooi = diff_hooi.frobenius_norm();
    let relative_error_hooi = error_hooi / original_norm;

    println!("Reconstruction:");
    println!("  - Reconstruction error: {:.6}", error_hooi);
    println!("  - Relative error: {:.6}", relative_error_hooi);
    println!();

    // ========================================================================
    // Example 3: Compare HOSVD vs HOOI
    // ========================================================================
    println!("Example 3: HOSVD vs HOOI comparison");
    println!("{}", "-".repeat(80));

    let comp_shape = vec![50, 50, 50];
    let comp_tensor = DenseND::<f64>::random_uniform(&comp_shape, 0.0, 1.0);
    let comp_ranks = vec![25, 25, 25];

    println!("Tensor shape: {:?}", comp_shape);
    println!("Ranks: {:?}", comp_ranks);
    println!();

    // HOSVD
    let start = std::time::Instant::now();
    let tucker_hosvd_result = tucker_hosvd(&comp_tensor, &comp_ranks)?;
    let hosvd_time = start.elapsed();

    let hosvd_recon = tucker_hosvd_result.reconstruct()?;
    let hosvd_diff = &comp_tensor - &hosvd_recon;
    let hosvd_error = hosvd_diff.frobenius_norm() / comp_tensor.frobenius_norm();

    // HOOI
    let start = std::time::Instant::now();
    let tucker_hooi_comp = tucker_hooi(&comp_tensor, &comp_ranks, 20, 1e-4)?;
    let hooi_time = start.elapsed();

    let hooi_recon = tucker_hooi_comp.reconstruct()?;
    let hooi_diff = &comp_tensor - &hooi_recon;
    let hooi_error = hooi_diff.frobenius_norm() / comp_tensor.frobenius_norm();

    println!("HOSVD:");
    println!("  - Time: {:.2}ms", hosvd_time.as_secs_f64() * 1000.0);
    println!("  - Relative error: {:.6}", hosvd_error);
    println!();

    println!("HOOI:");
    println!("  - Time: {:.2}ms", hooi_time.as_secs_f64() * 1000.0);
    println!("  - Iterations: {}", tucker_hooi_comp.iters);
    println!("  - Relative error: {:.6}", hooi_error);
    println!(
        "  - Improvement: {:.2}%",
        (hosvd_error - hooi_error) / hosvd_error * 100.0
    );
    println!();

    // ========================================================================
    // Example 4: Compression analysis with varying ranks
    // ========================================================================
    println!("Example 4: Compression vs accuracy trade-off");
    println!("{}", "-".repeat(80));

    let data_shape = vec![80, 80, 80];
    let data_tensor = DenseND::<f64>::random_uniform(&data_shape, 0.0, 1.0);
    let original_elements: usize = data_shape.iter().product();

    println!("Original tensor: {:?}", data_shape);
    println!("Original elements: {}", original_elements);
    println!();

    println!("Rank | Compression | Rel. Error | Time (ms)");
    println!("{}", "-".repeat(55));

    for &r in &[10, 20, 30, 40, 50, 60] {
        let ranks = vec![r, r, r];
        let start = std::time::Instant::now();
        let tucker = tucker_hosvd(&data_tensor, &ranks)?;
        let elapsed = start.elapsed();

        let recon = tucker.reconstruct()?;
        let diff = &data_tensor - &recon;
        let error = diff.frobenius_norm() / data_tensor.frobenius_norm();

        println!(
            "{:4} | {:10.2}x | {:10.6} | {:8.2}",
            r,
            tucker.compression_ratio(),
            error,
            elapsed.as_secs_f64() * 1000.0
        );
    }
    println!();

    // ========================================================================
    // Example 5: Non-cubic tensors
    // ========================================================================
    println!("Example 5: Non-cubic tensor decomposition");
    println!("{}", "-".repeat(80));

    let asymmetric_cases = vec![
        (vec![100, 80, 60], vec![50, 40, 30]),
        (vec![120, 40, 40], vec![60, 20, 20]),
        (vec![60, 60, 120], vec![30, 30, 60]),
    ];

    for (shape, ranks) in asymmetric_cases {
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        let start = std::time::Instant::now();
        let tucker = tucker_hosvd(&tensor, &ranks)?;
        let elapsed = start.elapsed();

        let recon = tucker.reconstruct()?;
        let diff = &tensor - &recon;
        let error = diff.frobenius_norm() / tensor.frobenius_norm();

        println!(
            "Shape {:?} -> {:?}: compression={:.2}x, error={:.6}, time={:.2}ms",
            shape,
            ranks,
            tucker.compression_ratio(),
            error,
            elapsed.as_secs_f64() * 1000.0
        );
    }
    println!();

    // ========================================================================
    // Example 6: Factor orthogonality verification
    // ========================================================================
    println!("Example 6: Verifying factor orthogonality");
    println!("{}", "-".repeat(80));

    let ortho_tensor = DenseND::<f64>::random_uniform(&[50, 50, 50], 0.0, 1.0);
    let ortho_ranks = vec![30, 30, 30];
    let tucker = tucker_hosvd(&ortho_tensor, &ortho_ranks)?;

    println!("Checking orthogonality (U^T U ≈ I) for each factor:");
    for (mode, factor) in tucker.factors.iter().enumerate() {
        // Compute U^T U
        let gram = factor.t().dot(factor);

        // Check if diagonal is close to 1 and off-diagonal close to 0
        let (rows, cols) = gram.dim();
        let mut max_diag_error: f64 = 0.0;
        let mut max_offdiag: f64 = 0.0;

        for i in 0..rows {
            for j in 0..cols {
                if i == j {
                    max_diag_error = max_diag_error.max((gram[[i, j]] - 1.0).abs());
                } else {
                    max_offdiag = max_offdiag.max(gram[[i, j]].abs());
                }
            }
        }

        println!(
            "  Mode {}: max |diag - 1| = {:.2e}, max |off-diag| = {:.2e}",
            mode, max_diag_error, max_offdiag
        );
    }
    println!();

    // ========================================================================
    // Example 7: HOOI convergence behavior
    // ========================================================================
    println!("Example 7: HOOI convergence behavior");
    println!("{}", "-".repeat(80));

    let conv_tensor = DenseND::<f64>::random_uniform(&[40, 40, 40], 0.0, 1.0);
    let conv_ranks = vec![20, 20, 20];

    println!("Iterations | Rel. Error | Time (ms)");
    println!("{}", "-".repeat(40));

    for &iters in &[1, 5, 10, 20, 50] {
        let start = std::time::Instant::now();
        let tucker = tucker_hooi(&conv_tensor, &conv_ranks, iters, 1e-10)?;
        let elapsed = start.elapsed();

        let recon = tucker.reconstruct()?;
        let diff = &conv_tensor - &recon;
        let error = diff.frobenius_norm() / conv_tensor.frobenius_norm();

        println!(
            "{:10} | {:10.6} | {:8.2}",
            tucker.iters,
            error,
            elapsed.as_secs_f64() * 1000.0
        );
    }
    println!();

    // ========================================================================
    // Example 8: Core tensor analysis
    // ========================================================================
    println!("Example 8: Core tensor statistics");
    println!("{}", "-".repeat(80));

    let core_tensor = DenseND::<f64>::random_uniform(&[60, 60, 60], 0.0, 1.0);
    let core_ranks = vec![30, 30, 30];
    let tucker = tucker_hosvd(&core_tensor, &core_ranks)?;

    println!("Core tensor shape: {:?}", tucker.core.shape());

    let core_data = tucker.core.as_slice();
    let mean = core_data.iter().sum::<f64>() / core_data.len() as f64;
    let std = (core_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / core_data.len() as f64)
        .sqrt();
    let min = core_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = core_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("Core statistics:");
    println!("  - Mean: {:.6}", mean);
    println!("  - Std: {:.6}", std);
    println!("  - Min: {:.6}", min);
    println!("  - Max: {:.6}", max);
    println!("  - Norm: {:.6}", tucker.core.frobenius_norm());
    println!();

    println!("{}", "=".repeat(80));
    println!("Tucker Decomposition Example Complete!");
    println!("{}", "=".repeat(80));

    Ok(())
}
