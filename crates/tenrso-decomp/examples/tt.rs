//! TT-SVD (Tensor Train via SVD) Example
//!
//! This example demonstrates Tensor Train decomposition, which factorizes
//! a high-order tensor into a sequence of 3-mode core tensors.
//!
//! TT decomposition is particularly useful for:
//! - High-order tensors (5+ modes)
//! - Quantum computing / tensor networks
//! - Memory-efficient representation
//!
//! Run with:
//! ```bash
//! cargo run --example tt
//! ```

use tenrso_core::DenseND;
use tenrso_decomp::tt_svd;

fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(80));
    println!("TT-SVD (Tensor Train) Decomposition Example");
    println!("{}", "=".repeat(80));
    println!();

    // ========================================================================
    // Example 1: Basic TT-SVD on a 4D tensor
    // ========================================================================
    println!("Example 1: Basic TT-SVD decomposition (4D tensor)");
    println!("{}", "-".repeat(80));

    let shape = vec![16, 16, 16, 16];
    let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
    let max_ranks = vec![8, 8, 8]; // n_modes - 1 ranks
    let tolerance = 1e-6;

    println!("Tensor shape: {:?}", shape);
    println!("Max TT-ranks: {:?}", max_ranks);
    println!("Tolerance: {}", tolerance);
    println!();

    let start = std::time::Instant::now();
    let tt = tt_svd(&tensor, &max_ranks, tolerance)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  - Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  - Actual TT-ranks: {:?}", tt.ranks);
    println!("  - Core tensor shapes:");
    for (i, core) in tt.cores.iter().enumerate() {
        println!("    Core {}: {:?}", i, core.shape());
    }
    println!("  - Compression ratio: {:.2}x", tt.compression_ratio());
    println!();

    // Verify reconstruction
    let reconstructed = tt.reconstruct()?;
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
    // Example 2: High-order tensor (6D)
    // ========================================================================
    println!("Example 2: High-order TT decomposition (6D tensor)");
    println!("{}", "-".repeat(80));

    let shape_6d = vec![8, 8, 8, 8, 8, 8];
    let tensor_6d = DenseND::<f64>::random_uniform(&shape_6d, 0.0, 1.0);
    let max_ranks_6d = vec![10, 10, 10, 10, 10]; // 5 ranks for 6D tensor
    let tolerance_6d = 1e-8;

    println!("Tensor shape: {:?}", shape_6d);
    println!("Max TT-ranks: {:?}", max_ranks_6d);
    let original_elements: usize = shape_6d.iter().product();
    println!("Original elements: {}", original_elements);
    println!();

    let start = std::time::Instant::now();
    let tt_6d = tt_svd(&tensor_6d, &max_ranks_6d, tolerance_6d)?;
    let elapsed = start.elapsed();

    println!("Results:");
    println!("  - Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  - Actual TT-ranks: {:?}", tt_6d.ranks);
    println!("  - Compression ratio: {:.2}x", tt_6d.compression_ratio());

    let recon_6d = tt_6d.reconstruct()?;
    let error_6d = (&tensor_6d - &recon_6d).frobenius_norm() / tensor_6d.frobenius_norm();
    println!("  - Relative error: {:.6}", error_6d);
    println!();

    // ========================================================================
    // Example 3: Compression analysis with varying ranks
    // ========================================================================
    println!("Example 3: Compression vs accuracy trade-off");
    println!("{}", "-".repeat(80));

    let comp_shape = vec![10, 10, 10, 10, 10];
    let comp_tensor = DenseND::<f64>::random_uniform(&comp_shape, 0.0, 1.0);
    let comp_original_elements: usize = comp_shape.iter().product();

    println!("Tensor shape: {:?}", comp_shape);
    println!("Original elements: {}", comp_original_elements);
    println!();

    println!("Max Rank | Actual Ranks          | Compression | Rel. Error | Time (ms)");
    println!("{}", "-".repeat(80));

    for &max_rank in &[2, 5, 10, 15, 20] {
        let max_ranks = vec![max_rank; comp_shape.len() - 1];
        let start = std::time::Instant::now();
        let tt = tt_svd(&comp_tensor, &max_ranks, 1e-10)?;
        let elapsed = start.elapsed();

        let recon = tt.reconstruct()?;
        let error = (&comp_tensor - &recon).frobenius_norm() / comp_tensor.frobenius_norm();

        println!(
            "{:8} | {:21?} | {:10.2}x | {:10.6} | {:8.2}",
            max_rank,
            tt.ranks,
            tt.compression_ratio(),
            error,
            elapsed.as_secs_f64() * 1000.0
        );
    }
    println!();

    // ========================================================================
    // Example 4: Tolerance-based rank truncation
    // ========================================================================
    println!("Example 4: Tolerance-based rank truncation");
    println!("{}", "-".repeat(80));

    let tol_shape = vec![12, 12, 12, 12];
    let tol_tensor = DenseND::<f64>::random_uniform(&tol_shape, 0.0, 1.0);
    let max_ranks_high = vec![50, 50, 50]; // High max rank

    println!("Tensor shape: {:?}", tol_shape);
    println!("Max TT-ranks: {:?}", max_ranks_high);
    println!();

    println!("Tolerance | Actual Ranks    | Compression | Rel. Error");
    println!("{}", "-".repeat(65));

    for &tol in &[1e-2, 1e-4, 1e-6, 1e-8, 1e-10] {
        let tt = tt_svd(&tol_tensor, &max_ranks_high, tol)?;
        let recon = tt.reconstruct()?;
        let error = (&tol_tensor - &recon).frobenius_norm() / tol_tensor.frobenius_norm();

        println!(
            "{:9.0e} | {:15?} | {:10.2}x | {:10.6}",
            tol,
            tt.ranks,
            tt.compression_ratio(),
            error
        );
    }
    println!();

    // ========================================================================
    // Example 5: Non-uniform mode sizes
    // ========================================================================
    println!("Example 5: Non-uniform mode sizes");
    println!("{}", "-".repeat(80));

    let asymmetric_cases = vec![
        vec![20, 15, 10, 8],
        vec![30, 10, 10, 10],
        vec![8, 8, 8, 8, 8, 8, 8],
    ];

    for shape in asymmetric_cases {
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let max_ranks = vec![10; shape.len() - 1];

        let start = std::time::Instant::now();
        let tt = tt_svd(&tensor, &max_ranks, 1e-8)?;
        let elapsed = start.elapsed();

        let recon = tt.reconstruct()?;
        let error = (&tensor - &recon).frobenius_norm() / tensor.frobenius_norm();

        let original_elements: usize = shape.iter().product();
        println!(
            "Shape {:?}: elements={}, compression={:.2}x, error={:.6}, time={:.2}ms",
            shape,
            original_elements,
            tt.compression_ratio(),
            error,
            elapsed.as_secs_f64() * 1000.0
        );
    }
    println!();

    // ========================================================================
    // Example 6: Core tensor analysis
    // ========================================================================
    println!("Example 6: TT core tensor statistics");
    println!("{}", "-".repeat(80));

    let analysis_tensor = DenseND::<f64>::random_uniform(&[10, 10, 10, 10, 10], 0.0, 1.0);
    let tt = tt_svd(&analysis_tensor, &vec![8, 8, 8, 8], 1e-6)?;

    println!("TT-ranks: {:?}", tt.ranks);
    println!();

    for (i, core) in tt.cores.iter().enumerate() {
        let core_data = core.as_slice().expect("Core tensor should be contiguous");
        let mean = core_data.iter().sum::<f64>() / core_data.len() as f64;
        let std = (core_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / core_data.len() as f64)
            .sqrt();
        let min = core_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = core_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let norm = core_data.iter().map(|&x| x * x).sum::<f64>().sqrt();

        println!("Core {} (shape {:?}):", i, core.shape());
        println!("  Mean: {:.6}, Std: {:.6}", mean, std);
        println!("  Min: {:.6}, Max: {:.6}", min, max);
        println!("  Norm: {:.6}", norm);
        println!();
    }

    // ========================================================================
    // Example 7: Memory efficiency for high-order tensors
    // ========================================================================
    println!("Example 7: Memory efficiency analysis");
    println!("{}", "-".repeat(80));

    let memory_cases = vec![
        (vec![4; 10], 3),  // 4^10 tensor with rank-3
        (vec![5; 8], 5),   // 5^8 tensor with rank-5
        (vec![6; 7], 8),   // 6^7 tensor with rank-8
        (vec![10; 6], 10), // 10^6 tensor with rank-10
    ];

    println!("Shape                | Rank | Original Elements | TT Elements | Compression");
    println!("{}", "-".repeat(85));

    for (shape, rank) in memory_cases {
        let original_elements: usize = shape.iter().product();
        let _max_ranks = vec![rank; shape.len() - 1];

        // Estimate TT storage (without creating the tensor)
        let mut tt_elements = 0;
        let n_modes = shape.len();

        for i in 0..n_modes {
            let r_left = if i == 0 { 1 } else { rank };
            let r_right = if i == n_modes - 1 { 1 } else { rank };
            tt_elements += r_left * shape[i] * r_right;
        }

        let compression = original_elements as f64 / tt_elements as f64;

        println!(
            "{:20?} | {:4} | {:17} | {:11} | {:10.2}x",
            shape, rank, original_elements, tt_elements, compression
        );
    }
    println!();

    // ========================================================================
    // Example 8: Comparison with CP and Tucker
    // ========================================================================
    println!("Example 8: TT vs CP vs Tucker decomposition");
    println!("{}", "-".repeat(80));

    let compare_shape = vec![10, 10, 10, 10];
    let compare_tensor = DenseND::<f64>::random_uniform(&compare_shape, 0.0, 1.0);

    // TT decomposition
    let start = std::time::Instant::now();
    let tt = tt_svd(&compare_tensor, &vec![10, 10, 10], 1e-6)?;
    let tt_time = start.elapsed();
    let tt_recon = tt.reconstruct()?;
    let tt_error = (&compare_tensor - &tt_recon).frobenius_norm() / compare_tensor.frobenius_norm();

    use tenrso_decomp::{cp_als, tucker_hosvd, InitStrategy};

    // CP decomposition (rank-10)
    let start = std::time::Instant::now();
    let cp = cp_als(&compare_tensor, 10, 50, 1e-6, InitStrategy::Random)?;
    let cp_time = start.elapsed();
    let cp_recon = cp.reconstruct(&compare_shape)?;
    let cp_error = (&compare_tensor - &cp_recon).frobenius_norm() / compare_tensor.frobenius_norm();

    // Tucker decomposition
    let start = std::time::Instant::now();
    let tucker = tucker_hosvd(&compare_tensor, &vec![8, 8, 8, 8])?;
    let tucker_time = start.elapsed();
    let tucker_recon = tucker.reconstruct()?;
    let tucker_error =
        (&compare_tensor - &tucker_recon).frobenius_norm() / compare_tensor.frobenius_norm();

    println!("Tensor shape: {:?}", compare_shape);
    println!();
    println!("Method  | Compression | Rel. Error | Time (ms)");
    println!("{}", "-".repeat(55));
    println!(
        "TT      | {:10.2}x | {:10.6} | {:8.2}",
        tt.compression_ratio(),
        tt_error,
        tt_time.as_secs_f64() * 1000.0
    );
    println!(
        "CP-10   | {:10.2}x | {:10.6} | {:8.2}",
        compare_shape.iter().product::<usize>() as f64
            / (compare_shape.iter().sum::<usize>() * 10) as f64,
        cp_error,
        cp_time.as_secs_f64() * 1000.0
    );
    println!(
        "Tucker  | {:10.2}x | {:10.6} | {:8.2}",
        tucker.compression_ratio(),
        tucker_error,
        tucker_time.as_secs_f64() * 1000.0
    );
    println!();

    println!("{}", "=".repeat(80));
    println!("TT-SVD Example Complete!");
    println!("{}", "=".repeat(80));

    Ok(())
}
