//! Complete CP-ALS workflow example using tenrso-kernels utilities
//!
//! This example demonstrates a full CP-ALS (Canonical Polyadic Alternating Least Squares)
//! decomposition iteration using the batch operations and normalization utilities.
//!
//! Run with: cargo run --example cp_als_workflow --features parallel

use scirs2_core::ndarray_ext::Array2;
use tenrso_core::DenseND;
use tenrso_kernels::{
    cp_reconstruct, denormalize_factor, frobenius_norm_tensor, mttkrp_all_modes, normalize_factor,
    validate_factor_shapes,
};

fn main() {
    println!("=== CP-ALS Workflow Example ===\n");

    // Problem setup
    let size = 30;
    let rank = 5;
    let n_modes = 3;
    let max_iters = 20;
    let tol = 1e-6;

    println!("Problem:");
    println!("  Tensor shape: {}×{}×{}", size, size, size);
    println!("  CP rank: {}", rank);
    println!("  Max iterations: {}", max_iters);
    println!("  Tolerance: {:.1e}\n", tol);

    // Create a test tensor (in practice, this would be your data)
    println!("Creating test tensor...");
    let tensor = DenseND::<f64>::random_uniform(&[size, size, size], 0.0, 1.0);
    let tensor_norm = frobenius_norm_tensor(&tensor.view());
    println!("  Tensor Frobenius norm: {:.4}\n", tensor_norm);

    // Initialize factor matrices randomly
    println!("Initializing {} random factor matrices...", n_modes);
    let mut factors: Vec<Array2<f64>> = (0..n_modes)
        .map(|_| {
            Array2::from_shape_fn((size, rank), |(i, j)| {
                // Simple pseudo-random initialization
                let seed = (i * 31 + j * 17) as f64;
                seed.sin().abs()
            })
        })
        .collect();

    // Validate initial factor shapes
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    match validate_factor_shapes(&[size, size, size], &factor_views) {
        Ok(r) => println!("  Factor shapes validated (rank: {})", r),
        Err(e) => {
            eprintln!("  Error: {}", e);
            return;
        }
    }

    // Store norms for later denormalization
    let mut all_norms: Vec<Vec<f64>>;

    println!("\n=== Beginning CP-ALS Iterations ===\n");

    let mut prev_error = f64::INFINITY;

    for iter in 0..max_iters {
        // 1. Normalize all factors
        let mut normalized_factors = Vec::with_capacity(n_modes);
        let mut norms_this_iter = Vec::with_capacity(n_modes);

        for factor in factors.iter() {
            let (normalized, norms) = normalize_factor(&factor.view());
            normalized_factors.push(normalized);
            norms_this_iter.push(norms);
        }

        all_norms = norms_this_iter;

        // 2. Compute MTTKRP for all modes in one call
        let normalized_views: Vec<_> = normalized_factors.iter().map(|f| f.view()).collect();

        let updated_factors = match mttkrp_all_modes(&tensor.view(), &normalized_views) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("MTTKRP failed: {}", e);
                return;
            }
        };

        // 3. Update factors
        factors = updated_factors;

        // 4. Denormalize the last factor to recover scale
        // (In practice, you'd accumulate norms across iterations)
        let last_mode = n_modes - 1;
        factors[last_mode] = denormalize_factor(&factors[last_mode].view(), &all_norms[last_mode]);

        // 5. Compute reconstruction and error
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let reconstructed = match cp_reconstruct(&factor_views, None) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Reconstruction failed: {}", e);
                return;
            }
        };

        // Compute relative error
        let recon_dense = DenseND::from_array(reconstructed);
        let error = compute_reconstruction_error(&tensor, &recon_dense);
        let rel_error = error / tensor_norm;

        // Progress reporting
        if iter % 5 == 0 || iter == max_iters - 1 {
            println!(
                "Iteration {:2}: error = {:.6e}, rel_error = {:.6e}",
                iter + 1,
                error,
                rel_error
            );
        }

        // Convergence check
        let error_change = (prev_error - error).abs();
        if error_change < tol {
            println!(
                "\nConverged after {} iterations! (error change: {:.2e})",
                iter + 1,
                error_change
            );
            break;
        }

        prev_error = error;
    }

    println!("\n=== CP-ALS Workflow Complete ===\n");

    // Final statistics
    println!("Final factor statistics:");
    for (mode_idx, factor) in factors.iter().enumerate() {
        let (nrows, ncols) = factor.dim();
        let factor_norm: f64 = factor.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!(
            "  Factor {}: shape={}×{}, norm={:.4}",
            mode_idx, nrows, ncols, factor_norm
        );
    }

    println!("\nKey features demonstrated:");
    println!("  ✓ validate_factor_shapes() - Pre-iteration validation");
    println!("  ✓ normalize_factor() - Numerical stability");
    println!("  ✓ mttkrp_all_modes() - Batch MTTKRP computation");
    println!("  ✓ denormalize_factor() - Scale recovery");
    println!("  ✓ cp_reconstruct() - Tensor reconstruction");
    println!("  ✓ frobenius_norm_tensor() - Error computation");
}

/// Compute reconstruction error between original and reconstructed tensors
fn compute_reconstruction_error(original: &DenseND<f64>, reconstructed: &DenseND<f64>) -> f64 {
    assert_eq!(original.shape(), reconstructed.shape());

    let orig_view = original.view();
    let recon_view = reconstructed.view();

    let mut error_sq = 0.0;
    for (a, b) in orig_view.iter().zip(recon_view.iter()) {
        let diff = a - b;
        error_sq += diff * diff;
    }

    error_sq.sqrt()
}
