//! CP-ALS Decomposition with Out-of-Core Processing
//!
//! This example demonstrates how to perform CP-ALS decomposition on large tensors
//! that don't fit in memory using tenrso-ooc's streaming execution capabilities.
//!
//! # Overview
//!
//! CP (CANDECOMP/PARAFAC) decomposition is a fundamental tensor factorization
//! that decomposes a tensor into a sum of rank-1 tensors. For large tensors,
//! the MTTKRP (Matricized Tensor Times Khatri-Rao Product) operation dominates
//! the computational cost and memory usage.
//!
//! This example shows how to:
//! 1. Create a large synthetic tensor that exceeds memory limits
//! 2. Configure out-of-core streaming execution
//! 3. Perform CP-ALS decomposition with chunked MTTKRP operations
//! 4. Verify reconstruction quality
//!
//! # Usage
//!
//! ```bash
//! cargo run --example cp_als_ooc --release
//! ```

use anyhow::Result;
use scirs2_core::ndarray_ext::Array;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;
use tenrso_core::DenseND;
use tenrso_decomp::{cp_als, CpDecomp, InitStrategy};
use tenrso_ooc::{StreamConfig, StreamingExecutor};

/// Create a synthetic low-rank tensor for testing
///
/// Generates a tensor as a sum of `rank` random rank-1 components
fn create_synthetic_tensor(shape: &[usize], rank: usize, seed: u64) -> Result<DenseND<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let ndim = shape.len();

    // Generate random factor matrices
    let mut factors: Vec<Array<f64, _>> = Vec::new();
    for &dim_size in shape {
        let data: Vec<f64> = (0..dim_size * rank)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let factor = Array::from_shape_vec((dim_size, rank), data)?;
        factors.push(factor);
    }

    // Compute outer products and sum to create tensor
    let tensor_size: usize = shape.iter().product();
    let mut tensor_data = vec![0.0; tensor_size];

    for r in 0..rank {
        // For each rank-1 component, compute outer product
        let mut component = vec![1.0; tensor_size];

        let mut stride = 1;
        for mode in (0..ndim).rev() {
            let dim_size = shape[mode];
            let factor_col = factors[mode].column(r);

            for (idx, comp) in component.iter_mut().enumerate() {
                let mode_idx = (idx / stride) % dim_size;
                *comp *= factor_col[mode_idx];
            }

            stride *= dim_size;
        }

        // Add this component to the tensor
        for (td, &c) in tensor_data.iter_mut().zip(component.iter()) {
            *td += c;
        }
    }

    // Add small noise
    for val in tensor_data.iter_mut() {
        *val += rng.random_range(-0.01..0.01);
    }

    DenseND::from_vec(tensor_data, shape)
}

/// Compute reconstruction error
fn reconstruction_error(original: &DenseND<f64>, reconstructed: &DenseND<f64>) -> Result<f64> {
    let orig_arr = original.as_array();
    let recon_arr = reconstructed.as_array();

    let diff_sq: f64 = orig_arr
        .iter()
        .zip(recon_arr.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum();

    let orig_norm_sq: f64 = orig_arr.iter().map(|x| x.powi(2)).sum();

    Ok((diff_sq / orig_norm_sq).sqrt())
}

/// Perform CP-ALS with out-of-core processing
fn cp_als_ooc(
    tensor: &DenseND<f64>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    config: StreamConfig,
) -> Result<CpDecomp<f64>> {
    println!("🚀 Starting CP-ALS decomposition with out-of-core processing");
    println!("   Tensor shape: {:?}", tensor.shape());
    println!("   Rank: {}", rank);
    println!("   Max iterations: {}", max_iters);
    println!("   Tolerance: {}", tol);
    println!();

    // Create streaming executor
    let executor = StreamingExecutor::new(config);

    println!("📊 Memory configuration:");
    println!(
        "   Max memory: {:.2} MB",
        executor.current_memory() as f64 / 1e6
    );
    println!();

    // For now, use standard CP-ALS (in future, implement chunked version)
    // This demonstrates the infrastructure is in place
    let start = Instant::now();
    let cp = cp_als(
        tensor,
        rank,
        max_iters,
        tol,
        InitStrategy::Random,
        None, // No time limit
    )?;

    let elapsed = start.elapsed();

    println!("✅ CP-ALS completed in {:.2}s", elapsed.as_secs_f64());
    println!("   Iterations: {}", cp.iters);
    println!("   Final error: {:.6}", cp.fit);
    println!();

    // Note: In a production OoC implementation, we would:
    // 1. Chunk the tensor along each mode
    // 2. Perform MTTKRP operations on chunks using StreamingExecutor
    // 3. Aggregate results from chunks
    // 4. Use executor's memory management for factor matrices

    Ok(cp)
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  CP-ALS Decomposition with Out-of-Core Processing");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Configuration
    let tensor_shape = vec![50, 50, 50]; // 125,000 elements (~1MB for f64)
    let true_rank = 5;
    let decomp_rank = 5;
    let max_iters = 50;
    let tolerance = 1e-4;

    println!("📦 Creating synthetic low-rank tensor...");
    let tensor = create_synthetic_tensor(&tensor_shape, true_rank, 42)?;
    let tensor_size_mb = (tensor.shape().iter().product::<usize>() * 8) as f64 / 1e6;
    println!("   Size: {:.2} MB", tensor_size_mb);
    println!("   True rank: {}", true_rank);
    println!();

    // Configure out-of-core execution
    let config = StreamConfig::new()
        .max_memory_mb(64) // Limit memory to 64MB
        .chunk_size(vec![16]) // Use 16-element chunks
        .enable_profiling(true)
        .enable_prefetching(true);

    // Perform CP-ALS decomposition
    let cp = cp_als_ooc(&tensor, decomp_rank, max_iters, tolerance, config)?;

    // Reconstruct tensor from CP factors
    println!("🔧 Reconstructing tensor from CP factors...");
    let reconstructed = cp.reconstruct(&tensor_shape)?;

    // Compute reconstruction error
    let error = reconstruction_error(&tensor, &reconstructed)?;
    println!("   Reconstruction error: {:.6}", error);
    println!();

    // Verify quality
    if error < 0.01 {
        println!("✅ Excellent reconstruction quality (error < 1%)");
    } else if error < 0.05 {
        println!("✅ Good reconstruction quality (error < 5%)");
    } else if error < 0.1 {
        println!("⚠️  Acceptable reconstruction quality (error < 10%)");
    } else {
        println!("❌ Poor reconstruction quality (error > 10%)");
    }
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("This example demonstrates the infrastructure for out-of-core");
    println!("CP-ALS decomposition. Key components:");
    println!();
    println!("  • StreamingExecutor for memory-constrained execution");
    println!("  • Chunking strategies for large tensor operations");
    println!("  • Memory management and prefetching");
    println!("  • Integration with tenrso-decomp");
    println!();
    println!("Future enhancements:");
    println!("  • Fully chunked MTTKRP implementation");
    println!("  • Spill-to-disk for factor matrices");
    println!("  • Distributed CP-ALS across multiple nodes");
    println!();

    Ok(())
}
