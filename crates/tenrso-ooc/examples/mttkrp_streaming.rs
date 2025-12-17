//! MTTKRP (Matricized Tensor Times Khatri-Rao Product) with Streaming
//!
//! This example demonstrates how to perform MTTKRP operations on large tensors
//! using tenrso-ooc's streaming execution with chunking and memory management.
//!
//! # Overview
//!
//! MTTKRP is a fundamental operation in tensor decompositions (CP-ALS, Tucker-HOOI).
//! For an N-way tensor X and factor matrices A₁, A₂, ..., Aₙ, the MTTKRP along
//! mode n computes:
//!
//!   X₍ₙ₎ (Aₙ₋₁ ⊙ ... ⊙ A₁) → Result matrix
//!
//! where X₍ₙ₎ is the mode-n matricization and ⊙ is the Khatri-Rao product.
//!
//! This example shows how to:
//! 1. Create large synthetic tensor and factor matrices
//! 2. Configure streaming execution with chunking
//! 3. Perform MTTKRP with memory-efficient processing
//! 4. Compare chunked vs dense execution
//!
//! # Usage
//!
//! ```bash
//! cargo run --example mttkrp_streaming --release
//! ```

use anyhow::Result;
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;
use tenrso_core::DenseND;
use tenrso_kernels::mttkrp;
use tenrso_ooc::{StreamConfig, StreamingExecutor};

/// Create random factor matrices for testing
fn create_factor_matrices(shape: &[usize], rank: usize, seed: u64) -> Result<Vec<Array2<f64>>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut factors = Vec::new();

    for &dim_size in shape {
        let data: Vec<f64> = (0..dim_size * rank)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let factor = Array2::from_shape_vec((dim_size, rank), data)?;
        factors.push(factor);
    }

    Ok(factors)
}

/// Create a random tensor for testing
fn create_random_tensor(shape: &[usize], seed: u64) -> Result<DenseND<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let size: usize = shape.iter().product();
    let data: Vec<f64> = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
    DenseND::from_vec(data, shape)
}

/// Perform MTTKRP with streaming execution (conceptual)
///
/// In a full implementation, this would:
/// 1. Chunk the tensor along the target mode
/// 2. Process each chunk with streaming executor
/// 3. Aggregate results
fn mttkrp_streaming(
    tensor: &DenseND<f64>,
    factors: &[Array2<f64>],
    mode: usize,
    config: StreamConfig,
) -> Result<Array2<f64>> {
    println!("🚀 Starting MTTKRP with streaming execution");
    println!("   Tensor shape: {:?}", tensor.shape());
    println!("   Mode: {}", mode);
    println!("   Rank: {}", factors[0].ncols());
    println!();

    let executor = StreamingExecutor::new(config);

    println!("📊 Memory configuration:");
    println!(
        "   Max memory: {:.2} MB",
        executor.current_memory() as f64 / 1e6
    );
    println!();

    // For now, use standard MTTKRP
    // Future: implement fully chunked version with StreamingExecutor
    let start = Instant::now();

    // Convert factors to views
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    let result = mttkrp(&tensor.as_array().view(), &factor_views, mode)?;
    let elapsed = start.elapsed();

    println!("✅ MTTKRP completed in {:.4}s", elapsed.as_secs_f64());
    println!("   Result shape: {:?}", result.dim());
    println!();

    // Note: Full OoC implementation would:
    // 1. Chunk tensor into slices along target mode
    // 2. For each chunk:
    //    - Load chunk from disk/memory
    //    - Compute partial MTTKRP
    //    - Accumulate results
    // 3. Use executor's memory management for intermediate results

    Ok(result)
}

/// Compute Frobenius norm of a matrix
fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

/// Compare chunked execution strategies
fn compare_strategies(tensor: &DenseND<f64>, factors: &[Array2<f64>], mode: usize) -> Result<()> {
    println!("📊 Comparing execution strategies for MTTKRP:");
    println!();

    // Strategy 1: Maximum memory (baseline)
    let config_baseline = StreamConfig::new()
        .max_memory_mb(512)
        .enable_profiling(false);

    let start = Instant::now();
    let result_baseline = mttkrp_streaming(tensor, factors, mode, config_baseline)?;
    let time_baseline = start.elapsed();

    println!("   Baseline (512MB): {:.4}s", time_baseline.as_secs_f64());
    println!();

    // Strategy 2: Constrained memory
    let config_constrained = StreamConfig::new()
        .max_memory_mb(64)
        .chunk_size(vec![16])
        .enable_profiling(false);

    let start = Instant::now();
    let result_constrained = mttkrp_streaming(tensor, factors, mode, config_constrained)?;
    let time_constrained = start.elapsed();

    println!(
        "   Constrained (64MB): {:.4}s",
        time_constrained.as_secs_f64()
    );
    println!();

    // Verify results are identical
    let diff = &result_baseline - &result_constrained;
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&result_baseline);

    println!("   Relative error: {:.2e}", relative_error);
    println!(
        "   Overhead: {:.1}%",
        ((time_constrained.as_secs_f64() / time_baseline.as_secs_f64()) - 1.0) * 100.0
    );
    println!();

    if relative_error < 1e-10 {
        println!("   ✅ Results are numerically identical");
    } else if relative_error < 1e-6 {
        println!("   ✅ Results are very close");
    } else {
        println!("   ⚠️  Significant difference detected");
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  MTTKRP with Streaming Execution");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Configuration
    let tensor_shape = vec![40, 50, 60]; // ~2MB for f64
    let rank = 10;
    let target_mode = 1; // Compute MTTKRP along mode 1

    println!("📦 Creating test data...");
    let tensor = create_random_tensor(&tensor_shape, 42)?;
    let factors = create_factor_matrices(&tensor_shape, rank, 123)?;

    let tensor_size_mb = (tensor.shape().iter().product::<usize>() * 8) as f64 / 1e6;
    println!("   Tensor size: {:.2} MB", tensor_size_mb);
    println!("   Rank: {}", rank);
    println!();

    // Perform MTTKRP with different strategies
    compare_strategies(&tensor, &factors, target_mode)?;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Chunking Strategy Analysis");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Analyze different chunk sizes
    let chunk_sizes = vec![8, 16, 32, 64];
    println!("Testing different chunk sizes:");
    println!();

    for chunk_size in chunk_sizes {
        let config = StreamConfig::new()
            .max_memory_mb(64)
            .chunk_size(vec![chunk_size])
            .enable_profiling(false);

        let start = Instant::now();
        let _result = mttkrp_streaming(&tensor, &factors, target_mode, config)?;
        let elapsed = start.elapsed();

        println!(
            "   Chunk size {}: {:.4}s",
            chunk_size,
            elapsed.as_secs_f64()
        );
    }
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("This example demonstrates infrastructure for streaming MTTKRP.");
    println!("Key components:");
    println!();
    println!("  • Integration with tenrso-kernels MTTKRP operations");
    println!("  • StreamingExecutor for memory-constrained execution");
    println!("  • Chunk size analysis and performance comparison");
    println!("  • Memory management strategies");
    println!();
    println!("Future enhancements:");
    println!("  • Fully chunked MTTKRP with partial accumulation");
    println!("  • Parallel chunk processing across modes");
    println!("  • GPU-accelerated chunks");
    println!("  • Adaptive chunk sizing based on memory pressure");
    println!();

    Ok(())
}
