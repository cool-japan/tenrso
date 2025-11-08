//! Example demonstrating MTTKRP (Matricized Tensor Times Khatri-Rao Product)
//!
//! MTTKRP is the core computational kernel in CP-ALS tensor decomposition.
//! This example shows how MTTKRP is used to update factor matrices during
//! CP decomposition iterations.
//!
//! Run with: cargo run --example mttkrp_cp --features parallel

use scirs2_core::ndarray_ext::{Array, Array2};
use tenrso_core::DenseND;
use tenrso_kernels::{cp_reconstruct, mttkrp, mttkrp_blocked, mttkrp_blocked_parallel};

fn main() {
    println!("=== MTTKRP and CP Decomposition Example ===\n");

    // Example 1: Basic MTTKRP operation
    println!("1. Basic MTTKRP Operation");
    println!("-------------------------");

    // Create a small 3D tensor (3×4×5)
    let shape = vec![3, 4, 5];
    let tensor = DenseND::<f64>::from_array(Array::from_shape_fn(shape.clone(), |idx| {
        (idx[0] + idx[1] * 2 + idx[2] * 3) as f64
    }));

    println!("Tensor shape: {:?}", tensor.shape());

    // Create factor matrices for rank-2 decomposition
    let rank = 2;
    let u0 = Array2::<f64>::from_shape_fn((3, rank), |(i, j)| (i + j) as f64 + 1.0);
    let u1 = Array2::<f64>::from_shape_fn((4, rank), |(i, j)| (i + j) as f64 + 1.0);
    let u2 = Array2::<f64>::from_shape_fn((5, rank), |(i, j)| (i + j) as f64 + 1.0);

    println!("Factor matrices:");
    println!("  U[0]: {} × {}", u0.shape()[0], u0.shape()[1]);
    println!("  U[1]: {} × {}", u1.shape()[0], u1.shape()[1]);
    println!("  U[2]: {} × {}", u2.shape()[0], u2.shape()[1]);
    println!();

    // Compute MTTKRP for mode 0
    let mode = 0;
    let result = mttkrp(&tensor.view(), &[u0.view(), u1.view(), u2.view()], mode).unwrap();

    println!("MTTKRP result for mode {}: {:?}", mode, result.shape());
    println!("Expected shape: [{}, {}]", shape[mode], rank);
    println!("Result:\n{:?}\n", result);

    // Example 2: MTTKRP in CP-ALS iteration
    println!("2. CP-ALS Iteration (Simplified)");
    println!("---------------------------------");

    // Create a simple rank-2 tensor
    let true_rank = 2;
    let factors_true: Vec<Array2<f64>> = vec![
        Array2::from_shape_fn((3, true_rank), |(i, j)| {
            if j == 0 {
                (i + 1) as f64
            } else {
                (3 - i) as f64
            }
        }),
        Array2::from_shape_fn((4, true_rank), |(i, j)| {
            if j == 0 {
                (i + 1) as f64
            } else {
                (4 - i) as f64
            }
        }),
        Array2::from_shape_fn((5, true_rank), |(i, j)| {
            if j == 0 {
                (i + 1) as f64
            } else {
                (5 - i) as f64
            }
        }),
    ];

    // Reconstruct the tensor from true factors
    let factor_views: Vec<_> = factors_true.iter().map(|f| f.view()).collect();
    let true_tensor = cp_reconstruct(&factor_views, None).unwrap();

    println!(
        "Generated rank-{} tensor with shape: {:?}",
        true_rank,
        true_tensor.shape()
    );

    // Initialize random factor matrices for decomposition
    let mut factors = [
        Array2::<f64>::from_shape_fn((3, true_rank), |(i, j)| ((i + j) as f64).sin() + 1.5),
        Array2::<f64>::from_shape_fn((4, true_rank), |(i, j)| ((i * j) as f64).cos() + 1.5),
        Array2::<f64>::from_shape_fn((5, true_rank), |(i, j)| ((i + j * 2) as f64).sin() + 1.5),
    ];

    println!("\nRunning 5 CP-ALS iterations...");

    for iter in 0..5 {
        // Update each factor matrix using MTTKRP
        for mode in 0..3 {
            let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
            let mttkrp_result = mttkrp(&true_tensor.view(), &factor_views, mode).unwrap();

            // In real CP-ALS, we would solve a least-squares problem here
            // For this example, we just use the MTTKRP result directly
            factors[mode] = mttkrp_result;

            // Normalize columns
            for col in 0..true_rank {
                let norm: f64 = factors[mode]
                    .column(col)
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f64>()
                    .sqrt();
                if norm > 1e-10 {
                    for row in 0..factors[mode].shape()[0] {
                        factors[mode][[row, col]] /= norm;
                    }
                }
            }
        }

        // Compute reconstruction error
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let reconstructed = cp_reconstruct(&factor_views, None).unwrap();

        let error: f64 = true_tensor
            .iter()
            .zip(reconstructed.iter())
            .map(|(t, r)| (t - r).powi(2))
            .sum::<f64>()
            .sqrt();

        println!(
            "  Iteration {}: Reconstruction error = {:.6}",
            iter + 1,
            error
        );
    }
    println!();

    // Example 3: Blocked MTTKRP for larger tensors
    println!("3. Blocked MTTKRP (Cache-Optimized)");
    println!("------------------------------------");

    let large_shape = vec![40, 40, 40];
    let large_rank = 10;
    let large_tensor =
        DenseND::<f64>::from_array(Array::from_shape_fn(large_shape.clone(), |idx| {
            ((idx[0] + idx[1] + idx[2]) as f64).sin()
        }));

    let factors_large: Vec<Array2<f64>> = (0..3)
        .map(|i| {
            Array2::<f64>::from_shape_fn((large_shape[i], large_rank), |(j, k)| {
                ((i + j + k) as f64).cos() + 1.0
            })
        })
        .collect();

    let factor_views: Vec<_> = factors_large.iter().map(|f| f.view()).collect();

    println!("Tensor shape: {:?}", large_tensor.shape());
    println!("Rank: {}", large_rank);

    // Standard MTTKRP
    let start = std::time::Instant::now();
    let standard_result = mttkrp(&large_tensor.view(), &factor_views, 1).unwrap();
    let standard_time = start.elapsed();

    // Blocked MTTKRP with tile size 16
    let tile_size = 16;
    let start = std::time::Instant::now();
    let blocked_result = mttkrp_blocked(&large_tensor.view(), &factor_views, 1, tile_size).unwrap();
    let blocked_time = start.elapsed();

    println!("\nStandard MTTKRP time: {:?}", standard_time);
    println!(
        "Blocked MTTKRP time (tile={}): {:?}",
        tile_size, blocked_time
    );

    // Verify results match
    let max_diff = standard_result
        .iter()
        .zip(blocked_result.iter())
        .map(|(s, b)| (s - b).abs())
        .fold(0.0, f64::max);
    println!("Maximum difference: {:.2e}", max_diff);
    println!("Results match: {}", max_diff < 1e-10);

    // Parallel blocked MTTKRP
    #[cfg(feature = "parallel")]
    {
        let start = std::time::Instant::now();
        let parallel_result =
            mttkrp_blocked_parallel(&large_tensor.view(), &factor_views, 1, tile_size).unwrap();
        let parallel_time = start.elapsed();

        println!("\nBlocked parallel MTTKRP time: {:?}", parallel_time);
        println!(
            "Speedup over standard: {:.2}x",
            standard_time.as_secs_f64() / parallel_time.as_secs_f64()
        );

        let max_diff_parallel = standard_result
            .iter()
            .zip(parallel_result.iter())
            .map(|(s, p)| (s - p).abs())
            .fold(0.0, f64::max);
        println!("Maximum difference (parallel): {:.2e}", max_diff_parallel);
        println!("Results match: {}", max_diff_parallel < 1e-10);
    }
    println!();

    // Example 4: Understanding MTTKRP output
    println!("4. Understanding MTTKRP Output");
    println!("-------------------------------");

    let small_shape = vec![2, 3, 4];
    let small_rank = 2;
    let small_tensor = DenseND::<f64>::ones(&small_shape);

    let small_factors: Vec<Array2<f64>> = vec![
        Array2::ones((2, small_rank)),
        Array2::ones((3, small_rank)),
        Array2::ones((4, small_rank)),
    ];

    let small_views: Vec<_> = small_factors.iter().map(|f| f.view()).collect();

    for mode in 0..3 {
        let result = mttkrp(&small_tensor.view(), &small_views, mode).unwrap();
        let product_size: usize = small_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != mode)
            .map(|(_, &s)| s)
            .product();

        println!("Mode {}: shape {:?}", mode, result.shape());
        println!(
            "  Expected values: {} (product of other mode sizes)",
            product_size
        );
        println!("  Actual first element: {:.1}", result[[0, 0]]);
    }

    println!("\n=== Example Complete ===");
}
