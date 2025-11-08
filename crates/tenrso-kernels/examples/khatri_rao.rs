//! Example demonstrating Khatri-Rao product operations
//!
//! The Khatri-Rao product is a column-wise Kronecker product used extensively
//! in tensor decompositions, particularly in CP-ALS algorithms.
//!
//! Run with: cargo run --example khatri_rao --features parallel

use scirs2_core::ndarray_ext::{array, Array2};
use tenrso_kernels::{khatri_rao, khatri_rao_parallel};

fn main() {
    println!("=== Khatri-Rao Product Example ===\n");

    // Example 1: Basic Khatri-Rao product
    println!("1. Basic Khatri-Rao Product");
    println!("---------------------------");

    let a = array![[1.0, 2.0], [3.0, 4.0],];

    let b = array![[5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];

    println!("Matrix A (2×2):");
    println!("{:?}\n", a);

    println!("Matrix B (3×2):");
    println!("{:?}\n", b);

    let result = khatri_rao(&a.view(), &b.view());

    println!("Khatri-Rao Product A ⊙ B (6×2):");
    println!("{:?}\n", result);
    println!("Expected structure: Each column is the Kronecker product of corresponding columns");
    println!();

    // Example 2: Verify column-wise Kronecker structure
    println!("2. Verifying Column-wise Structure");
    println!("-----------------------------------");

    // First column: a[:,0] ⊗ b[:,0]
    println!("Column 0:");
    println!("  a[0,0] * b[:,0] = 1.0 * [5, 7, 9] = [5, 7, 9]");
    println!("  a[1,0] * b[:,0] = 3.0 * [5, 7, 9] = [15, 21, 27]");
    println!("  Actual: {:?}\n", result.column(0).to_vec());

    // Second column: a[:,1] ⊗ b[:,1]
    println!("Column 1:");
    println!("  a[0,1] * b[:,1] = 2.0 * [6, 8, 10] = [12, 16, 20]");
    println!("  a[1,1] * b[:,1] = 4.0 * [6, 8, 10] = [24, 32, 40]");
    println!("  Actual: {:?}\n", result.column(1).to_vec());

    // Example 3: Use case in tensor decomposition
    println!("3. Application: Factor Matrices in CP Decomposition");
    println!("--------------------------------------------------");

    let rank = 5;
    let factor_a =
        Array2::<f64>::from_shape_fn((10, rank), |(i, j)| (i as f64 + 1.0) * (j as f64 + 1.0));
    let factor_b = Array2::<f64>::from_shape_fn((8, rank), |(i, j)| {
        (i as f64 * 0.5 + 1.0) * (j as f64 + 1.0)
    });

    println!(
        "Factor matrix A: {} × {}",
        factor_a.shape()[0],
        factor_a.shape()[1]
    );
    println!(
        "Factor matrix B: {} × {}",
        factor_b.shape()[0],
        factor_b.shape()[1]
    );

    let kr_product = khatri_rao(&factor_a.view(), &factor_b.view());
    println!(
        "Khatri-Rao product: {} × {}",
        kr_product.shape()[0],
        kr_product.shape()[1]
    );
    println!("Expected dimensions: {} × {}", 10 * 8, rank);
    println!();

    // Example 4: Parallel computation
    #[cfg(feature = "parallel")]
    {
        println!("4. Parallel Khatri-Rao Product");
        println!("-------------------------------");

        let large_a = Array2::<f64>::from_shape_fn((100, 20), |(i, j)| ((i + j) as f64).sin());
        let large_b = Array2::<f64>::from_shape_fn((150, 20), |(i, j)| ((i * j) as f64).cos());

        println!(
            "Large matrix A: {} × {}",
            large_a.shape()[0],
            large_a.shape()[1]
        );
        println!(
            "Large matrix B: {} × {}",
            large_b.shape()[0],
            large_b.shape()[1]
        );

        // Serial computation
        let start = std::time::Instant::now();
        let serial_result = khatri_rao(&large_a.view(), &large_b.view());
        let serial_time = start.elapsed();

        // Parallel computation
        let start = std::time::Instant::now();
        let parallel_result = khatri_rao_parallel(&large_a.view(), &large_b.view());
        let parallel_time = start.elapsed();

        println!(
            "Result dimensions: {} × {}",
            serial_result.shape()[0],
            serial_result.shape()[1]
        );
        println!("Serial time: {:?}", serial_time);
        println!("Parallel time: {:?}", parallel_time);
        println!(
            "Speedup: {:.2}x",
            serial_time.as_secs_f64() / parallel_time.as_secs_f64()
        );

        // Verify results match
        let max_diff = serial_result
            .iter()
            .zip(parallel_result.iter())
            .map(|(s, p)| (s - p).abs())
            .fold(0.0, f64::max);
        println!("Maximum difference: {:.2e}", max_diff);
        println!("Results match: {}", max_diff < 1e-10);
        println!();
    }

    // Example 5: Mathematical properties
    println!("5. Mathematical Properties");
    println!("--------------------------");

    let x = array![[1.0], [2.0]];
    let y = array![[3.0], [4.0], [5.0]];

    let kr = khatri_rao(&x.view(), &y.view());
    println!("For single-column matrices:");
    println!("A = {:?}", x.column(0).to_vec());
    println!("B = {:?}", y.column(0).to_vec());
    println!("A ⊙ B = {:?}", kr.column(0).to_vec());
    println!("This is equivalent to the standard Kronecker product for vectors");
    println!();

    // Example 6: Identity property
    println!("6. Property: Identity Matrix");
    println!("-----------------------------");

    let id = Array2::<f64>::eye(3);
    let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let kr_id = khatri_rao(&id.view(), &m.view());
    println!("I ⊙ M where I is identity (3×3) and M is (2×3):");
    println!("Result shape: {:?}", kr_id.shape());
    println!("Result:\n{:?}", kr_id);
    println!();

    println!("=== Example Complete ===");
}
