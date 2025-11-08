//! Example demonstrating n-mode products and Tucker decomposition operations
//!
//! N-mode products are fundamental operations in Tucker decomposition and
//! tensor-tensor multiplication. This example shows various uses of n-mode
//! products including Tucker reconstruction.
//!
//! Run with: cargo run --example nmode_tucker

use scirs2_core::ndarray_ext::{Array, Array2};
use std::collections::HashMap;
use tenrso_core::DenseND;
use tenrso_kernels::{nmode_product, nmode_products_seq, tucker_operator, tucker_reconstruct};

fn main() {
    println!("=== N-Mode Product and Tucker Decomposition Example ===\n");

    // Example 1: Basic n-mode product
    println!("1. Basic N-Mode Product (Tensor-Matrix Multiplication)");
    println!("-------------------------------------------------------");

    // Create a 3D tensor (2×3×4)
    let tensor = DenseND::<f64>::from_array(Array::from_shape_fn(vec![2, 3, 4], |idx| {
        (idx[0] * 12 + idx[1] * 4 + idx[2]) as f64 + 1.0
    }));

    println!("Original tensor shape: {:?}", tensor.shape());
    println!("First slice (mode-0, index 0):");
    for i in 0..3 {
        for j in 0..4 {
            print!("{:4.0} ", tensor.view()[[0, i, j]]);
        }
        println!();
    }
    println!();

    // Apply n-mode product on mode 1 with a 5×3 matrix
    let matrix = Array2::<f64>::from_shape_fn((5, 3), |(i, j)| (i + j) as f64 + 1.0);
    println!(
        "Matrix for mode-1 product ({} × {}):",
        matrix.shape()[0],
        matrix.shape()[1]
    );
    println!("{:?}\n", matrix);

    let result = nmode_product(&tensor.view(), &matrix.view(), 1).unwrap();
    println!("Result shape after mode-1 product: {:?}", result.shape());
    println!("Expected: [2, 5, 4] (mode-1 dimension changed from 3 to 5)");
    println!();

    // Example 2: Sequential n-mode products
    println!("2. Sequential N-Mode Products");
    println!("------------------------------");

    let tensor_3d = DenseND::<f64>::from_array(Array::from_shape_fn(vec![3, 4, 5], |idx| {
        (idx[0] + idx[1] * 2 + idx[2] * 3) as f64
    }));

    let m0 = Array2::<f64>::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as f64 + 1.0);
    let m1 = Array2::<f64>::from_shape_fn((3, 4), |(i, j)| (i + j) as f64 + 1.0);
    let m2 = Array2::<f64>::from_shape_fn((4, 5), |(i, j)| (i + j * 2) as f64 + 1.0);

    println!("Original shape: {:?}", tensor_3d.shape());
    println!("Applying matrices:");
    println!("  Mode 0: {} × {}", m0.shape()[0], m0.shape()[1]);
    println!("  Mode 1: {} × {}", m1.shape()[0], m1.shape()[1]);
    println!("  Mode 2: {} × {}", m2.shape()[0], m2.shape()[1]);

    let result = nmode_products_seq(
        &tensor_3d.view(),
        &[(&m0.view(), 0), (&m1.view(), 1), (&m2.view(), 2)],
    )
    .unwrap();

    println!("\nResult shape: {:?}", result.shape());
    println!("Expected: [2, 3, 4] (all modes transformed)");
    println!();

    // Example 3: Identity property
    println!("3. Identity Property of N-Mode Product");
    println!("---------------------------------------");

    let tensor_small = DenseND::<f64>::from_array(Array::from_shape_fn(vec![3, 3, 3], |idx| {
        (idx[0] + idx[1] + idx[2]) as f64
    }));

    let identity = Array2::<f64>::eye(3);

    println!("Tensor shape: {:?}", tensor_small.shape());
    println!("Applying identity matrix (3×3) on mode 1...");

    let result_identity = nmode_product(&tensor_small.view(), &identity.view(), 1).unwrap();

    println!("Result shape: {:?}", result_identity.shape());

    // Verify the tensor is unchanged
    let max_diff = tensor_small
        .view()
        .iter()
        .zip(result_identity.iter())
        .map(|(t, r)| (t - r).abs())
        .fold(0.0, f64::max);

    println!("Maximum difference: {:.2e}", max_diff);
    println!("Tensor unchanged: {}", max_diff < 1e-10);
    println!();

    // Example 4: Tucker operator (multi-mode products)
    println!("4. Tucker Operator (Multi-Mode Products)");
    println!("-----------------------------------------");

    // Create a core tensor (2×2×2)
    let core = DenseND::<f64>::from_array(Array::from_shape_fn(vec![2, 2, 2], |idx| {
        (idx[0] * 4 + idx[1] * 2 + idx[2]) as f64 + 1.0
    }));

    println!("Core tensor shape: {:?}", core.shape());
    println!("Core values:");
    for i in 0..2 {
        println!("  Slice {}:", i);
        for j in 0..2 {
            print!("    [");
            for k in 0..2 {
                print!("{:.0} ", core.view()[[i, j, k]]);
            }
            println!("]");
        }
    }
    println!();

    // Create factor matrices for Tucker reconstruction
    let u0 = Array2::<f64>::from_shape_fn((4, 2), |(i, j)| {
        if j == 0 {
            (i + 1) as f64
        } else {
            (4 - i) as f64
        }
    });
    let u1 = Array2::<f64>::from_shape_fn((5, 2), |(i, j)| {
        if j == 0 {
            (i + 1) as f64
        } else {
            (5 - i) as f64
        }
    });
    let u2 = Array2::<f64>::from_shape_fn((6, 2), |(i, j)| {
        if j == 0 {
            (i + 1) as f64
        } else {
            (6 - i) as f64
        }
    });

    println!("Factor matrices:");
    println!(
        "  U[0]: {} × {} (transforms mode 0: 2 → 4)",
        u0.shape()[0],
        u0.shape()[1]
    );
    println!(
        "  U[1]: {} × {} (transforms mode 1: 2 → 5)",
        u1.shape()[0],
        u1.shape()[1]
    );
    println!(
        "  U[2]: {} × {} (transforms mode 2: 2 → 6)",
        u2.shape()[0],
        u2.shape()[1]
    );

    // Create factor map for tucker operator
    let mut factor_map = HashMap::new();
    factor_map.insert(0, u0.view());
    factor_map.insert(1, u1.view());
    factor_map.insert(2, u2.view());

    let reconstructed = tucker_operator(&core.view(), &factor_map).unwrap();
    println!("\nReconstructed tensor shape: {:?}", reconstructed.shape());
    println!("Expected: [4, 5, 6]");
    println!();

    // Example 5: Partial Tucker reconstruction
    println!("5. Partial Tucker Reconstruction");
    println!("---------------------------------");

    // Apply factors only to specific modes
    let mut partial_map = HashMap::new();
    partial_map.insert(0, u0.view());
    partial_map.insert(2, u2.view());

    let partial_result = tucker_operator(&core.view(), &partial_map).unwrap();
    println!("Applying factors only to modes 0 and 2");
    println!("Result shape: {:?}", partial_result.shape());
    println!("Expected: [4, 2, 6] (mode 1 unchanged)");
    println!();

    // Example 6: Tucker reconstruction using ordered interface
    println!("6. Tucker Reconstruction (Ordered Interface)");
    println!("---------------------------------------------");

    let factors = vec![u0.view(), u1.view(), u2.view()];
    let full_reconstruction = tucker_reconstruct(&core.view(), &factors).unwrap();

    println!(
        "Full reconstruction shape: {:?}",
        full_reconstruction.shape()
    );
    println!("Expected: [4, 5, 6]");

    // Verify both methods give the same result
    let max_diff = reconstructed
        .iter()
        .zip(full_reconstruction.iter())
        .map(|(r1, r2)| (r1 - r2).abs())
        .fold(0.0, f64::max);

    println!("Difference between methods: {:.2e}", max_diff);
    println!("Methods match: {}", max_diff < 1e-10);
    println!();

    // Example 7: Tucker compression and reconstruction
    println!("7. Tucker Compression Example");
    println!("------------------------------");

    // Create a larger tensor
    let large_tensor = DenseND::<f64>::from_array(Array::from_shape_fn(vec![20, 25, 30], |idx| {
        ((idx[0] + idx[1] + idx[2]) as f64).sin()
    }));

    println!("Original tensor: {:?}", large_tensor.shape());
    let original_size: usize = large_tensor.shape().iter().product();
    println!("Number of elements: {}", original_size);

    // Define compressed core size
    let core_shape = vec![8, 10, 12];
    let core_compressed =
        DenseND::<f64>::from_array(Array::from_shape_fn(core_shape.clone(), |idx| {
            ((idx[0] + idx[1] * 2 + idx[2] * 3) as f64).cos()
        }));

    // Create factor matrices
    let factors_compress = [
        Array2::<f64>::from_shape_fn((20, 8), |(i, j)| ((i + j) as f64).sin() + 1.0),
        Array2::<f64>::from_shape_fn((25, 10), |(i, j)| ((i * j) as f64).cos() + 1.0),
        Array2::<f64>::from_shape_fn((30, 12), |(i, j)| ((i + j * 2) as f64).sin() + 1.0),
    ];

    let factor_views: Vec<_> = factors_compress.iter().map(|f| f.view()).collect();
    let decompressed = tucker_reconstruct(&core_compressed.view(), &factor_views).unwrap();

    println!("\nCompressed representation:");
    println!(
        "  Core: {:?} ({} elements)",
        core_shape,
        core_shape.iter().product::<usize>()
    );
    println!(
        "  Factors: 20×8 + 25×10 + 30×12 = {} elements",
        20 * 8 + 25 * 10 + 30 * 12
    );
    let compressed_size = core_shape.iter().product::<usize>() + 20 * 8 + 25 * 10 + 30 * 12;
    println!("  Total: {} elements", compressed_size);
    println!(
        "  Compression ratio: {:.2}×",
        original_size as f64 / compressed_size as f64
    );

    println!("\nDecompressed tensor shape: {:?}", decompressed.shape());
    println!();

    // Example 8: Mode ordering in sequential products
    println!("8. Effect of Mode Ordering");
    println!("--------------------------");

    let test_tensor = DenseND::<f64>::from_array(Array::from_shape_fn(vec![3, 4, 5], |idx| {
        (idx[0] + idx[1] + idx[2]) as f64
    }));

    let test_matrices = [
        Array2::<f64>::eye(3),
        Array2::<f64>::eye(4),
        Array2::<f64>::eye(5),
    ];

    // Order 1: [0, 1, 2]
    let result1 = nmode_products_seq(
        &test_tensor.view(),
        &[
            (&test_matrices[0].view(), 0),
            (&test_matrices[1].view(), 1),
            (&test_matrices[2].view(), 2),
        ],
    )
    .unwrap();

    // Order 2: [2, 1, 0]
    let result2 = nmode_products_seq(
        &test_tensor.view(),
        &[
            (&test_matrices[2].view(), 2),
            (&test_matrices[1].view(), 1),
            (&test_matrices[0].view(), 0),
        ],
    )
    .unwrap();

    println!("Applying identity matrices in different orders:");
    println!("  Order [0, 1, 2]: shape {:?}", result1.shape());
    println!("  Order [2, 1, 0]: shape {:?}", result2.shape());

    let max_diff_order = result1
        .iter()
        .zip(result2.iter())
        .map(|(r1, r2)| (r1 - r2).abs())
        .fold(0.0, f64::max);

    println!("Difference: {:.2e}", max_diff_order);
    println!("Order doesn't affect result: {}", max_diff_order < 1e-10);

    println!("\n=== Example Complete ===");
}
