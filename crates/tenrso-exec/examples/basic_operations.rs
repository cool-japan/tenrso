//! Basic operations example for tenrso-exec
//!
//! This example demonstrates:
//! - Creating tensors
//! - Using the einsum_ex builder API
//! - Element-wise operations
//! - Reduction operations
//! - Memory pooling and statistics

use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::{einsum_ex, CpuExecutor, ElemOp, ExecHints, ReduceOp, TenrsoExecutor};

fn main() -> anyhow::Result<()> {
    println!("TenRSo Executor - Basic Operations Example");
    println!("==========================================\n");

    // Example 1: Matrix multiplication using einsum
    println!("1. Matrix Multiplication (einsum)");
    println!("----------------------------------");

    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let b = DenseND::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])?;

    println!("Matrix A (2x3):");
    println!("{:?}", a.view());
    println!("\nMatrix B (3x2):");
    println!("{:?}", b.view());

    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);

    let result = einsum_ex::<f64>("ij,jk->ik")
        .inputs(&[handle_a, handle_b])
        .run()?;

    println!("\nResult C = A @ B (2x2):");
    println!("{:?}\n", result.as_dense().unwrap().view());

    // Example 2: Three-tensor contraction
    println!("2. Three-Tensor Contraction");
    println!("----------------------------");

    let x = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let y = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    let z = DenseND::from_vec(vec![9.0, 10.0, 11.0, 12.0], &[2, 2])?;

    let handle_x = TensorHandle::from_dense_auto(x);
    let handle_y = TensorHandle::from_dense_auto(y);
    let handle_z = TensorHandle::from_dense_auto(z);

    let result = einsum_ex::<f64>("ij,jk,kl->il")
        .inputs(&[handle_x, handle_y, handle_z])
        .hints(&ExecHints::default())
        .run()?;

    println!("Result of X @ Y @ Z:");
    println!("{:?}\n", result.as_dense().unwrap().view());

    // Example 3: Element-wise operations
    println!("3. Element-wise Operations");
    println!("--------------------------");

    let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    println!("Original tensor:");
    println!("{:?}", tensor.view());

    let handle = TensorHandle::from_dense_auto(tensor);
    let mut executor = CpuExecutor::new();

    // Negation
    let neg_result = executor.elem_op(ElemOp::Neg, &handle)?;
    println!("\nNegation:");
    println!("{:?}", neg_result.as_dense().unwrap().view());

    // Absolute value
    let abs_result = executor.elem_op(ElemOp::Abs, &neg_result)?;
    println!("\nAbsolute value:");
    println!("{:?}", abs_result.as_dense().unwrap().view());

    // Exponential
    let exp_tensor = DenseND::from_vec(vec![0.0, 1.0, 2.0], &[3])?;
    let exp_handle = TensorHandle::from_dense_auto(exp_tensor);
    let exp_result = executor.elem_op(ElemOp::Exp, &exp_handle)?;
    println!("\nExponential of [0, 1, 2]:");
    println!("{:?}", exp_result.as_dense().unwrap().view());

    // Example 4: Reduction operations
    println!("\n4. Reduction Operations");
    println!("-----------------------");

    let matrix = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    println!("Matrix (2x3):");
    println!("{:?}", matrix.view());

    let matrix_handle = TensorHandle::from_dense_auto(matrix);

    // Sum along axis 0 (rows)
    let sum_axis0 = executor.reduce(ReduceOp::Sum, &matrix_handle, &[0])?;
    println!("\nSum along axis 0:");
    println!("{:?}", sum_axis0.as_dense().unwrap().view());

    // Sum along axis 1 (columns)
    let sum_axis1 = executor.reduce(ReduceOp::Sum, &matrix_handle, &[1])?;
    println!("\nSum along axis 1:");
    println!("{:?}", sum_axis1.as_dense().unwrap().view());

    // Mean along axis 0
    let mean_axis0 = executor.reduce(ReduceOp::Mean, &matrix_handle, &[0])?;
    println!("\nMean along axis 0:");
    println!("{:?}", mean_axis0.as_dense().unwrap().view());

    // Example 5: Memory pool statistics
    println!("\n5. Memory Pool Statistics");
    println!("-------------------------");

    let executor = CpuExecutor::with_threads(4);
    println!("Created executor with 4 threads");

    // Perform some operations
    for i in 0..5 {
        let a = DenseND::from_vec(vec![1.0; 100], &[10, 10])?;
        let b = DenseND::from_vec(vec![2.0; 100], &[10, 10])?;
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);

        let _result = einsum_ex::<f64>("ij,jk->ik")
            .inputs(&[handle_a, handle_b])
            .run()?;

        if i % 2 == 1 {
            let (hits, misses, hit_rate) = executor.pool_stats();
            println!(
                "After iteration {}: hits={}, misses={}, hit_rate={:.2}%",
                i + 1,
                hits,
                misses,
                hit_rate * 100.0
            );
        }
    }

    let (hits, misses, hit_rate) = executor.pool_stats();
    println!(
        "\nFinal pool statistics: hits={}, misses={}, hit_rate={:.2}%",
        hits,
        misses,
        hit_rate * 100.0
    );

    println!("\n✓ All examples completed successfully!");

    Ok(())
}
