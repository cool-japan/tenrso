//! Advanced operations example for tenrso-exec
//!
//! This example demonstrates advanced usage patterns:
//! - Combining multiple operations in a pipeline
//! - Mixed einsum + element-wise + reduction workflows
//! - Custom computation patterns
//! - Performance considerations

use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::{einsum_ex, CpuExecutor, ElemOp, ReduceOp, TenrsoExecutor};

fn main() -> anyhow::Result<()> {
    println!("TenRSo Executor - Advanced Operations Example");
    println!("==============================================\n");

    // Example 1: Softmax computation using mixed operations
    println!("1. Softmax Implementation");
    println!("-------------------------");

    let logits = DenseND::from_vec(vec![2.0, 1.0, 0.1, 3.0], &[4])?;
    println!("Input logits: {:?}", logits.view());

    let mut executor = CpuExecutor::new();
    let logits_handle = TensorHandle::from_dense_auto(logits);

    // Step 1: Compute exponentials
    let exp_logits = executor.elem_op(ElemOp::Exp, &logits_handle)?;
    println!("\nExp(logits): {:?}", exp_logits.as_dense().unwrap().view());

    // Step 2: Sum all exponentials
    let sum_exp = executor.reduce(ReduceOp::Sum, &exp_logits, &[0])?;
    let sum_exp_dense = sum_exp.as_dense().unwrap();
    let sum_val = if sum_exp_dense.shape().is_empty() {
        sum_exp_dense.view()[[]]
    } else {
        sum_exp_dense.view()[[0]]
    };
    println!("Sum of exp: {:.4}", sum_val);

    // Step 3: Divide each by sum (normalized probabilities)
    // For now, we'll demonstrate the concept without actual division operator
    println!("Softmax probabilities computed (conceptually)\n");

    // Example 2: Batch matrix multiplication with normalization
    println!("2. Batch Matrix Operations");
    println!("--------------------------");

    // Create a batch of 3 matrices, each 4x4
    let batch_size = 3;
    let dim = 4;

    let matrices: Vec<DenseND<f64>> = (0..batch_size)
        .map(|i| {
            let data: Vec<f64> = (0..dim * dim)
                .map(|j| ((i * dim * dim + j) as f64) / 10.0)
                .collect();
            DenseND::from_vec(data, &[dim, dim]).unwrap()
        })
        .collect();

    println!("Created batch of {} matrices ({}x{})", batch_size, dim, dim);

    // Compute pairwise products and sum
    let mut total_sum = 0.0;
    for (i, matrix) in matrices.iter().enumerate() {
        let handle = TensorHandle::from_dense_auto(matrix.clone());

        // Square each element
        let squared = executor.elem_op(ElemOp::Sqr, &handle)?;

        // Sum all elements
        let sum_result = executor.reduce(ReduceOp::Sum, &squared, &[0, 1])?;
        let sum_dense = sum_result.as_dense().unwrap();
        let sum_val = if sum_dense.shape().is_empty() {
            sum_dense.view()[[]]
        } else {
            sum_dense.view()[[0]]
        };

        total_sum += sum_val;
        println!("  Matrix {}: Sum of squares = {:.2}", i, sum_val);
    }
    println!("Total sum across batch: {:.2}\n", total_sum);

    // Example 3: Attention-like computation (simplified)
    println!("3. Attention-like Computation");
    println!("------------------------------");

    // Query, Key, Value matrices
    let query = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let key = DenseND::from_vec(vec![0.5, 1.5, 2.5, 3.5], &[2, 2])?;
    let value = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;

    println!("Query matrix:\n{:?}", query.view());
    println!("\nKey matrix:\n{:?}", key.view());
    println!("\nValue matrix:\n{:?}", value.view());

    // Compute attention scores: Q @ K^T
    // For simplicity, we'll compute Q @ K (without transpose for now)
    let q_handle = TensorHandle::from_dense_auto(query);
    let k_handle = TensorHandle::from_dense_auto(key);
    let v_handle = TensorHandle::from_dense_auto(value);

    let scores = einsum_ex::<f64>("ij,jk->ik")
        .inputs(&[q_handle, k_handle])
        .run()?;

    println!(
        "\nAttention scores (Q @ K):\n{:?}",
        scores.as_dense().unwrap().view()
    );

    // Apply softmax-like transformation (simplified)
    let scores_exp = executor.elem_op(ElemOp::Exp, &scores)?;
    println!(
        "\nExp(scores):\n{:?}",
        scores_exp.as_dense().unwrap().view()
    );

    // Compute final attention output: scores @ V
    let attention_output = einsum_ex::<f64>("ij,jk->ik")
        .inputs(&[scores_exp, v_handle])
        .run()?;

    println!(
        "\nAttention output:\n{:?}\n",
        attention_output.as_dense().unwrap().view()
    );

    // Example 4: Trigonometric transformations
    println!("4. Trigonometric Features");
    println!("-------------------------");

    let angles = DenseND::from_vec(
        vec![
            0.0,
            std::f64::consts::PI / 6.0,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 3.0,
            std::f64::consts::PI / 2.0,
        ],
        &[5],
    )?;

    println!("Angles (in radians): {:?}", angles.view());

    let angles_handle = TensorHandle::from_dense_auto(angles);

    // Compute sin and cos features
    let sin_features = executor.elem_op(ElemOp::Sin, &angles_handle)?;
    let cos_features = executor.elem_op(ElemOp::Cos, &angles_handle)?;

    println!(
        "\nSin features: {:?}",
        sin_features.as_dense().unwrap().view()
    );
    println!(
        "Cos features: {:?}",
        cos_features.as_dense().unwrap().view()
    );

    // Example 5: Statistical moments
    println!("\n5. Statistical Moments");
    println!("----------------------");

    let data = DenseND::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    )?;

    println!("Data matrix (3x4):\n{:?}", data.view());

    let data_handle = TensorHandle::from_dense_auto(data);

    // Compute mean along different axes
    let mean_axis0 = executor.reduce(ReduceOp::Mean, &data_handle, &[0])?;
    let mean_axis1 = executor.reduce(ReduceOp::Mean, &data_handle, &[1])?;

    println!(
        "\nMean along axis 0: {:?}",
        mean_axis0.as_dense().unwrap().view()
    );
    println!(
        "Mean along axis 1: {:?}",
        mean_axis1.as_dense().unwrap().view()
    );

    // Compute sum of squares (for variance calculation)
    let squared_data = executor.elem_op(ElemOp::Sqr, &data_handle)?;
    let sum_sq = executor.reduce(ReduceOp::Sum, &squared_data, &[0, 1])?;

    let sum_sq_val = {
        let sum_sq_dense = sum_sq.as_dense().unwrap();
        if sum_sq_dense.shape().is_empty() {
            sum_sq_dense.view()[[]]
        } else {
            sum_sq_dense.view()[[0]]
        }
    };

    println!("Sum of squares: {:.2}", sum_sq_val);

    // Example 6: Pipeline of operations
    println!("\n6. Operation Pipeline");
    println!("---------------------");

    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let input_handle = TensorHandle::from_dense_auto(input);

    println!(
        "Starting value:\n{:?}",
        input_handle.as_dense().unwrap().view()
    );

    // Pipeline: Square -> Sqrt -> Log -> Exp (should roughly return to original)
    let step1 = executor.elem_op(ElemOp::Sqr, &input_handle)?;
    println!("\nAfter square:\n{:?}", step1.as_dense().unwrap().view());

    let step2 = executor.elem_op(ElemOp::Sqrt, &step1)?;
    println!("After sqrt:\n{:?}", step2.as_dense().unwrap().view());

    let step3 = executor.elem_op(ElemOp::Log, &step2)?;
    println!("After log:\n{:?}", step3.as_dense().unwrap().view());

    let step4 = executor.elem_op(ElemOp::Exp, &step3)?;
    println!(
        "After exp (back to ~original):\n{:?}",
        step4.as_dense().unwrap().view()
    );

    println!("\n✓ All advanced examples completed successfully!");
    println!("\nThese examples demonstrate how to:");
    println!("  • Build complex computational pipelines");
    println!("  • Combine einsum, element-wise, and reduction operations");
    println!("  • Implement higher-level algorithms (softmax, attention, etc.)");
    println!("  • Work with batched data and multiple tensors");

    Ok(())
}
