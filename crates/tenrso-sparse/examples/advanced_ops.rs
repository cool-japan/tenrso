//! Advanced Sparse Operations Example
//!
//! This example demonstrates advanced sparse matrix operations including:
//! - Element-wise mathematical operations
//! - Sparse pattern analysis
//! - Structural operations (stacking, triangular extraction)
//! - Norm calculations
//! - Reductions
//!
//! Run with: cargo run --example advanced_ops --features csf

use tenrso_sparse::{CooTensor, CsrMatrix};

fn main() -> anyhow::Result<()> {
    println!("=== TenRSo Sparse: Advanced Operations Example ===\n");

    // Create test matrices
    let indices = vec![
        vec![0, 0],
        vec![0, 2],
        vec![1, 1],
        vec![2, 0],
        vec![2, 2],
        vec![2, 3],
        vec![3, 1],
        vec![3, 3],
    ];
    let values = vec![4.0, -3.0, 5.0, -2.0, 6.0, -1.0, 7.0, 8.0];
    let coo = CooTensor::new(indices, values, vec![4, 4])?;
    let csr = CsrMatrix::from_coo(&coo)?;

    println!("Test matrix (4x4, {} non-zeros):", csr.nnz());
    println!("{:?}\n", csr.to_dense()?);

    // 1. Element-wise Mathematical Operations
    println!("1. Element-wise Mathematical Operations");
    use tenrso_sparse::ops::*;

    let abs_matrix = sparse_abs_csr(&csr)?;
    println!("   Absolute values: {:?}", abs_matrix.values());

    let sqrt_abs = sparse_sqrt_csr(&abs_matrix)?;
    println!("   Square root: {:?}", sqrt_abs.values());

    let rounded = sparse_round_csr(&sqrt_abs)?;
    println!("   Rounded: {:?}", rounded.values());

    let clipped = sparse_clip_csr(&csr, -2.0, 5.0)?;
    println!("   Clipped [-2, 5]: {:?}\n", clipped.values());

    // 2. Sparse Pattern Analysis
    println!("2. Sparse Pattern Analysis");
    use tenrso_sparse::patterns::*;

    let (lower_bw, upper_bw) = bandwidth(&csr);
    println!("   Bandwidth: lower={}, upper={}", lower_bw, upper_bw);

    let is_sym = is_structurally_symmetric(&csr);
    println!("   Structurally symmetric: {}", is_sym);

    let is_diag_dom = is_row_diagonally_dominant(&csr);
    println!("   Row diagonally dominant: {}", is_diag_dom);

    let pattern = analyze_pattern(&csr, Some(0.0));
    println!("   Pattern analysis:");
    println!("     - Min nnz/row: {}", pattern.min_nnz_per_row);
    println!("     - Max nnz/row: {}", pattern.max_nnz_per_row);
    println!("     - Avg nnz/row: {:.2}", pattern.avg_nnz_per_row);
    println!(
        "     - Bandwidth: {}+{}\n",
        pattern.lower_bandwidth, pattern.upper_bandwidth
    );

    // 3. Structural Operations
    println!("3. Structural Operations");
    use tenrso_sparse::structural::*;

    let upper_tri = triu_csr(&csr, 0)?;
    println!("   Upper triangular (k=0):");
    println!("   {:?}", upper_tri.to_dense()?);

    let lower_tri = tril_csr(&csr, 0)?;
    println!("   Lower triangular (k=0):");
    println!("   {:?}", lower_tri.to_dense()?);

    let diagonal = diagonal_csr(&csr, 0);
    println!("   Diagonal extraction: {:?}\n", diagonal);

    // 4. Norm Calculations
    println!("4. Norm Calculations");
    use tenrso_sparse::norms::*;

    let frob_norm = frobenius_norm_csr(&csr);
    println!("   Frobenius norm: {:.4}", frob_norm);

    let l1_norm = l1_norm_csr(&csr);
    println!("   L1 norm: {:.4}", l1_norm);

    let inf_norm = infinity_norm_csr(&csr);
    println!("   Infinity norm: {:.4}", inf_norm);

    let mat_1_norm = matrix_1_norm_csr(&csr);
    println!("   Matrix 1-norm (max col sum): {:.4}\n", mat_1_norm);

    // 5. Reductions
    println!("5. Reduction Operations");
    use tenrso_sparse::reductions::*;

    let sum_val = sum(&coo);
    println!("   Sum of all elements: {:.2}", sum_val);

    let max_val = max(&coo)?;
    println!("   Maximum element: {:.2}", max_val);

    let mean_val = mean(&coo)?;
    println!("   Mean (including zeros): {:.4}", mean_val);

    let sum_axis0 = sum_axis(&coo, 0)?;
    println!(
        "   Sum along axis 0 (column sums): {:?}\n",
        sum_axis0.values()
    );

    // 6. Binary Operations with Two Matrices
    println!("6. Binary Sparse Operations");
    let indices2 = vec![
        vec![0, 0],
        vec![0, 2],
        vec![1, 1],
        vec![2, 0],
        vec![2, 2],
        vec![2, 3],
        vec![3, 1],
        vec![3, 3],
    ];
    let values2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let coo2 = CooTensor::new(indices2, values2, vec![4, 4])?;
    let csr2 = CsrMatrix::from_coo(&coo2)?;

    let added = sparse_add_csr(&csr, &csr2, 1.0, 1.0)?;
    println!("   Matrix addition (A + B):");
    println!("   Non-zeros: {} → {}", csr.nnz(), added.nnz());

    let multiplied = sparse_multiply_csr(&csr, &csr2)?;
    println!("   Element-wise multiply (Hadamard product):");
    println!("   Non-zeros: {}\n", multiplied.nnz());

    // 7. Advanced Math Operations
    println!("7. Advanced Mathematical Operations");
    let trig_indices = vec![vec![0, 1], vec![0, 2], vec![2, 0], vec![2, 2]];
    let trig_values = vec![0.5, 1.0, 1.5, 2.0];
    let trig_coo = CooTensor::new(trig_indices, trig_values, vec![3, 3])?;
    let trig_test = CsrMatrix::from_coo(&trig_coo)?;

    let sin_result = sparse_sin_csr(&trig_test)?;
    println!("   sin(A): {:?}", sin_result.values());

    let exp_result = sparse_exp_csr(&trig_test)?;
    println!("   exp(A): {:?}", exp_result.values());

    let hypot_result = sparse_hypot_csr(&trig_test, &trig_test)?;
    println!("   hypot(A, A): {:?}\n", hypot_result.values());

    println!("=== Example Complete ===");
    println!("Explore the full API in the documentation");
    Ok(())
}
