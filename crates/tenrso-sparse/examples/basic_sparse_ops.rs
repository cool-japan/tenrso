//! Basic Sparse Matrix Operations Example
//!
//! This example demonstrates fundamental sparse matrix operations including:
//! - Creating sparse matrices in different formats
//! - Converting between formats
//! - Sparse matrix-vector multiplication (SpMV)
//! - Sparse matrix-matrix multiplication (SpMM)
//!
//! Run with: cargo run --example basic_sparse_ops --features csf

use scirs2_core::ndarray_ext::array;
use tenrso_sparse::{CooTensor, CsrMatrix};

fn main() -> anyhow::Result<()> {
    println!("=== TenRSo Sparse: Basic Operations Example ===\n");

    // 1. Create a sparse matrix using COO format
    println!("1. Creating a 5x5 sparse matrix with ~20% density...");
    let indices = vec![
        vec![0, 0],
        vec![0, 2],
        vec![1, 1],
        vec![2, 0],
        vec![2, 3],
        vec![3, 4],
        vec![4, 2],
    ];
    let values = vec![5.0, 3.0, 8.0, 2.0, 6.0, 1.0, 4.0];
    let shape = vec![5, 5];

    let coo = CooTensor::new(indices, values, shape)?;
    println!(
        "   COO tensor: {} non-zeros, density: {:.1}%\n",
        coo.nnz(),
        coo.density() * 100.0
    );

    // 2. Convert to CSR for efficient row operations
    println!("2. Converting to CSR format for efficient operations...");
    let csr = CsrMatrix::from_coo(&coo)?;
    println!(
        "   CSR matrix: {}x{}, {} non-zeros\n",
        csr.nrows(),
        csr.ncols(),
        csr.nnz()
    );

    // 3. Sparse Matrix-Vector Multiplication
    println!("3. Performing SpMV (y = A * x)...");
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = csr.spmv(&x.view())?;
    println!("   Input vector:  {:?}", x);
    println!("   Output vector: {:?}\n", y);

    // 4. Sparse Matrix-Matrix Multiplication
    println!("4. Performing SpMM (C = A * B) with dense B...");
    let b = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0],];
    let c = csr.spmm(&b.view())?;
    println!("   Result shape: {}x{}", c.nrows(), c.ncols());
    println!("   Result:\n{:?}\n", c);

    // 5. Element-wise operations
    println!("5. Element-wise operations...");
    use tenrso_sparse::ops::{sparse_abs_csr, sparse_scale_csr, sparse_square_csr};

    let scaled = sparse_scale_csr(&csr, 2.0)?;
    println!("   Scaled by 2.0: {} non-zeros", scaled.nnz());

    let absolute = sparse_abs_csr(&csr)?;
    println!("   Absolute values: {} non-zeros", absolute.nnz());

    let squared = sparse_square_csr(&csr)?;
    println!("   Squared: {} non-zeros\n", squared.nnz());

    // 6. Convert to dense for visualization
    println!("6. Converting to dense for visualization...");
    let dense = csr.to_dense()?;
    println!("   Dense representation:");
    println!("{:?}\n", dense);

    println!("=== Example Complete ===");
    Ok(())
}
