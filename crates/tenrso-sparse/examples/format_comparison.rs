//! Sparse Format Comparison Example
//!
//! This example demonstrates the different sparse formats and their use cases:
//! - COO: Coordinate format for construction
//! - CSR: Compressed Sparse Row for row operations
//! - CSC: Compressed Sparse Column for column operations
//! - BCSR: Block CSR for block-structured matrices
//! - ELL: ELLPACK for GPU-friendly operations
//! - DIA: Diagonal format for banded matrices
//!
//! Run with: cargo run --example format_comparison --features csf

use tenrso_sparse::{CooTensor, CscMatrix, CsrMatrix, DiaMatrix, EllMatrix};

fn main() -> anyhow::Result<()> {
    println!("=== TenRSo Sparse: Format Comparison Example ===\n");

    // Create a test matrix (tridiagonal for DIA demonstration) using COO
    println!("Creating a 5x5 tridiagonal test matrix...");
    let indices = vec![
        vec![0, 0],
        vec![0, 1],
        vec![1, 0],
        vec![1, 1],
        vec![1, 2],
        vec![2, 1],
        vec![2, 2],
        vec![2, 3],
        vec![3, 2],
        vec![3, 3],
        vec![3, 4],
        vec![4, 3],
        vec![4, 4],
    ];
    let values = vec![
        2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0,
    ];

    // COO: Good for construction
    println!("1. COO Format (Coordinate)");
    let coo = CooTensor::new(indices, values, vec![5, 5])?;
    println!("   Non-zeros: {}", coo.nnz());
    println!("   Use case: Matrix construction, flexible insertion");
    println!(
        "   Memory: {} indices + {} values\n",
        coo.indices().len(),
        coo.values().len()
    );

    let dense = coo.to_dense()?;
    println!("Dense representation:\n{:?}\n", dense);

    // CSR: Good for row operations
    println!("2. CSR Format (Compressed Sparse Row)");
    let csr = CsrMatrix::from_coo(&coo)?;
    println!("   Non-zeros: {}", csr.nnz());
    println!("   Row pointers: {} elements", csr.row_ptr().len());
    println!("   Use case: SpMV, row-wise operations");
    println!("   Best for: General sparse matrices\n");

    // CSC: Good for column operations
    println!("3. CSC Format (Compressed Sparse Column)");
    let csc = CscMatrix::from_coo(&coo)?;
    println!("   Non-zeros: {}", csc.nnz());
    println!("   Column pointers: {} elements", csc.col_ptr().len());
    println!("   Use case: Column-wise operations, transpose");
    println!("   Best for: QR decomposition, least squares\n");

    // Note: BCSR format available via BcsrMatrix::new() for block-structured matrices
    println!("4. BCSR Format (Block Compressed Sparse Row)");
    println!("   Use case: FEM, block-diagonal matrices");
    println!("   Best for: Matrices with block structure");
    println!("   Note: Use BcsrMatrix::new() for custom construction\n");

    // ELL: Good for GPU operations with uniform row sparsity
    println!("5. ELL Format (ELLPACK)");
    let ell = EllMatrix::from_csr(&csr);
    println!("   Max nnz per row: {}", ell.max_nnz_per_row());
    println!("   Fill efficiency: {:.1}%", ell.fill_efficiency() * 100.0);
    println!("   Use case: GPU computations, SIMD operations");
    println!("   Best for: Uniform row sparsity patterns\n");

    // DIA: Excellent for banded/diagonal matrices
    println!("6. DIA Format (Diagonal)");
    let dia = DiaMatrix::from_csr(&csr);
    let (lower, upper) = dia.bandwidth();
    println!("   Number of diagonals: {}", dia.num_diagonals());
    println!("   Bandwidth: lower={}, upper={}", lower, upper);
    println!("   Use case: PDE discretizations, banded systems");
    println!("   Best for: Tridiagonal, pentadiagonal matrices\n");

    // Performance comparison for SpMV
    println!("7. SpMV Performance Comparison");
    let x = scirs2_core::ndarray_ext::array![1.0, 2.0, 3.0, 4.0, 5.0];

    let y_csr = csr.spmv(&x.view())?;
    println!("   CSR SpMV result: {:?}", y_csr);

    let y_csc = csc.matvec(&x.view())?;
    println!("   CSC SpMV result: {:?}", y_csc);

    let y_ell = ell.spmv(&x)?;
    println!("   ELL SpMV result: {:?}", y_ell);

    let y_dia = dia.spmv(&x)?;
    println!("   DIA SpMV result: {:?}\n", y_dia);

    // Format recommendation
    println!("8. Format Selection Guidelines:");
    println!("   - Constructing/modifying → COO");
    println!("   - Row operations/SpMV → CSR");
    println!("   - Column operations → CSC");
    println!("   - Block structure (FEM) → BCSR");
    println!("   - GPU/SIMD operations → ELL");
    println!("   - Banded matrices (PDEs) → DIA");
    println!("   - Very sparse (< 0.1%) → CSF/HiCOO\n");

    println!("=== Example Complete ===");
    println!("See FORMAT_GUIDE.md for detailed format selection criteria");
    Ok(())
}
