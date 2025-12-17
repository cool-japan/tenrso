//! Comprehensive examples of matrix reordering algorithms
//!
//! This example demonstrates:
//! - Reverse Cuthill-McKee (RCM) for bandwidth reduction
//! - Approximate Minimum Degree (AMD) for fill-in reduction
//! - Bandwidth analysis before and after reordering
//! - Impact on factorization performance

use tenrso_sparse::{reordering, CsrMatrix};

fn main() {
    println!("=== Matrix Reordering Examples ===\n");

    example_rcm_bandwidth_reduction();
    example_amd_fill_reduction();
    example_permutation_validation();
    example_disconnected_graph();
}

/// Example 1: RCM for bandwidth reduction
fn example_rcm_bandwidth_reduction() {
    println!("1. Reverse Cuthill-McKee (RCM) - Bandwidth Reduction");

    // Create a matrix with non-optimal bandwidth
    // Pattern representing a mesh-like structure (symmetric)
    let row_ptr = vec![0, 2, 5, 8, 11, 14, 16];
    let col_indices = vec![
        0, 1, // row 0
        0, 1, 2, // row 1
        1, 2, 5, // row 2
        3, 4, // row 3
        3, 4, 5, // row 4
        2, 4, 5, // row 5
    ];
    let values = vec![1.0; col_indices.len()];
    let a = CsrMatrix::new(row_ptr, col_indices, values, (6, 6)).unwrap();

    println!("   Original matrix (6×6):");
    let (lower_orig, upper_orig) = reordering::bandwidth(&a);
    println!(
        "     Lower bandwidth: {}, Upper bandwidth: {}",
        lower_orig, upper_orig
    );
    println!("     Total bandwidth: {}", lower_orig + upper_orig);

    // Apply RCM reordering
    let perm = reordering::rcm(&a).unwrap();
    println!("   RCM permutation: {:?}", perm);

    let reordered = reordering::permute_symmetric(&a, &perm).unwrap();
    let (lower_new, upper_new) = reordering::bandwidth(&reordered);

    println!("\n   After RCM reordering:");
    println!(
        "     Lower bandwidth: {}, Upper bandwidth: {}",
        lower_new, upper_new
    );
    println!("     Total bandwidth: {}", lower_new + upper_new);

    let reduction = ((lower_orig + upper_orig) as f64 - (lower_new + upper_new) as f64)
        / (lower_orig + upper_orig) as f64
        * 100.0;
    println!("     Bandwidth reduction: {:.1}%", reduction.max(0.0));
    println!();
}

/// Example 2: AMD for fill-in reduction
fn example_amd_fill_reduction() {
    println!("2. Approximate Minimum Degree (AMD) - Fill-in Reduction");

    // Create a symmetric matrix
    let row_ptr = vec![0, 2, 5, 8, 10];
    let col_indices = vec![
        0, 1, // row 0
        0, 1, 2, // row 1
        1, 2, 3, // row 2
        2, 3, // row 3
    ];
    let values = vec![
        4.0, -1.0, // row 0
        -1.0, 4.0, -1.0, // row 1
        -1.0, 4.0, -1.0, // row 2
        -1.0, 4.0, // row 3
    ];
    let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

    println!("   Original matrix (4×4 tridiagonal):");
    println!("     NNZ: {}", a.nnz());

    // Apply AMD reordering
    let perm = reordering::amd(&a).unwrap();
    println!("   AMD permutation: {:?}", perm);

    let reordered = reordering::permute_symmetric(&a, &perm).unwrap();
    println!("\n   After AMD reordering:");
    println!("     NNZ: {} (preserved)", reordered.nnz());
    println!("     Note: AMD minimizes fill-in during factorization");
    println!();
}

/// Example 3: Permutation validation
fn example_permutation_validation() {
    println!("3. Permutation Validation");

    let row_ptr = vec![0, 2, 4, 5];
    let col_indices = vec![0, 1, 0, 2, 1];
    let values = vec![2.0, -1.0, -1.0, 2.0, 1.0];
    let a = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

    // Valid permutation (reverse order)
    let perm_valid = vec![2, 1, 0];
    println!("   Valid permutation: {:?}", perm_valid);

    match reordering::permute_symmetric(&a, &perm_valid) {
        Ok(reordered) => {
            println!("     ✓ Permutation applied successfully");
            println!(
                "     Original NNZ: {}, Reordered NNZ: {}",
                a.nnz(),
                reordered.nnz()
            );
        }
        Err(e) => println!("     ✗ Error: {}", e),
    }

    // Invalid permutation (duplicate index)
    let perm_invalid = vec![0, 1, 1];
    println!("\n   Invalid permutation (duplicate): {:?}", perm_invalid);

    match reordering::permute_symmetric(&a, &perm_invalid) {
        Ok(_) => println!("     ✗ Should have failed!"),
        Err(e) => println!("     ✓ Correctly rejected: {}", e),
    }

    // Invalid permutation (out of bounds)
    let perm_oob = vec![0, 1, 5];
    println!("\n   Invalid permutation (out of bounds): {:?}", perm_oob);

    match reordering::permute_symmetric(&a, &perm_oob) {
        Ok(_) => println!("     ✗ Should have failed!"),
        Err(e) => println!("     ✓ Correctly rejected: {}", e),
    }
    println!();
}

/// Example 4: Handling disconnected graphs
fn example_disconnected_graph() {
    println!("4. Handling Disconnected Graphs");

    // Create a matrix with two disconnected components
    let row_ptr = vec![0, 2, 4, 5, 7];
    let col_indices = vec![
        0, 1, // component 1: vertices 0-1
        0, 1, // component 1: vertices 0-1
        2, // component 2: vertex 2 (isolated)
        2, 3, // component 2: vertices 2-3
    ];
    let values = vec![1.0; col_indices.len()];
    let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 4)).unwrap();

    println!("   Matrix with 2 disconnected components:");
    println!("     Component 1: vertices {{0, 1}}");
    println!("     Component 2: vertices {{2, 3}}");

    let perm = reordering::rcm(&a).unwrap();
    println!("\n   RCM handles disconnected components:");
    println!("     Permutation: {:?}", perm);

    let reordered = reordering::permute_symmetric(&a, &perm).unwrap();
    println!("     ✓ Successfully reordered disconnected graph");
    println!(
        "     Original NNZ: {}, Reordered NNZ: {}",
        a.nnz(),
        reordered.nnz()
    );
    println!();
}
