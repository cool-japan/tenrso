//! Tensor matricization (unfold/fold) operations examples.
//!
//! This example demonstrates mode-n unfolding and folding, which are critical
//! operations for tensor decompositions like CP-ALS, Tucker-HOOI, and TT-SVD.
//!
//! Unfold (matricization): Rearranges a tensor into a matrix along a specified mode
//! Fold (tensorization): Inverse operation that reconstructs a tensor from a matrix
//!
//! Run with:
//! ```bash
//! cargo run --example unfold_fold
//! ```

use tenrso_core::DenseND;

fn main() {
    println!("=== TenRSo Core: Unfold/Fold (Matricization) Examples ===\n");

    // Example 1: Basic unfold/fold operations
    example_basic_unfold_fold();

    // Example 2: Unfold along different modes
    example_different_modes();

    // Example 3: Roundtrip verification (unfold → fold)
    example_roundtrip();

    // Example 4: Use case for tensor decompositions
    example_decomposition_usage();

    println!("\n=== All examples completed successfully! ===");
}

fn example_basic_unfold_fold() {
    println!("--- Example 1: Basic Unfold/Fold Operations ---");

    // Create a 3D tensor [2, 3, 4]
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let tensor = DenseND::from_vec(data, &[2, 3, 4]).unwrap();

    println!("Original tensor shape: {:?}", tensor.shape());
    println!("Original tensor size: {} elements", tensor.len());
    println!("Sample elements:");
    println!("  [0, 0, 0]: {}", tensor[&[0, 0, 0]]);
    println!("  [0, 1, 2]: {}", tensor[&[0, 1, 2]]);
    println!("  [1, 2, 3]: {}", tensor[&[1, 2, 3]]);

    // Unfold along mode 0
    let unfolded_0 = tensor.unfold(0).unwrap();
    println!("\nUnfolded along mode 0:");
    println!("  Matrix shape: {:?}", unfolded_0.shape());
    println!("  (mode 0 size: 2, others: 3*4 = 12)");

    // Fold back to original shape
    let folded_0 = DenseND::fold(&unfolded_0, &[2, 3, 4], 0).unwrap();
    println!("\nFolded back from mode 0:");
    println!("  Tensor shape: {:?}", folded_0.shape());
    println!("  Sample elements (should match original):");
    println!("    [0, 0, 0]: {}", folded_0[&[0, 0, 0]]);
    println!("    [0, 1, 2]: {}", folded_0[&[0, 1, 2]]);
    println!("    [1, 2, 3]: {}", folded_0[&[1, 2, 3]]);

    println!();
}

fn example_different_modes() {
    println!("--- Example 2: Unfold Along Different Modes ---");

    let tensor = DenseND::<f64>::ones(&[2, 3, 4]);
    println!("Tensor shape: {:?}", tensor.shape());

    // Unfold along each mode
    for mode in 0..3 {
        let unfolded = tensor.unfold(mode).unwrap();
        let mode_size = tensor.shape()[mode];
        let other_size: usize = tensor
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != mode)
            .map(|(_, &s)| s)
            .product();

        println!("\nMode {} unfold:", mode);
        println!("  Matrix shape: {:?}", unfolded.shape());
        println!("  Expected: [{}, {}]", mode_size, other_size);
        println!(
            "  Verification: {} x {} = {} (total elements)",
            mode_size,
            other_size,
            mode_size * other_size
        );
    }

    println!();
}

fn example_roundtrip() {
    println!("--- Example 3: Roundtrip Verification (Unfold → Fold) ---");

    // Create a tensor with distinct values
    let data: Vec<f64> = (1..=60).map(|x| x as f64).collect();
    let original = DenseND::from_vec(data, &[3, 4, 5]).unwrap();

    println!("Original tensor [3, 4, 5]:");
    println!("  Sample elements:");
    println!("    [0, 0, 0]: {}", original[&[0, 0, 0]]);
    println!("    [1, 2, 3]: {}", original[&[1, 2, 3]]);
    println!("    [2, 3, 4]: {}", original[&[2, 3, 4]]);

    // Test roundtrip for each mode
    for mode in 0..3 {
        println!("\n  Testing mode {}:", mode);

        // Unfold
        let unfolded = original.unfold(mode).unwrap();
        println!("    Unfolded shape: {:?}", unfolded.shape());

        // Fold back
        let reconstructed = DenseND::fold(&unfolded, original.shape(), mode).unwrap();
        println!("    Reconstructed shape: {:?}", reconstructed.shape());

        // Verify all elements match
        let mut all_match = true;
        let shape = original.shape();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    if (original[&[i, j, k]] - reconstructed[&[i, j, k]]).abs() > 1e-10 {
                        all_match = false;
                        break;
                    }
                }
            }
        }
        println!("    Roundtrip successful: {}", all_match);
    }

    println!();
}

fn example_decomposition_usage() {
    println!("--- Example 4: Use Case for Tensor Decompositions ---");

    // Create a tensor representing a 3-way data array
    // (e.g., users × items × contexts)
    let tensor = DenseND::<f64>::random_uniform(&[10, 15, 8], 0.0, 1.0);
    println!("Tensor shape [users=10, items=15, contexts=8]:");
    println!("  Total elements: {}", tensor.len());

    println!("\n--- Simulating CP-ALS decomposition workflow ---");

    // In CP-ALS, we need to unfold along each mode to compute factor matrices
    println!("\nStep 1: Unfold along mode 0 (users)");
    let unfolded_users = tensor.unfold(0).unwrap();
    println!("  Matrix shape: {:?}", unfolded_users.shape());
    println!("  This matrix will be used to update user factor matrix");

    println!("\nStep 2: Unfold along mode 1 (items)");
    let unfolded_items = tensor.unfold(1).unwrap();
    println!("  Matrix shape: {:?}", unfolded_items.shape());
    println!("  This matrix will be used to update item factor matrix");

    println!("\nStep 3: Unfold along mode 2 (contexts)");
    let unfolded_contexts = tensor.unfold(2).unwrap();
    println!("  Matrix shape: {:?}", unfolded_contexts.shape());
    println!("  This matrix will be used to update context factor matrix");

    println!("\n--- Simulating Tucker decomposition workflow ---");

    // In Tucker-HOSVD, we compute SVD of each unfolding
    println!("\nTucker uses mode-n unfoldings for computing factor matrices:");
    for mode in 0..3 {
        let unfolded = tensor.unfold(mode).unwrap();
        println!("  Mode {}: {:?} matrix for SVD", mode, unfolded.shape());
    }

    println!("\n--- Higher-order tensors ---");

    // 4D tensor (batch, height, width, channels)
    let tensor_4d = DenseND::<f64>::zeros(&[32, 64, 64, 3]);
    println!("\n4D tensor [batch=32, height=64, width=64, channels=3]:");

    for mode in 0..4 {
        let unfolded = tensor_4d.unfold(mode).unwrap();
        println!("  Mode {} unfold: {:?}", mode, unfolded.shape());
    }

    println!();
}
