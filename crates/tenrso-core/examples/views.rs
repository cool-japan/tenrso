//! Tensor views and shape operations examples.
//!
//! This example demonstrates:
//! - Creating immutable and mutable views
//! - Zero-copy operations when possible
//! - Reshape operations
//! - Permute (generalized transpose) operations
//!
//! Run with:
//! ```bash
//! cargo run --example views
//! ```

use tenrso_core::DenseND;

fn main() {
    println!("=== TenRSo Core: Views and Shape Operations ===\n");

    // Example 1: Views for zero-copy access
    example_views();

    // Example 2: Reshape operations
    example_reshape();

    // Example 3: Permute (transpose) operations
    example_permute();

    // Example 4: Complex shape manipulations
    example_complex_shapes();

    println!("\n=== All examples completed successfully! ===");
}

fn example_views() {
    println!("--- Example 1: Views for Zero-Copy Access ---");

    let mut tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    println!("Original tensor [2, 3]:");
    println!("  Element at [0, 0]: {}", tensor[&[0, 0]]);
    println!("  Element at [1, 2]: {}", tensor[&[1, 2]]);

    // Create an immutable view
    {
        let view = tensor.view();
        println!("\nImmutable view:");
        println!("  View shape: {:?}", view.shape());
        println!("  View element [0, 1]: {}", view[[0, 1]]);
        // Cannot modify through immutable view:
        // view[[0, 0]] = 10.0; // This would not compile
    }

    // Create a mutable view and modify
    {
        let mut view_mut = tensor.view_mut();
        println!("\nMutable view (modifying elements):");
        view_mut[[0, 0]] = 100.0;
        view_mut[[1, 2]] = 600.0;
        println!("  Modified [0, 0]: {}", view_mut[[0, 0]]);
        println!("  Modified [1, 2]: {}", view_mut[[1, 2]]);
    }

    // Changes are reflected in the original tensor
    println!("\nOriginal tensor after view modifications:");
    println!("  Element at [0, 0]: {}", tensor[&[0, 0]]);
    println!("  Element at [1, 2]: {}", tensor[&[1, 2]]);

    println!();
}

fn example_reshape() {
    println!("--- Example 2: Reshape Operations ---");

    // Create a 2x3x4 tensor
    let tensor = DenseND::<f64>::from_elem(&[2, 3, 4], 1.0);
    println!("Original shape: {:?}", tensor.shape());
    println!("Total elements: {}", tensor.len());

    // Reshape to 6x4
    let reshaped_1 = tensor.reshape(&[6, 4]).unwrap();
    println!("\nReshape to [6, 4]:");
    println!("  New shape: {:?}", reshaped_1.shape());
    println!("  Total elements: {}", reshaped_1.len());

    // Reshape to 8x3
    let reshaped_2 = tensor.reshape(&[8, 3]).unwrap();
    println!("\nReshape to [8, 3]:");
    println!("  New shape: {:?}", reshaped_2.shape());
    println!("  Total elements: {}", reshaped_2.len());

    // Reshape to 1D
    let reshaped_flat = tensor.reshape(&[24]).unwrap();
    println!("\nReshape to [24] (flatten):");
    println!("  New shape: {:?}", reshaped_flat.shape());
    println!("  Rank: {}", reshaped_flat.rank());

    // Reshape to 4D
    let reshaped_4d = tensor.reshape(&[2, 2, 2, 3]).unwrap();
    println!("\nReshape to [2, 2, 2, 3] (4D):");
    println!("  New shape: {:?}", reshaped_4d.shape());
    println!("  Rank: {}", reshaped_4d.rank());

    // Error: incompatible size
    match tensor.reshape(&[5, 5]) {
        Ok(_) => println!("\nUnexpected success!"),
        Err(e) => println!("\nExpected error for incompatible reshape: {}", e),
    }

    println!();
}

fn example_permute() {
    println!("--- Example 3: Permute (Transpose) Operations ---");

    // Create a tensor with data we can track
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let tensor = DenseND::from_vec(data, &[2, 3, 4]).unwrap();

    println!("Original tensor [2, 3, 4]:");
    println!("  Shape: {:?}", tensor.shape());
    println!("  Element at [0, 0, 0]: {}", tensor[&[0, 0, 0]]);
    println!("  Element at [1, 2, 3]: {}", tensor[&[1, 2, 3]]);

    // Standard transpose (swap first two axes)
    let transposed = tensor.permute(&[1, 0, 2]).unwrap();
    println!("\nPermute [1, 0, 2] - swap first two axes:");
    println!("  New shape: {:?}", transposed.shape());
    println!(
        "  Original [1, 2, 3] -> Permuted [2, 1, 3]: {}",
        transposed[&[2, 1, 3]]
    );

    // Cycle axes: move last to front
    let cycled = tensor.permute(&[2, 0, 1]).unwrap();
    println!("\nPermute [2, 0, 1] - cycle axes (last to front):");
    println!("  New shape: {:?}", cycled.shape());

    // Reverse axes
    let reversed = tensor.permute(&[2, 1, 0]).unwrap();
    println!("\nPermute [2, 1, 0] - reverse all axes:");
    println!("  New shape: {:?}", reversed.shape());

    // 2D transpose
    let matrix = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    println!("\n2D Matrix [2, 3]:");
    println!("  Original shape: {:?}", matrix.shape());
    println!("  Element at [0, 2]: {}", matrix[&[0, 2]]);

    let transposed_matrix = matrix.permute(&[1, 0]).unwrap();
    println!("\nTransposed [3, 2]:");
    println!("  New shape: {:?}", transposed_matrix.shape());
    println!(
        "  Original [0, 2] -> Transposed [2, 0]: {}",
        transposed_matrix[&[2, 0]]
    );

    println!();
}

fn example_complex_shapes() {
    println!("--- Example 4: Complex Shape Manipulations ---");

    // Start with a 4D tensor
    let tensor = DenseND::<f64>::ones(&[2, 3, 4, 5]);
    println!("Starting with 4D tensor [2, 3, 4, 5]");
    println!("  Total elements: {}", tensor.len());

    // Reshape to merge dimensions
    let merged = tensor.reshape(&[6, 20]).unwrap();
    println!("\nMerge to [6, 20]:");
    println!("  Shape: {:?}", merged.shape());

    // Reshape back to different 4D
    let reshaped_4d = merged.reshape(&[3, 2, 5, 4]).unwrap();
    println!("\nReshape to different 4D [3, 2, 5, 4]:");
    println!("  Shape: {:?}", reshaped_4d.shape());

    // Permute and then reshape
    let permuted = reshaped_4d.permute(&[3, 2, 1, 0]).unwrap();
    println!("\nPermute [3, 2, 1, 0]:");
    println!("  Shape: {:?}", permuted.shape());

    let final_reshape = permuted.reshape(&[20, 6]).unwrap();
    println!("\nFinal reshape to [20, 6]:");
    println!("  Shape: {:?}", final_reshape.shape());

    // Demonstrate batch operations
    let batch_tensor = DenseND::<f64>::zeros(&[32, 28, 28, 3]);
    println!("\n\nBatch of images [batch=32, height=28, width=28, channels=3]:");
    println!("  Total elements: {}", batch_tensor.len());

    // Flatten spatial dimensions
    let flattened = batch_tensor.reshape(&[32, 28 * 28 * 3]).unwrap();
    println!("\nFlatten spatial + channels [32, 2352]:");
    println!("  Shape: {:?}", flattened.shape());

    // Move channels to different position
    let channels_first = batch_tensor.permute(&[0, 3, 1, 2]).unwrap();
    println!("\nMove channels first [batch, channels, height, width]:");
    println!("  Shape: {:?}", channels_first.shape());

    println!();
}
