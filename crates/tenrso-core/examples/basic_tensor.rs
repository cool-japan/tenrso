//! Basic tensor creation and manipulation examples.
//!
//! This example demonstrates the core functionality of tenrso-core:
//! - Creating tensors with different initialization methods
//! - Accessing tensor properties (rank, shape, size)
//! - Indexing and modifying tensor elements
//! - Using TensorHandle with axis metadata
//!
//! Run with:
//! ```bash
//! cargo run --example basic_tensor
//! ```

use tenrso_core::{AxisMeta, DenseND, TensorHandle};

fn main() {
    println!("=== TenRSo Core: Basic Tensor Examples ===\n");

    // Example 1: Creating tensors with different methods
    example_creation();

    // Example 2: Tensor properties and inspection
    example_properties();

    // Example 3: Indexing and modification
    example_indexing();

    // Example 4: Using TensorHandle with axis metadata
    example_tensor_handle();

    // Example 5: Random initialization
    example_random();

    println!("\n=== All examples completed successfully! ===");
}

fn example_creation() {
    println!("--- Example 1: Tensor Creation ---");

    // Create a tensor of zeros
    let zeros = DenseND::<f64>::zeros(&[2, 3]);
    println!("Zeros tensor [2, 3]:");
    println!("  Shape: {:?}", zeros.shape());
    println!("  First element: {}", zeros[&[0, 0]]);

    // Create a tensor of ones
    let ones = DenseND::<f64>::ones(&[3, 4]);
    println!("\nOnes tensor [3, 4]:");
    println!("  Shape: {:?}", ones.shape());
    println!("  Element at [1, 2]: {}", ones[&[1, 2]]);

    // Create a tensor filled with a specific value
    let fives = DenseND::from_elem(&[2, 2, 2], 5.0);
    println!("\nTensor filled with 5.0 [2, 2, 2]:");
    println!("  Shape: {:?}", fives.shape());
    println!("  Element at [0, 1, 1]: {}", fives[&[0, 1, 1]]);

    // Create a tensor from a vector
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let from_vec = DenseND::from_vec(data, &[2, 3]).unwrap();
    println!("\nTensor from vector [2, 3]:");
    println!("  Shape: {:?}", from_vec.shape());
    println!("  Element at [0, 0]: {}", from_vec[&[0, 0]]);
    println!("  Element at [1, 2]: {}", from_vec[&[1, 2]]);

    println!();
}

fn example_properties() {
    println!("--- Example 2: Tensor Properties ---");

    let tensor = DenseND::<f64>::zeros(&[10, 20, 30]);

    println!("Tensor shape: [10, 20, 30]");
    println!("  Rank (number of dimensions): {}", tensor.rank());
    println!("  Shape: {:?}", tensor.shape());
    println!("  Total elements: {}", tensor.len());
    println!("  Is empty? {}", tensor.is_empty());

    // Create a 4D tensor
    let tensor_4d = DenseND::<f32>::ones(&[2, 3, 4, 5]);
    println!("\n4D tensor [2, 3, 4, 5]:");
    println!("  Rank: {}", tensor_4d.rank());
    println!("  Total elements: {}", tensor_4d.len());

    println!();
}

fn example_indexing() {
    println!("--- Example 3: Indexing and Modification ---");

    let mut tensor = DenseND::<f64>::zeros(&[3, 4]);

    // Read elements
    println!("Initial tensor [3, 4] (all zeros)");
    println!("  Element at [0, 0]: {}", tensor[&[0, 0]]);
    println!("  Element at [2, 3]: {}", tensor[&[2, 3]]);

    // Modify elements
    tensor[&[0, 0]] = 1.0;
    tensor[&[1, 2]] = 3.5;
    tensor[&[2, 3]] = 42.0;

    println!("\nAfter modification:");
    println!("  Element at [0, 0]: {}", tensor[&[0, 0]]);
    println!("  Element at [1, 2]: {}", tensor[&[1, 2]]);
    println!("  Element at [2, 3]: {}", tensor[&[2, 3]]);

    // Demonstrate bounds checking (will panic if uncommented)
    // println!("  Element at [10, 10]: {}", tensor[&[10, 10]]); // Panics!

    println!();
}

fn example_tensor_handle() {
    println!("--- Example 4: TensorHandle with Axis Metadata ---");

    // Create tensor with automatic axis naming
    let tensor = DenseND::<f64>::ones(&[32, 128, 256]);
    let handle_auto = TensorHandle::from_dense_auto(tensor);

    println!("TensorHandle with automatic axis naming:");
    println!("  Rank: {}", handle_auto.rank());
    println!(
        "  Axis 0: {} (size: {})",
        handle_auto.axes[0].name, handle_auto.axes[0].size
    );
    println!(
        "  Axis 1: {} (size: {})",
        handle_auto.axes[1].name, handle_auto.axes[1].size
    );
    println!(
        "  Axis 2: {} (size: {})",
        handle_auto.axes[2].name, handle_auto.axes[2].size
    );

    // Create tensor with custom axis names
    let tensor2 = DenseND::<f64>::zeros(&[32, 128, 256]);
    let axes = vec![
        AxisMeta::new("batch", 32),
        AxisMeta::new("sequence", 128),
        AxisMeta::new("features", 256),
    ];
    let handle = TensorHandle::from_dense(tensor2, axes);

    println!("\nTensorHandle with custom axis names:");
    println!("  Rank: {}", handle.rank());
    for (i, axis) in handle.axes.iter().enumerate() {
        println!("  Axis {}: \"{}\" (size: {})", i, axis.name, axis.size);
    }

    // Access the underlying dense tensor
    if let Some(dense) = handle.as_dense() {
        println!("\nAccessing underlying dense tensor:");
        println!("  Total elements: {}", dense.len());
        println!("  First element: {}", dense[&[0, 0, 0]]);
    }

    println!();
}

fn example_random() {
    println!("--- Example 5: Random Initialization ---");

    // Uniform distribution
    let uniform = DenseND::<f64>::random_uniform(&[3, 4], 0.0, 1.0);
    println!("Random uniform [3, 4] in range [0.0, 1.0):");
    println!("  Shape: {:?}", uniform.shape());
    println!("  Sample elements:");
    println!("    [0, 0]: {:.4}", uniform[&[0, 0]]);
    println!("    [1, 2]: {:.4}", uniform[&[1, 2]]);
    println!("    [2, 3]: {:.4}", uniform[&[2, 3]]);

    // Normal distribution
    let normal = DenseND::<f64>::random_normal(&[3, 4], 0.0, 1.0);
    println!("\nRandom normal [3, 4] (mean=0.0, std=1.0):");
    println!("  Shape: {:?}", normal.shape());
    println!("  Sample elements:");
    println!("    [0, 0]: {:.4}", normal[&[0, 0]]);
    println!("    [1, 2]: {:.4}", normal[&[1, 2]]);
    println!("    [2, 3]: {:.4}", normal[&[2, 3]]);

    // Larger tensor with uniform distribution
    let large_uniform = DenseND::<f64>::random_uniform(&[10, 10], -1.0, 1.0);
    println!("\nLarge random uniform [10, 10] in range [-1.0, 1.0):");
    println!("  Total elements: {}", large_uniform.len());
    println!("  Corner elements:");
    println!("    [0, 0]: {:.4}", large_uniform[&[0, 0]]);
    println!("    [9, 9]: {:.4}", large_uniform[&[9, 9]]);

    println!();
}
