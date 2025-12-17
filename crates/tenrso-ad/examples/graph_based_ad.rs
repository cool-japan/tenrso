//! Graph-Based Automatic Differentiation Example
//!
//! This example demonstrates the graph-based AD system, which provides
//! PyTorch-style automatic differentiation with dynamic computation graphs.
//!
//! The graph-based approach automatically records operations during the
//! forward pass and constructs a computation graph, then performs backward
//! propagation in reverse topological order.

use anyhow::Result;
use scirs2_core::ndarray_ext::array;
use tenrso_ad::graph::ComputationGraph;

fn main() -> Result<()> {
    println!("=== Graph-Based Automatic Differentiation ===\n");

    // Example 1: Basic Operations
    println!("Example 1: Basic Operations");
    basic_operations()?;

    // Example 2: Neural Network Forward/Backward
    println!("\nExample 2: Simple Neural Network");
    simple_neural_network()?;

    // Example 3: Complex Computation Graph
    println!("\nExample 3: Complex Computation Graph");
    complex_graph()?;

    // Example 4: Dynamic Control Flow
    println!("\nExample 4: Dynamic Control Flow");
    dynamic_control_flow()?;

    // Example 5: Training Loop Simulation
    println!("\nExample 5: Training Loop Simulation");
    training_loop()?;

    // Example 6: Graph Statistics
    println!("\nExample 6: Graph Statistics");
    graph_statistics()?;

    Ok(())
}

/// Example 1: Basic arithmetic operations with gradients
fn basic_operations() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();

    // Create input variables
    let x = graph.variable(array![2.0, 3.0].into_dyn(), true)?;
    let y = graph.variable(array![4.0, 5.0].into_dyn(), true)?;

    // Build computation: z = (x + y) * x
    let sum = graph.add(&x, &y)?;
    let z = graph.mul(&sum, &x)?;

    // Forward pass values
    let z_val = graph.value(&z)?;
    println!("  x = [2.0, 3.0]");
    println!("  y = [4.0, 5.0]");
    println!("  z = (x + y) * x = [{}, {}]", z_val[[0]], z_val[[1]]);

    // Backward pass
    graph.backward(&graph.sum(&z)?)?;

    // Get gradients
    let grad_x = graph.gradient(&x)?;
    let grad_y = graph.gradient(&y)?;

    println!("  ∂z/∂x = [{}, {}]", grad_x[[0]], grad_x[[1]]);
    println!("  ∂z/∂y = [{}, {}]", grad_y[[0]], grad_y[[1]]);

    // Verify gradients
    // d/dx [(x+y)*x] = 2x + y = [8, 11]
    // d/dy [(x+y)*x] = x = [2, 3]
    assert!((grad_x[[0]] - 8.0).abs() < 1e-6);
    assert!((grad_x[[1]] - 11.0).abs() < 1e-6);
    assert!((grad_y[[0]] - 2.0).abs() < 1e-6);
    assert!((grad_y[[1]] - 3.0).abs() < 1e-6);

    println!("  ✓ Gradients verified!");

    Ok(())
}

/// Example 2: Simple neural network with one hidden layer
fn simple_neural_network() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();

    // Input
    let x = graph.variable(array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), true)?;

    // Layer 1: W1, b1
    let w1 = graph.variable(array![[0.5, 0.6], [0.7, 0.8]].into_dyn(), true)?;
    let b1 = graph.variable(array![0.1, 0.2].into_dyn(), true)?;

    // Forward: h = ReLU(x @ W1 + b1)
    let xw1 = graph.matmul(&x, &w1)?;
    let h_pre = graph.add(&xw1, &b1)?;
    let h = graph.relu(&h_pre)?;

    // Layer 2: W2, b2
    let w2 = graph.variable(array![[0.3], [0.4]].into_dyn(), true)?;
    let b2 = graph.variable(array![0.05].into_dyn(), true)?;

    // Output: y = h @ W2 + b2
    let hw2 = graph.matmul(&h, &w2)?;
    let y = graph.add(&hw2, &b2)?;

    // Loss: mean squared error (simplified - just sum for now)
    let loss = graph.sum(&y)?;

    let loss_val = graph.value(&loss)?;
    println!("  Forward pass loss: {:.4}", loss_val[[]]);

    // Backward pass
    graph.backward(&loss)?;

    // Check that all parameters have gradients
    println!("  Gradients computed for:");
    if graph.has_gradient(&w1) {
        println!("    - W1 ✓");
    }
    if graph.has_gradient(&b1) {
        println!("    - b1 ✓");
    }
    if graph.has_gradient(&w2) {
        println!("    - W2 ✓");
    }
    if graph.has_gradient(&b2) {
        println!("    - b2 ✓");
    }

    Ok(())
}

/// Example 3: Complex computation graph with multiple operations
fn complex_graph() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();

    let x = graph.variable(array![2.0].into_dyn(), true)?;

    // Build complex expression: f(x) = exp(x^2) + log(x + 1)
    let x_squared = graph.pow(&x, 2.0)?;
    let exp_part = graph.exp(&x_squared)?;

    let one = graph.constant(array![1.0].into_dyn())?;
    let x_plus_1 = graph.add(&x, &one)?;
    let log_part = graph.log(&x_plus_1)?;

    let result = graph.add(&exp_part, &log_part)?;

    let result_val = graph.value(&result)?;
    println!("  f(x) = exp(x²) + log(x+1)");
    println!("  f(2.0) = {:.4}", result_val[[]]);

    // Backward pass
    graph.backward(&result)?;
    let grad_x = graph.gradient(&x)?;

    println!("  df/dx at x=2.0 = {:.4}", grad_x[[]]);

    // Analytical gradient: df/dx = 2x*exp(x²) + 1/(x+1)
    // At x=2: 2*2*exp(4) + 1/3 = 4*exp(4) + 0.333... ≈ 218.73
    let expected = 4.0 * (4.0_f64).exp() + 1.0 / 3.0;
    println!("  Expected: {:.4}", expected);
    assert!((grad_x[[]] - expected).abs() < 1e-2);
    println!("  ✓ Gradient verified!");

    Ok(())
}

/// Example 4: Dynamic control flow - graph changes based on runtime values
fn dynamic_control_flow() -> Result<()> {
    println!("  Building different graphs based on input values...");

    for &threshold in &[0.0, 0.5, 1.0] {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![threshold].into_dyn(), true)?;

        // Different operations based on value
        let y = if threshold > 0.5 {
            // Use sigmoid for large values
            graph.sigmoid(&x)?
        } else {
            // Use tanh for small values
            graph.tanh(&x)?
        };

        graph.backward(&y)?;
        let grad_x = graph.gradient(&x)?;

        let activation = if threshold > 0.5 { "sigmoid" } else { "tanh" };
        println!(
            "    x={:.1} -> {} -> grad={:.4}",
            threshold,
            activation,
            grad_x[[]]
        );
    }

    println!("  ✓ Dynamic control flow working!");

    Ok(())
}

/// Example 5: Simulate a training loop with graph reuse
fn training_loop() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();

    // Model parameters (simple linear model: y = w*x + b)
    let mut w_data = array![0.5].into_dyn();
    let mut b_data = array![0.0].into_dyn();

    let learning_rate = 0.1;
    let num_epochs = 5;

    println!("  Training y = w*x + b to fit y = 2*x + 1");
    println!("  Initial: w={:.4}, b={:.4}", w_data[[]], b_data[[]]);

    for epoch in 0..num_epochs {
        // Clear graph from previous iteration
        graph.zero_grad();

        // Create variables with current parameter values
        let w = graph.variable(w_data.clone(), true)?;
        let b = graph.variable(b_data.clone(), true)?;

        // Training data: x=1, target=3
        let x = graph.constant(array![1.0].into_dyn())?;
        let target = 3.0; // 2*1 + 1 = 3

        // Forward pass
        let wx = graph.mul(&w, &x)?;
        let prediction = graph.add(&wx, &b)?;

        // Loss: (prediction - target)^2
        let target_var = graph.constant(array![target].into_dyn())?;
        let diff = graph.sub(&prediction, &target_var)?;
        let loss = graph.pow(&diff, 2.0)?;

        let loss_val = graph.value(&loss)?;

        // Backward pass
        graph.backward(&loss)?;

        // Get gradients
        let grad_w = graph.gradient(&w)?;
        let grad_b = graph.gradient(&b)?;

        // Update parameters
        w_data = &w_data - &grad_w.mapv(|g| g * learning_rate);
        b_data = &b_data - &grad_b.mapv(|g| g * learning_rate);

        println!(
            "  Epoch {}: loss={:.4}, w={:.4}, b={:.4}",
            epoch + 1,
            loss_val[[]],
            w_data[[]],
            b_data[[]]
        );
    }

    println!("  Final: w={:.4}, b={:.4}", w_data[[]], b_data[[]]);
    println!("  Target: w=2.0000, b=1.0000");
    println!("  ✓ Training completed!");

    Ok(())
}

/// Example 6: Graph statistics and inspection
fn graph_statistics() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();

    // Build a moderately complex graph
    let x = graph.variable(array![1.0, 2.0].into_dyn(), true)?;
    let y = graph.variable(array![3.0, 4.0].into_dyn(), true)?;

    let sum = graph.add(&x, &y)?;
    let prod = graph.mul(&x, &y)?;
    let exp_sum = graph.exp(&sum)?;
    let log_prod = graph.log(&prod)?;
    let combined = graph.add(&exp_sum, &log_prod)?;
    let _ = graph.sum(&combined)?;

    // Get and display statistics
    let stats = graph.stats();
    println!("{}", stats);

    Ok(())
}
