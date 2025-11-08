//! Example of creating custom AD operations with the generic adapter
//!
//! This example demonstrates:
//! - Implementing the AdOperation trait for custom operations
//! - Using the GenericAdAdapter to record and execute backward passes
//! - Composing multiple operations
//!
//! Run with: cargo run --example custom_ad_operations

use anyhow::Result;
use tenrso_ad::hooks::{AdContext, AdOperation, GenericAdAdapter};
use tenrso_core::DenseND;

/// Custom operation: Element-wise ReLU activation
#[derive(Debug)]
struct ReluOp {
    input: DenseND<f64>,
}

impl ReluOp {
    fn new(input: DenseND<f64>) -> Self {
        Self { input }
    }
}

impl AdOperation<f64> for ReluOp {
    fn forward(&self) -> Result<Vec<DenseND<f64>>> {
        let mut output = self.input.clone();

        for idx in 0..output.len() {
            let multi_idx = linear_to_multi_index(output.shape(), idx);
            let val = *output.get(&multi_idx).unwrap();
            *output.get_mut(&multi_idx).unwrap() = if val > 0.0 { val } else { 0.0 };
        }

        Ok(vec![output])
    }

    fn backward(&self, output_grads: &[DenseND<f64>]) -> Result<Vec<DenseND<f64>>> {
        let grad_output = &output_grads[0];
        let mut grad_input = DenseND::zeros(self.input.shape());

        for idx in 0..self.input.len() {
            let multi_idx = linear_to_multi_index(self.input.shape(), idx);
            let x = *self.input.get(&multi_idx).unwrap();
            let g = *grad_output.get(&multi_idx).unwrap();

            // ReLU gradient: g if x > 0, else 0
            *grad_input.get_mut(&multi_idx).unwrap() = if x > 0.0 { g } else { 0.0 };
        }

        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "relu"
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

/// Custom operation: Element-wise sigmoid activation
#[derive(Debug)]
struct SigmoidOp {
    input: DenseND<f64>,
}

impl SigmoidOp {
    fn new(input: DenseND<f64>) -> Self {
        Self { input }
    }
}

impl AdOperation<f64> for SigmoidOp {
    fn forward(&self) -> Result<Vec<DenseND<f64>>> {
        let mut output = self.input.clone();

        for idx in 0..output.len() {
            let multi_idx = linear_to_multi_index(output.shape(), idx);
            let val = *output.get(&multi_idx).unwrap();
            *output.get_mut(&multi_idx).unwrap() = 1.0 / (1.0 + (-val).exp());
        }

        Ok(vec![output])
    }

    fn backward(&self, output_grads: &[DenseND<f64>]) -> Result<Vec<DenseND<f64>>> {
        let grad_output = &output_grads[0];
        let mut grad_input = DenseND::zeros(self.input.shape());

        for idx in 0..self.input.len() {
            let multi_idx = linear_to_multi_index(self.input.shape(), idx);
            let x = *self.input.get(&multi_idx).unwrap();
            let g = *grad_output.get(&multi_idx).unwrap();

            // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x)) * g
            let sigmoid_x = 1.0 / (1.0 + (-x).exp());
            *grad_input.get_mut(&multi_idx).unwrap() = sigmoid_x * (1.0 - sigmoid_x) * g;
        }

        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "sigmoid"
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

/// Custom operation: Scaled addition (y = a*x + b)
#[derive(Debug)]
struct ScaledAddOp {
    input: DenseND<f64>,
    scale: f64,
    bias: f64,
}

impl ScaledAddOp {
    fn new(input: DenseND<f64>, scale: f64, bias: f64) -> Self {
        Self { input, scale, bias }
    }
}

impl AdOperation<f64> for ScaledAddOp {
    fn forward(&self) -> Result<Vec<DenseND<f64>>> {
        let mut output = self.input.clone();

        for idx in 0..output.len() {
            let multi_idx = linear_to_multi_index(output.shape(), idx);
            let val = *output.get(&multi_idx).unwrap();
            *output.get_mut(&multi_idx).unwrap() = self.scale * val + self.bias;
        }

        Ok(vec![output])
    }

    fn backward(&self, output_grads: &[DenseND<f64>]) -> Result<Vec<DenseND<f64>>> {
        let grad_output = &output_grads[0];
        let mut grad_input = DenseND::zeros(self.input.shape());

        for idx in 0..self.input.len() {
            let multi_idx = linear_to_multi_index(self.input.shape(), idx);
            let g = *grad_output.get(&multi_idx).unwrap();

            // Gradient w.r.t. input: scale * g
            *grad_input.get_mut(&multi_idx).unwrap() = self.scale * g;
        }

        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "scaled_add"
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

/// Helper function to convert linear index to multi-dimensional index
fn linear_to_multi_index(shape: &[usize], linear_idx: usize) -> Vec<usize> {
    let mut multi_idx = vec![0; shape.len()];
    let mut remaining = linear_idx;

    for (dim, &size) in shape.iter().enumerate().rev() {
        multi_idx[dim] = remaining % size;
        remaining /= size;
    }

    multi_idx
}

fn main() -> Result<()> {
    println!("=== Custom AD Operations ===\n");

    // Example 1: Single ReLU operation
    println!("Example 1: ReLU Activation");
    println!("---------------------------");

    let input = DenseND::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[2, 3])?;
    println!("Input:\n{:?}", input.as_array());

    let relu_op = ReluOp::new(input.clone());
    let output = relu_op.forward()?;
    println!("\nReLU(input):\n{:?}", output[0].as_array());

    let grad_output = DenseND::ones(&[2, 3]);
    let grad_inputs = relu_op.backward(&[grad_output])?;
    println!("\nGradient:\n{:?}", grad_inputs[0].as_array());

    // Example 2: Sigmoid activation
    println!("\n\nExample 2: Sigmoid Activation");
    println!("------------------------------");

    let input2 = DenseND::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5])?;
    println!("Input: {:?}", input2.as_array());

    let sigmoid_op = SigmoidOp::new(input2.clone());
    let output2 = sigmoid_op.forward()?;
    println!("Sigmoid(input): {:?}", output2[0].as_array());

    let grad_output2 = DenseND::ones(&[5]);
    let grad_inputs2 = sigmoid_op.backward(&[grad_output2])?;
    println!("Gradient: {:?}", grad_inputs2[0].as_array());

    // Example 3: Composed operations with AD adapter
    println!("\n\nExample 3: Composed Operations");
    println!("--------------------------------");

    let mut adapter = GenericAdAdapter::<f64>::new();

    let x = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    println!("Initial input x:\n{:?}", x.as_array());

    // Build computation graph: y = ReLU(2*x + 1)
    println!("\nBuilding computation: y = ReLU(2*x + 1)");

    // Step 1: z = 2*x + 1
    let scaled_op = Box::new(ScaledAddOp::new(x.clone(), 2.0, 1.0));
    let _op1_id = adapter.register_operation(scaled_op);
    println!("Registered operation 1: scaled_add");

    // Step 2: y = ReLU(z)
    // (In a real implementation, we'd need to pass the output of op1 to op2)
    let z = DenseND::from_vec(vec![3.0, 5.0, 7.0, 9.0], &[2, 2])?; // 2*x + 1
    let relu_op2 = Box::new(ReluOp::new(z.clone()));
    let _op2_id = adapter.register_operation(relu_op2);
    println!("Registered operation 2: relu");

    println!("\nTotal operations recorded: {}", adapter.num_operations());

    // Forward pass (manually computed for illustration)
    println!("\nForward pass:");
    println!("  z = 2*x + 1 = {:?}", z.as_array());
    println!("  y = ReLU(z) = {:?}", z.as_array()); // ReLU does nothing since all positive

    // Backward pass (note: using dummy operation ID since we don't use the actual result)
    println!("\nBackward pass:");
    let grad_y = DenseND::<f64>::ones(&[2, 2]);
    println!("  Starting with ∂L/∂y = {:?}", grad_y.as_array());

    // Note: In a real implementation, we'd track operation IDs properly
    println!("  (Backward pass demonstration - see adapter implementation)");

    // Example 4: Demonstrate recording mode control
    println!("\n\nExample 4: Recording Mode Control");
    println!("-----------------------------------");

    let mut adapter2 = GenericAdAdapter::<f64>::new();
    println!("Initial recording state: {}", adapter2.is_recording());

    adapter2.stop_recording();
    println!("After stop_recording(): {}", adapter2.is_recording());

    // Operations registered while not recording should be ignored
    let dummy_op = Box::new(ReluOp::new(x.clone()));
    adapter2.register_operation(dummy_op);
    println!(
        "Operations after registering while stopped: {}",
        adapter2.num_operations()
    );

    adapter2.start_recording();
    println!("After start_recording(): {}", adapter2.is_recording());

    let active_op = Box::new(SigmoidOp::new(x.clone()));
    adapter2.register_operation(active_op);
    println!(
        "Operations after registering while recording: {}",
        adapter2.num_operations()
    );

    // Clear the adapter
    adapter2.clear();
    println!("After clear(): {} operations", adapter2.num_operations());

    println!("\n=== All examples completed successfully! ===");

    Ok(())
}
