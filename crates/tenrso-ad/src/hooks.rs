//! Integration hooks for external AD frameworks
//!
//! This module provides traits and adapters for integrating TenRSo operations
//! with external automatic differentiation (AD) frameworks like Tensorlogic.
//!
//! # Design Philosophy
//!
//! The integration is designed to be:
//! - **Framework-agnostic**: Generic traits that any AD system can implement
//! - **Zero-cost abstraction**: No runtime overhead for non-AD code paths
//! - **Minimally invasive**: TenRSo operations don't need to know about AD
//! - **Composable**: Multiple AD frameworks can coexist
//!
//! # Example
//!
//! ```rust,ignore
//! // Define how your AD framework wraps operations
//! struct MyAdAdapter;
//!
//! impl<T> AdContext<T> for MyAdAdapter {
//!     fn register_operation(&mut self, op: Box<dyn AdOperation<T>>) -> OperationId {
//!         // Record operation in your computation graph
//!     }
//!
//!     fn backward(&mut self, op_id: OperationId, grad: &DenseND<T>) {
//!         // Trigger backward pass
//!     }
//! }
//! ```

use anyhow::Result;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use tenrso_core::DenseND;

/// Unique identifier for operations in the AD graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(pub u64);

/// Trait for differentiable operations
///
/// External AD frameworks implement this to wrap TenRSo operations
/// with gradient computation logic.
pub trait AdOperation<T>: Debug
where
    T: Float + 'static,
{
    /// Execute the forward pass
    ///
    /// # Returns
    ///
    /// The output tensor(s) from this operation
    fn forward(&self) -> Result<Vec<DenseND<T>>>;

    /// Execute the backward pass
    ///
    /// # Arguments
    ///
    /// * `output_grads` - Gradients w.r.t. outputs (∂L/∂output)
    ///
    /// # Returns
    ///
    /// Gradients w.r.t. inputs (∂L/∂input)
    fn backward(&self, output_grads: &[DenseND<T>]) -> Result<Vec<DenseND<T>>>;

    /// Get a unique name for this operation (for debugging)
    fn name(&self) -> &str;

    /// Get the number of inputs
    fn num_inputs(&self) -> usize;

    /// Get the number of outputs
    fn num_outputs(&self) -> usize;
}

/// Trait for AD framework context management
///
/// This represents the "tape" or "graph" that records operations
/// for later differentiation.
pub trait AdContext<T>
where
    T: Float + 'static,
{
    /// Register a differentiable operation with the context
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to register
    ///
    /// # Returns
    ///
    /// Unique identifier for this operation in the graph
    fn register_operation(&mut self, op: Box<dyn AdOperation<T>>) -> OperationId;

    /// Execute backward pass starting from a specific operation
    ///
    /// # Arguments
    ///
    /// * `op_id` - The operation to start backpropagation from
    /// * `grad` - Initial gradient (usually ∂L/∂output for final layer)
    fn backward(&mut self, op_id: OperationId, grad: &DenseND<T>) -> Result<()>;

    /// Clear the tape/graph
    fn clear(&mut self);

    /// Get gradient for a specific input tensor
    ///
    /// # Arguments
    ///
    /// * `tensor_id` - Identifier for the tensor whose gradient is needed
    fn get_gradient(&self, tensor_id: u64) -> Option<&DenseND<T>>;
}

/// Generic adapter for external AD frameworks
///
/// This is a reference implementation showing how to wrap TenRSo
/// operations for use with an AD framework.
pub struct GenericAdAdapter<T>
where
    T: Float + 'static,
{
    /// Recorded operations in forward pass order
    operations: Vec<Box<dyn AdOperation<T>>>,

    /// Gradient storage (tensor_id -> gradient)
    gradients: std::collections::HashMap<u64, DenseND<T>>,

    /// Next operation ID
    next_op_id: u64,

    /// Recording mode flag
    recording: bool,
}

impl<T> GenericAdAdapter<T>
where
    T: Float + 'static,
{
    /// Create a new AD adapter
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            gradients: std::collections::HashMap::new(),
            next_op_id: 0,
            recording: true,
        }
    }

    /// Start recording operations
    pub fn start_recording(&mut self) {
        self.recording = true;
    }

    /// Stop recording operations
    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Get number of recorded operations
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }
}

impl<T> Default for GenericAdAdapter<T>
where
    T: Float + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AdContext<T> for GenericAdAdapter<T>
where
    T: Float + 'static,
{
    fn register_operation(&mut self, op: Box<dyn AdOperation<T>>) -> OperationId {
        if !self.recording {
            return OperationId(0); // No-op when not recording
        }

        let op_id = OperationId(self.next_op_id);
        self.next_op_id += 1;
        self.operations.push(op);
        op_id
    }

    fn backward(&mut self, _op_id: OperationId, grad: &DenseND<T>) -> Result<()> {
        // Simple reverse-mode AD: walk operations backwards
        let mut current_grads = vec![grad.clone()];

        for op in self.operations.iter().rev() {
            // Compute gradients w.r.t. inputs
            let input_grads = op.backward(&current_grads)?;

            // Store gradients (in a real implementation, would accumulate by tensor ID)
            for (idx, grad) in input_grads.iter().enumerate() {
                let tensor_id = self.operations.len() as u64 + idx as u64;
                self.gradients.insert(tensor_id, grad.clone());
            }

            // Pass gradients to previous operation
            current_grads = input_grads;
        }

        Ok(())
    }

    fn clear(&mut self) {
        self.operations.clear();
        self.gradients.clear();
        self.next_op_id = 0;
    }

    fn get_gradient(&self, tensor_id: u64) -> Option<&DenseND<T>> {
        self.gradients.get(&tensor_id)
    }
}

/// Placeholder for Tensorlogic integration
///
/// This will be implemented once Tensorlogic's API is finalized.
/// For now, this demonstrates the expected interface.
#[derive(Debug)]
pub struct TensorlogicAdapter {
    _private: (),
}

impl TensorlogicAdapter {
    /// Create a new Tensorlogic adapter
    ///
    /// # Note
    ///
    /// This is a placeholder. Actual implementation pending Tensorlogic API.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Register TenRSo operations with Tensorlogic
    ///
    /// # Note
    ///
    /// This is a placeholder. Actual implementation pending Tensorlogic API.
    pub fn register_tenrso_ops(&mut self) {
        // TODO: Register einsum, decompositions, etc. with Tensorlogic
    }

    /// Convert TenRSo tensor to Tensorlogic tensor
    ///
    /// # Note
    ///
    /// This is a placeholder. Actual implementation pending Tensorlogic API.
    pub fn to_tensorlogic<T>(&self, _tensor: &DenseND<T>) -> Result<()>
    where
        T: Float + 'static,
    {
        anyhow::bail!("Tensorlogic integration not yet implemented")
    }

    /// Convert Tensorlogic tensor to TenRSo tensor
    ///
    /// # Note
    ///
    /// This is a placeholder. Actual implementation pending Tensorlogic API.
    pub fn from_tensorlogic<T>(&self) -> Result<DenseND<T>>
    where
        T: Float + 'static,
    {
        anyhow::bail!("Tensorlogic integration not yet implemented")
    }
}

impl Default for TensorlogicAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_ad_adapter_creation() {
        let adapter = GenericAdAdapter::<f64>::new();
        assert!(adapter.is_recording());
        assert_eq!(adapter.num_operations(), 0);
    }

    #[test]
    fn test_generic_ad_adapter_recording() {
        let mut adapter = GenericAdAdapter::<f64>::new();
        assert!(adapter.is_recording());

        adapter.stop_recording();
        assert!(!adapter.is_recording());

        adapter.start_recording();
        assert!(adapter.is_recording());
    }

    #[test]
    fn test_generic_ad_adapter_clear() {
        let mut adapter = GenericAdAdapter::<f64>::new();
        adapter.clear();
        assert_eq!(adapter.num_operations(), 0);
    }

    #[test]
    fn test_tensorlogic_adapter_creation() {
        let _adapter = TensorlogicAdapter::new();
        // Just verify it can be created
    }
}
