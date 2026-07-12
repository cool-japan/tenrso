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

use crate::registry::{AdScalar, OpRegistry};

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

/// Tensorlogic integration entry point.
///
/// The adapter owns an [`OpRegistry`] holding every differentiable TenRSo
/// operation (einsum, element-wise ops, reductions, CP/Tucker/TT
/// reconstruction) together with its VJP rule. That registry is what an
/// external engine consumes:
///
/// * [`TensorlogicAdapter::register_tenrso_ops`] populates it,
/// * [`TensorlogicAdapter::registered_ops`] enumerates the op names,
/// * [`TensorlogicAdapter::registry`] hands out the rules themselves.
///
/// With the (default-off) `tensorlogic` feature the adapter additionally builds
/// a [`crate::tensorlogic_bridge::TenrsoTlExecutor`] — a real implementation of
/// Tensorlogic's `TlExecutor` / `TlAutodiff` traits backed by this registry —
/// and converts tensors to and from Tensorlogic's `ArrayD`-based tensor type.
///
/// # Example
///
/// ```rust
/// use tenrso_ad::hooks::TensorlogicAdapter;
///
/// let mut adapter = TensorlogicAdapter::<f64>::new();
/// adapter.register_tenrso_ops().expect("registration must succeed on a fresh adapter");
///
/// assert!(adapter.registered_ops().contains(&"einsum"));
/// assert!(adapter.registered_ops().contains(&"cp_reconstruct"));
/// ```
#[derive(Debug)]
pub struct TensorlogicAdapter<T = f64>
where
    T: AdScalar,
{
    registry: OpRegistry<T>,
}

impl<T> TensorlogicAdapter<T>
where
    T: AdScalar,
{
    /// Create an adapter with an empty registry.
    pub fn new() -> Self {
        Self {
            registry: OpRegistry::new(),
        }
    }

    /// Create an adapter whose registry already holds every built-in op.
    pub fn with_builtins() -> Self {
        Self {
            registry: OpRegistry::with_builtins(),
        }
    }

    /// Register every TenRSo operation (forward + VJP) in this adapter's registry.
    ///
    /// This is the real registration path: after it returns, the registry
    /// contains `einsum`, the element-wise unary/binary ops, the reductions and
    /// the CP/Tucker/TT reconstruction rules, each with a gradient rule that is
    /// verified against finite differences in `tests/registry_gradcheck.rs`.
    ///
    /// # Errors
    ///
    /// Fails if one of the built-in names collides with an op the caller already
    /// registered (registration never silently shadows).
    pub fn register_tenrso_ops(&mut self) -> Result<()> {
        self.registry.register_builtins()
    }

    /// Borrow the registry.
    pub fn registry(&self) -> &OpRegistry<T> {
        &self.registry
    }

    /// Mutably borrow the registry (to add engine-specific ops).
    pub fn registry_mut(&mut self) -> &mut OpRegistry<T> {
        &mut self.registry
    }

    /// Names of every registered operation, sorted.
    pub fn registered_ops(&self) -> Vec<&str> {
        self.registry.names()
    }

    /// Build a Tensorlogic executor backed by this adapter's registry.
    ///
    /// Requires the `tensorlogic` feature.
    #[cfg(feature = "tensorlogic")]
    pub fn executor(&self) -> crate::tensorlogic_bridge::TenrsoTlExecutor<T> {
        crate::tensorlogic_bridge::TenrsoTlExecutor::with_registry(self.registry.clone())
    }

    /// Convert a TenRSo tensor into Tensorlogic's dense tensor type.
    ///
    /// Tensorlogic's SciRS2 backend represents tensors as `ArrayD<f64>`
    /// (`tensorlogic_scirs_backend::Scirs2Tensor`), so this is a genuine
    /// conversion: the element type is cast to `f64` and the shape is preserved.
    ///
    /// Requires the `tensorlogic` feature.
    ///
    /// # Errors
    ///
    /// Fails if an element cannot be represented as `f64`.
    #[cfg(feature = "tensorlogic")]
    pub fn to_tensorlogic(
        &self,
        tensor: &DenseND<T>,
    ) -> Result<scirs2_core::ndarray_ext::ArrayD<f64>> {
        crate::tensorlogic_bridge::dense_to_tl(tensor)
    }

    /// Convert a Tensorlogic dense tensor (`ArrayD<f64>`) into a TenRSo tensor.
    ///
    /// Requires the `tensorlogic` feature.
    ///
    /// # Errors
    ///
    /// Fails if an element cannot be represented in `T`.
    #[cfg(feature = "tensorlogic")]
    pub fn from_tensorlogic(
        &self,
        tensor: &scirs2_core::ndarray_ext::ArrayD<f64>,
    ) -> Result<DenseND<T>> {
        crate::tensorlogic_bridge::tl_to_dense(tensor)
    }
}

impl<T> Default for TensorlogicAdapter<T>
where
    T: AdScalar,
{
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
    fn test_tensorlogic_adapter_starts_empty() {
        let adapter = TensorlogicAdapter::<f64>::new();
        assert!(adapter.registry().is_empty());
        assert!(adapter.registered_ops().is_empty());
    }

    #[test]
    fn test_tensorlogic_adapter_registers_real_ops() {
        let mut adapter = TensorlogicAdapter::<f64>::new();
        adapter
            .register_tenrso_ops()
            .expect("registering builtins into an empty registry must succeed");

        let ops = adapter.registered_ops();
        for expected in [
            "einsum",
            "relu",
            "add",
            "reduce_sum",
            "reduce_max",
            "cp_reconstruct",
            "tucker_reconstruct",
            "tt_reconstruct",
        ] {
            assert!(
                ops.contains(&expected),
                "missing registered op '{expected}'"
            );
        }

        // The registered rules are real: run one forward + backward through the
        // registry to prove the adapter is not just holding names.
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let params = crate::registry::OpParams::einsum("ij,jk->ik");

        let c = adapter
            .registry()
            .forward("einsum", &[a.clone(), b.clone()], &params)
            .expect("einsum forward");
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);

        let grads = adapter
            .registry()
            .vjp("einsum", &[a, b], &DenseND::ones(&[2, 2]), &params)
            .expect("einsum vjp");
        assert_eq!(grads.len(), 2);
        // grad_a = grad_c @ b^T = [[11, 15], [11, 15]]
        assert_eq!(grads[0].as_slice(), &[11.0, 15.0, 11.0, 15.0]);
    }

    #[test]
    fn test_tensorlogic_adapter_double_registration_fails() {
        let mut adapter = TensorlogicAdapter::<f64>::with_builtins();
        assert!(adapter.register_tenrso_ops().is_err());
    }
}
