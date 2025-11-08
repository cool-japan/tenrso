//! # tenrso-exec
//!
//! Unified execution API for TenRSo.
//!
//! This crate provides:
//! - `einsum_ex` - the main public API for tensor contractions
//! - `TenrsoExecutor` trait for different backends
//! - CPU executor with automatic representation selection
//! - Integration with planner and all backend operations

#![deny(warnings)]

pub mod executor;
pub mod hints;
pub mod ops;

// Re-exports
pub use executor::*;
pub use hints::*;

use scirs2_core::numeric::{Float, FromPrimitive, Num};

/// Execute an einsum contraction with hints
///
/// # Example
/// ```ignore
/// use tenrso_exec::{einsum_ex, ExecHints};
///
/// let y = einsum_ex::<f32>("bij,bjk->bik")
///     .inputs(&[A, B])
///     .hints(&ExecHints::default())
///     .run()?;
/// ```
pub fn einsum_ex<T>(spec: &str) -> EinsumBuilder<'_, T>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
{
    EinsumBuilder::new(spec)
}

/// Builder for einsum operations
pub struct EinsumBuilder<'a, T>
where
    T: Clone + Num + Float + FromPrimitive,
{
    spec: String,
    inputs: Option<&'a [tenrso_core::TensorHandle<T>]>,
    hints: ExecHints,
}

impl<'a, T> EinsumBuilder<'a, T>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
{
    /// Create a new einsum builder
    pub fn new(spec: impl Into<String>) -> Self {
        Self {
            spec: spec.into(),
            inputs: None,
            hints: ExecHints::default(),
        }
    }

    /// Set input tensors
    ///
    /// # Arguments
    ///
    /// * `inputs` - Slice of tensor handles to use as inputs
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = einsum_ex::<f32>("ij,jk->ik")
    ///     .inputs(&[tensor_a, tensor_b])
    ///     .run()?;
    /// ```
    pub fn inputs(mut self, inputs: &'a [tenrso_core::TensorHandle<T>]) -> Self {
        self.inputs = Some(inputs);
        self
    }

    /// Set execution hints
    ///
    /// # Arguments
    ///
    /// * `hints` - Execution hints for optimization
    ///
    /// # Example
    ///
    /// ```ignore
    /// let hints = ExecHints {
    ///     prefer_sparse: true,
    ///     ..Default::default()
    /// };
    /// let result = einsum_ex::<f32>("ij,jk->ik")
    ///     .inputs(&[tensor_a, tensor_b])
    ///     .hints(&hints)
    ///     .run()?;
    /// ```
    pub fn hints(mut self, hints: &ExecHints) -> Self {
        self.hints = hints.clone();
        self
    }

    /// Execute the einsum operation
    ///
    /// # Returns
    ///
    /// Result containing the output tensor handle
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No inputs were provided
    /// - Input count doesn't match the einsum spec
    /// - Execution fails
    pub fn run(self) -> anyhow::Result<tenrso_core::TensorHandle<T>> {
        let inputs = self
            .inputs
            .ok_or_else(|| anyhow::anyhow!("No inputs provided to einsum_ex"))?;

        // Use CpuExecutor for execution
        let mut executor = CpuExecutor::new();
        executor.einsum(&self.spec, inputs, &self.hints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenrso_core::{DenseND, TensorHandle};

    #[test]
    fn test_einsum_ex_builder_matmul() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);

        let result = einsum_ex::<f64>("ij,jk->ik")
            .inputs(&[handle_a, handle_b])
            .run()
            .unwrap();

        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);

        // Verify computation: [1*5+2*7, 1*6+2*8] = [19, 22]
        //                      [3*5+4*7, 3*6+4*8] = [43, 50]
        let result_view = result_dense.view();
        let diff1: f64 = result_view[[0, 0]] - 19.0;
        let diff2: f64 = result_view[[0, 1]] - 22.0;
        assert!(diff1.abs() < 1e-10);
        assert!(diff2.abs() < 1e-10);
    }

    #[test]
    fn test_einsum_ex_builder_with_hints() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);

        let hints = ExecHints::default();

        let result = einsum_ex::<f64>("ij,jk->ik")
            .inputs(&[handle_a, handle_b])
            .hints(&hints)
            .run()
            .unwrap();

        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
    }

    #[test]
    fn test_einsum_ex_builder_three_tensors() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let c = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let handle_c = TensorHandle::from_dense_auto(c);

        let result = einsum_ex::<f64>("ij,jk,kl->il")
            .inputs(&[handle_a, handle_b, handle_c])
            .run()
            .unwrap();

        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);

        // Verify non-zero result (detailed computation would be complex)
        let result_view = result_dense.view();
        let val: f64 = result_view[[0, 0]];
        assert!(val.abs() > 0.0);
    }

    #[test]
    fn test_einsum_ex_builder_no_inputs() {
        let result = einsum_ex::<f64>("ij,jk->ik").run();

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No inputs provided"));
    }
}
