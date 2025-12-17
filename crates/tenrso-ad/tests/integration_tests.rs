//! Integration tests for tenrso-ad
//!
//! These tests verify end-to-end AD workflows combining multiple components.

use anyhow::Result;
use tenrso_ad::grad::{CpReconstructionGrad, TuckerReconstructionGrad};
use tenrso_ad::gradcheck::{check_gradient, GradCheckConfig};
use tenrso_ad::hooks::{AdContext, GenericAdAdapter};
use tenrso_ad::vjp::{EinsumVjp, VjpOp};
use tenrso_core::DenseND;
use tenrso_exec::ops::execute_dense_contraction;
use tenrso_planner::EinsumSpec;

use scirs2_core::ndarray_ext::Array2;

/// Test end-to-end einsum VJP with gradient checking
#[test]
fn test_einsum_vjp_with_gradcheck() -> Result<()> {
    // Matrix multiplication: C = A @ B
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;

    let spec = EinsumSpec::parse("ij,jk->ik")?;
    let c = execute_dense_contraction(&spec, &a, &b)?;

    // Test VJP for input A
    let grad_c = DenseND::ones(c.shape());
    let vjp_ctx = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
    let _grads = vjp_ctx.vjp(&grad_c)?;

    // Verify gradient w.r.t. A using finite differences
    let f = |x: &DenseND<f64>| execute_dense_contraction(&spec, x, &b);
    let df = |_x: &DenseND<f64>, grad_y: &DenseND<f64>| {
        let vjp = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
        let grads = vjp.vjp(grad_y)?;
        Ok(grads[0].clone())
    };

    let config = GradCheckConfig {
        epsilon: 1e-5,
        rtol: 1e-3,
        atol: 1e-5,
        use_central_diff: true,
        verbose: false,
    };

    let result = check_gradient(f, df, &a, &grad_c, &config)?;
    assert!(result.passed, "Gradient check failed for einsum VJP");
    assert!(result.max_rel_diff < 1e-2, "Relative error too large");

    Ok(())
}

/// Test CP reconstruction gradient with multiple factors
#[test]
fn test_cp_reconstruction_full_workflow() -> Result<()> {
    // Create a 3-mode tensor decomposition
    let rank = 3;
    let factors = vec![
        Array2::from_shape_vec(
            (4, rank),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )?,
        Array2::from_shape_vec(
            (5, rank),
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
            ],
        )?,
        Array2::from_shape_vec(
            (6, rank),
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5,
                0.0, 0.5,
            ],
        )?,
    ];

    let grad_ctx = CpReconstructionGrad::new(factors.clone(), None);
    let grad_output = DenseND::ones(&[4, 5, 6]);

    // Compute gradients
    let factor_grads = grad_ctx.compute_factor_gradients(&grad_output)?;

    // Verify shapes
    assert_eq!(factor_grads.len(), 3);
    assert_eq!(factor_grads[0].shape(), &[4, rank]);
    assert_eq!(factor_grads[1].shape(), &[5, rank]);
    assert_eq!(factor_grads[2].shape(), &[6, rank]);

    // Verify gradients are not all zeros (sanity check)
    for (i, grad) in factor_grads.iter().enumerate() {
        let sum: f64 = grad.iter().copied().sum();
        assert!(sum.abs() > 1e-6, "Gradient for factor {} is all zeros", i);
    }

    Ok(())
}

/// Test Tucker reconstruction gradient with mode-n products
#[test]
fn test_tucker_reconstruction_full_workflow() -> Result<()> {
    // Create Tucker decomposition components
    let core = DenseND::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
    )?;

    let factors = vec![
        Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.3, 0.7])?,
        Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 0.0, 1.0, 0.6, 0.4, 0.4, 0.6, 0.8, 0.2],
        )?,
        Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.3,
                0.3, 0.4,
            ],
        )?,
    ];

    let grad_ctx = TuckerReconstructionGrad::new(core.clone(), factors.clone());
    let grad_output = DenseND::ones(&[4, 5, 6]);

    // Compute gradients for core
    let grad_core = grad_ctx.compute_core_gradient(&grad_output)?;
    assert_eq!(grad_core.shape(), core.shape());

    // Compute gradients for factors
    let factor_grads = grad_ctx.compute_factor_gradients(&grad_output)?;
    assert_eq!(factor_grads.len(), 3);
    assert_eq!(factor_grads[0].shape(), &[4, 2]);
    assert_eq!(factor_grads[1].shape(), &[5, 2]);
    assert_eq!(factor_grads[2].shape(), &[6, 3]);

    // Verify non-zero gradients
    let core_sum: f64 = grad_core.as_array().iter().map(|x: &f64| x.abs()).sum();
    assert!(core_sum > 1e-6, "Core gradient is all zeros");

    Ok(())
}

/// Test chained VJP operations
#[test]
fn test_chained_vjp_operations() -> Result<()> {
    // Test: D = (A @ B) @ C
    // This requires chaining two einsum operations
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    let c = DenseND::from_vec(vec![9.0, 10.0, 11.0, 12.0], &[2, 2])?;

    // Forward pass
    let spec_ab = EinsumSpec::parse("ij,jk->ik")?;
    let spec_abc = EinsumSpec::parse("ij,jk->ik")?;

    let ab = execute_dense_contraction(&spec_ab, &a, &b)?;
    let abc = execute_dense_contraction(&spec_abc, &ab, &c)?;

    // Backward pass
    let grad_abc = DenseND::ones(abc.shape());

    // First VJP: gradient w.r.t. AB and C
    let vjp_abc = EinsumVjp::new(spec_abc, ab.clone(), c.clone());
    let grads_abc = vjp_abc.vjp(&grad_abc)?;
    let grad_ab = &grads_abc[0];
    let grad_c = &grads_abc[1];

    // Second VJP: gradient w.r.t. A and B
    let vjp_ab = EinsumVjp::new(spec_ab, a.clone(), b.clone());
    let grads_ab = vjp_ab.vjp(grad_ab)?;
    let grad_a = &grads_ab[0];
    let grad_b = &grads_ab[1];

    // Verify shapes
    assert_eq!(grad_a.shape(), a.shape());
    assert_eq!(grad_b.shape(), b.shape());
    assert_eq!(grad_c.shape(), c.shape());

    // Verify non-zero gradients
    assert!(grad_a.as_array().iter().any(|x: &f64| x.abs() > 1e-10));
    assert!(grad_b.as_array().iter().any(|x: &f64| x.abs() > 1e-10));
    assert!(grad_c.as_array().iter().any(|x: &f64| x.abs() > 1e-10));

    Ok(())
}

/// Test generic AD adapter with multiple operations
#[test]
fn test_generic_ad_adapter_workflow() -> Result<()> {
    use tenrso_ad::hooks::AdOperation;

    // Define a simple squaring operation for testing
    #[derive(Debug)]
    struct SquareOp {
        input: DenseND<f64>,
    }

    impl AdOperation<f64> for SquareOp {
        fn forward(&self) -> Result<Vec<DenseND<f64>>> {
            let mut result = self.input.clone();
            for idx in 0..result.len() {
                let multi_idx = linear_to_multi_index(result.shape(), idx);
                let val = *result.get(&multi_idx).unwrap();
                *result.get_mut(&multi_idx).unwrap() = val * val;
            }
            Ok(vec![result])
        }

        fn backward(&self, output_grads: &[DenseND<f64>]) -> Result<Vec<DenseND<f64>>> {
            let grad_out = &output_grads[0];
            let mut grad_in = DenseND::zeros(self.input.shape());

            for idx in 0..self.input.len() {
                let multi_idx = linear_to_multi_index(self.input.shape(), idx);
                let x = *self.input.get(&multi_idx).unwrap();
                let g = *grad_out.get(&multi_idx).unwrap();
                *grad_in.get_mut(&multi_idx).unwrap() = 2.0 * x * g;
            }

            Ok(vec![grad_in])
        }

        fn name(&self) -> &str {
            "square"
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn num_outputs(&self) -> usize {
            1
        }
    }

    // Helper function
    fn linear_to_multi_index(shape: &[usize], linear_idx: usize) -> Vec<usize> {
        let mut multi_idx = vec![0; shape.len()];
        let mut remaining = linear_idx;
        for (dim, &size) in shape.iter().enumerate().rev() {
            multi_idx[dim] = remaining % size;
            remaining /= size;
        }
        multi_idx
    }

    // Create adapter and register operation
    let mut adapter = GenericAdAdapter::<f64>::new();
    let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let op = Box::new(SquareOp {
        input: input.clone(),
    });

    let op_id = adapter.register_operation(op);
    assert_eq!(adapter.num_operations(), 1);

    // Test backward pass
    let grad_output = DenseND::ones(&[2, 2]);
    adapter.backward(op_id, &grad_output)?;

    // Verify adapter state
    assert!(adapter.num_operations() > 0);

    Ok(())
}

/// Test gradient checking with different configurations
#[test]
fn test_gradcheck_configurations() -> Result<()> {
    let f = |x: &DenseND<f64>| {
        let mut result = x.clone();
        for idx in 0..x.len() {
            let multi_idx = linear_to_multi_index(x.shape(), idx);
            let val = *x.get(&multi_idx).unwrap();
            // f(x) = x^3
            *result.get_mut(&multi_idx).unwrap() = val * val * val;
        }
        Ok(result)
    };

    let df = |x: &DenseND<f64>, grad_y: &DenseND<f64>| {
        let mut grad_x = DenseND::zeros(x.shape());
        for idx in 0..x.len() {
            let multi_idx = linear_to_multi_index(x.shape(), idx);
            let x_val = *x.get(&multi_idx).unwrap();
            let gy_val = *grad_y.get(&multi_idx).unwrap();
            // df/dx = 3x^2
            *grad_x.get_mut(&multi_idx).unwrap() = 3.0 * x_val * x_val * gy_val;
        }
        Ok(grad_x)
    };

    fn linear_to_multi_index(shape: &[usize], linear_idx: usize) -> Vec<usize> {
        let mut multi_idx = vec![0; shape.len()];
        let mut remaining = linear_idx;
        for (dim, &size) in shape.iter().enumerate().rev() {
            multi_idx[dim] = remaining % size;
            remaining /= size;
        }
        multi_idx
    }

    let x = DenseND::from_vec(vec![0.5, 1.0, 1.5, 2.0], &[2, 2])?;
    let grad_y = DenseND::ones(&[2, 2]);

    // Test with central difference
    let config_central = GradCheckConfig {
        epsilon: 1e-5,
        rtol: 1e-3,
        atol: 1e-5,
        use_central_diff: true,
        verbose: false,
    };
    let result = check_gradient(f, df, &x, &grad_y, &config_central)?;
    assert!(result.passed, "Central difference gradient check failed");

    // Test with forward difference
    let config_forward = GradCheckConfig {
        epsilon: 1e-5,
        rtol: 1e-2,
        atol: 1e-4,
        use_central_diff: false,
        verbose: false,
    };
    let result = check_gradient(f, df, &x, &grad_y, &config_forward)?;
    assert!(result.passed, "Forward difference gradient check failed");

    Ok(())
}

/// Test CP gradient with weighted components
#[test]
fn test_cp_gradient_with_weights() -> Result<()> {
    use scirs2_core::ndarray_ext::Array1;

    let rank = 2;
    let factors = vec![
        Array2::from_shape_vec((3, rank), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?,
        Array2::from_shape_vec((4, rank), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])?,
    ];

    let weights = Some(Array1::from_vec(vec![2.0, 3.0]));

    let grad_ctx = CpReconstructionGrad::new(factors.clone(), weights.clone());
    let grad_output = DenseND::ones(&[3, 4]);

    let factor_grads = grad_ctx.compute_factor_gradients(&grad_output)?;

    // Verify shapes
    assert_eq!(factor_grads.len(), 2);
    assert_eq!(factor_grads[0].shape(), &[3, rank]);
    assert_eq!(factor_grads[1].shape(), &[4, rank]);

    // With weights, gradients should be scaled
    for grad in &factor_grads {
        let max_val = grad.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max);
        assert!(max_val > 1e-6, "Weighted gradients are too small");
    }

    Ok(())
}

/// Test scalar output VJP end-to-end (inner product)
#[test]
fn test_scalar_output_vjp_end_to_end() -> Result<()> {
    // Test inner product: y = sum_i(a[i] * b[i])
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
    let b = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0], &[5])?;

    let spec = EinsumSpec::parse("i,i->")?;
    let y = execute_dense_contraction(&spec, &a, &b)?;

    // Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 70
    assert!(y.shape().is_empty());
    assert_eq!(*y.get(&[]).unwrap(), 70.0);

    // Compute gradients
    let grad_y = DenseND::from_elem(&[], 1.0);
    let vjp_ctx = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
    let grads = vjp_ctx.vjp(&grad_y)?;

    // Verify gradient shapes
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), a.shape());
    assert_eq!(grads[1].shape(), b.shape());

    // Verify gradient values: grad_a = b, grad_b = a
    for i in 0..5 {
        assert_eq!(*grads[0].get(&[i]).unwrap(), *b.get(&[i]).unwrap());
        assert_eq!(*grads[1].get(&[i]).unwrap(), *a.get(&[i]).unwrap());
    }

    // Verify with gradient checking
    let f = |x: &DenseND<f64>| execute_dense_contraction(&spec, x, &b);
    let df = |_x: &DenseND<f64>, grad_output: &DenseND<f64>| {
        let vjp = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
        let grads = vjp.vjp(grad_output)?;
        Ok(grads[0].clone())
    };

    let config = GradCheckConfig {
        epsilon: 1e-5,
        rtol: 1e-3,
        atol: 1e-5,
        use_central_diff: true,
        verbose: false,
    };

    let result = check_gradient(f, df, &a, &grad_y, &config)?;
    assert!(result.passed, "Gradient check failed for scalar output VJP");

    Ok(())
}

/// Test edge case: single element tensor
#[test]
fn test_single_element_tensor_vjp() -> Result<()> {
    // Test with single-element tensors
    let a = DenseND::from_vec(vec![5.0], &[1, 1])?;
    let b = DenseND::from_vec(vec![7.0], &[1, 1])?;

    let spec = EinsumSpec::parse("ij,jk->ik")?;
    let c = execute_dense_contraction(&spec, &a, &b)?;

    // Expected: 5.0 * 7.0 = 35.0
    assert_eq!(*c.get(&[0, 0]).unwrap(), 35.0);

    // Compute gradients
    let grad_c = DenseND::ones(&[1, 1]);
    let vjp_ctx = EinsumVjp::new(spec, a.clone(), b.clone());
    let grads = vjp_ctx.vjp(&grad_c)?;

    // Verify gradient shapes
    assert_eq!(grads[0].shape(), &[1, 1]);
    assert_eq!(grads[1].shape(), &[1, 1]);

    // Verify gradient values
    assert_eq!(*grads[0].get(&[0, 0]).unwrap(), 7.0);
    assert_eq!(*grads[1].get(&[0, 0]).unwrap(), 5.0);

    Ok(())
}

/// Test large tensor gradient computation (performance/memory test)
#[test]
fn test_large_tensor_gradients() -> Result<()> {
    // Create moderately large tensors (not too large for CI)
    let size = 50; // 50x50 = 2500 elements
    let a = DenseND::<f64>::ones(&[size, size]);
    let b = DenseND::<f64>::ones(&[size, size]);

    let spec = EinsumSpec::parse("ij,jk->ik")?;
    let c = execute_dense_contraction(&spec, &a, &b)?;

    // Compute gradients
    let grad_c = DenseND::ones(c.shape());
    let vjp_ctx = EinsumVjp::new(spec, a.clone(), b.clone());
    let grads = vjp_ctx.vjp(&grad_c)?;

    // Verify gradient shapes
    assert_eq!(grads[0].shape(), a.shape());
    assert_eq!(grads[1].shape(), b.shape());

    // Verify gradient values (all should be size as f64)
    assert_eq!(
        *grads[0].get(&[0, 0]).unwrap(),
        size as f64,
        "Gradient value incorrect"
    );

    Ok(())
}
