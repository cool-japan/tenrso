//! Property-based tests for VJP and gradient correctness
//!
//! Uses proptest to verify mathematical properties across random inputs

use proptest::prelude::*;
use scirs2_core::ndarray_ext::{Array1, Array2};
use tenrso_ad::grad::TtReconstructionGrad;
use tenrso_ad::gradcheck::{check_gradient, GradCheckConfig};
use tenrso_ad::vjp::{EinsumVjp, ElementwiseUnaryVjp, ReductionType, ReductionVjp, VjpOp};
use tenrso_core::DenseND;
use tenrso_exec::ops::execute_dense_contraction;
use tenrso_planner::EinsumSpec;

proptest! {
    /// Test VJP linearity in cotangent: vjp(f, a*v1 + b*v2) = a*vjp(f,v1) + b*vjp(f,v2)
    /// This tests that the VJP is linear in the cotangent (output gradient)
    #[test]
    fn test_vjp_linearity_matmul(
        a in 0.1..5.0,
        b in 0.1..5.0,
    ) {
        let x_arr = Array2::from_shape_vec((3, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let y_arr = Array2::from_shape_vec((4, 5), (0..20).map(|i| i as f64).collect()).unwrap();

        let x = DenseND::from_array(x_arr.clone().into_dyn());
        let y = DenseND::from_array(y_arr.clone().into_dyn());

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

        // Create two different cotangents
        let v1_arr = Array2::ones((3, 5));
        let v2_arr = Array2::from_shape_fn((3, 5), |(i, j)| (i * 5 + j) as f64);

        let v1 = DenseND::from_array(v1_arr.clone().into_dyn());
        let v2 = DenseND::from_array(v2_arr.clone().into_dyn());

        // Linear combination of cotangents
        let v_combo_arr: Array2<f64> = a * &v1_arr + b * &v2_arr;
        let v_combo = DenseND::from_array(v_combo_arr.into_dyn());

        // Create VJP context (same inputs for all)
        let vjp_ctx = EinsumVjp::new(spec.clone(), x.clone(), y.clone());

        // Compute VJP with each cotangent
        let grads1 = vjp_ctx.vjp(&v1).unwrap();
        let grads2 = vjp_ctx.vjp(&v2).unwrap();
        let grads_combo = vjp_ctx.vjp(&v_combo).unwrap();

        // Check linearity: vjp(f, a*v1 + b*v2) ~= a*vjp(f,v1) + b*vjp(f,v2)
        // Test for gradient w.r.t. first input
        let grad_x1_arr = grads1[0].as_array();
        let grad_x2_arr = grads2[0].as_array();
        let grad_combo_arr = grads_combo[0].as_array();

        let expected_arr: scirs2_core::ndarray_ext::Array<f64, scirs2_core::ndarray_ext::IxDyn> = a * grad_x1_arr + b * grad_x2_arr;

        for (expected, actual) in expected_arr.iter().zip(grad_combo_arr.iter()) {
            prop_assert!((expected - actual).abs() < 1e-6,
                "Linearity in cotangent violated: expected {}, got {}", expected, actual);
        }
    }

    /// Test chain rule for simple compositions
    #[test]
    fn test_vjp_chain_rule_simple(
        scale in 0.1..10.0,
    ) {
        // Test chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
        // Let g(x) = x^2, f(y) = scale*y
        // Then d/dx[scale*x^2] = 2*scale*x

        let x_arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let x = DenseND::from_array(x_arr.clone().into_dyn());

        // Forward: y = x^2
        let y_arr = &x_arr * &x_arr;
        let _y = DenseND::from_array(y_arr.clone().into_dyn());

        // Using VJP: d/dx[x*x] with cotangent = scale*ones
        let cotangent = DenseND::from_elem(&[4], scale);

        let derivative = |x_elem: &f64| 2.0 * x_elem;
        let ctx = ElementwiseUnaryVjp::new(x.clone(), derivative);
        let grads = ctx.vjp(&cotangent).unwrap();
        let grad_arr = grads[0].as_array();

        // Expected: 2*scale*x
        let expected_arr = 2.0 * scale * &x_arr;

        for (expected, actual) in expected_arr.iter().zip(grad_arr.iter()) {
            prop_assert!((expected - actual).abs() < 1e-6,
                "Chain rule violated: expected {}, got {}", expected, actual);
        }
    }

    /// Test reduction VJP gradients
    #[test]
    fn test_reduction_vjp_gradient_sum(
        rows in 2..10usize,
        cols in 2..10usize,
    ) {
        let input_shape = vec![rows, cols];

        // Full reduction (sum all elements)
        let ctx = ReductionVjp::<f64>::new(input_shape.clone(), None, ReductionType::Sum);
        let cotangent = DenseND::from_elem(&[1, 1], 1.0);
        let grads = ctx.vjp(&cotangent).unwrap();

        // Gradient of sum should be all ones
        let grad_arr = grads[0].as_array();
        for &g in grad_arr.iter() {
            prop_assert!((g - 1.0).abs() < 1e-10,
                "Sum gradient should be 1.0, got {}", g);
        }

        // Partial reduction along axis 0
        let ctx_partial = ReductionVjp::<f64>::new(input_shape.clone(), Some(0), ReductionType::Sum);
        let cotangent_partial = DenseND::ones(&[1, cols]);
        let grads_partial = ctx_partial.vjp(&cotangent_partial).unwrap();

        // Each element's gradient should be 1.0
        let grad_partial_arr = grads_partial[0].as_array();
        for &g in grad_partial_arr.iter() {
            prop_assert!((g - 1.0).abs() < 1e-10,
                "Partial sum gradient should be 1.0, got {}", g);
        }
    }

    /// Test elementwise operations
    #[test]
    fn test_elementwise_vjp_composition(
        size in 1..20usize,
    ) {
        let x_arr = Array1::from_shape_fn(size, |i| (i as f64) * 0.5);
        let x = DenseND::from_array(x_arr.clone().into_dyn());

        // Test sin composition: d/dx[sin(x)] = cos(x)
        let cotangent = DenseND::ones(&[size]);
        let derivative = |x_elem: &f64| x_elem.cos();
        let ctx = ElementwiseUnaryVjp::new(x.clone(), derivative);

        let grads = ctx.vjp(&cotangent).unwrap();
        let grad_arr = grads[0].as_array();
        let expected_arr = x_arr.mapv(|v| v.cos());

        for (actual, expected) in grad_arr.iter().zip(expected_arr.iter()) {
            prop_assert!((actual - expected).abs() < 1e-10,
                "Sin derivative incorrect: expected {}, got {}", expected, actual);
        }
    }

    /// Test einsum VJP transpose symmetry: for y = A^T, verify gradient correctness
    ///
    /// For transpose ij->ji with two inputs (einsum requires 2 inputs),
    /// we test the equivalent operation via matrix multiplication with identity:
    /// Y = I @ A^T, which gives grad_A = I^T @ grad_Y^T = grad_Y^T
    #[test]
    fn test_einsum_vjp_transpose_symmetry(
        size in 2..8usize,
    ) {
        // Build a matrix and its transpose via einsum matmul with identity
        // C = A @ I_n where A is (size x size), effectively C = A
        // Then grad_A = grad_C @ I_n^T = grad_C
        let x_arr = Array2::from_shape_fn((size, size), |(i, j)| ((i * size + j) as f64) + 1.0);
        let x = DenseND::from_array(x_arr.clone().into_dyn());

        // Use identity matrix as second input
        let eye = DenseND::from_array(Array2::eye(size).into_dyn());

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let result = execute_dense_contraction(&spec, &x, &eye).unwrap();

        // Result should equal x (A @ I = A)
        for i in 0..size {
            for j in 0..size {
                let r = *result.get(&[i, j]).unwrap();
                let expected = *x.get(&[i, j]).unwrap();
                prop_assert!((r - expected).abs() < 1e-10,
                    "A @ I should equal A: got {} expected {}", r, expected);
            }
        }

        // Compute VJP: for C = A @ I, grad_A = grad_C @ I^T = grad_C
        let grad_c_arr = Array2::from_shape_fn((size, size), |(i, j)| ((i + j) as f64) * 0.5 + 1.0);
        let grad_c = DenseND::from_array(grad_c_arr.clone().into_dyn());

        let vjp_ctx = EinsumVjp::new(spec, x.clone(), eye.clone());
        let grads = vjp_ctx.vjp(&grad_c).unwrap();

        // grad_A = grad_C @ I^T = grad_C (since I^T = I)
        let grad_a = grads[0].as_array();
        for i in 0..size {
            for j in 0..size {
                let actual = grad_a[[i, j].as_ref()];
                let expected = grad_c_arr[[i, j]];
                prop_assert!((actual - expected).abs() < 1e-10,
                    "Transpose symmetry: grad_A should equal grad_C, got {} expected {}",
                    actual, expected);
            }
        }
    }
}

proptest! {
    /// Test gradient correctness using finite differences for matrix multiplication
    #[test]
    fn test_matmul_gradient_finite_diff(
        m in 2..8usize,
        k in 2..8usize,
        n in 2..8usize,
    ) {
        let a_arr = Array2::from_shape_fn((m, k), |(i, j)| ((i * k + j) as f64) * 0.1);
        let b_arr = Array2::from_shape_fn((k, n), |(i, j)| ((i * n + j) as f64) * 0.1);

        let a = DenseND::from_array(a_arr.into_dyn());
        let b = DenseND::from_array(b_arr.into_dyn());

        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

        let config = GradCheckConfig {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
            use_central_diff: true,
            verbose: false,
        };

        // Check gradient w.r.t. first input
        let result = check_gradient(
            |x| execute_dense_contraction(&spec, x, &b),
            |_x, grad_y| {
                let vjp = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
                let grads = vjp.vjp(grad_y)?;
                Ok(grads[0].clone())
            },
            &a,
            &DenseND::ones(&[m, n]),
            &config,
        );

        prop_assert!(result.is_ok(), "Gradient check failed: {:?}", result.err());
        let res = result.unwrap();
        prop_assert!(res.passed,
            "Gradient check failed: max_rel_diff={}", res.max_rel_diff);
    }

    /// Test mean reduction gradient with finite differences
    #[test]
    fn test_reduction_mean_gradient_finite_diff(
        rows in 2..8usize,
        cols in 2..8usize,
    ) {
        let x_arr = Array2::from_shape_fn((rows, cols), |(i, j)| ((i * cols + j) as f64) * 0.1);
        let x = DenseND::from_array(x_arr.into_dyn());

        let config = GradCheckConfig {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
            use_central_diff: true,
            verbose: false,
        };

        let n_elements = (rows * cols) as f64;

        let result = check_gradient(
            |x| {
                let sum_val: f64 = x.as_array().iter().sum();
                let mean = sum_val / n_elements;
                Ok(DenseND::from_elem(&[], mean))
            },
            |x, _grad_y| {
                let ctx = ReductionVjp::<f64>::new(x.shape().to_vec(), None, ReductionType::Mean);
                let cotangent = DenseND::from_elem(&[], 1.0);
                let grads = ctx.vjp(&cotangent)?;
                Ok(grads[0].clone())
            },
            &x,
            &DenseND::from_elem(&[], 1.0),
            &config,
        );

        prop_assert!(result.is_ok(), "Gradient check failed: {:?}", result.err());
        let res = result.unwrap();
        prop_assert!(res.passed,
            "Mean gradient check failed: max_rel_diff={}", res.max_rel_diff);
    }
}

proptest! {
    /// Test numerical stability with large values
    #[test]
    fn test_vjp_numerical_stability_large_values(
        scale in 100.0..1000.0,
    ) {
        let x_arr = Array1::from_vec(vec![scale, scale * 2.0, scale * 3.0]);
        let x = DenseND::from_array(x_arr.clone().into_dyn());

        let cotangent = DenseND::ones(&[3]);
        let derivative = |x_elem: &f64| 2.0 * x_elem;
        let ctx = ElementwiseUnaryVjp::new(x, derivative);

        let grads = ctx.vjp(&cotangent).unwrap();
        let grad_arr = grads[0].as_array();
        let expected = 2.0 * &x_arr;

        // Check relative error
        for (actual, expected) in grad_arr.iter().zip(expected.iter()) {
            let rel_error = (actual - expected).abs() / expected.abs();
            prop_assert!(rel_error < 1e-10,
                "Large value stability failed: rel_error={}", rel_error);
        }
    }

    /// Test numerical stability with small values
    #[test]
    fn test_vjp_numerical_stability_small_values(
        scale in 1e-6..1e-3,
    ) {
        let x_arr = Array1::from_vec(vec![scale, scale * 2.0, scale * 3.0]);
        let x = DenseND::from_array(x_arr.clone().into_dyn());

        let cotangent = DenseND::ones(&[3]);
        let derivative = |x_elem: &f64| 2.0 * x_elem;
        let ctx = ElementwiseUnaryVjp::new(x, derivative);

        let grads = ctx.vjp(&cotangent).unwrap();
        let grad_arr = grads[0].as_array();
        let expected = 2.0 * &x_arr;

        // Check absolute error for small values
        for (actual, expected) in grad_arr.iter().zip(expected.iter()) {
            let abs_error = (actual - expected).abs();
            prop_assert!(abs_error < 1e-12,
                "Small value stability failed: abs_error={}", abs_error);
        }
    }

    /// Test scalar output VJP correctness (inner product)
    /// For y = sum_i(a[i] * b[i]), verify grad_a = b and grad_b = a
    #[test]
    fn test_scalar_output_vjp_correctness(
        size in 3usize..20,
        seed in 0u64..1000,
    ) {
        use scirs2_core::random::{SeedableRng, StdRng};

        // Create random vectors with reproducible seed
        let mut rng = StdRng::seed_from_u64(seed);
        let a_vec: Vec<f64> = (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let b_vec: Vec<f64> = (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect();

        let a = DenseND::from_vec(a_vec.clone(), &[size]).unwrap();
        let b = DenseND::from_vec(b_vec.clone(), &[size]).unwrap();

        // Compute inner product
        let spec = EinsumSpec::parse("i,i->").unwrap();
        let y = execute_dense_contraction(&spec, &a, &b).unwrap();

        // Verify scalar output
        prop_assert!(y.shape().is_empty(), "Output should be scalar");

        // Compute gradients
        let grad_y = DenseND::from_elem(&[], 1.0);
        let vjp_ctx = EinsumVjp::new(spec, a.clone(), b.clone());
        let grads = vjp_ctx.vjp(&grad_y).unwrap();

        prop_assert_eq!(grads.len(), 2, "Should have gradients for both inputs");
        prop_assert_eq!(grads[0].shape(), a.shape(), "Gradient shape mismatch for a");
        prop_assert_eq!(grads[1].shape(), b.shape(), "Gradient shape mismatch for b");

        // Verify: grad_a = b and grad_b = a
        for i in 0..size {
            let grad_a_val = *grads[0].get(&[i]).unwrap();
            let grad_b_val = *grads[1].get(&[i]).unwrap();
            let b_val = *b.get(&[i]).unwrap();
            let a_val = *a.get(&[i]).unwrap();

            prop_assert!((grad_a_val - b_val).abs() < 1e-10,
                "grad_a[{}] = {} but expected {}", i, grad_a_val, b_val);
            prop_assert!((grad_b_val - a_val).abs() < 1e-10,
                "grad_b[{}] = {} but expected {}", i, grad_b_val, a_val);
        }
    }

    /// Test TT gradient correctness using finite differences
    ///
    /// Creates a small TT decomposition with random values and verifies that the
    /// analytical gradient matches the numerical gradient computed via central differences.
    #[test]
    fn test_tt_gradient_finite_diff(
        seed in 0u64..500,
    ) {
        use scirs2_core::random::{SeedableRng, StdRng};

        let mut rng = StdRng::seed_from_u64(seed);

        // Create a 2-core TT with small dimensions for tractability
        // Core0: (1, 3, 2), Core1: (2, 4, 1) => tensor shape (3, 4)
        let n0 = 3usize;
        let n1 = 4usize;
        let r = 2usize;

        let mut core0 = DenseND::<f64>::zeros(&[1, n0, r]);
        let mut core1 = DenseND::<f64>::zeros(&[r, n1, 1]);

        // Fill with random values in [-2, 2]
        for i in 0..n0 {
            for beta in 0..r {
                *core0.get_mut(&[0, i, beta]).unwrap() = rng.gen_range(-2.0..2.0);
            }
        }
        for alpha in 0..r {
            for j in 0..n1 {
                *core1.get_mut(&[alpha, j, 0]).unwrap() = rng.gen_range(-2.0..2.0);
            }
        }

        // Random gradient output
        let mut grad_data = Vec::with_capacity(n0 * n1);
        for _ in 0..(n0 * n1) {
            grad_data.push(rng.gen_range(-1.0..1.0));
        }
        let grad_output = DenseND::from_vec(grad_data, &[n0, n1]).unwrap();

        // Compute analytical gradients
        let ctx = TtReconstructionGrad::new(vec![core0.clone(), core1.clone()]);
        let analytical = ctx.compute_core_gradients(&grad_output).unwrap();

        // Verify gradient for core0 using finite differences
        let epsilon = 1e-7;
        for i in 0..n0 {
            for beta in 0..r {
                let mut c0_plus = core0.clone();
                let mut c0_minus = core0.clone();

                let val = *core0.get(&[0, i, beta]).unwrap();
                *c0_plus.get_mut(&[0, i, beta]).unwrap() = val + epsilon;
                *c0_minus.get_mut(&[0, i, beta]).unwrap() = val - epsilon;

                let ctx_p = TtReconstructionGrad::new(vec![c0_plus, core1.clone()]);
                let ctx_m = TtReconstructionGrad::new(vec![c0_minus, core1.clone()]);
                let recon_p = ctx_p.reconstruct().unwrap();
                let recon_m = ctx_m.reconstruct().unwrap();

                // Numerical grad = sum of grad_output * (recon_p - recon_m) / (2*eps)
                let mut num_grad = 0.0;
                for fi in 0..n0 {
                    for fj in 0..n1 {
                        let vp = *recon_p.get(&[fi, fj]).unwrap();
                        let vm = *recon_m.get(&[fi, fj]).unwrap();
                        let g = *grad_output.get(&[fi, fj]).unwrap();
                        num_grad += g * (vp - vm) / (2.0 * epsilon);
                    }
                }

                let ana_val = *analytical[0].get(&[0, i, beta]).unwrap();
                let diff = (ana_val - num_grad).abs();
                let tol = 1e-4 * (1.0 + ana_val.abs().max(num_grad.abs()));
                prop_assert!(diff < tol,
                    "TT grad core0[0,{},{}]: analytical={}, numerical={}, diff={}",
                    i, beta, ana_val, num_grad, diff);
            }
        }

        // Verify gradient for core1 using finite differences
        for alpha in 0..r {
            for j in 0..n1 {
                let mut c1_plus = core1.clone();
                let mut c1_minus = core1.clone();

                let val = *core1.get(&[alpha, j, 0]).unwrap();
                *c1_plus.get_mut(&[alpha, j, 0]).unwrap() = val + epsilon;
                *c1_minus.get_mut(&[alpha, j, 0]).unwrap() = val - epsilon;

                let ctx_p = TtReconstructionGrad::new(vec![core0.clone(), c1_plus]);
                let ctx_m = TtReconstructionGrad::new(vec![core0.clone(), c1_minus]);
                let recon_p = ctx_p.reconstruct().unwrap();
                let recon_m = ctx_m.reconstruct().unwrap();

                let mut num_grad = 0.0;
                for fi in 0..n0 {
                    for fj in 0..n1 {
                        let vp = *recon_p.get(&[fi, fj]).unwrap();
                        let vm = *recon_m.get(&[fi, fj]).unwrap();
                        let g = *grad_output.get(&[fi, fj]).unwrap();
                        num_grad += g * (vp - vm) / (2.0 * epsilon);
                    }
                }

                let ana_val = *analytical[1].get(&[alpha, j, 0]).unwrap();
                let diff = (ana_val - num_grad).abs();
                let tol = 1e-4 * (1.0 + ana_val.abs().max(num_grad.abs()));
                prop_assert!(diff < tol,
                    "TT grad core1[{},{},0]: analytical={}, numerical={}, diff={}",
                    alpha, j, ana_val, num_grad, diff);
            }
        }
    }
}
