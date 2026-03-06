//! Tests for CP decomposition module

use super::helpers::*;
use super::*;
use tenrso_core::DenseND;

// ========================================================================
// Basic CP-ALS Tests
// ========================================================================

#[test]
fn test_cp_als_basic() {
    let tensor = DenseND::<f64>::random_uniform(&[3, 4, 5], 0.0, 1.0);
    let result = cp_als(&tensor, 2, 10, 1e-4, InitStrategy::Svd, None);

    assert!(result.is_ok());
    let cp = result.expect("cp_als should succeed");

    assert_eq!(cp.factors.len(), 3);
    assert_eq!(cp.factors[0].shape(), &[3, 2]);
    assert_eq!(cp.factors[1].shape(), &[4, 2]);
    assert_eq!(cp.factors[2].shape(), &[5, 2]);
}

#[test]
fn test_gram_matrix() {
    use scirs2_core::ndarray_ext::array;

    let factor = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let gram = compute_gram_matrix(&factor);

    assert_eq!(gram.shape(), &[2, 2]);
    assert!((gram[[0, 0]] - 35.0_f64).abs() < 1e-10);
    assert!((gram[[1, 1]] - 56.0_f64).abs() < 1e-10);
}

// ========================================================================
// Non-Negative CP-ALS Tests
// ========================================================================

#[test]
fn test_cp_als_nonnegative() {
    let tensor = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
    let constraints = CpConstraints::nonnegative();
    let result = cp_als_constrained(
        &tensor,
        3,
        20,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    assert!(result.is_ok());
    let cp = result.expect("nonnegative CP should succeed");

    // Check that all factor values are non-negative
    for factor in &cp.factors {
        for &val in factor.iter() {
            assert!(
                val >= 0.0,
                "Factor value should be non-negative, got {}",
                val
            );
        }
    }
}

#[test]
fn test_cp_als_nonnegative_validates_input() {
    use scirs2_core::ndarray_ext::Array;

    // Create a tensor with negative values
    let data = Array::from_shape_vec(
        vec![3, 3, 3],
        (0..27)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect(),
    )
    .expect("shape should be valid");
    let tensor = DenseND::from_array(data.into_dyn());

    let constraints = CpConstraints::nonnegative();
    let result = cp_als_constrained(
        &tensor,
        2,
        10,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    // Should fail because tensor has negative values
    assert!(result.is_err());
    match result {
        Err(CpError::NonnegativeViolation) => {} // Expected
        Err(e) => panic!("Expected NonnegativeViolation, got {:?}", e),
        Ok(_) => panic!("Expected error for negative tensor values"),
    }
}

#[test]
fn test_cp_als_nonnegative_skip_validation() {
    use scirs2_core::ndarray_ext::Array;

    // Create a tensor with some negative values
    let data = Array::from_shape_vec(
        vec![3, 3, 3],
        (0..27)
            .map(|i| if i % 5 == 0 { -0.1 } else { 1.0 })
            .collect(),
    )
    .expect("shape should be valid");
    let tensor = DenseND::from_array(data.into_dyn());

    // Skip validation
    let constraints = CpConstraints {
        nonnegative: true,
        validate_nonneg_input: false,
        ..Default::default()
    };
    let result = cp_als_constrained(
        &tensor,
        2,
        10,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    // Should succeed because validation is skipped
    assert!(result.is_ok());
    let cp = result.expect("should succeed with skipped validation");

    // Factors should still be non-negative
    for factor in &cp.factors {
        for &val in factor.iter() {
            assert!(
                val >= 0.0,
                "Factor value should be non-negative, got {}",
                val
            );
        }
    }
}

#[test]
fn test_cp_als_nonnegative_with_nnsvd_init() {
    // NNSVD is designed for non-negative decompositions
    let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
    let constraints = CpConstraints::nonnegative();
    let result = cp_als_constrained(&tensor, 4, 30, 1e-4, InitStrategy::Nnsvd, constraints, None);

    assert!(result.is_ok());
    let cp = result.expect("NNCP with NNSVD init should succeed");

    // All factors must be non-negative
    for factor in &cp.factors {
        for &val in factor.iter() {
            assert!(val >= 0.0, "Factor should be non-negative, got {}", val);
        }
    }

    // Should achieve reasonable fit
    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

// ========================================================================
// L2 Regularization Tests
// ========================================================================

#[test]
fn test_cp_als_l2_regularized() {
    let tensor = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
    let constraints = CpConstraints::l2_regularized(0.01);
    let result = cp_als_constrained(
        &tensor,
        3,
        20,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    assert!(result.is_ok());
    let cp = result.expect("L2 regularized CP should succeed");

    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

#[test]
fn test_cp_als_l2_shrinks_factors() {
    // Higher L2 regularization should produce smaller factor norms
    let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);

    let cp_low = cp_als_constrained(
        &tensor,
        4,
        30,
        1e-4,
        InitStrategy::Svd,
        CpConstraints::l2_regularized(0.001),
        None,
    )
    .expect("low L2 should succeed");

    let cp_high = cp_als_constrained(
        &tensor,
        4,
        30,
        1e-4,
        InitStrategy::Svd,
        CpConstraints::l2_regularized(1.0),
        None,
    )
    .expect("high L2 should succeed");

    // Compute total factor norm for each
    let norm_low: f64 = cp_low
        .factors
        .iter()
        .map(|f| f.iter().map(|&v| v * v).sum::<f64>())
        .sum();
    let norm_high: f64 = cp_high
        .factors
        .iter()
        .map(|f| f.iter().map(|&v| v * v).sum::<f64>())
        .sum();

    // Higher regularization should produce smaller norms
    assert!(
        norm_high < norm_low,
        "Higher L2 reg should shrink factors: low={:.4}, high={:.4}",
        norm_low,
        norm_high
    );
}

// ========================================================================
// L1 Regularization Tests
// ========================================================================

#[test]
fn test_cp_als_l1_regularized() {
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
    let constraints = CpConstraints::l1_regularized(0.01);
    let result = cp_als_constrained(
        &tensor,
        3,
        30,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    assert!(result.is_ok());
    let cp = result.expect("L1 regularized CP should succeed");

    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

#[test]
fn test_cp_als_l1_promotes_sparsity() {
    // Higher L1 regularization should produce sparser factors
    let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);

    let cp_low = cp_als_constrained(
        &tensor,
        4,
        30,
        1e-4,
        InitStrategy::Svd,
        CpConstraints::l1_regularized(0.001),
        None,
    )
    .expect("low L1 should succeed");

    let cp_high = cp_als_constrained(
        &tensor,
        4,
        30,
        1e-4,
        InitStrategy::Svd,
        CpConstraints::l1_regularized(0.1),
        None,
    )
    .expect("high L1 should succeed");

    // Count near-zero elements (sparsity measure)
    let threshold = 1e-6;
    let sparse_count_low: usize = cp_low
        .factors
        .iter()
        .map(|f| f.iter().filter(|&&v| v.abs() < threshold).count())
        .sum();
    let sparse_count_high: usize = cp_high
        .factors
        .iter()
        .map(|f| f.iter().filter(|&&v| v.abs() < threshold).count())
        .sum();

    // Higher L1 regularization should produce more zero/near-zero elements
    assert!(
        sparse_count_high >= sparse_count_low,
        "Higher L1 should produce more sparse elements: low={}, high={}",
        sparse_count_low,
        sparse_count_high
    );
}

#[test]
fn test_soft_threshold_function() {
    use scirs2_core::ndarray_ext::array;

    let mut factor = array![[3.0_f64, -2.0, 0.5], [1.5, -0.3, 0.0]];
    soft_threshold(&mut factor, 1.0_f64);

    // Values with |x| > 1 should be shrunk by 1
    assert!((factor[[0, 0]] - 2.0_f64).abs() < 1e-10); // 3 - 1 = 2
    assert!((factor[[0, 1]] - (-1.0_f64)).abs() < 1e-10); // -2 + 1 = -1
                                                          // Values with |x| <= 1 should be zero
    assert!(factor[[0, 2]].abs() < 1e-10); // |0.5| < 1 -> 0
    assert!((factor[[1, 0]] - 0.5_f64).abs() < 1e-10); // 1.5 - 1 = 0.5
    assert!(factor[[1, 1]].abs() < 1e-10); // |-0.3| < 1 -> 0
    assert!(factor[[1, 2]].abs() < 1e-10); // 0 -> 0
}

// ========================================================================
// Elastic Net Tests
// ========================================================================

#[test]
fn test_cp_als_elastic_net() {
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
    let constraints = CpConstraints::elastic_net(0.01, 0.5); // 50% L1, 50% L2
    let result = cp_als_constrained(
        &tensor,
        3,
        30,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    assert!(result.is_ok());
    let cp = result.expect("elastic net CP should succeed");

    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

#[test]
fn test_elastic_net_components() {
    let en = RegularizationType::ElasticNet {
        lambda: 0.1,
        alpha: 0.7,
    };

    let l1 = en.l1_component();
    let l2 = en.l2_component();

    assert!((l1 - 0.07).abs() < 1e-10); // 0.7 * 0.1
    assert!((l2 - 0.03).abs() < 1e-10); // 0.3 * 0.1
}

// ========================================================================
// Tikhonov Regularization Tests
// ========================================================================

#[test]
fn test_cp_als_tikhonov_order_0() {
    // Order 0 Tikhonov = standard ridge regression
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
    let constraints = CpConstraints::tikhonov(0.01, 0);
    let result = cp_als_constrained(
        &tensor,
        3,
        30,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    assert!(result.is_ok());
    let cp = result.expect("Tikhonov order-0 should succeed");
    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

#[test]
fn test_cp_als_tikhonov_order_1() {
    // Order 1: promotes smoothness in factors
    let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
    let constraints = CpConstraints::tikhonov(0.01, 1);
    let result = cp_als_constrained(
        &tensor,
        4,
        30,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    assert!(result.is_ok());
    let cp = result.expect("Tikhonov order-1 should succeed");
    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

#[test]
fn test_cp_als_tikhonov_order_2() {
    // Order 2: stronger smoothness
    let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
    let constraints = CpConstraints::tikhonov(0.01, 2);
    let result = cp_als_constrained(
        &tensor,
        4,
        30,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    assert!(result.is_ok());
    let cp = result.expect("Tikhonov order-2 should succeed");
    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

#[test]
fn test_tikhonov_gram_modification_order_0() {
    use scirs2_core::ndarray_ext::Array2;

    let mut gram = Array2::<f64>::zeros((3, 3));
    gram[[0, 0]] = 1.0;
    gram[[1, 1]] = 2.0;
    gram[[2, 2]] = 3.0;

    apply_tikhonov_to_gram(&mut gram, 0.5, 0);

    // Should add 0.5 to diagonal
    assert!((gram[[0, 0]] - 1.5).abs() < 1e-10);
    assert!((gram[[1, 1]] - 2.5).abs() < 1e-10);
    assert!((gram[[2, 2]] - 3.5).abs() < 1e-10);
}

// ========================================================================
// Regularization Validation Tests
// ========================================================================

#[test]
fn test_regularization_validation() {
    // Valid regularizations
    assert!(RegularizationType::None.validate().is_ok());
    assert!(RegularizationType::L2 { lambda: 0.1 }.validate().is_ok());
    assert!(RegularizationType::L1 { lambda: 0.0 }.validate().is_ok());
    assert!(RegularizationType::ElasticNet {
        lambda: 0.1,
        alpha: 0.5
    }
    .validate()
    .is_ok());
    assert!(RegularizationType::Tikhonov {
        lambda: 0.1,
        order: 1
    }
    .validate()
    .is_ok());

    // Invalid regularizations
    assert!(RegularizationType::L2 { lambda: -0.1 }.validate().is_err());
    assert!(RegularizationType::L1 { lambda: -0.1 }.validate().is_err());
    assert!(RegularizationType::ElasticNet {
        lambda: 0.1,
        alpha: 1.5
    }
    .validate()
    .is_err());
    assert!(RegularizationType::Tikhonov {
        lambda: -0.1,
        order: 0
    }
    .validate()
    .is_err());
}

#[test]
fn test_constraints_effective_params() {
    // Legacy l2_reg
    let c1 = CpConstraints {
        l2_reg: 0.5,
        ..Default::default()
    };
    assert!((c1.effective_l2() - 0.5).abs() < 1e-10);
    assert!((c1.effective_l1()).abs() < 1e-10);

    // L2 regularization overrides l2_reg
    let c2 = CpConstraints {
        l2_reg: 0.5,
        regularization: RegularizationType::L2 { lambda: 0.1 },
        ..Default::default()
    };
    assert!((c2.effective_l2() - 0.1).abs() < 1e-10);

    // L1 regularization
    let c3 = CpConstraints::l1_regularized(0.3);
    assert!((c3.effective_l1() - 0.3).abs() < 1e-10);
    assert!((c3.effective_l2()).abs() < 1e-10);
}

// ========================================================================
// Constraint Combination Tests
// ========================================================================

#[test]
fn test_constraint_combinations() {
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
    let constraints = CpConstraints {
        nonnegative: true,
        l2_reg: 0.01,
        orthogonal: false,
        regularization: RegularizationType::L2 { lambda: 0.01 },
        validate_nonneg_input: true,
    };
    let result = cp_als_constrained(
        &tensor,
        3,
        20,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    assert!(result.is_ok());
    let cp = result.expect("combined constraints should succeed");

    // Check non-negativity is maintained with regularization
    for factor in &cp.factors {
        for &val in factor.iter() {
            assert!(val >= 0.0, "Factor value should be non-negative");
        }
    }

    assert!(cp.fit > 0.0);
}

#[test]
fn test_nonneg_with_l1() {
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
    let constraints = CpConstraints {
        nonnegative: true,
        regularization: RegularizationType::L1 { lambda: 0.01 },
        ..Default::default()
    };
    let result = cp_als_constrained(
        &tensor,
        3,
        30,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    assert!(result.is_ok());
    let cp = result.expect("nonneg + L1 should succeed");

    for factor in &cp.factors {
        for &val in factor.iter() {
            assert!(val >= 0.0, "Factor value should be non-negative with L1");
        }
    }
}

// ========================================================================
// Orthogonal CP Tests
// ========================================================================

#[test]
fn test_cp_als_orthogonal() {
    let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
    let constraints = CpConstraints::orthogonal();
    let result = cp_als_constrained(
        &tensor,
        4,
        10,
        1e-4,
        InitStrategy::Random,
        constraints,
        None,
    );

    if let Err(e) = &result {
        eprintln!("Orthogonal CP-ALS failed: {:?}", e);
    }
    assert!(result.is_ok());
    let cp = result.expect("orthogonal CP should succeed");

    for factor in &cp.factors {
        let gram = factor.t().dot(factor);
        for i in 0..gram.nrows() {
            for j in 0..gram.ncols() {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = gram[[i, j]];
                let diff = (actual - expected).abs();

                assert!(
                    diff < 0.1,
                    "Orthogonality check failed: gram[{},{}] = {:.4}, expected {}",
                    i,
                    j,
                    actual,
                    expected
                );
            }
        }
    }
}

// ========================================================================
// Initialization Tests
// ========================================================================

#[test]
fn test_nnsvd_initialization() {
    let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
    let result = cp_als(&tensor, 4, 20, 1e-4, InitStrategy::Nnsvd, None);

    assert!(result.is_ok());
    let cp = result.expect("NNSVD init should succeed");

    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
    assert_eq!(cp.factors.len(), 3);
    assert_eq!(cp.factors[0].shape(), &[8, 4]);
}

#[test]
fn test_leverage_score_initialization() {
    let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
    let result = cp_als(&tensor, 5, 20, 1e-4, InitStrategy::LeverageScore, None);

    assert!(result.is_ok());
    let cp = result.expect("leverage score init should succeed");

    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
    assert_eq!(cp.factors.len(), 3);
    assert!(cp.convergence.is_some());
}

// ========================================================================
// Convergence Tests
// ========================================================================

#[test]
fn test_convergence_diagnostics() {
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
    let result = cp_als(&tensor, 3, 10, 1e-4, InitStrategy::Random, None);

    assert!(result.is_ok());
    let cp = result.expect("convergence diagnostics test should succeed");

    assert!(cp.convergence.is_some());

    let conv = cp.convergence.expect("convergence should be present");

    assert!(!conv.fit_history.is_empty());
    assert!(conv.fit_history.len() <= 10);

    assert!((cp.fit - conv.fit_history[conv.fit_history.len() - 1]).abs() < 1e-10);

    match conv.reason {
        ConvergenceReason::FitTolerance
        | ConvergenceReason::MaxIterations
        | ConvergenceReason::Oscillation
        | ConvergenceReason::TimeLimit => {}
    }
}

#[test]
fn test_convergence_fit_history() {
    let tensor = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
    let result = cp_als(&tensor, 3, 30, 1e-6, InitStrategy::Svd, None);

    assert!(result.is_ok());
    let cp = result.expect("fit history test should succeed");
    let conv = cp.convergence.expect("convergence should be present");

    assert!(!conv.fit_history.is_empty());

    for &fit in &conv.fit_history {
        assert!(
            (0.0..=1.0).contains(&fit),
            "Fit value should be in [0,1], got {}",
            fit
        );
    }

    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

#[test]
fn test_oscillation_detection() {
    let tensor = DenseND::<f64>::random_uniform(&[4, 4, 4], 0.0, 1.0);
    let result = cp_als(&tensor, 2, 50, 1e-8, InitStrategy::Random, None);

    assert!(result.is_ok());
    let cp = result.expect("oscillation detection test should succeed");
    let conv = cp.convergence.expect("convergence should be present");

    assert!(conv.oscillation_count <= 50);

    if conv.oscillation_count > 0 {
        assert!(conv.oscillated);
    }
}

// ========================================================================
// Accelerated CP-ALS Tests
// ========================================================================

#[test]
fn test_cp_als_accelerated_basic() {
    let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
    let result = cp_als_accelerated(&tensor, 4, 30, 1e-4, InitStrategy::Random, None);

    assert!(result.is_ok());
    let cp = result.expect("accelerated CP should succeed");

    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
    assert_eq!(cp.factors.len(), 3);
    assert_eq!(cp.factors[0].shape(), &[8, 4]);
    assert!(cp.convergence.is_some());
}

#[test]
fn test_cp_als_accelerated_faster_convergence() {
    let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);

    let cp_standard = cp_als(&tensor, 3, 20, 1e-4, InitStrategy::Random, None)
        .expect("standard CP should succeed");

    let cp_accel = cp_als_accelerated(&tensor, 3, 20, 1e-4, InitStrategy::Random, None)
        .expect("accelerated CP should succeed");

    let fit_diff = (cp_standard.fit - cp_accel.fit).abs();
    assert!(
        fit_diff < 0.1,
        "Fits should be similar: standard={:.4}, accel={:.4}",
        cp_standard.fit,
        cp_accel.fit
    );
}

#[test]
fn test_cp_als_accelerated_reconstruction() {
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
    let cp = cp_als_accelerated(&tensor, 4, 30, 1e-4, InitStrategy::Random, None)
        .expect("accelerated CP should succeed");

    let reconstructed = cp
        .reconstruct(&[6, 6, 6])
        .expect("reconstruct should succeed");
    assert_eq!(reconstructed.shape(), &[6, 6, 6]);

    let diff = &tensor - &reconstructed;
    let error_norm = diff.frobenius_norm();
    let tensor_norm = tensor.frobenius_norm();
    let relative_error = error_norm / tensor_norm;
    let computed_fit = 1.0 - relative_error;

    let fit_diff = (cp.fit - computed_fit).abs();
    assert!(
        fit_diff < 0.1,
        "Fit should approximately match reconstruction: fit={:.4}, computed={:.4}",
        cp.fit,
        computed_fit
    );
}

// ========================================================================
// Completion Tests
// ========================================================================

#[test]
fn test_cp_completion_basic() {
    use scirs2_core::ndarray_ext::Array;

    let mut data = Array::zeros(vec![4, 4, 4]);
    let mut mask = Array::zeros(vec![4, 4, 4]);

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                if (i + j + k) % 2 == 0 {
                    data[[i, j, k]] = (i + j + k) as f64 / 10.0;
                    mask[[i, j, k]] = 1.0;
                }
            }
        }
    }

    let tensor = DenseND::from_array(data.into_dyn());
    let mask_tensor = DenseND::from_array(mask.into_dyn());

    let result = cp_completion(&tensor, &mask_tensor, 3, 50, 1e-4, InitStrategy::Random);

    assert!(result.is_ok());
    let cp = result.expect("completion should succeed");

    assert_eq!(cp.factors.len(), 3);
    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

#[test]
fn test_cp_completion_mask_validation() {
    use scirs2_core::ndarray_ext::Array;

    let data = Array::<f64, _>::zeros(vec![4, 4, 4]);
    let wrong_mask = Array::<f64, _>::zeros(vec![4, 4, 5]);

    let tensor = DenseND::from_array(data.into_dyn());
    let mask_tensor = DenseND::from_array(wrong_mask.into_dyn());

    let result = cp_completion(&tensor, &mask_tensor, 2, 50, 1e-4, InitStrategy::Random);

    assert!(result.is_err());
    match result {
        Err(CpError::ShapeMismatch(_)) => {}
        _ => panic!("Expected ShapeMismatch error"),
    }
}

#[test]
fn test_cp_completion_no_observed_entries() {
    use scirs2_core::ndarray_ext::Array;

    let data = Array::<f64, _>::zeros(vec![3, 3, 3]);
    let mask = Array::<f64, _>::zeros(vec![3, 3, 3]);

    let tensor = DenseND::from_array(data.into_dyn());
    let mask_tensor = DenseND::from_array(mask.into_dyn());

    let result = cp_completion(&tensor, &mask_tensor, 2, 50, 1e-4, InitStrategy::Random);

    assert!(result.is_err());
}

// ========================================================================
// Randomized CP Tests
// ========================================================================

#[test]
fn test_cp_randomized_basic() {
    let tensor = DenseND::<f64>::random_uniform(&[15, 15, 15], 0.0, 1.0);

    let rank = 5;
    let sketch_size = rank * 3;
    let result = cp_randomized(
        &tensor,
        rank,
        30,
        1e-4,
        InitStrategy::Random,
        sketch_size,
        5,
    );

    assert!(result.is_ok());
    let cp = result.expect("randomized CP should succeed");

    assert_eq!(cp.factors.len(), 3);
    assert!(cp.fit > 0.0 && cp.fit <= 1.0);
}

#[test]
fn test_cp_randomized_invalid_params() {
    let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);

    let result1 = cp_randomized(&tensor, 5, 20, 1e-4, InitStrategy::Random, 3, 5);
    assert!(result1.is_err(), "Should fail with sketch_size < rank");

    let result2 = cp_randomized(&tensor, 0, 20, 1e-4, InitStrategy::Random, 10, 5);
    assert!(result2.is_err(), "Should fail with rank = 0");
}

// ========================================================================
// Incremental CP Tests
// ========================================================================

#[test]
fn test_cp_incremental_append_mode() {
    let initial_tensor = DenseND::<f64>::random_uniform(&[20, 10, 10], 0.0, 1.0);
    let rank = 5;

    let cp_initial = cp_als(&initial_tensor, rank, 30, 1e-4, InitStrategy::Random, None)
        .expect("initial CP should succeed");

    let new_slice = DenseND::<f64>::random_uniform(&[5, 10, 10], 0.0, 1.0);

    let cp_updated = cp_als_incremental(
        &cp_initial,
        &new_slice,
        0,
        IncrementalMode::Append,
        10,
        1e-4,
    )
    .expect("incremental CP should succeed");

    assert_eq!(cp_updated.factors.len(), 3);
    assert_eq!(cp_updated.factors[0].shape()[0], 25);
    assert_eq!(cp_updated.factors[0].shape()[1], rank);
    assert!(cp_updated.fit > 0.0 && cp_updated.fit <= 1.0);
}

#[test]
fn test_cp_incremental_sliding_window() {
    let initial_tensor = DenseND::<f64>::random_uniform(&[20, 10, 10], 0.0, 1.0);
    let rank = 5;

    let cp_initial = cp_als(&initial_tensor, rank, 30, 1e-4, InitStrategy::Random, None)
        .expect("initial CP should succeed");

    let new_data = DenseND::<f64>::random_uniform(&[20, 10, 10], 0.0, 1.0);

    let cp_updated = cp_als_incremental(
        &cp_initial,
        &new_data,
        0,
        IncrementalMode::SlidingWindow { lambda: 0.9 },
        10,
        1e-4,
    )
    .expect("sliding window should succeed");

    assert_eq!(cp_updated.factors.len(), 3);
    assert_eq!(cp_updated.factors[0].shape()[0], 20);
    assert!(cp_updated.fit > 0.0 && cp_updated.fit <= 1.0);
}

#[test]
fn test_cp_incremental_invalid_mode() {
    let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
    let rank = 5;

    let cp = cp_als(&tensor, rank, 20, 1e-4, InitStrategy::Random, None)
        .expect("CP for invalid mode test should succeed");
    let new_data = DenseND::<f64>::random_uniform(&[5, 10, 10], 0.0, 1.0);

    let result = cp_als_incremental(&cp, &new_data, 3, IncrementalMode::Append, 5, 1e-4);

    assert!(result.is_err(), "Should fail with invalid mode");
}

#[test]
fn test_cp_incremental_shape_mismatch() {
    let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
    let rank = 5;

    let cp = cp_als(&tensor, rank, 20, 1e-4, InitStrategy::Random, None)
        .expect("CP for shape mismatch test should succeed");

    let new_data = DenseND::<f64>::random_uniform(&[5, 12, 10], 0.0, 1.0);

    let result = cp_als_incremental(&cp, &new_data, 0, IncrementalMode::Append, 5, 1e-4);

    assert!(result.is_err(), "Should fail with shape mismatch");
}

// ========================================================================
// Validate Non-negative Helper Test
// ========================================================================

#[test]
fn test_validate_nonnegative() {
    // All positive tensor
    let tensor_pos = DenseND::<f64>::random_uniform(&[3, 3, 3], 0.0, 1.0);
    assert!(validate_nonnegative(&tensor_pos).is_ok());

    // Tensor with negative values
    use scirs2_core::ndarray_ext::Array;
    let data = Array::from_shape_vec(vec![2, 2], vec![1.0, -0.5, 0.3, 2.0])
        .expect("shape should be valid");
    let tensor_neg = DenseND::from_array(data.into_dyn());
    assert!(validate_nonnegative(&tensor_neg).is_err());

    // Tensor with zeros (should pass)
    let data_zero =
        Array::from_shape_vec(vec![2, 2], vec![0.0, 0.0, 0.0, 0.0]).expect("shape should be valid");
    let tensor_zero = DenseND::from_array(data_zero.into_dyn());
    assert!(validate_nonnegative(&tensor_zero).is_ok());
}
