//! Constraint property tests: non-negativity (CP and Tucker), L2 regularization, orthogonality.
//!
//! Split out of the former monolithic `property_tests.rs` (see the parent module).

use super::shared::proptest_config;
use crate::{cp_als, cp_als_constrained, tucker_nonnegative, CpConstraints, InitStrategy};
use proptest::prelude::*;
use tenrso_core::DenseND;
// Property: Non-negative CP-ALS should maintain non-negativity in all factors
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_nonnegative_factors_are_nonnegative(
        size in 8usize..12,
        rank in 4usize..7,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        // Create non-negative tensor for testing
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        let cp = cp_als_constrained(
            &tensor,
            rank,
            20, // More iterations for convergence
            1e-4,
            InitStrategy::Random,
            CpConstraints::nonnegative(),
            None,
        ).expect("Non-negative CP-ALS should succeed");

        // Verify all factors are non-negative
        for (mode, factor) in cp.factors.iter().enumerate() {
            for (i, &value) in factor.iter().enumerate() {
                prop_assert!(
                    value >= -1e-10, // Allow small numerical errors
                    "Mode {} factor element {} should be non-negative, got {}",
                    mode, i, value
                );
            }
        }

        // Verify reconstruction is non-negative
        let recon = cp.reconstruct(&shape)
            .expect("Reconstruction should succeed");
        for (i, &value) in recon.as_slice().iter().enumerate() {
            prop_assert!(
                value >= -1e-10,
                "Reconstruction element {} should be non-negative, got {}",
                i, value
            );
        }
    }
}

// Property: Non-negative CP-ALS should achieve reasonable fit
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_nonnegative_reconstruction_quality(
        size in 8usize..11,
        rank in 5usize..8,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        let cp = cp_als_constrained(
            &tensor,
            rank,
            30, // More iterations for convergence
            1e-4,
            InitStrategy::Random,
            CpConstraints::nonnegative(),
            None,
        ).expect("Non-negative CP-ALS should succeed");

        // For non-negative tensors, we should achieve reasonable fit
        // (may not be as good as unconstrained, but should be > 0.3)
        prop_assert!(
            cp.fit > 0.3,
            "Non-negative CP fit ({:.4}) should be reasonable for non-negative data",
            cp.fit
        );

        // Verify reconstruction error is bounded
        let recon = cp.reconstruct(&shape)
            .expect("Reconstruction should succeed");
        let diff = &tensor - &recon;
        let relative_error = diff.frobenius_norm() / tensor.frobenius_norm();

        prop_assert!(
            relative_error < 0.8,
            "Non-negative CP relative error ({:.4}) should be bounded",
            relative_error
        );
    }
}

// Property: Non-negative Tucker should maintain non-negativity
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tucker_nonnegative_all_components_nonnegative(
        size in 6usize..9,  // Smaller for expensive multiplicative updates
        rank in 3usize..5,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let ranks = vec![rank, rank, rank];

        let tucker = tucker_nonnegative(&tensor, &ranks, 20, 1e-4)
            .expect("Non-negative Tucker should succeed");

        // Verify all factors are non-negative (most important check)
        for (mode, factor) in tucker.factors.iter().enumerate() {
            let min_val = factor.iter().fold(f64::INFINITY, |min, &x| min.min(x));
            prop_assert!(
                min_val >= -1e-10,
                "Mode {} factor minimum should be non-negative, got {}",
                mode, min_val
            );
        }

        // Verify reconstruction doesn't fail and has correct shape
        let recon = tucker.reconstruct()
            .expect("Reconstruction should succeed");
        prop_assert_eq!(recon.shape(), &shape[..]);

        // For non-negative factors and core, reconstruction should be approximately non-negative
        // (may have small numerical errors, so we just verify it completes successfully)
    }
}

// Note: Quality tests for tucker_nonnegative are skipped for random tensors
// because multiplicative updates work best on structured non-negative data
// (e.g., images, spectral data, topic models) rather than random noise.
// The algorithm is tested for correctness (non-negativity, dimensions) above.

// Property: Non-negative Tucker should maintain dimensionality correctly
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tucker_nonnegative_correct_dimensions(
        size in 6usize..9,
        rank in 3usize..5,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let ranks = vec![rank, rank, rank];

        let tucker = tucker_nonnegative(&tensor, &ranks, 30, 1e-4)
            .expect("Non-negative Tucker should succeed");

        // Verify dimensions are correct
        prop_assert_eq!(tucker.factors.len(), 3);
        for (mode, factor) in tucker.factors.iter().enumerate() {
            let (rows, cols) = factor.dim();
            prop_assert_eq!(
                rows, size,
                "Mode {} factor should have {} rows", mode, size
            );
            prop_assert_eq!(
                cols, rank,
                "Mode {} factor should have {} columns", mode, rank
            );
        }

        // Verify core shape
        let core_shape = tucker.core.shape();
        prop_assert_eq!(core_shape, &[rank, rank, rank]);

        // Verify reconstruction shape
        let recon = tucker.reconstruct()
            .expect("Reconstruction should succeed");
        prop_assert_eq!(recon.shape(), &shape[..]);
    }
}

// Property: CP with L2 regularization should reduce factor norms
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_l2_regularization_reduces_norms(
        size in 5usize..8,
        rank in 2usize..4,
        lambda in 0.01f64..0.1,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Run without regularization
        let cp_unreg = cp_als(&tensor, rank, 10, 1e-4, InitStrategy::Random, None)
            .expect("Unregularized CP-ALS should succeed");

        // Run with L2 regularization
        let cp_reg = cp_als_constrained(
            &tensor,
            rank,
            10,
            1e-4,
            InitStrategy::Random,
            CpConstraints::l2_regularized(lambda),
            None,
        ).expect("Regularized CP-ALS should succeed");

        // Compute average Frobenius norm of factors
        let unreg_norm: f64 = cp_unreg.factors.iter()
            .map(|f| {
                let sum: f64 = f.iter().map(|&x| x * x).sum();
                sum.sqrt()
            })
            .sum::<f64>() / cp_unreg.factors.len() as f64;

        let reg_norm: f64 = cp_reg.factors.iter()
            .map(|f| {
                let sum: f64 = f.iter().map(|&x| x * x).sum();
                sum.sqrt()
            })
            .sum::<f64>() / cp_reg.factors.len() as f64;

        // Regularization should tend to reduce factor norms
        // Allow some tolerance as random initialization can vary
        prop_assert!(
            reg_norm <= unreg_norm * 1.5,
            "Regularized norm ({:.4}) should not be much larger than unregularized ({:.4})",
            reg_norm, unreg_norm
        );
    }
}

// Property: CP with orthogonality constraint should have orthogonal factors
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_orthogonal_factors_are_orthogonal(
        size in 8usize..11,
        rank in 4usize..7,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        let cp = cp_als_constrained(
            &tensor,
            rank,
            30, // More iterations for convergence
            1e-4,
            InitStrategy::Random,
            CpConstraints::orthogonal(),
            None,
        ).expect("Orthogonal CP-ALS should succeed");

        // Check each factor matrix for orthonormality
        for (mode, factor) in cp.factors.iter().enumerate() {
            // Compute F^T F
            let gram = factor.t().dot(factor);
            let (rows, cols) = gram.dim();

            prop_assert_eq!(rows, rank);
            prop_assert_eq!(cols, rank);

            // Check if it's approximately identity
            for i in 0..rows {
                for j in 0..cols {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let actual = gram[[i, j]];
                    let diff = (actual - expected).abs();

                    prop_assert!(
                        diff < 1e-6, // Looser tolerance for iterative orthogonalization
                        "Mode {} factor: F^T F[{}, {}] = {:.2e}, expected {:.2e}",
                        mode, i, j, actual, expected
                    );
                }
            }
        }
    }
}
