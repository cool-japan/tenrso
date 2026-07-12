//! CP-ALS property tests: rank monotonicity, fit consistency, weights, scaling invariance.
//!
//! Split out of the former monolithic `property_tests.rs` (see the parent module).

use super::shared::proptest_config;
use crate::{cp_als, InitStrategy};
use proptest::prelude::*;
use tenrso_core::DenseND;
// Property: CP reconstruction error should decrease with higher rank
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_reconstruction_error_decreases_with_rank(
        size in 8usize..12,
        rank1 in 3usize..5,
        rank2 in 6usize..9,
    ) {
        prop_assume!(rank1 < rank2);
        prop_assume!(rank2 < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let original_norm = tensor.frobenius_norm();

        // Decompose with lower rank (reduced iterations for speed)
        let cp1 = cp_als(&tensor, rank1, 10, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS should succeed with lower rank");
        let recon1 = cp1.reconstruct(&shape)
            .expect("Reconstruction should succeed");
        let error1 = (&tensor - &recon1)
            .frobenius_norm() / original_norm;

        // Decompose with higher rank (reduced iterations for speed)
        let cp2 = cp_als(&tensor, rank2, 10, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS should succeed with higher rank");
        let recon2 = cp2.reconstruct(&shape)
            .expect("Reconstruction should succeed");
        let error2 = (&tensor - &recon2)
            .frobenius_norm() / original_norm;

        // Higher rank should have equal or lower error (with some tolerance for numerical issues)
        prop_assert!(
            error2 <= error1 * 1.1,
            "Error with rank {} ({:.6}) should be <= error with rank {} ({:.6})",
            rank2, error2, rank1, error1
        );
    }
}

// Property: CP reconstruction error should be bounded by fit
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_reconstruction_error_matches_fit(
        size in 8usize..12,
        rank in 4usize..7,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        let cp = cp_als(&tensor, rank, 10, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS should succeed");
        let recon = cp.reconstruct(&shape)
            .expect("Reconstruction should succeed");

        let original_norm = tensor.frobenius_norm();
        let diff = &tensor - &recon;
        let error = diff.frobenius_norm();
        let relative_error = error / original_norm;
        let computed_fit = 1.0 - relative_error;

        // The fit from CP-ALS should approximately match computed fit
        prop_assert!(
            (cp.fit - computed_fit).abs() < 0.05,
            "CP fit ({:.6}) should approximately match computed fit ({:.6})",
            cp.fit, computed_fit
        );
    }
}

// Property: CP weights should be non-negative if computed
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_weights_are_nonnegative(
        size in 8usize..12,
        rank in 4usize..7,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        let cp = cp_als(&tensor, rank, 10, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS should succeed");

        if let Some(weights) = &cp.weights {
            for (i, &weight) in weights.iter().enumerate() {
                prop_assert!(
                    weight >= 0.0,
                    "Weight {} should be non-negative, got {}",
                    i, weight
                );
            }
        }
    }
}

// Property: CP reconstruction should be invariant to factor scaling
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_reconstruction_invariant_to_scaling(
        size in 8usize..12,
        rank in 4usize..6,
        scale in 0.5f64..2.0,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        let mut cp = cp_als(&tensor, rank, 10, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS should succeed");

        let recon1 = cp.reconstruct(&shape)
            .expect("Reconstruction should succeed");

        // Scale first factor up, second factor down
        cp.factors[0] = cp.factors[0].mapv(|x| x * scale);
        cp.factors[1] = cp.factors[1].mapv(|x| x / scale);

        let recon2 = cp.reconstruct(&shape)
            .expect("Reconstruction should succeed");

        // Reconstructions should be nearly identical
        let diff = &recon1 - &recon2;
        let relative_diff = diff.frobenius_norm() / recon1.frobenius_norm();

        prop_assert!(
            relative_diff < 1e-10,
            "Reconstructions should be identical after scaling, got relative diff {}",
            relative_diff
        );
    }
}
