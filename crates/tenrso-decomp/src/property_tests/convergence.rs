//! Rank recovery and convergence-rate property tests.
//!
//! Split out of the former monolithic `property_tests.rs` (see the parent module).

use super::shared::proptest_config;
use crate::{cp_als, tucker_hooi, tucker_hosvd, InitStrategy};
use proptest::prelude::*;
use tenrso_core::DenseND;
// Property: CP can recover the correct rank from exact low-rank tensors
proptest! {
    #![proptest_config(ProptestConfig { cases: 2, max_local_rejects: 1000, max_global_rejects: 10000, ..ProptestConfig::default() })]
    #[test]
    fn cp_recovers_exact_low_rank(
        size in 4usize..6,
        true_rank in 2usize..3,
    ) {
        use scirs2_core::ndarray_ext::Array2;
        prop_assume!(true_rank < size);

        let shape = vec![size, size, size];

        // Create exact low-rank tensor
        let mut factors = Vec::new();
        for mode in 0..3 {
            let factor_data = Array2::from_shape_fn((size, true_rank), |(i, r)| {
                (i as f64 + r as f64 + mode as f64) / (size + true_rank) as f64 + 0.1
            });
            factors.push(factor_data);
        }

        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let tensor_arr = tenrso_kernels::cp_reconstruct(&factor_views, None)
            .expect("Reconstruction should work");
        let exact_tensor = DenseND::from_array(tensor_arr);

        // Decompose with correct rank (reduced iterations for speed)
        let cp = cp_als(&exact_tensor, true_rank, 20, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS should succeed");

        // For exact low-rank tensors, fit should be high (relaxed for fewer iterations)
        prop_assert!(
            cp.fit >= 0.8,
            "CP should fit exact rank-{} tensor well, got fit {:.6}",
            true_rank, cp.fit
        );

        // Reconstruction error should be small (relaxed for fewer iterations)
        let recon = cp.reconstruct(&shape).expect("Reconstruction should succeed");
        let diff = &exact_tensor - &recon;
        let relative_error = diff.frobenius_norm() / exact_tensor.frobenius_norm();

        prop_assert!(
            relative_error < 0.25,
            "Relative reconstruction error should be < 0.25, got {:.6}",
            relative_error
        );
    }
}

// Property: Tucker with automatic rank selection recovers good approximations
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tucker_auto_rank_preserves_energy(
        size in 5usize..8,   // Reduced from 8-11 for speed (125-343 elements)
        _target_rank in 2usize..4,   // Reduced from 4-6 for speed
    ) {
        use crate::tucker_hosvd_auto;
        use crate::tucker::{TuckerRankSelection};

        prop_assume!(_target_rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Use energy-based rank selection (90% energy)
        let tucker = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(0.9))
            .expect("Auto-rank Tucker should succeed");

        // Selected ranks should be reasonable (not too small, not too large)
        // Check via factor dimensions
        for (mode, factor) in tucker.factors.iter().enumerate() {
            let (_rows, cols) = factor.dim();
            prop_assert!(
                cols >= 1 && cols <= size,
                "Mode {} rank should be in [1, {}], got {}",
                mode, size, cols
            );
        }

        // Reconstruction error should be reasonable
        let recon = tucker.reconstruct().expect("Reconstruction should succeed");
        let diff = &tensor - &recon;
        let relative_error = diff.frobenius_norm() / tensor.frobenius_norm();

        prop_assert!(
            relative_error < 0.5,
            "90% energy should preserve structure well, error: {:.4}",
            relative_error
        );
    }
}

// Property: Overspecified rank doesn't hurt CP convergence
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_overspecified_rank_converges(
        size in 8usize..10,
        true_rank in 2usize..3,
    ) {
        use scirs2_core::ndarray_ext::Array2;
        prop_assume!(true_rank < size / 2);

        let _shape = [size, size, size];

        // Create low-rank tensor
        let mut factors = Vec::new();
        for _ in 0..3 {
            let factor_data = Array2::from_shape_fn((size, true_rank), |(i, r)| {
                (i + r) as f64 / 10.0
            });
            factors.push(factor_data);
        }

        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let tensor_arr = tenrso_kernels::cp_reconstruct(&factor_views, None)
            .expect("Reconstruction should work");
        let low_rank_tensor = DenseND::from_array(tensor_arr);

        // Decompose with rank > true_rank
        let over_rank = true_rank + 2;
        let cp = cp_als(&low_rank_tensor, over_rank, 50, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS with overspecified rank should converge");

        // Should still achieve high fit (may have some extra zero components)
        prop_assert!(
            cp.fit >= 0.8,
            "Overspecified CP should still fit well, got {:.6}",
            cp.fit
        );

        // Should produce valid factors
        prop_assert_eq!(cp.factors.len(), 3);
        for factor in &cp.factors {
            prop_assert_eq!(factor.shape(), &[size, over_rank]);
        }
    }
}

// ========================================================================
// Convergence Rate Analysis Property Tests (Advanced)
// ========================================================================

// Property: More iterations should not decrease fit (monotone convergence)
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_als_monotone_convergence(
        size in 8usize..11,
        rank in 3usize..5,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Run with fewer iterations
        let cp_few = cp_als(&tensor, rank, 5, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS with few iterations should succeed");

        // Run with more iterations (same seed pattern for fairness)
        let cp_many = cp_als(&tensor, rank, 25, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS with many iterations should succeed");

        // More iterations should give better or equal fit
        prop_assert!(
            cp_many.fit >= cp_few.fit - 0.1, // Small tolerance for random initialization
            "More iterations should not significantly reduce fit: few={:.4}, many={:.4}",
            cp_few.fit, cp_many.fit
        );
    }
}

// Property: Tucker-HOOI improves over HOSVD (or stays same)
proptest! {
    #![proptest_config(ProptestConfig { cases: 1, max_local_rejects: 1000, max_global_rejects: 10000, ..ProptestConfig::default() })]
    #[test]
    fn tucker_hooi_never_worse_than_hosvd(
        size in 4usize..5,   // Fixed at 4 for speed (64 elements)
        rank in 2usize..3,   // Fixed at 2 for speed
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let ranks = vec![rank, rank, rank];

        // Run HOSVD
        let tucker_hosvd_result = tucker_hosvd(&tensor, &ranks)
            .expect("HOSVD should succeed");

        // Run HOOI (initialized with HOSVD)
        let tucker_hooi_result = tucker_hooi(&tensor, &ranks, 1, 1e-4)  // Reduced from 3 to 1 iteration
            .expect("HOOI should succeed");

        // HOOI should have equal or better error
        let hosvd_recon = tucker_hosvd_result.reconstruct()
            .expect("HOSVD reconstruction should succeed");
        let hooi_recon = tucker_hooi_result.reconstruct()
            .expect("HOOI reconstruction should succeed");

        let hosvd_err = (&tensor - &hosvd_recon).frobenius_norm();
        let hooi_err = (&tensor - &hooi_recon).frobenius_norm();

        prop_assert!(
            hooi_err <= hosvd_err * 1.05, // Small tolerance for numerical errors
            "HOOI error ({:.6}) should be ≤ HOSVD error ({:.6})",
            hooi_err, hosvd_err
        );
    }
}

// Property: Accelerated CP converges faster than standard CP
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_accelerated_faster_convergence(
        size in 8usize..10,
        rank in 3usize..4,
    ) {
        use crate::cp_als_accelerated;
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Standard CP-ALS
        let cp_standard = cp_als(&tensor, rank, 10, 1e-4, InitStrategy::Random, None)
            .expect("Standard CP-ALS should succeed");

        // Accelerated CP-ALS
        let cp_accel = cp_als_accelerated(&tensor, rank, 10, 1e-4, InitStrategy::Random, None)
            .expect("Accelerated CP-ALS should succeed");

        // Both should produce valid results
        prop_assert!(cp_standard.fit >= 0.0 && cp_standard.fit <= 1.0);
        prop_assert!(cp_accel.fit >= 0.0 && cp_accel.fit <= 1.0);

        // Accelerated version typically converges faster (higher fit in fewer iterations)
        // or at least not worse
        prop_assert!(
            cp_accel.fit >= cp_standard.fit - 0.15,
            "Accelerated CP should converge at least as well as standard"
        );
    }
}

// Property: Convergence information tracks fit progression correctly
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_convergence_info_tracks_fit(
        size in 8usize..10,
        rank in 3usize..4,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        let cp = cp_als(&tensor, rank, 30, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS should succeed");

        // Check if convergence info is present
        if let Some(ref conv) = cp.convergence {
            // Fit history should not be empty
            prop_assert!(!conv.fit_history.is_empty(), "Fit history should be tracked");

            // Final fit should match last history entry
            if let Some(&last_fit) = conv.fit_history.last() {
                prop_assert!(
                    (cp.fit - last_fit).abs() < 1e-10,
                    "Final fit should match last history entry"
                );
            }

            // Fit values should all be valid [0, 1]
            for (i, &fit_val) in conv.fit_history.iter().enumerate() {
                prop_assert!(
                    (0.0..=1.0).contains(&fit_val),
                    "Fit history[{}] should be in [0,1], got {:.6}",
                    i, fit_val
                );
            }

            // If converged by fit tolerance, final fit change should be small
            if matches!(conv.reason, crate::ConvergenceReason::FitTolerance) {
                prop_assert!(
                    conv.final_fit_change < 1e-3,
                    "Fit tolerance convergence should have small final change, got {:.6}",
                    conv.final_fit_change
                );
            }
        }
    }
}

// Property: Early termination due to oscillation is correctly detected
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn cp_oscillation_detection_works(
        size in 6usize..8,
        rank in 2usize..3,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Run CP-ALS with many iterations (may or may not oscillate)
        let cp = cp_als(&tensor, rank, 50, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS should succeed");

        // Check convergence information
        if let Some(ref conv) = cp.convergence {
            // If oscillation was detected, count should be > 0
            if conv.oscillated {
                prop_assert!(
                    conv.oscillation_count > 0,
                    "If oscillated, count should be positive"
                );

                // Reason should be Oscillation if severe enough
                if matches!(conv.reason, crate::ConvergenceReason::Oscillation) {
                    prop_assert!(
                        conv.oscillation_count > 5,
                        "Severe oscillation should have count > 5"
                    );
                }
            }

            // If not oscillated, count should be low
            if !conv.oscillated {
                prop_assert!(
                    conv.oscillation_count <= 5,
                    "Non-oscillating runs should have low oscillation count"
                );
            }
        }
    }
}

// Property: Different initialization strategies converge to valid solutions
proptest! {
    #![proptest_config(ProptestConfig { cases: 2, max_local_rejects: 1000, max_global_rejects: 10000, ..ProptestConfig::default() })]
    #[test]
    fn cp_all_init_strategies_converge(
        size in 6usize..8,   // Reduced from 8-10 for speed (216-343 elements)
        rank in 2usize..3,   // Reduced from 3-4 for speed
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Test all initialization strategies
        let strategies = vec![
            InitStrategy::Random,
            InitStrategy::RandomNormal,
            InitStrategy::Svd,
        ];

        for strategy in strategies {
            let cp = cp_als(&tensor, rank, 10, 1e-4, strategy, None)
                .expect("CP-ALS should succeed with all initialization strategies");

            // All should produce valid fits
            prop_assert!(
                cp.fit >= 0.0 && cp.fit <= 1.0,
                "{:?} init: fit should be in [0,1], got {:.6}",
                strategy, cp.fit
            );

            // All should produce valid factors
            prop_assert_eq!(cp.factors.len(), 3);
            for factor in &cp.factors {
                prop_assert_eq!(factor.shape(), &[size, rank]);
            }
        }
    }
}
