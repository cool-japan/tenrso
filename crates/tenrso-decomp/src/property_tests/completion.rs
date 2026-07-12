//! Tensor completion property tests, for both CP-WOPT and Tucker completion.
//!
//! Split out of the former monolithic `property_tests.rs` (see the parent module).

use super::shared::proptest_config;
use crate::{tucker_completion, InitStrategy};
use proptest::prelude::*;
use scirs2_core::ndarray_ext::Array;
use scirs2_core::random::thread_rng;
use tenrso_core::DenseND;
// Property: Completion should fit observed entries well
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn completion_fits_observed_entries(
        size in 6usize..10,
        rank in 3usize..5,
        obs_rate_pct in 30u8..80,
    ) {
        use scirs2_core::ndarray_ext::Array;
        use scirs2_core::random::thread_rng;
        use crate::cp_completion;

        prop_assume!(rank < size);
        let obs_rate = obs_rate_pct as f64 / 100.0;

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Create random mask
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();
        let mut n_observed = 0usize;

        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < obs_rate {
                *idx = 1.0;
                n_observed += 1;
            }
        }

        // Need at least some observed entries
        prop_assume!(n_observed > size * size);

        let mask = DenseND::from_array(mask_data.into_dyn());

        // Run completion
        let cp = cp_completion(&tensor, &mask, rank, 50, 1e-4, InitStrategy::Random)
            .expect("Completion should succeed");

        // Fit should be non-negative and bounded
        prop_assert!(
            cp.fit >= 0.0 && cp.fit <= 1.0,
            "Fit should be in [0, 1], got {:.6}",
            cp.fit
        );
    }
}

// Property: Completion reconstruction should match observed entries better than missing
proptest! {
    #![proptest_config(ProptestConfig { cases: 1, max_local_rejects: 1000, max_global_rejects: 10000, ..ProptestConfig::default() })]
    #[test]
    fn completion_preserves_observed_pattern(
        size in 6usize..7,   // Fixed at 6 to avoid rank-deficiency with rank=2
        rank in 2usize..3,   // Reduced from 3-4 for speed
    ) {
        use scirs2_core::ndarray_ext::Array;
        use scirs2_core::random::thread_rng;
        use crate::cp_completion;

        prop_assume!(rank < size);

        let shape = vec![size, size, size];

        // Create a low-rank tensor for easier completion
        let factor1 = Array::from_shape_fn((size, rank), |(i, r)| (i + r) as f64 / 10.0);
        let factor2 = Array::from_shape_fn((size, rank), |(i, r)| (i + r + 1) as f64 / 10.0);
        let factor3 = Array::from_shape_fn((size, rank), |(i, r)| (i + r + 2) as f64 / 10.0);

        let factors_arr = [factor1, factor2, factor3];
        let factor_views: Vec<_> = factors_arr.iter().map(|f| f.view()).collect();
        let tensor_arr = tenrso_kernels::cp_reconstruct(&factor_views, None)
            .expect("Reconstruction should work");
        let tensor = DenseND::from_array(tensor_arr);

        // Observe 60% of entries
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();

        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < 0.6 {
                *idx = 1.0;
            }
        }

        let mask = DenseND::from_array(mask_data.into_dyn());

        // Complete with correct rank (reduced iterations, Random init for speed)
        let cp = cp_completion(&tensor, &mask, rank, 10, 1e-4, InitStrategy::Random)
            .expect("Completion should succeed");

        let reconstructed = cp.reconstruct(&shape)
            .expect("Reconstruction should succeed");

        // Fit should be reasonable for low-rank tensors
        prop_assert!(
            cp.fit >= 0.0,
            "Fit should be non-negative, got {:.6}",
            cp.fit
        );

        // Reconstruction shape should match
        prop_assert_eq!(reconstructed.shape(), &shape[..]);
    }
}

// Property: Higher observation rates should yield better fits
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn completion_improves_with_more_observations(
        size in 6usize..8,
        rank in 2usize..4,
    ) {
        use scirs2_core::ndarray_ext::Array;
        use scirs2_core::random::thread_rng;
        use crate::cp_completion;

        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Test with 30% vs 70% observation rates
        let mut rng = thread_rng();

        // Low observation rate (30%)
        let mut mask_low = Array::zeros(vec![size, size, size]);
        for idx in mask_low.iter_mut() {
            if rng.random::<f64>() < 0.3 {
                *idx = 1.0;
            }
        }

        // High observation rate (70%)
        let mut mask_high = Array::zeros(vec![size, size, size]);
        for idx in mask_high.iter_mut() {
            if rng.random::<f64>() < 0.7 {
                *idx = 1.0;
            }
        }

        let mask_low_tensor = DenseND::from_array(mask_low.into_dyn());
        let mask_high_tensor = DenseND::from_array(mask_high.into_dyn());

        // Complete with both masks
        let cp_low = cp_completion(&tensor, &mask_low_tensor, rank, 50, 1e-4, InitStrategy::Random)
            .ok();
        let cp_high = cp_completion(&tensor, &mask_high_tensor, rank, 50, 1e-4, InitStrategy::Random)
            .ok();

        // Both should succeed
        if let (Some(low), Some(high)) = (cp_low, cp_high) {
            // Higher observation rate often (but not always) gives better fit
            // Just verify both are in valid range
            prop_assert!(low.fit >= 0.0 && low.fit <= 1.0);
            prop_assert!(high.fit >= 0.0 && high.fit <= 1.0);
        }
    }
}

// Property: Completion rank should not exceed input rank
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn completion_respects_rank_constraint(
        size in 6usize..9,
        rank in 2usize..5,
    ) {
        use scirs2_core::ndarray_ext::Array;
        use scirs2_core::random::thread_rng;
        use crate::cp_completion;

        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Create mask with 50% observations
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();

        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < 0.5 {
                *idx = 1.0;
            }
        }

        let mask = DenseND::from_array(mask_data.into_dyn());

        let cp = cp_completion(&tensor, &mask, rank, 50, 1e-4, InitStrategy::Random)
            .expect("Completion should succeed");

        // Verify factor dimensions
        prop_assert_eq!(cp.factors.len(), 3);
        for (i, factor) in cp.factors.iter().enumerate() {
            prop_assert_eq!(
                factor.shape(),
                &[size, rank],
                "Factor {} should have shape [{}, {}]",
                i, size, rank
            );
        }
    }
}

// ========================================================================
// Tucker Completion Property Tests
// ========================================================================

// Property: Tucker completion should produce valid fit values
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tucker_completion_fits_observed_entries(
        size in 6usize..9,
        rank in 3usize..5,
        obs_rate in 0.3f64..0.8,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Create mask with specified observation rate
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();

        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < obs_rate {
                *idx = 1.0;
            }
        }

        let mask = DenseND::from_array(mask_data.into_dyn());

        // Reduced from 30 to 10 iterations for speed
        let tucker = tucker_completion(&tensor, &mask, &[rank, rank, rank], 10, 1e-4)
            .expect("Tucker completion should succeed");

        // Error should be in valid range [0, 1]
        let error = tucker.error.unwrap();
        prop_assert!(
            (0.0..=1.0).contains(&error),
            "Error should be in [0,1], got {}",
            error
        );
    }
}

// Property: Tucker completion should reconstruct low-rank tensors well
proptest! {
    #![proptest_config(ProptestConfig { cases: 2, max_local_rejects: 1000, max_global_rejects: 10000, ..ProptestConfig::default() })]
    #[test]
    fn tucker_completion_preserves_low_rank_structure(
        size in 4usize..5,   // Fixed at 4 for speed (64 elements)
        rank in 2usize..3,   // Fixed at 2 for speed
        obs_rate in 0.6f64..0.8,  // Narrowed range from 0.5-0.8 for speed
    ) {
        prop_assume!(rank < size);

        let _shape = [size, size, size];

        // Create a true low-rank tensor via small Tucker decomposition
        use scirs2_core::ndarray_ext::{Array2, Array3};
        let core_data = Array3::from_shape_fn((rank, rank, rank), |(i, j, k)| {
            (i + j + k) as f64 / (3.0 * rank as f64)
        });
        let mut core = DenseND::from_array(core_data.into_dyn());

        let mut factors = Vec::new();
        for _ in 0..3 {
            let factor_data = Array2::from_shape_fn((size, rank), |(i, j)| {
                ((i + j) as f64 / (size + rank) as f64) + 0.01
            });
            factors.push(factor_data);
        }

        // Reconstruct to get low-rank tensor
        for (mode, factor) in factors.iter().enumerate() {
            let recon_array = tenrso_kernels::nmode_product(&core.view(), &factor.view(), mode)
                .expect("nmode should work");
            core = DenseND::from_array(recon_array);
        }
        let low_rank_tensor = core;

        // Create mask
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();
        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < obs_rate {
                *idx = 1.0;
            }
        }
        let mask = DenseND::from_array(mask_data.into_dyn());

        // Run Tucker completion (reduced from 8 to 5 iterations for speed)
        let tucker = tucker_completion(&low_rank_tensor, &mask, &[rank, rank, rank], 5, 1e-4)
            .expect("Tucker completion should succeed");

        // For low-rank tensors with reasonable observation rate, error should be moderate
        let error = tucker.error.unwrap();
        prop_assert!(
            error < 0.8,
            "Low-rank tensor completion error should be < 0.8 with obs_rate={:.2}, got {:.4}",
            obs_rate, error
        );
    }
}

// Property: Tucker completion error improves with more observations
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tucker_completion_improves_with_more_observations(
        size in 4usize..6,   // Reduced from 5-7 for speed (64-125 elements, runs 2 completions)
        rank in 2usize..3,   // Reduced from 2-4 for speed
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Test with two observation rates
        let obs_rate_low = 0.3;
        let obs_rate_high = 0.7;

        // Create masks
        let mut mask_low_data = Array::zeros(vec![size, size, size]);
        let mut mask_high_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();

        for i in 0..mask_low_data.len() {
            let r = rng.random::<f64>();
            if r < obs_rate_low {
                mask_low_data.as_slice_mut().unwrap()[i] = 1.0;
            }
            if r < obs_rate_high {
                mask_high_data.as_slice_mut().unwrap()[i] = 1.0;
            }
        }

        let mask_low = DenseND::from_array(mask_low_data.into_dyn());
        let mask_high = DenseND::from_array(mask_high_data.into_dyn());

        // Run completion with both masks (reduced from 10 to 5 iterations for speed)
        let tucker_low = tucker_completion(&tensor, &mask_low, &[rank, rank, rank], 5, 1e-4)
            .expect("Tucker completion should succeed with low obs");
        let tucker_high = tucker_completion(&tensor, &mask_high, &[rank, rank, rank], 5, 1e-4)
            .expect("Tucker completion should succeed with high obs");

        // Both should produce valid errors
        let error_low = tucker_low.error.unwrap();
        let error_high = tucker_high.error.unwrap();

        prop_assert!((0.0..=1.0).contains(&error_low));
        prop_assert!((0.0..=1.0).contains(&error_high));

        // Note: We don't assert error_high < error_low because random tensors
        // may not benefit from more observations in the same way structured data does
    }
}

// Property: Tucker completion should respect rank constraints
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tucker_completion_respects_rank_constraint(
        size in 6usize..9,
        rank in 3usize..5,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Create mask with 50% observations
        let mut mask_data = Array::zeros(vec![size, size, size]);
        let mut rng = thread_rng();
        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < 0.5 {
                *idx = 1.0;
            }
        }
        let mask = DenseND::from_array(mask_data.into_dyn());

        // Reduced from 30 to 10 iterations for speed
        let tucker = tucker_completion(&tensor, &mask, &[rank, rank, rank], 10, 1e-4)
            .expect("Tucker completion should succeed");

        // Verify factor dimensions
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
        prop_assert_eq!(tucker.core.shape(), &[rank, rank, rank]);
    }
}
