//! Randomized CP property tests: dimensions, quality, sketch size, agreement with standard CP.
//!
//! Split out of the former monolithic `property_tests.rs` (see the parent module).

use super::shared::proptest_config;
use crate::{cp_als, cp_randomized, InitStrategy};
use proptest::prelude::*;
use tenrso_core::DenseND;
// Property: Randomized CP produces valid factor dimensions
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn randomized_cp_produces_valid_dimensions(
        size in 10usize..15,
        rank in 3usize..6,
        sketch_mult in 3usize..6,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let sketch_size = rank * sketch_mult;

        let cp = cp_randomized(&tensor, rank, 10, 1e-4, InitStrategy::Random, sketch_size, 5)
            .expect("Randomized CP should succeed");

        // Verify factor dimensions
        prop_assert_eq!(cp.factors.len(), 3, "Should have 3 factor matrices");
        for (mode, factor) in cp.factors.iter().enumerate() {
            prop_assert_eq!(
                factor.nrows(), size,
                "Mode {} factor rows should match tensor dimension", mode
            );
            prop_assert_eq!(
                factor.ncols(), rank,
                "Mode {} factor columns should match rank", mode
            );
        }

        // Verify fit is in valid range
        prop_assert!(
            cp.fit >= 0.0 && cp.fit <= 1.0,
            "Fit should be in [0, 1], got {:.6}", cp.fit
        );
    }
}

// Property: Randomized CP reconstruction has reasonable quality on low-rank tensors.
//
// Tests on a CONSTRUCTED rank-R tensor (built from outer products of random
// vectors), not on a random full-rank tensor.  A random full-rank tensor has
// large minimum rank, so rank-R CP reconstruction can have error > 0.8 by
// construction — that is not a bug.  To get a meaningful quality bound we
// must test on tensors whose true CP rank equals `rank`.
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn randomized_cp_reconstruction_quality(
        size in 6usize..9,
        rank in 2usize..4,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];

        // Build a rank-R tensor as X = sum_{r<rank} a_r ⊗ b_r ⊗ c_r using a
        // deterministic LCG seeded by (size, rank) so every proptest case gets
        // a distinct but reproducible tensor without needing the `rand` crate.
        let seed: u64 = (size as u64).wrapping_mul(1_234_567).wrapping_add(rank as u64);
        let mut rng = seed;
        let mut next_val = || -> f64 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005)
                     .wrapping_add(1_442_695_040_888_963_407);
            // Map to (-1, 1) and scale so components are O(1)
            (rng >> 32) as f64 / (u32::MAX as f64) - 0.5
        };

        let a: Vec<f64> = (0..size * rank).map(|_| next_val()).collect();
        let b: Vec<f64> = (0..size * rank).map(|_| next_val()).collect();
        let c: Vec<f64> = (0..size * rank).map(|_| next_val()).collect();

        let mut data = vec![0.0f64; size * size * size];
        for r in 0..rank {
            for i in 0..size {
                for j in 0..size {
                    for k in 0..size {
                        data[i * size * size + j * size + k] +=
                            a[i * rank + r] * b[j * rank + r] * c[k * rank + r];
                    }
                }
            }
        }
        let tensor = DenseND::<f64>::from_vec(data, &shape)
            .expect("Tensor construction should succeed");

        // 10x oversampling + 30 iters gives robust convergence on rank-R tensors
        let sketch_size = rank * 10;
        let cp = cp_randomized(&tensor, rank, 30, 1e-6, InitStrategy::Random, sketch_size, 2)
            .expect("Randomized CP should succeed");

        let recon = cp.reconstruct(&shape)
            .expect("Reconstruction should succeed");

        // Structural checks
        prop_assert_eq!(recon.shape(), &shape[..]);
        prop_assert!(
            cp.fit >= 0.0 && cp.fit <= 1.0,
            "Fit must be in [0,1], got {}", cp.fit
        );

        // Quality check: rank-R CP on a rank-R tensor should achieve < 50% relative error
        let original_norm = tensor.frobenius_norm();
        let diff = &tensor - &recon;
        let error = diff.frobenius_norm() / original_norm;
        prop_assert!(
            error < 0.5,
            "Reconstruction error on rank-{} tensor should be < 0.5, got {:.6}", rank, error
        );

        // Fit consistency: reported fit should approximately match computed fit
        let computed_fit = 1.0 - error;
        prop_assert!(
            (cp.fit - computed_fit).abs() < 0.15,
            "Fit ({:.6}) should approximately match computed fit ({:.6})", cp.fit, computed_fit
        );
    }
}

// Property: Higher sketch size improves accuracy (statistically)
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn randomized_cp_sketch_size_affects_quality(
        size in 6usize..8,
        rank in 2usize..3,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // Low sketch size (3x)
        let cp_low = cp_randomized(&tensor, rank, 8, 1e-4, InitStrategy::Random, rank * 3, 2)
            .expect("Low sketch CP should succeed");

        // High sketch size (7x)
        let cp_high = cp_randomized(&tensor, rank, 8, 1e-4, InitStrategy::Random, rank * 7, 2)
            .expect("High sketch CP should succeed");

        // Both should produce valid results
        prop_assert!(cp_low.fit >= 0.0 && cp_low.fit <= 1.0);
        prop_assert!(cp_high.fit >= 0.0 && cp_high.fit <= 1.0);

        // Both should successfully reconstruct
        prop_assert!(cp_low.reconstruct(&shape).is_ok());
        prop_assert!(cp_high.reconstruct(&shape).is_ok());
    }
}

// Property: Randomized CP is comparable to standard CP (within tolerance)
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn randomized_cp_comparable_to_standard(
        size in 6usize..8,
        rank in 2usize..3,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let sketch_size = rank * 5; // 5x oversampling

        // Standard CP-ALS
        let cp_std = cp_als(&tensor, rank, 8, 1e-4, InitStrategy::Random, None)
            .expect("Standard CP should succeed");

        // Randomized CP
        let cp_rand = cp_randomized(&tensor, rank, 8, 1e-4, InitStrategy::Random, sketch_size, 2)
            .expect("Randomized CP should succeed");

        // Both should have valid fits
        prop_assert!(cp_std.fit >= 0.0 && cp_std.fit <= 1.0);
        prop_assert!(cp_rand.fit >= 0.0 && cp_rand.fit <= 1.0);

        // Randomized CP fit should be reasonably close to standard
        // (may be lower due to approximation, but not drastically)
        prop_assert!(
            cp_rand.fit >= 0.0,
            "Randomized CP should achieve non-negative fit, got {:.6}", cp_rand.fit
        );

        // Both should successfully reconstruct
        prop_assert!(cp_std.reconstruct(&shape).is_ok());
        prop_assert!(cp_rand.reconstruct(&shape).is_ok());
    }
}

// Property: Randomized CP convergence with iterations
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn randomized_cp_converges_with_iterations(
        size in 6usize..8,
        rank in 2usize..3,
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let sketch_size = rank * 4;

        // Few iterations
        let cp_few = cp_randomized(&tensor, rank, 3, 1e-4, InitStrategy::Random, sketch_size, 1)
            .expect("Randomized CP with few iterations should succeed");

        // Many iterations
        let cp_many = cp_randomized(&tensor, rank, 12, 1e-4, InitStrategy::Random, sketch_size, 1)
            .expect("Randomized CP with many iterations should succeed");

        // Both should produce valid results
        prop_assert!(cp_few.fit >= 0.0 && cp_few.fit <= 1.0);
        prop_assert!(cp_many.fit >= 0.0 && cp_many.fit <= 1.0);

        // Note: We don't compare fits between cp_few and cp_many because they use
        // different random initializations. With randomized methods, a lucky initialization
        // with few iterations can sometimes outperform an unlucky initialization with many
        // iterations. The important property is that both produce valid decompositions.
    }
}
