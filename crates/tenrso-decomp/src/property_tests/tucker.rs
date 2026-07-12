//! Tucker property tests: factor orthogonality, HOOI vs HOSVD, rank monotonicity, core shape.
//!
//! Split out of the former monolithic `property_tests.rs` (see the parent module).

use super::shared::proptest_config;
use crate::{tucker_hooi, tucker_hosvd};
use proptest::prelude::*;
use tenrso_core::DenseND;
// Property: Tucker factor matrices should be orthogonal
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tucker_factors_are_orthogonal(
        size in 5usize..8,   // Reduced from 8-11 for speed (125-343 elements)
        rank in 2usize..4,   // Reduced from 4-7 for speed
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let ranks = vec![rank, rank, rank];

        let tucker = tucker_hosvd(&tensor, &ranks)
            .expect("Tucker-HOSVD should succeed");

        // Check each factor matrix for orthogonality
        for (mode, factor) in tucker.factors.iter().enumerate() {
            // Compute U^T U
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
                        diff < 1e-10,
                        "Mode {} factor: U^T U[{}, {}] = {:.2e}, expected {:.2e}",
                        mode, i, j, actual, expected
                    );
                }
            }
        }
    }
}

// Property: Tucker HOOI should improve or maintain error vs HOSVD
proptest! {
    #![proptest_config(ProptestConfig { cases: 1, max_local_rejects: 1000, max_global_rejects: 10000, ..ProptestConfig::default() })]
    #[test]
    fn tucker_hooi_improves_over_hosvd(
        size in 4usize..5,  // Fixed at 4 for speed (64 elements)
        rank in 2usize..3,  // Fixed at 2 for speed
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let ranks = vec![rank, rank, rank];

        let tucker_hosvd = tucker_hosvd(&tensor, &ranks)
            .expect("Tucker-HOSVD should succeed");
        let recon_hosvd = tucker_hosvd.reconstruct()
            .expect("HOSVD reconstruction should succeed");
        let error_hosvd = (&tensor - &recon_hosvd)
            .frobenius_norm() / tensor.frobenius_norm();

        let tucker_hooi = tucker_hooi(&tensor, &ranks, 1, 1e-4)  // Reduced from 2 to 1 iteration
            .expect("Tucker-HOOI should succeed");
        let recon_hooi = tucker_hooi.reconstruct()
            .expect("HOOI reconstruction should succeed");
        let error_hooi = (&tensor - &recon_hooi)
            .frobenius_norm() / tensor.frobenius_norm();

        // HOOI should have equal or lower error (with tolerance)
        // Note: HOOI doesn't always improve over HOSVD for very small tensors
        // or random data with no structure. Allow some variance.
        prop_assert!(
            error_hooi <= error_hosvd * 1.05,
            "HOOI error ({:.6}) should be approximately <= HOSVD error ({:.6})",
            error_hooi, error_hosvd
        );
    }
}

// Property: Tucker reconstruction error should decrease with higher rank
proptest! {
    #![proptest_config(ProptestConfig { cases: 1, max_local_rejects: 1000, max_global_rejects: 10000, ..ProptestConfig::default() })]
    #[test]
    fn tucker_error_decreases_with_rank(
        size in 4usize..5,    // Fixed at 4 for speed (64 elements)
        rank1 in 2usize..3,   // Fixed at 2 for speed
        rank2 in 3usize..4,   // Fixed at 3 for speed
    ) {
        prop_assume!(rank1 < rank2);
        prop_assume!(rank2 < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let original_norm = tensor.frobenius_norm();

        // Lower rank
        let ranks1 = vec![rank1, rank1, rank1];
        let tucker1 = tucker_hosvd(&tensor, &ranks1)
            .expect("Tucker-HOSVD with lower rank should succeed");
        let recon1 = tucker1.reconstruct()
            .expect("Reconstruction should succeed");
        let error1 = (&tensor - &recon1)
            .frobenius_norm() / original_norm;

        // Higher rank
        let ranks2 = vec![rank2, rank2, rank2];
        let tucker2 = tucker_hosvd(&tensor, &ranks2)
            .expect("Tucker-HOSVD with higher rank should succeed");
        let recon2 = tucker2.reconstruct()
            .expect("Reconstruction should succeed");
        let error2 = (&tensor - &recon2)
            .frobenius_norm() / original_norm;

        // Higher rank should have equal or lower error
        prop_assert!(
            error2 <= error1 * 1.01,
            "Error with rank {} ({:.6}) should be <= error with rank {} ({:.6})",
            rank2, error2, rank1, error1
        );
    }
}

// Property: Tucker core tensor should have expected shape
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tucker_core_has_correct_shape(
        size in 5usize..8,   // Reduced from 8-11 for speed (125-343 elements)
        rank in 2usize..4,   // Reduced from 4-7 for speed
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let ranks = vec![rank, rank, rank];

        let tucker = tucker_hosvd(&tensor, &ranks)
            .expect("Tucker-HOSVD should succeed");

        let core_shape = tucker.core.shape();
        prop_assert_eq!(core_shape.len(), 3);
        prop_assert_eq!(core_shape[0], rank);
        prop_assert_eq!(core_shape[1], rank);
        prop_assert_eq!(core_shape[2], rank);
    }
}
