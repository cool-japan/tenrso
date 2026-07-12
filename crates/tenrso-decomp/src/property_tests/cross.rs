//! Cross-method comparison: every decomposition must reconstruct to the original shape.
//!
//! Split out of the former monolithic `property_tests.rs` (see the parent module).

use crate::{cp_als, tt_svd, tucker_hosvd, InitStrategy};
use proptest::prelude::*;
use tenrso_core::DenseND;
// Property: All methods should produce valid reconstructions (same shape)
proptest! {
    #![proptest_config(ProptestConfig { cases: 3, max_local_rejects: 1000, max_global_rejects: 10000, ..ProptestConfig::default() })]
    #[test]
    fn all_methods_produce_valid_reconstructions(
        size in 4usize..6,   // Reduced from 5-8 for speed (64-125 elements)
        rank in 2usize..3,   // Reduced from 2-4 for speed
    ) {
        prop_assume!(rank < size);

        let shape = vec![size, size, size];
        let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

        // CP (reduced iterations for speed)
        let cp = cp_als(&tensor, rank, 5, 1e-4, InitStrategy::Random, None)
            .expect("CP-ALS should succeed");
        let cp_recon = cp.reconstruct(&shape)
            .expect("CP reconstruction should succeed");
        prop_assert_eq!(cp_recon.shape(), &shape[..]);

        // Tucker
        let ranks = vec![rank, rank, rank];
        let tucker = tucker_hosvd(&tensor, &ranks)
            .expect("Tucker-HOSVD should succeed");
        let tucker_recon = tucker.reconstruct()
            .expect("Tucker reconstruction should succeed");
        prop_assert_eq!(tucker_recon.shape(), &shape[..]);

        // TT
        let max_ranks = vec![rank; shape.len() - 1];
        let tt = tt_svd(&tensor, &max_ranks, 1e-6)
            .expect("TT-SVD should succeed");
        let tt_recon = tt.reconstruct()
            .expect("TT reconstruction should succeed");
        prop_assert_eq!(tt_recon.shape(), &shape[..]);
    }
}
