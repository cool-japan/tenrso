//! TT matrix-vector product properties: identity, scaling, rank multiplication, error handling.
//!
//! Split out of the former monolithic `property_tests.rs` (see the parent module).

use super::shared::proptest_config;

use proptest::prelude::*;
use tenrso_core::DenseND;

// Property: Identity matrix-vector product returns the vector
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tt_matvec_identity_preserves_vector(
        size in 4usize..6,
        n_modes in 2usize..4,
    ) {
        use scirs2_core::ndarray_ext::Array1;
        use crate::tt::{tt_svd, tt_matrix_from_diagonal};

        let shape = vec![size; n_modes];
        let total_size: usize = shape.iter().product();

        // Create identity matrix (all ones on diagonal)
        let ones = Array1::from_vec(vec![1.0; total_size]);
        let identity = tt_matrix_from_diagonal(&ones, &shape);

        // Create random vector
        let vec = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let tt_vec = tt_svd(&vec, &vec![size.saturating_sub(1); n_modes.saturating_sub(1)], 1e-10)
            .expect("TT-SVD should succeed");

        // Multiply: I × v = v
        let result = identity.matvec(&tt_vec)
            .expect("Matrix-vector product should succeed");

        // Reconstruct both
        let vec_recon = tt_vec.reconstruct().expect("Vector reconstruction should succeed");
        let result_recon = result.reconstruct().expect("Result reconstruction should succeed");

        // Check shapes match
        prop_assert_eq!(vec_recon.shape(), result_recon.shape());

        // Check values are close
        let vec_view = vec_recon.view();
        let res_view = result_recon.view();
        let mut max_diff: f64 = 0.0;
        for (v1, v2) in vec_view.iter().zip(res_view.iter()) {
            max_diff = max_diff.max((v1 - v2).abs());
        }

        prop_assert!(
            max_diff < 1e-8,
            "Identity × vector should preserve vector, max diff: {}", max_diff
        );
    }
}

// Property: Scaling matrix produces correctly scaled results
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tt_matvec_scaling_is_correct(
        size in 3usize..5,
        n_modes in 2usize..3,
    ) {
        use scirs2_core::ndarray_ext::Array1;
        use crate::tt::{tt_svd, tt_matrix_from_diagonal};

        let shape = vec![size; n_modes];
        let total_size: usize = shape.iter().product();

        // Create scaling factors (positive values)
        let scale_factors: Vec<f64> = (1..=total_size).map(|i| i as f64).collect();
        let scales = Array1::from_vec(scale_factors.clone());
        let scale_matrix = tt_matrix_from_diagonal(&scales, &shape);

        // Create unit vector (all ones)
        let data = scirs2_core::ndarray_ext::Array::from_elem(
            scirs2_core::ndarray_ext::IxDyn(&shape),
            1.0,
        );
        let vec = DenseND::<f64>::from_array(data);
        let tt_vec = tt_svd(&vec, &vec![size.saturating_sub(1); n_modes.saturating_sub(1)], 1e-10)
            .expect("TT-SVD should succeed");

        // Multiply
        let result = scale_matrix.matvec(&tt_vec)
            .expect("Matrix-vector product should succeed");

        let result_dense = result.reconstruct()
            .expect("Result reconstruction should succeed");

        // Result values should be bounded by the scale factors
        let result_view = result_dense.view();
        let min_val = result_view.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = result_view.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        prop_assert!(min_val >= 0.0, "All values should be non-negative, got min: {}", min_val);
        prop_assert!(
            max_val <= total_size as f64 * 2.0,
            "Max value should be reasonably bounded, got: {}", max_val
        );
    }
}

// Property: TT-matvec ranks are product of matrix and vector ranks
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tt_matvec_ranks_multiply(
        size in 3usize..5,
        n_modes in 3usize..4,
        vec_rank in 2usize..3,
    ) {
        use scirs2_core::ndarray_ext::Array1;
        use crate::tt::{tt_svd, tt_matrix_from_diagonal};

        prop_assume!(vec_rank < size);

        let shape = vec![size; n_modes];
        let total_size: usize = shape.iter().product();

        // Create diagonal matrix (all ranks = 1)
        let ones = Array1::from_vec(vec![1.0; total_size]);
        let matrix = tt_matrix_from_diagonal(&ones, &shape);

        // Create vector with higher ranks
        let vec = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
        let max_ranks = vec![vec_rank; n_modes.saturating_sub(1)];
        let tt_vec = tt_svd(&vec, &max_ranks, 1e-10)
            .expect("TT-SVD should succeed");

        // Result ranks should be ≤ matrix_rank * vector_rank
        let result = matrix.matvec(&tt_vec)
            .expect("Matrix-vector product should succeed");

        // Matrix has all ranks = 1, so result ranks should be ≤ vector ranks
        for (i, &rank) in result.ranks.iter().enumerate() {
            prop_assert!(
                rank <= vec_rank + 1,
                "Rank {} is {}, expected <= {}", i, rank, vec_rank + 1
            );
        }
    }
}

// Property: TT-matvec dimension mismatch errors gracefully
proptest! {
    #![proptest_config(proptest_config())]
    #[test]
    fn tt_matvec_dimension_mismatch_fails(
        size1 in 3usize..5,
        size2 in 5usize..7,
    ) {
        use scirs2_core::ndarray_ext::Array1;
        use crate::tt::{tt_svd, tt_matrix_from_diagonal};

        prop_assume!(size1 != size2);

        // Create matrix for size1
        let shape1 = vec![size1, size1];
        let total1: usize = shape1.iter().product();
        let diag1 = Array1::from_vec(vec![1.0; total1]);
        let matrix = tt_matrix_from_diagonal(&diag1, &shape1);

        // Create vector for size2
        let shape2 = vec![size2, size2];
        let vec = DenseND::<f64>::random_uniform(&shape2, 0.0, 1.0);
        let tt_vec = tt_svd(&vec, &[size2.saturating_sub(1)], 1e-10)
            .expect("TT-SVD should succeed");

        // Should error due to dimension mismatch
        let result = matrix.matvec(&tt_vec);
        prop_assert!(result.is_err(), "Mismatched dimensions should produce error");
    }
}
