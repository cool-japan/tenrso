//! Property-based tests for sparse tensor formats and operations
//!
//! These tests use proptest to verify algebraic properties and correctness
//! of sparse format conversions and operations against dense baselines.

use proptest::prelude::*;
use scirs2_core::ndarray_ext::Array1;
use tenrso_sparse::{CooTensor, CscMatrix, CsrMatrix, Mask};

// ============================================================================
// Test Utilities
// ============================================================================

// Type alias for sparse matrix strategy return type to reduce complexity
type SparseMatrixData = (Vec<(usize, usize)>, Vec<f64>, (usize, usize));

/// Generate a random sparse 2D matrix as COO with controlled density
fn sparse_matrix_strategy(
    nrows: usize,
    ncols: usize,
    max_nnz: usize,
) -> impl Strategy<Value = SparseMatrixData> {
    prop::collection::vec((0..nrows, 0..ncols), 0..=max_nnz).prop_flat_map(move |indices| {
        let len = indices.len();
        (
            Just(indices),
            prop::collection::vec(-100.0..100.0f64, len..=len),
            Just((nrows, ncols)),
        )
    })
}

// ============================================================================
// Format Conversion Roundtrip Properties
// ============================================================================

proptest! {
    /// Property: COO → Dense → COO preserves data
    #[test]
    fn prop_coo_dense_roundtrip(
        (indices, values, shape) in sparse_matrix_strategy(10, 10, 30)
    ) {
        // Create COO tensor (2D)
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices.clone(),
            values.clone(),
            vec![shape.0, shape.1],
        ).unwrap();

        // Deduplicate to handle potential duplicate indices
        coo.deduplicate();

        // Convert to dense and back
        let dense = coo.to_dense().unwrap();
        let coo2 = CooTensor::from_dense(&dense, 1e-10);

        // Verify NNZ is preserved
        prop_assert_eq!(coo.nnz(), coo2.nnz());

        // Verify dense representations match
        let dense2 = coo2.to_dense().unwrap();
        let dense_arr = dense.as_array();
        let dense2_arr = dense2.as_array();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                prop_assert!((dense_arr[[i, j]] - dense2_arr[[i, j]]).abs() < 1e-9);
            }
        }
    }

    /// Property: CSR → Dense → CSR preserves data
    #[test]
    fn prop_csr_dense_roundtrip(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20)
    ) {
        // Create COO first, then convert to CSR
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices,
            values,
            vec![shape.0, shape.1],
        ).unwrap();

        coo.deduplicate();

        let csr = CsrMatrix::from_coo(&coo).unwrap();

        // Convert to dense and back
        let dense = csr.to_dense().unwrap();
        let csr2 = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        // Verify NNZ is preserved
        prop_assert_eq!(csr.nnz(), csr2.nnz());

        // Verify dense representations match
        let dense2 = csr2.to_dense().unwrap();
        let dense_arr = dense.as_array();
        let dense2_arr = dense2.as_array();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                prop_assert!((dense_arr[[i, j]] - dense2_arr[[i, j]]).abs() < 1e-9);
            }
        }
    }

    /// Property: CSC → Dense → CSC preserves data
    #[test]
    fn prop_csc_dense_roundtrip(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20)
    ) {
        // Create COO first, then convert to CSC
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices,
            values,
            vec![shape.0, shape.1],
        ).unwrap();

        coo.deduplicate();

        let csc = CscMatrix::from_coo(&coo).unwrap();

        // Convert to dense and back
        let dense = csc.to_dense().unwrap();
        let csc2 = CscMatrix::from_dense(&dense, 1e-10).unwrap();

        // Verify NNZ is preserved
        prop_assert_eq!(csc.nnz(), csc2.nnz());

        // Verify dense representations match
        let dense2 = csc2.to_dense().unwrap();
        let dense_arr = dense.as_array();
        let dense2_arr = dense2.as_array();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                prop_assert!((dense_arr[[i, j]] - dense2_arr[[i, j]]).abs() < 1e-9);
            }
        }
    }

    /// Property: COO → CSR → COO preserves data
    #[test]
    fn prop_coo_csr_roundtrip(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20)
    ) {
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices,
            values,
            vec![shape.0, shape.1],
        ).unwrap();

        coo.deduplicate();

        let original_nnz = coo.nnz();
        let original_dense = coo.to_dense().unwrap();

        // COO → CSR → COO
        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let coo2 = csr.to_coo();

        prop_assert_eq!(original_nnz, coo2.nnz());

        let recovered_dense = coo2.to_dense().unwrap();
        let orig_arr = original_dense.as_array();
        let recov_arr = recovered_dense.as_array();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                prop_assert!((orig_arr[[i, j]] - recov_arr[[i, j]]).abs() < 1e-9);
            }
        }
    }

    /// Property: COO → CSC → COO preserves data
    #[test]
    fn prop_coo_csc_roundtrip(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20)
    ) {
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices,
            values,
            vec![shape.0, shape.1],
        ).unwrap();

        coo.deduplicate();

        let original_nnz = coo.nnz();
        let original_dense = coo.to_dense().unwrap();

        // COO → CSC → COO
        let csc = CscMatrix::from_coo(&coo).unwrap();
        let coo2 = csc.to_coo();

        prop_assert_eq!(original_nnz, coo2.nnz());

        let recovered_dense = coo2.to_dense().unwrap();
        let orig_arr = original_dense.as_array();
        let recov_arr = recovered_dense.as_array();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                prop_assert!((orig_arr[[i, j]] - recov_arr[[i, j]]).abs() < 1e-9);
            }
        }
    }

    /// Property: CSR → CSC → CSR preserves data
    #[test]
    fn prop_csr_csc_roundtrip(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20)
    ) {
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices,
            values,
            vec![shape.0, shape.1],
        ).unwrap();

        coo.deduplicate();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let original_dense = csr.to_dense().unwrap();

        // CSR → CSC → CSR
        let csc = CscMatrix::from_csr(&csr);
        let csr2 = csc.to_csr();

        prop_assert_eq!(csr.nnz(), csr2.nnz());

        let recovered_dense = csr2.to_dense().unwrap();
        let orig_arr = original_dense.as_array();
        let recov_arr = recovered_dense.as_array();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                prop_assert!((orig_arr[[i, j]] - recov_arr[[i, j]]).abs() < 1e-9);
            }
        }
    }
}

// ============================================================================
// Sparse Operations Correctness Properties
// ============================================================================

proptest! {
    /// Property: CSR SpMV matches dense matrix-vector multiply
    #[test]
    fn prop_csr_spmv_correctness(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20),
        vec_vals in prop::collection::vec(-10.0..10.0f64, 8)
    ) {
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices,
            values,
            vec![shape.0, shape.1],
        ).unwrap();

        coo.deduplicate();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let dense = csr.to_dense().unwrap();
        let dense_2d: scirs2_core::ndarray_ext::Array2<f64> =
            dense.view().to_owned().into_dimensionality().unwrap();
        let vec_x = Array1::from_vec(vec_vals);

        // Sparse SpMV
        let sparse_result = csr.spmv(&vec_x.view()).unwrap();

        // Dense baseline: y = A * x
        let dense_result = dense_2d.dot(&vec_x);

        // Compare results
        for i in 0..shape.0 {
            prop_assert!((sparse_result[i] - dense_result[i]).abs() < 1e-9,
                "SpMV mismatch at index {}: sparse={}, dense={}",
                i, sparse_result[i], dense_result[i]);
        }
    }

    /// Property: CSR SpMM matches dense matrix-matrix multiply
    #[test]
    fn prop_csr_spmm_correctness(
        (indices, values, shape) in sparse_matrix_strategy(6, 6, 15),
        b_vals in prop::collection::vec(-10.0..10.0f64, 6 * 4)
    ) {
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices,
            values,
            vec![shape.0, shape.1],
        ).unwrap();

        coo.deduplicate();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let dense = csr.to_dense().unwrap();
        let dense_2d: scirs2_core::ndarray_ext::Array2<f64> =
            dense.view().to_owned().into_dimensionality().unwrap();

        // Create dense matrix B (6×4)
        let b_array = Array1::from_vec(b_vals);
        let dense_b: scirs2_core::ndarray_ext::Array2<f64> =
            b_array.to_shape((6, 4)).unwrap().to_owned();

        // Sparse SpMM
        let sparse_result = csr.spmm(&dense_b.view()).unwrap();

        // Dense baseline: C = A * B
        let dense_result = dense_2d.dot(&dense_b);

        // Compare results
        for i in 0..shape.0 {
            for j in 0..4 {
                prop_assert!((sparse_result[[i, j]] - dense_result[[i, j]]).abs() < 1e-8,
                    "SpMM mismatch at [{}, {}]: sparse={}, dense={}",
                    i, j, sparse_result[[i, j]], dense_result[[i, j]]);
            }
        }
    }

    /// Property: CSC SpMM matches dense matrix-matrix multiply
    #[test]
    fn prop_csc_spmm_correctness(
        (indices, values, shape) in sparse_matrix_strategy(6, 6, 15),
        b_vals in prop::collection::vec(-10.0..10.0f64, 6 * 4)
    ) {
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices,
            values,
            vec![shape.0, shape.1],
        ).unwrap();

        coo.deduplicate();

        let csc = CscMatrix::from_coo(&coo).unwrap();
        let dense = csc.to_dense().unwrap();
        let dense_2d: scirs2_core::ndarray_ext::Array2<f64> =
            dense.view().to_owned().into_dimensionality().unwrap();

        // Create dense matrix B (6×4)
        let b_array = Array1::from_vec(b_vals);
        let dense_b: scirs2_core::ndarray_ext::Array2<f64> =
            b_array.to_shape((6, 4)).unwrap().to_owned();

        // Sparse SpMM
        let sparse_result = csc.spmm(&dense_b.view()).unwrap();

        // Dense baseline: C = A * B
        let dense_result = dense_2d.dot(&dense_b);

        // Compare results
        for i in 0..shape.0 {
            for j in 0..4 {
                prop_assert!((sparse_result[[i, j]] - dense_result[[i, j]]).abs() < 1e-8,
                    "CSC SpMM mismatch at [{}, {}]: sparse={}, dense={}",
                    i, j, sparse_result[[i, j]], dense_result[[i, j]]);
            }
        }
    }

    /// Property: CSC matvec matches dense matrix-vector multiply
    #[test]
    fn prop_csc_matvec_correctness(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20),
        vec_vals in prop::collection::vec(-10.0..10.0f64, 8)
    ) {
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(
            coo_indices,
            values,
            vec![shape.0, shape.1],
        ).unwrap();

        coo.deduplicate();

        let csc = CscMatrix::from_coo(&coo).unwrap();
        let dense = csc.to_dense().unwrap();
        let dense_2d: scirs2_core::ndarray_ext::Array2<f64> =
            dense.view().to_owned().into_dimensionality().unwrap();
        let vec_x = Array1::from_vec(vec_vals);

        // Sparse matvec
        let sparse_result = csc.matvec(&vec_x.view()).unwrap();

        // Dense baseline: y = A * x
        let dense_result = dense_2d.dot(&vec_x);

        // Compare results
        for i in 0..shape.0 {
            prop_assert!((sparse_result[i] - dense_result[i]).abs() < 1e-9,
                "CSC matvec mismatch at index {}: sparse={}, dense={}",
                i, sparse_result[i], dense_result[i]);
        }
    }
}

// ============================================================================
// Mask Set Operation Properties
// ============================================================================

proptest! {
    /// Property: Mask union is commutative
    #[test]
    fn prop_mask_union_commutative(
        indices1 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10),
        indices2 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10)
    ) {
        let mask1 = Mask::from_indices(indices1, vec![5, 5]).unwrap();
        let mask2 = Mask::from_indices(indices2, vec![5, 5]).unwrap();

        let union1 = mask1.union(&mask2).unwrap();
        let union2 = mask2.union(&mask1).unwrap();

        // Union is commutative
        prop_assert_eq!(union1.nnz(), union2.nnz());

        // All indices in union1 are in union2
        for idx in union1.iter() {
            prop_assert!(union2.contains(idx));
        }
    }

    /// Property: Mask intersection is commutative
    #[test]
    fn prop_mask_intersection_commutative(
        indices1 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10),
        indices2 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10)
    ) {
        let mask1 = Mask::from_indices(indices1, vec![5, 5]).unwrap();
        let mask2 = Mask::from_indices(indices2, vec![5, 5]).unwrap();

        let inter1 = mask1.intersection(&mask2).unwrap();
        let inter2 = mask2.intersection(&mask1).unwrap();

        // Intersection is commutative
        prop_assert_eq!(inter1.nnz(), inter2.nnz());

        // All indices in inter1 are in inter2
        for idx in inter1.iter() {
            prop_assert!(inter2.contains(idx));
        }
    }

    /// Property: Mask union contains both operands
    #[test]
    fn prop_mask_union_contains_operands(
        indices1 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10),
        indices2 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10)
    ) {
        let mask1 = Mask::from_indices(indices1, vec![5, 5]).unwrap();
        let mask2 = Mask::from_indices(indices2, vec![5, 5]).unwrap();

        let union = mask1.union(&mask2).unwrap();

        // Union contains all indices from mask1
        for idx in mask1.iter() {
            prop_assert!(union.contains(idx));
        }

        // Union contains all indices from mask2
        for idx in mask2.iter() {
            prop_assert!(union.contains(idx));
        }
    }

    /// Property: Mask intersection is subset of both operands
    #[test]
    fn prop_mask_intersection_subset(
        indices1 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10),
        indices2 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10)
    ) {
        let mask1 = Mask::from_indices(indices1, vec![5, 5]).unwrap();
        let mask2 = Mask::from_indices(indices2, vec![5, 5]).unwrap();

        let inter = mask1.intersection(&mask2).unwrap();

        // All indices in intersection are in mask1
        for idx in inter.iter() {
            prop_assert!(mask1.contains(idx));
        }

        // All indices in intersection are in mask2
        for idx in inter.iter() {
            prop_assert!(mask2.contains(idx));
        }
    }

    /// Property: Mask difference is subset of first operand
    #[test]
    fn prop_mask_difference_subset(
        indices1 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10),
        indices2 in prop::collection::vec(prop::collection::vec(0..5usize, 2..=2), 0..10)
    ) {
        let mask1 = Mask::from_indices(indices1, vec![5, 5]).unwrap();
        let mask2 = Mask::from_indices(indices2, vec![5, 5]).unwrap();

        let diff = mask1.difference(&mask2).unwrap();

        // All indices in difference are in mask1
        for idx in diff.iter() {
            prop_assert!(mask1.contains(idx));
        }

        // No indices in difference are in mask2
        for idx in diff.iter() {
            prop_assert!(!mask2.contains(idx));
        }
    }

    /// Property: Mask density is correct
    #[test]
    fn prop_mask_density(
        indices in prop::collection::vec(prop::collection::vec(0..4usize, 2..=2), 0..8)
    ) {
        let mask = Mask::from_indices(indices, vec![4, 4]).unwrap();

        let density = mask.density();
        let expected = mask.nnz() as f64 / 16.0;

        prop_assert!((density - expected).abs() < 1e-10,
            "Density mismatch: got {}, expected {}", density, expected);
    }

    // ============================================================================
    // Element-wise Operation Properties
    // ============================================================================

    /// Property: Sparse clip ensures all values are within bounds
    #[test]
    fn prop_clip_bounds(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20)
    ) {
        use tenrso_sparse::ops::sparse_clip_csr;

        // Create CSR matrix
        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(coo_indices, values, vec![shape.0, shape.1]).unwrap();
        coo.deduplicate();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        // Clip to range [0.0, 50.0]
        let clipped = sparse_clip_csr(&csr, 0.0, 50.0).unwrap();

        // All values should be within bounds
        for &val in clipped.values() {
            prop_assert!(val >= 0.0, "Value {} is below minimum", val);
            prop_assert!(val <= 50.0, "Value {} is above maximum", val);
        }

        // NNZ should be preserved (clipping doesn't remove elements)
        prop_assert_eq!(csr.nnz(), clipped.nnz());
    }

    /// Property: Sparse floor returns integer values
    #[test]
    fn prop_floor_integer(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20)
    ) {
        use tenrso_sparse::ops::sparse_floor_csr;

        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(coo_indices, values, vec![shape.0, shape.1]).unwrap();
        coo.deduplicate();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let floored = sparse_floor_csr(&csr).unwrap();

        // All values should be integers (floor of originals)
        for (&original, &floored_val) in csr.values().iter().zip(floored.values().iter()) {
            prop_assert_eq!(floored_val, original.floor());
            prop_assert_eq!(floored_val.fract(), 0.0, "Floor should produce integer value");
        }

        // NNZ preserved
        prop_assert_eq!(csr.nnz(), floored.nnz());
    }

    /// Property: Sparse ceil returns integer values
    #[test]
    fn prop_ceil_integer(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20)
    ) {
        use tenrso_sparse::ops::sparse_ceil_csr;

        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(coo_indices, values, vec![shape.0, shape.1]).unwrap();
        coo.deduplicate();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let ceiled = sparse_ceil_csr(&csr).unwrap();

        // All values should be integers (ceil of originals)
        for (&original, &ceiled_val) in csr.values().iter().zip(ceiled.values().iter()) {
            prop_assert_eq!(ceiled_val, original.ceil());
            prop_assert_eq!(ceiled_val.fract(), 0.0, "Ceil should produce integer value");
        }

        // NNZ preserved
        prop_assert_eq!(csr.nnz(), ceiled.nnz());
    }

    /// Property: Sparse round returns integer values
    #[test]
    fn prop_round_integer(
        (indices, values, shape) in sparse_matrix_strategy(8, 8, 20)
    ) {
        use tenrso_sparse::ops::sparse_round_csr;

        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        let mut coo = CooTensor::new(coo_indices, values, vec![shape.0, shape.1]).unwrap();
        coo.deduplicate();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let rounded = sparse_round_csr(&csr).unwrap();

        // All values should be integers (round of originals)
        for (&original, &rounded_val) in csr.values().iter().zip(rounded.values().iter()) {
            prop_assert_eq!(rounded_val, original.round());
            prop_assert_eq!(rounded_val.fract(), 0.0, "Round should produce integer value");
        }

        // NNZ preserved
        prop_assert_eq!(csr.nnz(), rounded.nnz());
    }

    /// Property: Hypot satisfies sqrt(a² + b²) = hypot(a, b)
    #[test]
    fn prop_hypot_pythagorean(
        (indices, values, shape) in sparse_matrix_strategy(6, 6, 15)
    ) {
        use tenrso_sparse::ops::sparse_hypot_csr;

        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        // Use absolute values to avoid negative sqrt
        let abs_values: Vec<f64> = values.iter().map(|&v| v.abs()).collect();

        let mut coo = CooTensor::new(coo_indices, abs_values, vec![shape.0, shape.1]).unwrap();
        coo.deduplicate();
        let a = CsrMatrix::from_coo(&coo).unwrap();
        let b = a.clone(); // Use same matrix for simplicity

        let result = sparse_hypot_csr(&a, &b).unwrap();

        // Verify hypot(a, a) = sqrt(a² + a²) = sqrt(2*a²) = a*sqrt(2)
        for (&a_val, &result_val) in a.values().iter().zip(result.values().iter()) {
            let expected = a_val * 2.0_f64.sqrt();
            prop_assert!((result_val - expected).abs() < 1e-9,
                "Hypot mismatch: got {}, expected {}", result_val, expected);
        }
    }

    /// Property: Divide and multiply are inverse operations
    #[test]
    fn prop_divide_multiply_inverse(
        (indices, values, shape) in sparse_matrix_strategy(6, 6, 15)
    ) {
        use tenrso_sparse::ops::{sparse_divide_csr, sparse_multiply_csr};

        let coo_indices: Vec<Vec<usize>> = indices.iter()
            .map(|(i, j)| vec![*i, *j])
            .collect();

        // Use non-zero values to avoid division by zero
        let nonzero_values: Vec<f64> = values.iter()
            .map(|&v| if v.abs() < 0.1 { 1.0 } else { v })
            .collect();

        let mut coo = CooTensor::new(coo_indices, nonzero_values, vec![shape.0, shape.1]).unwrap();
        coo.deduplicate();
        let a = CsrMatrix::from_coo(&coo).unwrap();
        let b = a.clone();

        // (A ./ B) .* B should approximately equal A for overlapping elements
        let divided = sparse_divide_csr(&a, &b).unwrap();
        let restored = sparse_multiply_csr(&divided, &b).unwrap();

        // Verify values match original
        for (&original, &restored_val) in a.values().iter().zip(restored.values().iter()) {
            prop_assert!((original - restored_val).abs() < 1e-9,
                "Divide-multiply inverse failed: {} != {}", original, restored_val);
        }
    }
}
