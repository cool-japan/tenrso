//! Property-based tests for tensor operations
//!
//! This module uses proptest to verify mathematical properties of tensor operations
//! across a wide range of randomly generated inputs.

#[cfg(test)]
mod tests {
    use crate::DenseND;
    use proptest::prelude::*;

    // Strategy for generating valid tensor shapes (1-4D, reasonable sizes)
    fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
        prop::collection::vec(2usize..10, 1..=4)
    }

    #[test]
    fn test_proptest_smoke() {
        // Simple smoke test to verify proptest is working
        let tensor = DenseND::<f64>::zeros(&[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    proptest! {
        #[test]
        fn prop_reshape_preserves_size(shape in shape_strategy()) {
            let tensor = DenseND::<f64>::zeros(&shape);
            let original_len = tensor.len();

            // Generate a valid reshape target
            let total_size: usize = shape.iter().product();
            let new_shape = if total_size > 1 {
                vec![total_size / 2, 2]
            } else {
                vec![1]
            };

            if let Ok(reshaped) = tensor.reshape(&new_shape) {
                prop_assert_eq!(reshaped.len(), original_len);
            }
        }

        #[test]
        fn prop_reshape_roundtrip(shape in shape_strategy()) {
            let tensor = DenseND::<f64>::ones(&shape);
            let original_shape = tensor.shape().to_vec();

            // Reshape to flat and back
            let flat = tensor.reshape(&[tensor.len()]).unwrap();
            prop_assert_eq!(flat.shape(), &[tensor.len()]);

            let restored = flat.reshape(&original_shape).unwrap();
            prop_assert_eq!(restored.shape(), original_shape.as_slice());
        }

        #[test]
        fn prop_permute_preserves_size(shape in shape_strategy()) {
            let tensor = DenseND::<f64>::zeros(&shape);
            let rank = tensor.rank();
            let original_len = tensor.len();

            // Generate a valid permutation
            let mut perm: Vec<usize> = (0..rank).collect();
            // Simple reversal as a permutation
            perm.reverse();

            let permuted = tensor.permute(&perm).unwrap();
            prop_assert_eq!(permuted.len(), original_len);
            prop_assert_eq!(permuted.rank(), rank);
        }

        #[test]
        fn prop_permute_identity(shape in shape_strategy()) {
            let tensor = DenseND::<f64>::ones(&shape);
            let rank = tensor.rank();

            // Identity permutation [0, 1, 2, ...]
            let identity: Vec<usize> = (0..rank).collect();
            let permuted = tensor.permute(&identity).unwrap();

            prop_assert_eq!(permuted.shape(), tensor.shape());
        }

        #[test]
        fn prop_unfold_fold_roundtrip(shape in shape_strategy()) {
            let tensor = DenseND::<f64>::ones(&shape);
            let rank = tensor.rank();

            // Test unfold-fold roundtrip for each mode
            for mode in 0..rank {
                let unfolded = tensor.unfold(mode).unwrap();
                prop_assert_eq!(unfolded.ndim(), 2);
                prop_assert_eq!(unfolded.shape()[0], shape[mode]);

                let folded = DenseND::fold(&unfolded, &shape, mode).unwrap();
                prop_assert_eq!(folded.shape(), tensor.shape());
            }
        }

        #[test]
        fn prop_unfold_dimensions(shape in shape_strategy()) {
            let tensor = DenseND::<f64>::zeros(&shape);
            let rank = tensor.rank();

            for mode in 0..rank {
                let unfolded = tensor.unfold(mode).unwrap();
                let mode_size = shape[mode];
                let other_size: usize = shape.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != mode)
                    .map(|(_, &s)| s)
                    .product();

                prop_assert_eq!(unfolded.shape(), [mode_size, other_size]);
            }
        }

        #[test]
        fn prop_random_uniform_bounds(
            rows in 2usize..20,
            cols in 2usize..20,
            low in -10.0f64..10.0,
            high in -10.0f64..10.0
        ) {
            if low >= high {
                return Ok(());
            }

            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], low, high);

            // Check all values are in range [low, high)
            for &val in tensor.as_slice() {
                prop_assert!(val >= low && val < high,
                    "Value {} outside range [{}, {})", val, low, high);
            }
        }

        #[test]
        fn prop_from_vec_consistency(
            rows in 2usize..20,
            cols in 2usize..20
        ) {
            let size = rows * cols;
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

            let tensor = DenseND::from_vec(data.clone(), &[rows, cols]).unwrap();

            prop_assert_eq!(tensor.shape(), &[rows, cols]);
            prop_assert_eq!(tensor.len(), size);

            // Verify data is in correct order
            for i in 0..rows {
                for j in 0..cols {
                    let expected = (i * cols + j) as f64;
                    prop_assert_eq!(tensor[&[i, j]], expected);
                }
            }
        }

        #[test]
        fn prop_index_bounds(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            let tensor = DenseND::<f64>::zeros(&[rows, cols]);

            // Valid indices should work
            for i in 0..rows {
                for j in 0..cols {
                    prop_assert!(tensor.get(&[i, j]).is_some());
                }
            }

            // Out of bounds should return None
            prop_assert!(tensor.get(&[rows, 0]).is_none());
            prop_assert!(tensor.get(&[0, cols]).is_none());
            prop_assert!(tensor.get(&[rows + 1, cols + 1]).is_none());
        }

        #[test]
        fn prop_frobenius_norm_positive(shape in shape_strategy()) {
            let tensor = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
            let norm = tensor.frobenius_norm();
            prop_assert!(norm >= 0.0);
        }

        #[test]
        fn prop_frobenius_norm_zero_tensor(shape in shape_strategy()) {
            let tensor = DenseND::<f64>::zeros(&shape);
            let norm = tensor.frobenius_norm();
            prop_assert!((norm - 0.0).abs() < 1e-10);
        }

        #[test]
        fn prop_addition_commutative(shape in shape_strategy()) {
            let a = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
            let b = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

            let sum1 = &a + &b;
            let sum2 = &b + &a;

            // Check shapes match
            prop_assert_eq!(sum1.shape(), sum2.shape());

            // Check all elements are equal (within floating point tolerance)
            for (v1, v2) in sum1.as_slice().iter().zip(sum2.as_slice().iter()) {
                prop_assert!((v1 - v2).abs() < 1e-10);
            }
        }

        #[test]
        fn prop_subtraction_inverse(shape in shape_strategy()) {
            let a = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);
            let b = DenseND::<f64>::random_uniform(&shape, 0.0, 1.0);

            let diff = &a - &b;
            let restored = &diff + &b;

            // Check shape
            prop_assert_eq!(restored.shape(), a.shape());

            // Check elements (within tolerance)
            for (orig, rest) in a.as_slice().iter().zip(restored.as_slice().iter()) {
                prop_assert!((orig - rest).abs() < 1e-10);
            }
        }

        #[test]
        fn prop_is_contiguous_after_creation(shape in shape_strategy()) {
            let tensor = DenseND::<f64>::zeros(&shape);
            prop_assert!(tensor.is_contiguous());
        }

        #[test]
        fn prop_is_square_correctness(
            size in 2usize..20
        ) {
            let square = DenseND::<f64>::zeros(&[size, size]);
            prop_assert!(square.is_square());

            let rect = DenseND::<f64>::zeros(&[size, size + 1]);
            prop_assert!(!rect.is_square());

            let tensor_3d = DenseND::<f64>::zeros(&[size, size, size]);
            prop_assert!(!tensor_3d.is_square());
        }

        #[test]
        fn prop_squeeze_unsqueeze_roundtrip(
            rows in 2usize..10,
            cols in 2usize..10,
            axis in 0usize..=3
        ) {
            let tensor = DenseND::<f64>::ones(&[rows, cols]);
            let rank = tensor.rank();

            if axis <= rank {
                let unsqueezed = tensor.unsqueeze(axis).unwrap();
                prop_assert_eq!(unsqueezed.rank(), rank + 1);
                prop_assert_eq!(unsqueezed.shape()[axis], 1);

                let squeezed = unsqueezed.squeeze_axis(axis).unwrap();
                prop_assert_eq!(squeezed.shape(), tensor.shape());
            }
        }

        #[test]
        fn prop_squeeze_removes_all_singletons(
            size1 in 2usize..10,
            size2 in 2usize..10
        ) {
            // Create tensor with singleton dimensions interspersed
            let tensor = DenseND::<f64>::zeros(&[1, size1, 1, size2, 1]);
            let squeezed = tensor.squeeze();

            // All singleton dimensions should be removed
            prop_assert_eq!(squeezed.shape(), &[size1, size2]);
            prop_assert!(squeezed.shape().iter().all(|&s| s > 1));
        }

        #[test]
        fn prop_unsqueeze_preserves_size(
            rows in 2usize..10,
            cols in 2usize..10,
            axis in 0usize..=2
        ) {
            let tensor = DenseND::<f64>::ones(&[rows, cols]);
            let original_len = tensor.len();

            let unsqueezed = tensor.unsqueeze(axis).unwrap();
            prop_assert_eq!(unsqueezed.len(), original_len);
        }

        #[test]
        fn prop_broadcast_preserves_values(
            rows in 2usize..8,
            cols in 2usize..8
        ) {
            // Test broadcasting (rows, 1) to (rows, cols)
            let tensor = DenseND::<f64>::ones(&[rows, 1]);
            let broadcast = tensor.broadcast_to(&[rows, cols]).unwrap();

            prop_assert_eq!(broadcast.shape(), &[rows, cols]);

            // All values should be 1.0
            for i in 0..rows {
                for j in 0..cols {
                    prop_assert_eq!(broadcast[&[i, j]], 1.0);
                }
            }
        }

        #[test]
        fn prop_broadcast_add_commutative(
            dim1 in 2usize..6,
            dim2 in 2usize..6
        ) {
            // Test that broadcasting addition is commutative
            // (dim1, 1) + (1, dim2) should equal (1, dim2) + (dim1, 1)
            let a = DenseND::<f64>::ones(&[dim1, 1]);
            let b = DenseND::<f64>::ones(&[1, dim2]);

            let result1 = &a + &b;
            let result2 = &b + &a;

            prop_assert_eq!(result1.shape(), result2.shape());

            // Check all elements are equal
            for i in 0..dim1 {
                for j in 0..dim2 {
                    prop_assert_eq!(result1[&[i, j]], result2[&[i, j]]);
                }
            }
        }

        #[test]
        fn prop_broadcast_consistent_with_manual(
            rows in 2usize..6,
            cols in 2usize..6
        ) {
            // Broadcasting (1, cols) to (rows, cols) should match manual expansion
            let tensor = DenseND::<f64>::random_uniform(&[1, cols], 0.0, 10.0);
            let broadcast = tensor.broadcast_to(&[rows, cols]).unwrap();

            // Check that each row is identical to the original
            for i in 0..rows {
                for j in 0..cols {
                    prop_assert_eq!(broadcast[&[i, j]], tensor[&[0, j]]);
                }
            }
        }

        #[test]
        fn prop_broadcast_3d(
            d1 in 2usize..5,
            d2 in 2usize..5,
            d3 in 2usize..5
        ) {
            // Test 3D broadcasting: (1, d2, 1) to (d1, d2, d3)
            let tensor = DenseND::<f64>::ones(&[1, d2, 1]);
            let broadcast = tensor.broadcast_to(&[d1, d2, d3]).unwrap();

            prop_assert_eq!(broadcast.shape(), &[d1, d2, d3]);

            // All values should be 1.0
            for i in 0..d1 {
                for j in 0..d2 {
                    for k in 0..d3 {
                        prop_assert_eq!(broadcast[&[i, j, k]], 1.0);
                    }
                }
            }
        }

        #[test]
        fn prop_broadcast_rank_promotion(
            inner_dim in 2usize..8,
            outer_dim in 2usize..8
        ) {
            // Test broadcasting from lower rank to higher rank
            // (inner_dim,) to (outer_dim, inner_dim)
            let tensor = DenseND::<f64>::ones(&[inner_dim]);
            let broadcast = tensor.broadcast_to(&[outer_dim, inner_dim]).unwrap();

            prop_assert_eq!(broadcast.shape(), &[outer_dim, inner_dim]);

            // All values should be 1.0
            for i in 0..outer_dim {
                for j in 0..inner_dim {
                    prop_assert_eq!(broadcast[&[i, j]], 1.0);
                }
            }
        }

        #[test]
        fn prop_median_within_range(
            size in 5usize..100
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.0, 100.0);
            let median = tensor.median();
            let min = tensor.min();
            let max = tensor.max();

            // Median should be within [min, max]
            prop_assert!(median >= *min);
            prop_assert!(median <= *max);
        }

        #[test]
        fn prop_median_constant_tensor(
            size in 5usize..100,
            value in -100.0f64..100.0
        ) {
            let tensor = DenseND::<f64>::from_elem(&[size], value);
            let median = tensor.median();

            // Median of constant tensor should equal the constant
            prop_assert!((median - value).abs() < 1e-10);
        }

        #[test]
        fn prop_quantile_bounds(
            size in 5usize..100,
            q in 0.0f64..=1.0
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.0, 100.0);
            let quantile = tensor.quantile(q);
            let min = tensor.min();
            let max = tensor.max();

            // Quantile should be within [min, max]
            prop_assert!(quantile >= *min - 1e-10);
            prop_assert!(quantile <= *max + 1e-10);
        }

        #[test]
        fn prop_quantile_endpoints(
            size in 5usize..100
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.0, 100.0);
            let q0 = tensor.quantile(0.0);
            let q1 = tensor.quantile(1.0);
            let min = tensor.min();
            let max = tensor.max();

            // q(0) should be min, q(1) should be max
            prop_assert!((q0 - *min).abs() < 1e-10);
            prop_assert!((q1 - *max).abs() < 1e-10);
        }

        #[test]
        fn prop_quantile_monotonic(
            size in 5usize..100,
            q1 in 0.0f64..0.5,
            q2 in 0.5f64..1.0
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.0, 100.0);
            let val1 = tensor.quantile(q1);
            let val2 = tensor.quantile(q2);

            // Quantiles should be monotonically increasing
            prop_assert!(val1 <= val2 + 1e-10);
        }

        #[test]
        fn prop_percentile_equals_quantile(
            size in 5usize..100,
            p in 0.0f64..=100.0
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.0, 100.0);
            let percentile = tensor.percentile(p);
            let quantile = tensor.quantile(p / 100.0);

            // percentile(p) should equal quantile(p/100)
            prop_assert!((percentile - quantile).abs() < 1e-10);
        }

        #[test]
        fn prop_median_equals_50th_percentile(
            size in 5usize..100
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.0, 100.0);
            let median = tensor.median();
            let p50 = tensor.percentile(50.0);

            // median should equal 50th percentile
            prop_assert!((median - p50).abs() < 1e-10);
        }

        #[test]
        fn prop_median_axis_output_shape(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.0, 100.0);

            let median_axis0 = tensor.median_axis(0, false).unwrap();
            prop_assert_eq!(median_axis0.shape(), &[cols]);

            let median_axis1 = tensor.median_axis(1, false).unwrap();
            prop_assert_eq!(median_axis1.shape(), &[rows]);
        }

        #[test]
        fn prop_quantile_axis_output_shape(
            rows in 2usize..10,
            cols in 2usize..10,
            q in 0.0f64..=1.0
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.0, 100.0);

            let q_axis0 = tensor.quantile_axis(q, 0, false).unwrap();
            prop_assert_eq!(q_axis0.shape(), &[cols]);

            let q_axis1 = tensor.quantile_axis(q, 1, false).unwrap();
            prop_assert_eq!(q_axis1.shape(), &[rows]);
        }

        #[test]
        fn prop_median_axis_within_range(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.0, 100.0);
            let median_axis0 = tensor.median_axis(0, false).unwrap();

            // Each median value should be within the global range
            let global_min = *tensor.min();
            let global_max = *tensor.max();

            for val in median_axis0.as_slice() {
                prop_assert!(*val >= global_min - 1e-10);
                prop_assert!(*val <= global_max + 1e-10);
            }
        }

        // ============================================================================
        // Product Reduction Properties
        // ============================================================================

        #[test]
        fn prop_prod_positive_values(
            size in 2usize..20
        ) {
            // Product of positive values should be positive
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.5, 2.0);
            let product = tensor.prod();
            prop_assert!(product > 0.0);
        }

        #[test]
        fn prop_prod_ones_tensor(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // Product of all ones should be 1.0
            let tensor = DenseND::<f64>::ones(&[rows, cols]);
            let product = tensor.prod();
            prop_assert!((product - 1.0).abs() < 1e-10);
        }

        #[test]
        fn prop_prod_axis_output_shape(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.5, 2.0);

            let prod_axis0 = tensor.prod_axis(0, false).unwrap();
            prop_assert_eq!(prod_axis0.shape(), &[cols]);

            let prod_axis1 = tensor.prod_axis(1, false).unwrap();
            prop_assert_eq!(prod_axis1.shape(), &[rows]);
        }

        #[test]
        fn prop_prod_axis_ones(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // Product along any axis for ones tensor should be all ones
            let tensor = DenseND::<f64>::ones(&[rows, cols]);
            let prod_axis0 = tensor.prod_axis(0, false).unwrap();

            for val in prod_axis0.as_slice() {
                prop_assert!((*val - 1.0).abs() < 1e-10);
            }
        }

        // ============================================================================
        // Cumulative Operation Properties
        // ============================================================================

        #[test]
        fn prop_cumsum_monotonic_increasing(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // Cumsum of positive values should be monotonically increasing
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.1, 1.0);
            let cumsum = tensor.cumsum(1).unwrap();

            // Check each row is monotonically increasing
            for i in 0..rows {
                for j in 1..cols {
                    prop_assert!(cumsum[&[i, j]] >= cumsum[&[i, j-1]]);
                }
            }
        }

        #[test]
        fn prop_cumsum_preserves_shape(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.0, 1.0);
            let cumsum = tensor.cumsum(0).unwrap();

            prop_assert_eq!(cumsum.shape(), tensor.shape());
        }

        #[test]
        fn prop_cumsum_first_element(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // First element along cumsum axis should equal original first element
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.0, 10.0);
            let cumsum = tensor.cumsum(1).unwrap();

            for i in 0..rows {
                prop_assert!((cumsum[&[i, 0]] - tensor[&[i, 0]]).abs() < 1e-10);
            }
        }

        #[test]
        fn prop_cumsum_last_equals_sum(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // Last element of cumsum should equal sum along that axis
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.0, 10.0);
            let cumsum = tensor.cumsum(1).unwrap();
            let sum_axis = tensor.sum_axis(1, false).unwrap();

            for i in 0..rows {
                prop_assert!((cumsum[&[i, cols-1]] - sum_axis[&[i]]).abs() < 1e-8);
            }
        }

        #[test]
        fn prop_cumprod_positive_values(
            size in 2usize..15
        ) {
            // Cumprod of positive values should remain positive
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.5, 2.0);
            let cumprod = tensor.cumprod(0).unwrap();

            for val in cumprod.as_slice() {
                prop_assert!(*val > 0.0);
            }
        }

        #[test]
        fn prop_cumprod_first_element(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // First element along cumprod axis should equal original first element
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.5, 2.0);
            let cumprod = tensor.cumprod(0).unwrap();

            for j in 0..cols {
                prop_assert!((cumprod[&[0, j]] - tensor[&[0, j]]).abs() < 1e-10);
            }
        }

        // ============================================================================
        // Norm Properties
        // ============================================================================

        #[test]
        fn prop_norm_l1_positive(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], -10.0, 10.0);
            let norm = tensor.norm_l1();
            prop_assert!(norm >= 0.0);
        }

        #[test]
        fn prop_norm_l1_zero_tensor(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            let tensor = DenseND::<f64>::zeros(&[rows, cols]);
            let norm = tensor.norm_l1();
            prop_assert!((norm - 0.0).abs() < 1e-10);
        }

        #[test]
        fn prop_norm_l2_equals_frobenius(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // L2 norm should equal Frobenius norm
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], -10.0, 10.0);
            let l2 = tensor.norm_l2();
            let frob = tensor.frobenius_norm();
            prop_assert!((l2 - frob).abs() < 1e-10);
        }

        #[test]
        fn prop_norm_linf_is_max_abs(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // Linf norm should equal max absolute value
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], -10.0, 10.0);
            let linf = tensor.norm_linf();
            let max_abs = tensor.as_slice().iter()
                .map(|x| x.abs())
                .fold(0.0f64, f64::max);

            prop_assert!((linf - max_abs).abs() < 1e-10);
        }

        #[test]
        fn prop_norm_lp_positive(
            rows in 2usize..10,
            cols in 2usize..10,
            p in 1.0f64..10.0
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], -10.0, 10.0);
            let norm = tensor.norm_lp(p);
            prop_assert!(norm >= 0.0);
        }

        #[test]
        fn prop_norm_lp_endpoints(
            rows in 2usize..8,
            cols in 2usize..8
        ) {
            // L1 norm via norm_lp(1.0) should match norm_l1()
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], -10.0, 10.0);
            let l1_direct = tensor.norm_l1();
            let l1_via_lp = tensor.norm_lp(1.0);
            prop_assert!((l1_direct - l1_via_lp).abs() < 1e-8);

            // L2 norm via norm_lp(2.0) should match norm_l2()
            let l2_direct = tensor.norm_l2();
            let l2_via_lp = tensor.norm_lp(2.0);
            prop_assert!((l2_direct - l2_via_lp).abs() < 1e-8);
        }

        #[test]
        fn prop_norm_lp_monotonicity(
            rows in 2usize..6,
            cols in 2usize..6
        ) {
            // For p1 < p2, we have ||x||_p1 >= ||x||_p2 (in general)
            // Test L1 >= L2 >= Linf property
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], -10.0, 10.0);
            let l1 = tensor.norm_l1();
            let l2 = tensor.norm_l2();
            let linf = tensor.norm_linf();

            // This property holds: Linf <= L2 <= L1 (scaled by dimension)
            prop_assert!(linf <= l2 + 1e-10);
            prop_assert!(linf <= l1 + 1e-10);
        }

        // ============================================================================
        // Sorting Properties
        // ============================================================================

        #[test]
        fn prop_sort_preserves_shape(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.0, 100.0);
            let sorted = tensor.sort(1).unwrap();
            prop_assert_eq!(sorted.shape(), tensor.shape());
        }

        #[test]
        fn prop_sort_is_sorted(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // Check that each row is sorted after sort(1)
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.0, 100.0);
            let sorted = tensor.sort(1).unwrap();

            for i in 0..rows {
                for j in 1..cols {
                    prop_assert!(sorted[&[i, j]] >= sorted[&[i, j-1]] - 1e-10);
                }
            }
        }

        #[test]
        fn prop_sort_preserves_values(
            size in 2usize..20
        ) {
            // Sorting should preserve the multiset of values
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.0, 100.0);
            let sorted = tensor.sort(0).unwrap();

            let mut orig_vals: Vec<_> = tensor.as_slice().to_vec();
            let mut sorted_vals: Vec<_> = sorted.as_slice().to_vec();
            orig_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for (o, s) in orig_vals.iter().zip(sorted_vals.iter()) {
                prop_assert!((o - s).abs() < 1e-10);
            }
        }

        #[test]
        fn prop_argsort_indices_valid(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            let tensor = DenseND::<f64>::random_uniform(&[rows, cols], 0.0, 100.0);
            let indices = tensor.argsort(1).unwrap();

            // All indices should be in range [0, cols)
            for idx in indices.as_slice() {
                prop_assert!(*idx < cols);
            }
        }

        #[test]
        fn prop_argsort_produces_sorted_array(
            size in 2usize..20
        ) {
            // Using argsort indices should produce sorted array
            let tensor = DenseND::<f64>::random_uniform(&[size], 0.0, 100.0);
            let indices = tensor.argsort(0).unwrap();

            // Manually apply indices to verify sorting
            let mut sorted_vals = Vec::with_capacity(size);
            for i in 0..size {
                let idx = indices[&[i]];
                sorted_vals.push(tensor[&[idx]]);
            }

            // Check sorted_vals is sorted
            for i in 1..size {
                prop_assert!(sorted_vals[i] >= sorted_vals[i-1] - 1e-10);
            }
        }

        // ============================================================================
        // Dot Product Properties
        // ============================================================================

        #[test]
        fn prop_dot_vector_commutative(
            size in 2usize..20
        ) {
            // Vector dot product should be commutative: a·b = b·a
            let a = DenseND::<f64>::random_uniform(&[size], -10.0, 10.0);
            let b = DenseND::<f64>::random_uniform(&[size], -10.0, 10.0);

            let dot_ab = a.dot(&b).unwrap();
            let dot_ba = b.dot(&a).unwrap();

            // Should return [1] shaped tensor for vector dot product
            prop_assert_eq!(dot_ab.shape(), &[1]);
            prop_assert_eq!(dot_ba.shape(), &[1]);

            // Values should be equal
            prop_assert!((dot_ab[&[0]] - dot_ba[&[0]]).abs() < 1e-8);
        }

        #[test]
        fn prop_dot_vector_zero(
            size in 2usize..20
        ) {
            // Dot product with zero vector should be zero
            let a = DenseND::<f64>::random_uniform(&[size], -10.0, 10.0);
            let zero = DenseND::<f64>::zeros(&[size]);

            let result = a.dot(&zero).unwrap();
            prop_assert_eq!(result.shape(), &[1]);
            prop_assert!((result[&[0]] - 0.0).abs() < 1e-10);
        }

        #[test]
        fn prop_dot_matrix_vector_shape(
            rows in 2usize..10,
            cols in 2usize..10
        ) {
            // Matrix-vector dot product should produce vector of length rows
            let matrix = DenseND::<f64>::random_uniform(&[rows, cols], -10.0, 10.0);
            let vector = DenseND::<f64>::random_uniform(&[cols], -10.0, 10.0);

            let result = matrix.dot(&vector).unwrap();
            prop_assert_eq!(result.shape(), &[rows]);
        }

        #[test]
        fn prop_dot_matrix_matrix_shape(
            m in 2usize..8,
            n in 2usize..8,
            k in 2usize..8
        ) {
            // Matrix-matrix dot product: (m,n) · (n,k) = (m,k)
            let a = DenseND::<f64>::random_uniform(&[m, n], -10.0, 10.0);
            let b = DenseND::<f64>::random_uniform(&[n, k], -10.0, 10.0);

            let result = a.dot(&b).unwrap();
            prop_assert_eq!(result.shape(), &[m, k]);
        }

        #[test]
        fn prop_dot_identity_matrix(
            size in 2usize..10
        ) {
            // A · I = A (dot product with identity)
            let matrix = DenseND::<f64>::random_uniform(&[size, size], -10.0, 10.0);
            let identity = DenseND::<f64>::eye(size);

            let result = matrix.dot(&identity).unwrap();

            // Result should equal original matrix
            for i in 0..size {
                for j in 0..size {
                    prop_assert!((result[&[i, j]] - matrix[&[i, j]]).abs() < 1e-8);
                }
            }
        }
    }
}
