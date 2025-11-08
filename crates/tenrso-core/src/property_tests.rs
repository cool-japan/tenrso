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
    }
}
