//! Integration tests for tenrso-core
//!
//! These tests verify end-to-end functionality and cross-module interactions.

use tenrso_core::{AxisMeta, DenseND, TensorHandle};

#[test]
fn test_tensor_handle_creation_and_access() {
    // Create a dense tensor
    let data = DenseND::<f64>::zeros(&[2, 3, 4]);

    // Create axes metadata
    let axes = vec![
        AxisMeta::new("batch", 2),
        AxisMeta::new("height", 3),
        AxisMeta::new("width", 4),
    ];

    // Wrap in TensorHandle
    let handle = TensorHandle::from_dense(data.clone(), axes);

    // Verify shape and rank
    assert_eq!(handle.rank(), 3);
    assert_eq!(handle.shape().as_slice(), &[2, 3, 4]);

    // Access as dense
    let dense_ref = handle.as_dense().unwrap();
    assert_eq!(dense_ref.shape(), &[2, 3, 4]);
}

#[test]
fn test_unfold_fold_roundtrip_integration() {
    // Create a 3D tensor
    let data = DenseND::<f64>::from_vec((1..=24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Test unfold-fold roundtrip for all modes
    for mode in 0..3 {
        let unfolded = data.unfold(mode).unwrap();
        let folded = DenseND::fold(&unfolded, &[2, 3, 4], mode).unwrap();

        // Verify roundtrip
        assert_eq!(data.shape(), folded.shape());

        // Verify values match
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let orig = data[&[i, j, k]];
                    let result = folded[&[i, j, k]];
                    assert!(
                        (orig - result).abs() < 1e-10,
                        "Mismatch at [{}, {}, {}]",
                        i,
                        j,
                        k
                    );
                }
            }
        }
    }
}

#[test]
fn test_reshape_permute_chain() {
    let data = DenseND::<f64>::from_vec((1..=24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Reshape then permute
    let reshaped = data.reshape(&[6, 4]).unwrap();
    let permuted = reshaped.permute(&[1, 0]).unwrap();

    assert_eq!(permuted.shape(), &[4, 6]);

    // Permute then reshape (where possible)
    let permuted2 = data.permute(&[2, 0, 1]).unwrap();
    let reshaped2 = permuted2.reshape(&[4, 6]).unwrap();

    assert_eq!(reshaped2.shape(), &[4, 6]);
}

#[test]
fn test_statistical_operations_integration() {
    let data = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    // Global statistics
    let sum = data.sum();
    let mean = data.mean();
    let min = data.min();
    let max = data.max();

    assert_eq!(sum, 21.0);
    assert_eq!(mean, 3.5);
    assert_eq!(min, &1.0);
    assert_eq!(max, &6.0);

    // Axis-wise statistics
    let sum_axis0 = data.sum_axis(0, false).unwrap();
    assert_eq!(sum_axis0.shape(), &[3]);

    let mean_axis1 = data.mean_axis(1, false).unwrap();
    assert_eq!(mean_axis1.shape(), &[2]);
}

#[test]
fn test_covariance_correlation_integration() {
    // Test covariance between two variables
    let x = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let y = DenseND::<f64>::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();

    // Perfect linear relationship: y = 2x
    let cov = x.covariance(&y).unwrap();
    assert!(cov > 0.0, "Covariance should be positive");

    let corr = x.correlation(&y).unwrap();
    assert!(
        (corr - 1.0).abs() < 1e-10,
        "Correlation should be perfect (1.0)"
    );

    // Test covariance matrix
    let data = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let cov_matrix = data.covariance_matrix(false).unwrap();
    assert_eq!(cov_matrix.shape(), &[2, 2]);

    // Covariance matrix should be symmetric
    assert!((cov_matrix[&[0, 1]] - cov_matrix[&[1, 0]]).abs() < 1e-10);
}

#[test]
fn test_windowing_operations_integration() {
    // Test sliding window for 1D signal
    let signal = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let windows = signal.sliding_window(3, 1, 0).unwrap();

    assert_eq!(windows.shape(), &[3, 3]);
    // First window should be [1, 2, 3]
    assert_eq!(windows[&[0, 0]], 1.0);
    assert_eq!(windows[&[0, 1]], 2.0);
    assert_eq!(windows[&[0, 2]], 3.0);

    // Test strided view for downsampling
    let data = DenseND::<f64>::from_vec((1..=9).map(|x| x as f64).collect(), &[3, 3]).unwrap();
    let strided = data.strided_view(&[2, 2]).unwrap();

    assert_eq!(strided.shape(), &[2, 2]);
    assert_eq!(strided[&[0, 0]], 1.0); // data[0, 0]
    assert_eq!(strided[&[0, 1]], 3.0); // data[0, 2]
    assert_eq!(strided[&[1, 0]], 7.0); // data[2, 0]
    assert_eq!(strided[&[1, 1]], 9.0); // data[2, 2]

    // Test extract_patches for 2D image
    let img = DenseND::<f64>::from_vec((1..=16).map(|x| x as f64).collect(), &[4, 4]).unwrap();
    let patches = img.extract_patches((2, 2), (2, 2)).unwrap();

    assert_eq!(patches.shape(), &[2, 2, 2, 2]); // 2x2 grid of 2x2 patches
}

#[test]
fn test_arithmetic_operations_integration() {
    let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    // Element-wise addition
    let sum = &a + &b;
    assert_eq!(sum[&[0, 0]], 6.0);
    assert_eq!(sum[&[1, 1]], 12.0);

    // Element-wise subtraction
    let diff = &b - &a;
    assert_eq!(diff[&[0, 0]], 4.0);
    assert_eq!(diff[&[1, 1]], 4.0);

    // Scalar multiplication
    let scaled = &a * 2.0;
    assert_eq!(scaled[&[0, 0]], 2.0);
    assert_eq!(scaled[&[1, 1]], 8.0);
}

#[test]
fn test_linear_algebra_integration() {
    let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    // Matrix multiplication
    let product = a.matmul(&b).unwrap();
    assert_eq!(product.shape(), &[2, 2]);
    // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    assert_eq!(product[&[0, 0]], 19.0);
    assert_eq!(product[&[0, 1]], 22.0);
    assert_eq!(product[&[1, 0]], 43.0);
    assert_eq!(product[&[1, 1]], 50.0);

    // Transpose
    let transposed = a.transpose().unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed[&[0, 0]], 1.0);
    assert_eq!(transposed[&[0, 1]], 3.0);
    assert_eq!(transposed[&[1, 0]], 2.0);
    assert_eq!(transposed[&[1, 1]], 4.0);

    // Trace
    let trace = a.trace().unwrap();
    assert_eq!(trace, 5.0); // 1 + 4 = 5
}

#[test]
fn test_comparison_and_masking_integration() {
    let data = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

    // Greater than comparison
    let gt_mask = data.mask_gt(3.0);
    assert_eq!(gt_mask, vec![false, false, false, true, true]);

    // Count elements satisfying predicate
    let count = data.count_if(|&x| x > 3.0);
    assert_eq!(count, 2);

    // Filter by mask
    let filtered = data.select_mask(&gt_mask).unwrap();
    assert_eq!(filtered.shape(), &[2]);
    assert_eq!(filtered[&[0]], 4.0);
    assert_eq!(filtered[&[1]], 5.0);
}

#[test]
fn test_array_manipulation_integration() {
    let data = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

    // Roll elements
    let rolled = data.roll(2, 0).unwrap();
    assert_eq!(rolled[&[0]], 3.0);
    assert_eq!(rolled[&[1]], 4.0);
    assert_eq!(rolled[&[2]], 1.0);
    assert_eq!(rolled[&[3]], 2.0);

    // Flip array
    let flipped = data.flip(0).unwrap();
    assert_eq!(flipped[&[0]], 4.0);
    assert_eq!(flipped[&[1]], 3.0);
    assert_eq!(flipped[&[2]], 2.0);
    assert_eq!(flipped[&[3]], 1.0);

    // Tile array
    let tiled = data.tile(&[2]).unwrap();
    assert_eq!(tiled.shape(), &[8]);
    assert_eq!(tiled[&[4]], 1.0); // Second repetition starts
}

#[test]
fn test_advanced_indexing_integration() {
    let data = DenseND::<f64>::from_vec((1..=10).map(|x| x as f64).collect(), &[10]).unwrap();

    // Take specific indices
    let indices = vec![0, 2, 4, 6];
    let taken = data.select_indices(&indices, 0).unwrap();
    assert_eq!(taken.shape(), &[4]);
    assert_eq!(taken[&[0]], 1.0);
    assert_eq!(taken[&[1]], 3.0);
    assert_eq!(taken[&[2]], 5.0);
    assert_eq!(taken[&[3]], 7.0);

    // Unique values
    let dup_data = DenseND::<f64>::from_vec(vec![1.0, 2.0, 2.0, 3.0, 1.0, 3.0], &[6]).unwrap();
    let unique = dup_data.unique();
    assert_eq!(unique.shape(), &[3]);

    let (unique_vals, counts) = dup_data.unique_with_counts();
    assert_eq!(unique_vals.shape(), &[3]);
    assert_eq!(counts.shape(), &[3]);
}

#[test]
fn test_padding_operations_integration() {
    use tenrso_core::PadMode;

    let data = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    // Constant padding
    let padded = data.pad(&[(1, 1), (1, 1)], PadMode::Constant, 0.0).unwrap();
    assert_eq!(padded.shape(), &[4, 4]);
    assert_eq!(padded[&[0, 0]], 0.0); // Padding
    assert_eq!(padded[&[1, 1]], 1.0); // Original data at padded[1,1] = data[0,0]
    assert_eq!(padded[&[1, 2]], 2.0); // Original data at padded[1,2] = data[0,1]
    assert_eq!(padded[&[2, 1]], 3.0); // Original data at padded[2,1] = data[1,0]
    assert_eq!(padded[&[2, 2]], 4.0); // Original data at padded[2,2] = data[1,1]

    // Edge padding - skip for now as it has complex logic
    // let edge_padded = data.pad(&[(1, 1), (1, 1)], PadMode::Edge, 0.0).unwrap();
    // assert_eq!(edge_padded.shape(), &[4, 4]);
}

#[test]
fn test_cumulative_operations_integration() {
    let data = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    // Cumulative sum along axis
    let cumsum = data.cumsum(1).unwrap();
    assert_eq!(cumsum.shape(), &[2, 3]);
    // First row: [1, 3, 6] (1, 1+2, 1+2+3)
    assert_eq!(cumsum[&[0, 0]], 1.0);
    assert_eq!(cumsum[&[0, 1]], 3.0);
    assert_eq!(cumsum[&[0, 2]], 6.0);

    // Cumulative product
    let cumprod = data.cumprod(1).unwrap();
    assert_eq!(cumprod.shape(), &[2, 3]);
    // First row: [1, 2, 6] (1, 1*2, 1*2*3)
    assert_eq!(cumprod[&[0, 0]], 1.0);
    assert_eq!(cumprod[&[0, 1]], 2.0);
    assert_eq!(cumprod[&[0, 2]], 6.0);
}

#[test]
fn test_norm_operations_integration() {
    let data = DenseND::<f64>::from_vec(vec![3.0, 4.0], &[2]).unwrap();

    // L1 norm (Manhattan)
    let l1 = data.norm_l1();
    assert_eq!(l1, 7.0);

    // L2 norm (Euclidean)
    let l2 = data.norm_l2();
    assert_eq!(l2, 5.0); // sqrt(3^2 + 4^2) = 5

    // Linf norm (Maximum)
    let linf = data.norm_linf();
    assert_eq!(linf, 4.0);

    // Lp norm
    let lp = data.norm_lp(2.0);
    assert!((lp - 5.0).abs() < 1e-10);
}

#[test]
fn test_sorting_integration() {
    let data = DenseND::<f64>::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]).unwrap();

    // Sort along axis
    let sorted = data.sort(1).unwrap();
    assert_eq!(sorted.shape(), &[2, 3]);
    // First row sorted: [1, 3, 4]
    assert_eq!(sorted[&[0, 0]], 1.0);
    assert_eq!(sorted[&[0, 1]], 3.0);
    assert_eq!(sorted[&[0, 2]], 4.0);

    // Argsort
    let indices = data.argsort(1).unwrap();
    assert_eq!(indices.shape(), &[2, 3]);
}

#[test]
fn test_broadcasting_integration() {
    let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let b = DenseND::<f64>::from_vec(vec![10.0, 20.0], &[1, 2]).unwrap();

    // Broadcast to compatible shape
    let broadcasted_a = a.broadcast_to(&[3, 2]).unwrap();
    assert_eq!(broadcasted_a.shape(), &[3, 2]);
    assert_eq!(broadcasted_a[&[0, 0]], 1.0);
    assert_eq!(broadcasted_a[&[0, 1]], 1.0);

    // Broadcasting in addition
    let sum = &broadcasted_a + &b.broadcast_to(&[3, 2]).unwrap();
    assert_eq!(sum.shape(), &[3, 2]);
}

#[test]
fn test_pooling_operations_integration() {
    // Create a 4x4 test image
    let data = DenseND::<f64>::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[4, 4],
    )
    .unwrap();

    // Test max pooling 2x2 with stride 2 (non-overlapping)
    let max_pooled = data.max_pool_2d((2, 2), None).unwrap();
    assert_eq!(max_pooled.shape(), &[2, 2]);
    assert_eq!(max_pooled[&[0, 0]], 6.0); // max of [1, 2, 5, 6]
    assert_eq!(max_pooled[&[0, 1]], 8.0); // max of [3, 4, 7, 8]
    assert_eq!(max_pooled[&[1, 0]], 14.0); // max of [9, 10, 13, 14]
    assert_eq!(max_pooled[&[1, 1]], 16.0); // max of [11, 12, 15, 16]

    // Test average pooling 2x2 with stride 2
    let avg_pooled = data.avg_pool_2d((2, 2), None).unwrap();
    assert_eq!(avg_pooled.shape(), &[2, 2]);
    assert_eq!(avg_pooled[&[0, 0]], 3.5); // avg of [1, 2, 5, 6] = 14/4
    assert_eq!(avg_pooled[&[0, 1]], 5.5); // avg of [3, 4, 7, 8] = 22/4
    assert_eq!(avg_pooled[&[1, 0]], 11.5); // avg of [9, 10, 13, 14] = 46/4
    assert_eq!(avg_pooled[&[1, 1]], 13.5); // avg of [11, 12, 15, 16] = 54/4

    // Test overlapping pooling (stride < kernel_size)
    let overlap_pooled = data.max_pool_2d((2, 2), Some((1, 1))).unwrap();
    assert_eq!(overlap_pooled.shape(), &[3, 3]);
    assert_eq!(overlap_pooled[&[0, 0]], 6.0);
    assert_eq!(overlap_pooled[&[1, 1]], 11.0); // max of [6, 7, 10, 11]

    // Test adaptive pooling
    let adaptive = data.adaptive_avg_pool_2d((2, 2)).unwrap();
    assert_eq!(adaptive.shape(), &[2, 2]);
    // Each 2x2 block of the 4x4 input is averaged
    assert_eq!(adaptive[&[0, 0]], 3.5); // avg of first 2x2 quadrant
}

#[test]
fn test_pooling_with_windowing_integration() {
    // Combine windowing and pooling for CNN-like operations
    let img = DenseND::<f64>::from_vec((1..=64).map(|x| x as f64).collect(), &[8, 8]).unwrap();

    // First, extract 4x4 patches with stride 2
    let patches = img.extract_patches((4, 4), (2, 2)).unwrap();
    assert_eq!(patches.shape(), &[3, 3, 4, 4]); // 3x3 grid of 4x4 patches

    // Then we could pool each patch (in practice, would reshape and pool)
    // For now, verify patch extraction works correctly
    assert!(patches[&[0, 0, 0, 0]] == 1.0); // First patch, first element

    // Test strided view followed by max pooling
    let strided = img.strided_view(&[2, 2]).unwrap();
    assert_eq!(strided.shape(), &[4, 4]);

    let pooled = strided.max_pool_2d((2, 2), None).unwrap();
    assert_eq!(pooled.shape(), &[2, 2]);
}
