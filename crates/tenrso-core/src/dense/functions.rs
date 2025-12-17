//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DenseND;
use scirs2_core::ndarray_ext::{Array, IxDyn};
use scirs2_core::numeric::Num;

pub(crate) fn shapes_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);
    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }
    true
}
fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    if !shapes_broadcastable(shape1, shape2) {
        return None;
    }
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };
        result.push(dim1.max(dim2));
    }
    result.reverse();
    Some(result)
}
pub(crate) fn broadcast_copy<T>(
    src: &Array<T, IxDyn>,
    dst: &mut Array<T, IxDyn>,
    src_shape: &[usize],
    dst_shape: &[usize],
) -> anyhow::Result<()>
where
    T: Clone + Num,
{
    let src_rank = src_shape.len();
    let dst_rank = dst_shape.len();
    let rank_diff = dst_rank.saturating_sub(src_rank);
    let map_index = |dst_idx: &[usize]| -> Vec<usize> {
        let mut src_idx = Vec::with_capacity(src_rank);
        for (i, &src_dim) in src_shape.iter().enumerate() {
            let dst_dim_idx = rank_diff + i;
            let dst_val = dst_idx[dst_dim_idx];
            src_idx.push(if src_dim == 1 { 0 } else { dst_val });
        }
        src_idx
    };
    let total_elements: usize = dst_shape.iter().product();
    for flat_idx in 0..total_elements {
        let mut dst_idx = Vec::with_capacity(dst_rank);
        let mut remaining = flat_idx;
        for i in (0..dst_rank).rev() {
            let dim_size = dst_shape[i];
            dst_idx.insert(0, remaining % dim_size);
            remaining /= dim_size;
        }
        let src_idx = map_index(&dst_idx);
        dst[IxDyn(&dst_idx)] = src[IxDyn(&src_idx)].clone();
    }
    Ok(())
}
pub(crate) fn generate_indices(shape: &[usize], axis: usize, axis_value: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut indices = vec![0; shape.len()];
    indices[axis] = axis_value;
    fn recurse(
        shape: &[usize],
        axis: usize,
        indices: &mut [usize],
        current_dim: usize,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current_dim == shape.len() {
            result.push(indices.to_vec());
            return;
        }
        if current_dim == axis {
            recurse(shape, axis, indices, current_dim + 1, result);
        } else {
            for i in 0..shape[current_dim] {
                indices[current_dim] = i;
                recurse(shape, axis, indices, current_dim + 1, result);
            }
        }
    }
    recurse(shape, axis, &mut indices, 0, &mut result);
    result
}
impl<'b, T> std::ops::Sub<&'b DenseND<T>> for &DenseND<T>
where
    T: Clone + Num,
{
    type Output = DenseND<T>;
    fn sub(self, rhs: &'b DenseND<T>) -> Self::Output {
        let target_shape = broadcast_shape(self.shape(), rhs.shape())
            .expect("Shapes are not broadcastable for subtraction");
        let lhs_broadcast = if self.shape() != target_shape.as_slice() {
            self.broadcast_to(&target_shape)
                .expect("Failed to broadcast left operand")
        } else {
            self.clone()
        };
        let rhs_broadcast = if rhs.shape() != target_shape.as_slice() {
            rhs.broadcast_to(&target_shape)
                .expect("Failed to broadcast right operand")
        } else {
            rhs.clone()
        };
        let result = &lhs_broadcast.data - &rhs_broadcast.data;
        DenseND { data: result }
    }
}
impl<'b, T> std::ops::Add<&'b DenseND<T>> for &DenseND<T>
where
    T: Clone + Num,
{
    type Output = DenseND<T>;
    fn add(self, rhs: &'b DenseND<T>) -> Self::Output {
        let target_shape = broadcast_shape(self.shape(), rhs.shape())
            .expect("Shapes are not broadcastable for addition");
        let lhs_broadcast = if self.shape() != target_shape.as_slice() {
            self.broadcast_to(&target_shape)
                .expect("Failed to broadcast left operand")
        } else {
            self.clone()
        };
        let rhs_broadcast = if rhs.shape() != target_shape.as_slice() {
            rhs.broadcast_to(&target_shape)
                .expect("Failed to broadcast right operand")
        } else {
            rhs.clone()
        };
        let result = &lhs_broadcast.data + &rhs_broadcast.data;
        DenseND { data: result }
    }
}
impl<T> std::ops::Mul<T> for &DenseND<T>
where
    T: Clone + Num + scirs2_core::ndarray_ext::ScalarOperand,
{
    type Output = DenseND<T>;
    fn mul(self, scalar: T) -> Self::Output {
        let result = &self.data * scalar;
        DenseND { data: result }
    }
}
impl<T> std::ops::Div<T> for &DenseND<T>
where
    T: Clone + Num + scirs2_core::ndarray_ext::ScalarOperand,
{
    type Output = DenseND<T>;
    fn div(self, scalar: T) -> Self::Output {
        let result = &self.data / scalar;
        DenseND { data: result }
    }
}
impl<T> std::ops::Add<T> for &DenseND<T>
where
    T: Clone + Num + scirs2_core::ndarray_ext::ScalarOperand,
{
    type Output = DenseND<T>;
    fn add(self, scalar: T) -> Self::Output {
        let result = &self.data + scalar;
        DenseND { data: result }
    }
}
impl<T> std::ops::Sub<T> for &DenseND<T>
where
    T: Clone + Num + scirs2_core::ndarray_ext::ScalarOperand,
{
    type Output = DenseND<T>;
    fn sub(self, scalar: T) -> Self::Output {
        let result = &self.data - scalar;
        DenseND { data: result }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_create_zeros() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.len(), 24);
        assert_eq!(tensor[&[0, 0, 0]], 0.0);
    }
    #[test]
    fn test_create_ones() {
        let tensor = DenseND::<f64>::ones(&[2, 3]);
        assert_eq!(tensor[&[0, 0]], 1.0);
        assert_eq!(tensor[&[1, 2]], 1.0);
    }
    #[test]
    fn test_from_elem() {
        let tensor = DenseND::from_elem(&[2, 3], 5.0);
        assert_eq!(tensor[&[0, 0]], 5.0);
        assert_eq!(tensor[&[1, 2]], 5.0);
    }
    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = DenseND::from_vec(data, &[2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor[&[0, 0]], 1.0);
        assert_eq!(tensor[&[1, 2]], 6.0);
    }
    #[test]
    fn test_reshape() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let reshaped = tensor.reshape(&[6, 4]).unwrap();
        assert_eq!(reshaped.shape(), &[6, 4]);
        assert_eq!(reshaped.len(), 24);
    }
    #[test]
    fn test_permute() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let permuted = tensor.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.shape(), &[4, 2, 3]);
    }
    #[test]
    fn test_unfold() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let unfolded = tensor.unfold(1).unwrap();
        assert_eq!(unfolded.shape(), &[3, 8]);
    }
    #[test]
    fn test_unfold_fold_roundtrip() {
        let shape = vec![2, 3, 4];
        let tensor = DenseND::<f64>::ones(&shape);
        for mode in 0..3 {
            let unfolded = tensor.unfold(mode).unwrap();
            let folded = DenseND::fold(&unfolded, &shape, mode).unwrap();
            assert_eq!(folded.shape(), tensor.shape());
            assert_eq!(folded.len(), tensor.len());
        }
    }
    #[test]
    fn test_random_uniform() {
        let tensor = DenseND::<f64>::random_uniform(&[10, 10], 0.0, 1.0);
        assert_eq!(tensor.shape(), &[10, 10]);
        for val in tensor.data.iter() {
            assert!(*val >= 0.0 && *val < 1.0);
        }
    }
    #[test]
    fn test_random_normal() {
        let tensor = DenseND::<f64>::random_normal(&[10, 10], 0.0, 1.0);
        assert_eq!(tensor.shape(), &[10, 10]);
    }
    #[test]
    fn test_squeeze_all() {
        let tensor = DenseND::<f64>::zeros(&[2, 1, 3, 1, 4]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[2, 3, 4]);
    }
    #[test]
    fn test_squeeze_all_ones() {
        let tensor = DenseND::<f64>::zeros(&[1, 1, 1]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[1]);
    }
    #[test]
    fn test_squeeze_no_singletons() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[2, 3, 4]);
    }
    #[test]
    fn test_squeeze_axis() {
        let tensor = DenseND::<f64>::zeros(&[2, 1, 3, 4]);
        let squeezed = tensor.squeeze_axis(1).unwrap();
        assert_eq!(squeezed.shape(), &[2, 3, 4]);
    }
    #[test]
    fn test_squeeze_axis_error() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        assert!(tensor.squeeze_axis(1).is_err());
    }
    #[test]
    fn test_unsqueeze_beginning() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let unsqueezed = tensor.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 2, 3, 4]);
    }
    #[test]
    fn test_unsqueeze_middle() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let unsqueezed = tensor.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 1, 3, 4]);
    }
    #[test]
    fn test_unsqueeze_end() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let unsqueezed = tensor.unsqueeze(3).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 3, 4, 1]);
    }
    #[test]
    fn test_unsqueeze_squeeze_roundtrip() {
        let tensor = DenseND::<f64>::ones(&[2, 3, 4]);
        let unsqueezed = tensor.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 1, 3, 4]);
        let squeezed = unsqueezed.squeeze_axis(1).unwrap();
        assert_eq!(squeezed.shape(), &[2, 3, 4]);
    }
    #[test]
    fn test_concatenate_axis_0() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);
        let concat = DenseND::concatenate(&[a, b], 0).unwrap();
        assert_eq!(concat.shape(), &[4, 3]);
    }
    #[test]
    fn test_concatenate_axis_1() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 4]);
        let concat = DenseND::concatenate(&[a, b], 1).unwrap();
        assert_eq!(concat.shape(), &[2, 7]);
    }
    #[test]
    fn test_concatenate_multiple() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);
        let c = DenseND::<f64>::from_elem(&[2, 3], 5.0);
        let concat = DenseND::concatenate(&[a, b, c], 0).unwrap();
        assert_eq!(concat.shape(), &[6, 3]);
    }
    #[test]
    fn test_stack_axis_0() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);
        let stacked = DenseND::stack(&[a, b], 0).unwrap();
        assert_eq!(stacked.shape(), &[2, 2, 3]);
    }
    #[test]
    fn test_stack_axis_1() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);
        let stacked = DenseND::stack(&[a, b], 1).unwrap();
        assert_eq!(stacked.shape(), &[2, 2, 3]);
    }
    #[test]
    fn test_stack_axis_2() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);
        let c = DenseND::<f64>::from_elem(&[2, 3], 5.0);
        let stacked = DenseND::stack(&[a, b, c], 2).unwrap();
        assert_eq!(stacked.shape(), &[2, 3, 3]);
    }
    #[test]
    fn test_scalar_multiply() {
        let tensor = DenseND::<f64>::from_elem(&[2, 3], 2.0);
        let result = &tensor * 3.0;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[&[0, 0]], 6.0);
        assert_eq!(result[&[1, 2]], 6.0);
    }
    #[test]
    fn test_scalar_divide() {
        let tensor = DenseND::<f64>::from_elem(&[2, 3], 12.0);
        let result = &tensor / 4.0;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[&[0, 0]], 3.0);
        assert_eq!(result[&[1, 2]], 3.0);
    }
    #[test]
    fn test_scalar_add() {
        let tensor = DenseND::<f64>::from_elem(&[2, 3], 2.0);
        let result = &tensor + 5.0;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[&[0, 0]], 7.0);
        assert_eq!(result[&[1, 2]], 7.0);
    }
    #[test]
    fn test_scalar_subtract() {
        let tensor = DenseND::<f64>::from_elem(&[2, 3], 10.0);
        let result = &tensor - 3.0;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[&[0, 0]], 7.0);
        assert_eq!(result[&[1, 2]], 7.0);
    }
    #[test]
    fn test_split_axis_0() {
        let tensor = DenseND::<f64>::ones(&[6, 4]);
        let splits = tensor.split(3, 0).unwrap();
        assert_eq!(splits.len(), 3);
        for split in &splits {
            assert_eq!(split.shape(), &[2, 4]);
        }
    }
    #[test]
    fn test_split_axis_1() {
        let tensor = DenseND::<f64>::ones(&[6, 4]);
        let splits = tensor.split(2, 1).unwrap();
        assert_eq!(splits.len(), 2);
        for split in &splits {
            assert_eq!(split.shape(), &[6, 2]);
        }
    }
    #[test]
    fn test_split_error_not_divisible() {
        let tensor = DenseND::<f64>::ones(&[7, 4]);
        assert!(tensor.split(3, 0).is_err());
    }
    #[test]
    fn test_chunk_equal_size() {
        let tensor = DenseND::<f64>::ones(&[6, 4]);
        let chunks = tensor.chunk(2, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.shape(), &[2, 4]);
        }
    }
    #[test]
    fn test_chunk_last_smaller() {
        let tensor = DenseND::<f64>::ones(&[7, 4]);
        let chunks = tensor.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[3, 4]);
        assert_eq!(chunks[1].shape(), &[3, 4]);
        assert_eq!(chunks[2].shape(), &[1, 4]);
    }
    #[test]
    fn test_split_concatenate_roundtrip() {
        let tensor = DenseND::<f64>::ones(&[6, 4]);
        let splits = tensor.split(3, 0).unwrap();
        let reconstructed = DenseND::concatenate(&splits, 0).unwrap();
        assert_eq!(reconstructed.shape(), tensor.shape());
    }
    #[test]
    fn test_transpose() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let transposed = tensor.transpose().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed[&[0, 0]], 1.0);
        assert_eq!(transposed[&[0, 1]], 4.0);
        assert_eq!(transposed[&[1, 0]], 2.0);
        assert_eq!(transposed[&[1, 1]], 5.0);
    }
    #[test]
    fn test_transpose_error_3d() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        assert!(tensor.transpose().is_err());
    }
    #[test]
    fn test_matmul() {
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c[&[0, 0]], 19.0);
        assert_eq!(c[&[0, 1]], 22.0);
        assert_eq!(c[&[1, 0]], 43.0);
        assert_eq!(c[&[1, 1]], 50.0);
    }
    #[test]
    fn test_matmul_rectangular() {
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }
    #[test]
    fn test_matmul_error_mismatch() {
        let a = DenseND::<f64>::zeros(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[4, 2]);
        assert!(a.matmul(&b).is_err());
    }
    #[test]
    fn test_hadamard() {
        let a = DenseND::<f64>::from_elem(&[2, 3], 2.0);
        let b = DenseND::<f64>::from_elem(&[2, 3], 3.0);
        let c = a.hadamard(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c[&[0, 0]], 6.0);
        assert_eq!(c[&[1, 2]], 6.0);
    }
    #[test]
    fn test_hadamard_error_shape_mismatch() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::ones(&[3, 2]);
        assert!(a.hadamard(&b).is_err());
    }
    #[test]
    fn test_diagonal_square() {
        let tensor =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
                .unwrap();
        let diag = tensor.diagonal().unwrap();
        assert_eq!(diag.shape(), &[3]);
        assert_eq!(diag[&[0]], 1.0);
        assert_eq!(diag[&[1]], 5.0);
        assert_eq!(diag[&[2]], 9.0);
    }
    #[test]
    fn test_diagonal_rectangular() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let diag = tensor.diagonal().unwrap();
        assert_eq!(diag.shape(), &[2]);
        assert_eq!(diag[&[0]], 1.0);
        assert_eq!(diag[&[1]], 5.0);
    }
    #[test]
    fn test_trace() {
        let tensor =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
                .unwrap();
        let trace = tensor.trace().unwrap();
        assert_eq!(trace, 15.0);
    }
    #[test]
    fn test_trace_error_non_square() {
        let tensor = DenseND::<f64>::zeros(&[2, 3]);
        assert!(tensor.trace().is_err());
    }
    #[test]
    fn test_broadcast_to_simple() {
        let tensor = DenseND::<f64>::ones(&[3, 1]);
        let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(broadcast[&[i, j]], 1.0);
            }
        }
    }
    #[test]
    fn test_broadcast_to_scalar() {
        let tensor = DenseND::<f64>::from_vec(vec![2.0], &[1]).unwrap();
        let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(broadcast[&[i, j]], 2.0);
            }
        }
    }
    #[test]
    fn test_broadcast_to_3d() {
        let tensor = DenseND::<f64>::ones(&[1, 3, 1]);
        let broadcast = tensor.broadcast_to(&[2, 3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(broadcast[&[i, j, k]], 1.0);
                }
            }
        }
    }
    #[test]
    fn test_broadcast_to_rank_mismatch() {
        let tensor = DenseND::<f64>::ones(&[4]);
        let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[3, 4]);
    }
    #[test]
    fn test_broadcast_to_incompatible() {
        let tensor = DenseND::<f64>::ones(&[3, 2]);
        let result = tensor.broadcast_to(&[3, 4]);
        assert!(result.is_err());
    }
    #[test]
    fn test_broadcast_to_same_shape() {
        let tensor = DenseND::<f64>::ones(&[3, 4]);
        let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[3, 4]);
    }
    #[test]
    fn test_broadcast_add() {
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]).unwrap();
        let result = &a + &b;
        assert_eq!(result.shape(), &[3, 4]);
        assert_eq!(result[&[0, 0]], 11.0);
        assert_eq!(result[&[0, 1]], 21.0);
        assert_eq!(result[&[1, 0]], 12.0);
        assert_eq!(result[&[2, 3]], 43.0);
    }
    #[test]
    fn test_broadcast_sub() {
        let a =
            DenseND::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let result = &a - &b;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[&[0, 0]], 9.0);
        assert_eq!(result[&[0, 1]], 18.0);
        assert_eq!(result[&[0, 2]], 27.0);
        assert_eq!(result[&[1, 0]], 39.0);
    }
    #[test]
    fn test_broadcast_complex_3d() {
        let a = DenseND::<f64>::ones(&[2, 1, 4]);
        let b = DenseND::<f64>::ones(&[3, 1]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(result[&[i, j, k]], 2.0);
                }
            }
        }
    }
    #[test]
    fn test_broadcast_scalar_like() {
        let a = DenseND::<f64>::from_vec(vec![5.0], &[1]).unwrap();
        let b = DenseND::<f64>::ones(&[3, 4]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(result[&[i, j]], 6.0);
            }
        }
    }
    #[test]
    fn test_broadcast_matching_shapes() {
        let a = DenseND::<f64>::ones(&[3, 4]);
        let b = DenseND::<f64>::ones(&[3, 4]);
        let result = &a + &b;
        assert_eq!(result.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(result[&[i, j]], 2.0);
            }
        }
    }
}
