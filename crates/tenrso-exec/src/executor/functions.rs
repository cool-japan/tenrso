//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{BinaryOp, ElemOp, ReduceOp, ScatterMode};
use crate::hints::ExecHints;
use anyhow::Result;
use scirs2_core::numeric::Num;
use tenrso_core::{Axis, TensorHandle};
/// Main executor trait for tensor operations
pub trait TenrsoExecutor<T>
where
    T: Clone + Num + 'static,
{
    /// Execute an einsum contraction
    fn einsum(
        &mut self,
        spec: &str,
        inputs: &[TensorHandle<T>],
        hints: &ExecHints,
    ) -> Result<TensorHandle<T>>;
    /// Apply unary element-wise operation
    fn elem_op(&mut self, op: ElemOp, x: &TensorHandle<T>) -> Result<TensorHandle<T>>;
    /// Apply binary element-wise operation
    fn binary_op(
        &mut self,
        op: BinaryOp,
        x: &TensorHandle<T>,
        y: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>>;
    /// Apply reduction operation
    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &TensorHandle<T>,
        axes: &[Axis],
    ) -> Result<TensorHandle<T>>;
    /// Clip tensor values to be within [min_val, max_val]
    fn clip(&mut self, x: &TensorHandle<T>, min_val: T, max_val: T) -> Result<TensorHandle<T>>;
    /// Softmax operation along specified axis
    /// Computes: exp(x) / sum(exp(x), axis)
    /// Uses numerically stable implementation: exp(x - max(x))
    fn softmax(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>>;
    /// Log-softmax operation along specified axis (numerically stable)
    /// Computes: x - log(sum(exp(x), axis))
    fn log_softmax(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>>;
    /// Transpose/permute tensor axes
    /// Reorders the dimensions according to the provided axes permutation
    fn transpose(&mut self, x: &TensorHandle<T>, axes: &[Axis]) -> Result<TensorHandle<T>>;
    /// Reshape tensor to new shape
    /// Total number of elements must remain constant
    fn reshape(&mut self, x: &TensorHandle<T>, new_shape: &[usize]) -> Result<TensorHandle<T>>;
    /// Concatenate tensors along specified axis
    fn concatenate(&mut self, tensors: &[TensorHandle<T>], axis: Axis) -> Result<TensorHandle<T>>;
    /// Split tensor along specified axis into chunks
    fn split(
        &mut self,
        x: &TensorHandle<T>,
        num_splits: usize,
        axis: Axis,
    ) -> Result<Vec<TensorHandle<T>>>;
    /// Layer normalization (fused operation)
    /// Normalizes over the last dimension: (x - mean) / sqrt(var + eps)
    fn layer_norm(&mut self, x: &TensorHandle<T>, eps: T) -> Result<TensorHandle<T>>;
    /// Batch normalization (fused operation)
    /// Normalizes over the batch dimension (first axis)
    fn batch_norm(&mut self, x: &TensorHandle<T>, eps: T) -> Result<TensorHandle<T>>;
    /// Conditional selection: where(condition, x, y)
    /// Returns x where condition is true (>0), y otherwise
    /// All tensors must have the same shape
    fn where_op(
        &mut self,
        condition: &TensorHandle<T>,
        x: &TensorHandle<T>,
        y: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>>;
    /// Masked selection: select values from x where mask is true (>0)
    /// Returns a 1D tensor containing selected values
    fn masked_select(
        &mut self,
        x: &TensorHandle<T>,
        mask: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>>;
    /// Element-wise modulo operation: x % divisor
    fn modulo(&mut self, x: &TensorHandle<T>, divisor: T) -> Result<TensorHandle<T>>;
    /// Element-wise remainder operation (same as modulo for positive numbers)
    fn remainder(&mut self, x: &TensorHandle<T>, divisor: T) -> Result<TensorHandle<T>>;
    /// Max pooling operation (1D)
    /// Applies max pooling with specified kernel size and stride
    fn max_pool_1d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: usize,
        stride: usize,
    ) -> Result<TensorHandle<T>>;
    /// Average pooling operation (1D)
    /// Applies average pooling with specified kernel size and stride
    fn avg_pool_1d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: usize,
        stride: usize,
    ) -> Result<TensorHandle<T>>;
    /// Max pooling operation (2D)
    /// Applies max pooling with specified kernel size and stride
    fn max_pool_2d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<TensorHandle<T>>;
    /// Average pooling operation (2D)
    /// Applies average pooling with specified kernel size and stride
    fn avg_pool_2d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<TensorHandle<T>>;
    /// 1D Convolution operation
    /// Applies 1D convolution: output\[i\] = sum_j(input\[i+j\] * kernel\[j\])
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape \[batch, in_channels, length\]
    /// * `kernel` - Convolution kernel of shape \[out_channels, in_channels, kernel_size\]
    /// * `bias` - Optional bias of shape \[out_channels\]
    /// * `stride` - Stride for the convolution
    /// * `padding` - Padding to apply (left, right)
    fn conv1d(
        &mut self,
        x: &TensorHandle<T>,
        kernel: &TensorHandle<T>,
        bias: Option<&TensorHandle<T>>,
        stride: usize,
        padding: (usize, usize),
    ) -> Result<TensorHandle<T>>;
    /// 2D Convolution operation
    /// Applies 2D convolution for image processing
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape \[batch, in_channels, height, width\]
    /// * `kernel` - Convolution kernel of shape \[out_channels, in_channels, kernel_h, kernel_w\]
    /// * `bias` - Optional bias of shape \[out_channels\]
    /// * `stride` - Stride for the convolution (stride_h, stride_w)
    /// * `padding` - Padding to apply (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
    fn conv2d(
        &mut self,
        x: &TensorHandle<T>,
        kernel: &TensorHandle<T>,
        bias: Option<&TensorHandle<T>>,
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    ) -> Result<TensorHandle<T>>;
    /// 3D Convolution operation
    /// Applies 3D convolution for volumetric data (video, medical imaging, etc.)
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape \[batch, in_channels, depth, height, width\]
    /// * `kernel` - Convolution kernel of shape \[out_channels, in_channels, kernel_d, kernel_h, kernel_w\]
    /// * `bias` - Optional bias of shape \[out_channels\]
    /// * `stride` - Stride for the convolution (stride_d, stride_h, stride_w)
    /// * `padding` - Padding to apply (pad_d_front, pad_d_back, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
    fn conv3d(
        &mut self,
        x: &TensorHandle<T>,
        kernel: &TensorHandle<T>,
        bias: Option<&TensorHandle<T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize, usize, usize, usize),
    ) -> Result<TensorHandle<T>>;
    /// Gather operation - selects values along an axis using indices
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `axis` - Axis along which to gather
    /// * `indices` - Integer indices to gather (as Float tensor, will be cast to usize)
    fn gather(
        &mut self,
        x: &TensorHandle<T>,
        axis: Axis,
        indices: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>>;
    /// Scatter operation - writes values to an output tensor using indices
    ///
    /// # Arguments
    /// * `shape` - Shape of the output tensor
    /// * `axis` - Axis along which to scatter
    /// * `indices` - Integer indices where to write (as Float tensor, will be cast to usize)
    /// * `values` - Values to write at the indices
    fn scatter(
        &mut self,
        shape: &[usize],
        axis: Axis,
        indices: &TensorHandle<T>,
        values: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>>;
    /// Matrix determinant operation
    /// Computes the determinant of a square matrix or batch of matrices
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [..., N, N] (last two dimensions must be square)
    ///
    /// # Returns
    /// Tensor of shape [...] containing determinants
    fn determinant(&mut self, x: &TensorHandle<T>) -> Result<TensorHandle<T>>;
    /// Matrix inverse operation
    /// Computes the inverse of a square matrix or batch of matrices
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [..., N, N] (last two dimensions must be square)
    ///
    /// # Returns
    /// Tensor of shape [..., N, N] containing matrix inverses
    fn matrix_inverse(&mut self, x: &TensorHandle<T>) -> Result<TensorHandle<T>>;
    /// Solve linear system Ax = b
    /// Solves the linear system of equations for x
    ///
    /// # Arguments
    /// * `a` - Coefficient matrix of shape [..., N, N]
    /// * `b` - Right-hand side of shape [..., N] or [..., N, K]
    ///
    /// # Returns
    /// Solution tensor x of the same shape as b
    fn solve(&mut self, a: &TensorHandle<T>, b: &TensorHandle<T>) -> Result<TensorHandle<T>>;

    /// Advanced gather operation with negative indices support
    /// Gathers values from `x` along the specified `axis` using `indices`.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `axis` - Axis along which to gather
    /// * `indices` - Integer indices (as Float tensor)
    /// * `allow_negative` - Whether to allow Python-style negative indices
    ///
    /// # Returns
    /// Tensor with gathered values
    fn advanced_gather(
        &mut self,
        x: &TensorHandle<T>,
        axis: Axis,
        indices: &TensorHandle<T>,
        allow_negative: bool,
    ) -> Result<TensorHandle<T>>;

    /// Advanced scatter operation with accumulation modes
    /// Scatters `values` into an output tensor along the specified `axis` using `indices`.
    ///
    /// # Arguments
    /// * `shape` - Shape of the output tensor
    /// * `axis` - Axis along which to scatter
    /// * `indices` - Integer indices where to write
    /// * `values` - Values to scatter
    /// * `mode` - Scatter mode (Replace, Add, Max, Min)
    ///
    /// # Returns
    /// Output tensor with scattered values
    fn advanced_scatter(
        &mut self,
        shape: &[usize],
        axis: Axis,
        indices: &TensorHandle<T>,
        values: &TensorHandle<T>,
        mode: ScatterMode,
    ) -> Result<TensorHandle<T>>;

    /// Fancy indexing with boolean masks
    /// Selects elements from `x` where `mask` is true (> 0).
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `mask` - Boolean mask tensor (same shape as x)
    ///
    /// # Returns
    /// 1D tensor containing selected elements
    fn fancy_index_mask(
        &mut self,
        x: &TensorHandle<T>,
        mask: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>>;

    /// Tile operation - repeats a tensor along each dimension
    /// Constructs an array by repeating `x` the number of times given by `reps`.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `reps` - Number of repetitions along each dimension
    ///
    /// # Returns
    /// Tiled tensor
    fn tile(&mut self, x: &TensorHandle<T>, reps: &[usize]) -> Result<TensorHandle<T>>;

    /// Pad operation - pads a tensor with a constant value
    /// Pads the input tensor with the specified value.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `pad_width` - Number of values padded to edges of each axis (before, after) for each dimension
    /// * `constant_value` - The value to set the padded values
    ///
    /// # Returns
    /// Padded tensor
    fn pad(
        &mut self,
        x: &TensorHandle<T>,
        pad_width: &[(usize, usize)],
        constant_value: T,
    ) -> Result<TensorHandle<T>>;

    /// Flip operation - reverses the order of elements along specified axes
    /// Reverses the order of elements in an array along the given axes.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `axes` - Axes along which to flip
    ///
    /// # Returns
    /// Flipped tensor
    fn flip(&mut self, x: &TensorHandle<T>, axes: &[Axis]) -> Result<TensorHandle<T>>;

    /// Squeeze operation - removes dimensions of size 1
    /// Removes single-dimensional entries from the shape of an array.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `axes` - Optional axes to squeeze. If None, all axes of size 1 are removed
    ///
    /// # Returns
    /// Squeezed tensor
    fn squeeze(&mut self, x: &TensorHandle<T>, axes: Option<&[Axis]>) -> Result<TensorHandle<T>>;

    /// Unsqueeze/expand_dims operation - adds a dimension of size 1
    /// Expands the shape of an array by inserting a new axis.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `axis` - Position where new axis is to be inserted
    ///
    /// # Returns
    /// Tensor with expanded dimensions
    fn unsqueeze(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>>;

    /// Stack operation - joins tensors along a new axis
    /// Joins a sequence of tensors along a new axis.
    ///
    /// # Arguments
    /// * `tensors` - Sequence of tensors to stack
    /// * `axis` - Axis along which to stack
    ///
    /// # Returns
    /// Stacked tensor
    fn stack(&mut self, tensors: &[TensorHandle<T>], axis: Axis) -> Result<TensorHandle<T>>;

    /// Repeat operation - repeats elements of an array
    /// Repeat elements of an array along each axis.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `repeats` - Number of repetitions for each element along each axis
    /// * `axis` - Axis along which to repeat values
    ///
    /// # Returns
    /// Repeated tensor
    fn repeat(
        &mut self,
        x: &TensorHandle<T>,
        repeats: usize,
        axis: Axis,
    ) -> Result<TensorHandle<T>>;

    /// Roll operation - rolls array elements along an axis
    /// Shifts elements along an axis, wrapping around at the boundaries.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `shift` - Number of places to shift (positive or negative)
    /// * `axis` - Axis along which to roll
    ///
    /// # Returns
    /// Rolled tensor
    fn roll(&mut self, x: &TensorHandle<T>, shift: isize, axis: Axis) -> Result<TensorHandle<T>>;

    /// ArgMax operation - indices of maximum values along an axis
    /// Returns the indices of the maximum values along an axis.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `axis` - Axis along which to find argmax
    ///
    /// # Returns
    /// Tensor containing indices of maximum values
    fn argmax(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>>;

    /// ArgMin operation - indices of minimum values along an axis
    /// Returns the indices of the minimum values along an axis.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `axis` - Axis along which to find argmin
    ///
    /// # Returns
    /// Tensor containing indices of minimum values
    fn argmin(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>>;
}
