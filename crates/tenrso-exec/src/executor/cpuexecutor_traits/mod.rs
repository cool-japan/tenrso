//! # CpuExecutor - Trait Implementations
//!
//! This module contains trait implementations for `CpuExecutor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `TenrsoExecutor`
//!
//! ## Module Layout
//!
//! The original 2221-line monolithic `impl TenrsoExecutor<T> for CpuExecutor`
//! block has been split into per-category helper modules. Each trait method
//! below is a thin delegation to a free function in the matching submodule.
//!
//! - `contraction` - einsum
//! - `elementwise` - elem_op, binary_op, clip, modulo
//! - `reduction`   - reduce, softmax, log_softmax, layer_norm, batch_norm, argmax, argmin
//! - `shape`       - transpose, reshape, concatenate, split, tile, pad, flip, squeeze,
//!   unsqueeze, stack, repeat, roll
//! - `indexing`    - where_op, masked_select, gather, scatter, advanced_gather,
//!   advanced_scatter, fancy_index_mask
//! - `conv_pool`   - max_pool_1d/2d, avg_pool_1d/2d, conv1d/2d/3d
//! - `linalg`      - determinant, matrix_inverse, solve

mod contraction;
mod conv_pool;
mod elementwise;
mod indexing;
mod linalg;
mod reduction;
mod shape;

use super::functions::TenrsoExecutor;
use super::types::{BinaryOp, CpuExecutor, ElemOp, ReduceOp, ScatterMode};
use crate::hints::ExecHints;
use anyhow::Result;
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{Axis, TensorHandle};

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TenrsoExecutor<T> for CpuExecutor
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    fn einsum(
        &mut self,
        spec: &str,
        inputs: &[TensorHandle<T>],
        hints: &ExecHints,
    ) -> Result<TensorHandle<T>> {
        contraction::einsum(self, spec, inputs, hints)
    }

    fn elem_op(&mut self, op: ElemOp, x: &TensorHandle<T>) -> Result<TensorHandle<T>> {
        elementwise::elem_op(self, op, x)
    }

    fn binary_op(
        &mut self,
        op: BinaryOp,
        x: &TensorHandle<T>,
        y: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
        elementwise::binary_op(self, op, x, y)
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &TensorHandle<T>,
        axes: &[Axis],
    ) -> Result<TensorHandle<T>> {
        reduction::reduce(self, op, x, axes)
    }

    fn clip(&mut self, x: &TensorHandle<T>, min_val: T, max_val: T) -> Result<TensorHandle<T>> {
        elementwise::clip(self, x, min_val, max_val)
    }

    fn softmax(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
        reduction::softmax(self, x, axis)
    }

    fn log_softmax(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
        reduction::log_softmax(self, x, axis)
    }

    fn transpose(&mut self, x: &TensorHandle<T>, axes: &[Axis]) -> Result<TensorHandle<T>> {
        shape::transpose(self, x, axes)
    }

    fn reshape(&mut self, x: &TensorHandle<T>, new_shape: &[usize]) -> Result<TensorHandle<T>> {
        shape::reshape(self, x, new_shape)
    }

    fn concatenate(&mut self, tensors: &[TensorHandle<T>], axis: Axis) -> Result<TensorHandle<T>> {
        shape::concatenate(self, tensors, axis)
    }

    fn split(
        &mut self,
        x: &TensorHandle<T>,
        num_splits: usize,
        axis: Axis,
    ) -> Result<Vec<TensorHandle<T>>> {
        shape::split(self, x, num_splits, axis)
    }

    fn layer_norm(&mut self, x: &TensorHandle<T>, eps: T) -> Result<TensorHandle<T>> {
        reduction::layer_norm(self, x, eps)
    }

    fn batch_norm(&mut self, x: &TensorHandle<T>, eps: T) -> Result<TensorHandle<T>> {
        reduction::batch_norm(self, x, eps)
    }

    fn where_op(
        &mut self,
        condition: &TensorHandle<T>,
        x: &TensorHandle<T>,
        y: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
        indexing::where_op(self, condition, x, y)
    }

    fn masked_select(
        &mut self,
        x: &TensorHandle<T>,
        mask: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
        indexing::masked_select(self, x, mask)
    }

    fn modulo(&mut self, x: &TensorHandle<T>, divisor: T) -> Result<TensorHandle<T>> {
        elementwise::modulo(self, x, divisor)
    }

    fn remainder(&mut self, x: &TensorHandle<T>, divisor: T) -> Result<TensorHandle<T>> {
        elementwise::modulo(self, x, divisor)
    }

    fn max_pool_1d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: usize,
        stride: usize,
    ) -> Result<TensorHandle<T>> {
        conv_pool::max_pool_1d(self, x, kernel_size, stride)
    }

    fn avg_pool_1d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: usize,
        stride: usize,
    ) -> Result<TensorHandle<T>> {
        conv_pool::avg_pool_1d(self, x, kernel_size, stride)
    }

    fn max_pool_2d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<TensorHandle<T>> {
        conv_pool::max_pool_2d(self, x, kernel_size, stride)
    }

    fn avg_pool_2d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<TensorHandle<T>> {
        conv_pool::avg_pool_2d(self, x, kernel_size, stride)
    }

    fn conv1d(
        &mut self,
        x: &TensorHandle<T>,
        kernel: &TensorHandle<T>,
        bias: Option<&TensorHandle<T>>,
        stride: usize,
        padding: (usize, usize),
    ) -> Result<TensorHandle<T>> {
        conv_pool::conv1d(self, x, kernel, bias, stride, padding)
    }

    fn conv2d(
        &mut self,
        x: &TensorHandle<T>,
        kernel: &TensorHandle<T>,
        bias: Option<&TensorHandle<T>>,
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    ) -> Result<TensorHandle<T>> {
        conv_pool::conv2d(self, x, kernel, bias, stride, padding)
    }

    fn conv3d(
        &mut self,
        x: &TensorHandle<T>,
        kernel: &TensorHandle<T>,
        bias: Option<&TensorHandle<T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize, usize, usize, usize),
    ) -> Result<TensorHandle<T>> {
        conv_pool::conv3d(self, x, kernel, bias, stride, padding)
    }

    fn gather(
        &mut self,
        x: &TensorHandle<T>,
        axis: Axis,
        indices: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
        indexing::gather(self, x, axis, indices)
    }

    fn scatter(
        &mut self,
        shape: &[usize],
        axis: Axis,
        indices: &TensorHandle<T>,
        values: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
        indexing::scatter(self, shape, axis, indices, values)
    }

    fn determinant(&mut self, x: &TensorHandle<T>) -> Result<TensorHandle<T>> {
        linalg::determinant(self, x)
    }

    fn matrix_inverse(&mut self, x: &TensorHandle<T>) -> Result<TensorHandle<T>> {
        linalg::matrix_inverse(self, x)
    }

    fn solve(&mut self, a: &TensorHandle<T>, b: &TensorHandle<T>) -> Result<TensorHandle<T>> {
        linalg::solve(self, a, b)
    }

    fn advanced_gather(
        &mut self,
        x: &TensorHandle<T>,
        axis: Axis,
        indices: &TensorHandle<T>,
        allow_negative: bool,
    ) -> Result<TensorHandle<T>> {
        indexing::advanced_gather(self, x, axis, indices, allow_negative)
    }

    fn advanced_scatter(
        &mut self,
        shape: &[usize],
        axis: Axis,
        indices: &TensorHandle<T>,
        values: &TensorHandle<T>,
        mode: ScatterMode,
    ) -> Result<TensorHandle<T>> {
        indexing::advanced_scatter(self, shape, axis, indices, values, mode)
    }

    fn fancy_index_mask(
        &mut self,
        x: &TensorHandle<T>,
        mask: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
        indexing::fancy_index_mask(self, x, mask)
    }

    fn tile(&mut self, x: &TensorHandle<T>, reps: &[usize]) -> Result<TensorHandle<T>> {
        shape::tile(self, x, reps)
    }

    fn pad(
        &mut self,
        x: &TensorHandle<T>,
        pad_width: &[(usize, usize)],
        constant_value: T,
    ) -> Result<TensorHandle<T>> {
        shape::pad(self, x, pad_width, constant_value)
    }

    fn flip(&mut self, x: &TensorHandle<T>, axes: &[Axis]) -> Result<TensorHandle<T>> {
        shape::flip(self, x, axes)
    }

    fn squeeze(&mut self, x: &TensorHandle<T>, axes: Option<&[Axis]>) -> Result<TensorHandle<T>> {
        shape::squeeze(self, x, axes)
    }

    fn unsqueeze(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
        shape::unsqueeze(self, x, axis)
    }

    fn stack(&mut self, tensors: &[TensorHandle<T>], axis: Axis) -> Result<TensorHandle<T>> {
        shape::stack(self, tensors, axis)
    }

    fn repeat(
        &mut self,
        x: &TensorHandle<T>,
        repeats: usize,
        axis: Axis,
    ) -> Result<TensorHandle<T>> {
        shape::repeat(self, x, repeats, axis)
    }

    fn roll(&mut self, x: &TensorHandle<T>, shift: isize, axis: Axis) -> Result<TensorHandle<T>> {
        shape::roll(self, x, shift, axis)
    }

    fn argmax(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
        reduction::argmax(self, x, axis)
    }

    fn argmin(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
        reduction::argmin(self, x, axis)
    }
}
