//! Tests for TenrsoExecutor trait implementations
//!
//! This module contains comprehensive tests for all executor operations,
//! split into per-category submodules. Each submodule file contains bare
//! `#[test] fn ...` items; the whole tree is already `#[cfg(test)]`-gated
//! by the parent `executor/mod.rs` declaration.
//!
//! ## Layout
//!
//! - `einsum_tests`     - `einsum` trait method
//! - `elementwise_tests` - elem_op, binary_op, clip, modulo, remainder
//! - `reduction_tests`   - reduce, softmax, log_softmax, layer_norm, batch_norm,
//!   argmax, argmin
//! - `shape_tests`       - transpose, reshape, concatenate, split, tile, pad,
//!   flip, squeeze, unsqueeze, stack, repeat, roll
//! - `indexing_tests`    - where_op, masked_select, gather, scatter,
//!   advanced_gather, advanced_scatter, fancy_index_mask
//! - `conv_pool_tests`   - max_pool_1d/2d, avg_pool_1d/2d, conv1d/2d/3d
//! - `linalg_tests`      - determinant, matrix_inverse, solve
//! - `pool_tests`        - memory pool behaviour and automatic-pooling integration
//! - `thread_pool_tests` - `with_threads(n)` really bounds parallelism to `n`

mod conv_pool_tests;
mod einsum_tests;
mod elementwise_tests;
mod indexing_tests;
mod linalg_tests;
mod pool_tests;
mod reduction_tests;
mod shape_tests;
mod thread_pool_tests;
