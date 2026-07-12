# tenrso-core TODO

> **Milestone:** M0 → M1 → M2+ (Complete)
> **Version:** 0.1.0
> **Status:** 0.1.0 — 178 unit tests passing (100%)
> **Last Updated:** 2026-04-14

---

## M0: Foundation Types - COMPLETE

- [x] Type aliases (`Axis`, `Rank`, `Shape`)
- [x] `AxisMeta` struct with name and size
- [x] `TensorRepr` enum (Dense/Sparse/LowRank placeholders)
- [x] `TensorHandle` unified wrapper
- [x] Basic accessor methods (`rank()`, `shape()`)
- [x] Module structure (types, dense, ops)
- [x] CI passing (fmt, clippy, test)

---

## M1: Dense Tensor Implementation - COMPLETE

### Core Dense Tensor - COMPLETE

- [x] Implement `DenseND<T>` structure
  - [x] Wrap `scirs2_core::ndarray_ext::Array<T, IxDyn>`
  - [x] Track strides and offset (via ndarray)
  - [x] Support arbitrary rank (1D to N-D)
  - [x] Generic over scalar types (via num-traits)

- [x] Tensor creation methods
  - [x] `zeros(shape: &[usize]) -> Self`
  - [x] `ones(shape: &[usize]) -> Self`
  - [x] `from_elem(shape, value) -> Self`
  - [x] `from_array(array: Array<T, IxDyn>) -> Self`
  - [x] `from_vec(vec, shape) -> Result<Self>`

- [x] Initialization strategies
  - [x] `random_uniform(shape, low, high) -> Self`
  - [x] `random_normal(shape, mean, std) -> Self`
  - [x] `from_leverage_scores(tensor, mode, rank) -> Self` (CP init, feature `linalg`) — Complete (2026-06-10)

### Views and Slicing - COMPLETE

- [x] Implement views via ndarray's ArrayView/ArrayViewMut
  - [x] `view(&self) -> ArrayView<T, IxDyn>`
  - [x] `view_mut(&mut self) -> ArrayViewMut<T, IxDyn>`
- [x] Panic-free indexing (`get`, `get_mut`)
- [x] Index trait (`tensor[&[i,j,k]]`)
- [x] IndexMut trait (`tensor[&[i,j,k]] = value`)
- [x] Bounds checking (via ndarray)

### Unfold/Fold Operations - COMPLETE

- [x] Mode-n unfold (matricization)
  - [x] `unfold(&self, mode: Axis) -> Array2<T>`
  - [x] Efficient memory layout (minimize copies)
  - [x] Support for all modes (0..rank)
  - [x] Correctness tests

- [x] Mode-n fold (tensorization)
  - [x] `fold(matrix: &Array2<T>, shape: &[usize], mode: Axis) -> Result<Self>`
  - [x] Inverse of unfold operation
  - [x] Shape validation
  - [x] Roundtrip tests (unfold -> fold == identity)

### Shape Operations - COMPLETE

- [x] Reshape
  - [x] `reshape(&self, new_shape: &[usize]) -> Result<Self>`
  - [x] Validate total size matches
  - [x] Zero-copy when possible (contiguous)
  - [x] Copy when necessary (non-contiguous)

- [x] Permute (transpose)
  - [x] `permute(&self, axes: &[Axis]) -> Self`
  - [x] Arbitrary axis permutation
  - [x] Validate axes are valid permutation
  - [x] Efficient implementation (stride manipulation via ndarray)

- [x] Flatten/ravel
  - [x] `flatten() -> Self` - Flatten to 1D
  - [x] `ravel() -> Self` - Alias for flatten

- [x] Axis manipulation
  - [x] `swapaxes(ax1, ax2) -> Self`
  - [x] `moveaxis(src, dst) -> Self`

- [x] Squeeze/Unsqueeze - COMPLETE (M2+)
  - [x] `squeeze(&self) -> Self` (drops every size-1 axis; collapses to rank-0 when all axes are 1)
  - [x] `squeeze_axis(&self, axis: Axis) -> Result<Self>`
  - [x] `squeeze_axes(&self, axes: &[Axis]) -> Result<Self>`
  - [x] `unsqueeze(&self, axis: Axis) -> Result<Self>`
  - [x] `unsqueeze_axes(&self, axes: &[Axis]) -> Result<Self>` (indices interpreted against final shape)

### Integration with TensorHandle - COMPLETE

- [x] `TensorHandle::from_dense(data: DenseND<T>, axes: Vec<AxisMeta>) -> Self`
- [x] `TensorHandle::from_dense_auto(data: DenseND<T>) -> Self` (auto-naming)
- [x] `TensorHandle::as_dense(&self) -> Option<&DenseND<T>>`
- [x] `TensorHandle::as_dense_mut(&mut self) -> Option<&mut DenseND<T>>`
- [x] Axis metadata validation (shape matches axes)

### Testing - COMPLETE

- [x] Unit tests - passing
  - [x] Tensor creation methods (zeros, ones, from_elem, from_vec)
  - [x] View operations (view, view_mut)
  - [x] Unfold/fold correctness
  - [x] Reshape validation
  - [x] Permute correctness
  - [x] Random initialization (uniform, normal)
  - [x] Unfold-fold roundtrip

- [x] Property tests (proptest) - passing
  - [x] Property-based testing infrastructure
  - [x] Randomized test cases

### Documentation - COMPLETE

- [x] Rustdoc for all public types
- [x] Examples in docstrings
- [x] Module-level documentation (lib.rs, types.rs, dense.rs, ops.rs)
- [x] `examples/basic_tensor.rs`
- [x] `examples/views.rs`
- [x] `examples/unfold_fold.rs`

### Performance - COMPLETE

- [x] Benchmark tensor creation (zeros, ones, from_elem, from_vec, random)
- [x] Benchmark view/slice operations
- [x] Benchmark unfold/fold (including CP-ALS and Tucker use cases)
- [x] Benchmark reshape/permute (including common patterns)
- [x] Criterion infrastructure set up
- [x] Compare against ndarray baseline — `benches/ndarray_baseline.rs` (2026-07-11). Pairs
      `dense_nd/<op>` vs `ndarray/<op>` at 64³ and 256³. Wrapper overhead is ~0 for
      construction/reshape/permute/hadamard/mul_scalar/view/index (same underlying call).
      Found two genuine perf bugs: `Add`/`Sub` clone both operands even when shapes already
      match (3x the needed allocation; 3.6x–33x slower in practice) and `unfold` performs two
      full copies via generic `permute`+`reshape` composition where one suffices (2x bytes
      moved; 1.3x–1.9x slower). Also confirmed `reshape`/`permute` are never actually
      zero-copy despite docs (see allocation profiling below) — filed as follow-ups, not
      fixed here (out of scope for this benchmark task).
- [x] Profile memory allocations — `benches/alloc_profile.rs` (2026-07-11). Custom counting
      `#[global_allocator]` wrapping `System`; reports allocation count + bytes per op.
      Confirms: `reshape`/`permute` always allocate a full-tensor copy (never truly
      zero-copy, unlike the raw consuming path which is 0 allocations); `Add` does 4 allocs
      (~3x tensor size) vs raw's 1; `unfold` does 5 allocs (~2x tensor bytes) vs a hand-fused
      single-pass raw unfold's 4 (~1x); `sum`/`mean`/`view`/`index_get` allocate nothing in
      either implementation.

---

## M2+: Advanced Features - COMPLETE

### Fancy Indexing - COMPLETE

- [x] Boolean mask selection (`select_mask`)
- [x] Index array selection along axis (`select_indices`)
- [x] Filter by predicate (`filter`)

### Comparison and Utility Operations - COMPLETE

- [x] Element-wise comparison operations
  - [x] Greater than (`gt`), less than (`lt`)
  - [x] Greater/less than or equal (`gte`, `lte`)
  - [x] Element-wise equality (`eq_elementwise`)
- [x] Masking operations
  - [x] Threshold masking (`mask_gt`, `mask_lt`)
  - [x] Conditional counting (`count_if`)
- [x] Value clipping (`clip`)
- [x] Mapping operations (`map`, `map_inplace`)
- [x] Fill operation (`fill`)
- [x] Full tensor creation (`full`)

### Array Combining - COMPLETE

- [x] Concatenate along axis (`concatenate`)
- [x] Stack along new axis (`stack`)
- [x] Split into n parts (`split`)
- [x] Chunk into fixed-size pieces (`chunk`)
- [x] Diagonal extraction (`diagonal`)
- [x] Trace (`trace`)

### Advanced Statistics - COMPLETE

- [x] Variance (`variance`, `variance_axis`)
- [x] Standard deviation (`std`, `std_axis`)
- [x] Median (`median`, `median_axis`)
- [x] Quantile (`quantile`, `quantile_axis`)
- [x] Percentile (`percentile`, `percentile_axis`)
- [x] argmax, argmin
- [x] all, any (logical reductions)
- [x] roll, flip, tile (array manipulation)

### Linear Algebra - COMPLETE

- [x] Matrix multiplication (`matmul`)
- [x] Determinant (`det`)
- [x] Matrix inverse (`inv`)
- [x] Linear solve (`solve`)
- [x] 37+ elementwise algebra operations

### Activation Functions - COMPLETE

- [x] relu, sigmoid, tanh, gelu, and more

### Convolution and FFT - COMPLETE

- [x] Convolution operations (im2col + GEMM, 5-10x faster)
- [x] FFT operations (20+ functions)

### Advanced Creation Methods - COMPLETE

- [x] `eye(n)` - Identity matrix
- [x] `arange(start, stop, step)` - Evenly spaced values
- [x] `linspace(start, stop, num)` - Linearly spaced values

### Serialization - COMPLETE

- [x] Serde support for `DenseND<T>` (feature-gated)
- [x] Serde support for `TensorHandle<T>` (feature-gated)
- [x] Serde support for `TensorRepr<T>` (feature-gated)
- [x] Serde support for `AxisMeta` (feature-gated)
- [x] Binary format (efficient) — COMPLETE (oxicode, `binary` feature, `save_binary`/`load_binary`)
- [x] JSON format (human-readable) - COMPLETE (save_json, load_json, to_json_string, from_json_str, 2026-06-03)
- [ ] Arrow IPC format - Deferred to tenrso-ooc

### Broadcasting - COMPLETE

- [x] `broadcast_to(shape)` - Broadcast tensor to new shape
- [x] Broadcasting in arithmetic operations (Add, Sub)

### Indexing Safety - COMPLETE

- [x] Panic-free `get`/`get_mut` variants
- [x] Bounds checking for all indexing paths
- [x] Multiple safety levels

---

## M3+: Future Enhancements - PLANNED

### Memory Management

- [ ] Custom allocators - Future
- [ ] Memory pooling - Future
- [ ] Lazy allocation - Future
- [ ] Memory-mapped tensors (via tenrso-ooc) - Deferred to tenrso-ooc

### Type System

- [ ] Const generic shapes (when Rust supports) - Blocked by Rust language
- [ ] Compile-time shape checking (where possible) - Future
- [ ] Type-level rank tracking - Future

### Interop

- [x] Sparse interop via boolean masks — Complete (2026-06-10, `to_bool_mask`, `apply_bool_mask`, `sparse_indices` on DenseND)
- [x] `from_leverage_scores(tensor, mode, rank) -> Self` (CP init, feature `linalg`) — Complete (2026-06-10)
- [x] Squeeze/Unsqueeze shape ops - COMPLETE (see `shape_ops.rs`)

---

## Dependencies

### Current (All In Use)

- scirs2-core (ndarray_ext, random) - In use
- scirs2-linalg (SVD, QR, GEMM) - In use
- smallvec - In use
- num-traits - In use
- anyhow - In use
- thiserror - In use
- proptest (dev) - In use
- criterion (dev) - In use

---

## Refactoring History

### Dense Module Split (2025-11-25)

Refactored `dense.rs` (3131 lines) into 4 modules using splitrs:
- `dense/types.rs` - DenseND struct and core methods
- `dense/densend_traits.rs` - Trait implementations
- `dense/functions.rs` - Helper functions and tests
- `dense/mod.rs` - Module organization

### Advanced Refactoring (2025-11-26)

Split `types.rs` (2,549 lines) into 10 functional modules by operation type:
- `types.rs` - Core struct and basic methods (291 lines)
- `creation.rs` - Initialization methods (121 lines)
- `shape_ops.rs` - Shape manipulation (436 lines)
- `combining.rs` - Concatenate/stack/split/chunk (277 lines)
- `indexing.rs` - Element access and selection (215 lines)
- `algebra.rs` - Linear algebra operations (262 lines)
- `statistics.rs` - Statistical reductions (348 lines)
- `elementwise.rs` - Element-wise ops (232 lines)
- `comparison.rs` - Comparison and logical ops (212 lines)
- `manipulation.rs` - Array manipulation (175 lines)

All modules under 500 lines, clear separation of concerns.

---

## RC.1 Status (2026-03-06)

- **Tests:** 138 passing (100%)
- **Warnings:** 0 (clippy strict mode)
- **Unsafe blocks:** 0
- **todo!()/unimplemented!():** 0
- **Documentation coverage:** 100% of public APIs
- **SCIRS2 compliance:** Full (no direct ndarray/rand imports)
- **Source:** 20 files, ~16K lines
- **Methods:** 189+ tensor operations implemented

## 0.1.0 Status (2026-04-14)

- **Tests:** 178 unit + 18 integration + 240 doctests (100%)
- **Warnings:** 0 (clippy strict mode)
- **New in this update:** `squeeze` now collapses all-size-1 inputs to rank-0;
  added `squeeze_axes`, `unsqueeze_axes`. `squeeze_axis` / `unsqueeze`
  annotated `#[inline]`. 27 new unit tests in `dense::shape_ops::tests`.
