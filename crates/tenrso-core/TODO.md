# tenrso-core TODO

> **Milestone:** M0 → M1
> **Status:** Foundation types complete, dense implementation pending

---

## M0: Foundation Types - ✅ COMPLETE

- [x] Type aliases (`Axis`, `Rank`, `Shape`)
- [x] `AxisMeta` struct with name and size
- [x] `TensorRepr` enum (Dense/Sparse/LowRank placeholders)
- [x] `TensorHandle` unified wrapper
- [x] Basic accessor methods (`rank()`, `shape()`)
- [x] Module structure (types, dense, ops)
- [x] CI passing (fmt, clippy, test)

---

## M1: Dense Tensor Implementation - ✅ COMPLETE

### Core Dense Tensor - ✅ COMPLETE

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
  - [ ] `random_leverage_scores(tensor, rank) -> Self` (for CP init) - ⏳ Future M2

### Views and Slicing - ✅ LARGELY COMPLETE

- [x] Implement views via ndarray's ArrayView/ArrayViewMut
  - [x] `view(&self) -> ArrayView<T, IxDyn>`
  - [x] `view_mut(&mut self) -> ArrayViewMut<T, IxDyn>`
- [ ] Advanced slicing operations - ⏳ Future (can use ndarray's slicing directly)
  - [ ] Custom `slice` API wrapper
  - [ ] Support for ranges (`0..10`, `..`, `5..`)
  - [ ] Support for stepped ranges (`0..10:2`)

- [x] Indexing operations
  - [x] Index trait (`tensor[&[i,j,k]]`)
  - [x] IndexMut trait (`tensor[&[i,j,k]] = value`)
  - [x] Bounds checking (via ndarray)
  - [ ] Panic-free indexing variants (`get`, `get_mut`) - ⏳ Future

### Unfold/Fold Operations - ✅ COMPLETE

- [x] Mode-n unfold (matricization)
  - [x] `unfold(&self, mode: Axis) -> Array2<T>`
  - [x] Efficient memory layout (minimize copies)
  - [x] Support for all modes (0..rank)
  - [x] Correctness tests ✅

- [x] Mode-n fold (tensorization)
  - [x] `fold(matrix: &Array2<T>, shape: &[usize], mode: Axis) -> Result<Self>`
  - [x] Inverse of unfold operation
  - [x] Shape validation
  - [x] Roundtrip tests (unfold → fold == identity) ✅

### Shape Operations - ✅ COMPLETE

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

- [ ] Squeeze/Unsqueeze - ⏳ Future M2+
  - [ ] `squeeze(&self, axis: Option<Axis>) -> Self`
  - [ ] `unsqueeze(&self, axis: Axis) -> Self`
  - [ ] Remove/add singleton dimensions

### Integration with TensorHandle - ✅ COMPLETE

- [x] `TensorHandle::from_dense(data: DenseND<T>, axes: Vec<AxisMeta>) -> Self`
- [x] `TensorHandle::from_dense_auto(data: DenseND<T>) -> Self` (auto-naming)
- [x] `TensorHandle::as_dense(&self) -> Option<&DenseND<T>>`
- [x] `TensorHandle::as_dense_mut(&mut self) -> Option<&mut DenseND<T>>`
- [x] Axis metadata validation (shape matches axes)

### Testing - ✅ LARGELY COMPLETE

- [x] Unit tests - ✅ 19 tests passing
  - [x] Tensor creation methods (zeros, ones, from_elem, from_vec)
  - [x] View operations (view, view_mut)
  - [x] Unfold/fold correctness ✅
  - [x] Reshape validation ✅
  - [x] Permute correctness ✅
  - [x] Random initialization (uniform, normal)
  - [x] Unfold-fold roundtrip ✅

- [x] Property tests (proptest) - ✅ 17 tests passing
  - [x] Property-based testing infrastructure in place
  - [x] Randomized test cases
  - [ ] Additional edge case coverage - ⏳ Future

- [ ] Integration tests - ⏳ Next step
  - [x] Cross-crate usage with tenrso-kernels (via existing integration tests)
  - [x] Axis metadata consistency (validated in TensorHandle)
  - [ ] Memory layout verification - ⏳ Future benchmark

### Documentation - ✅ COMPLETE (2025-11-04)

- [x] Rustdoc for all public types ✅
- [x] Examples in docstrings ✅
- [x] Module-level documentation (lib.rs, types.rs, dense.rs, ops.rs) ✅
- [x] `examples/basic_tensor.rs` ✅
- [x] `examples/views.rs` ✅
- [x] `examples/unfold_fold.rs` ✅
- [ ] README updates with usage examples - ⏳ Optional (covered in module docs)

### Performance - ✅ COMPLETE (2025-11-04)

- [x] Benchmark tensor creation (zeros, ones, from_elem, from_vec, random) ✅
- [x] Benchmark view/slice operations ✅
- [x] Benchmark unfold/fold (including CP-ALS and Tucker use cases) ✅
- [x] Benchmark reshape/permute (including common patterns) ✅
- [x] Set up criterion infrastructure ✅
- [x] Comprehensive benchmark suites created ✅
- [ ] Compare against ndarray baseline - ⏳ Future optimization
- [ ] Profile memory allocations - ⏳ Future optimization

---

## M2+: Future Enhancements - ⏳ PLANNED

### Advanced Features

- [ ] Broadcasting support
- [ ] Fancy indexing (boolean masks, index arrays)
- [ ] Concatenate/stack operations
- [ ] Split/chunk operations
- [ ] Diagonal extraction
- [ ] Trace operations

### Memory Management

- [ ] Custom allocators
- [ ] Memory pooling
- [ ] Lazy allocation
- [ ] Memory-mapped tensors (via tenrso-ooc)

### Type System

- [ ] Const generic shapes (when Rust supports)
- [ ] Compile-time shape checking (where possible)
- [ ] Type-level rank tracking

### Serialization

- [ ] Serde support for DenseND
- [ ] Binary format (efficient)
- [ ] JSON format (human-readable)
- [ ] Arrow IPC format (via tenrso-ooc)

---

## Dependencies

### Current

- scirs2-core (ndarray_ext, random) - ✅ In use
- smallvec - ✅ In use
- num-traits - ✅ In use
- anyhow - ✅ In use
- thiserror - ✅ In use

### Future

- proptest (property testing) - ⏳ Needed for M1
- criterion (benchmarking) - ⏳ Needed for M1

---

## Blockers

- None currently

---

## Notes

- **Memory layout:** Default to C-contiguous (row-major) for cache efficiency
- **Safety:** All indexing must be bounds-checked; no UB
- **SciRS2 usage:** Always use `scirs2_core::ndarray_ext`, never raw `ndarray`
- **Views:** Implement using ndarray's view mechanism (ArrayView/ArrayViewMut)
- **Unfold:** Follow standard tensor unfolding conventions (mode-n fibers become rows)

---

## Questions for Review

1. Should we support both C-contiguous and Fortran-contiguous layouts?
2. What should be the default initialization for `DenseND::default()`?
3. Should `unfold` return owned Array2 or a view?
4. How to handle non-contiguous reshape (copy vs error)?

---

## Notes

- **Status:** ✅ **M1 COMPLETE + Documentation + Performance + Enhanced Safety COMPLETE!**
- **Test Results:** 70 tests passing (22 unit/property + 48 doc tests)
- **Key Features Ready:**
  - ✅ DenseND<T> with all creation methods
  - ✅ Unfold/fold operations (critical for decompositions)
  - ✅ Reshape and permute
  - ✅ Random initialization
  - ✅ TensorHandle integration
  - ✅ **Panic-free indexing** (get/get_mut)
  - ✅ **Utility methods** (is_contiguous, is_square, shape_vec)
  - ✅ Comprehensive documentation with examples
  - ✅ Complete benchmark suite
  - ✅ **Enhanced property-based testing**

- **Documentation Deliverables:**
  - ✅ Enhanced module-level documentation in lib.rs
  - ✅ Comprehensive rustdoc for all public types in types.rs
  - ✅ Mathematical background documentation in ops.rs
  - ✅ 3 example programs (basic_tensor, views, unfold_fold)
  - ✅ 48 doc tests passing (demonstrating all APIs)

- **Performance Deliverables:**
  - ✅ Criterion benchmarking infrastructure
  - ✅ `benches/tensor_creation.rs` - Creation methods, indexing
  - ✅ `benches/unfold_fold.rs` - Matricization for decompositions
  - ✅ `benches/reshape_permute.rs` - Shape operations, views
  - ✅ All benchmarks compile and run successfully

- **Safety & Robustness Enhancements:**
  - ✅ Panic-free indexing methods (get, get_mut) for safer access
  - ✅ Utility methods for common checks (is_contiguous, is_square, shape_vec)
  - ✅ Enhanced property tests (12 property-based tests total)
  - ✅ Edge case testing (single element, singleton dimensions)
  - ✅ 4D tensor support in property tests
  - ✅ Comprehensive bounds checking and error handling

- **Dependencies Satisfied:** tenrso-decomp can now begin implementation!
- **Next Priority:** Start CP-ALS implementation in tenrso-decomp (M2)
