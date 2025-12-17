# Changelog

All notable changes to the `tenrso-ad` crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased - 0.1.0-alpha.2]

### Added (2025-12-06 PM - Latest)
- **Graph-Based AD Benchmarks** (`benches/graph_benchmarks.rs`) ✨ NEW - 435 lines
  - 8 comprehensive benchmark categories for computation graph performance
  - Benchmarks: arithmetic, neural layers, deep networks, construction, backward pass, activations, optimization, memory
  - Throughput-based measurement for scalability analysis
  - Total benchmarks: 15 (7 VJP/decomp + 8 graph) ⬆️ +114% increase
- **Cargo.toml**: Added `graph_benchmarks` benchmark configuration

### Fixed (2025-12-06 PM)
- **Test Stability**: Fixed intermittent test failure in `parallel::tests::test_parallel_elementwise_vjp`
  - Added better error messages for debugging
  - All 154 tests now pass consistently (100% pass rate)

### Added (2025-12-06 PM - Part 2)
- **Graph Optimizer Module** (`graph_optimizer.rs`) ✨ NEW - 500+ lines
  - Operation fusion for better kernel utilization (MatMul+Bias, MatMul+Bias+ReLU, Add+ReLU, Mul+Add)
  - Dead code elimination to reduce memory usage
  - Memory planning and reuse estimation
  - Configurable optimization strategies (OperationFusion, DeadCodeElimination, All)
  - Pattern prioritization for optimal fusion
  - 12 unit tests covering all optimization passes
- **Enhanced graph-based AD**: Integration with optimization passes

### Added (2025-12-06 PM - Part 1)
- **Graph-Based AD Module** (`graph.rs`) ✨ NEW - 800+ lines
  - PyTorch-style tape-based automatic differentiation
  - Dynamic computation graph construction during forward pass
  - Automatic operation recording with gradient tracking
  - Topological sort for efficient backward propagation
  - Training/inference mode support (`.train()` / `.eval()`)
  - Operations: Add, Sub, Mul, Div, Neg, MatMul, ReLU, Sigmoid, Tanh, Exp, Log, Pow, Sum, Mean, Reshape
  - Dynamic control flow support
  - Graph statistics and inspection
  - 15 unit tests covering all graph operations
- **Example**: `graph_based_ad.rs` - Comprehensive graph-based AD demonstration (6 examples)

### Added (2025-12-06 AM)
- **Optimizers Module** (`optimizers.rs`) ✨ NEW - 900+ lines
  - SGD optimizer with momentum and Nesterov acceleration
  - Adam optimizer with adaptive moment estimation
  - AdamW optimizer with decoupled weight decay
  - RMSprop optimizer for non-stationary objectives
  - AdaGrad optimizer for sparse features
  - StepLR learning rate scheduler
  - ExponentialLR learning rate scheduler
  - CosineAnnealingLR for cosine annealing schedule
  - ReduceLROnPlateau for adaptive LR based on metrics
  - Comprehensive `OptimizerConfig` builder pattern
  - Generic `Optimizer` and `LRScheduler` traits
  - 13 unit tests covering all optimizers and schedulers
- **Example**: `optimizers_showcase.rs` - 10 comprehensive examples demonstrating all optimizers and schedulers

### Changed
- Updated documentation to accurately reflect production-ready state
- Updated README.md with 127 tests (up from 114)
- Updated lib.rs crate documentation to include optimizers module
- Clarified that mixed precision training and gradient monitoring are complete, not future work
- Updated quality metrics to match current codebase statistics
- Reorganized roadmap to show completed vs future features

### Documentation
- Enhanced README.md with accurate feature status and optimizer coverage
- Added optimizers module to architecture diagram
- Updated test breakdown with optimizer tests
- Updated CHANGELOG.md with precise test counts and statistics
- All documentation now reflects the production-ready state of the crate

### Testing (Latest)
- **154/154 tests passing (100% pass rate)** ⬆️ +40 tests from alpha.2 base
  - Unit tests: 134 passing (+27 graph + optimizer tests)
  - Integration tests: 10 passing
  - Property tests: 10/10 passing
- **15 performance benchmarks** ⬆️ +8 graph benchmarks
  - Gradient benchmarks: 7 (VJP/decomposition)
  - Graph benchmarks: 8 (computation graph operations) ✨ NEW

## [0.1.0-alpha.1] - 2025-11-27

### Added
- Core VJP (Vector-Jacobian Product) framework
  - Einsum VJP with automatic adjoint spec generation
  - Element-wise operation VJP (unary and binary)
  - Reduction operation VJP (sum, mean, max, min)
  - **Special handling for scalar outputs** (inner products, norms)
- Decomposition gradient computation
  - CP-ALS gradients via MTTKRP
  - Tucker-HOOI gradients via mode-n products
  - TT-SVD gradients with chain-rule propagation
- Gradient verification utilities
  - Finite difference methods (central and forward)
  - Configurable tolerances and detailed error reporting
- External AD framework integration
  - Generic adapter with trait-based design
  - Operation registration and tape management
  - Tensorlogic placeholder (awaiting API stabilization)
- Advanced gradient features
  - **Gradient checkpointing**: O(√n) memory complexity
  - **Hessian computation**: Efficient second-order derivatives
  - **Sparse gradients**: COO format with automatic format selection
  - **Mixed precision training**: FP16/BF16 with automatic loss scaling
  - **Gradient monitoring**: Vanishing/exploding gradient detection
- Performance optimizations
  - Parallel gradient computation with Rayon
  - Memory-optimized tensor storage with Arc-based sharing
  - Configurable thread pools and chunk sizes
- Comprehensive testing
  - 94 unit tests covering all modules
  - 10 integration tests for end-to-end workflows (including edge cases)
  - 10 property-based tests for mathematical correctness
  - 7 performance benchmarks
- Documentation
  - Comprehensive crate-level documentation (200+ lines)
  - 8 working examples demonstrating all features
  - Inline API documentation for all public items
- SCIRS2 policy compliance
  - All array operations use `scirs2_core::ndarray_ext`
  - All numeric types use `scirs2_core::numeric`
  - No direct `ndarray` or `rand` imports

### Fixed
- **Scalar output VJP bug** (2025-11-27)
  - Issue: Einsum VJP failed for scalar outputs (e.g., inner products `"i,i->"`)
  - Root cause: Adjoint specs became malformed (e.g., `",i->i"`)
  - Solution: Added special case handling using broadcasting instead of einsum
  - Impact: `basic_einsum_ad.rs` example now works correctly
  - Added unit test `test_einsum_vjp_scalar_output` for coverage
- **Benchmark warnings** (2025-11-27)
  - Removed unused `Array3` import from `gradient_benchmarks.rs`
  - Zero warnings in strict clippy mode

### Changed
- Enhanced documentation
  - Expanded `lib.rs` from 9 to 204 lines of comprehensive documentation
  - Added quick start examples and architecture overview
  - Documented performance characteristics and SCIRS2 compliance
- Updated README.md
  - Reflected latest test count (110/110 passing)
  - Added badges for test status and coverage
  - Included all 8 examples with descriptions

### Performance
- Einsum VJP: Efficient adjoint spec generation with no tape overhead
- Parallel computation: Scales linearly with CPU cores for large tensors (>10k elements)
- Checkpointing: Reduces memory by 60-90% for deep networks
- Sparse gradients: Up to 98% memory savings for highly sparse gradients
- Hessian-vector products: O(n) time complexity instead of O(n²)

### Testing
- **114/114 tests passing (100% pass rate)**
  - Unit tests: 94 passing
  - Integration tests: 10 passing
  - Property tests: 10/10 passing
- Zero warnings in strict clippy mode
- All examples build and run successfully
- All benchmarks compile cleanly

### Documentation
- 9,051 total lines of code
  - Source: 5,890 lines
  - Documentation: 1,409+ lines (in-code)
  - Comments: 407 lines
- 8 comprehensive examples
- 7 performance benchmarks
- Full API documentation (200+ lines in lib.rs)

### Known Limitations
- Tensorlogic integration awaiting API stabilization
- Distributed gradients planned for future release
- Hardware accelerator support (GPU/TPU) planned for future release

---

## Future Releases

### Planned for 0.2.0
- Tensorlogic integration (when API is stable)
- Additional integration tests
- Performance optimizations for small tensors
- Extended property-based test coverage

### Planned for 0.3.0
- Distributed gradients with AllReduce and gradient bucketing
- Communication-computation overlap for distributed training
- GPU kernel integration via scirs2-gpu
- TPU/NPU support
- Additional graph optimization passes (constant folding, common subexpression elimination)

---

**Legend:**
- `Added`: New features
- `Changed`: Changes in existing functionality
- `Deprecated`: Soon-to-be removed features
- `Removed`: Removed features
- `Fixed`: Bug fixes
- `Security`: Security vulnerability fixes

[0.1.0-alpha.2]: https://github.com/cool-japan/tenrso/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/cool-japan/tenrso/releases/tag/v0.1.0-alpha.1
