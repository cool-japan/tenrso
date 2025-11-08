# tenrso-planner TODO

> **Milestone:** M4
> **Status:** 🔄 IN PROGRESS

---

## M4: Planning Implementation - 🔄 IN PROGRESS

### Core Planning

- [x] **Einsum Specification Parser** - ✅ COMPLETE
  - [x] Parse Einstein summation notation ("ijk,jkl->il")
  - [x] Validate subscripts and shape compatibility
  - [x] Identify contracted and output indices
  - [x] Support output inference (auto-detection)
  - [x] 16 unit tests + 1 doc test
  - **Module:** `parser.rs` (318 lines)

- [x] **Cost Model** - ✅ COMPLETE
  - [x] TensorStats struct (dense/sparse statistics)
  - [x] FLOPs estimation for pairwise contractions
  - [x] Memory usage estimation (input + output)
  - [x] Output sparsity prediction (density product heuristic)
  - [x] CostEstimate with complete analysis
  - [x] 14 comprehensive unit tests
  - **Module:** `cost.rs` (462 lines)

- [x] **Contraction Order Search** - ✅ GREEDY COMPLETE
  - [x] Greedy heuristic (minimize cost at each step)
  - [x] IntermediateTensor tracking
  - [x] Pairwise contraction spec computation
  - [x] Plan data structure with nodes and order
  - [x] 7 unit tests (matmul, chain, 3-tensor, error cases)
  - [x] Dynamic programming (optimal for small graphs) - ✅ COMPLETE
  - **Module:** `order.rs` (362 lines)

- [x] **Representation Selection** - ✅ COMPLETE
  - [x] Dense vs sparse threshold
  - [x] Low-rank approximation heuristics
  - [x] Cost-benefit analysis
  - [x] Memory pressure consideration
  - [x] Conversion cost estimation
  - [x] 17 unit tests
  - **Module:** `repr.rs` (440 lines)

- [x] **Tiling Strategy** - ✅ COMPLETE
  - [x] Cache-aware tile sizes (L1/L2/L3)
  - [x] Memory budget constraints
  - [x] Matrix multiplication blocking
  - [x] Cache miss estimation
  - [x] 14 unit tests
  - **Module:** `tiling.rs` (422 lines)

---

## Testing & Documentation

- [x] Unit tests for parser - ✅ 16 tests
- [x] Unit tests for cost model - ✅ 14 tests
- [x] Unit tests for greedy planner - ✅ 7 tests
- [x] Unit tests for DP planner - ✅ 11 tests
- [x] Unit tests for representation selection - ✅ 17 tests
- [x] Unit tests for tiling strategy - ✅ 14 tests
- [x] API structure tests - ✅ 3 tests
- [x] Property tests for plan optimality - ✅ 7 tests
- [x] Benchmarks for plan quality - ✅ 7 benchmark suites
- [x] Examples and documentation - ✅ 3 examples + enhanced docs
- [ ] Integration tests with tenrso-exec - ⏳ Depends on exec implementation

**Total Tests:** ✅ **82 tests passing** (75 unit + 7 property + 2 doc)

---

## API Structure

### Public API (`api.rs`)
- `Plan` - Complete contraction plan with cost estimates
- `PlanNode` - Individual contraction step
- `ContractionSpec` - Einsum-style contraction specification
- `ReprHint` - Dense/Sparse/LowRank/Auto
- `PlanHints` - User preferences (memory budget, device, sparsity)
- `Planner` trait - Main interface for planning

### Parser (`parser.rs`)
- `EinsumSpec` - Parsed einsum specification
- `validate_shapes()` - Shape consistency checking
- `compute_output_shape()` - Output shape calculation

### Cost Model (`cost.rs`)
- `TensorStats` - Dense/sparse tensor statistics
- `estimate_flops()` - FLOPs estimation for contractions
- `estimate_memory()` - Memory usage estimation
- `estimate_output_sparsity()` - Sparsity prediction
- `CostEstimate` - Complete cost analysis

### Order Planning (`order.rs`)
- `greedy_planner()` - Greedy contraction order search
- `dp_planner()` - Dynamic programming (placeholder)

---

**Last Updated:** 2025-11-04 (Evening)

---

## Progress Notes

### 2025-11-04 (Late Evening) - M4 Core Planning Complete ✅

**Implemented:**
1. Complete einsum parser with validation and output inference
2. Comprehensive cost model with dense/sparse support
3. Greedy contraction order planner with O(n³) complexity
4. **Representation selection with sparse/dense/low-rank heuristics**
5. **Cache-aware tiling strategies for L1/L2/L3**
6. 66 tests covering all core functionality

**Key Achievements:**
- Parser handles complex multi-tensor specs (e.g., "ijk,jkl,klm->ilm")
- Cost model supports both dense and sparse tensors
- Greedy planner finds locally optimal contraction orders
- Representation selection chooses optimal formats based on sparsity and memory
- Tiling strategies optimize for cache locality and memory budgets
- All tests passing in full workspace (343 total)

**Module Summary:**
- `api.rs` (189 lines): Plan data structures
- `parser.rs` (318 lines): Einsum specification parser
- `cost.rs` (462 lines): Cost model for FLOPs and memory
- `order.rs` (362 lines): Greedy contraction order planner
- `repr.rs` (440 lines): Representation selection heuristics
- `tiling.rs` (422 lines): Cache-aware tiling strategies

**Next Steps:**
1. Integrate with tenrso-exec for plan execution
2. Add dynamic programming planner for small graphs
3. Create integration tests and benchmarks
4. Performance tuning and optimization

### 2025-11-06 - M4 Enhanced with DP Planner, Property Tests, Benchmarks ✅

**Major Enhancements:**
1. **Dynamic Programming Planner** - Full O(3ⁿ) optimal contraction order search
   - Bitmask DP with memoization
   - Backtracking for plan reconstruction
   - Automatic fallback to greedy for n > 20
   - 11 comprehensive tests including optimality verification

2. **Property-Based Testing** - 7 proptest suites
   - Plan correctness for various topologies
   - DP optimality (cost ≤ greedy)
   - Determinism verification
   - Memory and cost invariants

3. **Benchmark Suite** - 7 comprehensive benchmarks
   - Matrix chain multiplication (greedy vs DP)
   - Star topology contractions
   - Dense network planning
   - Plan quality comparison
   - Size sensitivity analysis

4. **Usage Examples** - 3 detailed examples
   - `basic_matmul.rs`: Simple matrix multiplication
   - `matrix_chain.rs`: Greedy vs DP comparison
   - `planning_hints.rs`: Advanced hint usage

5. **Enhanced Documentation**
   - Comprehensive module-level docs with quickstart
   - Complexity analysis and performance notes
   - References to literature and related work
   - 2 doc tests (both passing)

**Module Summary (Updated):**
- `api.rs` (182 lines): Plan data structures
- `parser.rs` (318 lines): Einsum specification parser
- `cost.rs` (462 lines): Cost model for FLOPs and memory
- `order.rs` (972 lines): Greedy + DP planners with property tests
- `repr.rs` (440 lines): Representation selection heuristics
- `tiling.rs` (422 lines): Cache-aware tiling strategies

**Test Coverage:** 82 tests (75 unit + 7 property + 2 doc)
**Benchmarks:** 7 comprehensive benchmark suites
**Examples:** 3 fully documented examples

**Status:** M4 core planning complete with production-grade testing and documentation

### 2025-11-06 (Final) - M4 Complete with Advanced Features ✅

**Final Enhancements:**

6. **Planner Trait Implementations** - Struct-based planners
   - `GreedyPlanner` and `DPPlanner` structs implementing `Planner` trait
   - Support for trait object polymorphism
   - Clean API with `new()` and `Default` implementations
   - 5 additional tests for trait functionality
   - Doc tests with usage examples

7. **Plan Comparison & Analysis** - Production utilities
   - `Plan::compare()` - detailed comparison between two plans
   - `Plan::analyze()` - comprehensive plan statistics
   - `Plan::is_better_than()` - metric-based plan comparison
   - `PlanComparison` struct with improvement percentages
   - `PlanAnalysis` struct with detailed metrics
   - `PlanMetric` enum (FLOPs, Memory, Steps, Combined)
   - 5 comprehensive tests for comparison/analysis

8. **Serde Support** - Serialization/deserialization
   - Optional `serde` feature flag
   - Full serialization support for all plan data structures
   - JSON serialization/deserialization
   - Backward-compatible with zero-cost abstraction
   - 3 comprehensive serde tests
   - Production-ready for plan persistence and sharing

**Final Statistics:**
- **Total Tests:** 99 tests (95 unit + 4 doc) ✅
- **Test Success Rate:** 100% passing
- **Code Coverage:** Comprehensive
- **Features:**
  - Default: Core planning functionality
  - `serde`: Optional serialization support

**Module Summary (Final):**
- `api.rs` (491 lines): Plan data structures + comparison/analysis + serde
- `parser.rs` (318 lines): Einsum specification parser
- `cost.rs` (462 lines): Cost model for FLOPs and memory
- `order.rs` (1098 lines): Greedy + DP planners + trait impls + property tests
- `repr.rs` (440 lines): Representation selection heuristics
- `tiling.rs` (422 lines): Cache-aware tiling strategies
- **Total:** ~3,200 lines of production code

**Examples & Documentation:**
- 3 comprehensive examples
- 4 doc tests (all passing)
- Extensive API documentation
- Performance benchmarks (7 suites)

**Key Capabilities:**
✅ Optimal contraction order planning (DP)
✅ Fast heuristic planning (Greedy)
✅ Plan comparison and analysis
✅ Serialization/deserialization
✅ Property-based testing
✅ Comprehensive benchmarking
✅ Production-ready error handling
✅ Zero-cost abstractions
✅ Trait-based polymorphism

**Status:** ✨ **M4 COMPLETE** - Production-grade tensor contraction planner

### 2025-11-04 (AM) - M3 Complete, M4 Started

- **M3 Status:** Sparse formats and operations complete (113 tests)
- **Dependencies:** Can now use sparse statistics for planning
- **Status:** M4 foundation laid with parser, cost model, and greedy planner
