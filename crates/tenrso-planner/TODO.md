# tenrso-planner TODO

> **Milestone:** M4
> **Version:** 0.1.0-rc.1
> **Status:** RC.1 — 271 tests passing (100%) — 2026-03-06
> **Last Updated:** 2026-03-06

---

## M4: Planning Implementation - COMPLETE

### Core Planning

- [x] **Einsum Specification Parser** - COMPLETE
  - [x] Parse Einstein summation notation ("ijk,jkl->il")
  - [x] Validate subscripts and shape compatibility
  - [x] Identify contracted and output indices
  - [x] Support output inference (auto-detection)
  - [x] 16 unit tests + 1 doc test
  - **Module:** `parser.rs` (318 lines)

- [x] **Cost Model** - COMPLETE
  - [x] TensorStats struct (dense/sparse statistics)
  - [x] FLOPs estimation for pairwise contractions
  - [x] Memory usage estimation (input + output)
  - [x] Output sparsity prediction (density product heuristic)
  - [x] CostEstimate with complete analysis
  - [x] ML-based cost models (linear + polynomial regression)
  - [x] 14 comprehensive unit tests
  - **Module:** `cost.rs` (462 lines)

- [x] **Contraction Order Search** - COMPLETE
  - [x] Greedy heuristic (minimize cost at each step)
  - [x] IntermediateTensor tracking
  - [x] Pairwise contraction spec computation
  - [x] Plan data structure with nodes and order
  - [x] 7 unit tests (matmul, chain, 3-tensor, error cases)
  - [x] Dynamic programming (optimal for small graphs) - COMPLETE
  - [x] Beam Search planner (configurable width)
  - [x] Simulated Annealing planner
  - [x] Genetic Algorithm planner (struct + Planner trait)
  - [x] Adaptive planner (auto-selects algorithm)
  - [x] Plan refinement via local search
  - **Module:** `order/` (2,155+ lines across 8 files)

- [x] **Representation Selection** - COMPLETE
  - [x] Dense vs sparse threshold
  - [x] Low-rank approximation heuristics
  - [x] Cost-benefit analysis
  - [x] Memory pressure consideration
  - [x] Conversion cost estimation
  - [x] 17 unit tests
  - **Module:** `repr.rs` (440 lines)

- [x] **Tiling Strategy** - COMPLETE
  - [x] Cache-aware tile sizes (L1/L2/L3)
  - [x] Memory budget constraints
  - [x] Matrix multiplication blocking
  - [x] Cache miss estimation
  - [x] Multi-level (nested) tiling for modern cache hierarchies
  - [x] 14+ unit tests
  - **Module:** `tiling.rs` (807 lines)

- [x] **Plan Management** - COMPLETE
  - [x] Plan caching (LRU/LFU/ARC eviction policies)
  - [x] Plan profiling and execution time tracking
  - [x] Parallel planning support
  - [x] Plan comparison (`Plan::compare()`, `Plan::is_better_than()`)
  - [x] Plan analysis (`Plan::analyze()`)
  - [x] Serialization/deserialization (optional `serde` feature)

---

## Testing & Documentation

- [x] Unit tests for parser - 16 tests
- [x] Unit tests for cost model - 14 tests
- [x] Unit tests for greedy planner - 7 tests
- [x] Unit tests for DP planner - 11 tests
- [x] Unit tests for beam search planner - 6 tests
- [x] Unit tests for simulated annealing planner - 5 tests
- [x] Unit tests for genetic algorithm planner - 6 tests
- [x] Unit tests for adaptive planner - 9 tests
- [x] Unit tests for representation selection - 17 tests
- [x] Unit tests for tiling strategy - 14 tests
- [x] API structure tests - 3 tests
- [x] Property tests for plan optimality - 7 tests
- [x] Benchmarks for plan quality - 14 benchmark suites
- [x] Examples and documentation - 4 examples

**Total Tests:** 271 tests passing (100%)

---

## API Structure

### Public API (`api.rs`)
- `Plan` - Complete contraction plan with cost estimates
- `PlanNode` - Individual contraction step
- `ContractionSpec` - Einsum-style contraction specification
- `ReprHint` - Dense/Sparse/LowRank/Auto
- `PlanHints` - User preferences (memory budget, device, sparsity)
- `Planner` trait - Main interface for planning
- `PlanComparison` - Plan comparison result with improvement percentages
- `PlanAnalysis` - Detailed plan metrics
- `PlanMetric` enum - FLOPs, Memory, Steps, Combined

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

### Order Planning (`order/`)
- `greedy_planner()` / `GreedyPlanner` - Greedy contraction order search
- `dp_planner()` / `DPPlanner` - Dynamic programming (optimal for n<=20)
- `BeamSearchPlanner` - Beam search with configurable width
- `SimulatedAnnealingPlanner` - Stochastic search with temperature
- `GeneticAlgorithmPlanner` - Evolutionary search (fast/default/high_quality presets)
- `AdaptivePlanner` - Automatic algorithm selection
- `refine_plan()` - Post-planning local search improvement

---

## Module Structure

```
src/
├── lib.rs              - Module exports and documentation
├── api.rs              - Plan data structures + comparison/analysis + serde
├── parser.rs           - Einsum specification parser
├── cost.rs             - Cost model for FLOPs and memory
├── repr.rs             - Representation selection heuristics
├── tiling.rs           - Single + multi-level tiling strategies
└── order/
    ├── mod.rs
    ├── types.rs        - 6 planner structs
    ├── functions.rs    - Core planning functions
    ├── greedyplanner_traits.rs
    ├── dpplanner_traits.rs
    ├── beamsearchplanner_traits.rs
    ├── simulatedannealingplanner_traits.rs
    ├── adaptiveplanner_traits.rs
    └── geneticalgorithmplanner_traits.rs
```

---

## Planner Comparison

| Algorithm | Struct | Time | Space | Quality | Best For |
|-----------|--------|------|-------|---------|----------|
| **Greedy** | yes | O(n³) | O(n) | Good | General, fast |
| **Beam Search** | yes | O(n³k) | O(kn) | Better | Medium networks |
| **DP** | yes | O(3^n) | O(2^n) | Optimal | n <= 20 |
| **Simulated Annealing** | yes | O(iter*n) | O(n) | Variable | Large, quality |
| **Genetic Algorithm** | yes | O(gen*pop*n³) | O(pop*n) | High | Very large, complex |
| **Adaptive** | yes | Varies | Varies | Auto | All cases |

---

**Milestone M4:** COMPLETE
**Last Updated:** 2026-03-06
