# Changelog

All notable changes to the `tenrso-planner` crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SciRS2-Core integration for professional-grade RNG (2025-11-27)
  - Replaced custom LCG with `scirs2_core::random::StdRng`
  - Simulated Annealing planner now uses SciRS2-Core RNG
  - Genetic Algorithm planner now uses SciRS2-Core RNG
  - Maintained reproducibility with fixed seeds
- Plan visualization and debugging utilities (2025-12-06)
  - `Display` trait implementation for `Plan`, `PlanNode`, and `ReprHint`
  - `Plan::to_dot()` - Export plans as Graphviz DOT format for visualization
  - `Plan::validate()` - Comprehensive plan correctness validation
  - 13 new unit tests for visualization and validation features

### Changed
- Enhanced documentation to reflect SciRS2-Core integration
- Updated README with comprehensive planner comparison table

### Fixed
- Fixed rustdoc broken intra-doc links to `Planner` trait (2025-12-06)
  - Added `#[allow(unused_imports)]` for doc-only import in `types.rs`
  - All documentation now builds without warnings
- Fixed workspace policy compliance in Cargo.toml (2025-12-06)
  - Changed `version` to use `workspace = true`
  - Changed `serde`, `criterion`, and `serde_json` to use workspace versions
  - Follows CLAUDE.md workspace policy requirement
- Fixed clippy warning about collapsible if statements

## [0.1.0-alpha.1] - 2025-11-26

### Added
- Complete einsum specification parser with validation
- Comprehensive cost model for FLOPs and memory estimation
- Six production-grade planning algorithms:
  - Greedy planner (O(n³) heuristic)
  - Dynamic Programming planner (optimal for n ≤ 20)
  - Beam Search planner (configurable width)
  - Simulated Annealing planner (stochastic search)
  - Genetic Algorithm planner (evolutionary optimization)
  - Adaptive planner (auto-selects best algorithm)
- Representation selection (dense/sparse/low-rank)
- Cache-aware tiling strategies (L1/L2/L3)
- Multi-level tiling for modern CPU hierarchies
- Plan comparison and analysis utilities
- Plan refinement via local search
- Serde support for plan serialization (optional feature)
- 136 comprehensive unit tests
- 7 property-based tests
- 8 doc tests
- 14 benchmark suites covering all planners
- 4 detailed examples:
  - `basic_matmul.rs` - Simple matrix multiplication
  - `matrix_chain.rs` - Greedy vs DP comparison
  - `planning_hints.rs` - Advanced hint usage
  - `genetic_algorithm.rs` - GA planner showcase

### Features
- `serde` - Optional serialization/deserialization support

### Performance
- Greedy: ~1ms for 10 tensors
- Beam Search (k=5): ~5ms for 10 tensors
- DP: ~1ms for 5 tensors, ~100ms for 10 tensors
- Simulated Annealing: ~10-100ms (configurable)
- Genetic Algorithm: ~50-500ms (configurable)

### Quality
- Zero compilation warnings
- Zero clippy warnings
- 100% test pass rate
- Full rustdoc coverage
- Production-ready error handling

## [0.1.0-alpha.0] - 2025-11-04

### Added
- Initial M4 milestone implementation
- Basic einsum parser
- Greedy contraction order planner
- Cost estimation framework
- Tensor statistics tracking
- Dense and sparse tensor support

---

**Legend:**
- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security improvements
