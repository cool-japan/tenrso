# tenrso-exec

Unified execution API for TenRSo tensor operations.

## Overview

`tenrso-exec` provides the main user-facing API for executing tensor operations:

- **`einsum_ex`** - Unified einsum contraction interface
- **TenrsoExecutor trait** - Backend abstraction (CPU, GPU)
- **Execution hints** - Control representation, tiling, masking
- **Auto-optimization** - Automatic planner integration

All tensor operations (dense, sparse, low-rank) go through this unified interface.

## Features

- Single API for all tensor representations
- Automatic optimization via planner
- Memory pooling and device management
- Parallel execution
- Custom execution hints

## Usage

```toml
[dependencies]
tenrso-exec = "0.1"
```

### Basic Einsum (TODO: M4)

```rust
use tenrso_exec::{einsum_ex, ExecHints};

// Simple matrix multiplication
let C = einsum_ex::<f32>("ij,jk->ik")
    .inputs(&[A, B])
    .run()?;
```

### With Hints (TODO: M4)

```rust
// Tensor contraction with optimization hints
let result = einsum_ex::<f32>("bij,bjk->bik")
    .inputs(&[A, B])
    .hints(&ExecHints {
        prefer_lowrank: true,
        prefer_sparse: true,
        tile_kb: Some(512),
        ..Default::default()
    })
    .run()?;
```

### Element-wise & Reductions

```rust
use tenrso_exec::{CpuExecutor, TenrsoExecutor, ElemOp, ReduceOp};

let mut exec = CpuExecutor::new();

// Element-wise operation
let abs_tensor = exec.elem_op(ElemOp::Abs, &tensor)?;

// Reduction
let sum = exec.reduce(ReduceOp::Sum, &tensor, &[0, 1])?;
```

### Performance Configuration

`tenrso-exec` includes advanced optimization features that can be configured per executor:

```rust
use tenrso_exec::CpuExecutor;

// Default: all optimizations enabled
let mut exec = CpuExecutor::new();

// Custom configuration with selective optimizations
let mut exec = CpuExecutor::new()
    .with_simd(true)                    // SIMD-accelerated operations
    .with_tiled_reductions(true)        // Cache-friendly blocked reductions
    .with_vectorized_broadcast(true);   // Optimized broadcasting patterns

// Disable all optimizations (for debugging or baseline comparison)
let mut exec = CpuExecutor::unoptimized();
```

#### Optimization Features

- **SIMD Operations** (`enable_simd`):
  - Vectorized element-wise operations (neg, abs, exp, log, sin, cos, etc.)
  - Vectorized binary operations (add, sub, mul, div, etc.)
  - Automatically activated for tensors ≥1024 elements
  - Typical speedup: 2-4× for simple ops, up to 8× for expensive ops (exp, sin)

- **Tiled Reductions** (`enable_tiled_reductions`):
  - Cache-friendly blocked reductions using 4KB tiles
  - Optimizes sum, mean, max, min operations
  - Automatically activated for tensors ≥100K elements
  - Typical speedup: 1.5-3× for large tensors (reduces cache misses)

- **Vectorized Broadcasting** (`enable_vectorized_broadcast`):
  - Pattern-aware broadcasting with specialized kernels
  - Detects common patterns (scalar, same-shape, axis-specific)
  - Parallel execution for large operations
  - Typical speedup: 1.5-2× for broadcast-heavy workloads

#### When to Use Each Optimization

**Enable SIMD when**:
- Working with large vectors/tensors (>1K elements)
- Performing many element-wise operations
- Using expensive math functions (exp, log, trigonometric)

**Enable Tiled Reductions when**:
- Reducing very large tensors (>100K elements)
- Memory bandwidth is a bottleneck
- Working with multi-dimensional reductions

**Disable optimizations when**:
- Debugging numerical differences
- Profiling baseline performance
- Working with very small tensors (<1K elements)
- Comparing against reference implementations

#### Performance Tuning Guidelines

1. **Default configuration is optimal for most workloads**:
   ```rust
   let mut exec = CpuExecutor::new(); // All optimizations enabled
   ```

2. **For debugging or numerical verification**:
   ```rust
   let mut exec = CpuExecutor::unoptimized();
   ```

3. **For memory-constrained environments**:
   ```rust
   let mut exec = CpuExecutor::new()
       .with_tiled_reductions(false); // Reduce memory footprint
   ```

4. **For maximum throughput on modern CPUs**:
   ```rust
   let mut exec = CpuExecutor::new(); // All optimizations enabled by default
   ```

#### Benchmarking

Run comprehensive benchmarks to measure optimization impact:

```bash
# Run all benchmarks
cargo bench

# Run optimization-specific benchmarks
cargo bench --bench optimization_benchmarks

# Compare optimized vs unoptimized performance
cargo bench --bench optimization_benchmarks -- simd
cargo bench --bench optimization_benchmarks -- tiled
```

Benchmark results include:
- SIMD element-wise operations at various tensor sizes
- Tiled reductions vs standard reductions
- Combined optimization pipeline performance
- Automatic threshold detection verification

## API Reference

### Einsum Builder

```rust
pub fn einsum_ex<T>(spec: &str) -> EinsumBuilder<T>

impl<T> EinsumBuilder<T> {
    pub fn inputs(self, tensors: &[TensorHandle<T>]) -> Self;
    pub fn hints(self, hints: &ExecHints) -> Self;
    pub fn run(self) -> Result<TensorHandle<T>>;
}
```

### Execution Hints

```rust
pub struct ExecHints {
    pub mask: Option<MaskPack>,
    pub subset: Option<SubsetSpec>,
    pub prefer_sparse: bool,
    pub prefer_lowrank: bool,
    pub tile_kb: Option<usize>,
}
```

### Executor Trait

```rust
pub trait TenrsoExecutor<T> {
    fn einsum(&mut self, spec: &str, inputs: &[TensorHandle<T>], hints: &ExecHints)
        -> Result<TensorHandle<T>>;
    fn elem_op(&mut self, op: ElemOp, x: &TensorHandle<T>) -> Result<TensorHandle<T>>;
    fn reduce(&mut self, op: ReduceOp, x: &TensorHandle<T>, axes: &[Axis])
        -> Result<TensorHandle<T>>;
}
```

## Dependencies

- **tenrso-core** - Tensor types
- **tenrso-kernels** - Tensor kernels
- **tenrso-sparse** - Sparse operations
- **tenrso-decomp** - Decompositions
- **tenrso-planner** - Contraction planning
- **tenrso-ooc** (optional) - Out-of-core support

## License

Apache-2.0
