# tenrso-exec

[![Crates.io](https://img.shields.io/crates/v/tenrso-exec)](https://crates.io/crates/tenrso-exec)
[![Documentation](https://docs.rs/tenrso-exec/badge.svg)](https://docs.rs/tenrso-exec)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**Unified execution API for TenRSo tensor operations.**

Part of the [TenRSo](https://github.com/cool-japan/tenrso) tensor computing stack.

## Overview

`tenrso-exec` provides the main user-facing API for executing tensor operations:

- **`einsum_ex`** - Unified einsum contraction interface
- **TenrsoExecutor trait** - Backend abstraction (CPU, GPU)
- **Execution hints** - Control representation, tiling, masking
- **Auto-optimization** - Automatic planner integration
- **Memory pooling** - Thread-local buffer pools with smart heuristics

All tensor operations (dense, sparse, low-rank) go through this unified interface.

## Features

- Single API for all tensor representations
- Automatic optimization via planner
- Memory pooling (phases 1-5.1): thread-local, smart heuristics, auto pooling
- SIMD operations: vectorized element-wise, 2-8x speedup
- Tiled reductions: cache-friendly blocked reductions
- Vectorized broadcasting: pattern-aware kernels
- Conv1d/2d/3d with pooling layers
- Shape manipulation: concat, tile, pad, flip, squeeze, unsqueeze, stack, repeat, roll
- Advanced indexing: gather, scatter, fancy (boolean mask) indexing
- Parallel execution
- Custom execution hints

## Installation

```toml
[dependencies]
tenrso-exec = "0.1.0-rc.1"
```

## Quick Start

### Basic Einsum

```rust
use tenrso_exec::{einsum_ex, ExecHints};

// Simple matrix multiplication
let c = einsum_ex::<f32>("ij,jk->ik")
    .inputs(&[a, b])
    .run()?;
```

### With Hints

```rust
// Tensor contraction with optimization hints
let result = einsum_ex::<f32>("bij,bjk->bik")
    .inputs(&[a, b])
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

### Convolutions

```rust
let result = exec.conv2d(
    &input,    // [batch, height, width, channels]
    &kernel,   // [out_channels, kH, kW, in_channels]
    stride,
    padding,
)?;
```

## Performance Configuration

`tenrso-exec` includes advanced optimization features configurable per executor:

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

### Optimization Features

- **SIMD Operations** (`enable_simd`):
  - Vectorized element-wise operations (neg, abs, exp, log, sin, cos, etc.)
  - Vectorized binary operations (add, sub, mul, div, etc.)
  - Automatically activated for tensors >= 1024 elements
  - Typical speedup: 2-4x for simple ops, up to 8x for expensive ops (exp, sin)

- **Tiled Reductions** (`enable_tiled_reductions`):
  - Cache-friendly blocked reductions using 4KB tiles
  - Optimizes sum, mean, max, min operations
  - Automatically activated for tensors >= 100K elements
  - Typical speedup: 1.5-3x for large tensors (reduces cache misses)

- **Vectorized Broadcasting** (`enable_vectorized_broadcast`):
  - Pattern-aware broadcasting with specialized kernels
  - Detects common patterns (scalar, same-shape, axis-specific)
  - Parallel execution for large operations
  - Typical speedup: 1.5-2x for broadcast-heavy workloads

## Memory Pooling

Automatic buffer pooling for common tensor operations with zero API changes:

```rust
// Pool automatically used for broadcasting, conv1d/2d/3d,
// concatenate, max_pool_2d, avg_pool_2d, tile, pad, flip
let result = exec.conv2d(&input, &kernel, stride, padding)?;

// Manual pool management
let buf = exec.acquire_f32(&[batch, height, width]);
// ... use buf ...
exec.release_f32(&[batch, height, width], buf);

// Pool statistics
let stats = exec.get_pool_stats_f32();
println!("Hit rate: {:.1}%", stats.hit_rate * 100.0);
```

**Pooled operations (10 total):**
- Binary ops with broadcasting
- Conv1d, Conv2d, Conv3d
- Concatenate, MaxPool2d, AvgPool2d, Tile, Pad, Flip

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
    fn binary_op(&mut self, op: BinaryOp, a: &TensorHandle<T>, b: &TensorHandle<T>)
        -> Result<TensorHandle<T>>;
    fn matmul(&mut self, a: &TensorHandle<T>, b: &TensorHandle<T>)
        -> Result<TensorHandle<T>>;
    fn conv2d(&mut self, input: &TensorHandle<T>, kernel: &TensorHandle<T>,
              stride: usize, padding: usize) -> Result<TensorHandle<T>>;
    // ... and many more
}
```

### Supported Operations

**Element-wise:** Neg, Abs, Exp, Log, Sqrt, Sin, Cos, Tanh, Sigmoid, ReLU, GELU, Recip, Sign

**Binary:** Add, Sub, Mul, Div, Pow, Max, Min

**Reductions:** Sum, Mean, Max, Min, Prod, All, Any, ArgMax, ArgMin

**Shape:** Reshape, Permute, Squeeze, Unsqueeze, Expand, Stack, Repeat, Roll

**Tensor manipulation:** Tile, Pad, Flip, Concatenate, Slice, Gather, Scatter

**Convolutions:** Conv1d, Conv2d, Conv3d, MaxPool2d, AvgPool2d

## Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run optimization-specific benchmarks
cargo bench --bench optimization_benchmarks

# Compare optimized vs unoptimized performance
cargo bench --bench optimization_benchmarks -- simd
cargo bench --bench optimization_benchmarks -- tiled
```

## Testing

```bash
cargo test --package tenrso-exec
```

**Test Coverage:** 244 tests (100% passing)

## Dependencies

- **tenrso-core** - Tensor types
- **tenrso-kernels** - Tensor kernels
- **tenrso-sparse** - Sparse operations
- **tenrso-decomp** - Decompositions
- **tenrso-planner** - Contraction planning
- **tenrso-ooc** (optional) - Out-of-core support

## License

Apache-2.0

## Related Projects

- [`tenrso-core`](../tenrso-core) - Core tensor data structures
- [`tenrso-planner`](../tenrso-planner) - Contraction planning
- [`tenrso-ooc`](../tenrso-ooc) - Out-of-core processing
- [`tenrso`](../tenrso) - Main TenRSo library

---

**Status:** Alpha (production-ready internals) | **Version:** 0.1.0-rc.1 | **Tests:** 244/244 passing
