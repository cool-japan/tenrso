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
tenrso-exec = "0.1.0"
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

`ExecHints` selects *which output cells to compute*. Either spelling — a dense
`mask` or a sparse `subset` of flat indices — routes the einsum through the
masked engine, which computes only the selected cells.

```rust
// Compute only the diagonal of the 3x3 output.
let result = einsum_ex::<f32>("ij,jk->ik")
    .inputs(&[a, b])
    .hints(
        &ExecHints::new()
            .with_sparse(true)
            .with_subset(vec![0, 4, 8], vec![3, 3]),
    )
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

// Default: all optimizations enabled, parallel regions on rayon's ambient pool
let mut exec = CpuExecutor::new();

// Selective configuration
let mut exec = CpuExecutor::new()
    .with_simd(true)                 // AVX2 kernels for the exp/log family
    .with_blocked_reductions(true);  // multi-accumulator reductions

// Bound the executor's parallelism to a private pool of 4 threads
let mut exec = CpuExecutor::with_threads(4)?;

// Disable all optimizations (for debugging or baseline comparison)
let mut exec = CpuExecutor::unoptimized();
```

These knobs apply to the inherent methods that read executor configuration
(`parallel_elem_op`, `scalar_op`, `parallel_binary_op`, `full_reduce`). The
`TenrsoExecutor` *trait* methods are a simpler path that deliberately does not
consult executor configuration.

### Optimization Features

All figures below were measured on a Xeon Gold 5315Y, medians over repeated runs
on a contended machine. They are the numbers the current code actually produces,
not targets.

- **SIMD element-wise** (`enable_simd`):
  - Real AVX2 + FMA kernels for `exp`, `log`, and the six activations built on
    them (`sigmoid`, `tanh`, `gelu`, `elu`, `selu`, `softplus`).
  - Measured **~2-5x** vs scalar libm for `exp` (1M f64: 7.44 ms -> 1.98 ms).
    Accurate to ~1 ULP; NaN/inf/out-of-range lanes fall back to scalar libm and
    are bit-identical to it.
  - **No SIMD path for bandwidth-bound ops** (`neg`, `abs`, `relu`, `sqrt`,
    `sqr`, `recip`, `sign`, `sin`, `cos`). These are already at memory speed, and
    an intrinsic `relu` measured *slower* than `mapv` (8M f64: 20.5 ms -> 35.0 ms),
    so no kernel is provided and none is claimed.
  - Requires x86_64 with AVX2 + FMA, a contiguous tensor of `f32`/`f64`, and
    >= 256 elements; otherwise the ordinary path runs.

- **Blocked reductions** (`enable_blocked_reductions`):
  - Keeps 8 independent accumulators so float addition's latency chain stops
    being the bottleneck; blocks are then spread over rayon.
  - `sum`, `mean`, `prod`, `max`, `min` over >= 1024 contiguous elements.
  - Measured **~9x** on 4M f64 (11.5 ms -> 1.35 ms).
  - Deterministic: the result does not depend on the thread count. It is *not*
    bit-identical to naive left-to-right accumulation (a different summation
    order rounds differently); set `enable_blocked_reductions = false` for the
    exact naive order.

- **Thread count** (`CpuExecutor::with_threads(n)`):
  - Builds a private rayon pool of exactly `n` threads and installs the parallel
    element-wise, scalar, binary and reduction regions into it, so `n` is a real
    bound. `n = 0` uses rayon's ambient global pool.
  - `effective_num_threads()` reports the count that will actually be used.
  - The einsum/GEMM contraction path is not routed through this pool and still
    uses the ambient one.

- **Broadcasting**: binary ops broadcast via stride-0 views rather than
  reconstructing subscripts per element (measured 1010 ms -> 55 ms, **18x**, on an
  8.4M-element `(512,1,64) + (512,256,64)` add). This is unconditional — there is
  no knob to turn it off, because there is no reason to.

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
    /// Compute only the output cells this boolean mask selects.
    pub mask: Option<MaskPack>,
    /// The same, by flat index — O(selected) memory instead of O(output).
    pub subset: Option<SubsetSpec>,
    /// Route through the sparse (masked) engine. Required by `mask`/`subset`.
    pub prefer_sparse: bool,
}
```

Every field is read by the executor. `mask` and `subset` are two spellings of
one selection, so supplying both is an error rather than a silent precedence
rule.

The dense GEMM engine has no tunable tile size to expose: `f32`/`f64` are routed
to `matrixmultiply`, which does its own register/cache blocking and beats the
portable blocked kernel by 4.8-6.1x on this workload. There is deliberately no
`tile_kb` knob, because honouring one would mean abandoning the faster kernel.

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

**Status:** Stable | **Version:** 0.1.0 | **Tests:** 244/244 passing
