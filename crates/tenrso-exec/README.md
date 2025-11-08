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
