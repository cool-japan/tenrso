# tenrso-ad

Automatic differentiation support for TenRSo tensor operations.

## Overview

`tenrso-ad` provides efficient gradient computation for tensor operations:

- **Custom VJP rules** - Vector-Jacobian Products for tensor contractions
- **Decomposition gradients** - Backprop through CP, Tucker, TT
- **Integration hooks** - Connect to external AD frameworks (Tensorlogic)
- **Efficient backprop** - Avoid AD tape blow-up with custom rules

## Features

- Hand-written gradients for tensor primitives
- Efficient backward passes using contraction planning
- Memory-efficient gradient accumulation
- Integration with external AD systems

## Usage

```toml
[dependencies]
tenrso-ad = "0.1"
```

### Custom VJP (TODO: M6)

```rust
use tenrso_ad::vjp_einsum;

// Forward pass
let y = einsum("ijk,jkl->il", &[x1, x2])?;

// Backward pass with cotangent
let grad_y = Array::ones(y.raw_dim());
let grads = vjp_einsum("ijk,jkl->il", &[x1, x2], &grad_y)?;

assert_eq!(grads.len(), 2);  // Gradients for x1 and x2
```

### Decomposition Gradients (TODO: M6)

```rust
use tenrso_ad::cp_als_vjp;

// Forward: CP decomposition
let cp = cp_als(&tensor, rank=64)?;

// Backward: gradient w.r.t. tensor
let grad_tensor = cp_als_vjp(&cp, &grad_output)?;
```

## API Reference

### VJP for Contractions

```rust
pub fn vjp_einsum<T>(
    spec: &str,
    inputs: &[&Array<T, IxDyn>],
    cotangent: &Array<T, IxDyn>,
) -> Result<Vec<Array<T, IxDyn>>>
```

### Decomposition Gradients

```rust
pub fn cp_als_vjp<T>(...) -> Result<Array<T, IxDyn>>;
pub fn tucker_hooi_vjp<T>(...) -> Result<Array<T, IxDyn>>;
pub fn tt_svd_vjp<T>(...) -> Result<Vec<Array<T, IxDyn>>>;
```

## External Integration

### Tensorlogic Hooks (TODO: M6)

```rust
use tenrso_ad::TensorlogicAdapter;

// Register TenRSo operations with Tensorlogic
let adapter = TensorlogicAdapter::new();
adapter.register_einsum_op()?;
adapter.register_decomp_ops()?;
```

## Performance

- Gradient computation uses same planner as forward pass
- Memory-efficient: gradients computed in optimal order
- Parallelized gradient accumulation

## Dependencies

- **tenrso-core** - Tensor types
- **tenrso-exec** - Execution interface
- **tenrso-decomp** - Decomposition operations
- **scirs2-core** - Array operations

## License

Apache-2.0
