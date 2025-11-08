# Claude Code Guide for TenRSo

> **Project:** TenRSo — Tensor Computing Stack for COOLJAPAN
> **Purpose:** Production-grade tensor operations with generalized contraction, decompositions, and out-of-core processing
> **Last Updated:** 2025-11-03

---

## 1. Project Overview

TenRSo is a **Rust-native tensor computing stack** providing:
- **Generalized contraction + planner** with cost-based optimization
- **Sparse/low-rank mixed execution** (Dense ⇄ Sparse ⇄ CP/Tucker/TT)
- **Tensor decompositions** (CP-ALS, Tucker-HOOI, TT-SVD)
- **Out-of-core processing** via Arrow/Parquet/mmap
- **AD-ready** with custom VJP/grad rules

**Consumers:** Tensorlogic (runtime backend), scientific/HPC apps, ML systems, quantum/tensor-network research

---

## 2. Build & Test Commands

### Quick Start

```bash
# Build entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Format code
cargo fmt --all

# Lint with clippy
cargo clippy --workspace -- -D warnings

# Check without building
cargo check --workspace

# Build release
cargo build --workspace --release
```

### CI/CD Compliance

```bash
# Run full CI checks locally
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps --all-features
```

### Feature-Gated Builds

```bash
# Build with all features
cargo build --workspace --all-features

# Build without OoC support
cargo build --workspace --no-default-features

# Build with CSF sparse format
cargo build -p tenrso-sparse --features csf

# Build with parallel execution
cargo build -p tenrso-kernels --features parallel
```

---

## 3. Architecture

TenRSo uses a modular Cargo workspace with 8 specialized crates:

### Core Infrastructure

- **`tenrso-core`** - Dense tensors, axis metadata, views, unfold/fold, reshape/permute
- **`tenrso-exec`** - Unified execution API (`einsum_ex`), backend orchestration

### Computation Engines

- **`tenrso-kernels`** - Khatri-Rao, Kronecker, Hadamard, n-mode products, MTTKRP
- **`tenrso-decomp`** - CP-ALS, Tucker-HOOI/HOSVD, TT-SVD, reconstruction
- **`tenrso-sparse`** - COO/CSR/BCSR, CSF/HiCOO, masked einsum, SpMM/SpSpMM

### Optimization & Scalability

- **`tenrso-planner`** - Contraction order search, representation selection, tiling
- **`tenrso-ooc`** - Arrow/Parquet I/O, mmap, chunked execution, streaming

### Automatic Differentiation

- **`tenrso-ad`** - Custom VJP/grad rules for contractions and decompositions

---

## 4. Critical Dependency: SciRS2-Core

**⚠️ MANDATORY:** TenRSo **MUST** use `scirs2-core` for all scientific computing operations.

### 4.1 Golden Rules

1. **NEVER** import `ndarray` directly
   ```rust
   // ❌ FORBIDDEN
   use ndarray::Array;

   // ✅ REQUIRED
   use scirs2_core::ndarray_ext::Array;
   ```

2. **NEVER** import `rand` directly
   ```rust
   // ❌ FORBIDDEN
   use rand::thread_rng;

   // ✅ REQUIRED
   use scirs2_core::random::thread_rng;
   ```

3. **ALWAYS** use SciRS2's:
   - SIMD operations for tensor kernels
   - GPU abstractions (when available)
   - Memory management for large tensors
   - Profiling/benchmarking utilities

4. **Test arrays:** Use `scirs2_core::ndarray_ext` for test data
   ```rust
   // ✅ Use scirs2_core for test arrays
   #[cfg(test)]
   use scirs2_core::ndarray_ext::{array, Array};
   ```

### 4.2 SciRS2-Core Capabilities

**Array Operations:**
- Multi-dimensional arrays via `ndarray_ext`
- Views, slicing, broadcasting
- Element-wise operations with SIMD
- Parallel iteration

**Random Number Generation:**
- Statistical distributions (`Uniform`, `Normal`, etc.)
- Reproducible seeding
- Thread-safe RNG

**Linear Algebra:**
- SVD, QR, eigendecomposition via `scirs2-linalg`
- Matrix factorizations
- Least-squares solvers

**Performance:**
- SIMD acceleration (AVX2, AVX-512)
- Parallel execution via Rayon
- GPU support (feature-gated)

**Memory Management:**
- Memory-mapped arrays
- Chunked allocation
- Buffer pools

**Advanced:**
- Complex numbers
- Scientific constants
- Profiling hooks

---

## 5. Module-Specific SciRS2 Usage

### 5.1 `tenrso-core`

**Purpose:** Dense tensor foundation, axis metadata, views

**SciRS2 Integration:**
```rust
use scirs2_core::ndarray_ext::{Array, ArrayView, Axis, Ix, IxDyn, s};
use scirs2_core::random::{Rng, thread_rng};

pub struct DenseND<T> {
    data: Array<T, IxDyn>,
    axes: Vec<AxisMeta>,
}

impl<T> DenseND<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        let data = Array::zeros(IxDyn(shape));
        // ...
    }
}
```

**Key Operations:**
- Tensor creation/initialization
- Reshape/permute using `ndarray_ext` methods
- Unfold/fold for mode-n operations
- Memory-efficient views

### 5.2 `tenrso-kernels`

**Purpose:** Fast tensor kernels (Khatri-Rao, Kronecker, MTTKRP)

**SciRS2 Integration:**
```rust
use scirs2_core::ndarray_ext::{Array2, ArrayView2, Zip};
use scirs2_core::parallel::par_azip;

pub fn khatri_rao<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
    // Use SIMD-accelerated operations
    // Use parallel iteration for large matrices
}
```

**Key Operations:**
- SIMD-optimized element-wise products
- Parallel Khatri-Rao products
- Cache-friendly MTTKRP
- Blocked matrix operations

### 5.3 `tenrso-decomp`

**Purpose:** CP-ALS, Tucker-HOOI, TT-SVD decompositions

**SciRS2 Integration:**
```rust
use scirs2_linalg::{SVD, QR};
use scirs2_optimize::LeastSquares;
use scirs2_core::ndarray_ext::Array2;

pub fn tucker_hosvd<T>(
    tensor: &Array<T, IxDyn>,
    ranks: &[usize],
) -> Result<(Array<T, IxDyn>, Vec<Array2<T>>)> {
    // Unfold tensor along each mode
    // Compute SVD using scirs2_linalg
    let (u, _s, _vt) = unfolded.svd(true, true)?;
    // ...
}
```

**Key Operations:**
- SVD for Tucker/TT decompositions
- QR for orthogonal factor updates
- Least-squares for CP-ALS
- Convergence criteria via optimization

### 5.4 `tenrso-sparse`

**Purpose:** Sparse tensor formats (COO/CSR/BCSR/CSF)

**SciRS2 Integration:**
```rust
use scirs2_sparse::{CsrMatrix, CooMatrix};
use scirs2_core::ndarray_ext::Array;

pub struct SparseND<T> {
    // Use SciRS2 sparse formats where applicable
    // Extend to N-D with CSF/HiCOO
}
```

**Key Operations:**
- COO/CSR via `scirs2_sparse`
- Custom CSF/HiCOO for N-D sparsity
- Sparse-dense conversions
- Masked operations

### 5.5 `tenrso-planner`

**Purpose:** Contraction order planning, representation selection

**SciRS2 Integration:**
```rust
use scirs2_core::ndarray_ext::Array;
// Minimal SciRS2 usage - mostly graph algorithms
```

**Key Operations:**
- Cost model calculations
- Statistics gathering (nnz, density)
- Tiling size estimation

### 5.6 `tenrso-ooc`

**Purpose:** Out-of-core processing via Arrow/Parquet/mmap

**SciRS2 Integration:**
```rust
use scirs2_core::ndarray_ext::Array;
use scirs2_core::memory::MemoryMappedArray;
```

**Key Operations:**
- Memory-mapped tensor views
- Chunk buffer management
- Streaming with back-pressure

### 5.7 `tenrso-exec`

**Purpose:** Unified execution API (`einsum_ex`)

**SciRS2 Integration:**
```rust
use scirs2_core::ndarray_ext::Array;
use scirs2_core::parallel::ThreadPool;

pub trait TenrsoExecutor<T> {
    fn einsum(
        &mut self,
        spec: &str,
        inputs: &[TensorHandle<T>],
        hints: &ExecHints,
    ) -> Result<TensorHandle<T>>;
}
```

**Key Operations:**
- Parallel execution orchestration
- Memory pool management
- Device abstraction (CPU/GPU)

### 5.8 `tenrso-ad`

**Purpose:** Custom VJP/grad rules for AD

**SciRS2 Integration:**
```rust
use scirs2_core::ndarray_ext::Array;
// Custom gradient implementations
// All operations use scirs2_core
```

**Key Operations:**
- Efficient backward passes
- Gradient computation for decompositions
- Hooks for external AD frameworks

---

## 6. Testing Strategy

### 6.1 Unit Tests

Each module has dedicated `#[test]` functions:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_khatri_rao() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let result = khatri_rao(&a.view(), &b.view());
        // assertions...
    }
}
```

### 6.2 Integration Tests

Place in `tests/` directories:

```rust
// crates/tenrso-exec/tests/einsum_integration.rs
use tenrso_core::TensorHandle;
use tenrso_exec::{einsum_ex, ExecHints};

#[test]
fn test_einsum_matmul() {
    // End-to-end einsum test
}
```

### 6.3 Property Tests

Use `proptest` or `quickcheck` for mathematical properties:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn cp_decomp_reconstruction_error(rank in 1..100usize) {
        // Verify CP reconstruction quality
    }
}
```

### 6.4 Benchmarks

Place in `benches/` directories:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_mttkrp(c: &mut Criterion) {
    c.bench_function("mttkrp_256x256x256", |b| {
        // benchmark code
    });
}

criterion_group!(benches, bench_mttkrp);
criterion_main!(benches);
```

---

## 7. Performance Targets

Refer to `blueprint.md` section 7 for specific targets:

- **Einsum contraction**: ≥ 80% of OpenBLAS baseline
- **Masked operations**: ≥ 5× speedup vs dense naive
- **CP-ALS**: < 2s / 10 iters (256³, rank-64)
- **Tucker-HOOI**: < 3s / 10 iters (512×512×128)
- **TT-SVD**: < 2s build time (32⁶)

---

## 8. Code Quality Standards

### 8.1 No Warnings Policy

All code **MUST** compile without warnings:

```rust
#![deny(warnings)]
```

### 8.2 Documentation

All public APIs **MUST** have:
- Rustdoc comments with `///`
- Examples that compile (`#[doc(test)]`)
- Algorithmic complexity notes

### 8.3 Error Handling

Use `Result<T, Error>` over panics:

```rust
use anyhow::Result;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DecompError {
    #[error("SVD failed: {0}")]
    SvdFailed(String),
}
```

### 8.4 Safety

- **Minimize `unsafe`** code
- **Fuzz all `unsafe` blocks**
- **Bounds-check all indexing**

### 8.5 Determinism

- Document tie-breaking strategies
- Use fixed seeds for stochastic initialization
- Ensure reproducible results

---

## 9. Common Patterns

### 9.1 Creating Dense Tensors

```rust
use scirs2_core::ndarray_ext::Array;
use tenrso_core::{TensorHandle, AxisMeta};

let data = Array::zeros(vec![10, 20, 30]);
let axes = vec![
    AxisMeta::new("batch", 10),
    AxisMeta::new("height", 20),
    AxisMeta::new("width", 30),
];
let tensor = TensorHandle::from_dense(data, axes);
```

### 9.2 Einsum Execution

```rust
use tenrso_exec::{einsum_ex, ExecHints};

let result = einsum_ex::<f32>("bij,bjk->bik")
    .inputs(&[tensor_a, tensor_b])
    .hints(&ExecHints {
        prefer_sparse: true,
        tile_kb: Some(512),
        ..Default::default()
    })
    .run()?;
```

### 9.3 CP Decomposition

```rust
use tenrso_decomp::cp_als;

let cp = cp_als(
    &dense_tensor,
    rank = 64,
    iters = 50,
    tol = 1e-4,
    nonneg = false,
)?;
```

---

## 10. Roadmap Awareness

Understand the 16-20 week development plan in `ROADMAP.md`:

- **M0 (Week 0-1):** Repo hygiene, core dense tensor ← **YOU ARE HERE**
- **M1 (Week 2-4):** Kernels (Khatri-Rao, MTTKRP)
- **M2 (Week 5-6):** Decompositions (CP, Tucker, TT)
- **M3 (Week 7-8):** Sparse formats + masked einsum
- **M4 (Week 9-10):** Planner v0
- **M5 (Week 11-12):** Out-of-core v0
- **M6 (Week 13-16):** AD hooks

---

## 11. When in Doubt

1. **Check `blueprint.md`** for architectural decisions
2. **Check `SCIRS2_INTEGRATION_POLICY.md`** for dependency rules
3. **Check `CONTRIBUTING.md`** for PR process
4. **Ask @cool-japan maintainers** via GitHub issues

---

## 12. Quick Reference Card

```rust
// ✅ ALWAYS DO THIS
use scirs2_core::ndarray_ext::{Array, array};  // array! macro included
use scirs2_core::random::thread_rng;
use scirs2_linalg::SVD;

// ❌ NEVER DO THIS
use ndarray::Array;  // FORBIDDEN
use rand::thread_rng; // FORBIDDEN

// 📝 DOCUMENT EVERYTHING
/// Computes the Khatri-Rao product of two matrices.
///
/// # Complexity
/// O(n * m * p) where n, m, p are matrix dimensions
pub fn khatri_rao<T>(...) -> Array2<T> { ... }

// 🎯 ERROR HANDLING
use anyhow::Result;
pub fn decompose(...) -> Result<TensorHandle<T>> { ... }

// 🚀 PERFORMANCE
#![deny(warnings)]
// Write fast, safe, tested code
```

---

**This guide is your companion for maintaining TenRSo's production-grade quality.**
