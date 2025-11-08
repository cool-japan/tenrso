# SciRS2 Integration Policy for TenRSo

> **Status:** Active
> **Last Updated:** 2025-11-03
> **Owners:** @cool-japan

---

## 1. Core Mandate

TenRSo is a production-grade tensor computing stack that **must use SciRS2 as its scientific computing foundation**. This policy establishes clear guidelines for when and how to integrate SciRS2 crates into TenRSo.

### 1.1 Architectural Hierarchy

```
TenRSo (tensor computing)
    ↓ uses
SciRS2 (scientific computing primitives)
    ↓ builds on
Core Rust Scientific Libs (ndarray, nalgebra, etc.)
```

**Critical Rule:** TenRSo **NEVER** directly imports `ndarray`, `rand`, `nalgebra`, or other low-level scientific libraries. All numerical operations **MUST** route through SciRS2 equivalents.

---

## 2. Core Principles

### 2.1 Foundation, Not Dependency Bloat

- Add SciRS2 crates **only when actually needed** by TenRSo functionality
- Each integration must demonstrate **concrete usage** in TenRSo's tensor operations
- Avoid speculative "we might need this later" dependencies

### 2.2 Evidence-Based Integration

Every SciRS2 crate addition requires:
1. **Clear justification** tied to specific TenRSo features
2. **Documented use cases** in tensor computing workflows
3. **Code examples** showing how TenRSo will consume the crate
4. **Performance rationale** (why SciRS2 vs. implementing directly)

### 2.3 Maintain Separation of Concerns

- **SciRS2** = general scientific computing primitives
- **TenRSo** = tensor-specific operations (einsum, decompositions, sparse tensors)
- TenRSo builds **on top of** SciRS2, not alongside it

---

## 3. Essential Dependencies (Always Required)

### 3.1 `scirs2-core`

**Status:** ✅ MANDATORY
**Why:** Foundation for all numerical work in TenRSo

**Usage:**
- Dense tensor storage via `scirs2_core::ndarray_ext`
- Random initialization via `scirs2_core::random`
- SIMD operations for tensor kernels
- Memory management for large tensors

**Critical Pattern:**
```rust
// ❌ WRONG - never import ndarray directly
use ndarray::Array;

// ✅ CORRECT - always use scirs2_core
use scirs2_core::ndarray_ext::Array;
```

**Prohibited Direct Dependencies:**
```rust
// ❌ FORBIDDEN - Direct external dependencies
use ndarray::Array;        // Use scirs2_core::ndarray_ext instead
use rand::thread_rng;      // Use scirs2_core::random instead
use num_traits::Num;       // Use scirs2_core::numeric instead
use num_complex::Complex;  // Use scirs2_core (root) instead
```

### 3.2 `scirs2-linalg`

**Status:** ✅ MANDATORY
**Why:** Required for tensor decompositions (CP, Tucker, TT)

**Usage:**
- SVD for Tucker-HOSVD and TT-SVD
- QR decomposition for CP-ALS
- Matrix operations in MTTKRP
- Eigenvalue computations

### 3.3 `scirs2` (meta-crate)

**Status:** ✅ MANDATORY
**Why:** Unified access to core scientific computing capabilities

---

## 4. Highly Likely Required

These crates will almost certainly be needed for TenRSo's core functionality:

| Crate | Justification | Status |
|-------|---------------|--------|
| `scirs2-optimize` | Nonlinear optimization for CP-ALS, Tucker-HOOI | Pending |
| `scirs2-sparse` | Sparse tensor formats (COO/CSR/CSF) | Pending |
| `scirs2-stats` | Statistical operations in decompositions | Pending |
| `scirs2-special` | Special functions (gamma, beta) for initialization | Pending |
| `scirs2-fft` | FFT-based tensor operations | Pending |
| `scirs2-parallel` | Parallel tensor contractions | Pending |

---

## 5. Conditionally Required

These crates may be needed based on specific TenRSo features:

| Crate | Condition | Priority |
|-------|-----------|----------|
| `scirs2-cluster` | If implementing distributed tensor operations | Low |
| `scirs2-graph` | If tensor network graph optimization | Medium |
| `scirs2-metrics` | If built-in performance metrics | Low |
| `scirs2-signal` | If tensor-based signal processing | Low |

---

## 6. Likely Not Required

These crates are probably not relevant to TenRSo's tensor computing focus:

- `scirs2-image` - Image processing (use TenRSo tensors directly)
- `scirs2-nlp` - NLP-specific operations
- `scirs2-vision` - Computer vision
- `scirs2-audio` - Audio processing
- `scirs2-io` - Generic I/O (TenRSo has specialized Arrow/Parquet I/O)

---

## 7. Integration Workflow

### 7.1 Before Adding a SciRS2 Crate

1. **Document the need** in a GitHub issue
   - Which TenRSo feature requires this?
   - What specific SciRS2 functionality will be used?
   - Why can't we implement this directly in TenRSo?

2. **Provide code examples** showing actual usage

3. **Get approval** from @cool-japan maintainers

### 7.2 After Adding a SciRS2 Crate

1. **Update this policy** with justification and usage patterns
2. **Document in relevant crate READMEs** how SciRS2 is used
3. **Add tests** demonstrating the integration
4. **Update CI/CD** to prevent direct low-level imports

---

## 8. Enforcement Mechanisms

### 8.1 Prohibited Direct Dependencies

**❌ FORBIDDEN in ALL TenRSo Crates:**

The following dependencies **MUST NEVER** appear in any `Cargo.toml` under `crates/`:

```toml
# ❌ POLICY VIOLATIONS - Use scirs2-core instead
rand = { workspace = true }              # Use scirs2_core::random
rand_distr = { workspace = true }        # Use scirs2_core::random
rand_core = { workspace = true }         # Use scirs2_core::random
rand_chacha = { workspace = true }       # Use scirs2_core::random
ndarray = { workspace = true }           # Use scirs2_core::ndarray_ext
ndarray-rand = { workspace = true }      # Use scirs2_core::ndarray_ext
ndarray-stats = { workspace = true }     # Use scirs2_core::ndarray_ext
num-traits = { workspace = true }        # Use scirs2_core::numeric
num-complex = { workspace = true }       # Use scirs2_core (root)
num-integer = { workspace = true }       # Use scirs2_core::numeric
nalgebra = { workspace = true }          # Use scirs2_linalg
rayon = { workspace = true }             # Use scirs2_core::parallel_ops
```

### 8.2 CI/CD Checks

```bash
# Forbidden patterns that will fail CI
grep -r "^use ndarray::" crates/     # Must fail
grep -r "^use rand::" crates/        # Must fail
grep -r "^use num_traits::" crates/  # Must fail
grep -r "^use num_complex::" crates/ # Must fail
grep -r "^use nalgebra::" crates/    # Must fail
grep -r "^use rayon::" crates/       # Must fail (use scirs2_core::parallel_ops)

# Check Cargo.toml files
find crates/ -name "Cargo.toml" -exec grep -l "^rand = " {} \;          # Must be empty
find crates/ -name "Cargo.toml" -exec grep -l "^ndarray = " {} \;       # Must be empty
find crates/ -name "Cargo.toml" -exec grep -l "^num-traits = " {} \;    # Must be empty
find crates/ -name "Cargo.toml" -exec grep -l "^num-complex = " {} \;   # Must be empty
find crates/ -name "Cargo.toml" -exec grep -l "^nalgebra = " {} \;      # Must be empty
find crates/ -name "Cargo.toml" -exec grep -l "^rayon = " {} \;         # Must be empty
```

### 8.3 Code Review Requirements

Every PR that adds a SciRS2 dependency must:
- [ ] Include justification in this policy document
- [ ] Show concrete usage in TenRSo code
- [ ] Pass CI checks for forbidden direct imports
- [ ] Include tests using the new SciRS2 functionality

### 8.4 Quarterly Audits

Review all SciRS2 dependencies and remove any that:
- Have zero usage in TenRSo codebase
- Can be replaced with direct TenRSo implementations
- Are no longer maintained upstream

---

## 9. Critical Usage Patterns

### 9.1 Array Operations (Production Code)

```rust
// ✅ CORRECT - production tensor operations
use scirs2_core::ndarray_ext::{Array, ArrayView, s};

pub struct DenseND<T> {
    data: Array<T, ndarray::IxDyn>,
}
```

### 9.2 Array Operations (Test Code)

```rust
// ✅ CORRECT - use scirs2_core for all tests
use scirs2_core::ndarray_ext::array;

#[test]
fn test_tensor_add() {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    // ...
}
```

### 9.3 Random Number Generation

```rust
// ✅ CORRECT - use SciRS2 random
use scirs2_core::random::{Rng, thread_rng, distributions::Uniform};

let mut rng = thread_rng();
let init_values: Vec<f64> = (0..rank)
    .map(|_| rng.sample(Uniform::new(-0.1, 0.1)))
    .collect();
```

### 9.4 Linear Algebra

```rust
// ✅ CORRECT - use SciRS2 linalg
use scirs2_linalg::{SVD, QR};

// Tucker decomposition using SVD
let (u, s, vt) = unfolded_matrix.svd(true, true)?;
```

### 9.5 Numeric Traits

```rust
// ✅ CORRECT - use SciRS2 numeric
use scirs2_core::numeric::{Num, Float, NumAssign, FloatConst};

// Tensor operations with generic numeric types
pub fn tensor_op<T>(x: &DenseND<T>) -> Result<T>
where
    T: Float + NumAssign,
{
    // Implementation
}

// ❌ WRONG - direct num_traits import
use num_traits::{Num, Float};  // FORBIDDEN
```

### 9.6 Parallel Processing (Rayon)

```rust
// ✅ CORRECT - use scirs2_core::parallel_ops
use scirs2_core::parallel_ops::*;

// Parallel iteration with Rayon
let results: Vec<f64> = (0..1000)
    .into_par_iter()
    .map(|x| expensive_computation(x))
    .collect();

// Parallel matrix operations
use scirs2_core::parallel_ops::{ThreadPoolBuilder, ThreadPool};

let pool = ThreadPoolBuilder::new()
    .num_threads(8)
    .build()
    .unwrap();

pool.install(|| {
    // Parallel work here
});

// ❌ WRONG - direct rayon import
use rayon::prelude::*;        // FORBIDDEN
use rayon::ThreadPoolBuilder; // FORBIDDEN
```

### 9.7 SIMD Operations

```rust
// ✅ CORRECT - use scirs2_core SIMD modules
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::simd::*;

// SIMD-accelerated element-wise operations
let a = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let b = array![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

// Automatic SIMD acceleration based on CPU capabilities
let result = f64::simd_add(&a.view(), &b.view());

// Platform capability detection
use scirs2_core::simd_ops::PlatformCapabilities;
let caps = PlatformCapabilities::detect();
if caps.avx2_available {
    // Use AVX2-optimized path
}

// ❌ WRONG - manual SIMD intrinsics
// Don't use std::arch directly unless absolutely necessary
// use std::arch::x86_64::*;  // Avoid unless critical
```

### 9.8 GPU Acceleration

```rust
// ✅ CORRECT - use scirs2_core::gpu
#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuContext};

#[cfg(feature = "gpu")]
pub fn gpu_accelerated_operation(data: &Array2<f64>) -> Result<Array2<f64>> {
    // Automatic backend selection (CUDA, Metal, OpenCL, WebGPU)
    let backend = GpuBackend::preferred();

    if backend.is_available() {
        // Use GPU acceleration
        let ctx = GpuContext::new(backend)?;
        ctx.execute_kernel(data)
    } else {
        // CPU fallback
        cpu_operation(data)
    }
}

// ❌ WRONG - direct GPU framework imports
// use cudarc::*;     // FORBIDDEN
// use metal::*;      // FORBIDDEN
// use wgpu::*;       // FORBIDDEN (use scirs2_core::gpu instead)
```

---

## 10. Complete Dependency Mapping

**External Crate → SciRS2 Module → Usage in TenRSo**

| External Crate | SciRS2 Module | TenRSo Usage |
|----------------|---------------|--------------|
| `ndarray` | `scirs2_core::ndarray_ext` | All array operations |
| `rand` | `scirs2_core::random` | All RNG operations |
| `num-traits` | `scirs2_core::numeric` | All numeric traits (Num, Float, etc.) |
| `num-complex` | `scirs2_core` (root) | Complex numbers |
| `nalgebra` | `scirs2_linalg` | Linear algebra operations |
| `rayon` | `scirs2_core::parallel_ops` | Parallel iteration, ThreadPool |
| SIMD intrinsics | `scirs2_core::simd`, `scirs2_core::simd_ops` | Vectorized operations, platform detection |
| GPU frameworks | `scirs2_core::gpu` | GPU acceleration (CUDA, Metal, OpenCL, WebGPU) |

**Note on Performance Features:**
- **Rayon**: Always use `scirs2_core::parallel_ops::*` for parallel iteration
- **SIMD**: Use `scirs2_core::simd_ops::SimdUnifiedOps` for automatic SIMD acceleration
- **GPU**: Use `scirs2_core::gpu::GpuBackend::preferred()` for automatic backend selection

---

## 10. Module-Specific Guidance

### 10.1 `tenrso-core`

**SciRS2 Usage:**
- `scirs2_core::ndarray_ext` for dense tensor storage
- `scirs2_core::random` for tensor initialization
- `scirs2_core::simd`, `scirs2_core::simd_ops` for fast element-wise ops

### 10.2 `tenrso-kernels`

**SciRS2 Usage:**
- `scirs2_linalg` for matrix-based kernels
- `scirs2_core::parallel_ops` for parallelized MTTKRP
- `scirs2_core::simd_ops::SimdUnifiedOps` for Khatri-Rao/Kronecker products

### 10.3 `tenrso-decomp`

**SciRS2 Usage:**
- `scirs2_linalg::SVD` for Tucker/TT decompositions
- `scirs2_optimize` for ALS iteration convergence
- `scirs2_linalg::QR` for CP-ALS factor updates

### 10.4 `tenrso-sparse`

**SciRS2 Usage:**
- `scirs2_sparse` for COO/CSR format operations
- `scirs2_core::ndarray_ext` for dense-sparse conversions

### 10.5 `tenrso-planner`

**SciRS2 Usage:**
- `scirs2_graph` (if using graph-based planning)
- `scirs2_optimize` for cost minimization

### 10.6 `tenrso-ooc`

**SciRS2 Usage:**
- Minimal - use Arrow/Parquet directly
- `scirs2_core` for chunk buffers

### 10.7 `tenrso-exec`

**SciRS2 Usage:**
- `scirs2_core::parallel_ops` for multi-threaded execution
- `scirs2_core::memory` for buffer management
- `scirs2_core::gpu` (optional) for GPU-accelerated execution

### 10.8 `tenrso-ad`

**SciRS2 Usage:**
- Custom VJP rules using `scirs2_core`
- All gradient operations via `scirs2_core::ndarray_ext`

---

## 11. Decision Log

### 2025-11-03: Initial Policy

- Established mandatory use of `scirs2-core` and `scirs2-linalg`
- Defined evidence-based integration workflow
- Set up CI/CD enforcement mechanisms

### 2025-11-03: Complete Policy Enforcement

**Context:** Discovered residual `num-traits` direct dependencies after initial policy creation

**Actions Taken:**
1. **Removed all `num-traits` dependencies** from 4 crates:
   - `tenrso-core/Cargo.toml`
   - `tenrso-kernels/Cargo.toml`
   - `tenrso-decomp/Cargo.toml`
   - `tenrso-sparse/Cargo.toml`

2. **Replaced all imports** throughout codebase:
   - `use num_traits::{Num, Float, ...}` → `use scirs2_core::numeric::{Num, Float, ...}`
   - Updated 10+ source files across all crates

3. **Upgraded to Rust 1.90** to resolve bincode 2.0.1 compatibility

4. **Upgraded Arrow 53 → 57** for Rust 1.90 compatibility

5. **Added comprehensive forbidden dependency list** based on QuantRS2 policy:
   - rand, rand_distr, rand_core, rand_chacha
   - ndarray, ndarray-rand, ndarray-stats
   - num-traits, num-complex, num-integer
   - nalgebra

6. **Validated compliance:** All 102 tests passing with zero forbidden imports

**Key Lessons:**
- QuantRS2's SCIRS2_INTEGRATION_POLICY.md provides the definitive forbidden dependency list
- `scirs2_core::numeric` provides complete re-export of `num_traits`
- Incremental policy enforcement leads to incomplete compliance
- Must audit **all** crates systematically

**Status:** ✅ Full SciRS2 policy compliance achieved

### 2025-11-04: Performance Features Policy Declaration

**Context:** Added explicit guidance for Rayon, SIMD, and GPU acceleration via scirs2-core

**Actions Taken:**
1. **Added Rayon to forbidden dependencies list**
   - Must use `scirs2_core::parallel_ops::*` instead of direct `rayon` imports
   - Re-exports rayon::prelude, ThreadPool, ThreadPoolBuilder

2. **Documented SIMD usage patterns**
   - Must use `scirs2_core::simd` and `scirs2_core::simd_ops` modules
   - Provides `SimdUnifiedOps` trait for automatic SIMD acceleration
   - Platform capability detection via `PlatformCapabilities::detect()`

3. **Documented GPU acceleration patterns**
   - Must use `scirs2_core::gpu` module (feature-gated)
   - Provides unified interface for CUDA, Metal, OpenCL, WebGPU
   - Automatic backend selection via `GpuBackend::preferred()`

4. **Updated module-specific guidance**
   - `tenrso-core`: Uses SIMD for element-wise ops
   - `tenrso-kernels`: Uses parallel_ops and simd_ops for tensor kernels
   - `tenrso-exec`: Uses parallel_ops and optional GPU acceleration

5. **Added CI/CD enforcement**
   - Added check for direct `rayon::` imports
   - Added check for `rayon` in Cargo.toml files

**Key Principles:**
- **Rayon**: Always use `scirs2_core::parallel_ops` for parallel iteration
- **SIMD**: Use `scirs2_core::simd_ops::SimdUnifiedOps` for vectorized ops
- **GPU**: Use `scirs2_core::gpu` with automatic backend selection
- All performance features route through scirs2-core for consistency

**Status:** ✅ Performance features policy fully documented

---

## 12. Contact & Questions

For questions about this policy:
- Open a GitHub issue with label `scirs2-integration`
- Tag @cool-japan maintainers
- Reference this policy document in discussions

**This is a living document.** Update it as TenRSo's needs evolve.
