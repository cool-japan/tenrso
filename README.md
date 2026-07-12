# TenRSo — Tensor Computing Stack for COOLJAPAN

> **Mission:** A production-grade, Rust-native tensor stack providing **generalized contraction + planner**, **sparse/low-rank mixed execution**, **decompositions (CP/Tucker/TT)**, and **out-of-core** processing—cleanly separated from SciRS2's matrix-centric linalg.

## Features

- **One API, many representations**: Dense ⇄ Sparse (COO/CSR/BCSR/CSF/HiCOO) ⇄ Low-rank (CP/Tucker/TT) chosen automatically by a planner
- **Generalized contraction with cost-based planning**: Optimal contraction order, tiling, streaming, native masked/subset reductions
- **Decompositions & primitives**: First-class CP-ALS, Tucker-HOOI, TT-SVD, plus Khatri–Rao/Kronecker/Hadamard, n-mode products, MTTKRP
- **Scale beyond memory**: Out-of-core (Parquet/Arrow/mmap), chunked execution, spill-to-disk with deterministic tiles
- **AD-ready**: Custom VJP/grad rules for contraction and decompositions
- **Production discipline**: Strong CI, perf budgets, no-panic kernels, semver stability

## Crates

- `tenrso-core`: Dense tensors, axis metadata, views, unfold/fold, reshape/permute
- `tenrso-kernels`: Khatri–Rao, Kronecker, Hadamard, n-mode products, MTTKRP, TTM/TTT
- `tenrso-decomp`: CP-ALS, Tucker-HOOI, TT-SVD decompositions
- `tenrso-sparse`: COO/CSR/BCSR + CSF/HiCOO, SpMM/SpSpMM, masked-einsum
- `tenrso-planner`: Contraction order, representation selection, tiling/streaming/OoC
- `tenrso-ooc`: Arrow/Parquet readers, chunked/mmap streaming
- `tenrso-exec`: Unified execution API (dense/sparse/low-rank mixing)
- `tenrso-ad`: Custom VJP/grad rules, hooks to external AD

## Installation

**Version:** 0.1.0

Add TenRSo crates to your `Cargo.toml`:

```toml
[dependencies]
tenrso-core = "0.1.0"
tenrso-exec = "0.1.0"
tenrso-decomp = "0.1.0"
```

Or use the workspace in development:

```bash
git clone https://github.com/cool-japan/tenrso.git
cd tenrso
cargo build --workspace
cargo test --workspace
```

## Quick Start

### Basic Tensor Operations

```rust
use tenrso_core::{TensorHandle, AxisMeta};
use scirs2_core::ndarray_ext::Array;

// Create a 3D tensor
let data = Array::zeros(vec![10, 20, 30]);
let axes = vec![
    AxisMeta::new("batch", 10),
    AxisMeta::new("height", 20),
    AxisMeta::new("width", 30),
];
let tensor = TensorHandle::from_dense(data, axes);
```

### Einsum Contractions

```rust
use tenrso_exec::{einsum_ex, ExecHints};

// Unified contraction with automatic optimization.
// f32/f64 dispatch to a native matrixmultiply GEMM, parallel over the batch.
let y = einsum_ex::<f32>("bij,bjk->bik")
    .inputs(&[A, B])
    .hints(&ExecHints::default())
    .run()?;
```

### Tensor Decompositions

```rust
use tenrso_decomp::{cp_als, tucker_hooi, tt_svd};

// CP decomposition
let cp = cp_als(&tensor, rank=64, iters=50, tol=1e-4, nonneg=false)?;

// Tucker decomposition
let tucker = tucker_hooi(&tensor, &[64, 64, 32], 30, 1e-4)?;

// Tensor Train decomposition
let tt = tt_svd(&tensor, 1e-6)?;
```

### Sparse Tensors

```rust
use tenrso_sparse::{Csr, MaskPack};
use tenrso_exec::einsum_ex;

// Create sparse tensor
let sparse = TensorHandle::from_csr(Csr::<f32>::from_triplets(...));

// Masked einsum
let mask = MaskPack::from_indices(...);
let result = einsum_ex::<f32>("ab,bc->ac")
    .inputs(&[sparse, dense])
    .hints(&ExecHints{
        mask: Some(mask),
        prefer_sparse: true,
        ..Default::default()
    })
    .run()?;
```

## Documentation

- [**CHANGELOG**](CHANGELOG.md) - Release notes and version history ✨ NEW
- [Blueprint](blueprint.md) - Architecture and design decisions
- [Roadmap](ROADMAP.md) - Development timeline and milestones
- [Contributing](CONTRIBUTING.md) - Development guidelines and RFC process
- [SciRS2 Integration Policy](SCIRS2_INTEGRATION_POLICY.md) - Dependency guidelines
- [Claude Guide](CLAUDE.md) - Development guide for AI assistance

API documentation: `cargo doc --workspace --no-deps --open`

### Latest Release

**0.1.0 — Stable** (2026-04-14): branch-cut from RC.1 with no code changes vs RC.1 at release time.

**What's New in 0.1.0-rc.1:**
- All milestones M0-M6 complete
- 2,109 tests passing (100% pass rate, 14 skipped)
- TT-SVD gradient backward pass fully implemented (tenrso-ad)
- Masked einsum and subset reductions (tenrso-sparse)
- CP decomposition regularization (L1/L2) and cross-validation rank selection (tenrso-decomp)
- Executor element-wise and reduction operations (tenrso-exec)
- Zero compiler/clippy warnings, zero slow tests (>30s)
- Test suite runtime reduced 4.8x (963s to 198s)

See [CHANGELOG.md](CHANGELOG.md) for complete release notes.

## Project Status

**0.1.0 RELEASED** - 2026-04-14 - Production-Ready Quality

| Crate | Tests | Status |
|-------|-------|--------|
| tenrso-core | 138 | Complete |
| tenrso-kernels | 264 | Complete |
| tenrso-decomp | 165 | Complete |
| tenrso-sparse | 426 | Complete |
| tenrso-planner | 271 | Complete |
| tenrso-ooc | 238 | Complete |
| tenrso-exec | 244 | Complete |
| tenrso-ad | 154 | Complete |
| **Total** | **2,109** | **All milestones M0-M6 complete** |

See [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md) for details.

## Performance Targets

- **Einsum (CPU, float32)** - ≥ 80% of OpenBLAS baseline
- **Masked operations** - ≥ 5× speedup vs dense naive
- **CP-ALS** - < 2s / 10 iters (256³, rank-64)
- **Tucker-HOOI** - < 3s / 10 iters (512×512×128)
- **TT-SVD** - < 2s build time (32⁶)

## Building from Source

```bash
# Clone repository
git clone https://github.com/cool-japan/tenrso.git
cd tenrso

# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Build with all features
cargo build --workspace --all-features

# Run benchmarks
cargo bench --workspace
```

## Feature Flags

- `cpu` (default) - CPU backend
- `simd` - SIMD acceleration
- `gpu` - GPU backend (future)
- `serde` - Serialization support
- `parquet` - Parquet I/O
- `arrow` - Arrow I/O
- `ooc` - Out-of-core processing
- `csf` - CSF/HiCOO sparse formats

## Sponsorship

TenRSo is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find TenRSo useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

Apache-2.0

## Citation

If you use TenRSo in your research, please cite:

```bibtex
@software{tenrso2025,
  title = {TenRSo: Tensor Computing Stack for COOLJAPAN},
  author = {COOLJAPAN Contributors},
  year = {2025},
  url = {https://github.com/cool-japan/tenrso}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Coding standards
- Pull request process
- RFC process for major changes

## Contact

- **Issues:** [GitHub Issues](https://github.com/cool-japan/tenrso/issues)
- **Discussions:** [GitHub Discussions](https://github.com/cool-japan/tenrso/discussions)
- **Owner:** @cool-japan
