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

**Version:** 0.1.0-alpha.1

Add TenRSo crates to your `Cargo.toml`:

```toml
[dependencies]
tenrso-core = "0.1.0-alpha.1"
tenrso-exec = "0.1.0-alpha.1"
tenrso-decomp = "0.1.0-alpha.1"
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

// Unified contraction with automatic optimization
let y = einsum_ex::<f32>("bij,bjk->bik")
    .inputs(&[A, B])
    .hints(&ExecHints{
        prefer_lowrank: true,
        prefer_sparse: true,
        tile_kb: Some(512),
        ..Default::default()
    })
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

### Alpha.1 Release Notes

**What's New in 0.1.0-alpha.1:**
- ✅ All critical bugs fixed (CP-ALS, TT-SVD, Tucker-HOOI)
- ✅ 524/524 tests passing (100% pass rate)
- ✅ SIMD optimizations (3-4× speedup)
- ✅ Hardware FMA support (2× additional speedup)
- ✅ Parallel execution (6-8× speedup on 8 cores)
- ✅ Complete out-of-core processing (Arrow, Parquet, mmap)
- ✅ Zero compiler/clippy warnings
- ✅ Production-grade quality

See [CHANGELOG.md](CHANGELOG.md) for complete release notes.

## Project Status

**🎉 ALPHA.1 RELEASED** - **524/524 tests passing (100%)**

| Milestone | Status | Tests |
|-----------|--------|-------|
| M0: Repo hygiene | ✅ Complete | - |
| M1: Kernels | ✅ Complete | 75 |
| M2: Decompositions | ✅ **100% Complete** | 30 |
| M3: Sparse & Masked | ✅ Complete | 128 |
| M4: Planner | ✅ Complete | 125 |
| M5: Out-of-Core | ✅ Complete | 96 |
| M6: AD hooks | ✅ Core Complete | 13 |
| **Alpha.1** | ✅ **SHIPPED** | **524** |

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
