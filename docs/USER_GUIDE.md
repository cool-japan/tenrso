# TenRSo User Guide

> Reference guide for TenRSo — a Rust-native tensor computing stack providing
> generalized contraction with cost-based planning, mixed dense/sparse/low-rank
> execution, tensor decompositions (CP/Tucker/TT), out-of-core processing, and
> automatic differentiation.
>
> For hands-on, runnable walkthroughs see [`TUTORIALS.md`](TUTORIALS.md) in this
> same directory. This guide is the reference companion: concepts, APIs, and
> honest performance numbers.

**Workspace version:** 0.1.0 · **Rust edition:** 2021 · **Toolchain:** 1.90.0 (pinned in `rust-toolchain.toml`)

---

## Table of Contents

1. [Installation and Feature Flags](#1-installation-and-feature-flags)
2. [The Crate Map](#2-the-crate-map)
3. [Core Concepts (`tenrso-core`)](#3-core-concepts-tenrso-core)
4. [The Einsum Spec Language and Its Limits](#4-the-einsum-spec-language-and-its-limits)
5. [The Execution Model (`tenrso-exec` + `tenrso-planner`)](#5-the-execution-model-tenrso-exec--tenrso-planner)
6. [Decompositions (`tenrso-decomp`)](#6-decompositions-tenrso-decomp)
7. [Sparse Formats (`tenrso-sparse`)](#7-sparse-formats-tenrso-sparse)
8. [Out-of-Core Processing (`tenrso-ooc`)](#8-out-of-core-processing-tenrso-ooc)
9. [Automatic Differentiation (`tenrso-ad`)](#9-automatic-differentiation-tenrso-ad)
10. [The Mandatory SciRS2-Core Policy](#10-the-mandatory-scirs2-core-policy)
11. [Performance Notes (Honest Numbers)](#11-performance-notes-honest-numbers)
12. [Where to Go Next](#12-where-to-go-next)

---

## 1. Installation and Feature Flags

### 1.1 Adding TenRSo to a project

Pick only the crates you need — there is no requirement to depend on the whole
stack. A typical dense/decomposition project needs three crates:

```toml
[dependencies]
tenrso-core   = "0.1.0"
tenrso-exec   = "0.1.0"
tenrso-decomp = "0.1.0"
```

Or, for convenience, depend on the `tenrso` facade crate, which re-exports every
component crate as a module plus a `prelude`:

```toml
[dependencies]
tenrso = "0.1.0"
```

```rust,ignore
use tenrso::prelude::*; // DenseND, TensorHandle, cp_als, einsum_ex, ...
```

Working from a clone of the workspace itself:

```bash
git clone https://github.com/cool-japan/tenrso.git
cd tenrso
cargo build --workspace
cargo test --workspace
```

### 1.2 Feature flags, crate by crate

Feature names are **not** uniform across crates — always check the crate you
depend on directly. These are the actual flags declared in each crate's
`Cargo.toml` (not the aspirational list in the root `README.md`, which predates
several of these):

| Crate | Default features | Notable optional features |
|---|---|---|
| `tenrso-core` | *(none)* | `parallel` (Rayon via scirs2-core), `serde`, `linalg` (pulls `scirs2-linalg`), `fft` (pulls `scirs2-fft`), `binary` (oxicode-backed serialization), `json` |
| `tenrso-kernels` | `parallel`, `sparse` | `csf` (enables CSF/HiCOO-aware kernels; implies `sparse`) |
| `tenrso-decomp` | `parallel` | `sparse` (enables `cp_als_sparse` and sparse-tensor kernels) |
| `tenrso-sparse` | `parallel` | `csf` (enables the `Csf`/`HiCoo` N-D formats) |
| `tenrso-planner` | *(none)* | `serde` |
| `tenrso-exec` | `ooc` (pulls in `tenrso-ooc`) | — |
| `tenrso-ooc` | `arrow`, `parquet`, `mmap`, `parallel`, `compression`, `lz4-compression`, `zstd-compression`, `tracing`, `lock-free` | `opentelemetry`, `prometheus-metrics`, `jemalloc`, `mimalloc-allocator`, `distributed`, `gpu`-adjacent backends (`cuda`, `rocm`, `metal`, `vulkan`, each default-off) |
| `tenrso-ad` | *(none)* | `tensorlogic` (integration gate; returns a specific error until the external Tensorlogic API stabilizes) |
| `tenrso` (facade) | `ooc` | `ad` (re-exports `tenrso_ad`), `sparse`, `csf`, `full` (all of the above) |

Common combinations:

```bash
# Everything, for exploring the whole stack
cargo build --workspace --all-features

# Sparse CP-ALS + CSF/HiCOO kernels
cargo build -p tenrso-decomp --features sparse
cargo build -p tenrso-sparse --features csf

# Parallel execution (Rayon, routed through scirs2-core)
cargo build -p tenrso-kernels --features parallel

# Minimal dense-only build, no out-of-core dependency chain
cargo build -p tenrso-exec --no-default-features
```

Note: `tenrso-ad` unconditionally depends on `tenrso-exec` with its *default*
features, so if anything in your dependency graph pulls in `tenrso-ad`, the
`ooc` feature (and therefore `tenrso-ooc`) gets pulled in transitively too —
Cargo unifies features across the whole build, not per binary.

---

## 2. The Crate Map

TenRSo is nine crates in one Cargo workspace. Reach for the one matching your
task:

| Crate | Reach for it when you need to... |
|---|---|
| **`tenrso-core`** | Represent a dense N-D tensor, track named axes, reshape/permute/unfold, or hold any tensor via the unified `TensorHandle`. Every other crate builds on this one. |
| **`tenrso-kernels`** | Call a specific numerical primitive directly: Khatri-Rao, Kronecker, Hadamard, n-mode product (TTM/TTT), MTTKRP (plus blocked/fused/parallel variants), outer products, tensor contractions/reductions. Decomposition algorithms are built from these. |
| **`tenrso-decomp`** | Factorize a tensor: CP-ALS (with constrained/regularized/randomized/completion/sparse variants), Tucker-HOSVD/HOOI (with automatic rank selection), TT-SVD/TT-rounding, plus a rank-selection toolkit (AIC/BIC/MDL, cross-validation, elbow/scree analysis). |
| **`tenrso-sparse`** | Store or operate on sparse data: 8 formats (COO/CSR/CSC/BCSR/ELL/DIA/CSF/HiCOO), SpMV/SpMM/SpSpMM, masked einsum, iterative solvers (CG/BiCGSTAB/GMRES), matrix reordering, graph algorithms. |
| **`tenrso-planner`** | Decide a contraction order for 3+ tensors, estimate FLOPs/memory ahead of time, or choose between dense/sparse/low-rank representations. Six algorithms: greedy, DP (optimal), beam search, simulated annealing, genetic, and an adaptive meta-planner that picks among them. |
| **`tenrso-ooc`** | Work with tensors that don't comfortably fit in RAM: Arrow/Parquet I/O, memory-mapped access, deterministic chunk graphs, spill-to-disk policies, prefetching. |
| **`tenrso-exec`** | Actually *run* an einsum string end-to-end (`einsum_ex`) without hand-rolling planner + kernel calls yourself; also exposes element-wise/reduction ops and a pooled `CpuExecutor`. |
| **`tenrso-ad`** | Differentiate through a contraction or decomposition: VJP rules, custom gradients for CP/Tucker/TT, finite-difference gradient checking, checkpointing, mixed precision, Hessian-vector products, optimizers (SGD/Adam/AdamW/...). |
| **`tenrso`** (facade) | Depend on one crate and get everything re-exported as modules (`tenrso::core`, `tenrso::decomp`, ...) plus a curated `prelude`. |

A typical dependency direction: `tenrso-core` → `tenrso-kernels` → `tenrso-decomp`;
`tenrso-core` + `tenrso-planner` → `tenrso-exec`; `tenrso-sparse` is consumed by
`tenrso-kernels`/`tenrso-decomp` behind the `sparse`/`csf` features; `tenrso-ooc`
is consumed by `tenrso-exec` behind its `ooc` feature; `tenrso-ad` sits on top of
`tenrso-core` + `tenrso-exec` + `tenrso-decomp` + `tenrso-planner`.

---

## 3. Core Concepts (`tenrso-core`)

### 3.1 `DenseND<T>` — the dense tensor type

`DenseND<T>` wraps a row-major (C-contiguous by default), bounds-checked N-D
array. It is generic over the element type (`f32`, `f64`, ...), not over rank.

```rust
use tenrso_core::DenseND;

let zeros = DenseND::<f64>::zeros(&[2, 3]);
let ones  = DenseND::<f64>::ones(&[3, 4]);
let fives = DenseND::from_elem(&[2, 2, 2], 5.0);
let from_data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

let uniform = DenseND::<f64>::random_uniform(&[3, 4], 0.0, 1.0);
let normal  = DenseND::<f64>::random_normal(&[3, 4], 0.0, 1.0);

assert_eq!(zeros.rank(), 2);
assert_eq!(zeros.shape(), &[2, 3]);
assert_eq!(zeros.len(), 6);

// Indexing is bounds-checked
let mut t = DenseND::<f64>::zeros(&[3, 4]);
t[&[0, 0]] = 1.0;
assert_eq!(t[&[0, 0]], 1.0);
```

Random initialization (`random_uniform`/`random_normal`) is backed by
`scirs2_core::random`, never by the `rand` crate directly (see [§10](#10-the-mandatory-scirs2-core-policy)).

### 3.2 Views, reshape, permute

```rust
use tenrso_core::DenseND;

let mut tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

// Zero-copy immutable / mutable views
let view = tensor.view();
assert_eq!(view.shape(), &[2, 3]);
{
    let mut view_mut = tensor.view_mut();
    view_mut[[0, 0]] = 100.0;
}

// Reshape (zero-copy when the tensor is contiguous)
let t = DenseND::<f64>::zeros(&[2, 3, 4]);
let reshaped = t.reshape(&[6, 4]).unwrap();
assert_eq!(reshaped.shape(), &[6, 4]);

// Permute axes (generalized transpose)
let permuted = t.permute(&[2, 0, 1]).unwrap();
assert_eq!(permuted.shape(), &[4, 2, 3]);
```

### 3.3 Unfold / fold (matricization)

Unfold turns mode `m` of an N-D tensor into the rows of a matrix (the other
modes are flattened, in order, into the columns); `fold` is the exact inverse.
This is the core primitive that every decomposition (CP, Tucker, TT) is built
from:

```rust
use tenrso_core::DenseND;

let tensor = DenseND::<f64>::ones(&[2, 3, 4]);

let unfolded = tensor.unfold(1).unwrap();
assert_eq!(unfolded.shape(), &[3, 8]); // 8 = 2 * 4

let folded = DenseND::fold(&unfolded, &[2, 3, 4], 1).unwrap();
assert_eq!(folded.shape(), &[2, 3, 4]);
```

### 3.4 `AxisMeta` and `TensorHandle<T>`

`AxisMeta` attaches a symbolic name (and size) to an axis, purely for
debuggability — it does not change how data is stored or indexed.

`TensorHandle<T>` is the type that the rest of the stack (planner, executor,
AD) actually passes around. It wraps a `TensorRepr<T>` enum with three
variants — `Dense(DenseND<T>)`, `Sparse(SparseND<T>)`, `LowRank(LowRank<T>)` —
so a single handle can carry a dense tensor, a sparse one, or a CP/Tucker/TT
factorization, transparently:

```rust
use tenrso_core::{AxisMeta, DenseND, TensorHandle};

// With explicit axis names
let tensor = DenseND::<f64>::zeros(&[32, 128, 256]);
let axes = vec![
    AxisMeta::new("batch", 32),
    AxisMeta::new("sequence", 128),
    AxisMeta::new("features", 256),
];
let handle = TensorHandle::from_dense(tensor, axes);
assert_eq!(handle.rank(), 3);
assert_eq!(handle.axes[0].name, "batch");

// Or with automatic "axis_0", "axis_1", ... naming
let handle2 = TensorHandle::from_dense_auto(DenseND::<f64>::ones(&[2, 3, 4]));
assert_eq!(handle2.axes[0].name, "axis_0");

// Get back to a dense view/tensor regardless of internal representation
if let Some(dense) = handle2.as_dense() {
    assert_eq!(dense.len(), 24);
}
```

`einsum_ex` (see [§5](#5-the-execution-model-tenrso-exec--tenrso-planner)) takes
`&[TensorHandle<T>]`, not `&[DenseND<T>]` — this is what lets the same call
site transparently accept dense, sparse, or low-rank inputs.

---

## 4. The Einsum Spec Language and Its Limits

TenRSo's einsum notation (parsed by `tenrso_planner::EinsumSpec::parse`,
`crates/tenrso-planner/src/parser.rs`) is intentionally a strict subset of
NumPy/`opt_einsum` notation:

- Input subscripts are comma-separated, e.g. `"ijk,jkl"`.
- An explicit output is written after `->`, e.g. `"ij,jk->ik"`. If `->` is
  omitted, the output is **inferred** as every index that appears, in order of
  first appearance (this differs from NumPy's implicit-mode dedup/sort rule —
  TenRSo does not deduplicate or alphabetize; e.g. `"ij,jk"` infers `"ijk"`).
- Only **lowercase ASCII letters `a`–`z`** are valid index characters, for both
  inputs and the output. Uppercase letters, digits, and any other symbol are
  rejected at parse time.
- An index repeated within a single input (e.g. `"ii->i"` for a diagonal, or
  `"ii->"` for a trace) **parses** without error — the parser does not require
  indices to be unique per input. **However**, as of this writing, actually
  *executing* **any single-input spec** (repeated index or not — this also
  affects a plain same-tensor axis reduction like `"ij->i"`) through
  `einsum_ex`/`CpuExecutor::einsum` does **not** perform what the spec
  describes: with exactly one input tensor, the planner produces zero
  pairwise contraction steps (there is nothing to pair), so the executor's
  contraction loop never runs and the **original tensor is silently returned
  unchanged** — wrong shape, wrong values, no error raised. This was confirmed
  by direct testing while writing this guide (reproduction below) and is a
  real correctness gap, not a documentation nuance. **Do not pass a
  single-tensor einsum spec to `einsum_ex`.** For single-tensor axis
  reductions, use `CpuExecutor::reduce` instead (§5.4) — e.g.
  `executor.reduce(ReduceOp::Sum, &handle, &[1])` correctly sums axis 1 and
  was verified to produce the right answer where the equivalent `"ij->i"`
  spec silently did not. Two-or-more-input specs are unaffected by this
  specific issue (every multi-input example elsewhere in this guide and in
  `TUTORIALS.md` was independently executed and verified correct).
- An index that appears in every input and *not* in the output is contracted
  (summed over); an index that appears in the output must appear in at least
  one input, or parsing fails.

**Explicitly not supported: ellipsis (`...`) broadcasting.** NumPy/PyTorch-style
`"...ij,...jk->...ik"` for batched/broadcast contractions over an unspecified
number of leading dimensions does **not** parse — the character `.` is not
`a`-`z` and is rejected immediately:

```rust
use tenrso_planner::EinsumSpec;

let result = EinsumSpec::parse("...ij,...jk->...ik");
assert!(result.is_err());
// Error: "Input 0 contains invalid characters (only lowercase a-z allowed)"
```

If you need batch dimensions, name them explicitly instead — this is the
pattern used throughout the codebase (e.g. `"bij,bjk->bik"` for a batched
matmul with a single leading batch axis `b`). There is no way to express "N
arbitrary leading batch axes" in one spec string; use one concrete letter per
batch axis you actually have.

A second practical limit: because indices are single ASCII letters, a single
contraction spec supports at most 26 distinct index labels.

**Reproducing the single-input silent-passthrough issue.** The following
compiles and runs against this workspace exactly as shown (on the same 3×3
matrix `[[1,2,3],[4,5,6],[7,8,9]]` used elsewhere in this guide). All three
`einsum_ex` calls succeed (`Ok`, no error) and all three print the *original*
3×3 matrix completely unchanged; the final call, using the documented
`CpuExecutor::reduce` API instead of an einsum spec, correctly computes the
row sums:

```rust,ignore
use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::{einsum_ex, CpuExecutor, ReduceOp, TenrsoExecutor};

let m = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?;
let handle = TensorHandle::from_dense_auto(m);

let diag = einsum_ex::<f64>("ii->i").inputs(&[handle.clone()]).run()?;
println!("{:?}", diag.as_dense().unwrap().shape());   // prints [3, 3], not the expected [3]

let trace = einsum_ex::<f64>("ii->").inputs(&[handle.clone()]).run()?;
println!("{:?}", trace.as_dense().unwrap().shape());   // prints [3, 3], not the expected [] (scalar)

let row_sum_via_einsum = einsum_ex::<f64>("ij->i").inputs(&[handle.clone()]).run()?;
println!("{:?}", row_sum_via_einsum.as_dense().unwrap().shape()); // prints [3, 3], not the expected [3]

// The correct, working way to do this: CpuExecutor::reduce.
let mut executor = CpuExecutor::new();
let row_sums = executor.reduce(ReduceOp::Sum, &handle, &[1])?;
println!("{:?}", row_sums.as_dense().unwrap().as_slice()); // correctly prints [6.0, 15.0, 24.0]
```

**Root cause** (`crates/tenrso-exec/src/executor/types.rs`,
`execute_einsum_with_planner`/`execute_plan`): for exactly one input tensor,
`greedy_planner` produces a plan with zero pairwise contraction steps (there
is nothing to pair up), so `execute_plan`'s step loop never executes and the
single input is returned exactly as received. No error is surfaced anywhere
in this path. This is a genuine correctness bug, not merely a missing
feature — it was found while verifying this guide's own claims and is not
fixed here (see the project's contribution process); until it is fixed
upstream, do not rely on any single-input einsum spec through `einsum_ex`,
regardless of whether it repeats an index.

---

## 5. The Execution Model (`tenrso-exec` + `tenrso-planner`)

### 5.1 `einsum_ex` — the one-call API

For the common case — "run this einsum string on these tensors" — use the
`einsum_ex` builder. It internally invokes `CpuExecutor`, which consults the
planner for contraction order and representation choices:

```rust,ignore
use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::{einsum_ex, ExecHints};

let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
let b = DenseND::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])?;

let result = einsum_ex::<f64>("ij,jk->ik")
    .inputs(&[TensorHandle::from_dense_auto(a), TensorHandle::from_dense_auto(b)])
    .hints(&ExecHints::default())
    .run()?;

assert_eq!(result.as_dense().unwrap().shape(), &[2, 2]);
```

`einsum_ex` supports any number of input tensors (not just two) — the planner
picks a binary contraction order internally. See Tutorial 2 for a worked
three-tensor example.

### 5.2 `ExecHints` — steering execution

```rust
use tenrso_exec::ExecHints;

let hints = ExecHints {
    mask: None,             // Some(MaskPack): compute only the cells this bitmap selects
    subset: None,           // Some(SubsetSpec): the same, by flat index (sparse spelling)
    prefer_sparse: true,    // route through the masked/sparse engine
};

// Builders are equivalent and usually shorter:
let hints = ExecHints::new()
    .with_sparse(true)
    .with_subset(vec![0, 4, 8], vec![3, 3]); // only the diagonal of a 3x3 output
```

`ExecHints` describes *which output cells you want*, and every field is read by
the executor. `mask` and `subset` are two spellings of one selection — a dense
`Vec<bool>` (`O(output)` memory) and a list of flat row-major indices
(`O(selected)` memory) — so supplying both at once is an error, not a silent
precedence rule. Both need `prefer_sparse: true`; that flag is what switches
`einsum_ex` from the dense GEMM engine to the masked engine.

Whether the masked engine reaches a *specialized* kernel (matmul, element-wise,
outer product — see `tenrso_sparse::masked_einsum`) depends on the einsum
pattern; unmatched patterns fall back to a generic masked loop. The selection
itself is always honoured: unselected output cells are never computed.

There is no `tile_kb` knob and no `prefer_lowrank` knob. `f32`/`f64` contractions
go to a native `matrixmultiply` GEMM, which does its own register/cache blocking
and is 4.8-6.1x faster than the portable blocked kernel a user-supplied tile size
could steer — so a tile budget could only make things slower. And there is no
low-rank execution path for `einsum_ex` to prefer: it requires dense operands.

### 5.3 The planner: contraction order and cost estimation

For 3+ input tensors, the order in which you contract pairs changes the total
FLOP count by orders of magnitude (the classical matrix-chain problem,
generalized to tensor networks). `tenrso-planner` provides six algorithms:

| Planner | Complexity | Optimality | Use when |
|---|---|---|---|
| `greedy_planner` | O(n³) | heuristic | many tensors (>10), planning time matters |
| `BeamSearchPlanner` | tunable | better than greedy | 8–20 tensors |
| `dp_planner` | O(3ⁿ) time, O(2ⁿ) space | **optimal** | ≤20 tensors, need a provably best order |
| `SimulatedAnnealingPlanner` | tunable | stochastic | >20 tensors, quality over speed |
| `GeneticAlgorithmPlanner` | tunable | stochastic | >20 tensors, largest time budget |
| `AdaptivePlanner` | auto-selects | — | default choice; picks DP for n≤5, beam/DP for 5<n≤20, greedy/SA/GA above that |

```rust
use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};

let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
let shapes = vec![vec![100, 200], vec![200, 300]];
let plan = greedy_planner(&spec, &shapes, &PlanHints::default()).unwrap();

println!("contraction steps: {}", plan.nodes.len());
println!("estimated FLOPs:   {:.2e}", plan.estimated_flops);
println!("estimated memory:  {} bytes", plan.estimated_memory);
```

The plan also carries a `repr` hint per node — the planner's cost model looks
at density/rank statistics (`nnz`, sparsity hints in `PlanHints`) to decide
whether a given intermediate is worth materializing as dense, sparse, or
low-rank. `tenrso-exec`'s `CpuExecutor` is what actually acts on that choice at
runtime.

### 5.4 Element-wise and reduction operations

`CpuExecutor` also exposes non-contraction operations directly (useful when you
don't want to route a plain elementwise op through an einsum string):

```rust,ignore
use tenrso_exec::{CpuExecutor, ElemOp, ReduceOp};

let mut executor = CpuExecutor::new();
let neg = executor.elem_op(ElemOp::Neg, &handle)?;
let abs = executor.elem_op(ElemOp::Abs, &neg)?;
let row_sums = executor.reduce(ReduceOp::Sum, &matrix_handle, &[1])?;
```

`CpuExecutor` maintains a thread-local memory pool; `executor.pool_stats()`
returns `(hits, misses, hit_rate)` if you want to check whether pooling is
actually helping your workload.

---

## 6. Decompositions (`tenrso-decomp`)

All three decomposition families operate on `DenseND<T>` (and, for CP under the
`sparse` feature, on `tenrso_sparse::CooTensor<T>`), and all return a `*Decomp`
struct with a `.reconstruct()` method that rebuilds an approximation of the
original tensor.

### 6.1 CP-ALS (Canonical Polyadic / CANDECOMP-PARAFAC)

Factorizes a tensor into a sum of `rank` rank-1 outer products:
`X ≈ Σᵣ λᵣ (a₁ᵣ ⊗ a₂ᵣ ⊗ ... ⊗ aₙᵣ)`. Good default for factor-analysis-style
problems (chemometrics, EEG/fMRI, blind source separation) where you have a
prior belief about the number of latent components.

```rust
use tenrso_core::DenseND;
use tenrso_decomp::{cp_als, InitStrategy};

let tensor = DenseND::<f64>::random_uniform(&[50, 50, 50], 0.0, 1.0);
let cp = cp_als(&tensor, 10, 100, 1e-4, InitStrategy::Random, None).unwrap();

println!("converged in {} iterations, fit = {:.4}", cp.iters, cp.fit);
let approx = cp.reconstruct(tensor.shape()).unwrap();
```

- **Convergence:** ALS iterates until `max_iters` is hit or the fit
  improvement between iterations drops below `tol`; pass `Some(Duration)` as
  the last argument for a wall-clock cutoff instead.
- **Initialization (`InitStrategy`):** `Random` (uniform), `RandomNormal`, or
  `Svd` (HOSVD-based warm start — usually converges in fewer iterations, at
  the cost of an upfront SVD per mode).
- **Rank selection:** there is no oracle; use `tenrso_decomp::rank_selection`
  (§6.4) or simply sweep candidate ranks and compare `cp.fit`.
- **Variants:** `cp_als_constrained` (non-negativity, L1/L2 regularization,
  orthogonality), `cp_als_accelerated` (line search), `cp_randomized`
  (sketching for large tensors), `cp_completion` (missing-data / CP-WOPT), and
  — behind the `sparse` feature — `cp_als_sparse` for `CooTensor` input
  (`O(nnz · rank)` MTTKRP instead of densifying first).
- **A worked example with a known ground truth** (so you can see a fit of
  ~1.0 rather than guessing whether the number is "good") is in Tutorial 3.

### 6.2 Tucker (Higher-Order SVD / HOOI)

Factorizes a tensor into a (usually much smaller) core tensor plus one
orthogonal factor matrix per mode: `X ≈ G ×₁ U₁ ×₂ U₂ ×₃ ... ×ₙ Uₙ`. Good for
compression, feature extraction, and any workload where different modes
plausibly have different intrinsic dimensionality (unlike CP, per-mode ranks
need not match).

```rust
use tenrso_core::DenseND;
use tenrso_decomp::{tucker_hooi, tucker_hosvd, tucker_hosvd_auto, TuckerRankSelection};

let tensor = DenseND::<f64>::random_uniform(&[30, 30, 30], 0.0, 1.0);

// One-pass SVD-based decomposition (fast, not iteratively optimal)
let hosvd = tucker_hosvd(&tensor, &[15, 15, 15]).unwrap();

// Iterative refinement (HOOI): monotonically improves on HOSVD at the same ranks
let hooi = tucker_hooi(&tensor, &[15, 15, 15], 20, 1e-4).unwrap();

// Automatic rank selection: keep 90% of the singular-value energy per mode
let auto = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(0.9)).unwrap();
let selected_ranks: Vec<usize> = auto.factors.iter().map(|f| f.ncols()).collect();
```

- **HOSVD vs HOOI:** HOSVD computes each factor from an independent SVD of
  that mode's unfolding — one pass, no iteration. HOOI alternates: fix all
  factors but one, project the tensor down, refine that factor, repeat. HOOI's
  reconstruction error is always `<=` HOSVD's at identical ranks (verified in
  the crate's own integration tests), at the cost of `max_iters` extra passes.
- **Rank selection (`tucker_hosvd_auto`):** `TuckerRankSelection::Energy(f)`
  keeps enough singular values per mode to retain fraction `f` of that mode's
  squared-singular-value energy; `TuckerRankSelection::Threshold(t)` instead
  keeps singular values `> t × σ_max`. Both operate per-mode independently, so
  the total reconstruction error is bounded by (not equal to) the sum of the
  per-mode discarded energy — see Tutorial 4 for a concrete before/after.
- **Convergence (HOOI):** stops at `max_iters` or when the relative change in
  reconstruction error drops below `tol`.

### 6.3 Tensor Train (TT-SVD)

Represents an N-way tensor as a chain of 3-way cores,
`X(i₁,...,iₙ) = G₁[i₁] G₂[i₂] ... Gₙ[iₙ]` (each `Gₖ[iₖ]` is an
`r_{k-1} × r_k` matrix slice). This is the format of choice once you're past
~5–6 modes, where CP/Tucker either can't express the necessary structure
compactly or become numerically awkward — TT is standard in tensor-network and
quantum many-body contexts, and for stochastic-PDE discretizations.

```rust
use tenrso_core::DenseND;
use tenrso_decomp::{tt_round, tt_svd};

let tensor = DenseND::<f64>::random_uniform(&[16, 16, 16, 16], 0.0, 1.0);

// max_ranks has length (n_modes - 1): one interior bond rank per junction
let tt = tt_svd(&tensor, &[8, 8, 8], 1e-6).unwrap();
println!("actual TT-ranks: {:?}", tt.ranks);
println!("compression: {:.2}x", tt.compression_ratio());

// Re-round to a smaller rank budget (further lossy compression)
let rounded = tt_round(&tt, &[4, 4, 4], 1e-6).unwrap();
```

- **Convergence / truncation:** `tt_svd` truncates each bond's SVD to the
  smaller of `max_ranks[k]` and whatever rank is needed to keep the truncated
  singular values' contribution under `tol` (relative Frobenius sense) — it is
  not iterative the way ALS is; one left-to-right sweep produces the result.
- **`tt_round`:** recompresses an existing `TTDecomp` to a smaller rank budget
  without rebuilding from the dense tensor — this is the operation you use
  after e.g. a `tt_add` or `tt_hadamard` inflates the ranks and you want to
  bring the representation back down.
- **Other TT operations:** `tt_add`, `tt_dot` (inner product without
  reconstructing to dense), `tt_hadamard` — all operate directly on TT cores.
- **A worked example with a known TT-rank ground truth** — so "43× compression
  at zero error, then 144× compression at meaningfully higher error" is a
  concrete, reproducible number rather than a vague claim — is in Tutorial 5.

### 6.4 Rank selection toolkit (`tenrso_decomp::rank_selection`)

Model-order selection utilities shared across CP/Tucker/TT:

- **Information criteria:** `InformationCriterion::{AIC, BIC, MDL}`, applied
  via `select_rank_auto(&errors, &num_params_per_rank, num_observations, strategy)`.
  `cp_num_params(shape, rank)` / `tucker_num_params(shape, &ranks)` compute the
  parameter counts these criteria need.
- **Cross-validation:** `create_cv_split(shape, train_fraction)` produces a
  train/validation mask pair; combine with `cp_completion` (train on the
  masked tensor) and `masked_reconstruction_error` (evaluate on the held-out
  mask).
- **Elbow / scree analysis:** `ScreePlotData::new(singular_values, thresh_a, thresh_b)`
  exposes `.suggested_rank` (elbow detection), `.suggested_rank_90` /
  `.suggested_rank_95` (fixed variance thresholds), and
  `.rank_for_variance(threshold)` for arbitrary thresholds.
- **`RankSelectionStrategy::Combined(criterion)`** runs an information
  criterion and cross-checks it against elbow detection.

See `crates/tenrso-decomp/examples/rank_selection.rs` for all six strategies
exercised end-to-end (this file is a real, compiling example in the
workspace — run it with `cargo run --example rank_selection -p tenrso-decomp`).

---

## 7. Sparse Formats (`tenrso-sparse`)

Eight formats are provided; each is a genuinely different storage/performance
tradeoff, not just a naming variant. Full write-ups (with worked examples and
decision trees) live in `crates/tenrso-sparse/FORMAT_GUIDE.md` and
`FORMAT_SELECTION_GUIDE.md` — this is the condensed version:

| Format | Dimensionality | Best for | Avoid when |
|---|---|---|---|
| **COO** | N-D | Incremental construction, format conversion, staging | Repeated arithmetic (no locality) |
| **CSR** | 2-D | Row access, SpMV, SpMM — the default general-purpose 2-D format | Column-heavy access patterns |
| **CSC** | 2-D | Column access, transposes (CSR ↔ CSC), factorizations that iterate columns | Row-heavy access patterns |
| **BCSR** (block CSR / BSR) | 2-D | Block-structured sparsity (FE meshes, blocked graphs) — dense SIMD-friendly blocks | Irregular/scattered sparsity with no block structure |
| **ELL** (ELLPACK) | 2-D | Uniform nonzeros-per-row, SIMD/GPU-style coalesced access | Highly variable row degree (wastes padding) |
| **DIA** (diagonal) | 2-D | Banded matrices — PDE stencils, tri/penta-diagonal systems | Non-banded sparsity |
| **CSF** *(feature `csf`)* | N-D | Sparse-tensor MTTKRP via fiber-tree traversal — the sparse-tensor decomposition workhorse | Plain 2-D problems (use CSR) |
| **HiCOO** *(feature `csf`)* | N-D | Very sparse, block-clustered high-order tensors | Scattered (non-clustered) nonzero patterns |

```rust
use tenrso_sparse::{CooTensor, CsrMatrix};

// Build incrementally in COO...
let indices = vec![vec![0, 0], vec![0, 2], vec![1, 1], vec![2, 0], vec![3, 3]];
let values  = vec![4.0, -1.0, 4.0, -1.0, 4.0];
let coo = CooTensor::new(indices, values, vec![4, 4]).unwrap();

// ...then convert to CSR for repeated row-oriented operations
let csr = CsrMatrix::from_coo(&coo).unwrap();
let dense = csr.to_dense().unwrap();
```

### 7.1 Masked einsum

When only a known subset of an einsum's output is needed, `masked_einsum`
computes just those entries and returns a `CooTensor` — for a mask with density
`d`, matmul-shaped patterns cost `O(d · M · N · K)` instead of `O(M · N · K)`:

```rust
use tenrso_core::DenseND;
use tenrso_sparse::mask::Mask;
use tenrso_sparse::masked_einsum::masked_einsum;

let a = DenseND::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
let mask = Mask::from_indices(vec![vec![0, 0], vec![1, 1]], vec![2, 2]).unwrap();

let result = masked_einsum("ij,jk->ik", &[&a, &b], &mask).unwrap();
assert_eq!(result.nnz(), 2); // only the two masked output cells were computed
```

Three patterns have specialized kernels (`ij,jk->ik` matmul,
`ij,ij->ij` element-wise, `i,j->ij` outer product); anything else falls back
to a generic (correct, but slower) masked contraction loop. `ExecHints{ mask:
Some(_), prefer_sparse: true, .. }` routes `einsum_ex` through this same path.

---

## 8. Out-of-Core Processing (`tenrso-ooc`)

### 8.1 When you need it

Reach for `tenrso-ooc` once a tensor (or an intermediate result) no longer
comfortably fits in RAM, or you want to persist tensors to disk in a
columnar/interoperable format (Arrow/Parquet) rather than an ad hoc binary
blob. For tensors that *do* fit in memory, plain `DenseND` + `tenrso-exec` is
simpler and faster — don't reach for chunking machinery you don't need.

### 8.2 Arrow / Parquet I/O

```rust,ignore
use tenrso_core::DenseND;
use tenrso_ooc::arrow_io::{ArrowReader, ArrowWriter};

let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

let mut writer = ArrowWriter::new("/tmp/example.arrow")?;
writer.write(&tensor)?;
writer.finish()?;

let mut reader = ArrowReader::open("/tmp/example.arrow")?;
let loaded: DenseND<f64> = reader.read()?; // whole tensor, reconstructed with its original shape
```

`ParquetWriter`/`ParquetReader` (feature `parquet`, default-on) have an
identical shape. Both `read()` calls load the **entire** tensor back into a
`DenseND` in one call — the out-of-core story for a single Arrow/Parquet file
is "smaller on disk, columnar, interoperable," not "streamed element by
element off disk." For genuinely chunked *processing* of a large tensor, use
the chunking primitives below on the loaded (or memory-mapped) tensor.

### 8.3 Chunking and streaming

`ChunkSpec` deterministically partitions a shape into tiles (by fixed tile
size or by a fixed number of chunks per dimension); `ChunkSpec::iter()` walks
every chunk in row-major order:

```rust
use tenrso_ooc::ChunkSpec;

let spec = ChunkSpec::tile_size(&[40, 25], &[8, 25]).unwrap(); // 5 row-chunks
assert_eq!(spec.total_chunks(), 5);

for chunk_idx in spec.iter() {
    let (start, end) = spec.chunk_bounds(&chunk_idx);
    // process tensor[start[0]..end[0], start[1]..end[1]] ...
}
```

`StreamingExecutor` (`StreamConfig::new().max_memory_mb(..).chunk_size(..)`)
applies this same tiling to specific operations — `matmul_chunked`,
`add_chunked`, `multiply_chunked`, `fma_chunked`, and friends — bounding peak
memory and optionally spilling to disk (`enable_spill(true)`) when a chunk
doesn't fit. `MmapTensor::open` gives a memory-mapped, zero-copy view over a
tensor written with `write_tensor_binary`, letting the OS page data in on
demand instead of loading everything up front. See Tutorial 7 for a full
write → read-back → chunked-processing walkthrough with real output.

### 8.4 Other capabilities (browse, don't assume)

`tenrso-ooc` is the largest crate in the workspace and also contains: spill
policies (`SpillPolicy`: LRU/LFU/FIFO/MRU, plus an ML-driven
`MLEvictionPolicy`), tiered memory management (`TieredMemoryManager`), NUMA
awareness, prefetching (including a lock-free variant behind the `lock-free`
feature), a `ChunkGraph` for deterministic multi-op streaming schedules with
memory-constrained execution ordering, data-integrity checksums, and
compression (`oxiarc-lz4`/`oxiarc-zstd`, both Pure-Rust). GPU *device
enumeration* exists behind default-off `cuda`/`rocm`/`metal`/`vulkan`
features; as of this writing, tensor kernels are **not** dispatched to any
detected GPU — buffer operations always execute on the CPU regardless of which
backends are compiled in (check the module doc-comment in `src/gpu.rs` for the
current, honestly-reported state before relying on this).

---

## 9. Automatic Differentiation (`tenrso-ad`)

`tenrso-ad` provides reverse-mode (VJP) rules hand-written for TenRSo's own
operations — it is not a general eager-mode autodiff tape, and it does not
depend on an external AD framework by default.

```rust,ignore
use tenrso_ad::vjp::{EinsumVjp, VjpOp};
use tenrso_ad::gradcheck::{check_gradient, GradCheckConfig};
use tenrso_core::DenseND;
use tenrso_exec::ops::execute_dense_contraction;
use tenrso_planner::EinsumSpec;

let spec = EinsumSpec::parse("ij,jk->ik")?;
let c = execute_dense_contraction(&spec, &a, &b)?;

let grad_c = DenseND::ones(c.shape());              // dL/dC, e.g. from L = sum(C)
let vjp = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
let grads = vjp.vjp(&grad_c)?;                        // [dL/dA, dL/dB]

// Cross-check the analytical gradient against central finite differences
let result = check_gradient(
    |x: &DenseND<f64>| execute_dense_contraction(&spec, x, &b),
    |_x, grad_y| Ok(EinsumVjp::new(spec.clone(), a.clone(), b.clone()).vjp(grad_y)?[0].clone()),
    &a, &grad_c, &GradCheckConfig::default(),
)?;
assert!(result.passed);
```

- **Custom gradients for decompositions:** the `grad` module has hand-derived
  backward rules for CP-ALS, Tucker-HOOI, and TT-SVD reconstruction, avoiding
  the tape blow-up a naive autodiff-through-ALS-iterations approach would
  incur.
- **Gradient checking (`gradcheck`):** `check_gradient` compares an analytical
  gradient against a central- or forward-difference numerical estimate;
  `GradCheckConfig` controls step size (`epsilon`), tolerances (`rtol`/`atol`),
  and which finite-difference scheme to use. This is the standard way to
  validate any new hand-written gradient rule — see Tutorial 8.
- **Beyond basic VJP:** gradient checkpointing (`checkpoint`, O(√n) memory for
  a sequence of operations), sparse gradients (`sparse_grad`), mixed-precision
  training utilities (`mixed_precision`), Hessian-vector products (`hessian`),
  gradient monitoring/clipping/compression (`monitoring`, `utils`,
  `compression`), and optimizers (`optimizers`: SGD, Adam, AdamW, RMSprop,
  AdaGrad, LR schedulers).
- **External-framework hooks (`hooks`):** a trait-based registration point for
  wiring TenRSo's forward/backward rules into an external AD system (the
  Tensorlogic integration is tracked as in-progress in the project's
  `TODO.md` — the hook surface exists, but the registration of einsum/
  decomposition gradients with Tensorlogic itself is not yet wired up).

---

## 10. The Mandatory SciRS2-Core Policy

Every crate under `crates/` in this workspace is **forbidden** from directly
depending on `ndarray`, `rand`, `rand_distr`, `num-traits`, `num-complex`,
`nalgebra`, or `rayon` — see `SCIRS2_INTEGRATION_POLICY.md` at the workspace
root for the full, enforced list and the CI checks that verify it. Instead:

```rust,ignore
// FORBIDDEN anywhere under crates/
use ndarray::Array;
use rand::thread_rng;
use rayon::prelude::*;

// REQUIRED instead
use scirs2_core::ndarray_ext::{Array, ArrayView, Ix2, s, array};
use scirs2_core::random::{thread_rng, Rng, distributions::Uniform};
use scirs2_core::parallel_ops::*; // Rayon re-exported through scirs2-core
```

**What this means as a consumer of TenRSo (not just a contributor):** this
policy is enforced on TenRSo's *own* source tree, not on arbitrary downstream
application code — nothing stops your `main.rs` from `use ndarray::Array`
directly. In practice, though, you will usually want `scirs2-core` anyway,
because TenRSo's public API is built on its types: `DenseND<T>` wraps
`scirs2_core::ndarray_ext::Array<T, IxDyn>` internally, `tenrso_kernels`
functions like `cp_reconstruct`/`tucker_reconstruct` take
`scirs2_core::ndarray_ext::{Array2, ArrayView, ArrayView2}` directly, and
`DenseND::random_uniform`/`random_normal` are backed by
`scirs2_core::random`. If you need to build a factor matrix by hand to feed
into `cp_reconstruct` or `tucker_reconstruct` (as the tutorials in this guide
do to construct known-rank test tensors), you'll be reaching for
`scirs2_core::ndarray_ext::Array2::from_shape_vec` — using plain `ndarray`
types would not type-check against these signatures without an extra
conversion step.

`scirs2-core` (pinned to `0.6.0` in the workspace `Cargo.toml`, with the
`array`, `random`, and `parallel` features enabled) also provides:
SIMD-accelerated element-wise ops (`scirs2_core::simd_ops::SimdUnifiedOps`,
with `PlatformCapabilities::detect()` for runtime AVX2/AVX-512 checks),
`scirs2-linalg` (SVD/QR — the backbone of Tucker/TT/CP-init), and an optional
`scirs2_core::gpu` abstraction (feature-gated, used for the honest GPU
enumeration described in [§8.4](#84-other-capabilities-browse-dont-assume)).

---

## 11. Performance Notes (Honest Numbers)

These are the project's own measured numbers (from `TODO.md` and the
per-crate `PERFORMANCE.md`/lib.rs docs), not marketing copy. Read them before
you plan capacity around this stack.

### 11.1 Kernels (`tenrso-kernels`, per its own `PERFORMANCE.md`/lib.rs docs)

| Operation | Measured throughput |
|---|---|
| Khatri-Rao (serial) | ~1.5 Gelem/s |
| Khatri-Rao (parallel) | ~3× the serial rate |
| MTTKRP (blocked, parallel) | ~13.3 Gelem/s (peak) |
| N-mode product | >5 Gelem/s sustained |
| Hadamard (in-place) | ~11 Gelem/s, ~2.7× faster than the allocating variant |

### 11.2 Decomposition targets vs. measured reality

The ROADMAP's original performance targets (16-core CPU, dense f32/f64) and
the project's own most recent measurements (`TODO.md`, 8-core x86_64 AVX2,
pure Rust, 2026-05-30) do **not** all agree — and the gap is documented rather
than hidden:

| Workload | Target | Measured | Verdict |
|---|---|---|---|
| CP-ALS, 256³, rank-64, 10 iters | < 2s | **~21–44s** | **Missed, substantially.** Root cause (documented): each MTTKRP reads a 134MB unfolded matrix plus a 33MB Khatri-Rao product per iteration — the Khatri-Rao matrix alone exceeds typical L3 cache. The pure-Rust `matrixmultiply` GEMM this stack uses runs at ~3–4 GFLOP/s on this shape, versus ~50 GFLOP/s for multi-threaded OpenBLAS DGEMM. This is a **memory-bandwidth limit inherent to a pure-Rust, no-BLAS-backend design**, not a bug to be tuned away — parallelizing further *increases* cache pressure rather than reducing wall time. A BLAS backend would be required to hit the original 2s target; none is currently wired in (by the workspace's Pure-Rust policy). |
| Tucker-HOOI, target ranks [64,64,32] | < 3s / 10 iters | ~35–50% faster than a prior (mis-specified) benchmark; completes on the documented target shape | The original in-repo benchmark had used the *wrong* ranks ([256,256,64], which happened to disable randomized SVD); at the ranks the ROADMAP actually specifies, randomized SVD activates on every mode and the benchmark is meaningfully faster than before the fix. No absolute wall-clock number against the <3s target is published as of this writing. |
| TT-SVD, 32⁶, eps=1e-6 | < 2s build | 32⁴ baseline preserved at 2.79–3s; 32⁶ expected to complete without timeout after a Gram-matrix-based thin-SVD fix, but **not yet separately measured** | A naive Gaussian-sketch SVD would need to allocate an ~8.4GB random matrix for the 32⁶ case; `thin_svd_via_gram` avoids this by computing the Gram matrix instead. Treat the 32⁶ number as "should work," not "measured passing," until re-benchmarked. |
| TT memory reduction | ≥ 10× | **20,459×** (32⁶ case) | Passed, by a wide margin. |
| Einsum vs. OpenBLAS baseline | ≥ 80% of OpenBLAS | **not measurable** | The workspace's Pure-Rust policy means there is no OpenBLAS in the dependency tree to compare against; this target is structurally unmeasurable under the project's own constraints, not merely "not yet run." |
| Masked einsum vs. dense naive | ≥ 5× speedup | benchmark harness exists (`crates/tenrso-sparse/benches/masked_einsum_bench.rs`, compares 50%/90%/99% sparsity vs. dense) | No single headline number is published in the tracked docs as of this writing; run the benchmark yourself for your target sparsity if you need a number to plan around. |

**Practical takeaway:** if your workload looks like "CP-ALS on a 256³-or-larger
dense tensor at rank 64", budget tens of seconds per 10 iterations on current
hardware, not the originally-hoped-for 2 seconds, until a BLAS-backed GEMM
path lands. Smaller problems, sparser inputs, or Tucker/TT (which lean more on
SVD than on dense GEMM-heavy MTTKRP) are comparatively closer to their
targets.

---

## 12. Where to Go Next

- **[`TUTORIALS.md`](TUTORIALS.md)** — eight complete, executed, progressive
  walkthroughs: first tensor, einsum + planner, CP, Tucker + rank selection,
  TT + rounding, sparse + masked einsum, out-of-core + chunking, and autodiff.
- **`examples/`** directories inside each crate (`crates/*/examples/*.rs`) —
  464 example programs at last count (per `TODO.md`), covering far more ground
  than any guide can. Run any of them with
  `cargo run --example <name> -p <crate>`.
- **`crates/tenrso/tests/`** — cross-crate integration and regression tests
  (`kernels_decomp_integration.rs`, `regression_suite.rs`) are a good source of
  additional, definitely-correct usage patterns.
- Per-crate `README.md` and `TODO.md` for crate-specific detail and known
  in-progress work.
- `cargo doc --workspace --no-deps --open` for full API documentation.
