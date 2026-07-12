# TenRSo Tutorials

> Eight progressive, complete, **runnable** programs — each one was written to a
> scratch crate, compiled, and executed against this workspace before being
> pasted into this document. See "How these were verified" at the end for
> exactly what that means and what to expect if you run them yourself.
>
> For API reference, the crate map, and honest performance numbers, see the
> companion [`USER_GUIDE.md`](USER_GUIDE.md) in this same directory.

---

## Before you start

Every tutorial below is a complete `fn main() -> anyhow::Result<()>` — drop one
into `src/main.rs` of a new binary crate. A minimal `Cargo.toml` covering
Tutorials 1, 2, 3, 4, 5, 6, and 8 looks like this:

```toml
[package]
name = "tenrso_tutorials"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow      = "1"
scirs2-core = { version = "0.6", features = ["array", "random", "parallel"] }
tenrso-core     = "0.1.0"
tenrso-kernels  = "0.1.0"
tenrso-decomp   = "0.1.0"
tenrso-sparse   = "0.1.0"
tenrso-planner  = "0.1.0"
tenrso-exec     = "0.1.0"
tenrso-ad       = "0.1.0"
```

Tutorial 7 (out-of-core) additionally needs `tenrso-ooc = "0.1.0"`. If you are
working from a clone of the `tenrso` workspace itself, you don't need any of
this — `cargo new` a scratch crate outside the workspace, or just create a
`src/bin/<name>.rs` file under any crate's own package and `cargo run --example`
it; the workspace's `Cargo.lock` will resolve everything for you.

**A note on the exact numbers you'll see:** Tutorials 3, 4, and 5 build a
*synthetic* tensor with a known rank so the decomposition has real ground
truth to recover, using `DenseND::random_uniform`/`random_normal`, which are
**not seeded**. Shapes, ranks, and pass/fail outcomes are stable across runs;
specific floating-point values (fit, error, iteration counts) will differ
slightly each time you run them. Where useful, this doc notes which numbers
are structural (always true) versus illustrative (representative, will vary).

---

## Tutorial 1 — First Tensor: Create, Reshape, Permute, Unfold

The foundation of everything else: `DenseND<T>`, TenRSo's dense N-D tensor
type. This tutorial creates a 3D tensor, reshapes it, permutes its axes, and
unfolds/folds it (the matricization operation every decomposition algorithm is
built from).

```rust
//! Tutorial 1: First tensor -- create, reshape, permute, unfold.
use tenrso_core::DenseND;

fn main() -> anyhow::Result<()> {
    // 1. Create a 3D tensor from a row-major Vec<f64>.
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let tensor = DenseND::from_vec(data, &[2, 3, 4])?;
    println!("shape:  {:?}", tensor.shape());
    println!("rank:   {}", tensor.rank());
    println!("len:    {}", tensor.len());
    println!("[0,0,0]: {}", tensor[&[0, 0, 0]]);
    println!("[1,2,3]: {}", tensor[&[1, 2, 3]]);

    // 2. Reshape (zero-copy when contiguous).
    let reshaped = tensor.reshape(&[6, 4])?;
    println!("\nreshaped to {:?}", reshaped.shape());

    // 3. Permute (generalized transpose).
    let permuted = tensor.permute(&[2, 0, 1])?;
    println!("permuted [2,0,1] -> shape {:?}", permuted.shape());
    // Original element [1,2,3] should now live at permuted index [3,1,2].
    println!(
        "original[1,2,3]={} == permuted[3,1,2]={}",
        tensor[&[1, 2, 3]],
        permuted[&[3, 1, 2]]
    );

    // 4. Unfold (mode-n matricization) and fold back.
    let unfolded = tensor.unfold(1)?;
    println!("\nunfold(mode=1) shape: {:?}", unfolded.shape());
    let folded = DenseND::fold(&unfolded, &[2, 3, 4], 1)?;
    println!("fold back shape: {:?}", folded.shape());

    let mut max_diff = 0.0_f64;
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                let d = (tensor[&[i, j, k]] - folded[&[i, j, k]]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
    }
    println!("max |original - unfold->fold| = {:.3e}", max_diff);

    Ok(())
}
```

**Actual output** (deterministic — no randomness involved):

```text
shape:  [2, 3, 4]
rank:   3
len:    24
[0,0,0]: 0
[1,2,3]: 23

reshaped to [6, 4]
permuted [2,0,1] -> shape [4, 2, 3]
original[1,2,3]=23 == permuted[3,1,2]=23

unfold(mode=1) shape: [3, 8]
fold back shape: [2, 3, 4]
max |original - unfold->fold| = 0.000e0
```

**What to notice:** `permute` really does move data logically (element `23` at
original index `[1,2,3]` is exactly the element at `[3,1,2]` in the permuted
tensor), and `unfold` followed by `fold` is an exact round trip to machine
precision.

---

## Tutorial 2 — Einsum: Matmul, Batched Contraction, Multi-Tensor Contraction via the Planner

`einsum_ex` is the one-call API for contractions; for anything with 3+ input
tensors, the *order* you contract pairs in matters, which is where
`tenrso-planner` comes in. This tutorial does a plain matmul, a batched matmul
(one explicit batch axis `b`), and then a three-tensor chain contraction where
we inspect the planner's chosen contraction order and cost estimate before
executing it.

```rust
//! Tutorial 2: Einsum -- matmul, batched contraction, multi-tensor contraction via the planner.
use tenrso_core::{DenseND, TensorHandle};
use tenrso_exec::{einsum_ex, CpuExecutor, ExecHints, TenrsoExecutor};
use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};

fn main() -> anyhow::Result<()> {
    // 1. Plain matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let b = DenseND::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])?;

    let handle_a = TensorHandle::from_dense_auto(a);
    let handle_b = TensorHandle::from_dense_auto(b);

    let c = einsum_ex::<f64>("ij,jk->ik")
        .inputs(&[handle_a, handle_b])
        .run()?;
    println!("A @ B =\n{:?}", c.as_dense().unwrap().view());

    // 2. Batched matmul: C[b,i,k] = sum_j A[b,i,j] * B[b,j,k]
    let batch = 2usize;
    let a_batched = DenseND::from_vec((0..batch * 2 * 3).map(|x| x as f64 + 1.0).collect(), &[batch, 2, 3])?;
    let b_batched = DenseND::from_vec((0..batch * 3 * 2).map(|x| x as f64 + 1.0).collect(), &[batch, 3, 2])?;

    let handle_ab = TensorHandle::from_dense_auto(a_batched);
    let handle_bb = TensorHandle::from_dense_auto(b_batched);

    let c_batched = einsum_ex::<f64>("bij,bjk->bik")
        .inputs(&[handle_ab, handle_bb])
        .hints(&ExecHints::default())
        .run()?;
    println!("\nBatched result shape: {:?}", c_batched.as_dense().unwrap().shape());

    // 3. Multi-tensor contraction planned explicitly via the planner, then executed.
    let spec = EinsumSpec::parse("ij,jk,kl->il")?;
    let shapes = vec![vec![2, 3], vec![3, 4], vec![4, 2]];
    let hints = PlanHints::default();
    let plan = greedy_planner(&spec, &shapes, &hints)?;
    println!(
        "\nPlanned {} contraction step(s), estimated FLOPs: {:.2e}",
        plan.nodes.len(),
        plan.estimated_flops
    );

    let x = DenseND::from_vec((0..6).map(|v| v as f64 + 1.0).collect(), &[2, 3])?;
    let y = DenseND::from_vec((0..12).map(|v| v as f64 + 1.0).collect(), &[3, 4])?;
    let z = DenseND::from_vec((0..8).map(|v| v as f64 + 1.0).collect(), &[4, 2])?;

    let hx = TensorHandle::from_dense_auto(x);
    let hy = TensorHandle::from_dense_auto(y);
    let hz = TensorHandle::from_dense_auto(z);

    let mut executor = CpuExecutor::new();
    let result = executor.einsum("ij,jk,kl->il", &[hx, hy, hz], &ExecHints::default())?;
    println!("3-tensor contraction result shape: {:?}", result.as_dense().unwrap().shape());
    println!("result:\n{:?}", result.as_dense().unwrap().view());

    Ok(())
}
```

**Actual output** (deterministic — no randomness involved):

```text
A @ B =
[[58.0, 64.0],
 [139.0, 154.0]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2

Batched result shape: [2, 2, 2]

Planned 2 contraction step(s), estimated FLOPs: 8.00e1
3-tensor contraction result shape: [2, 2]
result:
[[812.0, 1838.0],
 [1000.0, 2260.0]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2
```

**What to notice:** a 3-tensor chain `"ij,jk,kl->il"` compiles down to exactly
2 binary contraction steps (as it must — 3 inputs always need `n-1 = 2`
pairwise contractions, regardless of order); the planner's `estimated_flops`
is a pre-execution cost estimate you can use to compare candidate orders
*before* running anything.

---

## Tutorial 3 — CP Decomposition End-to-End

CP-ALS factorizes a tensor into a sum of rank-1 outer products. To make "is
this fit good?" answerable rather than a guess, this tutorial first builds a
tensor with a **known** CP-rank (by summing 6 random rank-1 components via
`tenrso_kernels::cp_reconstruct`), then runs CP-ALS at the matching rank and
checks that it recovers the ground truth.

```rust
//! Tutorial 3: CP decomposition end-to-end -- decompose, inspect fit, reconstruct, measure error.
use scirs2_core::ndarray_ext::Array2;
use tenrso_core::DenseND;
use tenrso_decomp::{cp_als, InitStrategy};
use tenrso_kernels::cp_reconstruct;

/// Build a mode_size x rank random factor matrix (values in [-1, 1)).
fn random_factor(mode_size: usize, rank: usize) -> Array2<f64> {
    let flat = DenseND::<f64>::random_uniform(&[mode_size, rank], -1.0, 1.0);
    Array2::from_shape_vec((mode_size, rank), flat.as_slice().to_vec())
        .expect("shape matches flat data length")
}

fn main() -> anyhow::Result<()> {
    let shape = [20usize, 18, 16];
    let rank = 6;

    // Build a *known* rank-6 tensor by summing 6 random rank-1 components, so
    // CP-ALS at rank=6 has a ground truth to recover (rather than fitting noise).
    let factors: Vec<Array2<f64>> = shape.iter().map(|&m| random_factor(m, rank)).collect();
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    let ground_truth = cp_reconstruct(&factor_views, None)?;
    let tensor = DenseND::from_array(ground_truth);

    let cp = cp_als(&tensor, rank, 50, 1e-6, InitStrategy::Svd, None)?;

    println!("CP-ALS on a synthetic rank-{rank} tensor of shape {:?}", shape);
    println!("  iterations: {}", cp.iters);
    println!("  fit:        {:.6}", cp.fit);
    for (i, factor) in cp.factors.iter().enumerate() {
        println!("  factor[{i}] shape: {:?}", factor.shape());
    }

    let reconstructed = cp.reconstruct(&shape)?;
    let original_norm = tensor.frobenius_norm();
    let error = (&tensor - &reconstructed).frobenius_norm();
    let relative_error = error / original_norm;

    println!("\noriginal norm:      {:.6}", original_norm);
    println!("reconstruction err: {:.6}", error);
    println!("relative error:     {:.6}", relative_error);
    println!("fit (1 - rel_err):  {:.6}", 1.0 - relative_error);

    Ok(())
}
```

**Representative output** (shapes and near-zero error are structural;
iteration count and the exact 6th decimal will vary run to run since the
ground truth is unseeded random data):

```text
CP-ALS on a synthetic rank-6 tensor of shape [20, 18, 16]
  iterations: 18
  fit:        1.000000
  factor[0] shape: [20, 6]
  factor[1] shape: [18, 6]
  factor[2] shape: [16, 6]

original norm:      37.479316
reconstruction err: 0.000011
relative error:     0.000000
fit (1 - rel_err):  1.000000
```

**What to notice:** fit reaches essentially `1.0` because the target tensor's
true CP-rank exactly matches the rank we asked for — this is what "CP-ALS is
working correctly" looks like. If you instead run `cp_als` at rank 6 against
an *arbitrary* (full-rank) tensor of the same shape, expect a much lower fit
(often ~0.4–0.6) — that's not a bug, it's rank-6 genuinely being insufficient
to explain generic data. Use the rank-selection toolkit
(`tenrso_decomp::rank_selection`, see `USER_GUIDE.md` §6.4) rather than
guessing when you don't know the true rank ahead of time.

---

## Tutorial 4 — Tucker Decomposition + Rank Selection

Tucker factorizes a tensor into a core tensor plus one factor matrix per mode,
and (unlike CP) lets each mode have a different rank. This tutorial builds a
tensor with a known multilinear rank of `[5,5,5]`, then compares: (a)
oversized manual ranks via HOSVD and HOOI, (b) automatic energy-based rank
selection at two different energy thresholds.

```rust
//! Tutorial 4: Tucker decomposition + automatic rank selection.
use scirs2_core::ndarray_ext::Array2;
use tenrso_core::DenseND;
use tenrso_decomp::{tucker_hooi, tucker_hosvd, tucker_hosvd_auto, TuckerRankSelection};
use tenrso_kernels::tucker_reconstruct;

/// Build a mode_size x rank random factor matrix (values in [-1, 1)).
fn random_factor(mode_size: usize, rank: usize) -> Array2<f64> {
    let flat = DenseND::<f64>::random_uniform(&[mode_size, rank], -1.0, 1.0);
    Array2::from_shape_vec((mode_size, rank), flat.as_slice().to_vec())
        .expect("shape matches flat data length")
}

fn main() -> anyhow::Result<()> {
    let shape = vec![30, 28, 26];
    let true_ranks = [5usize, 5, 5];

    // Build a tensor with a *known* multilinear rank of [5,5,5]: a small random
    // core expanded by random factor matrices. Any Tucker rank >= 5 per mode
    // should reconstruct it almost exactly; this makes rank-selection behavior
    // easy to interpret.
    let core = DenseND::<f64>::random_uniform(&true_ranks, 0.0, 1.0);
    let factors: Vec<Array2<f64>> = shape
        .iter()
        .zip(true_ranks.iter())
        .map(|(&m, &r)| random_factor(m, r))
        .collect();
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    let ground_truth = tucker_reconstruct(&core.view(), &factor_views)?;
    let tensor = DenseND::from_array(ground_truth);

    // 1. Manual ranks via Tucker-HOSVD (one-pass SVD), deliberately oversized.
    let ranks = vec![15, 14, 13];
    let hosvd = tucker_hosvd(&tensor, &ranks)?;
    let hosvd_err = (&tensor - &hosvd.reconstruct()?).frobenius_norm() / tensor.frobenius_norm();
    println!(
        "True multilinear rank: {:?}\n",
        true_ranks
    );
    println!("HOSVD manual ranks {:?}: relative error = {:.6}", ranks, hosvd_err);

    // 2. Iterative refinement via Tucker-HOOI at the same ranks.
    let hooi = tucker_hooi(&tensor, &ranks, 20, 1e-6)?;
    let hooi_err = (&tensor - &hooi.reconstruct()?).frobenius_norm() / tensor.frobenius_norm();
    println!(
        "HOOI  manual ranks {:?}: relative error = {:.6} ({} iterations)",
        ranks, hooi_err, hooi.iters
    );

    // 3. Automatic rank selection: keep 90% of the singular-value energy per mode.
    let auto90 = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(0.9))?;
    let ranks90: Vec<usize> = auto90.factors.iter().map(|f| f.ncols()).collect();
    let err90 = (&tensor - &auto90.reconstruct()?).frobenius_norm() / tensor.frobenius_norm();
    println!("\nAuto rank (90% energy): ranks = {:?}", ranks90);
    println!("  core shape: {:?}", auto90.core.shape());
    println!("  relative error: {:.6}", err90);

    // 4. Automatic rank selection: keep 99.9% of the singular-value energy per mode.
    let auto999 = tucker_hosvd_auto(&tensor, TuckerRankSelection::Energy(0.999))?;
    let ranks999: Vec<usize> = auto999.factors.iter().map(|f| f.ncols()).collect();
    let err999 = (&tensor - &auto999.reconstruct()?).frobenius_norm() / tensor.frobenius_norm();
    println!("\nAuto rank (99.9% energy): ranks = {:?}", ranks999);
    println!("  relative error: {:.6}", err999);

    Ok(())
}
```

**Representative output** (the zero errors at the true rank and at 99.9%
energy are structural; the exact 90%-energy ranks/error vary a little run to
run because the ground truth is unseeded random data):

```text
True multilinear rank: [5, 5, 5]

HOSVD manual ranks [15, 14, 13]: relative error = 0.000000
HOOI  manual ranks [15, 14, 13]: relative error = 0.000000 (2 iterations)

Auto rank (90% energy): ranks = [2, 2, 2]
  core shape: [2, 2, 2]
  relative error: 0.375725

Auto rank (99.9% energy): ranks = [5, 5, 5]
  relative error: 0.000000
```

**What to notice:** oversized manual ranks (`[15,14,13]`, well above the true
`[5,5,5]`) reconstruct essentially exactly, as expected — HOOI needed only 2
iterations to converge because HOSVD was already very close to optimal at
this rank. The 90%-energy auto-selection deliberately under-selects (`[2,2,2]`)
because a random 5-dimensional subspace's singular values don't decay sharply
until index 5 — 90% retained energy per mode is not enough to capture the
whole signal. Raising the threshold to 99.9% correctly recovers the true rank
`[5,5,5]` with zero error. This is the real, interpretable tradeoff
energy-based rank selection gives you — see `USER_GUIDE.md` §6.2 for why the
total error is bounded by, not equal to, the per-mode discarded energy.

---

## Tutorial 5 — Tensor Train: TT-SVD, Compression Ratio, TT-Rounding

TT represents an N-way tensor as a chain of 3-way cores — the standard choice
once you're past ~5-6 modes. This tutorial builds a 4D tensor with a known
TT-rank of 4 (by constructing TT cores directly and expanding them to dense),
recovers it with `tt_svd`, then uses `tt_round` to recompress to a smaller
rank budget and observes the resulting compression/accuracy tradeoff.

```rust
//! Tutorial 5: Tensor Train -- TT-SVD, compression ratio, TT-rounding.
use scirs2_core::ndarray_ext::Array3;
use tenrso_core::DenseND;
use tenrso_decomp::tt::{tt_round, tt_svd, TTDecomp};

/// Build a random TT core of shape (r_left, mode_size, r_right).
fn random_core(r_left: usize, mode_size: usize, r_right: usize) -> Array3<f64> {
    let flat = DenseND::<f64>::random_uniform(&[r_left, mode_size, r_right], -1.0, 1.0);
    Array3::from_shape_vec((r_left, mode_size, r_right), flat.as_slice().to_vec())
        .expect("shape matches flat data length")
}

fn main() -> anyhow::Result<()> {
    let shape = vec![12usize, 12, 12, 12];
    let true_rank = 4usize;

    // Build a tensor with a *known* TT-rank of 4 by constructing TT cores
    // directly and expanding them to dense. This gives TT-SVD a real
    // low-rank structure to recover (instead of fitting incompressible noise).
    let cores = vec![
        random_core(1, shape[0], true_rank),
        random_core(true_rank, shape[1], true_rank),
        random_core(true_rank, shape[2], true_rank),
        random_core(true_rank, shape[3], 1),
    ];
    let ground_truth_tt = TTDecomp {
        cores,
        ranks: vec![true_rank, true_rank, true_rank],
        shape: shape.clone(),
        error: None,
    };
    let tensor = ground_truth_tt.reconstruct()?;
    println!("True TT-rank: {true_rank} (uniform across {} interior bonds)\n", shape.len() - 1);

    // 1. TT-SVD with a generous max rank and tight tolerance: should recover
    //    the true rank-4 structure almost exactly.
    let max_ranks = vec![8, 8, 8];
    let tt = tt_svd(&tensor, &max_ranks, 1e-8)?;

    println!("TT-SVD on shape {:?} (max_ranks={:?})", shape, max_ranks);
    println!("  recovered TT-ranks: {:?}", tt.ranks);
    for (i, core) in tt.cores.iter().enumerate() {
        println!("  core[{i}] shape: {:?}", core.shape());
    }
    println!("  compression ratio: {:.2}x", tt.compression_ratio());

    let recon = tt.reconstruct()?;
    let err = (&tensor - &recon).frobenius_norm() / tensor.frobenius_norm();
    println!("  relative reconstruction error: {:.6}", err);

    // 2. TT-rounding: recompress to a smaller rank budget and see the tradeoff.
    let round_ranks = vec![2, 2, 2];
    let tt_rounded = tt_round(&tt, &round_ranks, 1e-10)?;
    println!("\nAfter tt_round to ranks {:?}:", round_ranks);
    println!("  new TT-ranks: {:?}", tt_rounded.ranks);
    println!("  compression ratio: {:.2}x", tt_rounded.compression_ratio());

    let recon_rounded = tt_rounded.reconstruct()?;
    let err_rounded = (&tensor - &recon_rounded).frobenius_norm() / tensor.frobenius_norm();
    println!("  relative reconstruction error: {:.6}", err_rounded);

    Ok(())
}
```

**Representative output** (the near-zero error at the true rank, and the
[2,2,2]-vs-larger-error tradeoff after rounding, are structural; the exact
compression ratio and the rounded error's later decimals vary a little run to
run since the ground truth is unseeded random data — the recovered TT-ranks
are consistently `[4, 4, 4]` at the `tol=1e-8` used here, matching the true
rank, though a looser or tighter tolerance can recover a slightly different
rank at one or more bonds, e.g. `[4, 5, 4]`, without changing the conclusion):

```text
True TT-rank: 4 (uniform across 3 interior bonds)

TT-SVD on shape [12, 12, 12, 12] (max_ranks=[8, 8, 8])
  recovered TT-ranks: [4, 4, 4]
  core[0] shape: [1, 12, 4]
  core[1] shape: [4, 12, 4]
  core[2] shape: [4, 12, 4]
  core[3] shape: [4, 12, 1]
  compression ratio: 43.20x
  relative reconstruction error: 0.000000

After tt_round to ranks [2, 2, 2]:
  new TT-ranks: [2, 2, 2]
  compression ratio: 144.00x
  relative reconstruction error: 0.785564
```

**What to notice:** at the true rank, TT-SVD gets a >40x compression ratio at
essentially zero error. Rounding down further to rank 2 more than triples the
compression ratio (144x) but at a real accuracy cost (~0.79 relative error) —
rank 2 genuinely cannot represent a rank-4 structure. This is the honest
compression/accuracy tradeoff `tt_round` lets you explore without
re-decomposing from the dense tensor each time.

---

## Tutorial 6 — Sparse Tensors: Build COO, Convert Formats, Masked Einsum

This tutorial builds a sparse matrix directly in COO (coordinate) format,
converts it to CSR for efficient row access, converts back to dense to
inspect it, then uses `masked_einsum` to compute only the diagonal of a dense
matmul — verifying the masked result against a full dense computation.

```rust
//! Tutorial 6: Sparse tensors -- build COO, convert formats, masked einsum.
use tenrso_core::DenseND;
use tenrso_sparse::mask::Mask;
use tenrso_sparse::masked_einsum::masked_einsum;
use tenrso_sparse::{CooTensor, CsrMatrix};

fn main() -> anyhow::Result<()> {
    // 1. Build a sparse 4x4 matrix directly in COO format.
    let indices = vec![vec![0, 0], vec![0, 2], vec![1, 1], vec![2, 0], vec![3, 3]];
    let values = vec![4.0, -1.0, 4.0, -1.0, 4.0];
    let shape = vec![4, 4];
    let coo = CooTensor::new(indices, values, shape.clone())?;
    println!(
        "COO: {} non-zeros out of {} ({:.1}% density)",
        coo.nnz(),
        shape.iter().product::<usize>(),
        coo.density() * 100.0
    );

    // 2. Convert COO -> CSR for efficient row access, and back to dense.
    let csr = CsrMatrix::from_coo(&coo)?;
    println!("CSR: {}x{}, {} non-zeros", csr.nrows(), csr.ncols(), csr.nnz());
    let dense = csr.to_dense()?;
    println!("Dense reconstruction:\n{:?}", dense);

    // 3. Masked einsum: only compute the output entries selected by a mask.
    let a = DenseND::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
    )?;
    let b = DenseND::from_vec(
        vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        &[3, 3],
    )?;

    // Only compute the diagonal of A @ B.
    let mask = Mask::from_indices(vec![vec![0, 0], vec![1, 1], vec![2, 2]], vec![3, 3])?;
    let masked_result = masked_einsum("ij,jk->ik", &[&a, &b], &mask)?;
    println!(
        "\nMasked einsum (diagonal only) produced {} non-zero output entries",
        masked_result.nnz()
    );
    for (idx, val) in masked_result
        .indices()
        .iter()
        .zip(masked_result.values().iter())
    {
        println!("  out{:?} = {}", idx, val);
    }

    // Cross-check the diagonal against a full dense matmul.
    let mut full = vec![0.0; 9];
    for i in 0..3 {
        for k in 0..3 {
            let mut s = 0.0;
            for j in 0..3 {
                s += a[&[i, j]] * b[&[j, k]];
            }
            full[i * 3 + k] = s;
        }
    }
    println!("Dense diagonal check: [{}, {}, {}]", full[0], full[4], full[8]);

    Ok(())
}
```

**Actual output** (deterministic — no randomness involved):

```text
COO: 5 non-zeros out of 16 (31.2% density)
CSR: 4x4, 5 non-zeros
Dense reconstruction:
DenseND { shape: [4, 4], rank: 2, data: [[4.0, 0.0, -1.0, 0.0],
 [0.0, 4.0, 0.0, 0.0],
 [-1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 4.0]], shape=[4, 4], strides=[4, 1], layout=Cc (0x5), dynamic ndim=2 }

Masked einsum (diagonal only) produced 3 non-zero output entries
  out[0, 0] = 30
  out[1, 1] = 69
  out[2, 2] = 90
Dense diagonal check: [30, 69, 90]
```

**What to notice:** `masked_einsum` returns a `CooTensor` with exactly 3
entries (one per masked output position) — it never materializes the other 6
entries of the 3×3 result — and those 3 values agree exactly with the
corresponding diagonal of a full dense matmul.

---

## Tutorial 7 — Out-of-Core: Write to Arrow, Read Back, Stream in Chunks

This tutorial writes a tensor to an Arrow IPC file, reads it back into a fresh
`DenseND`, and then processes the loaded tensor in row-chunks using
`ChunkSpec`/`ChunkIterator` — accumulating a running sum one chunk at a time
and cross-checking it against a direct full-tensor sum.

Because `tenrso-ooc` is a large, separately-versioned dependency (Arrow,
Parquet, mmap, compression, ...), it's worth keeping it in its own binary
crate rather than pulling it into every project that only needs
`tenrso-core`/`tenrso-exec`.

```rust
//! Tutorial 7: Out-of-core -- write a tensor to Arrow, read it back, and stream it
//! back in chunks using `ChunkSpec`/`ChunkIterator`.
use std::env;
use tenrso_core::DenseND;
use tenrso_ooc::arrow_io::{ArrowReader, ArrowWriter};
use tenrso_ooc::{ChunkSpec, ChunkIndex};

fn main() -> anyhow::Result<()> {
    let path = env::temp_dir().join("tenrso_tutorial7.arrow");

    // 1. Build a modest tensor and write it to an Arrow IPC file.
    let shape = [40usize, 25];
    let data: Vec<f64> = (0..shape[0] * shape[1]).map(|k| k as f64 * 0.5).collect();
    let tensor = DenseND::<f64>::from_vec(data, &shape)?;

    {
        let mut writer = ArrowWriter::new(&path)?;
        writer.write(&tensor)?;
        writer.finish()?;
    }
    println!("Wrote tensor {:?} to {:?}", tensor.shape(), path);

    // 2. Read it back into a fresh DenseND and verify equality.
    let loaded = {
        let mut reader = ArrowReader::open(&path)?;
        reader.read()?
    };
    println!("Read back shape: {:?}", loaded.shape());

    let mut max_diff = 0.0_f64;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let d = (tensor[&[i, j]] - loaded[&[i, j]]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    println!("max |original - roundtrip| = {:.3e}", max_diff);

    // 3. Stream the loaded tensor back in row-chunks using ChunkSpec/ChunkIterator,
    //    accumulating a running sum without materializing more than one chunk
    //    of row-indices at a time.
    let chunk_spec = ChunkSpec::tile_size(loaded.shape(), &[8, shape[1]])?;
    println!(
        "\nChunking {:?} into {} chunk(s) of up to {:?} rows each",
        loaded.shape(),
        chunk_spec.total_chunks(),
        chunk_spec.chunk_size()
    );

    let mut streamed_sum = 0.0_f64;
    let mut chunks_seen = 0usize;
    for chunk_idx in chunk_spec.iter() {
        let (start, end) = chunk_spec.chunk_bounds(&chunk_idx);
        let mut chunk_sum = 0.0_f64;
        for i in start[0]..end[0] {
            for j in start[1]..end[1] {
                chunk_sum += loaded[&[i, j]];
            }
        }
        println!(
            "  chunk {:?}: rows {}..{} -> partial sum {:.2}",
            ChunkIndex::new(chunk_idx.coords.clone()).coords,
            start[0],
            end[0],
            chunk_sum
        );
        streamed_sum += chunk_sum;
        chunks_seen += 1;
    }

    let direct_sum: f64 = loaded.as_slice().iter().sum();
    println!(
        "\nchunks processed: {chunks_seen}, streamed sum = {streamed_sum:.6}, direct sum = {direct_sum:.6}"
    );

    std::fs::remove_file(&path).ok();
    Ok(())
}
```

**Actual output** (deterministic — no randomness involved):

```text
Wrote tensor [40, 25] to "/tmp/tenrso_tutorial7.arrow"
Read back shape: [40, 25]
max |original - roundtrip| = 0.000e0

Chunking [40, 25] into 5 chunk(s) of up to [8, 25] rows each
  chunk [0, 0]: rows 0..8 -> partial sum 9950.00
  chunk [1, 0]: rows 8..16 -> partial sum 29950.00
  chunk [2, 0]: rows 16..24 -> partial sum 49950.00
  chunk [3, 0]: rows 24..32 -> partial sum 69950.00
  chunk [4, 0]: rows 32..40 -> partial sum 89950.00

chunks processed: 5, streamed sum = 249750.000000, direct sum = 249750.000000
```

**What to notice:** the Arrow round trip is exact (`max diff = 0`), and the
chunk-by-chunk streamed sum matches the direct full-tensor sum exactly — this
is the pattern to follow for processing a tensor larger than you want to hold
in one contiguous pass, whether that data came from Arrow, Parquet, or a
memory-mapped binary file (`tenrso_ooc::mmap_io::MmapTensor`). See
`USER_GUIDE.md` §8 for the honest scope of what "out-of-core" means here: both
`ArrowReader::read()` and `ParquetReader::read()` load the *whole* tensor back
in one call — chunking is something you apply afterward (or via
`MmapTensor`/`StreamingExecutor`), not something the file readers do for you
automatically.

---

## Tutorial 8 — Autodiff: Define a Contraction, Get Its VJP, Verify with Finite Differences

This tutorial computes `C = A @ B` as an explicit einsum contraction, gets the
vector-Jacobian product (gradient) with respect to both inputs assuming
`L = sum(C)`, and then verifies the analytical gradient against a
central-difference numerical estimate using the crate's built-in gradient
checker.

```rust
//! Tutorial 8: Autodiff -- define a contraction, get its VJP, verify with the
//! finite-difference gradient checker.
use tenrso_ad::gradcheck::{check_gradient, GradCheckConfig};
use tenrso_ad::vjp::{EinsumVjp, VjpOp};
use tenrso_core::DenseND;
use tenrso_exec::ops::execute_dense_contraction;
use tenrso_planner::EinsumSpec;

fn main() -> anyhow::Result<()> {
    // Forward pass: C = A @ B
    let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;

    let spec = EinsumSpec::parse("ij,jk->ik")?;
    let c = execute_dense_contraction(&spec, &a, &b)?;
    println!("C = A @ B:\n{:?}", c.as_array());

    // Backward pass: assume dL/dC = 1 everywhere (i.e. L = sum(C)).
    let grad_c = DenseND::ones(c.shape());
    let vjp_ctx = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
    let grads = vjp_ctx.vjp(&grad_c)?;
    let grad_a = &grads[0];
    let grad_b = &grads[1];
    println!("\ndL/dA:\n{:?}", grad_a.as_array());
    println!("dL/dB:\n{:?}", grad_b.as_array());

    // Verify dL/dA against a central-difference numerical gradient.
    let f = |x: &DenseND<f64>| execute_dense_contraction(&spec, x, &b);
    let df = |_x: &DenseND<f64>, grad_y: &DenseND<f64>| {
        let vjp = EinsumVjp::new(spec.clone(), a.clone(), b.clone());
        let grads = vjp.vjp(grad_y)?;
        Ok(grads[0].clone())
    };

    let config = GradCheckConfig {
        epsilon: 1e-5,
        rtol: 1e-3,
        atol: 1e-5,
        use_central_diff: true,
        verbose: true,
    };

    let result = check_gradient(f, df, &a, &grad_c, &config)?;
    println!(
        "\ngradient check: passed={}, max_abs_diff={:.3e}, max_rel_diff={:.3e}",
        result.passed, result.max_abs_diff, result.max_rel_diff
    );
    assert!(result.passed, "finite-difference gradient check must pass");

    Ok(())
}
```

**Actual output** (deterministic apart from floating-point noise in the last
1-2 significant digits of the finite-difference diffs, which will always be
comfortably within the `1e-9`-ish range shown below):

```text
C = A @ B:
[[19.0, 22.0],
 [43.0, 50.0]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2

dL/dA:
[[11.0, 15.0],
 [11.0, 15.0]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2
dL/dB:
[[4.0, 4.0],
 [6.0, 6.0]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2
✓ Gradient check passed!
  Max absolute difference: 6.49e-10
  Max relative difference: 5.90e-11

gradient check: passed=true, max_abs_diff=6.494e-10, max_rel_diff=5.903e-11
```

**What to notice:** `dL/dA[i,j] = sum_k B[j,k]` and `dL/dB[j,k] = sum_i A[i,j]`
by the standard matmul VJP rule (you can hand-verify: row sums of `B` are
`[11, 15]` matching every row of `dL/dA`; column sums of `A` are `[4, 6]`
matching every row of `dL/dB`). The finite-difference check confirms the
hand-written VJP rule matches a numerical gradient to ~10 significant digits —
this is the standard way to validate *any* new gradient rule you write against
this crate's operations.

---

## How these were verified

All eight programs above were written into scratch Cargo binaries, built with
`cargo build`, and run with `cargo run`, against an isolated, clean `git
worktree` snapshot of this repository's committed `HEAD` (rather than the live
working tree, which had unrelated in-progress edits to a few crates at
verification time — see the accompanying task report for detail). Output
blocks are pasted verbatim from those runs; only Tutorials 3, 4, and 5 involve
unseeded randomness in their *inputs* (the synthetic ground-truth tensors),
which is called out explicitly above each such output block.

---

## Where to go next

- [`USER_GUIDE.md`](USER_GUIDE.md) — the reference companion to this document:
  installation, feature flags, the crate map, core concepts, the einsum spec
  language's limits, the execution model, decompositions, sparse formats,
  out-of-core processing, automatic differentiation, the SciRS2-Core policy,
  and honest performance numbers.
- `crates/*/examples/*.rs` across the workspace — 464 example programs at last
  count, covering considerably more ground (randomized methods, incremental
  CP, gradient checkpointing, mixed-precision training, NUMA-aware OoC,
  iterative sparse solvers, and more) than any single guide can.
