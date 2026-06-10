//! Comprehensive regression test suite for TenRSo
//!
//! Tests full end-to-end pipelines across multiple crates, covering milestones M0-M6.
//! Each group exercises pipelines NOT covered by `kernels_decomp_integration.rs`.
//!
//! ## Test Groups
//!
//! - **Group 1 (2 tests):** TT-SVD end-to-end (decompose, round, reconstruct)
//! - **Group 2 (1 test):** OoC Arrow IPC round-trip
//! - **Group 3 (2 tests):** Planner + executor integration
//! - **Group 4 (1 test):** Sparse CP pipeline (COO → cp_als_sparse → fit check)
//! - **Group 5 (1 test):** AD gradient consistency for MatMul via ComputationGraph
//!
//! ## SciRS2 Policy
//!
//! All array creation uses `DenseND` or `scirs2_core::ndarray_ext`.
//! No direct `ndarray` imports; no direct `rand` imports.

use scirs2_core::ndarray_ext::Array3;
use tenrso_core::{DenseND, TensorHandle};
use tenrso_decomp::tt::{tt_round, tt_svd};
use tenrso_exec::{einsum_ex, CpuExecutor, ExecHints, TenrsoExecutor};
use tenrso_planner::{greedy_planner, AdaptivePlanner, EinsumSpec, PlanHints, Planner};

// ─── Helper: relative Frobenius reconstruction error ────────────────────────

/// ||X - X_hat||_F / ||X||_F.  Returns 0.0 if ||X||_F < 1e-30.
fn relative_error(original: &DenseND<f64>, reconstructed: &DenseND<f64>) -> f64 {
    let ov = original.view();
    let rv = reconstructed.view();
    let mut diff_sq = 0.0_f64;
    let mut orig_sq = 0.0_f64;
    for (&o, &r) in ov.iter().zip(rv.iter()) {
        let d = o - r;
        diff_sq += d * d;
        orig_sq += o * o;
    }
    if orig_sq < 1e-30 {
        return 0.0;
    }
    (diff_sq / orig_sq).sqrt()
}

// ────────────────────────────────────────────────────────────────────────────
// Group 1: TT-SVD end-to-end
// ────────────────────────────────────────────────────────────────────────────

/// Build a 4D tensor of shape [3,3,3,3] with deterministic values,
/// decompose via `tt_svd` (max_rank=9, eps=1e-10), then apply `tt_round`
/// with the same max_ranks and eps=1e-6 (lossless rounding), reconstruct,
/// and verify relative error < 0.01.
///
/// The [3,3,3,3] tensor has a mode-0 unfolding of shape (3, 27) → max rank 3,
/// mode-1 unfolding (9, 9) → max rank 9, mode-2 unfolding (27, 3) → max rank 3.
/// Using max_ranks=[9,9,9] ensures no truncation beyond the natural rank.
#[test]
fn regression_tt_svd_round_trip() {
    // Deterministic: fill with sine of index (avoids exact low-rank structure)
    let total = 3 * 3 * 3 * 3; // 81
    let data: Vec<f64> = (0..total)
        .map(|i| (i as f64 * 0.3).sin() + (i as f64 * 0.07).cos())
        .collect();
    let tensor = DenseND::<f64>::from_vec(data, &[3, 3, 3, 3]).expect("from_vec [3,3,3,3]");

    // TT-SVD: 4 modes → max_ranks has length 3 (N-1 interior ranks)
    // Use max_rank=9 and near-zero tolerance for a near-exact decomposition
    let max_ranks = vec![9, 9, 9];
    let tt = tt_svd(&tensor, &max_ranks, 1e-10).expect("tt_svd should succeed on [3,3,3,3]");

    // Verify TT structure invariants
    assert_eq!(tt.cores.len(), 4, "TT should have 4 cores for a 4D tensor");
    assert_eq!(
        tt.ranks.len(),
        3,
        "TT should have 3 interior ranks for a 4D tensor"
    );
    assert_eq!(tt.shape, vec![3, 3, 3, 3], "TT shape mismatch");

    // TT-round: apply rounding with the same max_ranks to verify round-trip stability
    let tt_rounded =
        tt_round(&tt, &max_ranks, 1e-8).expect("tt_round should succeed on valid TTDecomp");

    // Verify rounded TT structure is still valid
    assert_eq!(
        tt_rounded.cores.len(),
        4,
        "Rounded TT should still have 4 cores"
    );

    // Reconstruct
    let reconstructed = tt_rounded
        .reconstruct()
        .expect("TTDecomp::reconstruct should succeed after rounding");

    assert_eq!(
        reconstructed.shape(),
        tensor.shape(),
        "reconstructed shape must match original"
    );

    let err = relative_error(&tensor, &reconstructed);
    assert!(err.is_finite(), "relative error must be finite, got {err}");
    assert!(
        err < 0.01,
        "TT-SVD + round-trip relative error {err:.6} must be < 0.01"
    );
}

/// Build a 3-core TT manually (shapes [1,4,2], [2,4,2], [2,4,1]),
/// expand to a dense tensor via `TTDecomp::reconstruct()`, then
/// call `tt_svd` on the expansion and verify reconstruction error < 0.05.
#[test]
fn regression_tt_reconstruct_quality() {
    // Build 3 TT cores deterministically using simple patterns
    // Core 0: shape [1, 4, 2]
    let mut core0 = Array3::<f64>::zeros((1, 4, 2));
    for i in 0..4 {
        core0[[0, i, 0]] = (i + 1) as f64 * 0.5;
        core0[[0, i, 1]] = (i + 1) as f64 * 0.25;
    }

    // Core 1: shape [2, 4, 2]
    let mut core1 = Array3::<f64>::zeros((2, 4, 2));
    for r in 0..2 {
        for i in 0..4 {
            core1[[r, i, 0]] = ((r * 4 + i) as f64 + 1.0) * 0.3;
            core1[[r, i, 1]] = ((r * 4 + i) as f64 + 1.0) * 0.15;
        }
    }

    // Core 2: shape [2, 4, 1]
    let mut core2 = Array3::<f64>::zeros((2, 4, 1));
    for r in 0..2 {
        for i in 0..4 {
            core2[[r, i, 0]] = ((r * 4 + i) as f64 + 1.0) * 0.2;
        }
    }

    let manual_tt = tenrso_decomp::tt::TTDecomp {
        cores: vec![core0, core1, core2],
        ranks: vec![2, 2],
        shape: vec![4, 4, 4],
        error: None,
    };

    // Expand to dense
    let dense = manual_tt
        .reconstruct()
        .expect("manual TTDecomp::reconstruct should succeed");

    assert_eq!(dense.shape(), &[4, 4, 4], "dense expansion shape mismatch");

    // Now decompose the expanded tensor with tt_svd and check quality
    let tt_recovered = tt_svd(&dense, &[4, 4], 1e-6)
        .expect("tt_svd on manually-constructed dense tensor should succeed");

    let recon = tt_recovered
        .reconstruct()
        .expect("TTDecomp::reconstruct on recovered TT should succeed");

    let err = relative_error(&dense, &recon);
    assert!(err.is_finite(), "relative error must be finite, got {err}");
    assert!(
        err < 0.05,
        "TT reconstruction relative error {err:.6} must be < 0.05"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Group 2: OoC Arrow IPC round-trip
// ────────────────────────────────────────────────────────────────────────────

/// Write a DenseND<f64> tensor of shape [4,5] to an Arrow IPC file in
/// `std::env::temp_dir()`, read it back, and verify element-wise equality < 1e-12.
#[test]
fn regression_ooc_arrow_round_trip() {
    use std::env;
    use tenrso_ooc::arrow_io::{ArrowReader, ArrowWriter};

    let temp_path = env::temp_dir().join("regression_arrow_4x5.arrow");

    // Build a [4,5] tensor with known values: entry [i,j] = i*5.0 + j as f64
    let data: Vec<f64> = (0..20).map(|k| k as f64 * 1.5 + 0.1).collect();
    let original =
        DenseND::<f64>::from_vec(data, &[4, 5]).expect("DenseND::from_vec [4,5] should succeed");

    // Write
    {
        let mut writer =
            ArrowWriter::new(&temp_path).expect("ArrowWriter::new should succeed for temp path");
        writer
            .write(&original)
            .expect("ArrowWriter::write should succeed");
        writer.finish().expect("ArrowWriter::finish should succeed");
    }

    // Read
    let loaded = {
        let mut reader = ArrowReader::open(&temp_path)
            .expect("ArrowReader::open should succeed for written file");
        reader
            .read()
            .expect("ArrowReader::read should return valid DenseND<f64>")
    };

    // Cleanup regardless of outcome
    std::fs::remove_file(&temp_path).ok();

    // Verify shape
    assert_eq!(
        loaded.shape(),
        original.shape(),
        "round-tripped shape mismatch: {:?} vs {:?}",
        loaded.shape(),
        original.shape()
    );

    // Verify element-wise equality
    let ov = original.view();
    let lv = loaded.view();
    for i in 0..4 {
        for j in 0..5 {
            let diff = (ov[[i, j]] - lv[[i, j]]).abs();
            assert!(
                diff < 1e-12,
                "Arrow round-trip mismatch at [{i},{j}]: original={:.15}, loaded={:.15}, diff={diff:.2e}",
                ov[[i, j]],
                lv[[i, j]]
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Group 3: Planner + executor integration
// ────────────────────────────────────────────────────────────────────────────

/// Plan "ij,jk->ik" with `greedy_planner` for shapes [4,6]×[6,5],
/// then execute via `CpuExecutor::einsum` on sequential-filled tensors.
/// Verify result against hand-computed matrix multiplication (tolerance 1e-10).
#[test]
fn regression_planner_exec_matmul() {
    // ── Planning ─────────────────────────────────────────────────────────────
    let spec = EinsumSpec::parse("ij,jk->ik").expect("EinsumSpec::parse should succeed");
    let shapes = vec![vec![4, 6], vec![6, 5]];
    let hints = PlanHints::default();
    let plan = greedy_planner(&spec, &shapes, &hints).expect("greedy_planner should succeed");

    // A 2-input spec should produce exactly 1 contraction node
    assert!(
        !plan.nodes.is_empty(),
        "plan should have at least one node, got 0"
    );
    assert!(
        plan.estimated_flops > 0.0,
        "plan estimated_flops must be positive"
    );

    // ── Tensor construction: A[4,6] and B[6,5] with sequential values ────────
    // A[i,j] = (i*6 + j + 1) as f64
    let a_data: Vec<f64> = (0..24).map(|k| k as f64 + 1.0).collect();
    let b_data: Vec<f64> = (0..30).map(|k| k as f64 + 1.0).collect();

    let a = DenseND::<f64>::from_vec(a_data, &[4, 6]).expect("DenseND A from_vec");
    let b = DenseND::<f64>::from_vec(b_data, &[6, 5]).expect("DenseND B from_vec");

    // ── Execution via CpuExecutor ─────────────────────────────────────────────
    let handle_a = TensorHandle::from_dense_auto(a.clone());
    let handle_b = TensorHandle::from_dense_auto(b.clone());

    let mut executor = CpuExecutor::new();
    let result_handle = executor
        .einsum("ij,jk->ik", &[handle_a, handle_b], &ExecHints::default())
        .expect("CpuExecutor::einsum should succeed for 'ij,jk->ik'");

    let result_dense = result_handle
        .as_dense()
        .expect("result should be a dense tensor");

    assert_eq!(result_dense.shape(), &[4, 5], "result shape must be [4,5]");

    // ── Hand-compute expected result ─────────────────────────────────────────
    // C[i,k] = sum_j A[i,j] * B[j,k]
    let av = a.view();
    let bv = b.view();
    let rv = result_dense.view();

    for i in 0..4 {
        for k in 0..5 {
            let expected: f64 = (0..6).map(|j| av[[i, j]] * bv[[j, k]]).sum();
            let actual = rv[[i, k]];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-10,
                "matmul mismatch at [{i},{k}]: expected={expected:.6}, actual={actual:.6}, diff={diff:.2e}"
            );
        }
    }
}

/// Plan "ijk,jkl,lm->im" with `AdaptivePlanner` for shapes [3,4,5]×[4,5,6]×[6,7].
/// Assert the plan has at least 2 contraction steps. Execute via `CpuExecutor`
/// on random (but deterministic LCG) tensors and verify the result is finite
/// with shape [3,7].
#[test]
fn regression_adaptive_planner_3tensor() {
    // ── Planning ─────────────────────────────────────────────────────────────
    let planner = AdaptivePlanner::new();
    let shapes = vec![vec![3, 4, 5], vec![4, 5, 6], vec![6, 7]];
    let hints = PlanHints::default();
    let plan = planner
        .make_plan("ijk,jkl,lm->im", &shapes, &hints)
        .expect("AdaptivePlanner::make_plan should succeed for 3-tensor spec");

    // 3 inputs → at least 2 binary contraction steps
    assert!(
        plan.nodes.len() >= 2,
        "plan for 3-tensor contraction must have >= 2 nodes, got {}",
        plan.nodes.len()
    );

    // ── Tensor construction with deterministic LCG values ────────────────────
    // LCG: state = state * 1664525 + 1013904223 (mod 2^32), value = state / 2^32
    fn lcg_fill(size: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..size)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 16) & 0xFFFF) as f64 / 65535.0
            })
            .collect()
    }

    let a_data = lcg_fill(3 * 4 * 5, 42);
    let b_data = lcg_fill(4 * 5 * 6, 137);
    let c_data = lcg_fill(6 * 7, 7);

    let ta = DenseND::<f64>::from_vec(a_data, &[3, 4, 5]).expect("DenseND A [3,4,5]");
    let tb = DenseND::<f64>::from_vec(b_data, &[4, 5, 6]).expect("DenseND B [4,5,6]");
    let tc = DenseND::<f64>::from_vec(c_data, &[6, 7]).expect("DenseND C [6,7]");

    // ── Execution ─────────────────────────────────────────────────────────────
    let ha = TensorHandle::from_dense_auto(ta);
    let hb = TensorHandle::from_dense_auto(tb);
    let hc = TensorHandle::from_dense_auto(tc);

    let result_handle = einsum_ex::<f64>("ijk,jkl,lm->im")
        .inputs(&[ha, hb, hc])
        .hints(&ExecHints::default())
        .run()
        .expect("einsum_ex 'ijk,jkl,lm->im' should succeed");

    let result_dense = result_handle
        .as_dense()
        .expect("result should be a dense tensor");

    assert_eq!(result_dense.shape(), &[3, 7], "result shape must be [3,7]");

    // All values must be finite
    let rv = result_dense.view();
    for &v in rv.iter() {
        assert!(v.is_finite(), "result contains non-finite value: {v}");
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Group 4: Sparse CP pipeline
// ────────────────────────────────────────────────────────────────────────────

/// Build a 5×6×7 rank-2 dense tensor from outer products using deterministic
/// LCG PRNG, convert to COO, run `cp_als_sparse` (rank=2, iters=50, tol=1e-4),
/// and assert fit > 0.5 and reconstructed shape = [5,6,7].
#[cfg(feature = "sparse")]
#[test]
fn regression_sparse_cp_round_trip() {
    use tenrso_decomp::cp::{cp_als_sparse, InitStrategy};
    use tenrso_sparse::coo::CooTensor;

    let shape = [5usize, 6, 7];
    let total: usize = shape.iter().product(); // 210

    // Deterministic LCG factor vectors for rank-2 construction
    fn lcg_vec(size: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..size)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                0.1 + ((state >> 16) & 0xFFFF) as f64 / 65535.0 * 0.9
            })
            .collect()
    }

    // Two sets of factor vectors for rank-2 outer product sum
    let u0_r0 = lcg_vec(shape[0], 11);
    let u1_r0 = lcg_vec(shape[1], 22);
    let u2_r0 = lcg_vec(shape[2], 33);
    let u0_r1 = lcg_vec(shape[0], 44);
    let u1_r1 = lcg_vec(shape[1], 55);
    let u2_r1 = lcg_vec(shape[2], 66);

    // Construct dense tensor = rank-1 component 0 + rank-1 component 1
    let mut dense_data = vec![0.0f64; total];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let idx = i * shape[1] * shape[2] + j * shape[2] + k;
                dense_data[idx] = u0_r0[i] * u1_r0[j] * u2_r0[k] + u0_r1[i] * u1_r1[j] * u2_r1[k];
            }
        }
    }

    // Convert dense tensor to COO (store all elements as "sparse")
    let mut indices = Vec::with_capacity(total);
    let mut values = Vec::with_capacity(total);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let idx = i * shape[1] * shape[2] + j * shape[2] + k;
                let v = dense_data[idx];
                // Only include elements above a small threshold (keep density ~100%)
                if v.abs() > 1e-15 {
                    indices.push(vec![i, j, k]);
                    values.push(v);
                }
            }
        }
    }
    let coo = CooTensor::<f64>::new(indices, values, shape.to_vec())
        .expect("CooTensor::new should succeed for valid rank-2 tensor entries");

    // Run sparse CP-ALS
    let cp = cp_als_sparse(
        &coo,
        2,    // rank
        50,   // max_iters
        1e-4, // tol
        InitStrategy::Random,
        None,
    )
    .expect("cp_als_sparse should succeed on a rank-2 COO tensor");

    // Fit must be reasonable (> 0.5 for a rank-2 tensor with rank-2 decomposition)
    assert!(
        cp.fit > 0.5,
        "sparse CP fit {:.4} must be > 0.5 for a rank-2 input with rank=2",
        cp.fit
    );

    // Reconstruct and verify shape
    let reconstructed = cp
        .reconstruct(&shape)
        .expect("CpDecomp::reconstruct should succeed");

    assert_eq!(
        reconstructed.shape(),
        &shape,
        "reconstructed shape mismatch: {:?} vs {:?}",
        reconstructed.shape(),
        shape
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Group 5: AD gradient consistency
// ────────────────────────────────────────────────────────────────────────────

/// Build a 2×3 matrix A and 3×2 matrix B in the AD computation graph.
/// Compute C = A @ B (MatMul). Compute loss = sum(C). Run backward.
/// Verify grad_A[i,j] = sum_k(B[j,k]) = row sum of B^T,
/// which equals the column sum of B for column j (tolerance 1e-10).
#[test]
fn regression_ad_matmul_gradient() {
    use scirs2_core::ndarray_ext::{ArrayD, IxDyn};
    use tenrso_ad::graph::ComputationGraph;

    // A: 2×3, values 1..6
    let a_vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // B: 3×2, values 1..6
    let b_vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a_arr =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), a_vals.clone()).expect("A array construction");
    let b_arr =
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), b_vals.clone()).expect("B array construction");

    let graph = ComputationGraph::<f64>::new();

    let var_a = graph
        .variable(a_arr, true)
        .expect("graph.variable A should succeed");
    let var_b = graph
        .variable(b_arr, true)
        .expect("graph.variable B should succeed");

    // C = A @ B  (shape [2,2])
    let var_c = graph
        .matmul(&var_a, &var_b)
        .expect("graph.matmul should succeed for 2×3 @ 3×2");

    // loss = sum(C)  (scalar)
    let var_loss = graph
        .sum(&var_c)
        .expect("graph.sum should succeed for 2×2 matrix");

    // Backward pass
    graph
        .backward(&var_loss)
        .expect("graph.backward should succeed from scalar loss");

    // Retrieve gradients
    let grad_a = graph
        .gradient(&var_a)
        .expect("gradient for A should exist after backward");
    let grad_b = graph
        .gradient(&var_b)
        .expect("gradient for B should exist after backward");

    // ── Verify grad_A ─────────────────────────────────────────────────────────
    // loss = sum(C) = sum(A @ B)
    // d(loss)/d(A[i,j]) = sum_k (d C[i,k] / d A[i,j]) * 1
    //                   = sum_k B[j,k]   (since C[i,k] = sum_j A[i,j]*B[j,k])
    // So grad_A[i,j] = sum_k B[j,k] = B.sum(axis=1)[j]
    assert_eq!(grad_a.shape(), &[2, 3], "grad_A shape must be [2,3]");

    // B is [3,2]: B.sum(axis=1)[j] = B[j,0] + B[j,1]
    // Using b_vals = [1,2, 3,4, 5,6] (row-major)
    // B[0,*] = [1,2] → sum = 3
    // B[1,*] = [3,4] → sum = 7
    // B[2,*] = [5,6] → sum = 11
    let expected_grad_a_col: [f64; 3] = [3.0, 7.0, 11.0]; // indexed by j
    for i in 0..2 {
        for j in 0..3 {
            let diff = (grad_a[[i, j]] - expected_grad_a_col[j]).abs();
            assert!(
                diff < 1e-10,
                "grad_A[{i},{j}] = {:.10}, expected {:.10}, diff {diff:.2e}",
                grad_a[[i, j]],
                expected_grad_a_col[j]
            );
        }
    }

    // ── Verify grad_B ─────────────────────────────────────────────────────────
    // d(loss)/d(B[j,k]) = sum_i A[i,j]  (i.e. A.sum(axis=0)[j])
    // A is [2,3]: A.sum(axis=0)[j] = A[0,j] + A[1,j]
    // Using a_vals = [1,2,3, 4,5,6] (row-major)
    // A[:,0] = [1,4] → sum = 5
    // A[:,1] = [2,5] → sum = 7
    // A[:,2] = [3,6] → sum = 9
    // grad_B[j,k] = A[:,j].sum() for all k
    assert_eq!(grad_b.shape(), &[3, 2], "grad_B shape must be [3,2]");
    let expected_grad_b_row: [f64; 3] = [5.0, 7.0, 9.0]; // indexed by j (rows of B)
    for j in 0..3 {
        for k in 0..2 {
            let diff = (grad_b[[j, k]] - expected_grad_b_row[j]).abs();
            assert!(
                diff < 1e-10,
                "grad_B[{j},{k}] = {:.10}, expected {:.10}, diff {diff:.2e}",
                grad_b[[j, k]],
                expected_grad_b_row[j]
            );
        }
    }
}
