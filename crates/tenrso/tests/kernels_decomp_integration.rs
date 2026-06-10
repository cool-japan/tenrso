//! Cross-crate integration tests: tenrso-kernels <-> tenrso-decomp
//!
//! Verifies that the Tucker and CP kernel functions produce numerically
//! consistent results when paired with Tucker and CP decomposition algorithms.
//! All array creation uses `DenseND` from `tenrso_core`; no direct `ndarray`
//! construction per SCIRS2_INTEGRATION_POLICY.md.
//!
//! ## Test Groups
//!
//! - **Group A (4 tests):** Tucker kernel <-> Tucker decomp consistency
//! - **Group B (3 tests):** CP kernel <-> CP decomp consistency
//! - **Group C (1 test):** MTTKRP variant equivalence

use tenrso_core::DenseND;
use tenrso_decomp::{
    cp::cp_als,
    tucker::{tucker_hooi, tucker_hosvd},
    InitStrategy,
};
use tenrso_kernels::{mttkrp, mttkrp_fused, tucker_reconstruct};

// ─── Helper: Frobenius norm of a DenseND tensor ─────────────────────────────

fn frob_norm(tensor: &DenseND<f64>) -> f64 {
    tensor.view().iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Relative Frobenius reconstruction error ||X - Xhat||_F / ||X||_F.
///
/// Returns the ratio; returns 0.0 if the reference tensor has zero norm.
fn relative_reconstruction_error(original: &DenseND<f64>, reconstructed: &DenseND<f64>) -> f64 {
    let orig_view = original.view();
    let recon_view = reconstructed.view();

    let mut diff_sq = 0.0_f64;
    let mut orig_sq = 0.0_f64;

    for (&o, &r) in orig_view.iter().zip(recon_view.iter()) {
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
// Group A: Tucker kernel <-> Tucker decomp consistency
// ────────────────────────────────────────────────────────────────────────────

/// Test 1 -- `tucker_reconstruct` (kernel) produces the correct output shape
/// from a Tucker-HOSVD decomposition and the error is finite and < 1.0.
#[test]
fn test_tucker_hosvd_reconstruct_via_kernel() {
    let tensor = DenseND::<f64>::random_uniform(&[8, 6, 5], 0.0, 1.0);
    let tucker = tucker_hosvd(&tensor, &[4, 3, 3])
        .expect("tucker_hosvd should succeed on valid input");

    let factor_views: Vec<_> = tucker.factors.iter().map(|f| f.view()).collect();
    let core_view = tucker.core.view();

    let reconstructed_arr = tucker_reconstruct(&core_view, &factor_views)
        .expect("tucker_reconstruct kernel should succeed");

    // Shape must match original tensor
    assert_eq!(
        reconstructed_arr.shape(),
        tensor.shape(),
        "reconstructed shape must match original tensor shape"
    );

    // Wrap in DenseND for error computation
    let reconstructed = DenseND::from_array(reconstructed_arr);

    let err = relative_reconstruction_error(&tensor, &reconstructed);

    // Sanity: error must be finite (computation succeeded)
    assert!(err.is_finite(), "relative error must be finite, got {}", err);

    // For rank [4,3,3] on a [8,6,5] tensor, error must be < 1.0
    // (at minimum the decomposition is not worse than predicting zeros)
    assert!(
        err < 1.0,
        "relative Tucker-HOSVD reconstruction error {} must be < 1.0",
        err
    );
}

/// Test 2 -- Tucker-HOOI (iterative) must give reconstruction error <= HOSVD
/// at the same ranks.  HOOI is designed to minimise the approximation error
/// monotonically, so it must not be strictly worse than HOSVD.
#[test]
fn test_tucker_hooi_kernel_vs_hosvd_error() {
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);

    let hosvd = tucker_hosvd(&tensor, &[3, 3, 3])
        .expect("tucker_hosvd should succeed");
    let hooi = tucker_hooi(&tensor, &[3, 3, 3], 20, 1e-6)
        .expect("tucker_hooi should succeed");

    // Helper closure: reconstruct using the kernel and compute relative error
    let compute_error =
        |decomp: &tenrso_decomp::tucker::TuckerDecomp<f64>| -> f64 {
            let factor_views: Vec<_> = decomp.factors.iter().map(|f| f.view()).collect();
            let core_view = decomp.core.view();
            let recon_arr = tucker_reconstruct(&core_view, &factor_views)
                .expect("tucker_reconstruct should succeed for a valid decomposition");
            let recon = DenseND::from_array(recon_arr);
            relative_reconstruction_error(&tensor, &recon)
        };

    let hosvd_err = compute_error(&hosvd);
    let hooi_err = compute_error(&hooi);

    assert!(
        hooi_err <= hosvd_err + 1e-6,
        "Tucker-HOOI error ({}) must be <= Tucker-HOSVD error ({}) + 1e-6",
        hooi_err,
        hosvd_err
    );
}

/// Test 3 -- `TuckerDecomp::reconstruct()` (from decomp crate) must produce
/// output numerically identical to `tucker_reconstruct` (from kernels crate),
/// because they invoke the same underlying operations.
#[test]
fn test_tucker_operator_matches_decomp_reconstruct() {
    let tensor = DenseND::<f64>::random_uniform(&[5, 4, 6], 0.0, 1.0);
    let tucker = tucker_hosvd(&tensor, &[3, 3, 4])
        .expect("tucker_hosvd should succeed");

    // Path A: kernels tucker_reconstruct
    let factor_views_a: Vec<_> = tucker.factors.iter().map(|f| f.view()).collect();
    let core_view_a = tucker.core.view();
    let kernel_recon_arr = tucker_reconstruct(&core_view_a, &factor_views_a)
        .expect("tucker_reconstruct (kernel) should succeed");
    let kernel_recon = DenseND::from_array(kernel_recon_arr);

    // Path B: TuckerDecomp::reconstruct() from decomp
    let decomp_recon = tucker
        .reconstruct()
        .expect("TuckerDecomp::reconstruct should succeed");

    // Both paths must agree to machine precision (they use the same ops)
    let kr_view = kernel_recon.view();
    let dr_view = decomp_recon.view();

    for (&a, &b) in kr_view.iter().zip(dr_view.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "kernel_recon and decomp_recon values differ by more than 1e-10: {} vs {}",
            a,
            b
        );
    }
}

/// Test 4 -- Verify factor and core shapes from Tucker-HOSVD, then confirm
/// that `tucker_reconstruct` restores the original shape.
#[test]
fn test_tucker_full_pipeline_shapes() {
    let tensor = DenseND::<f64>::random_uniform(&[10, 8, 6], 0.0, 1.0);
    let tucker = tucker_hosvd(&tensor, &[5, 4, 3])
        .expect("tucker_hosvd should succeed");

    // Core must have the requested shape
    assert_eq!(
        tucker.core.shape(),
        &[5, 4, 3],
        "core shape mismatch"
    );

    // Factor shapes: (mode_size, rank)
    assert_eq!(tucker.factors[0].shape(), [10, 5], "factor-0 shape mismatch");
    assert_eq!(tucker.factors[1].shape(), [8, 4], "factor-1 shape mismatch");
    assert_eq!(tucker.factors[2].shape(), [6, 3], "factor-2 shape mismatch");

    // Reconstructed tensor must have original shape
    let factor_views: Vec<_> = tucker.factors.iter().map(|f| f.view()).collect();
    let core_view = tucker.core.view();
    let reconstructed = tucker_reconstruct(&core_view, &factor_views)
        .expect("tucker_reconstruct should succeed");

    assert_eq!(
        reconstructed.shape(),
        &[10usize, 8, 6],
        "reconstructed shape must match original tensor"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Group B: CP kernel <-> CP decomp consistency
// ────────────────────────────────────────────────────────────────────────────

/// Test 5 -- After CP-ALS, compute MTTKRP for every mode using the kernel.
/// Verify shapes and that all values are finite.
#[test]
fn test_cp_als_mttkrp_shapes() {
    let tensor = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
    let cp = cp_als(&tensor, 3, 10, 1e-4, InitStrategy::Random, None)
        .expect("cp_als should succeed on a well-shaped tensor");

    let factor_views: Vec<_> = cp.factors.iter().map(|f| f.view()).collect();
    let tensor_view = tensor.view();

    for (mode, &mode_size) in tensor.shape().iter().enumerate() {
        let result = mttkrp(&tensor_view, &factor_views, mode)
            .expect("mttkrp should succeed for a valid CP decomposition");

        assert_eq!(
            result.shape(),
            [mode_size, 3],
            "MTTKRP mode-{} shape must be [{}, 3]",
            mode,
            mode_size
        );

        assert!(
            result.iter().all(|v| v.is_finite()),
            "MTTKRP mode-{} result contains non-finite values",
            mode
        );
    }
}

/// Test 6 -- `mttkrp_fused` must agree with standard `mttkrp` to within 1e-10
/// on the CP factor matrices.  Both implement the same mathematical operation
/// (MTTKRP) with different algorithmic strategies.
#[test]
fn test_mttkrp_fused_equals_standard_on_cp_factors() {
    let tensor = DenseND::<f64>::random_uniform(&[5, 6, 7], 0.0, 1.0);
    let cp = cp_als(&tensor, 4, 15, 1e-4, InitStrategy::Random, None)
        .expect("cp_als should succeed on a well-shaped tensor");

    let factor_views: Vec<_> = cp.factors.iter().map(|f| f.view()).collect();
    let tensor_view = tensor.view();

    for mode in 0..3 {
        let standard = mttkrp(&tensor_view, &factor_views, mode)
            .expect("mttkrp standard should succeed");
        let fused = mttkrp_fused(&tensor_view, &factor_views, mode)
            .expect("mttkrp_fused should succeed");

        assert_eq!(
            standard.shape(),
            fused.shape(),
            "shapes must match for mode {}",
            mode
        );

        for (&s, &f) in standard.iter().zip(fused.iter()) {
            assert!(
                (s - f).abs() < 1e-10,
                "mttkrp_fused differs from mttkrp by more than 1e-10 at mode {}: {} vs {}",
                mode,
                s,
                f
            );
        }
    }
}

/// Test 7 -- More CP-ALS iterations must yield better or equal fit.
/// Uses a fixed-seed-like approach: same `InitStrategy::Random` but
/// `tol = 1e-10` forces full iterations without early stopping.
/// Also verifies factor shapes are valid.
#[test]
fn test_cp_reconstruction_error_decreases_with_iterations() {
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);

    let cp_5 = cp_als(&tensor, 4, 5, 1e-10, InitStrategy::Random, None)
        .expect("cp_als (5 iters) should succeed");
    let cp_50 = cp_als(&tensor, 4, 50, 1e-10, InitStrategy::Random, None)
        .expect("cp_als (50 iters) should succeed");

    // More iterations should produce a fit value that is at least as good.
    // fit = 1 - ||X - Xhat|| / ||X||, so higher is better.
    assert!(
        cp_50.fit >= cp_5.fit - 1e-6,
        "50-iteration CP fit ({}) must be >= 5-iteration fit ({}) - 1e-6",
        cp_50.fit,
        cp_5.fit
    );

    // Factor shapes must be consistent for the 50-iter run
    for (i, factor) in cp_50.factors.iter().enumerate() {
        assert_eq!(
            factor.shape()[1],
            4,
            "cp_50 factor-{} rank must be 4",
            i
        );
        assert_eq!(
            factor.shape()[0],
            tensor.shape()[i],
            "cp_50 factor-{} row count must match mode size",
            i
        );
    }

    // Factor shapes must be consistent for the 5-iter run
    for (i, factor) in cp_5.factors.iter().enumerate() {
        assert_eq!(
            factor.shape()[1],
            4,
            "cp_5 factor-{} rank must be 4",
            i
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Group C: MTTKRP variant consistency -- blocked vs standard
// ────────────────────────────────────────────────────────────────────────────

/// Test 8 -- `mttkrp_blocked` must agree with `mttkrp` on factor matrices
/// derived from Tucker HOSVD decomp with equal ranks across all modes.
///
/// Tucker factor matrices at equal rank form well-conditioned, orthogonal
/// column bases -- a harder test than random matrices because they span the
/// actual tensor subspace.
#[test]
fn test_mttkrp_blocked_equals_standard_on_tucker_factors() {
    use tenrso_kernels::mttkrp_blocked;

    let tensor = DenseND::<f64>::random_uniform(&[5, 6, 7], 0.0, 1.0);
    // Use equal ranks so all factor matrices have the same number of columns (= 3),
    // which is required for MTTKRP (all factors must share the same "CP rank").
    let tucker = tucker_hosvd(&tensor, &[3, 3, 3])
        .expect("tucker_hosvd should succeed");

    // All factors have shape [mode_size, 3] -- column counts are equal (3).
    let factor_views: Vec<_> = tucker.factors.iter().map(|f| f.view()).collect();
    let tensor_view = tensor.view();

    for mode in 0..3 {
        let standard = mttkrp(&tensor_view, &factor_views, mode)
            .expect("mttkrp standard should succeed");
        let blocked = mttkrp_blocked(&tensor_view, &factor_views, mode, 4)
            .expect("mttkrp_blocked should succeed");

        assert_eq!(
            standard.shape(),
            blocked.shape(),
            "shapes must match for mode {}",
            mode
        );

        for (&s, &b) in standard.iter().zip(blocked.iter()) {
            assert!(
                (s - b).abs() < 1e-10,
                "mttkrp_blocked differs from mttkrp by more than 1e-10 at mode {}: {} vs {}",
                mode,
                s,
                b
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Smoke test: verify frob_norm helper is consistent
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_frob_norm_helper_identity_matrix() {
    // The 3x3 identity matrix has Frobenius norm sqrt(3)
    let data: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let tensor = DenseND::from_vec(data, &[3, 3]).expect("from_vec should succeed");
    let norm = frob_norm(&tensor);
    assert!(
        (norm - (3.0_f64).sqrt()).abs() < 1e-10,
        "Frobenius norm of 3x3 identity must be sqrt(3), got {}",
        norm
    );
}
