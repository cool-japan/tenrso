//! Tensor Train (TT) rank compression via SVD-based rounding.
//!
//! Implements the Oseledets (2011) TT-rounding algorithm:
//! 1. Left-to-right QR sweep — produces left-canonical form, concentrates norm in last core
//! 2. Right-to-left SVD sweep — truncates each bond using an absolute per-bond error threshold
//!
//! The total approximation error satisfies
//! `||TT_rounded - TT_original||_F ≤ ε · ||TT_original||_F`
//! for `tt_round`, and hard per-bond rank caps for `tt_truncate`.
//!
//! # References
//!
//! - Oseledets, I. V. (2011). "Tensor-Train Decomposition". SIAM J. Sci. Comput.
//! - Holtz, S., Rohwedder, T., & Schneider, R. (2012). "The Alternating Linear Scheme
//!   for Tensor Optimization in the TT Format".

use crate::error::{KernelError, KernelResult};
use crate::tt_orthog::tt_left_orthogonalize;
use scirs2_core::ndarray_ext::{s, Array1, Array2, Array3, ScalarOperand};
use scirs2_core::num_traits::{Float, NumAssign};
use std::iter::Sum;

// ─── Private helpers ────────────────────────────────────────────────────────

/// Determine the minimum rank that keeps the truncation error within `delta_sq`.
///
/// Singular values must be in **descending** order (as returned by LAPACK).
/// `delta_sq` is an *absolute* squared threshold — the sum of dropped singular
/// values squared must stay ≤ `delta_sq`.
fn determine_rank_from_delta<T>(
    singular_values: &Array1<T>,
    delta_sq: T,
    max_rank: Option<usize>,
) -> usize
where
    T: Float,
{
    let n = singular_values.len();
    if n == 0 {
        return 0;
    }

    // Greedily drop smallest singular values while tail energy stays within budget.
    let mut rank = n;
    let mut tail_energy = T::zero();

    for i in (0..n).rev() {
        let candidate = tail_energy + singular_values[i] * singular_values[i];
        if candidate <= delta_sq {
            tail_energy = candidate;
            rank = i;
        } else {
            break;
        }
    }

    rank = rank.max(1);
    if let Some(max_r) = max_rank {
        rank = rank.min(max_r);
    }
    rank.min(n)
}

/// Determine truncation rank based on singular values, a *relative* `epsilon_sq`,
/// and an optional `max_rank`.
///
/// The threshold is `epsilon_sq * total_energy`, where `total_energy = Σ σᵢ²`.
/// The rank `r` is chosen so that `Σ_{i>r} σᵢ² ≤ epsilon_sq · Σ σᵢ²`.
#[allow(dead_code)]
pub(crate) fn determine_truncation_rank<T>(
    singular_values: &Array1<T>,
    epsilon_sq: T,
    max_rank: Option<usize>,
) -> usize
where
    T: Float,
{
    let n = singular_values.len();
    if n == 0 {
        return 0;
    }

    let total_energy: T = singular_values
        .iter()
        .map(|&s| s * s)
        .fold(T::zero(), |a, b| a + b);

    if total_energy <= T::zero() {
        return 1.min(n);
    }

    let threshold = epsilon_sq * total_energy;
    let mut cumulative_tail_energy = T::zero();
    let mut rank = n;

    for i in (0..n).rev() {
        cumulative_tail_energy = cumulative_tail_energy + singular_values[i] * singular_values[i];
        if cumulative_tail_energy > threshold {
            rank = i + 1;
            break;
        }
    }

    rank = rank.max(1);
    if let Some(max_r) = max_rank {
        rank = rank.min(max_r);
    }
    rank.min(n)
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Round TT tensor using SVD-based rank compression with relative error control.
///
/// Implements the two-phase Oseledets TT-rounding algorithm:
/// 1. **Left-to-right QR sweep** — brings TT to left-canonical form.
/// 2. **Right-to-left SVD sweep** — truncates each bond using a per-bond absolute
///    error threshold derived from `ε · ||TT||_F / sqrt(d-1)`.
///
/// The guarantee is `||TT_rounded - TT||_F ≤ ε · ||TT||_F` for `ε > 0`.
///
/// # Arguments
///
/// * `cores` - TT cores to round (modified in-place)
/// * `max_rank` - Optional hard cap applied at every bond *after* epsilon truncation
/// * `epsilon` - Relative Frobenius norm error tolerance (non-negative)
///
/// # Errors
///
/// Returns an error if the core list is empty, `epsilon < 0`, or any internal
/// linear algebra call fails.
///
/// # Complexity
///
/// O(∑ᵢ rᵢ² nᵢ) for the QR sweep + O(∑ᵢ rᵢ³) for the SVD sweep.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array3;
/// use tenrso_kernels::tt_round::tt_round;
///
/// let core1 = Array3::<f64>::from_elem((1, 10, 8), 0.1);
/// let core2 = Array3::<f64>::from_elem((8, 10, 8), 0.1);
/// let core3 = Array3::<f64>::from_elem((8, 10, 1), 0.1);
/// let mut cores = vec![core1, core2, core3];
///
/// // Round with epsilon=1e-6, no max rank
/// tt_round(&mut cores, None, 1e-6).unwrap();
///
/// // Optionally constrain max rank as well
/// let mut cores2 = vec![
///     Array3::<f64>::from_elem((1, 10, 8), 0.1),
///     Array3::<f64>::from_elem((8, 10, 8), 0.1),
///     Array3::<f64>::from_elem((8, 10, 1), 0.1),
/// ];
/// tt_round(&mut cores2, Some(5), 1e-6).unwrap();
/// ```
pub fn tt_round<T>(cores: &mut [Array3<T>], max_rank: Option<usize>, epsilon: T) -> KernelResult<()>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    if cores.is_empty() {
        return Err(KernelError::empty_input("tt_round", "cores"));
    }

    if epsilon < T::zero() {
        return Err(KernelError::operation_error(
            "tt_round",
            "epsilon must be non-negative",
        ));
    }

    let d = cores.len();

    // Single-core TT: no bonds to compress; max_rank is irrelevant for r_0 = r_1 = 1.
    if d == 1 {
        return Ok(());
    }

    // ── Phase 1: left-to-right QR orthogonalization ──────────────────────────
    // After this, cores[0..d-2] are left-orthogonal, and ||TT||_F = ||cores[d-1]||_F.
    tt_left_orthogonalize(cores)?;

    // ── Compute total squared norm from the last core ─────────────────────────
    let norm_sq: T = cores[d - 1]
        .iter()
        .map(|&x| x * x)
        .fold(T::zero(), |a, b| a + b);

    // ── Per-bond absolute squared error threshold ─────────────────────────────
    // Distribute error evenly: delta_k = ε * ||TT||_F / sqrt(d-1)
    // => delta_k² = norm_sq * ε² / (d-1)
    let d_f = T::from(d - 1).ok_or_else(|| {
        KernelError::operation_error("tt_round", "Failed to convert (d-1) to scalar type")
    })?;
    let delta_sq = norm_sq * epsilon * epsilon / d_f;

    // ── Phase 2: right-to-left SVD with truncation ────────────────────────────
    for k in (1..d).rev() {
        let (r_left, n_k, r_right) = {
            let s = cores[k].shape();
            (s[0], s[1], s[2])
        };

        // Right-unfold: (r_left, n_k * r_right)
        let core_mat: Array2<T> = cores[k]
            .view()
            .to_shape((r_left, n_k * r_right))
            .map_err(|e| {
                KernelError::operation_error("tt_round", format!("right-unfold k={}: {}", k, e))
            })?
            .to_owned();

        let (u, s_vals, vt) = scirs2_linalg::svd(&core_mat.view(), false, None)
            .map_err(|e| KernelError::operation_error("tt_round", format!("SVD k={}: {}", k, e)))?;

        let new_rank = determine_rank_from_delta(&s_vals, delta_sq, max_rank);
        let actual_rank = new_rank.min(s_vals.len());

        // New G_k = Vt[:actual_rank, :].reshape(actual_rank, n_k, r_right)
        let vt_trunc: Array2<T> = vt.slice(s![..actual_rank, ..]).to_owned();
        cores[k] = vt_trunc
            .to_shape((actual_rank, n_k, r_right))
            .map_err(|e| {
                KernelError::operation_error("tt_round", format!("reshape Vt k={}: {}", k, e))
            })?
            .to_owned();

        // Transfer = U[:, :actual_rank] * diag(S[:actual_rank])
        let mut transfer: Array2<T> = u.slice(s![.., ..actual_rank]).to_owned();
        for (col_idx, &sv) in s_vals.iter().take(actual_rank).enumerate() {
            transfer.column_mut(col_idx).mapv_inplace(|x| x * sv);
        }

        // G_{k-1} ← G_{k-1}.reshape(r_prev * n_prev, r_left) @ transfer, reshaped back
        let prev_shape = cores[k - 1].shape().to_vec();
        let (r_prev_left, n_prev) = (prev_shape[0], prev_shape[1]);

        let prev_mat: Array2<T> = cores[k - 1]
            .view()
            .to_shape((r_prev_left * n_prev, r_left))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_round",
                    format!("reshape G_{{k-1}} k={}: {}", k, e),
                )
            })?
            .to_owned();

        cores[k - 1] = prev_mat
            .dot(&transfer)
            .to_shape((r_prev_left, n_prev, actual_rank))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_round",
                    format!("reshape new G_{{k-1}} k={}: {}", k, e),
                )
            })?
            .to_owned();
    }

    Ok(())
}

/// Truncate TT ranks to specified per-bond maximum values.
///
/// Applies the same two-phase algorithm as [`tt_round`] but uses hard per-bond
/// rank caps (`max_ranks[k]`) instead of an epsilon threshold.  Singular values
/// beyond `max_ranks[k]` are discarded, introducing the minimum possible error
/// relative to the TT's actual energy distribution.
///
/// # Arguments
///
/// * `cores` - TT cores to truncate (modified in-place)
/// * `max_ranks` - Maximum rank for bond k (between core k and core k+1);
///   length must equal `cores.len() - 1`
///
/// # Errors
///
/// Returns an error if the core list is empty, `max_ranks.len() ≠ cores.len()-1`,
/// or any internal linear algebra call fails.
///
/// # Complexity
///
/// O(∑ᵢ rᵢ² nᵢ) QR sweep + O(∑ᵢ rᵢ³) SVD sweep.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array3;
/// use tenrso_kernels::tt_round::tt_truncate;
///
/// let core1 = Array3::<f64>::from_elem((1, 10, 8), 0.1);
/// let core2 = Array3::<f64>::from_elem((8, 10, 8), 0.1);
/// let core3 = Array3::<f64>::from_elem((8, 10, 1), 0.1);
/// let mut cores = vec![core1, core2, core3];
///
/// // Truncate bond 0→1 to rank 5, bond 1→2 to rank 4
/// tt_truncate(&mut cores, &[5, 4]).unwrap();
///
/// // Verify boundary ranks preserved
/// assert_eq!(cores[0].shape()[0], 1);
/// assert_eq!(cores[2].shape()[2], 1);
/// ```
pub fn tt_truncate<T>(cores: &mut [Array3<T>], max_ranks: &[usize]) -> KernelResult<()>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    if cores.is_empty() {
        return Err(KernelError::empty_input("tt_truncate", "cores"));
    }

    if max_ranks.len() != cores.len() - 1 {
        return Err(KernelError::dimension_mismatch(
            "tt_truncate",
            vec![cores.len() - 1],
            vec![max_ranks.len()],
            "max_ranks length must be cores.len() - 1",
        ));
    }

    let d = cores.len();

    // Single-core TT: no bonds to truncate.
    if d == 1 {
        return Ok(());
    }

    // ── Phase 1: left-to-right QR orthogonalization ──────────────────────────
    tt_left_orthogonalize(cores)?;

    // ── Phase 2: right-to-left SVD, applying per-bond max_ranks ──────────────
    // Bond k (0-indexed) sits between core k and core k+1; its max rank is max_ranks[k].
    // We process from k = d-1 down to k = 1 (using bond k-1 = max_ranks[k-1]).
    for k in (1..d).rev() {
        let bond_idx = k - 1; // max_ranks[bond_idx] caps this bond
        let (r_left, n_k, r_right) = {
            let s = cores[k].shape();
            (s[0], s[1], s[2])
        };

        let core_mat: Array2<T> = cores[k]
            .view()
            .to_shape((r_left, n_k * r_right))
            .map_err(|e| {
                KernelError::operation_error("tt_truncate", format!("right-unfold k={}: {}", k, e))
            })?
            .to_owned();

        let (u, s_vals, vt) = scirs2_linalg::svd(&core_mat.view(), false, None).map_err(|e| {
            KernelError::operation_error("tt_truncate", format!("SVD k={}: {}", k, e))
        })?;

        let new_rank = s_vals.len().min(max_ranks[bond_idx]).max(1);

        let vt_trunc: Array2<T> = vt.slice(s![..new_rank, ..]).to_owned();
        cores[k] = vt_trunc
            .to_shape((new_rank, n_k, r_right))
            .map_err(|e| {
                KernelError::operation_error("tt_truncate", format!("reshape Vt k={}: {}", k, e))
            })?
            .to_owned();

        let mut transfer: Array2<T> = u.slice(s![.., ..new_rank]).to_owned();
        for (col_idx, &sv) in s_vals.iter().take(new_rank).enumerate() {
            transfer.column_mut(col_idx).mapv_inplace(|x| x * sv);
        }

        let prev_shape = cores[k - 1].shape().to_vec();
        let (r_prev_left, n_prev) = (prev_shape[0], prev_shape[1]);

        let prev_mat: Array2<T> = cores[k - 1]
            .view()
            .to_shape((r_prev_left * n_prev, r_left))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_truncate",
                    format!("reshape G_{{k-1}} k={}: {}", k, e),
                )
            })?
            .to_owned();

        cores[k - 1] = prev_mat
            .dot(&transfer)
            .to_shape((r_prev_left, n_prev, new_rank))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_truncate",
                    format!("reshape new G_{{k-1}} k={}: {}", k, e),
                )
            })?
            .to_owned();
    }

    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tt_ops::tt_norm;
    use scirs2_core::ndarray_ext::{array, Array3};

    fn make_cores_f64(shapes: &[(usize, usize, usize)]) -> Vec<Array3<f64>> {
        shapes
            .iter()
            .enumerate()
            .map(|(i, &(rl, n, rr))| {
                Array3::from_shape_fn((rl, n, rr), |(r, j, s)| {
                    (i + 1) as f64 * (r + 1) as f64 + (j + 1) as f64 * 0.1 + (s + 1) as f64 * 0.01
                })
            })
            .collect()
    }

    fn tt_norm_and_reconstruct(cores: &[Array3<f64>]) -> f64 {
        let views: Vec<_> = cores.iter().map(|c| c.view()).collect();
        tt_norm(&views).unwrap_or(f64::NAN)
    }

    // ── tt_round: error-handling ─────────────────────────────────────────────

    #[test]
    fn test_tt_round_empty_cores() {
        let mut cores: Vec<Array3<f64>> = vec![];
        assert!(tt_round(&mut cores, Some(2), 1e-6).is_err());
    }

    #[test]
    fn test_tt_round_negative_epsilon() {
        let core1 = Array3::<f64>::ones((1, 3, 2));
        let core2 = Array3::<f64>::ones((2, 3, 1));
        let mut cores = vec![core1, core2];
        assert!(tt_round(&mut cores, Some(2), -0.1).is_err());
    }

    // ── tt_truncate: error-handling ──────────────────────────────────────────

    #[test]
    fn test_tt_truncate_wrong_ranks_length() {
        let core1 = Array3::<f64>::ones((1, 3, 2));
        let core2 = Array3::<f64>::ones((2, 3, 2));
        let core3 = Array3::<f64>::ones((2, 3, 1));
        let mut cores = vec![core1, core2, core3];
        assert!(tt_truncate(&mut cores, &[2, 2, 2]).is_err());
    }

    // ── single-core edge case ────────────────────────────────────────────────

    #[test]
    fn test_svd_round_single_core_unchanged() {
        let core1 = Array3::<f64>::ones((1, 10, 1));
        let original_norm = tt_norm_and_reconstruct(&[core1.clone()]);

        let mut cores = vec![core1];
        tt_round(&mut cores, Some(5), 1e-6).unwrap();

        assert_eq!(cores[0].shape(), &[1, 10, 1]);
        let rounded_norm = tt_norm_and_reconstruct(&cores);
        assert!((original_norm - rounded_norm).abs() < 1e-10);
    }

    // ── norm preservation with zero epsilon ──────────────────────────────────

    #[test]
    fn test_tt_round_zero_epsilon_preserves_norm() {
        let mut cores = make_cores_f64(&[(1, 5, 4), (4, 5, 4), (4, 5, 1)]);
        let original_norm = tt_norm_and_reconstruct(&cores);

        tt_round(&mut cores, None, 0.0).unwrap();

        let rounded_norm = tt_norm_and_reconstruct(&cores);
        let rel = (original_norm - rounded_norm).abs() / original_norm.max(1e-14);
        assert!(rel < 1e-8, "rel error {:.2e} with zero epsilon", rel);
    }

    // ── boundary ranks are preserved ─────────────────────────────────────────

    #[test]
    fn test_svd_round_preserves_boundary_ranks() {
        let mut cores = make_cores_f64(&[(1, 5, 6), (6, 5, 6), (6, 5, 1)]);
        tt_round(&mut cores, Some(3), 1e-6).unwrap();
        assert_eq!(cores[0].shape()[0], 1, "left boundary rank must stay 1");
        assert_eq!(cores[2].shape()[2], 1, "right boundary rank must stay 1");
    }

    // ── max_rank constraint is honoured ──────────────────────────────────────

    #[test]
    fn test_svd_round_max_rank_constraint() {
        let mut cores = make_cores_f64(&[(1, 4, 6), (6, 4, 6), (6, 4, 1)]);
        tt_round(&mut cores, Some(3), 1e-12).unwrap();

        for c in &cores {
            let s = c.shape();
            assert!(s[0] <= 3, "r_left {} > max_rank 3", s[0]);
            assert!(s[2] <= 3, "r_right {} > max_rank 3", s[2]);
        }
    }

    // ── tt_truncate per-bond rank caps ───────────────────────────────────────

    #[test]
    fn test_svd_truncate_per_bond_ranks() {
        let mut cores = make_cores_f64(&[(1, 4, 7), (7, 4, 8), (8, 4, 1)]);
        tt_truncate(&mut cores, &[3, 4]).unwrap();

        assert!(cores[0].shape()[2] <= 3, "bond 0 right rank");
        assert!(cores[1].shape()[0] <= 3, "bond 0 matches left of core 1");
        assert!(cores[1].shape()[2] <= 4, "bond 1 right rank");
        assert!(cores[2].shape()[0] <= 4, "bond 1 matches left of core 2");
    }

    // ── combined epsilon + max_rank ──────────────────────────────────────────

    #[test]
    fn test_svd_round_combined_constraints() {
        let mut cores = make_cores_f64(&[(1, 5, 8), (8, 5, 8), (8, 5, 1)]);
        let original_norm = tt_norm_and_reconstruct(&cores);

        tt_round(&mut cores, Some(4), 0.05).unwrap();

        for c in &cores {
            let s = c.shape();
            assert!(s[0] <= 4, "r_left {} > 4", s[0]);
            assert!(s[2] <= 4, "r_right {} > 4", s[2]);
        }

        let rounded_norm = tt_norm_and_reconstruct(&cores);
        let rel = (original_norm - rounded_norm).abs() / original_norm.max(1e-14);
        assert!(rel < 0.15, "rel error {:.2e} too large", rel);
    }

    // ── low-rank structure triggers actual rank reduction ────────────────────

    #[test]
    fn test_svd_round_rank_reduction_low_rank_structure() {
        // Build TT whose second bond has effective rank 2 out of 5.
        // Core1: (1, 4, 5), only first 2 slices non-zero.
        let mut core1 = Array3::<f64>::zeros((1, 4, 5));
        for i in 0..4 {
            core1[[0, i, 0]] = (i + 1) as f64;
            core1[[0, i, 1]] = (i + 2) as f64 * 0.5;
        }

        let mut core2 = Array3::<f64>::zeros((5, 4, 5));
        for i in 0..4 {
            for r in 0..2 {
                core2[[r, i, r]] = (i + r + 1) as f64 * 0.3;
            }
        }

        let mut core3 = Array3::<f64>::zeros((5, 4, 1));
        for i in 0..4 {
            for r in 0..2 {
                core3[[r, i, 0]] = (i + r + 1) as f64 * 0.2;
            }
        }

        let cores_ref: Vec<_> = [core1.view(), core2.view(), core3.view()].to_vec();
        let original_norm = tt_norm(&cores_ref).unwrap();
        assert!(original_norm > 0.0, "non-trivial TT required");

        let mut cores = vec![core1, core2, core3];
        tt_round(&mut cores, Some(2), 1e-10).unwrap();

        // All internal bond ranks should be ≤ 2
        assert!(
            cores[0].shape()[2] <= 2,
            "bond 0 rank {}",
            cores[0].shape()[2]
        );
        assert!(
            cores[1].shape()[0] <= 2,
            "bond 0 (core1 left) rank {}",
            cores[1].shape()[0]
        );
        assert!(
            cores[1].shape()[2] <= 2,
            "bond 1 rank {}",
            cores[1].shape()[2]
        );
        assert!(
            cores[2].shape()[0] <= 2,
            "bond 1 (core2 left) rank {}",
            cores[2].shape()[0]
        );
    }

    // ── very tight epsilon keeps most singular values ─────────────────────────

    #[test]
    fn test_svd_round_very_small_epsilon() {
        let mut cores = make_cores_f64(&[(1, 4, 5), (5, 4, 5), (5, 4, 1)]);
        let original_norm = tt_norm_and_reconstruct(&cores);

        tt_round(&mut cores, None, 1e-12).unwrap();

        let rounded_norm = tt_norm_and_reconstruct(&cores);
        let rel = (original_norm - rounded_norm).abs() / original_norm.max(1e-14);
        assert!(rel < 1e-8, "rel error {:.2e} with epsilon=1e-12", rel);
    }

    // ── moderate epsilon reduces norm slightly ────────────────────────────────

    #[test]
    fn test_tt_round_with_epsilon() {
        let mut cores = make_cores_f64(&[(1, 5, 4), (4, 5, 4), (4, 5, 1)]);
        let original_norm = tt_norm_and_reconstruct(&cores);

        tt_round(&mut cores, None, 0.1).unwrap();

        let rounded_norm = tt_norm_and_reconstruct(&cores);
        let rel = (original_norm - rounded_norm).abs() / original_norm.max(1e-14);
        assert!(rel < 0.5, "rel error {:.2e} unexpectedly large", rel);
    }

    // ── reconstruction fidelity check ────────────────────────────────────────

    #[test]
    fn test_tt_round_reconstruction_fidelity() {
        // Build a compressible TT from an outer product: each core is rank-1.
        let v1 = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let v2 = array![0.5_f64, 1.5, 2.5, 3.5, 4.5];
        let v3 = array![0.1_f64, 0.2, 0.3, 0.4, 0.5];

        let mut core1 = Array3::<f64>::zeros((1, 5, 1));
        let mut core2 = Array3::<f64>::zeros((1, 5, 1));
        let mut core3 = Array3::<f64>::zeros((1, 5, 1));
        for i in 0..5 {
            core1[[0, i, 0]] = v1[i];
            core2[[0, i, 0]] = v2[i];
            core3[[0, i, 0]] = v3[i];
        }

        let mut cores = vec![core1, core2, core3];
        let original_norm = tt_norm_and_reconstruct(&cores);

        tt_round(&mut cores, None, 1e-10).unwrap();

        let rounded_norm = tt_norm_and_reconstruct(&cores);
        let rel = (original_norm - rounded_norm).abs() / original_norm.max(1e-14);
        assert!(
            rel < 1e-8,
            "rank-1 TT should survive rounding: rel={:.2e}",
            rel
        );
    }

    // ── determine_rank_from_delta unit tests ─────────────────────────────────

    #[test]
    fn test_determine_rank_from_delta_zero_threshold() {
        let s = Array1::from_vec(vec![3.0_f64, 2.0, 1.0, 0.5]);
        // delta_sq = 0 => keep all
        assert_eq!(determine_rank_from_delta(&s, 0.0, None), 4);
    }

    #[test]
    fn test_determine_rank_from_delta_drops_trailing_small() {
        // 0.1^2 = 0.01, 0.2^2 = 0.04; budget = 0.06 => can drop both
        let s = Array1::from_vec(vec![3.0_f64, 2.0, 1.0, 0.2, 0.1]);
        let rank = determine_rank_from_delta(&s, 0.05_f64, None);
        assert!(rank <= 4, "expected rank ≤ 4, got {}", rank);
        assert!(rank >= 1);
    }

    #[test]
    fn test_determine_rank_from_delta_max_rank_cap() {
        let s = Array1::from_vec(vec![1.0_f64, 1.0, 1.0, 1.0, 1.0]);
        // max_rank=2 forces rank ≤ 2 regardless of threshold
        assert_eq!(determine_rank_from_delta(&s, 0.0, Some(2)), 2);
    }

    // ── determine_truncation_rank unit tests (existing helper, kept) ──────────

    #[test]
    fn test_determine_truncation_rank_all_equal() {
        let s = Array1::from_vec(vec![1.0_f64, 1.0, 1.0, 1.0]);
        let rank = determine_truncation_rank(&s, 0.01, None);
        assert!(rank >= 3);
    }

    #[test]
    fn test_determine_truncation_rank_decaying() {
        let s = Array1::from_vec(vec![1.0_f64, 0.5, 0.25, 0.125, 0.0625, 0.03125]);
        let rank = determine_truncation_rank(&s, 0.01, None);
        assert!(rank < 6);
        assert!(rank >= 3);
    }

    #[test]
    fn test_determine_truncation_rank_with_max_rank() {
        let s = Array1::from_vec(vec![1.0_f64, 0.9, 0.8, 0.7, 0.6, 0.5]);
        let rank = determine_truncation_rank(&s, 1e-12, Some(3));
        assert_eq!(rank, 3);
    }

    #[test]
    fn test_determine_truncation_rank_zero_values() {
        let s = Array1::from_vec(vec![1.0_f64, 0.5, 0.25, 0.0, 0.0]);
        let rank = determine_truncation_rank(&s, 1e-4, None);
        assert!(rank <= 3);
        assert!(rank >= 1);
    }

    // ── tt_truncate boundary ranks ────────────────────────────────────────────

    #[test]
    fn test_tt_truncate_boundary_ranks_preserved() {
        let mut cores = make_cores_f64(&[(1, 4, 5), (5, 4, 6), (6, 4, 1)]);
        tt_truncate(&mut cores, &[3, 3]).unwrap();
        assert_eq!(cores[0].shape()[0], 1);
        assert_eq!(cores[2].shape()[2], 1);
    }

    // ── 4-core TT round ───────────────────────────────────────────────────────

    #[test]
    fn test_tt_round_four_cores() {
        let mut cores = make_cores_f64(&[(1, 4, 6), (6, 4, 6), (6, 4, 6), (6, 4, 1)]);
        let original_norm = tt_norm_and_reconstruct(&cores);

        tt_round(&mut cores, Some(3), 0.01).unwrap();

        // All ranks ≤ 3 after rounding
        for c in &cores {
            let s = c.shape();
            assert!(s[0] <= 3, "r_left={} > 3", s[0]);
            assert!(s[2] <= 3, "r_right={} > 3", s[2]);
        }

        let rounded_norm = tt_norm_and_reconstruct(&cores);
        let rel = (original_norm - rounded_norm).abs() / original_norm.max(1e-14);
        assert!(rel < 0.2, "4-core rel error {:.2e}", rel);
    }

    // ── svd_truncate_bond helper (used indirectly via tt_truncate) ────────────

    #[test]
    fn test_svd_truncate_bond_shape_consistency() {
        let mut cores = make_cores_f64(&[(1, 3, 6), (6, 3, 6), (6, 3, 1)]);
        tt_truncate(&mut cores, &[4, 4]).unwrap();

        // Bond consistency: right rank of core k == left rank of core k+1
        assert_eq!(
            cores[0].shape()[2],
            cores[1].shape()[0],
            "bond 0 rank mismatch"
        );
        assert_eq!(
            cores[1].shape()[2],
            cores[2].shape()[0],
            "bond 1 rank mismatch"
        );
    }
}
