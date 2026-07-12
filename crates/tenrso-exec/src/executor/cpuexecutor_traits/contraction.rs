//! Contraction (einsum) helper for the `TenrsoExecutor` implementation.

use super::super::types::CpuExecutor;
use crate::hints::{ExecHints, MaskPack, SubsetSpec};
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{DenseND, TensorHandle};
use tenrso_planner::EinsumSpec;
use tenrso_sparse::mask::Mask;
use tenrso_sparse::masked_einsum::masked_einsum as sparse_masked_einsum;

pub(super) fn einsum<T>(
    executor: &mut CpuExecutor,
    spec: &str,
    inputs: &[TensorHandle<T>],
    hints: &ExecHints,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    // `mask` and `subset` are two spellings of one selection — a dense bitmap and
    // a sparse index list.  Both, with `prefer_sparse`, route to the masked
    // engine; both at once is an ambiguous request, so it is an error rather than
    // a silent precedence rule.
    if let Some(selection) = output_selection(hints)? {
        return einsum_masked_path(spec, inputs, &selection);
    }

    // Dense path (existing).
    let parsed_spec = EinsumSpec::parse(spec)?;
    if parsed_spec.num_inputs() != inputs.len() {
        return Err(anyhow!(
            "Spec expects {} inputs, got {}",
            parsed_spec.num_inputs(),
            inputs.len()
        ));
    }
    let dense_inputs: Vec<&DenseND<T>> = inputs
        .iter()
        .map(|h| {
            h.as_dense()
                .ok_or_else(|| anyhow!("Only dense tensors supported for now"))
        })
        .collect::<Result<Vec<_>>>()?;
    let dense_inputs_owned: Vec<DenseND<T>> = dense_inputs.iter().map(|&t| t.clone()).collect();
    let result = executor.execute_einsum_with_planner(&parsed_spec, &dense_inputs_owned, hints)?;
    Ok(TensorHandle::from_dense_auto(result))
}

// ---------------------------------------------------------------------------
// Output selection: mask (dense bitmap) or subset (sparse index list)
// ---------------------------------------------------------------------------

/// Resolve the hints' output selection, if any, into a single [`Mask`].
///
/// Returns `Ok(None)` when the dense engine should run: either `prefer_sparse` is
/// unset (the flag is what selects the masked engine), or neither selection field
/// carries data.
///
/// # Errors
///
/// If `mask` and `subset` both carry data.  They describe the same thing in two
/// representations, so supplying both is a contradiction the caller has to
/// resolve — quietly preferring one would hide a bug in their code.
fn output_selection(hints: &ExecHints) -> Result<Option<Mask>> {
    if !hints.prefer_sparse {
        return Ok(None);
    }

    let mask_pack = hints.mask.as_ref().filter(|m| m.mask.is_some());
    let subset = hints.subset.as_ref().filter(|s| s.indices.is_some());

    match (mask_pack, subset) {
        (Some(_), Some(_)) => Err(anyhow!(
            "ExecHints: `mask` and `subset` both select the output — supply exactly one \
             (`subset` is the sparse spelling of `mask`)"
        )),
        (Some(mask_pack), None) => convert_mask_pack(mask_pack).map(Some),
        (None, Some(subset)) => convert_subset_spec(subset).map(Some),
        (None, None) => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Masked einsum path
// ---------------------------------------------------------------------------

/// Route einsum through the sparse masked engine.
///
/// Extracts dense data from every input, calls `tenrso_sparse::masked_einsum`
/// with the resolved selection, then densifies the COO result before wrapping it
/// in a `TensorHandle`.  Only the selected output cells are computed; every other
/// cell of the returned tensor is zero.
fn einsum_masked_path<T>(
    spec: &str,
    inputs: &[TensorHandle<T>],
    mask: &Mask,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    // 1. Extract dense inputs — fail gracefully if any handle is not dense.
    let dense_inputs: Vec<&DenseND<T>> = inputs
        .iter()
        .map(|h| {
            h.as_dense().ok_or_else(|| {
                anyhow!("Masked einsum path requires all inputs to be dense tensors")
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // 2. Delegate to the sparse masked einsum engine.
    let coo = sparse_masked_einsum(spec, &dense_inputs, mask)
        .map_err(|e| anyhow!("masked_einsum failed: {}", e))?;

    // 3. Densify the COO result.
    let dense = coo
        .to_dense()
        .map_err(|e| anyhow!("CooTensor::to_dense failed: {}", e))?;

    // 4. Wrap in a TensorHandle.
    Ok(TensorHandle::from_dense_auto(dense))
}

/// Row-major strides of `shape`: `strides[d] = Π shape[d+1..]`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Expand a flat row-major offset into one index per axis.
fn unflatten(flat: usize, strides: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; strides.len()];
    let mut rem = flat;
    for (d, &stride) in strides.iter().enumerate() {
        // `stride` is a product of extents, all ≥ 1 for a validated shape.
        idx[d] = rem / stride;
        rem %= stride;
    }
    idx
}

/// Convert a flat row-major `MaskPack` into a multi-dimensional `Mask`.
fn convert_mask_pack(mp: &MaskPack) -> Result<Mask> {
    let mask_bits = mp
        .mask
        .as_ref()
        .ok_or_else(|| anyhow!("MaskPack has no mask data"))?;
    let shape = &mp.shape;

    if shape.is_empty() {
        anyhow::bail!("MaskPack shape cannot be empty");
    }
    let size: usize = shape.iter().product();
    if mask_bits.len() != size {
        return Err(anyhow!(
            "MaskPack: mask has {} entries but shape {:?} has {} cells",
            mask_bits.len(),
            shape,
            size
        ));
    }

    let strides = row_major_strides(shape);
    let indices: Vec<Vec<usize>> = mask_bits
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(flat, _)| unflatten(flat, &strides))
        .collect();

    Mask::from_indices(indices, shape.to_vec())
        .map_err(|e| anyhow!("MaskPack → Mask conversion failed: {}", e))
}

/// Convert a flat row-major `SubsetSpec` into a multi-dimensional `Mask`.
///
/// The sparse counterpart of [`convert_mask_pack`]: it walks the `s` selected
/// positions instead of all `N` cells of the output, which is the whole point of
/// the subset representation.  An index outside the shape is rejected — wrapping
/// it, or dropping it, would compute a silently different output than the caller
/// asked for.
fn convert_subset_spec(spec: &SubsetSpec) -> Result<Mask> {
    let flat_indices = spec
        .indices
        .as_ref()
        .ok_or_else(|| anyhow!("SubsetSpec has no indices"))?;
    let shape = &spec.shape;

    if shape.is_empty() {
        anyhow::bail!("SubsetSpec shape cannot be empty");
    }
    let size: usize = shape.iter().product();

    let strides = row_major_strides(shape);
    let indices: Vec<Vec<usize>> = flat_indices
        .iter()
        .map(|&flat| {
            if flat >= size {
                return Err(anyhow!(
                    "SubsetSpec: flat index {} is out of range for shape {:?} ({} cells)",
                    flat,
                    shape,
                    size
                ));
            }
            Ok(unflatten(flat, &strides))
        })
        .collect::<Result<Vec<_>>>()?;

    Mask::from_indices(indices, shape.to_vec())
        .map_err(|e| anyhow!("SubsetSpec → Mask conversion failed: {}", e))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::functions::TenrsoExecutor;
    use crate::executor::types::CpuExecutor;
    use crate::hints::{ExecHints, MaskPack};

    // ------------------------------------------------------------------
    // Test helpers
    // ------------------------------------------------------------------

    fn make_matrix(rows: usize, cols: usize, start: f64) -> TensorHandle<f64> {
        let data: Vec<f64> = (0..rows * cols).map(|i| start + i as f64).collect();
        TensorHandle::from_dense_auto(DenseND::from_vec(data, &[rows, cols]).unwrap())
    }

    fn make_vector(len: usize, start: f64) -> TensorHandle<f64> {
        let data: Vec<f64> = (0..len).map(|i| start + i as f64).collect();
        TensorHandle::from_dense_auto(DenseND::from_vec(data, &[len]).unwrap())
    }

    fn make_mask_diagonal(n: usize) -> MaskPack {
        let mut bits = vec![false; n * n];
        for i in 0..n {
            bits[i * n + i] = true;
        }
        MaskPack::new(bits, vec![n, n])
    }

    fn hints_sparse_mask(mask_pack: MaskPack) -> ExecHints {
        ExecHints {
            prefer_sparse: true,
            mask: Some(mask_pack),
            ..Default::default()
        }
    }

    // ------------------------------------------------------------------
    // Test 1: diagonal mask on 3×3 matmul output
    // ------------------------------------------------------------------
    #[test]
    fn masked_matmul_diagonal() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(3, 3, 1.0);
        let b = make_matrix(3, 3, 1.0);

        let hints = hints_sparse_mask(make_mask_diagonal(3));
        let result = ex
            .einsum("ij,jk->ik", &[a.clone(), b.clone()], &hints)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3, 3]);

        // Dense reference
        let dense_hints = ExecHints::default();
        let mut ex2 = CpuExecutor::new();
        let ref_result = ex2.einsum("ij,jk->ik", &[a, b], &dense_hints).unwrap();
        let ref_dense = ref_result.as_dense().unwrap();

        let rv = result_dense.view();
        let dv = ref_dense.view();

        // Diagonal elements must match
        for i in 0..3 {
            let diff = (rv[[i, i]] - dv[[i, i]]).abs();
            assert!(
                diff < 1e-10,
                "diagonal[{}] mismatch: {} vs {}",
                i,
                rv[[i, i]],
                dv[[i, i]]
            );
        }
        // Off-diagonal must be 0 (mask excluded them)
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(
                        rv[[i, j]].abs() < 1e-14,
                        "off-diag [{},{}] should be 0, got {}",
                        i,
                        j,
                        rv[[i, j]]
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 2: top-left 2×2 mask on 4×4 output
    // ------------------------------------------------------------------
    #[test]
    fn masked_matmul_half() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(4, 4, 1.0);
        let b = make_matrix(4, 4, 1.0);

        // Mask selects top-left 2×2 of 4×4 output
        let mut bits = vec![false; 16];
        for r in 0..2 {
            for c in 0..2 {
                bits[r * 4 + c] = true;
            }
        }
        let mask_pack = MaskPack::new(bits, vec![4, 4]);
        let hints = hints_sparse_mask(mask_pack);

        let result = ex
            .einsum("ij,jk->ik", &[a.clone(), b.clone()], &hints)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[4, 4]);

        let mut ex2 = CpuExecutor::new();
        let ref_result = ex2
            .einsum("ij,jk->ik", &[a, b], &ExecHints::default())
            .unwrap();
        let ref_dense = ref_result.as_dense().unwrap();

        let rv = result_dense.view();
        let dv = ref_dense.view();

        // Top-left 2×2 must match dense
        for r in 0..2 {
            for c in 0..2 {
                let diff = (rv[[r, c]] - dv[[r, c]]).abs();
                assert!(
                    diff < 1e-10,
                    "[{},{}] masked={} dense={}",
                    r,
                    c,
                    rv[[r, c]],
                    dv[[r, c]]
                );
            }
        }
        // Remaining positions must be 0
        for r in 0..4 {
            for c in 0..4 {
                if r >= 2 || c >= 2 {
                    assert!(
                        rv[[r, c]].abs() < 1e-14,
                        "unmasked [{},{}] should be 0, got {}",
                        r,
                        c,
                        rv[[r, c]]
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 3: masked element-wise (ij,ij->ij) on 3×3
    // ------------------------------------------------------------------
    #[test]
    fn masked_elementwise() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(3, 3, 1.0);
        let b = make_matrix(3, 3, 10.0);

        // Upper triangle mask
        let mut bits = vec![false; 9];
        for r in 0..3 {
            for c in r..3 {
                bits[r * 3 + c] = true;
            }
        }
        let mask_pack = MaskPack::new(bits.clone(), vec![3, 3]);
        let hints = hints_sparse_mask(mask_pack);

        let result = ex
            .einsum("ij,ij->ij", &[a.clone(), b.clone()], &hints)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3, 3]);

        let av = a.as_dense().unwrap().view();
        let bv = b.as_dense().unwrap().view();
        let rv = result_dense.view();

        for r in 0..3 {
            for c in 0..3 {
                let masked = bits[r * 3 + c];
                if masked {
                    let expected = av[[r, c]] * bv[[r, c]];
                    let diff = (rv[[r, c]] - expected).abs();
                    assert!(
                        diff < 1e-10,
                        "elt [{},{}]: got {} expected {}",
                        r,
                        c,
                        rv[[r, c]],
                        expected
                    );
                } else {
                    assert!(
                        rv[[r, c]].abs() < 1e-14,
                        "unmasked [{},{}] should be 0",
                        r,
                        c
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 4: masked outer product (i,j->ij)
    // ------------------------------------------------------------------
    #[test]
    fn masked_outer() {
        let mut ex = CpuExecutor::new();
        let a = make_vector(4, 1.0);
        let b = make_vector(3, 10.0);

        // Checkerboard mask on 4×3
        let mut bits = vec![false; 12];
        for r in 0..4 {
            for c in 0..3 {
                if (r + c) % 2 == 0 {
                    bits[r * 3 + c] = true;
                }
            }
        }
        let mask_pack = MaskPack::new(bits.clone(), vec![4, 3]);
        let hints = hints_sparse_mask(mask_pack);

        let result = ex
            .einsum("i,j->ij", &[a.clone(), b.clone()], &hints)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[4, 3]);

        let av = a.as_dense().unwrap().view();
        let bv = b.as_dense().unwrap().view();
        let rv = result_dense.view();

        for r in 0..4 {
            for c in 0..3 {
                let masked = bits[r * 3 + c];
                if masked {
                    let expected = av[[r]] * bv[[c]];
                    let diff = (rv[[r, c]] - expected).abs();
                    assert!(
                        diff < 1e-10,
                        "outer [{},{}]: got {} expected {}",
                        r,
                        c,
                        rv[[r, c]],
                        expected
                    );
                } else {
                    assert!(
                        rv[[r, c]].abs() < 1e-14,
                        "unmasked [{},{}] should be 0",
                        r,
                        c
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 5: full mask equals dense
    // ------------------------------------------------------------------
    #[test]
    fn full_mask_equals_dense() {
        let n = 3;
        let a = make_matrix(n, n, 1.0);
        let b = make_matrix(n, n, 1.0);

        // All-true mask
        let bits = vec![true; n * n];
        let mask_pack = MaskPack::new(bits, vec![n, n]);

        let mut ex_masked = CpuExecutor::new();
        let hints = hints_sparse_mask(mask_pack);
        let masked_result = ex_masked
            .einsum("ij,jk->ik", &[a.clone(), b.clone()], &hints)
            .unwrap();

        let mut ex_dense = CpuExecutor::new();
        let dense_result = ex_dense
            .einsum("ij,jk->ik", &[a, b], &ExecHints::default())
            .unwrap();

        let mr = masked_result.as_dense().unwrap().view();
        let dr = dense_result.as_dense().unwrap().view();

        for i in 0..n {
            for j in 0..n {
                let diff = (mr[[i, j]] - dr[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "full-mask [{},{}]: masked={} dense={}",
                    i,
                    j,
                    mr[[i, j]],
                    dr[[i, j]]
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 6: empty mask yields all-zero result
    // ------------------------------------------------------------------
    #[test]
    fn empty_mask() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(3, 3, 1.0);
        let b = make_matrix(3, 3, 1.0);

        let bits = vec![false; 9];
        let mask_pack = MaskPack::new(bits, vec![3, 3]);
        let hints = hints_sparse_mask(mask_pack);

        let result = ex.einsum("ij,jk->ik", &[a, b], &hints).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3, 3]);

        let rv = result_dense.view();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    rv[[i, j]].abs() < 1e-14,
                    "empty mask [{},{}] should be 0, got {}",
                    i,
                    j,
                    rv[[i, j]]
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 7: prefer_sparse=true but no mask → dense path, no error
    // ------------------------------------------------------------------
    #[test]
    fn no_mask_unaffected() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(2, 3, 1.0);
        let b = make_matrix(3, 2, 1.0);

        let hints = ExecHints {
            prefer_sparse: true,
            mask: None,
            ..Default::default()
        };

        let result = ex.einsum("ij,jk->ik", &[a, b], &hints).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        // Just verify it produced a non-zero result (dense path ran)
        let rv = result_dense.view();
        assert!(
            rv[[0, 0]].abs() > 0.0,
            "expected non-zero result from dense path"
        );
    }

    // ------------------------------------------------------------------
    // Test 8: prefer_sparse=false with mask → dense path used
    // ------------------------------------------------------------------
    #[test]
    fn mask_with_sparse_false() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(2, 2, 1.0);
        let b = make_matrix(2, 2, 1.0);

        // Diagonal mask, but prefer_sparse = false — must use dense path.
        let hints = ExecHints {
            prefer_sparse: false,
            mask: Some(make_mask_diagonal(2)),
            ..Default::default()
        };

        let result = ex.einsum("ij,jk->ik", &[a, b], &hints).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);
        // Dense path produces full result — off-diagonal should be non-zero.
        let rv = result_dense.view();
        // [0,1] = row0 of A · col1 of B = 1*2 + 2*4 = 10
        let diff = (rv[[0, 1]] - 10.0).abs();
        assert!(
            diff < 1e-10,
            "dense path [0,1] should be 10, got {}",
            rv[[0, 1]]
        );
    }

    // ==================================================================
    // `subset`: the sparse spelling of `mask`
    // ==================================================================

    fn hints_sparse_subset(indices: Vec<usize>, shape: Vec<usize>) -> ExecHints {
        ExecHints::new()
            .with_sparse(true)
            .with_subset(indices, shape)
    }

    /// Dense reference for `"ij,jk->ik"` on the two test matrices.
    fn dense_reference(a: TensorHandle<f64>, b: TensorHandle<f64>) -> DenseND<f64> {
        let mut ex = CpuExecutor::new();
        ex.einsum("ij,jk->ik", &[a, b], &ExecHints::default())
            .expect("dense einsum")
            .as_dense()
            .expect("dense result")
            .clone()
    }

    // ------------------------------------------------------------------
    // Test 9: a subset selects exactly the cells it names — the selected
    // cells match the dense result, every other cell is left at zero.
    // ------------------------------------------------------------------
    #[test]
    fn subset_selects_named_cells_only() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(3, 3, 1.0);
        let b = make_matrix(3, 3, 1.0);

        // Flat row-major indices into the 3×3 output: [0,0], [1,2], [2,1].
        let selected = vec![0, 5, 7];
        let hints = hints_sparse_subset(selected.clone(), vec![3, 3]);

        let result = ex
            .einsum("ij,jk->ik", &[a.clone(), b.clone()], &hints)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3, 3]);

        let reference = dense_reference(a, b);
        let rv = result_dense.view();
        let dv = reference.view();

        for i in 0..3 {
            for j in 0..3 {
                let flat = i * 3 + j;
                if selected.contains(&flat) {
                    assert!(
                        (rv[[i, j]] - dv[[i, j]]).abs() < 1e-12,
                        "selected cell [{i},{j}]: subset={} dense={}",
                        rv[[i, j]],
                        dv[[i, j]]
                    );
                } else {
                    assert_eq!(
                        rv[[i, j]],
                        0.0,
                        "unselected cell [{i},{j}] must not be computed, got {}",
                        rv[[i, j]]
                    );
                }
            }
        }

        // The dense result is non-zero at the unselected cells — otherwise this
        // test would pass against an engine that computed nothing at all.
        assert!(
            dv[[0, 1]].abs() > 0.0,
            "the dense reference must be non-zero where the subset omits cells"
        );
    }

    // ------------------------------------------------------------------
    // Test 10: subset and mask are two spellings of one selection — the
    // same selection through either field yields the same tensor.
    // ------------------------------------------------------------------
    #[test]
    fn subset_matches_equivalent_mask() {
        let a = make_matrix(4, 4, 1.0);
        let b = make_matrix(4, 4, 2.0);

        // Diagonal, expressed both ways.
        let flat: Vec<usize> = (0..4).map(|i| i * 4 + i).collect();

        let mut ex_subset = CpuExecutor::new();
        let via_subset = ex_subset
            .einsum(
                "ij,jk->ik",
                &[a.clone(), b.clone()],
                &hints_sparse_subset(flat, vec![4, 4]),
            )
            .unwrap();

        let mut ex_mask = CpuExecutor::new();
        let via_mask = ex_mask
            .einsum(
                "ij,jk->ik",
                &[a, b],
                &hints_sparse_mask(make_mask_diagonal(4)),
            )
            .unwrap();

        let sv = via_subset.as_dense().unwrap().view();
        let mv = via_mask.as_dense().unwrap().view();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    sv[[i, j]].to_bits(),
                    mv[[i, j]].to_bits(),
                    "subset and mask disagree at [{i},{j}]: {} vs {}",
                    sv[[i, j]],
                    mv[[i, j]]
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 11: subset is honoured only under `prefer_sparse`, exactly like
    // `mask` (see test 8) — and that is documented, not silent.
    // ------------------------------------------------------------------
    #[test]
    fn subset_with_sparse_false_uses_dense_path() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(2, 2, 1.0);
        let b = make_matrix(2, 2, 1.0);

        let hints = ExecHints::new()
            .with_sparse(false)
            .with_subset(vec![0], vec![2, 2]);

        let result = ex.einsum("ij,jk->ik", &[a, b], &hints).unwrap();
        let rv = result.as_dense().unwrap().view().to_owned();
        // Dense path computed every cell, including the ones the subset omits.
        assert!(
            (rv[[0, 1]] - 10.0).abs() < 1e-10,
            "dense path [0,1] should be 10, got {}",
            rv[[0, 1]]
        );
    }

    // ------------------------------------------------------------------
    // Test 12: an empty subset computes nothing.
    // ------------------------------------------------------------------
    #[test]
    fn empty_subset_yields_zeros() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(3, 3, 1.0);
        let b = make_matrix(3, 3, 1.0);

        let result = ex
            .einsum(
                "ij,jk->ik",
                &[a, b],
                &hints_sparse_subset(vec![], vec![3, 3]),
            )
            .unwrap();
        let rv = result.as_dense().unwrap().view().to_owned();
        assert_eq!(rv.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(rv[[i, j]], 0.0, "empty subset [{i},{j}] must be 0");
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 13: an out-of-range flat index is an error, not a wrapped cell.
    // ------------------------------------------------------------------
    #[test]
    fn subset_out_of_range_index_errors() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(3, 3, 1.0);
        let b = make_matrix(3, 3, 1.0);

        // The 3×3 output has 9 cells: index 9 does not exist.
        let err = ex
            .einsum(
                "ij,jk->ik",
                &[a, b],
                &hints_sparse_subset(vec![9], vec![3, 3]),
            )
            .expect_err("an out-of-range subset index must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("out of range"),
            "expected an out-of-range error, got: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // Test 14: mask and subset together is a contradiction, so it errors
    // rather than one silently winning.
    // ------------------------------------------------------------------
    #[test]
    fn mask_and_subset_together_error() {
        let mut ex = CpuExecutor::new();
        let a = make_matrix(2, 2, 1.0);
        let b = make_matrix(2, 2, 1.0);

        let hints = ExecHints {
            prefer_sparse: true,
            mask: Some(make_mask_diagonal(2)),
            subset: Some(crate::hints::SubsetSpec::new(vec![0], vec![2, 2])),
        };

        let err = ex
            .einsum("ij,jk->ik", &[a, b], &hints)
            .expect_err("mask + subset must be rejected");
        assert!(
            err.to_string().contains("both select the output"),
            "expected the mask/subset conflict error, got: {err}"
        );
    }
}
