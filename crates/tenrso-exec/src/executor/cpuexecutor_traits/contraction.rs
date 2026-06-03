//! Contraction (einsum) helper for the `TenrsoExecutor` implementation.

use super::super::types::CpuExecutor;
use crate::hints::{ExecHints, MaskPack};
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
    // When prefer_sparse AND mask is present with data → use masked einsum path.
    if hints.prefer_sparse {
        if let Some(mask_pack) = &hints.mask {
            if mask_pack.mask.is_some() {
                return einsum_masked_path(spec, inputs, mask_pack);
            }
        }
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
// Masked einsum path
// ---------------------------------------------------------------------------

/// Route einsum through the sparse masked engine.
///
/// Extracts dense data from every input, converts the flat `MaskPack` to a
/// `Mask`, calls `tenrso_sparse::masked_einsum`, then densifies the COO result
/// before wrapping it in a `TensorHandle`.
fn einsum_masked_path<T>(
    spec: &str,
    inputs: &[TensorHandle<T>],
    mask_pack: &MaskPack,
) -> Result<TensorHandle<T>>
where
    T: Clone
        + Num
        + std::ops::AddAssign
        + std::default::Default
        + Float
        + FromPrimitive
        + 'static,
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

    // 2. Convert MaskPack → Mask.
    let mask = convert_mask_pack(mask_pack)?;

    // 3. Delegate to the sparse masked einsum engine.
    let coo = sparse_masked_einsum(spec, &dense_inputs, &mask)
        .map_err(|e| anyhow!("masked_einsum failed: {}", e))?;

    // 4. Densify the COO result.
    let dense = coo
        .to_dense()
        .map_err(|e| anyhow!("CooTensor::to_dense failed: {}", e))?;

    // 5. Wrap in a TensorHandle.
    Ok(TensorHandle::from_dense_auto(dense))
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

    let ndim = shape.len();

    // Compute row-major strides: strides[d] = product of shape[d+1..].
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    // Convert flat boolean array to list of multi-dimensional indices.
    let indices: Vec<Vec<usize>> = mask_bits
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(flat, _)| {
            let mut idx = vec![0usize; ndim];
            let mut rem = flat;
            for d in 0..ndim {
                idx[d] = rem / strides[d];
                rem %= strides[d];
            }
            idx
        })
        .collect();

    Mask::from_indices(indices, shape.to_vec())
        .map_err(|e| anyhow!("MaskPack → Mask conversion failed: {}", e))
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
        let ref_result = ex2
            .einsum("ij,jk->ik", &[a, b], &dense_hints)
            .unwrap();
        let ref_dense = ref_result.as_dense().unwrap();

        let rv = result_dense.view();
        let dv = ref_dense.view();

        // Diagonal elements must match
        for i in 0..3 {
            let diff = (rv[[i, i]] - dv[[i, i]]).abs();
            assert!(diff < 1e-10, "diagonal[{}] mismatch: {} vs {}", i, rv[[i, i]], dv[[i, i]]);
        }
        // Off-diagonal must be 0 (mask excluded them)
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(rv[[i, j]].abs() < 1e-14, "off-diag [{},{}] should be 0, got {}", i, j, rv[[i, j]]);
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
                assert!(diff < 1e-10, "[{},{}] masked={} dense={}", r, c, rv[[r, c]], dv[[r, c]]);
            }
        }
        // Remaining positions must be 0
        for r in 0..4 {
            for c in 0..4 {
                if r >= 2 || c >= 2 {
                    assert!(
                        rv[[r, c]].abs() < 1e-14,
                        "unmasked [{},{}] should be 0, got {}",
                        r, c, rv[[r, c]]
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
                    assert!(diff < 1e-10, "elt [{},{}]: got {} expected {}", r, c, rv[[r, c]], expected);
                } else {
                    assert!(rv[[r, c]].abs() < 1e-14, "unmasked [{},{}] should be 0", r, c);
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
                    assert!(diff < 1e-10, "outer [{},{}]: got {} expected {}", r, c, rv[[r, c]], expected);
                } else {
                    assert!(rv[[r, c]].abs() < 1e-14, "unmasked [{},{}] should be 0", r, c);
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
                    i, j, mr[[i, j]], dr[[i, j]]
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

        let result = ex
            .einsum("ij,jk->ik", &[a, b], &hints)
            .unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3, 3]);

        let rv = result_dense.view();
        for i in 0..3 {
            for j in 0..3 {
                assert!(rv[[i, j]].abs() < 1e-14, "empty mask [{},{}] should be 0, got {}", i, j, rv[[i, j]]);
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
        assert!(rv[[0, 0]].abs() > 0.0, "expected non-zero result from dense path");
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
        assert!(diff < 1e-10, "dense path [0,1] should be 10, got {}", rv[[0, 1]]);
    }
}
