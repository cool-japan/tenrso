//! Decomposition reconstruction rules (CP, Tucker, TT).
//!
//! Each rule exposes the *reconstruction* operator of a decomposition as a
//! differentiable op: the inputs are the decomposition's parameters (factors,
//! core, cores) and the output is the reconstructed dense tensor.
//!
//! The gradients are the ones already implemented in [`crate::grad`]; this
//! module only supplies the matching forward pass and the tensor plumbing.
//!
//! # Layout conventions
//!
//! * **CP** — inputs are the factor matrices `U_1 … U_N`, each `(I_n, R)` with
//!   the same rank `R`. Reconstruction is
//!   `X[i_1,…,i_N] = Σ_r λ_r · Π_n U_n[i_n, r]` (with `λ = 1` unless the rule was
//!   built with [`CpReconstructionRule::with_weights`]).
//!   The forward pass is computed as `X_(0) = (U_0 ⊙λ) · (U_1 ⊙ … ⊙ U_N)ᵀ`,
//!   which is exactly the unfolding convention used by
//!   [`crate::grad::CpReconstructionGrad`] (row-major unfold ⇔ Khatri-Rao with
//!   the last mode varying fastest).
//! * **Tucker** — `inputs[0]` is the core, `inputs[1..]` are the factor matrices
//!   `U_n` of shape `(I_n, R_n)`. Reconstruction is `X = G ×₁ U_1 ×₂ … ×_N U_N`.
//! * **TT** — inputs are the TT cores, each of shape `(r_{k-1}, n_k, r_k)` with
//!   `r_0 = r_N = 1`.

use anyhow::{anyhow, bail, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};
use tenrso_core::DenseND;

use super::{as_matrix, from_matrix, AdScalar, Arity, OpParams, OpRule};
use crate::grad::{
    tt_reconstruct, CpReconstructionGrad, TtReconstructionGrad, TuckerReconstructionGrad,
};

/// CP (CANDECOMP/PARAFAC) reconstruction: factors → dense tensor.
#[derive(Debug, Clone)]
pub struct CpReconstructionRule<T>
where
    T: AdScalar,
{
    weights: Option<Array1<T>>,
    name: String,
}

impl<T> Default for CpReconstructionRule<T>
where
    T: AdScalar,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CpReconstructionRule<T>
where
    T: AdScalar,
{
    /// Unweighted CP reconstruction, registered as `"cp_reconstruct"`.
    pub fn new() -> Self {
        Self {
            weights: None,
            name: "cp_reconstruct".to_string(),
        }
    }

    /// CP reconstruction with component weights `λ`, registered under `name`.
    ///
    /// The weights are part of the *rule* (not of [`OpParams`]) because they are
    /// typed tensor data; the differentiable inputs remain the factor matrices.
    pub fn with_weights(name: impl Into<String>, weights: Array1<T>) -> Self {
        Self {
            weights: Some(weights),
            name: name.into(),
        }
    }

    /// The component weights, if any.
    pub fn weights(&self) -> Option<&Array1<T>> {
        self.weights.as_ref()
    }

    /// Collect the factor matrices and validate their shapes.
    fn factors(&self, inputs: &[DenseND<T>]) -> Result<Vec<Array2<T>>> {
        let mut factors = Vec::with_capacity(inputs.len());
        let mut rank: Option<usize> = None;

        for (mode, input) in inputs.iter().enumerate() {
            let matrix = as_matrix(input, &format!("CP factor {mode}"))?;
            let cols = matrix.shape()[1];
            match rank {
                None => rank = Some(cols),
                Some(expected) if expected != cols => {
                    bail!(
                        "CP factors must share the same rank: factor 0 has {expected} columns, \
                         factor {mode} has {cols}"
                    );
                }
                Some(_) => {}
            }
            factors.push(matrix);
        }

        let rank = rank.ok_or_else(|| anyhow!("CP reconstruction needs at least one factor"))?;
        if rank == 0 {
            bail!("CP rank must be greater than zero");
        }

        if let Some(weights) = &self.weights {
            if weights.len() != rank {
                bail!(
                    "CP weights have length {} but the factors have rank {rank}",
                    weights.len()
                );
            }
        }

        Ok(factors)
    }
}

impl<T> OpRule<T> for CpReconstructionRule<T>
where
    T: AdScalar,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn arity(&self) -> Arity {
        // At least two factors: the Khatri-Rao product of "all other modes"
        // needs one operand.
        Arity::AtLeast(2)
    }

    fn forward(&self, inputs: &[DenseND<T>], _params: &OpParams) -> Result<DenseND<T>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let factors = self.factors(inputs)?;

        let shape: Vec<usize> = factors.iter().map(|f| f.shape()[0]).collect();

        // Fold the weights into mode 0 — mathematically identical to scaling each
        // rank-one component, and it keeps the contraction a single GEMM.
        let mut first = factors[0].clone();
        if let Some(weights) = &self.weights {
            for (r, &weight) in weights.iter().enumerate() {
                for i in 0..first.shape()[0] {
                    first[[i, r]] = first[[i, r]] * weight;
                }
            }
        }

        // Khatri-Rao product of every factor except mode 0. Reuse the exact same
        // helper the CP gradient uses, so the two conventions cannot drift apart.
        let grad_ctx = CpReconstructionGrad::new(factors, None);
        let kr = grad_ctx.khatri_rao_except(0)?;

        // X_(0) = U_0 · KRᵀ  →  (I_0, Π_{n≠0} I_n)
        let unfolded: Array2<T> = first.dot(&kr.t());
        DenseND::fold(&unfolded, &shape, 0)
    }

    fn vjp(
        &self,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
        _params: &OpParams,
    ) -> Result<Vec<DenseND<T>>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let factors = self.factors(inputs)?;

        let expected_shape: Vec<usize> = factors.iter().map(|f| f.shape()[0]).collect();
        if output_grad.shape() != expected_shape.as_slice() {
            bail!(
                "CP reconstruction: output gradient shape {:?} does not match the reconstructed \
                 shape {:?}",
                output_grad.shape(),
                expected_shape
            );
        }

        let grad_ctx = CpReconstructionGrad::new(factors, self.weights.clone());
        let factor_grads = grad_ctx.compute_factor_gradients(output_grad)?;

        Ok(factor_grads.into_iter().map(from_matrix).collect())
    }
}

/// Tucker reconstruction: `[core, U_1, …, U_N]` → dense tensor.
#[derive(Debug, Clone, Copy, Default)]
pub struct TuckerReconstructionRule;

impl TuckerReconstructionRule {
    /// Create the rule (registered as `"tucker_reconstruct"`).
    pub fn new() -> Self {
        Self
    }

    /// Split the inputs into `(core, factors)` and validate their shapes.
    fn split<T: AdScalar>(inputs: &[DenseND<T>]) -> Result<(DenseND<T>, Vec<Array2<T>>)> {
        let core = inputs[0].clone();
        let factors: Vec<Array2<T>> = inputs[1..]
            .iter()
            .enumerate()
            .map(|(mode, input)| as_matrix(input, &format!("Tucker factor {mode}")))
            .collect::<Result<_>>()?;

        if core.rank() != factors.len() {
            bail!(
                "Tucker core has rank {} but {} factor matrices were supplied",
                core.rank(),
                factors.len()
            );
        }

        for (mode, factor) in factors.iter().enumerate() {
            if factor.shape()[1] != core.shape()[mode] {
                bail!(
                    "Tucker factor {mode} has {} columns but the core has extent {} along mode \
                     {mode}",
                    factor.shape()[1],
                    core.shape()[mode]
                );
            }
        }

        Ok((core, factors))
    }
}

impl<T> OpRule<T> for TuckerReconstructionRule
where
    T: AdScalar,
{
    fn name(&self) -> &str {
        "tucker_reconstruct"
    }

    fn arity(&self) -> Arity {
        // Core plus at least one factor.
        Arity::AtLeast(2)
    }

    fn forward(&self, inputs: &[DenseND<T>], _params: &OpParams) -> Result<DenseND<T>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let (core, factors) = Self::split(inputs)?;

        let mut result = core;
        for (mode, factor) in factors.iter().enumerate() {
            result = crate::grad::mode_n_product(&result, factor, mode)?;
        }
        Ok(result)
    }

    fn vjp(
        &self,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
        _params: &OpParams,
    ) -> Result<Vec<DenseND<T>>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let (core, factors) = Self::split(inputs)?;

        let expected_shape: Vec<usize> = factors.iter().map(|f| f.shape()[0]).collect();
        if output_grad.shape() != expected_shape.as_slice() {
            bail!(
                "Tucker reconstruction: output gradient shape {:?} does not match the \
                 reconstructed shape {:?}",
                output_grad.shape(),
                expected_shape
            );
        }

        let grad_ctx = TuckerReconstructionGrad::new(core, factors);
        let core_grad = grad_ctx.compute_core_gradient(output_grad)?;
        let factor_grads = grad_ctx.compute_factor_gradients(output_grad)?;

        let mut grads = Vec::with_capacity(inputs.len());
        grads.push(core_grad);
        grads.extend(factor_grads.into_iter().map(from_matrix));
        Ok(grads)
    }
}

/// Tensor-Train reconstruction: TT cores → dense tensor.
#[derive(Debug, Clone, Copy, Default)]
pub struct TtReconstructionRule;

impl TtReconstructionRule {
    /// Create the rule (registered as `"tt_reconstruct"`).
    pub fn new() -> Self {
        Self
    }

    /// Validate the TT core chain (rank-3 cores, matching bond dimensions,
    /// boundary ranks equal to one).
    fn validate<T: AdScalar>(cores: &[DenseND<T>]) -> Result<Vec<usize>> {
        if cores.is_empty() {
            bail!("TT reconstruction needs at least one core");
        }

        for (k, core) in cores.iter().enumerate() {
            if core.rank() != 3 {
                bail!(
                    "TT core {k} must be rank 3 (r_prev, n_k, r_k), got shape {:?}",
                    core.shape()
                );
            }
        }

        if cores[0].shape()[0] != 1 {
            bail!(
                "TT core 0 must have left rank 1, got {}",
                cores[0].shape()[0]
            );
        }
        let last = cores.len() - 1;
        if cores[last].shape()[2] != 1 {
            bail!(
                "TT core {last} must have right rank 1, got {}",
                cores[last].shape()[2]
            );
        }

        for k in 1..cores.len() {
            let left = cores[k - 1].shape()[2];
            let right = cores[k].shape()[0];
            if left != right {
                bail!(
                    "TT bond mismatch: core {} has right rank {left}, core {k} has left rank \
                     {right}",
                    k - 1
                );
            }
        }

        Ok(cores.iter().map(|core| core.shape()[1]).collect())
    }
}

impl<T> OpRule<T> for TtReconstructionRule
where
    T: AdScalar,
{
    fn name(&self) -> &str {
        "tt_reconstruct"
    }

    fn arity(&self) -> Arity {
        Arity::AtLeast(1)
    }

    fn forward(&self, inputs: &[DenseND<T>], _params: &OpParams) -> Result<DenseND<T>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        Self::validate(inputs)?;
        tt_reconstruct(inputs)
    }

    fn vjp(
        &self,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
        _params: &OpParams,
    ) -> Result<Vec<DenseND<T>>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let expected_shape = Self::validate(inputs)?;

        if output_grad.shape() != expected_shape.as_slice() {
            bail!(
                "TT reconstruction: output gradient shape {:?} does not match the reconstructed \
                 shape {:?}",
                output_grad.shape(),
                expected_shape
            );
        }

        let grad_ctx = TtReconstructionGrad::new(inputs.to_vec());
        grad_ctx.compute_core_gradients(output_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values (no RNG dependency in tests).
    fn ramp(n: usize, offset: f64) -> Vec<f64> {
        (0..n)
            .map(|i| ((i as f64) * 0.37 + offset).sin() + 1.5)
            .collect()
    }

    #[test]
    fn test_cp_forward_matches_naive_definition() {
        let (i0, i1, i2, rank) = (2usize, 3usize, 4usize, 2usize);
        let u0 = DenseND::from_vec(ramp(i0 * rank, 0.1), &[i0, rank]).unwrap();
        let u1 = DenseND::from_vec(ramp(i1 * rank, 0.7), &[i1, rank]).unwrap();
        let u2 = DenseND::from_vec(ramp(i2 * rank, 1.3), &[i2, rank]).unwrap();

        let rule = CpReconstructionRule::<f64>::new();
        let x = rule
            .forward(&[u0.clone(), u1.clone(), u2.clone()], &OpParams::none())
            .unwrap();

        assert_eq!(x.shape(), &[i0, i1, i2]);

        // Naive CP definition: X[i,j,k] = sum_r U0[i,r] U1[j,r] U2[k,r]
        for i in 0..i0 {
            for j in 0..i1 {
                for k in 0..i2 {
                    let mut expected = 0.0;
                    for r in 0..rank {
                        expected += u0.get(&[i, r]).unwrap()
                            * u1.get(&[j, r]).unwrap()
                            * u2.get(&[k, r]).unwrap();
                    }
                    let actual = *x.get(&[i, j, k]).unwrap();
                    assert!(
                        (actual - expected).abs() < 1e-12,
                        "CP mismatch at [{i},{j},{k}]: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_cp_weighted_forward_matches_naive_definition() {
        let (i0, i1, rank) = (3usize, 2usize, 2usize);
        let u0 = DenseND::from_vec(ramp(i0 * rank, 0.2), &[i0, rank]).unwrap();
        let u1 = DenseND::from_vec(ramp(i1 * rank, 0.9), &[i1, rank]).unwrap();
        let weights = Array1::from_vec(vec![0.5, 2.0]);

        let rule = CpReconstructionRule::with_weights("cp_weighted", weights.clone());
        let x = rule
            .forward(&[u0.clone(), u1.clone()], &OpParams::none())
            .unwrap();

        for i in 0..i0 {
            for j in 0..i1 {
                let mut expected = 0.0;
                for r in 0..rank {
                    expected += weights[r] * u0.get(&[i, r]).unwrap() * u1.get(&[j, r]).unwrap();
                }
                assert!((*x.get(&[i, j]).unwrap() - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_cp_rank_mismatch_errors() {
        let u0 = DenseND::<f64>::ones(&[2, 3]);
        let u1 = DenseND::<f64>::ones(&[2, 2]);
        let rule = CpReconstructionRule::<f64>::new();
        assert!(rule.forward(&[u0, u1], &OpParams::none()).is_err());
    }

    #[test]
    fn test_tucker_forward_matches_naive_definition() {
        let core = DenseND::from_vec(ramp(2 * 2, 0.4), &[2, 2]).unwrap();
        let u0 = DenseND::from_vec(ramp(3 * 2, 1.1), &[3, 2]).unwrap();
        let u1 = DenseND::from_vec(ramp(4 * 2, 2.2), &[4, 2]).unwrap();

        let rule = TuckerReconstructionRule::new();
        let x = rule
            .forward(&[core.clone(), u0.clone(), u1.clone()], &OpParams::none())
            .unwrap();

        assert_eq!(x.shape(), &[3, 4]);

        for i in 0..3 {
            for j in 0..4 {
                let mut expected = 0.0;
                for a in 0..2 {
                    for b in 0..2 {
                        expected += core.get(&[a, b]).unwrap()
                            * u0.get(&[i, a]).unwrap()
                            * u1.get(&[j, b]).unwrap();
                    }
                }
                assert!((*x.get(&[i, j]).unwrap() - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_tucker_shape_validation() {
        let core = DenseND::<f64>::ones(&[2, 2]);
        let u0 = DenseND::<f64>::ones(&[3, 3]); // 3 columns vs core extent 2
        let u1 = DenseND::<f64>::ones(&[4, 2]);
        let rule = TuckerReconstructionRule::new();
        assert!(rule.forward(&[core, u0, u1], &OpParams::none()).is_err());
    }

    #[test]
    fn test_tt_forward_matches_grad_module() {
        let core0 = DenseND::from_vec(ramp(4, 0.3), &[1, 2, 2]).unwrap();
        let core1 = DenseND::from_vec(ramp(6, 1.7), &[2, 3, 1]).unwrap();

        let rule = TtReconstructionRule::new();
        let x = rule
            .forward(&[core0.clone(), core1.clone()], &OpParams::none())
            .unwrap();

        assert_eq!(x.shape(), &[2, 3]);

        for i in 0..2 {
            for j in 0..3 {
                let mut expected = 0.0;
                for r in 0..2 {
                    expected += core0.get(&[0, i, r]).unwrap() * core1.get(&[r, j, 0]).unwrap();
                }
                assert!((*x.get(&[i, j]).unwrap() - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_tt_bond_validation() {
        let core0 = DenseND::<f64>::ones(&[1, 2, 3]);
        let core1 = DenseND::<f64>::ones(&[2, 3, 1]); // left rank 2 != right rank 3
        let rule = TtReconstructionRule::new();
        assert!(rule.forward(&[core0, core1], &OpParams::none()).is_err());
    }
}
