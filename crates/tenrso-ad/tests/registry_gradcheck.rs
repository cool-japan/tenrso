//! Finite-difference verification of every rule in the TenRSo op registry.
//!
//! Each registered rule is checked with [`tenrso_ad::gradcheck::check_gradient`]:
//! the analytic VJP produced by the rule is compared against a central-difference
//! numerical gradient of the rule's own forward pass, for **every** input of the
//! op.
//!
//! Non-differentiable points are avoided deliberately (values are kept away from
//! the kinks of `relu`/`abs`/`min`/`max`, away from zero for `ln`/`div`/`sqrt`,
//! and free of ties for the max/min reductions), because a finite-difference
//! check is meaningless at a kink.

use anyhow::Result;
use scirs2_core::ndarray_ext::Array1;
use tenrso_ad::gradcheck::{check_gradient, GradCheckConfig, GradCheckResult};
use tenrso_ad::registry::{
    BinaryOpKind, CpReconstructionRule, OpParams, OpRegistry, OpRule, ReduceKind,
    TtReconstructionRule, TuckerReconstructionRule, UnaryOpKind,
};
use tenrso_core::DenseND;

/// Deterministic, well-conditioned values (no RNG, reproducible).
fn values(n: usize, offset: f64, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|i| scale * (((i as f64) * 0.7321 + offset).sin() + 1.35))
        .collect()
}

fn tensor(shape: &[usize], offset: f64, scale: f64) -> DenseND<f64> {
    let n: usize = shape.iter().product();
    DenseND::from_vec(values(n, offset, scale), shape).expect("valid tensor")
}

/// Gradient check of `rule` with respect to input `k`.
fn check_input(
    rule: &dyn OpRule<f64>,
    inputs: &[DenseND<f64>],
    k: usize,
    params: &OpParams,
    grad_out: &DenseND<f64>,
) -> Result<GradCheckResult> {
    let forward = |x: &DenseND<f64>| -> Result<DenseND<f64>> {
        let mut probe = inputs.to_vec();
        probe[k] = x.clone();
        rule.forward(&probe, params)
    };

    let backward = |x: &DenseND<f64>, grad: &DenseND<f64>| -> Result<DenseND<f64>> {
        let mut probe = inputs.to_vec();
        probe[k] = x.clone();
        let grads = rule.vjp(&probe, grad, params)?;
        Ok(grads[k].clone())
    };

    check_gradient(
        forward,
        backward,
        &inputs[k],
        grad_out,
        &GradCheckConfig::default(),
    )
}

/// Gradient check of `rule` with respect to every input.
fn check_all_inputs(rule: &dyn OpRule<f64>, inputs: &[DenseND<f64>], params: &OpParams) {
    let output = rule
        .forward(inputs, params)
        .unwrap_or_else(|e| panic!("forward failed for '{}': {e:#}", rule.name()));

    // A non-uniform cotangent catches gradients that are only right for ones().
    let grad_out = tensor(output.shape(), 2.71, 0.8);

    for k in 0..inputs.len() {
        let result = check_input(rule, inputs, k, params, &grad_out)
            .unwrap_or_else(|e| panic!("gradcheck failed for '{}' input {k}: {e:#}", rule.name()));

        assert!(
            result.passed,
            "'{}' input {k}: gradient mismatch ({} / {} elements), max_abs={:.3e}, max_rel={:.3e}",
            rule.name(),
            result.num_failures,
            result.num_elements,
            result.max_abs_diff,
            result.max_rel_diff
        );
    }
}

fn registry() -> OpRegistry<f64> {
    OpRegistry::with_builtins()
}

// ---------------------------------------------------------------------------
// Einsum
// ---------------------------------------------------------------------------

#[test]
fn gradcheck_einsum_matmul() {
    let registry = registry();
    let rule = registry.require("einsum").expect("einsum rule");

    let a = tensor(&[3, 4], 0.1, 1.0);
    let b = tensor(&[4, 2], 1.3, 1.0);

    check_all_inputs(rule, &[a, b], &OpParams::einsum("ij,jk->ik"));
}

#[test]
fn gradcheck_einsum_batched() {
    let registry = registry();
    let rule = registry.require("einsum").expect("einsum rule");

    let a = tensor(&[2, 3, 4], 0.4, 1.0);
    let b = tensor(&[2, 4, 2], 2.1, 1.0);

    check_all_inputs(rule, &[a, b], &OpParams::einsum("bij,bjk->bik"));
}

#[test]
fn gradcheck_einsum_scalar_output() {
    let registry = registry();
    let rule = registry.require("einsum").expect("einsum rule");

    let a = tensor(&[5], 0.2, 1.0);
    let b = tensor(&[5], 1.9, 1.0);

    check_all_inputs(rule, &[a, b], &OpParams::einsum("i,i->"));
}

#[test]
fn gradcheck_einsum_ternary() {
    let registry = registry();
    let rule = registry.require("einsum").expect("einsum rule");

    let a = tensor(&[2, 3], 0.3, 1.0);
    let b = tensor(&[3, 4], 1.1, 1.0);
    let c = tensor(&[4, 2], 2.6, 1.0);

    check_all_inputs(rule, &[a, b, c], &OpParams::einsum("ij,jk,kl->il"));
}

#[test]
fn gradcheck_einsum_unary_transpose_and_sum() {
    let registry = registry();
    let rule = registry.require("einsum").expect("einsum rule");

    let x = tensor(&[3, 4], 0.9, 1.0);
    check_all_inputs(rule, std::slice::from_ref(&x), &OpParams::einsum("ij->ji"));
    check_all_inputs(rule, std::slice::from_ref(&x), &OpParams::einsum("ij->i"));

    let square = tensor(&[3, 3], 1.7, 1.0);
    check_all_inputs(
        rule,
        std::slice::from_ref(&square),
        &OpParams::einsum("ii->"),
    );
    check_all_inputs(
        rule,
        std::slice::from_ref(&square),
        &OpParams::einsum("ii->i"),
    );
}

#[test]
fn gradcheck_einsum_broadcast_adjoint() {
    // 'j' is summed away inside operand 0: the adjoint must expand it back.
    let registry = registry();
    let rule = registry.require("einsum").expect("einsum rule");

    let a = tensor(&[3, 4], 0.6, 1.0);
    let b = tensor(&[3], 2.4, 1.0);

    check_all_inputs(rule, &[a, b], &OpParams::einsum("ij,i->i"));
}

// ---------------------------------------------------------------------------
// Element-wise
// ---------------------------------------------------------------------------

#[test]
fn gradcheck_all_unary_rules() {
    let registry = registry();

    for kind in UnaryOpKind::ALL {
        let rule = registry.require(kind.name()).expect("unary rule");

        // Strictly positive, away from 0 (ln/sqrt/relu/abs are well defined and
        // differentiable there) and moderate in magnitude (exp stays finite).
        let x = tensor(&[3, 4], 0.25, 0.6);
        assert!(
            x.as_slice().iter().all(|&v| v > 0.1),
            "unary probe must stay away from the kinks"
        );

        check_all_inputs(rule, std::slice::from_ref(&x), &OpParams::none());
    }
}

#[test]
fn gradcheck_unary_rules_on_negative_values() {
    // Re-check the ops that are defined on negatives, including the strictly
    // negative branch of relu/abs.
    let registry = registry();

    for kind in [
        UnaryOpKind::Neg,
        UnaryOpKind::OneMinus,
        UnaryOpKind::Relu,
        UnaryOpKind::Sigmoid,
        UnaryOpKind::Tanh,
        UnaryOpKind::Exp,
        UnaryOpKind::Square,
        UnaryOpKind::Abs,
    ] {
        let rule = registry.require(kind.name()).expect("unary rule");
        let x = DenseND::from_vec(vec![-2.0, -0.75, -1.5, -3.25], &[2, 2]).expect("tensor");
        check_all_inputs(rule, std::slice::from_ref(&x), &OpParams::none());
    }
}

#[test]
fn gradcheck_all_binary_rules() {
    let registry = registry();

    for kind in BinaryOpKind::ALL {
        let rule = registry.require(kind.name()).expect("binary rule");

        // Distinct, strictly positive values: no min/max ties, no division by
        // zero, and every comparison is strict (so the numerical gradient of the
        // predicates is exactly zero).
        let x = tensor(&[3, 4], 0.15, 0.7);
        let y = tensor(&[3, 4], 3.05, 0.9);

        for (a, b) in x.as_slice().iter().zip(y.as_slice().iter()) {
            assert!(
                (a - b).abs() > 1e-3,
                "binary probe must avoid ties (got {a} vs {b})"
            );
            assert!(*b > 0.1, "binary probe must avoid division by zero");
        }

        check_all_inputs(rule, &[x, y], &OpParams::none());
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

#[test]
fn gradcheck_all_reductions_single_axis() {
    let registry = registry();

    for kind in ReduceKind::ALL {
        let rule = registry.require(kind.name()).expect("reduction rule");
        let x = tensor(&[3, 4], 0.11, 0.9);

        for axis in 0..2usize {
            check_all_inputs(rule, std::slice::from_ref(&x), &OpParams::axes(vec![axis]));
        }
    }
}

#[test]
fn gradcheck_all_reductions_multi_axis() {
    let registry = registry();

    for kind in ReduceKind::ALL {
        let rule = registry.require(kind.name()).expect("reduction rule");
        let x = tensor(&[2, 3, 2], 0.33, 0.8);

        // Full reduction and a middle-axis pair.
        check_all_inputs(
            rule,
            std::slice::from_ref(&x),
            &OpParams::axes(vec![0, 1, 2]),
        );
        check_all_inputs(rule, std::slice::from_ref(&x), &OpParams::axes(vec![0, 2]));
    }
}

#[test]
fn gradcheck_reduction_identity() {
    let registry = registry();
    let rule = registry.require("reduce_sum").expect("reduction rule");
    let x = tensor(&[2, 3], 0.5, 1.0);
    check_all_inputs(rule, std::slice::from_ref(&x), &OpParams::none());
}

// ---------------------------------------------------------------------------
// Decompositions
// ---------------------------------------------------------------------------

#[test]
fn gradcheck_cp_reconstruction() {
    let registry = registry();
    let rule = registry.require("cp_reconstruct").expect("cp rule");

    let rank = 3;
    let u0 = tensor(&[3, rank], 0.2, 0.9);
    let u1 = tensor(&[4, rank], 1.4, 0.8);
    let u2 = tensor(&[2, rank], 2.8, 1.1);

    check_all_inputs(rule, &[u0, u1, u2], &OpParams::none());
}

#[test]
fn gradcheck_cp_reconstruction_weighted() {
    // A weighted CP rule is a first-class registry citizen: register it and check
    // it exactly like a builtin.
    let mut registry = registry();
    let weights = Array1::from_vec(vec![0.75, -1.5, 2.25]);
    registry
        .register(Box::new(CpReconstructionRule::with_weights(
            "cp_reconstruct_weighted",
            weights,
        )))
        .expect("register weighted CP rule");

    let rule = registry
        .require("cp_reconstruct_weighted")
        .expect("weighted cp rule");

    let u0 = tensor(&[3, 3], 0.7, 0.9);
    let u1 = tensor(&[2, 3], 1.9, 1.2);

    check_all_inputs(rule, &[u0, u1], &OpParams::none());
}

#[test]
fn gradcheck_tucker_reconstruction() {
    let registry = registry();
    let rule = registry.require("tucker_reconstruct").expect("tucker rule");

    let core = tensor(&[2, 3, 2], 0.45, 0.9);
    let u0 = tensor(&[3, 2], 1.15, 1.0);
    let u1 = tensor(&[4, 3], 2.35, 0.8);
    let u2 = tensor(&[2, 2], 3.05, 1.1);

    check_all_inputs(rule, &[core, u0, u1, u2], &OpParams::none());
}

#[test]
fn gradcheck_tt_reconstruction() {
    let registry = registry();
    let rule = registry.require("tt_reconstruct").expect("tt rule");

    // Cores (1,3,2), (2,4,2), (2,2,1) -> a 3x4x2 tensor.
    let core0 = tensor(&[1, 3, 2], 0.31, 0.9);
    let core1 = tensor(&[2, 4, 2], 1.27, 0.7);
    let core2 = tensor(&[2, 2, 1], 2.19, 1.0);

    check_all_inputs(rule, &[core0, core1, core2], &OpParams::none());
}

#[test]
fn gradcheck_tt_reconstruction_two_cores() {
    let registry = registry();
    let rule = registry.require("tt_reconstruct").expect("tt rule");

    let core0 = tensor(&[1, 2, 3], 0.9, 1.0);
    let core1 = tensor(&[3, 3, 1], 2.2, 0.9);

    check_all_inputs(rule, &[core0, core1], &OpParams::none());
}

// ---------------------------------------------------------------------------
// Registry-level guarantees
// ---------------------------------------------------------------------------

#[test]
fn every_builtin_rule_is_covered_by_a_gradcheck() {
    // Guards against adding a builtin without a finite-difference test.
    let registry = registry();

    let mut expected: Vec<&str> = Vec::new();
    expected.push("einsum");
    expected.extend(UnaryOpKind::ALL.iter().map(|k| k.name()));
    expected.extend(BinaryOpKind::ALL.iter().map(|k| k.name()));
    expected.extend(ReduceKind::ALL.iter().map(|k| k.name()));
    expected.push("cp_reconstruct");
    expected.push("tucker_reconstruct");
    expected.push("tt_reconstruct");
    expected.sort_unstable();

    assert_eq!(
        registry.names(),
        expected,
        "the builtin set changed: add a gradcheck for the new rule"
    );
}

#[test]
fn registry_rejects_wrong_gradient_shapes() {
    // The registry validates what a rule returns, so a mis-shaped custom rule
    // cannot poison a tape.
    let registry = registry();
    let x = tensor(&[2, 3], 0.5, 1.0);

    let err = registry
        .vjp(
            "reduce_sum",
            std::slice::from_ref(&x),
            &DenseND::<f64>::ones(&[7]),
            &OpParams::axes(vec![0]),
        )
        .unwrap_err();
    assert!(err.to_string().contains("does not match"), "{err:#}");
}

#[test]
fn decomposition_rules_reject_wrong_arity() {
    let cp = CpReconstructionRule::<f64>::new();
    let single = tensor(&[3, 2], 0.4, 1.0);
    assert!(cp
        .forward(std::slice::from_ref(&single), &OpParams::none())
        .is_err());

    let tucker = TuckerReconstructionRule::new();
    assert!(OpRule::<f64>::forward(&tucker, &[], &OpParams::none()).is_err());

    let tt = TtReconstructionRule::new();
    assert!(OpRule::<f64>::forward(&tt, &[], &OpParams::none()).is_err());
}
