//! Tests for the in-place passes and for operation fusion.
//!
//! The fusion tests hold the pass to three independent bars:
//!
//! 1. **Numerics**: a fused plan's forward output equals the unfused plan's and
//!    the eager graph's value, elementwise.
//! 2. **Gradients**: fused VJPs equal the unfused plan's VJPs, equal
//!    `ComputationGraph::backward`, and equal central finite differences
//!    (`gradcheck::check_gradient`). A fused forward with a broken backward is
//!    a silent catastrophe, so it is checked three ways.
//! 3. **Savings**: node counts and *measured* buffer/element allocations drop
//!    by exactly the interior nodes the fusions eliminated.

use super::*;
use crate::gradcheck::{check_gradient, GradCheckConfig};
use crate::graph::{ComputationGraph, Operation, Variable};
use anyhow::Result;
use scirs2_core::ndarray_ext::{array, Array2, ArrayD, IxDyn};
use std::collections::HashMap;
use tenrso_core::DenseND;

/// Tolerance for exact-agreement checks between two analytic paths.
const EXACT_TOL: f64 = 1e-12;

// ============================================================================
// Helpers
// ============================================================================

fn count_op_kind<F>(graph: &ComputationGraph<f64>, pred: F) -> usize
where
    F: Fn(&Operation) -> bool,
{
    graph
        .snapshot_ops()
        .iter()
        .filter(|(_, op)| pred(op))
        .count()
}

/// Deterministic, well-conditioned filler that keeps ReLU pre-activations away
/// from 0 (a finite-difference check across a ReLU kink is meaningless).
fn filled(shape: &[usize], base: f64, step: f64) -> ArrayD<f64> {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n)
        .map(|i| base + step * ((i % 7) as f64 - 3.0) + 0.37 * ((i % 3) as f64))
        .collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape/len mismatch in test filler")
}

fn max_abs_diff(a: &ArrayD<f64>, b: &ArrayD<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape(), "shape mismatch in comparison");
    (a - b).mapv(f64::abs).iter().copied().fold(0.0, f64::max)
}

/// `out = relu(x @ w + b)` with a broadcast bias — the canonical fusible layer.
struct Mlp1 {
    graph: ComputationGraph<f64>,
    x: Variable,
    w: Variable,
    b: Variable,
    out: Variable,
}

fn build_mlp1(m: usize, k: usize, n: usize) -> Result<Mlp1> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(filled(&[m, k], 0.5, 0.25), true)?;
    let w = graph.variable(filled(&[k, n], 0.2, 0.15), true)?;
    let b = graph.variable(filled(&[n], 0.9, 0.30), true)?;
    let xw = graph.matmul(&x, &w)?;
    let xwb = graph.add(&xw, &b)?;
    let out = graph.relu(&xwb)?;
    Ok(Mlp1 {
        graph,
        x,
        w,
        b,
        out,
    })
}

fn feeds_of(
    graph: &ComputationGraph<f64>,
    vars: &[Variable],
) -> Result<HashMap<NodeId, ArrayD<f64>>> {
    let mut feeds = HashMap::new();
    for v in vars {
        feeds.insert(v.id(), graph.value(v)?);
    }
    Ok(feeds)
}

// ============================================================================
// Fusion: detection
// ============================================================================

#[test]
fn test_detect_matmul_bias_relu() -> Result<()> {
    let mlp = build_mlp1(4, 3, 5)?;
    let matches = detect_fusion_patterns(&mlp.graph, &[mlp.out.id()])?;

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].output, mlp.out.id());
    assert_eq!(
        matches[0].eliminated.len(),
        2,
        "matmul and add both absorbed"
    );
    match matches[0].fused {
        FusedOperation::MatMulBiasReLU { lhs, rhs, bias } => {
            assert_eq!(lhs, mlp.x.id());
            assert_eq!(rhs, mlp.w.id());
            assert_eq!(bias, mlp.b.id());
        }
        other => panic!("expected MatMulBiasReLU, got {:?}", other),
    }
    Ok(())
}

#[test]
fn test_detect_matmul_bias() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(filled(&[2, 3], 0.5, 0.2), true)?;
    let w = graph.variable(filled(&[3, 4], 0.3, 0.1), true)?;
    let b = graph.variable(filled(&[4], 0.7, 0.2), true)?;
    let xw = graph.matmul(&x, &w)?;
    let out = graph.add(&xw, &b)?;

    let matches = detect_fusion_patterns(&graph, &[out.id()])?;
    assert_eq!(matches.len(), 1);
    match matches[0].fused {
        FusedOperation::MatMulBias { lhs, rhs, bias } => {
            assert_eq!(lhs, x.id());
            assert_eq!(rhs, w.id());
            assert_eq!(bias, b.id());
        }
        other => panic!("expected MatMulBias, got {:?}", other),
    }
    assert_eq!(matches[0].eliminated, vec![xw.id()]);
    Ok(())
}

#[test]
fn test_detect_add_relu() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(filled(&[4], 0.5, 0.4), true)?;
    let y = graph.variable(filled(&[4], -0.2, 0.3), true)?;
    let s = graph.add(&x, &y)?;
    let out = graph.relu(&s)?;

    let matches = detect_fusion_patterns(&graph, &[out.id()])?;
    assert_eq!(matches.len(), 1);
    match matches[0].fused {
        FusedOperation::AddReLU { lhs, rhs } => {
            assert_eq!(lhs, x.id());
            assert_eq!(rhs, y.id());
        }
        other => panic!("expected AddReLU, got {:?}", other),
    }
    Ok(())
}

#[test]
fn test_detect_mul_add() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(filled(&[4], 0.5, 0.4), true)?;
    let y = graph.variable(filled(&[4], 1.2, 0.3), true)?;
    let c = graph.variable(filled(&[4], -0.7, 0.2), true)?;
    let p = graph.mul(&x, &y)?;
    let out = graph.add(&p, &c)?;

    let matches = detect_fusion_patterns(&graph, &[out.id()])?;
    assert_eq!(matches.len(), 1);
    match matches[0].fused {
        FusedOperation::MulAdd {
            x: fx,
            y: fy,
            c: fc,
        } => {
            assert_eq!(fx, x.id());
            assert_eq!(fy, y.id());
            assert_eq!(fc, c.id());
        }
        other => panic!("expected MulAdd, got {:?}", other),
    }
    Ok(())
}

// ============================================================================
// Fusion: soundness guards
// ============================================================================

#[test]
fn test_no_fusion_when_interior_has_two_consumers() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(filled(&[2, 3], 0.5, 0.2), true)?;
    let w = graph.variable(filled(&[3, 4], 0.3, 0.1), true)?;
    let b = graph.variable(filled(&[4], 0.7, 0.2), true)?;
    let xw = graph.matmul(&x, &w)?;
    let biased = graph.add(&xw, &b)?;
    // A second consumer of the matmul result: its buffer is still needed.
    let other = graph.exp(&xw)?;
    let out = graph.add(&biased, &other)?;

    let plan = compile_fused_plan(&graph, &[out.id()])?;
    assert_eq!(
        plan.stats().fusions_applied,
        0,
        "the matmul feeds two consumers, so nothing may be fused"
    );

    // ...and the plan still computes the right thing.
    let feeds = feeds_of(&graph, &[x, w, b])?;
    let exec = plan.forward(&feeds)?;
    assert!(max_abs_diff(exec.value(out.id())?, &graph.value(&out)?) < EXACT_TOL);
    Ok(())
}

#[test]
fn test_no_fusion_when_interior_is_an_output() -> Result<()> {
    let mlp = build_mlp1(3, 3, 3)?;
    // The user asked for the pre-activation by name, so it must be materialized.
    let pre = mlp
        .graph
        .node_parents(mlp.out.id())
        .expect("relu has a parent")[0];

    let plan = compile_fused_plan(&mlp.graph, &[mlp.out.id(), pre])?;
    assert_eq!(plan.stats().matmul_bias_relu, 0);
    assert_eq!(
        plan.stats().fusions_applied,
        1,
        "matmul+add still fuses; only the relu is blocked"
    );

    let feeds = feeds_of(&mlp.graph, &[mlp.x, mlp.w, mlp.b])?;
    let exec = plan.forward(&feeds)?;
    assert!(max_abs_diff(exec.value(mlp.out.id())?, &mlp.graph.value(&mlp.out)?) < EXACT_TOL);
    Ok(())
}

#[test]
fn test_compile_rejects_reshape_and_broadcast() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(filled(&[2, 3], 0.5, 0.2), true)?;
    let r = graph.reshape(&x, &[3, 2])?;
    let err = compile_fused_plan(&graph, &[r.id()]).expect_err("Reshape is not replayable");
    assert!(err.to_string().contains("Reshape"), "got: {}", err);

    let graph2 = ComputationGraph::<f64>::new();
    let y = graph2.variable(filled(&[3], 0.5, 0.2), true)?;
    let bc = graph2.broadcast(&y, &[2, 3])?;
    let err2 = compile_fused_plan(&graph2, &[bc.id()]).expect_err("Broadcast is not replayable");
    assert!(err2.to_string().contains("Broadcast"), "got: {}", err2);
    Ok(())
}

#[test]
fn test_forward_rejects_missing_feed() -> Result<()> {
    let mlp = build_mlp1(2, 2, 2)?;
    let plan = compile_fused_plan(&mlp.graph, &[mlp.out.id()])?;
    let feeds = feeds_of(&mlp.graph, &[mlp.x, mlp.w])?; // bias missing
    let err = plan
        .forward(&feeds)
        .expect_err("missing feed must be an error");
    assert!(err.to_string().contains("no feed value"), "got: {}", err);
    Ok(())
}

// ============================================================================
// Fusion: numerics, gradients, and measured savings
// ============================================================================

#[test]
fn test_fused_matches_unfused_forward_and_backward() -> Result<()> {
    let mlp = build_mlp1(6, 5, 4)?;
    let outputs = [mlp.out.id()];

    let fused = compile_plan(&mlp.graph, &outputs, &FusionConfig::default())?;
    let plain = compile_plan(
        &mlp.graph,
        &outputs,
        &FusionConfig {
            enable_fusion: false,
        },
    )?;

    // --- structure: measured, not asserted by construction -------------------
    assert_eq!(plain.stats().plan_steps, 3, "matmul, add, relu");
    assert_eq!(fused.stats().plan_steps, 1, "one MatMulBiasReLU");
    assert_eq!(fused.stats().fusions_applied, 1);
    assert_eq!(fused.stats().matmul_bias_relu, 1);
    assert_eq!(fused.stats().interior_nodes_eliminated, 2);
    // Both interior buffers are [6, 4] = 24 elements each.
    assert_eq!(fused.stats().interior_elements_eliminated, 48);

    let feeds = feeds_of(&mlp.graph, &[mlp.x, mlp.w, mlp.b])?;
    let fused_exec = fused.forward(&feeds)?;
    let plain_exec = plain.forward(&feeds)?;

    // --- measured allocations ----------------------------------------------
    assert_eq!(plain_exec.buffers_allocated(), 3);
    assert_eq!(fused_exec.buffers_allocated(), 1);
    assert_eq!(plain_exec.elements_allocated(), 72); // 3 x 24
    assert_eq!(fused_exec.elements_allocated(), 24); // 1 x 24

    // --- numerics -----------------------------------------------------------
    let eager = mlp.graph.value(&mlp.out)?;
    assert!(max_abs_diff(fused_exec.value(mlp.out.id())?, &eager) < EXACT_TOL);
    assert!(max_abs_diff(plain_exec.value(mlp.out.id())?, &eager) < EXACT_TOL);

    // --- gradients ----------------------------------------------------------
    let seed = ArrayD::from_elem(IxDyn(&[6, 4]), 1.0);
    let g_fused = fused.backward(&fused_exec, mlp.out.id(), &seed)?;
    let g_plain = plain.backward(&plain_exec, mlp.out.id(), &seed)?;

    for v in [mlp.x, mlp.w, mlp.b] {
        let gf = g_fused.get(&v.id()).expect("fused gradient");
        let gp = g_plain.get(&v.id()).expect("unfused gradient");
        assert_eq!(gf.shape(), mlp.graph.value(&v)?.shape());
        assert!(
            max_abs_diff(gf, gp) < EXACT_TOL,
            "fused/unfused gradient mismatch for {}: {}",
            v.id(),
            max_abs_diff(gf, gp)
        );
    }
    Ok(())
}

#[test]
fn test_fused_gradients_match_finite_differences() -> Result<()> {
    let mlp = build_mlp1(4, 3, 5)?;
    let plan = compile_fused_plan(&mlp.graph, &[mlp.out.id()])?;
    assert_eq!(plan.stats().fusions_applied, 1);

    let base = feeds_of(&mlp.graph, &[mlp.x, mlp.w, mlp.b])?;
    let seed = ArrayD::from_elem(IxDyn(&[4, 5]), 1.0);
    let config = GradCheckConfig {
        epsilon: 1e-6,
        rtol: 1e-5,
        atol: 1e-7,
        use_central_diff: true,
        verbose: false,
    };

    // Check every leaf, including the broadcast bias (whose VJP must sum over
    // the batch axis).
    for target in [mlp.x, mlp.w, mlp.b] {
        let f = |probe: &DenseND<f64>| -> Result<DenseND<f64>> {
            let mut feeds = base.clone();
            feeds.insert(target.id(), probe.as_array().clone());
            let exec = plan.forward(&feeds)?;
            Ok(DenseND::from_array(exec.value(mlp.out.id())?.clone()))
        };
        let df = |probe: &DenseND<f64>, grad_y: &DenseND<f64>| -> Result<DenseND<f64>> {
            let mut feeds = base.clone();
            feeds.insert(target.id(), probe.as_array().clone());
            let exec = plan.forward(&feeds)?;
            let grads = plan.backward(&exec, mlp.out.id(), grad_y.as_array())?;
            let g = grads
                .get(&target.id())
                .ok_or_else(|| anyhow::anyhow!("no gradient for {}", target.id()))?;
            Ok(DenseND::from_array(g.clone()))
        };

        let x0 = DenseND::from_array(base[&target.id()].clone());
        let gy = DenseND::from_array(seed.clone());
        let result = check_gradient(f, df, &x0, &gy, &config)?;
        assert!(
            result.passed,
            "fused gradcheck failed for leaf {}: max_abs_diff={:.3e} max_rel_diff={:.3e} \
             ({} of {} elements failed)",
            target.id(),
            result.max_abs_diff,
            result.max_rel_diff,
            result.num_failures,
            result.num_elements
        );
    }
    Ok(())
}

#[test]
fn test_fused_backward_matches_graph_backward() -> Result<()> {
    // Same-shape bias so the eager graph's Add VJP is exercised on the
    // non-broadcast path, where it and the plan must agree exactly.
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(filled(&[3, 3], 0.4, 0.3), true)?;
    let w = graph.variable(filled(&[3, 3], 0.25, 0.2), true)?;
    let b = graph.variable(filled(&[3, 3], 0.8, 0.35), true)?;
    let xw = graph.matmul(&x, &w)?;
    let xwb = graph.add(&xw, &b)?;
    let act = graph.relu(&xwb)?;
    let loss = graph.sum(&act)?;

    graph.backward(&loss)?;
    let eager_gx = graph.gradient(&x)?;
    let eager_gw = graph.gradient(&w)?;
    let eager_gb = graph.gradient(&b)?;

    let plan = compile_fused_plan(&graph, &[loss.id()])?;
    assert_eq!(plan.stats().fusions_applied, 1);
    assert_eq!(plan.stats().matmul_bias_relu, 1);

    let feeds = feeds_of(&graph, &[x, w, b])?;
    let exec = plan.forward(&feeds)?;
    let seed = ArrayD::from_elem(IxDyn(&[]), 1.0);
    let grads = plan.backward(&exec, loss.id(), &seed)?;

    assert!(max_abs_diff(grads.get(&x.id()).expect("gx"), &eager_gx) < EXACT_TOL);
    assert!(max_abs_diff(grads.get(&w.id()).expect("gw"), &eager_gw) < EXACT_TOL);
    assert!(max_abs_diff(grads.get(&b.id()).expect("gb"), &eager_gb) < EXACT_TOL);
    Ok(())
}

#[test]
fn test_add_relu_and_mul_add_gradcheck() -> Result<()> {
    // out = relu(x + y) * 1 ... and z = a * b + c, in one graph so both fused
    // kernels are differentiated together.
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(filled(&[5], 0.6, 0.35), true)?;
    let y = graph.variable(filled(&[5], 0.9, 0.2), true)?;
    let c = graph.variable(filled(&[5], -0.3, 0.25), true)?;

    let s = graph.add(&x, &y)?;
    let r = graph.relu(&s)?; // AddReLU
    let p = graph.mul(&r, &y)?;
    let out = graph.add(&p, &c)?; // MulAdd

    let plan = compile_fused_plan(&graph, &[out.id()])?;
    assert_eq!(plan.stats().add_relu, 1);
    assert_eq!(plan.stats().mul_add, 1);
    assert_eq!(plan.stats().fusions_applied, 2);
    assert_eq!(plan.stats().plan_steps, 2, "4 ops fused down to 2 kernels");
    assert_eq!(plan.stats().interior_nodes_eliminated, 2);

    let base = feeds_of(&graph, &[x, y, c])?;
    let exec = plan.forward(&base)?;
    assert!(max_abs_diff(exec.value(out.id())?, &graph.value(&out)?) < EXACT_TOL);

    let seed = ArrayD::from_elem(IxDyn(&[5]), 1.0);
    let config = GradCheckConfig {
        epsilon: 1e-6,
        rtol: 1e-5,
        atol: 1e-7,
        use_central_diff: true,
        verbose: false,
    };
    for target in [x, y, c] {
        let f = |probe: &DenseND<f64>| -> Result<DenseND<f64>> {
            let mut feeds = base.clone();
            feeds.insert(target.id(), probe.as_array().clone());
            let e = plan.forward(&feeds)?;
            Ok(DenseND::from_array(e.value(out.id())?.clone()))
        };
        let df = |probe: &DenseND<f64>, grad_y: &DenseND<f64>| -> Result<DenseND<f64>> {
            let mut feeds = base.clone();
            feeds.insert(target.id(), probe.as_array().clone());
            let e = plan.forward(&feeds)?;
            let g = plan.backward(&e, out.id(), grad_y.as_array())?;
            Ok(DenseND::from_array(
                g.get(&target.id())
                    .ok_or_else(|| anyhow::anyhow!("no gradient"))?
                    .clone(),
            ))
        };
        let x0 = DenseND::from_array(base[&target.id()].clone());
        let gy = DenseND::from_array(seed.clone());
        let result = check_gradient(f, df, &x0, &gy, &config)?;
        assert!(
            result.passed,
            "gradcheck failed for {}: max_abs_diff={:.3e}",
            target.id(),
            result.max_abs_diff
        );
    }
    Ok(())
}

#[test]
fn test_deep_mlp_fuses_every_layer() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let mut h = graph.variable(filled(&[8, 6], 0.3, 0.2), true)?;
    let mut params = vec![h];
    for layer in 0..3 {
        let w = graph.variable(filled(&[6, 6], 0.1 + 0.05 * layer as f64, 0.1), true)?;
        let b = graph.variable(filled(&[6], 0.5, 0.2), true)?;
        params.push(w);
        params.push(b);
        let z = graph.matmul(&h, &w)?;
        let zb = graph.add(&z, &b)?;
        h = graph.relu(&zb)?;
    }
    let loss = graph.sum(&h)?;

    let fused = compile_fused_plan(&graph, &[loss.id()])?;
    let plain = compile_plan(
        &graph,
        &[loss.id()],
        &FusionConfig {
            enable_fusion: false,
        },
    )?;

    assert_eq!(fused.stats().matmul_bias_relu, 3);
    assert_eq!(fused.stats().fusions_applied, 3);
    assert_eq!(plain.stats().plan_steps, 10); // 3 x (matmul, add, relu) + sum
    assert_eq!(fused.stats().plan_steps, 4); // 3 fused layers + sum
    assert_eq!(fused.stats().interior_nodes_eliminated, 6);

    let feeds = feeds_of(&graph, &params)?;
    let fused_exec = fused.forward(&feeds)?;
    let plain_exec = plain.forward(&feeds)?;

    assert_eq!(plain_exec.buffers_allocated(), 10);
    assert_eq!(fused_exec.buffers_allocated(), 4);
    assert!(
        fused_exec.elements_allocated() < plain_exec.elements_allocated(),
        "fusion must materialize fewer elements: {} vs {}",
        fused_exec.elements_allocated(),
        plain_exec.elements_allocated()
    );

    assert!(max_abs_diff(fused_exec.value(loss.id())?, &graph.value(&loss)?) < EXACT_TOL);

    let seed = ArrayD::from_elem(IxDyn(&[]), 1.0);
    let fused_grads = fused.backward(&fused_exec, loss.id(), &seed)?;
    let plain_grads = plain.backward(&plain_exec, loss.id(), &seed)?;

    // Every gradient keeps its leaf's shape, and fused == unfused everywhere.
    for p in &params {
        let gf = fused_grads.get(&p.id()).expect("fused gradient");
        let gp = plain_grads.get(&p.id()).expect("unfused gradient");
        assert_eq!(
            gf.shape(),
            graph.value(p)?.shape(),
            "gradient shape for {}",
            p.id()
        );
        assert!(
            max_abs_diff(gf, gp) < EXACT_TOL,
            "fused/unfused gradient mismatch for {}",
            p.id()
        );
    }

    // Cross-check against the eager tape for the matrix parameters.
    //
    // The bias vectors are deliberately excluded: `ComputationGraph`'s `Add`
    // VJP forwards the output gradient to *both* parents without un-broadcasting
    // it, so a bias of shape [width] broadcast against a [batch, width]
    // pre-activation comes back from `ComputationGraph::backward` with shape
    // [batch, width] — a wrong shape and an un-summed value. The plan's VJP is
    // the correct one (it un-broadcasts), which is why it disagrees here; it is
    // pinned against central finite differences below and in
    // `test_fused_gradients_match_finite_differences`. Comparing against the
    // eager bias gradient would mean asserting that our correct answer equals a
    // wrong one.
    graph.backward(&loss)?;
    for p in params
        .iter()
        .filter(|p| graph.value(p).map(|v| v.ndim() == 2).unwrap_or(false))
    {
        let eager = graph.gradient(p)?;
        let got = fused_grads.get(&p.id()).expect("plan gradient");
        assert!(
            max_abs_diff(got, &eager) < 1e-10,
            "gradient mismatch for {}: {}",
            p.id(),
            max_abs_diff(got, &eager)
        );
    }

    // The bias gradients: verified against central finite differences.
    let base = feeds.clone();
    let config = GradCheckConfig {
        epsilon: 1e-6,
        rtol: 1e-5,
        atol: 1e-7,
        use_central_diff: true,
        verbose: false,
    };
    let loss_seed = DenseND::from_array(seed.clone());
    for bias in params
        .iter()
        .filter(|p| graph.value(p).map(|v| v.ndim() == 1).unwrap_or(false))
    {
        let f = |probe: &DenseND<f64>| -> Result<DenseND<f64>> {
            let mut feeds = base.clone();
            feeds.insert(bias.id(), probe.as_array().clone());
            let e = fused.forward(&feeds)?;
            Ok(DenseND::from_array(e.value(loss.id())?.clone()))
        };
        let df = |probe: &DenseND<f64>, grad_y: &DenseND<f64>| -> Result<DenseND<f64>> {
            let mut feeds = base.clone();
            feeds.insert(bias.id(), probe.as_array().clone());
            let e = fused.forward(&feeds)?;
            let g = fused.backward(&e, loss.id(), grad_y.as_array())?;
            Ok(DenseND::from_array(
                g.get(&bias.id())
                    .ok_or_else(|| anyhow::anyhow!("no gradient"))?
                    .clone(),
            ))
        };
        let x0 = DenseND::from_array(base[&bias.id()].clone());
        let result = check_gradient(f, df, &x0, &loss_seed, &config)?;
        assert!(
            result.passed,
            "bias gradcheck failed for {}: max_abs_diff={:.3e}",
            bias.id(),
            result.max_abs_diff
        );
    }
    Ok(())
}

#[test]
fn test_plan_is_reusable_with_fresh_feeds() -> Result<()> {
    // The whole point of compiling: run the fused graph again on new data.
    let mlp = build_mlp1(4, 4, 3)?;
    let plan = compile_fused_plan(&mlp.graph, &[mlp.out.id()])?;

    let mut feeds = feeds_of(&mlp.graph, &[mlp.x, mlp.w, mlp.b])?;
    let first = plan.forward(&feeds)?;
    let first_out = first.value(mlp.out.id())?.clone();

    // New batch: same shapes, different values.
    feeds.insert(mlp.x.id(), filled(&[4, 4], -0.8, 0.5));
    let second = plan.forward(&feeds)?;
    let second_out = second.value(mlp.out.id())?.clone();

    assert!(
        max_abs_diff(&first_out, &second_out) > 1e-6,
        "a fresh feed must actually change the output"
    );

    // And it still matches an eager graph built on the new data.
    let check = ComputationGraph::<f64>::new();
    let x2 = check.variable(feeds[&mlp.x.id()].clone(), true)?;
    let w2 = check.variable(feeds[&mlp.w.id()].clone(), true)?;
    let b2 = check.variable(feeds[&mlp.b.id()].clone(), true)?;
    let out2 = check.relu(&check.add(&check.matmul(&x2, &w2)?, &b2)?)?;
    assert!(max_abs_diff(&second_out, &check.value(&out2)?) < EXACT_TOL);
    Ok(())
}

#[test]
fn test_matmul_bias_kernel_matches_ndarray() -> Result<()> {
    // The fused MatMulBias kernel is a real GEMM + bias, not a re-labelled add.
    let graph = ComputationGraph::<f64>::new();
    let a = graph.variable(filled(&[3, 4], 0.5, 0.3), true)?;
    let b = graph.variable(filled(&[4, 2], 0.2, 0.25), true)?;
    let bias = graph.variable(filled(&[2], 1.5, 0.4), true)?;
    let ab = graph.matmul(&a, &b)?;
    let out = graph.add(&ab, &bias)?;

    let plan = compile_fused_plan(&graph, &[out.id()])?;
    assert_eq!(plan.stats().matmul_bias, 1);
    assert_eq!(plan.stats().plan_steps, 1);

    let feeds = feeds_of(&graph, &[a, b, bias])?;
    let exec = plan.forward(&feeds)?;

    // Independent reference computed straight from ndarray.
    let a2: Array2<f64> = feeds[&a.id()]
        .clone()
        .into_dimensionality()
        .expect("2-D lhs");
    let b2: Array2<f64> = feeds[&b.id()]
        .clone()
        .into_dimensionality()
        .expect("2-D rhs");
    let mut reference = a2.dot(&b2);
    for mut row in reference.rows_mut() {
        for (j, v) in row.iter_mut().enumerate() {
            *v += feeds[&bias.id()][[j]];
        }
    }
    assert!(max_abs_diff(exec.value(out.id())?, &reference.into_dyn()) < EXACT_TOL);
    Ok(())
}

// ============================================================================
// In-place passes
// ============================================================================

#[test]
fn test_optimizer_creation() {
    let optimizer = GraphOptimizer::new();
    assert_eq!(optimizer.config.passes.len(), 1);
}

#[test]
fn test_optimizer_with_passes() {
    let optimizer = GraphOptimizer::new()
        .with_pass(OptimizationPass::CommonSubexpressionElimination)
        .with_pass(OptimizationPass::DeadCodeElimination);
    assert_eq!(optimizer.config.passes.len(), 3); // default All + 2 added
}

#[test]
fn test_config_builder() {
    let config = OptimizationConfig::new()
        .with_pass(OptimizationPass::CommonSubexpressionElimination)
        .verbose(true)
        .max_iterations(5);

    assert!(config.verbose);
    assert_eq!(config.max_iterations, 5);
    assert_eq!(config.passes.len(), 2);
}

#[test]
fn test_dead_code_elimination() {
    let optimizer = GraphOptimizer::new();
    let ops = vec![
        (NodeId(0), Operation::Input),
        (NodeId(1), Operation::Input),
        (
            NodeId(2),
            Operation::Add {
                lhs: NodeId(0),
                rhs: NodeId(1),
            },
        ),
        (
            NodeId(3),
            Operation::Mul {
                lhs: NodeId(0),
                rhs: NodeId(1),
            },
        ), // Unused
        (
            NodeId(4),
            Operation::Add {
                lhs: NodeId(2),
                rhs: NodeId(0),
            },
        ),
    ];

    let mut outputs = HashSet::new();
    outputs.insert(NodeId(4));

    let dead_nodes = optimizer.eliminate_dead_code(&ops, &outputs);
    assert_eq!(dead_nodes.len(), 1);
    assert!(dead_nodes.contains(&NodeId(3)));
}

#[test]
fn test_get_operation_inputs() {
    let add_op = Operation::Add {
        lhs: NodeId(1),
        rhs: NodeId(2),
    };
    assert_eq!(operation_inputs(&add_op), vec![NodeId(1), NodeId(2)]);

    let relu_op = Operation::ReLU { input: NodeId(3) };
    assert_eq!(operation_inputs(&relu_op), vec![NodeId(3)]);

    let input_op = Operation::Input;
    assert_eq!(operation_inputs(&input_op), vec![]);
}

#[test]
fn test_optimization_stats_reduction() {
    let stats = OptimizationStats {
        nodes_before: 100,
        nodes_after: 80,
        cse_nodes_eliminated: 5,
        dead_nodes_removed: 15,
        constants_folded: 0,
        iterations: 3,
    };

    assert!((stats.reduction_percent() - 20.0).abs() < 1e-10);
}

#[test]
fn test_memory_savings_estimation() {
    let optimizer = GraphOptimizer::new();
    let stats = OptimizationStats {
        nodes_before: 100,
        nodes_after: 80,
        cse_nodes_eliminated: 0,
        dead_nodes_removed: 20,
        constants_folded: 0,
        iterations: 1,
    };

    let savings = optimizer.estimate_memory_savings(&stats, 1024);
    assert_eq!(savings, 20 * 1024);
}

#[test]
fn test_stats_display_reports_only_what_it_measured() {
    let stats = OptimizationStats {
        nodes_before: 50,
        nodes_after: 40,
        cse_nodes_eliminated: 3,
        dead_nodes_removed: 7,
        constants_folded: 0,
        iterations: 2,
    };

    let output = format!("{}", stats);
    assert!(output.contains("Nodes before: 50"));
    assert!(output.contains("Nodes after: 40"));
    assert!(output.contains("CSE nodes eliminated: 3"));
    // The in-place optimizer cannot fuse, so it must never claim to.
    assert!(!output.to_lowercase().contains("fused"));
}

/// The regression this whole module exists for: CSE's node-dedup count must
/// never be reported as a fusion count. `OptimizationStats` no longer has a
/// fusion field at all, and fusion counts come only from a real fused plan.
#[test]
fn test_cse_count_is_not_a_fusion_count() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(array![1.0, 2.0].into_dyn(), true)?;
    let y = graph.variable(array![3.0, 4.0].into_dyn(), true)?;
    let _a = graph.add(&x, &y)?;
    let _b = graph.add(&y, &x)?; // commutative duplicate -> CSE eliminates one
    let out = graph.mul(&x, &y)?;

    let optimizer =
        GraphOptimizer::new().with_pass(OptimizationPass::CommonSubexpressionElimination);
    let stats = optimizer.optimize(&graph)?;
    assert!(
        stats.cse_nodes_eliminated >= 1,
        "CSE really did eliminate a duplicate"
    );

    // ...and a fusion plan over the surviving graph reports zero fusions,
    // because there is nothing fusible in it.
    let plan = compile_fused_plan(&graph, &[out.id()])?;
    assert_eq!(plan.stats().fusions_applied, 0);
    Ok(())
}

#[test]
fn test_optimize_eliminates_dead_nodes() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.constant(array![1.0, 2.0].into_dyn())?;
    let y = graph.constant(array![3.0, 4.0].into_dyn())?;
    let z = graph.add(&x, &y)?;
    let _w = graph.sum(&z)?;
    let _dead = graph.mul(&x, &y)?;

    let nodes_before = graph.num_nodes();
    let optimizer = GraphOptimizer::new();
    let stats = optimizer.optimize(&graph)?;

    assert!(stats.nodes_after <= nodes_before);
    assert_eq!(stats.nodes_before, nodes_before);
    assert!(stats.iterations >= 1);
    Ok(())
}

#[test]
fn test_optimize_constant_folding() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let a = graph.constant(array![2.0, 3.0].into_dyn())?;
    let b = graph.constant(array![4.0, 5.0].into_dyn())?;
    let _c = graph.add(&a, &b)?;

    let optimizer = GraphOptimizer::new();
    let stats = optimizer.optimize(&graph)?;
    assert!(stats.constants_folded >= 1);
    Ok(())
}

// ============================================================================
// Constant folding + CSE (moved from the pre-split module, unchanged)
// ============================================================================

const F64_TOL: f64 = 1e-10;

#[test]
fn test_constant_folding_basic() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let a = graph.constant(array![1.0, 2.0, 3.0].into_dyn())?;
    let b = graph.constant(array![10.0, 20.0, 30.0].into_dyn())?;
    let c = graph.add(&a, &b)?;

    let v_before = graph.value(&c)?;
    assert_eq!(
        count_op_kind(&graph, |op| matches!(op, Operation::Add { .. })),
        1
    );

    let stats = constant_folding(&graph);
    assert_eq!(stats.nodes_folded, 1);

    assert_eq!(
        count_op_kind(&graph, |op| matches!(op, Operation::Add { .. })),
        0
    );
    assert!(graph.is_constant(c.id()));

    let v_after = graph.value(&c)?;
    assert!(max_abs_diff(&v_before, &v_after) < F64_TOL);
    Ok(())
}

#[test]
fn test_constant_folding_chain() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let a = graph.constant(array![2.0].into_dyn())?;
    let b = graph.constant(array![3.0].into_dyn())?;
    let c = graph.constant(array![5.0].into_dyn())?;

    let ab = graph.mul(&a, &b)?;
    let abc = graph.add(&ab, &c)?;

    let expected = graph.value(&abc)?;
    let stats = constant_folding(&graph);
    assert_eq!(stats.nodes_folded, 2);
    assert!(graph.is_constant(ab.id()));
    assert!(graph.is_constant(abc.id()));

    let actual = graph.value(&abc)?;
    assert!(max_abs_diff(&expected, &actual) < F64_TOL);
    assert!((actual[[0]] - 11.0).abs() < F64_TOL);
    Ok(())
}

#[test]
fn test_constant_folding_not_folded_when_variable() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(array![1.0, 2.0].into_dyn(), true)?;
    let c = graph.constant(array![10.0, 20.0].into_dyn())?;
    let y = graph.add(&x, &c)?;

    let stats = constant_folding(&graph);
    assert_eq!(stats.nodes_folded, 0);
    assert!(!graph.is_constant(y.id()));
    assert!(matches!(
        graph.node_operation(y.id()),
        Some(Operation::Add { .. })
    ));
    Ok(())
}

#[test]
fn test_constant_folding_size_threshold() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let a = graph.constant(array![1.0, 2.0, 3.0, 4.0].into_dyn())?;
    let b = graph.constant(array![5.0, 6.0, 7.0, 8.0].into_dyn())?;
    let c = graph.add(&a, &b)?;

    let stats_small = constant_folding_with_threshold(&graph, 3);
    assert_eq!(stats_small.nodes_folded, 0);
    assert_eq!(stats_small.skipped_too_large, 1);
    assert!(!graph.is_constant(c.id()));

    let stats_ok = constant_folding_with_threshold(&graph, 4);
    assert_eq!(stats_ok.nodes_folded, 1);
    assert_eq!(stats_ok.skipped_too_large, 0);
    assert!(graph.is_constant(c.id()));
    Ok(())
}

#[test]
fn test_cse_add_dedup() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(array![1.0, 2.0].into_dyn(), true)?;
    let y = graph.variable(array![3.0, 4.0].into_dyn(), true)?;

    let s1 = graph.add(&x, &y)?;
    let s2 = graph.add(&x, &y)?;

    assert_eq!(
        count_op_kind(&graph, |op| matches!(op, Operation::Add { .. })),
        2
    );

    let stats = common_subexpression_elimination(&graph);
    assert_eq!(stats.nodes_eliminated, 1);
    assert_eq!(
        count_op_kind(&graph, |op| matches!(op, Operation::Add { .. })),
        1
    );

    let v = graph.value(&s1).or_else(|_| graph.value(&s2))?;
    assert!(max_abs_diff(&array![4.0, 6.0].into_dyn(), &v) < F64_TOL);
    Ok(())
}

#[test]
fn test_cse_commutativity() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(array![1.0].into_dyn(), true)?;
    let y = graph.variable(array![2.0].into_dyn(), true)?;

    let _s_xy = graph.add(&x, &y)?;
    let _s_yx = graph.add(&y, &x)?;
    let _p_xy = graph.mul(&x, &y)?;
    let _p_yx = graph.mul(&y, &x)?;

    let stats = common_subexpression_elimination(&graph);
    assert_eq!(stats.nodes_eliminated, 2);
    assert_eq!(
        count_op_kind(&graph, |op| matches!(op, Operation::Add { .. })),
        1
    );
    assert_eq!(
        count_op_kind(&graph, |op| matches!(op, Operation::Mul { .. })),
        1
    );
    Ok(())
}

#[test]
fn test_cse_non_commutative_preserved() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(array![5.0].into_dyn(), true)?;
    let y = graph.variable(array![3.0].into_dyn(), true)?;

    let _d_xy = graph.sub(&x, &y)?;
    let _d_yx = graph.sub(&y, &x)?;

    let stats = common_subexpression_elimination(&graph);
    assert_eq!(stats.nodes_eliminated, 0);
    assert_eq!(
        count_op_kind(&graph, |op| matches!(op, Operation::Sub { .. })),
        2
    );
    Ok(())
}

#[test]
fn test_cse_different_ops_not_deduped() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(array![3.0, 4.0].into_dyn(), true)?;
    let y = graph.variable(array![1.0, 2.0].into_dyn(), true)?;

    let _a = graph.add(&x, &y)?;
    let _m = graph.mul(&x, &y)?;
    let _s = graph.sub(&x, &y)?;

    let stats = common_subexpression_elimination(&graph);
    assert_eq!(stats.nodes_eliminated, 0);
    Ok(())
}

#[test]
fn test_optimize_graph_pipeline_end_to_end() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(array![1.0, 2.0, 3.0].into_dyn(), true)?;
    let two = graph.constant(array![2.0, 2.0, 2.0].into_dyn())?;
    let three = graph.constant(array![3.0, 3.0, 3.0].into_dyn())?;

    let s1 = graph.add(&two, &three)?;
    let s2 = graph.add(&three, &two)?;
    let m1 = graph.mul(&s1, &x)?;
    let m2 = graph.mul(&s2, &x)?;
    let out = graph.add(&m1, &m2)?;

    let expected = graph.value(&out)?;
    let nodes_before = graph.snapshot_ops().len();

    let stats = optimize_graph(&graph, &[out.id()]);

    assert_eq!(stats.nodes_before, nodes_before);
    assert!(stats.cse_pre.nodes_eliminated >= 1);
    assert!(stats.const_fold.nodes_folded >= 1);
    assert!(stats.nodes_after < stats.nodes_before);
    assert_eq!(
        count_op_kind(&graph, |op| matches!(op, Operation::Mul { .. })),
        1
    );

    let actual = graph.value(&out)?;
    assert!(max_abs_diff(&expected, &actual) < F64_TOL);
    assert!(max_abs_diff(&actual, &array![10.0, 20.0, 30.0].into_dyn()) < F64_TOL);
    Ok(())
}

#[test]
fn test_optimize_graph_round_trip_correctness() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(array![0.25, 0.5, 0.75].into_dyn(), true)?;
    let a = graph.constant(array![2.0, 2.0, 2.0].into_dyn())?;
    let b = graph.constant(array![4.0, 4.0, 4.0].into_dyn())?;

    let ab1 = graph.mul(&a, &b)?;
    let ab2 = graph.mul(&b, &a)?;
    let t1 = graph.mul(&ab1, &x)?;
    let t2 = graph.mul(&ab2, &x)?;
    let t3 = graph.relu(&t1)?;
    let t4 = graph.exp(&t2)?;
    let out = graph.add(&t3, &t4)?;

    let expected = graph.value(&out)?;
    let _stats = optimize_graph(&graph, &[out.id()]);
    let actual = graph.value(&out)?;

    assert!(max_abs_diff(&expected, &actual) < F64_TOL);
    Ok(())
}

#[test]
fn test_dce_on_graph_removes_unreachable() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let x = graph.variable(array![1.0].into_dyn(), true)?;
    let y = graph.variable(array![2.0].into_dyn(), true)?;
    let used = graph.add(&x, &y)?;
    let _unused = graph.mul(&x, &y)?;

    let before = graph.snapshot_ops().len();
    let stats = dead_code_elimination_on_graph(&graph, &[used.id()]);
    let after = graph.snapshot_ops().len();

    assert!(stats.nodes_removed >= 1);
    assert!(after < before);
    assert!(graph.node_operation(used.id()).is_some());
    Ok(())
}
