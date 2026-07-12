//! Operation fusion on a computation graph: what it eliminates, and what that costs.
//!
//! Builds an MLP (`loss = sum(relu(...relu(x @ w0 + b0)... @ wL + bL))`),
//! compiles it twice — once with fusion, once without — and then:
//!
//! 1. reports the *counted* structural savings (steps, buffers, elements),
//! 2. checks that the fused plan agrees with the unfused one numerically and
//!    on every gradient, and
//! 3. times forward+backward for both, reporting the median of several
//!    repetitions (a single timing on a contended machine is noise).
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p tenrso-ad --example graph_fusion
//! ```

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayD, IxDyn};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tenrso_ad::graph::{ComputationGraph, NodeId, Variable};
use tenrso_ad::graph_optimizer::{compile_plan, FusionConfig, FusionPlan};

const BATCH: usize = 64;
const WIDTH: usize = 128;
const LAYERS: usize = 3;
const REPS: usize = 9;

fn main() -> Result<()> {
    let graph = ComputationGraph::<f64>::new();
    let mut h = graph.variable(
        Array2::<f64>::from_shape_fn((BATCH, WIDTH), |(i, j)| {
            0.01 * (i as f64) - 0.02 * (j as f64) + 0.5
        })
        .into_dyn(),
        true,
    )?;
    let mut params: Vec<Variable> = vec![h];

    for layer in 0..LAYERS {
        let w = graph.variable(
            Array2::<f64>::from_shape_fn((WIDTH, WIDTH), |(i, j)| {
                0.001 * ((i + j + layer) % 11) as f64 - 0.004
            })
            .into_dyn(),
            true,
        )?;
        let b = graph.variable(Array1::<f64>::from_elem(WIDTH, 0.1).into_dyn(), true)?;
        params.push(w);
        params.push(b);

        let z = graph.matmul(&h, &w)?;
        let zb = graph.add(&z, &b)?;
        h = graph.relu(&zb)?;
    }
    let loss = graph.sum(&h)?;
    let outputs = [loss.id()];

    let fused = compile_plan(&graph, &outputs, &FusionConfig::default())?;
    let unfused = compile_plan(
        &graph,
        &outputs,
        &FusionConfig {
            enable_fusion: false,
        },
    )?;

    println!("=== Structure (counted, not estimated) ===");
    println!("graph nodes (live)       : {}", fused.stats().graph_nodes);
    println!(
        "plan steps  unfused/fused: {} / {}",
        unfused.stats().plan_steps,
        fused.stats().plan_steps
    );
    println!(
        "fusions applied          : {} (MatMulBiasReLU={}, MatMulBias={}, MulAdd={}, AddReLU={})",
        fused.stats().fusions_applied,
        fused.stats().matmul_bias_relu,
        fused.stats().matmul_bias,
        fused.stats().mul_add,
        fused.stats().add_relu
    );
    println!(
        "interior nodes eliminated: {} ({} elements of forward buffer, and the same again in backward)",
        fused.stats().interior_nodes_eliminated,
        fused.stats().interior_elements_eliminated
    );

    let feeds: HashMap<NodeId, ArrayD<f64>> = params
        .iter()
        .map(|v| graph.value(v).map(|val| (v.id(), val)))
        .collect::<Result<_>>()?;
    let seed = ArrayD::from_elem(IxDyn(&[]), 1.0);

    let fused_exec = fused.forward(&feeds)?;
    let unfused_exec = unfused.forward(&feeds)?;

    println!("\n=== Measured allocations (one forward pass) ===");
    println!(
        "buffers  unfused/fused: {} / {}",
        unfused_exec.buffers_allocated(),
        fused_exec.buffers_allocated()
    );
    println!(
        "elements unfused/fused: {} / {}  ({:.1}% fewer)",
        unfused_exec.elements_allocated(),
        fused_exec.elements_allocated(),
        100.0
            * (1.0
                - fused_exec.elements_allocated() as f64
                    / unfused_exec.elements_allocated() as f64)
    );

    // ---- correctness: fused must match unfused, forward and backward --------
    let diff = (fused_exec.value(loss.id())? - unfused_exec.value(loss.id())?)
        .mapv(f64::abs)
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let fused_grads = fused.backward(&fused_exec, loss.id(), &seed)?;
    let unfused_grads = unfused.backward(&unfused_exec, loss.id(), &seed)?;
    let mut grad_diff = 0.0_f64;
    for p in &params {
        let a = fused_grads.get(&p.id()).expect("fused gradient");
        let b = unfused_grads.get(&p.id()).expect("unfused gradient");
        grad_diff = grad_diff.max(
            (a - b)
                .mapv(f64::abs)
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
        );
    }
    println!("\n=== Agreement with the unfused plan ===");
    println!("max |Δ loss|      : {:.3e}", diff);
    println!("max |Δ gradient|  : {:.3e}", grad_diff);
    assert!(diff < 1e-12 && grad_diff < 1e-10, "fusion changed the math");

    // ---- timing -------------------------------------------------------------
    let unfused_times = time_plan(&unfused, &feeds, loss.id(), &seed)?;
    let fused_times = time_plan(&fused, &feeds, loss.id(), &seed)?;

    println!("\n=== forward + backward, {REPS} reps (machine may be contended) ===");
    report("unfused", &unfused_times);
    report("fused  ", &fused_times);
    let su = median(&unfused_times).as_secs_f64();
    let sf = median(&fused_times).as_secs_f64();
    println!("speedup (median)  : {:.2}x", su / sf);

    Ok(())
}

fn time_plan(
    plan: &FusionPlan,
    feeds: &HashMap<NodeId, ArrayD<f64>>,
    loss: NodeId,
    seed: &ArrayD<f64>,
) -> Result<Vec<Duration>> {
    // Warm up allocator / caches.
    for _ in 0..2 {
        let exec = plan.forward(feeds)?;
        let _ = plan.backward(&exec, loss, seed)?;
    }
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        let exec = plan.forward(feeds)?;
        let grads = plan.backward(&exec, loss, seed)?;
        times.push(start.elapsed());
        std::hint::black_box(grads.len());
    }
    Ok(times)
}

fn median(times: &[Duration]) -> Duration {
    let mut sorted = times.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}

fn report(label: &str, times: &[Duration]) {
    let mut sorted = times.to_vec();
    sorted.sort();
    println!(
        "{label}: median {:>8.3} ms   [min {:>8.3}, max {:>8.3}]",
        median(times).as_secs_f64() * 1e3,
        sorted[0].as_secs_f64() * 1e3,
        sorted[sorted.len() - 1].as_secs_f64() * 1e3
    );
}
