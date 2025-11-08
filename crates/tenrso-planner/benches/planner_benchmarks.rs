//! Benchmarks for tensor contraction planners
//!
//! Measures planning time and plan quality for various tensor network topologies.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use tenrso_planner::{dp_planner, greedy_planner, EinsumSpec, PlanHints};

/// Benchmark greedy planner on matrix chain multiplication
fn bench_greedy_matrix_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_matrix_chain");

    for n in [3, 4, 5, 6, 8, 10].iter() {
        let (spec, shapes) = create_matrix_chain(*n);
        let hints = PlanHints::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let _ = greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints));
            });
        });
    }

    group.finish();
}

/// Benchmark DP planner on small matrix chains
fn bench_dp_matrix_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("dp_matrix_chain");

    // DP is expensive, so test smaller sizes
    for n in [3, 4, 5, 6, 7, 8].iter() {
        let (spec, shapes) = create_matrix_chain(*n);
        let hints = PlanHints::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let _ = dp_planner(black_box(&spec), black_box(&shapes), black_box(&hints));
            });
        });
    }

    group.finish();
}

/// Benchmark greedy planner on star topology (all tensors share a common index)
fn bench_greedy_star(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_star");

    for n in [3, 5, 8, 10, 15, 20].iter() {
        let (spec, shapes) = create_star_topology(*n);
        let hints = PlanHints::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let _ = greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints));
            });
        });
    }

    group.finish();
}

/// Benchmark DP planner on star topology
fn bench_dp_star(c: &mut Criterion) {
    let mut group = c.benchmark_group("dp_star");

    // DP is expensive, so test smaller sizes
    for n in [3, 5, 8, 10].iter() {
        let (spec, shapes) = create_star_topology(*n);
        let hints = PlanHints::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let _ = dp_planner(black_box(&spec), black_box(&shapes), black_box(&hints));
            });
        });
    }

    group.finish();
}

/// Benchmark plan quality: compare greedy vs DP costs
fn bench_plan_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_quality");

    for n in [3, 4, 5, 6].iter() {
        let (spec, shapes) = create_matrix_chain(*n);
        let hints = PlanHints::default();

        // Measure greedy cost
        group.bench_with_input(BenchmarkId::new("greedy_cost", n), n, |b, _| {
            b.iter(|| {
                let plan = greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints))
                    .unwrap();
                black_box(plan.estimated_flops)
            });
        });

        // Measure DP cost
        group.bench_with_input(BenchmarkId::new("dp_cost", n), n, |b, _| {
            b.iter(|| {
                let plan =
                    dp_planner(black_box(&spec), black_box(&shapes), black_box(&hints)).unwrap();
                black_box(plan.estimated_flops)
            });
        });
    }

    group.finish();
}

/// Benchmark greedy planner on dense tensor networks
fn bench_greedy_dense_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_dense_network");

    for n in [4, 6, 8, 10, 12].iter() {
        let (spec, shapes) = create_dense_network(*n);
        let hints = PlanHints::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let _ = greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints));
            });
        });
    }

    group.finish();
}

/// Benchmark tensor size sensitivity
fn bench_size_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("size_sensitivity");

    for size in [10, 50, 100, 200, 500].iter() {
        let (spec, shapes) = create_matrix_chain_with_size(4, *size);
        let hints = PlanHints::default();

        group.bench_with_input(BenchmarkId::new("greedy", size), size, |b, _| {
            b.iter(|| {
                let _ = greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints));
            });
        });
    }

    group.finish();
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a matrix chain: A_ij * B_jk * C_kl * ... -> output
fn create_matrix_chain(n: usize) -> (EinsumSpec, Vec<Vec<usize>>) {
    let indices: Vec<char> = "ijklmnopqrstuvwxyz".chars().collect();

    let mut input_specs = Vec::new();
    let mut shapes = Vec::new();

    for i in 0..n {
        let idx1 = indices[i];
        let idx2 = indices[i + 1];
        input_specs.push(format!("{}{}", idx1, idx2));

        // Varying sizes for interesting optimization
        let size1 = 10 + i * 5;
        let size2 = 10 + (i + 1) * 5;
        shapes.push(vec![size1, size2]);
    }

    let output = format!("{}{}", indices[0], indices[n]);
    let spec_str = format!("{}->{}", input_specs.join(","), output);
    let spec = EinsumSpec::parse(&spec_str).expect("Valid spec");

    (spec, shapes)
}

/// Create a matrix chain with specific tensor size
fn create_matrix_chain_with_size(n: usize, size: usize) -> (EinsumSpec, Vec<Vec<usize>>) {
    let indices: Vec<char> = "ijklmnopqrstuvwxyz".chars().collect();

    let mut input_specs = Vec::new();
    let mut shapes = Vec::new();

    for i in 0..n {
        let idx1 = indices[i];
        let idx2 = indices[i + 1];
        input_specs.push(format!("{}{}", idx1, idx2));
        shapes.push(vec![size, size]);
    }

    let output = format!("{}{}", indices[0], indices[n]);
    let spec_str = format!("{}->{}", input_specs.join(","), output);
    let spec = EinsumSpec::parse(&spec_str).expect("Valid spec");

    (spec, shapes)
}

/// Create a star topology: all tensors share a common index
fn create_star_topology(n: usize) -> (EinsumSpec, Vec<Vec<usize>>) {
    let indices: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();

    let mut input_specs = Vec::new();
    let mut shapes = Vec::new();

    // Common index is 'i'
    // Tensors: ia, ib, ic, id, ...
    for (j, &idx) in indices.iter().enumerate().take(n) {
        input_specs.push(format!("i{}", idx));
        shapes.push(vec![10, 5 + j]);
    }

    // Output includes all indices
    let mut output = String::from("i");
    for &idx in indices.iter().take(n) {
        output.push(idx);
    }

    let spec_str = format!("{}->{}", input_specs.join(","), output);
    let spec = EinsumSpec::parse(&spec_str).expect("Valid spec");

    (spec, shapes)
}

/// Create a dense network where many indices are shared
fn create_dense_network(n: usize) -> (EinsumSpec, Vec<Vec<usize>>) {
    let indices: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();

    let mut input_specs = Vec::new();
    let mut shapes = Vec::new();

    // Create a cycle: ab, bc, cd, ..., za
    for i in 0..n {
        let idx1 = indices[i % indices.len()];
        let idx2 = indices[(i + 1) % indices.len()];
        input_specs.push(format!("{}{}", idx1, idx2));
        shapes.push(vec![8, 8]);
    }

    // Output is all unique indices in the cycle
    let mut output_indices = std::collections::HashSet::new();
    for spec in &input_specs {
        for c in spec.chars() {
            output_indices.insert(c);
        }
    }
    let mut output: Vec<char> = output_indices.into_iter().collect();
    output.sort();
    let output_str: String = output.into_iter().collect();

    let spec_str = format!("{}->{}", input_specs.join(","), output_str);
    let spec = EinsumSpec::parse(&spec_str).expect("Valid spec");

    (spec, shapes)
}

criterion_group!(
    benches,
    bench_greedy_matrix_chain,
    bench_dp_matrix_chain,
    bench_greedy_star,
    bench_dp_star,
    bench_plan_quality,
    bench_greedy_dense_network,
    bench_size_sensitivity,
);

criterion_main!(benches);
