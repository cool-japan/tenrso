//! Benchmarks for tensor contraction planners
//!
//! Measures planning time and plan quality for various tensor network topologies.
//! Includes all 6 planning algorithms: Greedy, Beam Search, DP, SA, GA, and Adaptive.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use tenrso_planner::{
    beam_search_planner, dp_planner, genetic_algorithm_planner, greedy_planner,
    simulated_annealing_planner, AdaptivePlanner, BeamSearchPlanner, DPPlanner, EinsumSpec,
    GeneticAlgorithmPlanner, GreedyPlanner, PlanHints, Planner, SimulatedAnnealingPlanner,
};

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

/// Benchmark Beam Search planner on matrix chains
fn bench_beam_search_matrix_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("beam_search_matrix_chain");

    for n in [3, 4, 5, 6, 8].iter() {
        let (spec, shapes) = create_matrix_chain(*n);
        let hints = PlanHints::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let _ =
                    beam_search_planner(black_box(&spec), black_box(&shapes), black_box(&hints), 5);
            });
        });
    }

    group.finish();
}

/// Benchmark Simulated Annealing planner on matrix chains
fn bench_sa_matrix_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("sa_matrix_chain");

    for n in [3, 4, 5, 6, 8, 10].iter() {
        let (spec, shapes) = create_matrix_chain(*n);
        let hints = PlanHints::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let _ = simulated_annealing_planner(
                    black_box(&spec),
                    black_box(&shapes),
                    black_box(&hints),
                    1000.0,
                    0.95,
                    500, // Reduced iterations for benchmarking
                );
            });
        });
    }

    group.finish();
}

/// Benchmark Genetic Algorithm planner on matrix chains
fn bench_ga_matrix_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga_matrix_chain");
    group.sample_size(10); // Reduce sample size for expensive GA

    for n in [3, 4, 5, 6, 8, 10].iter() {
        let (spec, shapes) = create_matrix_chain(*n);
        let hints = PlanHints::default();

        // Use fast preset for benchmarking
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let _ = genetic_algorithm_planner(
                    black_box(&spec),
                    black_box(&shapes),
                    black_box(&hints),
                    50,  // population
                    50,  // generations
                    0.2, // mutation_rate
                    3,   // elitism
                );
            });
        });
    }

    group.finish();
}

/// Benchmark Adaptive planner (auto-selection)
fn bench_adaptive_planner(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_planner");

    for n in [3, 5, 8, 10, 15].iter() {
        let (_, shapes) = create_matrix_chain(*n);
        let hints = PlanHints::default();
        let planner = AdaptivePlanner::new();

        // Need to format spec as string for Planner trait
        let spec_text = format!(
            "{}->{}",
            (0..*n)
                .map(|i| {
                    let indices: Vec<char> = "ijklmnopqrstuvwxyz".chars().collect();
                    format!("{}{}", indices[i], indices[i + 1])
                })
                .collect::<Vec<_>>()
                .join(","),
            {
                let indices: Vec<char> = "ijklmnopqrstuvwxyz".chars().collect();
                format!("{}{}", indices[0], indices[*n])
            }
        );

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let _ =
                    planner.make_plan(black_box(&spec_text), black_box(&shapes), black_box(&hints));
            });
        });
    }

    group.finish();
}

/// Comprehensive planner comparison on star topology
fn bench_planner_comparison_star(c: &mut Criterion) {
    let mut group = c.benchmark_group("planner_comparison_star");
    group.sample_size(10);

    let n = 8; // Fixed size for comparison
    let (spec, shapes) = create_star_topology(n);
    let hints = PlanHints::default();

    // Greedy
    group.bench_function("greedy", |b| {
        b.iter(|| {
            let _ = greedy_planner(black_box(&spec), black_box(&shapes), black_box(&hints));
        });
    });

    // Beam Search (k=5)
    group.bench_function("beam_search_k5", |b| {
        b.iter(|| {
            let _ = beam_search_planner(black_box(&spec), black_box(&shapes), black_box(&hints), 5);
        });
    });

    // DP
    group.bench_function("dp", |b| {
        b.iter(|| {
            let _ = dp_planner(black_box(&spec), black_box(&shapes), black_box(&hints));
        });
    });

    // SA (reduced iterations)
    group.bench_function("sa_fast", |b| {
        b.iter(|| {
            let _ = simulated_annealing_planner(
                black_box(&spec),
                black_box(&shapes),
                black_box(&hints),
                1000.0,
                0.95,
                300,
            );
        });
    });

    // GA (fast preset)
    group.bench_function("ga_fast", |b| {
        b.iter(|| {
            let _ = genetic_algorithm_planner(
                black_box(&spec),
                black_box(&shapes),
                black_box(&hints),
                50,
                50,
                0.2,
                3,
            );
        });
    });

    group.finish();
}

/// Benchmark struct-based planners using Planner trait
fn bench_planner_trait_polymorphism(c: &mut Criterion) {
    let mut group = c.benchmark_group("planner_trait");

    let n = 5;
    let (_, shapes) = create_matrix_chain(n);
    let hints = PlanHints::default();

    // Build spec string
    let spec_str = format!(
        "{}->{}",
        (0..n)
            .map(|i| {
                let indices: Vec<char> = "ijklmnopqrstuvwxyz".chars().collect();
                format!("{}{}", indices[i], indices[i + 1])
            })
            .collect::<Vec<_>>()
            .join(","),
        {
            let indices: Vec<char> = "ijklmnopqrstuvwxyz".chars().collect();
            format!("{}{}", indices[0], indices[n])
        }
    );

    // Benchmark each planner via trait
    let planners: Vec<(&str, Box<dyn Planner>)> = vec![
        ("greedy_struct", Box::new(GreedyPlanner::new())),
        (
            "beam_struct",
            Box::new(BeamSearchPlanner::with_beam_width(5)),
        ),
        ("dp_struct", Box::new(DPPlanner::new())),
        (
            "sa_struct",
            Box::new(SimulatedAnnealingPlanner::with_params(1000.0, 0.95, 300)),
        ),
        ("ga_struct", Box::new(GeneticAlgorithmPlanner::fast())),
        ("adaptive_struct", Box::new(AdaptivePlanner::new())),
    ];

    for (name, planner) in planners {
        group.bench_function(name, |b| {
            b.iter(|| {
                let _ =
                    planner.make_plan(black_box(&spec_str), black_box(&shapes), black_box(&hints));
            });
        });
    }

    group.finish();
}

/// Benchmark GA with different parameter configurations
fn bench_ga_parameter_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga_parameter_sensitivity");
    group.sample_size(10);

    let n = 6;
    let (spec, shapes) = create_matrix_chain(n);
    let hints = PlanHints::default();

    // Test different population sizes
    for pop_size in [25, 50, 100, 150].iter() {
        group.bench_with_input(
            BenchmarkId::new("population", pop_size),
            pop_size,
            |b, &pop| {
                b.iter(|| {
                    let _ = genetic_algorithm_planner(
                        black_box(&spec),
                        black_box(&shapes),
                        black_box(&hints),
                        pop,
                        50, // fixed generations
                        0.2,
                        3,
                    );
                });
            },
        );
    }

    // Test different generation counts
    for gen_count in [25, 50, 100, 150].iter() {
        group.bench_with_input(
            BenchmarkId::new("generations", gen_count),
            gen_count,
            |b, &gen| {
                b.iter(|| {
                    let _ = genetic_algorithm_planner(
                        black_box(&spec),
                        black_box(&shapes),
                        black_box(&hints),
                        50, // fixed population
                        gen,
                        0.2,
                        3,
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_greedy_matrix_chain,
    bench_dp_matrix_chain,
    bench_beam_search_matrix_chain,
    bench_sa_matrix_chain,
    bench_ga_matrix_chain,
    bench_greedy_star,
    bench_dp_star,
    bench_plan_quality,
    bench_greedy_dense_network,
    bench_size_sensitivity,
    bench_adaptive_planner,
    bench_planner_comparison_star,
    bench_planner_trait_polymorphism,
    bench_ga_parameter_sensitivity,
);

criterion_main!(benches);
