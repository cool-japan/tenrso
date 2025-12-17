//! Benchmarks for parallel planning strategies
//!
//! Run with: cargo bench --bench parallel_planners

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use tenrso_planner::{
    beam_search_planner, greedy_planner, EinsumSpec, EnsemblePlanner, ParallelBeamSearchPlanner,
    ParallelGreedyPlanner, PlanHints, Planner,
};

fn bench_ensemble_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_vs_sequential");

    // Test different problem sizes
    let test_cases = vec![
        (
            "3_tensors",
            "ij,jk,kl->il",
            vec![vec![50, 60], vec![60, 70], vec![70, 80]],
        ),
        (
            "4_tensors",
            "ij,jk,kl,lm->im",
            vec![vec![40, 50], vec![50, 60], vec![60, 70], vec![70, 80]],
        ),
        (
            "5_tensors",
            "ij,jk,kl,lm,mn->in",
            vec![
                vec![30, 40],
                vec![40, 50],
                vec![50, 60],
                vec![60, 70],
                vec![70, 80],
            ],
        ),
    ];

    for (name, spec_str, shapes) in test_cases {
        let spec = EinsumSpec::parse(spec_str).unwrap();
        let hints = PlanHints::default();

        // Benchmark sequential (greedy + beam_search)
        group.bench_with_input(
            BenchmarkId::new("sequential", name),
            &(&spec, &shapes, &hints),
            |b, (spec, shapes, hints)| {
                b.iter(|| {
                    let _ = black_box(greedy_planner(spec, shapes, hints).unwrap());
                    let _ = black_box(beam_search_planner(spec, shapes, hints, 5).unwrap());
                });
            },
        );

        // Benchmark ensemble (greedy + beam_search in parallel)
        group.bench_with_input(
            BenchmarkId::new("ensemble", name),
            &(spec_str, &shapes, &hints),
            |b, (spec_str, shapes, hints)| {
                let ensemble = EnsemblePlanner::new(vec!["greedy", "beam_search"]);
                b.iter(|| {
                    let _ = black_box(ensemble.plan(spec_str, shapes, hints).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_ensemble_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_size");

    let spec_str = "ij,jk,kl,lm->im";
    let shapes = vec![vec![40, 50], vec![50, 60], vec![60, 70], vec![70, 80]];
    let hints = PlanHints::default();

    // Test different ensemble sizes
    let ensemble_configs = vec![
        ("2_planners", vec!["greedy", "beam_search"]),
        ("3_planners", vec!["greedy", "beam_search", "dp"]),
        (
            "4_planners",
            vec!["greedy", "beam_search", "dp", "simulated_annealing"],
        ),
    ];

    for (name, planners) in ensemble_configs {
        group.bench_with_input(
            BenchmarkId::new("ensemble", name),
            &(&spec_str, &shapes, &hints),
            |b, (spec_str, shapes, hints)| {
                let ensemble = EnsemblePlanner::new(planners.clone());
                b.iter(|| {
                    let _ = black_box(ensemble.plan(spec_str, shapes, hints).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_beam_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_beam_search");

    let spec_str = "ij,jk,kl,lm->im";
    let shapes = vec![vec![40, 50], vec![50, 60], vec![60, 70], vec![70, 80]];
    let hints = PlanHints::default();

    // Test different beam widths
    let beam_widths = vec![3, 5, 8, 10];

    for width in beam_widths {
        let spec = EinsumSpec::parse(spec_str).unwrap();

        // Sequential beam search
        group.bench_with_input(
            BenchmarkId::new("sequential", width),
            &(&spec, &shapes, &hints),
            |b, (spec, shapes, hints)| {
                b.iter(|| {
                    let _ = black_box(beam_search_planner(spec, shapes, hints, width).unwrap());
                });
            },
        );

        // Parallel beam search
        group.bench_with_input(
            BenchmarkId::new("parallel", width),
            &(spec_str, &shapes, &hints),
            |b, (spec_str, shapes, hints)| {
                let planner = ParallelBeamSearchPlanner::new(width);
                b.iter(|| {
                    let _ = black_box(planner.make_plan(spec_str, shapes, hints).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_greedy");

    // Test with different numbers of tensors
    let test_cases = vec![
        (
            "10_tensors",
            "a0,a1,a2,a3,a4,a5,a6,a7,a8,a9->a0a9",
            vec![vec![10, 10]; 10],
        ),
        (
            "20_tensors",
            "b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19->b0b19",
            vec![vec![8, 8]; 20],
        ),
    ];

    for (name, spec_str, shapes) in test_cases {
        let spec = EinsumSpec::parse(spec_str).unwrap();
        let hints = PlanHints::default();

        // Sequential greedy
        group.bench_with_input(
            BenchmarkId::new("sequential", name),
            &(&spec, &shapes, &hints),
            |b, (spec, shapes, hints)| {
                b.iter(|| {
                    let _ = black_box(greedy_planner(spec, shapes, hints).unwrap());
                });
            },
        );

        // Parallel greedy
        group.bench_with_input(
            BenchmarkId::new("parallel", name),
            &(spec_str, &shapes, &hints),
            |b, (spec_str, shapes, hints)| {
                let planner = ParallelGreedyPlanner::new();
                b.iter(|| {
                    let _ = black_box(planner.make_plan(spec_str, shapes, hints).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_ensemble_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_metrics");

    let spec_str = "ij,jk,kl->il";
    let shapes = vec![vec![50, 60], vec![60, 70], vec![70, 80]];
    let hints = PlanHints::default();

    let metrics = vec!["flops", "memory", "combined"];

    for metric in metrics {
        group.bench_with_input(
            BenchmarkId::new("metric", metric),
            &(&spec_str, &shapes, &hints),
            |b, (spec_str, shapes, hints)| {
                let ensemble =
                    EnsemblePlanner::new(vec!["greedy", "beam_search", "dp"]).with_metric(metric);
                b.iter(|| {
                    let _ = black_box(ensemble.plan(spec_str, shapes, hints).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_ensemble_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_quality");
    group.sample_size(20); // Fewer samples for expensive operations

    let spec_str = "ij,jk,kl,lm->im";
    let shapes = vec![vec![40, 50], vec![50, 60], vec![60, 70], vec![70, 80]];
    let hints = PlanHints::default();

    // Compare quality (FLOPs) vs time tradeoff
    let configs = vec![
        ("fast", vec!["greedy"]),
        ("balanced", vec!["greedy", "beam_search"]),
        ("quality", vec!["greedy", "beam_search", "dp"]),
        (
            "best",
            vec!["greedy", "beam_search", "dp", "simulated_annealing"],
        ),
    ];

    for (name, planners) in configs {
        group.bench_with_input(
            BenchmarkId::new("config", name),
            &(&spec_str, &shapes, &hints),
            |b, (spec_str, shapes, hints)| {
                let ensemble = EnsemblePlanner::new(planners.clone());
                b.iter(|| {
                    let plan = black_box(ensemble.plan(spec_str, shapes, hints).unwrap());
                    // Return FLOPs for quality measurement
                    plan.estimated_flops
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_ensemble_vs_sequential,
    bench_ensemble_size,
    bench_parallel_beam_search,
    bench_parallel_greedy,
    bench_ensemble_metrics,
    bench_ensemble_quality,
);
criterion_main!(benches);
