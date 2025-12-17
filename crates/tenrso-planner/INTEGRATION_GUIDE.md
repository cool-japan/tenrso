# TenRSo-Planner Integration Guide

> **Version:** 0.1.0-alpha.2
> **Last Updated:** 2025-12-09

This guide demonstrates how to integrate tenrso-planner's advanced features into your tensor computation workflow.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Planning Algorithms](#planning-algorithms)
3. [Parallel Ensemble Planning](#parallel-ensemble-planning)
4. [ML-Based Cost Calibration](#ml-based-cost-calibration)
5. [Plan Caching](#plan-caching)
6. [Hardware Simulation](#hardware-simulation)
7. [Quality Tracking](#quality-tracking)
8. [Production Workflow](#production-workflow)
9. [Best Practices](#best-practices)

---

## Quick Start

```rust
use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};

// Parse Einstein summation notation
let spec = EinsumSpec::parse("ij,jk->ik")?;

// Define tensor shapes
let shapes = vec![vec![100, 200], vec![200, 300]];

// Create a plan
let hints = PlanHints::default();
let plan = greedy_planner(&spec, &shapes, &hints)?;

// Inspect results
println!("FLOPs: {:.2e}", plan.estimated_flops);
println!("Memory: {} bytes", plan.estimated_memory);
println!("Steps: {}", plan.nodes.len());
```

---

## Planning Algorithms

TenRSo-Planner provides 6 production-grade planning algorithms:

### 1. Greedy Planner (Fast, Good Quality)

```rust
use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};

let spec = EinsumSpec::parse("ij,jk,kl->il")?;
let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40]];
let hints = PlanHints::default();

let plan = greedy_planner(&spec, &shapes, &hints)?;
// O(n³) time, good for most cases
```

**Use when:** You need fast planning (< 1ms for 10 tensors)

### 2. Dynamic Programming (Optimal, Expensive)

```rust
use tenrso_planner::{dp_planner, EinsumSpec, PlanHints};

let spec = EinsumSpec::parse("ij,jk,kl->il")?;
let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40]];
let hints = PlanHints::default();

let plan = dp_planner(&spec, &shapes, &hints)?;
// O(3^n) time, guaranteed optimal
```

**Use when:** You need provably optimal plans (≤ 20 tensors)

### 3. Beam Search (Better Quality, Moderate Speed)

```rust
use tenrso_planner::{beam_search_planner, EinsumSpec, PlanHints};

let spec = EinsumSpec::parse("ij,jk,kl->il")?;
let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40]];
let hints = PlanHints::default();

let plan = beam_search_planner(&spec, &shapes, &hints, 5)?; // beam width = 5
// O(n³ * k) time, better than greedy
```

**Use when:** Medium networks (8-20 tensors), want better than greedy

### 4. Simulated Annealing (Stochastic, Escapes Local Minima)

```rust
use tenrso_planner::{SimulatedAnnealingPlanner, Planner, PlanHints};

let planner = SimulatedAnnealingPlanner::with_params(1000.0, 0.95, 1000);
let plan = planner.make_plan("ij,jk,kl->il", &shapes, &hints)?;
// Stochastic search, configurable iterations
```

**Use when:** Large networks, quality more important than planning speed

### 5. Genetic Algorithm (Population-Based, High Quality)

```rust
use tenrso_planner::{GeneticAlgorithmPlanner, Planner, PlanHints};

let planner = GeneticAlgorithmPlanner::fast(); // or ::high_quality()
let plan = planner.make_plan("ij,jk,kl->il", &shapes, &hints)?;
// Evolutionary search, best for complex topologies
```

**Use when:** Very large networks (> 20), need best quality with time budget

### 6. Adaptive Planner (⭐ Recommended)

```rust
use tenrso_planner::{AdaptivePlanner, Planner, PlanHints};

let planner = AdaptivePlanner::default();
let plan = planner.make_plan("ij,jk,kl->il", &shapes, &hints)?;
// Automatically selects best algorithm based on problem size
```

**Use when:** You want optimal results without manual algorithm selection

---

## Parallel Ensemble Planning

Run multiple planners concurrently and automatically select the best result:

```rust
use tenrso_planner::{EnsemblePlanner, PlanHints};

// Create ensemble with multiple planners
let ensemble = EnsemblePlanner::new(vec!["greedy", "beam_search", "dp"]);

// Run all planners in parallel
let plan = ensemble.plan("ij,jk,kl->il", &shapes, &hints)?;

// Automatically selects best plan by FLOPs
println!("Best plan: {:.2e} FLOPs", plan.estimated_flops);
```

### Configuring Selection Metric

```rust
// Select by memory usage
let ensemble = EnsemblePlanner::new(vec!["greedy", "dp"])
    .with_metric("memory");

// Select by combined FLOPs + memory
let ensemble = EnsemblePlanner::new(vec!["greedy", "dp"])
    .with_metric("combined");
```

### Performance

- **Speedup:** Near-linear with number of cores
- **Example:** 3 planners → 2.8x faster on 4-core system
- **Overhead:** ~1-2ms per planner for thread spawning

### Best Configurations

```rust
// Fast (2 planners): greedy + beam_search
let fast = EnsemblePlanner::new(vec!["greedy", "beam_search"]);

// Balanced (3 planners): add DP
let balanced = EnsemblePlanner::new(vec!["greedy", "beam_search", "dp"]);

// Best Quality (5 planners): add stochastic algorithms
let best = EnsemblePlanner::new(vec![
    "greedy", "beam_search", "dp",
    "simulated_annealing", "genetic_algorithm"
]);
```

---

## ML-Based Cost Calibration

Learn from execution history to improve cost predictions:

```rust
use tenrso_planner::{MLCostModel, ExecutionHistory, ExecutionRecord};
use std::time::SystemTime;

// Step 1: Create execution history
let mut history = ExecutionHistory::with_max_size(100);

// Step 2: Record actual execution results
history.record(ExecutionRecord {
    id: "matmul_1000x2000x3000".to_string(),
    predicted_flops: 12_000_000_000.0,  // What planner predicted
    actual_flops: 12_500_000_000.0,      // What actually happened
    predicted_time_ms: 100.0,
    actual_time_ms: 105.0,
    predicted_memory: 24_000_000,
    actual_memory: 25_000_000,
    timestamp: SystemTime::now(),
    planner: "greedy".to_string(),
});

// Step 3: Train ML cost model (requires ≥ 3 records)
let mut ml_model = MLCostModel::new();
ml_model.train(&history);

// Step 4: Use calibrated predictions
let predicted_flops = 10_000_000_000.0;
let calibrated_flops = ml_model.calibrate_flops(predicted_flops);
let calibrated_time = ml_model.calibrate_time(100.0, calibrated_flops);

println!("Original:   {:.2e} FLOPs", predicted_flops);
println!("Calibrated: {:.2e} FLOPs", calibrated_flops);
println!("Model R²:   {:.4}", ml_model.flops_r_squared());
```

### Per-Planner Calibration

Different planners may have different biases:

```rust
// Calibrate for specific planner
let cal_greedy = ml_model.calibrate_flops_for_planner(1e9, "greedy");
let cal_dp = ml_model.calibrate_flops_for_planner(1e9, "dp");

// Falls back to general model if planner-specific not available
let cal_unknown = ml_model.calibrate_flops_for_planner(1e9, "unknown");
```

---

## Plan Caching

Cache plans with LRU/LFU/ARC eviction policies:

```rust
use tenrso_planner::PlanCache;

// Create cache with LRU eviction (default)
let mut cache = PlanCache::new_lru(100);

// Or use LFU (frequency-based)
let mut cache = PlanCache::new_lfu(100);

// Or use ARC (adaptive, balances recency and frequency)
let mut cache = PlanCache::new_arc(100);

// Cache plans
let key = "ij,jk->ik:100x200:200x300";
cache.put(key.to_string(), plan.clone());

// Retrieve cached plan
if let Some(cached_plan) = cache.get(key) {
    println!("Cache hit! {:.2e} FLOPs", cached_plan.estimated_flops);
} else {
    // Cache miss - compute plan
    let plan = greedy_planner(&spec, &shapes, &hints)?;
    cache.put(key.to_string(), plan.clone());
}

// Check cache statistics
let stats = cache.stats();
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
```

---

## Hardware Simulation

Simulate plan execution on different hardware:

```rust
use tenrso_planner::{HardwareModel, PlanSimulator};

// Create simulator with hardware model
let cpu_low = HardwareModel::cpu_low_end();
let cpu_high = HardwareModel::cpu_high_end();
let gpu_v100 = HardwareModel::nvidia_volta();
let gpu_a100 = HardwareModel::nvidia_ampere();

let simulator_cpu = PlanSimulator::new(cpu_low);
let simulator_gpu = PlanSimulator::new(gpu_a100);

// Simulate plan execution
let sim_cpu = simulator_cpu.simulate(&plan)?;
let sim_gpu = simulator_gpu.simulate(&plan)?;

println!("CPU: {:.2} ms, {:.2} GB/s",
    sim_cpu.total_time_ms, sim_cpu.effective_bandwidth_gbps);
println!("GPU: {:.2} ms, {:.2} GB/s",
    sim_gpu.total_time_ms, sim_gpu.effective_bandwidth_gbps);

// Compare hardware for best choice
if sim_gpu.total_time_ms < sim_cpu.total_time_ms {
    println!("GPU is {:.2}x faster", sim_cpu.total_time_ms / sim_gpu.total_time_ms);
}
```

### Available Hardware Models

- **CPUs:** `cpu_low_end()`, `cpu_high_end()`
- **NVIDIA:** `nvidia_pascal()`, `nvidia_volta()`, `nvidia_turing()`, `nvidia_ampere()`, `nvidia_hopper()`
- **AMD:** `amd_cdna2()`

---

## Quality Tracking

Track plan quality over time:

```rust
use tenrso_planner::{ExecutionHistory, PlanQualityMetrics};

let history = ExecutionHistory::with_max_size(1000);

// Record executions...
// (see ML-Based Cost Calibration section)

// Compute quality metrics
let metrics = history.compute_metrics();

println!("Executions: {}", metrics.num_executions);
println!("Avg FLOPs error: {:.1}%", metrics.avg_flops_error * 100.0);
println!("Accuracy (10%): {:.1}%", metrics.accuracy_10pct * 100.0);

// Per-planner metrics
for (planner, planner_metrics) in &metrics.per_planner {
    println!("{}: {:.1}% error", planner, planner_metrics.avg_flops_error * 100.0);
}

// Find best planner
if let Some(best) = history.best_planner() {
    println!("Best planner: {}", best);
}
```

---

## Production Workflow

Recommended workflow integrating all features:

```rust
use tenrso_planner::*;
use std::sync::{Arc, Mutex};

// 1. Initialize components
let cache = Arc::new(Mutex::new(PlanCache::new_arc(1000)));
let history = Arc::new(Mutex::new(ExecutionHistory::with_max_size(10000)));
let ml_model = Arc::new(Mutex::new(MLCostModel::new()));

// 2. Plan with caching
fn plan_with_cache(
    spec: &str,
    shapes: &[Vec<usize>],
    cache: &Arc<Mutex<PlanCache>>,
) -> anyhow::Result<Plan> {
    let key = format!("{}:{:?}", spec, shapes);

    // Try cache first
    let mut cache_lock = cache.lock().unwrap();
    if let Some(plan) = cache_lock.get(&key) {
        return Ok(plan.clone());
    }
    drop(cache_lock);

    // Cache miss - use ensemble planner
    let ensemble = EnsemblePlanner::new(vec!["greedy", "beam_search", "dp"]);
    let plan = ensemble.plan(spec, shapes, &PlanHints::default())?;

    // Cache for future
    let mut cache_lock = cache.lock().unwrap();
    cache_lock.put(key, plan.clone());

    Ok(plan)
}

// 3. Execute and record results
fn execute_and_record(
    plan: &Plan,
    actual_flops: f64,
    actual_time_ms: f64,
    actual_memory: usize,
    history: &Arc<Mutex<ExecutionHistory>>,
) {
    let record = ExecutionRecord {
        id: "execution_id".to_string(),
        predicted_flops: plan.estimated_flops,
        actual_flops,
        predicted_time_ms: 100.0, // from simulation
        actual_time_ms,
        predicted_memory: plan.estimated_memory,
        actual_memory,
        timestamp: std::time::SystemTime::now(),
        planner: "ensemble".to_string(),
    };

    let mut history_lock = history.lock().unwrap();
    history_lock.record(record);
}

// 4. Periodically retrain ML model
fn retrain_ml_model(
    history: &Arc<Mutex<ExecutionHistory>>,
    ml_model: &Arc<Mutex<MLCostModel>>,
) {
    let history_lock = history.lock().unwrap();
    if history_lock.len() >= 10 {
        let mut model_lock = ml_model.lock().unwrap();
        model_lock.train(&history_lock);
        println!("ML model retrained with {} samples", history_lock.len());
        println!("FLOPs R²: {:.4}", model_lock.flops_r_squared());
    }
}

// 5. Use in production
let spec = "ij,jk,kl->il";
let shapes = vec![vec![100, 200], vec![200, 300], vec![300, 400]];

let plan = plan_with_cache(spec, &shapes, &cache)?;

// ... execute plan ...

execute_and_record(&plan, 1.2e9, 105.0, 25_000_000, &history);

// Retrain periodically (e.g., every 100 executions)
retrain_ml_model(&history, &ml_model);
```

---

## Best Practices

### 1. Algorithm Selection

- **Interactive/Development:** Use `AdaptivePlanner` for automatic selection
- **Production:** Use `EnsemblePlanner` with 2-3 fast planners for best quality within time budget
- **Batch/Offline:** Use `dp_planner` or full ensemble for optimal results

### 2. Caching Strategy

- Use **ARC** for general workloads (adapts to access patterns)
- Use **LFU** for workloads with hot patterns (repeated tensors)
- Use **LRU** for sequential/streaming workloads
- Set cache size to ~1000-10000 entries depending on memory constraints

### 3. ML Calibration

- Collect **≥ 100 execution records** before heavy reliance on ML model
- Check **R² scores** (aim for > 0.9 for good models)
- Retrain **periodically** (e.g., every 100-1000 executions)
- Use **per-planner calibration** for multi-algorithm workflows

### 4. Parallel Planning

- Use **2-3 planners** for fast results (greedy + beam_search)
- Use **5-6 planners** for best quality (includes SA, GA)
- Consider **thread overhead** for small problems (< 5 tensors)
- Set appropriate **beam widths** (3-10) for beam search

### 5. Quality Monitoring

- Track **accuracy percentages** (aim for > 80% within 10% tolerance)
- Monitor **per-planner metrics** to identify systematic biases
- Use **execution history** to detect performance regressions
- Set up **alerts** for prediction errors > 50%

---

## Examples

See the `examples/` directory for detailed usage:

- `ml_calibration.rs` - ML-based cost calibration walkthrough
- `parallel_ensemble.rs` - Parallel planning demonstration
- `basic_matmul.rs` - Simple matrix multiplication
- `matrix_chain.rs` - Greedy vs DP comparison
- `planning_hints.rs` - Advanced hint usage
- `genetic_algorithm.rs` - GA planner showcase
- `comprehensive_comparison.rs` - All planners side-by-side
- `plan_visualization.rs` - Visualization and debugging
- `advanced_features.rs` - Caching, simulation, profiling

Run with: `cargo run --example <name>`

---

## Benchmarks

Compare planning algorithms:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench --bench planner_benchmarks
cargo bench --bench comprehensive_comparison
cargo bench --bench parallel_planners
```

---

## Further Reading

- **API Documentation:** `cargo doc --open`
- **Source Code:** `src/` directory with extensive inline documentation
- **TODO.md:** Roadmap and implementation details
- **CLAUDE.md:** Integration guide for maintainers

---

## Support

For questions, issues, or contributions:

- **GitHub:** https://github.com/cool-japan/tenrso
- **Issues:** https://github.com/cool-japan/tenrso/issues

---

**Last Updated:** 2025-12-09
**Version:** 0.1.0-alpha.2
**Status:** Production-Ready + ML + Parallel
