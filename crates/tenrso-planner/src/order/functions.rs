//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::api::{ContractionSpec, Plan, PlanHints, PlanNode, ReprHint};
use crate::cost::{estimate_flops, TensorStats};
use crate::parser::EinsumSpec;
use anyhow::{anyhow, Result};
use scirs2_core::random::{Rng, SeedableRng, StdRng};
use std::collections::HashMap;

use super::types::IntermediateTensor;
/// Find the contraction spec for two intermediate tensors
fn compute_pairwise_spec(a: &IntermediateTensor, b: &IntermediateTensor) -> Result<EinsumSpec> {
    let indices_a: std::collections::HashSet<char> = a.indices.chars().collect();
    let indices_b: std::collections::HashSet<char> = b.indices.chars().collect();
    let shared: std::collections::HashSet<char> =
        indices_a.intersection(&indices_b).copied().collect();
    let mut output_indices = Vec::new();
    for c in a.indices.chars() {
        if !shared.contains(&c) && !output_indices.contains(&c) {
            output_indices.push(c);
        }
    }
    for c in b.indices.chars() {
        if !shared.contains(&c) && !output_indices.contains(&c) {
            output_indices.push(c);
        }
    }
    output_indices.sort();
    let output: String = output_indices.into_iter().collect();
    let spec_str = format!("{},{}->{}", a.indices, b.indices, output);
    EinsumSpec::parse(&spec_str)
}
/// Compute the output shape for a pairwise contraction
fn compute_pairwise_output_shape(
    spec: &EinsumSpec,
    a: &IntermediateTensor,
    b: &IntermediateTensor,
) -> Result<Vec<usize>> {
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (c, &size) in a.indices.chars().zip(a.shape.iter()) {
        dim_map.insert(c, size);
    }
    for (c, &size) in b.indices.chars().zip(b.shape.iter()) {
        if let Some(&prev_size) = dim_map.get(&c) {
            if prev_size != size {
                return Err(anyhow!(
                    "Dimension mismatch for index '{}': {} vs {}",
                    c,
                    prev_size,
                    size
                ));
            }
        } else {
            dim_map.insert(c, size);
        }
    }
    let output_shape: Vec<usize> = spec
        .output
        .chars()
        .map(|c| *dim_map.get(&c).unwrap_or(&1))
        .collect();
    Ok(output_shape)
}
/// Greedy contraction order planner
///
/// Repeatedly contracts the pair of tensors with minimum cost until only one remains.
///
/// # Algorithm
///
/// 1. Start with all input tensors
/// 2. Find all possible pairwise contractions
/// 3. Estimate cost for each pair
/// 4. Contract the pair with minimum cost
/// 5. Repeat until only one tensor remains
///
/// # Complexity
///
/// O(n^3) where n is the number of inputs (n-1 steps, each checking O(n^2) pairs)
pub fn greedy_planner(
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    _hints: &PlanHints,
) -> Result<Plan> {
    if spec.num_inputs() != shapes.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match shapes ({})",
            spec.num_inputs(),
            shapes.len()
        ));
    }
    let mut intermediates: Vec<IntermediateTensor> = spec
        .inputs
        .iter()
        .zip(shapes.iter())
        .enumerate()
        .map(|(i, (indices, shape))| {
            IntermediateTensor::from_input(indices.clone(), shape.clone(), i)
        })
        .collect();
    let mut plan = Plan::new();
    let mut total_flops = 0.0;
    let mut peak_memory = 0;
    let mut contraction_order = Vec::new();
    while intermediates.len() > 1 {
        let mut best_cost = f64::INFINITY;
        let mut best_pair = (0, 1);
        let mut best_spec = None;
        let mut best_shape = None;
        for i in 0..intermediates.len() {
            for j in (i + 1)..intermediates.len() {
                let a = &intermediates[i];
                let b = &intermediates[j];
                if let Ok(pairwise_spec) = compute_pairwise_spec(a, b) {
                    let stats = vec![a.stats.clone(), b.stats.clone()];
                    if let Ok(cost) = estimate_flops(&pairwise_spec, &stats) {
                        if cost < best_cost {
                            best_cost = cost;
                            best_pair = (i, j);
                            if let Ok(output_shape) =
                                compute_pairwise_output_shape(&pairwise_spec, a, b)
                            {
                                best_spec = Some(pairwise_spec);
                                best_shape = Some(output_shape);
                            }
                        }
                    }
                }
            }
        }
        if best_spec.is_none() || best_shape.is_none() {
            return Err(anyhow!("No valid contraction found"));
        }
        let pairwise_spec = best_spec.unwrap();
        let output_shape = best_shape.unwrap();
        let (i, j) = best_pair;
        let a = &intermediates[i];
        let b = &intermediates[j];
        let input_indices = vec![
            a.original_idx.unwrap_or(1000 + i),
            b.original_idx.unwrap_or(1000 + j),
        ];
        let node = PlanNode {
            inputs: input_indices.clone(),
            output_spec: ContractionSpec::new(
                vec![a.indices.clone(), b.indices.clone()],
                pairwise_spec.output.clone(),
            ),
            cost: best_cost,
            memory: output_shape.iter().product::<usize>() * 8,
            repr: ReprHint::Auto,
        };
        plan.nodes.push(node);
        total_flops += best_cost;
        peak_memory = peak_memory.max(output_shape.iter().product::<usize>() * 8);
        contraction_order.push(best_pair);
        let output_stats = TensorStats::dense(output_shape.clone());
        let intermediate = IntermediateTensor::from_contraction(
            pairwise_spec.output.clone(),
            output_shape,
            output_stats,
        );
        let (remove_first, remove_second) = if i > j { (i, j) } else { (j, i) };
        intermediates.remove(remove_first);
        intermediates.remove(remove_second);
        intermediates.push(intermediate);
    }
    plan.estimated_flops = total_flops;
    plan.estimated_memory = peak_memory;
    plan.order = contraction_order;
    Ok(plan)
}
/// Dynamic programming planner for optimal contraction order
///
/// Uses bitmask DP to find the globally optimal contraction sequence.
/// Best for small tensor networks (< 20 tensors).
///
/// # Algorithm
///
/// For each subset S of tensors (represented as bitmask):
/// 1. Try all ways to partition S into two non-empty subsets S1 and S2
/// 2. Cost(S) = min over partitions of: Cost(S1) + Cost(S2) + cost(contract(S1, S2))
/// 3. Use memoization to avoid recomputation
/// 4. Backtrack to reconstruct optimal plan
///
/// # Complexity
///
/// O(3^n) time, O(2^n) space where n is number of inputs
/// Practical limit: n ≤ 20 tensors
///
/// # Fallback
///
/// Falls back to greedy planner if n > 20 (too expensive)
pub fn dp_planner(spec: &EinsumSpec, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
    let n = spec.num_inputs();
    if n > 20 {
        log::warn!(
            "DP planner: too many inputs ({}), falling back to greedy",
            n
        );
        return greedy_planner(spec, shapes, hints);
    }
    if n != shapes.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match shapes ({})",
            n,
            shapes.len()
        ));
    }
    if n == 1 {
        return Ok(Plan::new());
    }
    if n == 2 {
        return greedy_planner(spec, shapes, hints);
    }
    let intermediates: Vec<IntermediateTensor> = spec
        .inputs
        .iter()
        .zip(shapes.iter())
        .enumerate()
        .map(|(i, (indices, shape))| {
            IntermediateTensor::from_input(indices.clone(), shape.clone(), i)
        })
        .collect();
    let num_states = 1 << n;
    let mut dp: Vec<Option<(f64, usize)>> = vec![None; num_states];
    let mut cached_contractions: HashMap<usize, IntermediateTensor> = HashMap::new();
    for (i, intermediate) in intermediates.iter().enumerate().take(n) {
        let mask = 1 << i;
        dp[mask] = Some((0.0, 0));
        cached_contractions.insert(mask, intermediate.clone());
    }
    for mask in 1..num_states {
        let popcount = mask.count_ones() as usize;
        if popcount <= 1 {
            continue;
        }
        let mut best_cost = f64::INFINITY;
        let mut best_partition = 0;
        let mut submask = mask;
        loop {
            if submask != 0 && submask != mask {
                let complement = mask ^ submask;
                if let (Some((cost1, _)), Some((cost2, _))) = (dp[submask], dp[complement]) {
                    if let (Some(tensor1), Some(tensor2)) = (
                        cached_contractions.get(&submask),
                        cached_contractions.get(&complement),
                    ) {
                        if let Ok(pairwise_spec) = compute_pairwise_spec(tensor1, tensor2) {
                            let stats = vec![tensor1.stats.clone(), tensor2.stats.clone()];
                            if let Ok(contraction_cost) = estimate_flops(&pairwise_spec, &stats) {
                                let total_cost = cost1 + cost2 + contraction_cost;
                                if total_cost < best_cost {
                                    best_cost = total_cost;
                                    best_partition = submask;
                                }
                            }
                        }
                    }
                }
            }
            if submask == 0 {
                break;
            }
            submask = (submask - 1) & mask;
        }
        if best_partition == 0 {
            return Err(anyhow!(
                "DP planner: no valid partition found for mask {}",
                mask
            ));
        }
        dp[mask] = Some((best_cost, best_partition));
        let submask = best_partition;
        let complement = mask ^ submask;
        if let (Some(tensor1), Some(tensor2)) = (
            cached_contractions.get(&submask),
            cached_contractions.get(&complement),
        ) {
            if let Ok(pairwise_spec) = compute_pairwise_spec(tensor1, tensor2) {
                if let Ok(output_shape) =
                    compute_pairwise_output_shape(&pairwise_spec, tensor1, tensor2)
                {
                    let output_stats = TensorStats::dense(output_shape.clone());
                    let intermediate = IntermediateTensor::from_contraction(
                        pairwise_spec.output.clone(),
                        output_shape,
                        output_stats,
                    );
                    cached_contractions.insert(mask, intermediate);
                }
            }
        }
    }
    let full_mask = (1 << n) - 1;
    let (total_cost, _) = dp[full_mask].ok_or_else(|| anyhow!("DP planner: no solution found"))?;
    let mut plan = Plan::new();
    plan.estimated_flops = total_cost;
    fn reconstruct_plan(
        mask: usize,
        dp: &[Option<(f64, usize)>],
        cached_contractions: &HashMap<usize, IntermediateTensor>,
        plan: &mut Plan,
        peak_memory: &mut usize,
    ) -> Result<()> {
        if mask.count_ones() == 1 {
            return Ok(());
        }
        let (_, best_partition) =
            dp[mask].ok_or_else(|| anyhow!("Missing DP state for mask {}", mask))?;
        let submask = best_partition;
        let complement = mask ^ submask;
        reconstruct_plan(submask, dp, cached_contractions, plan, peak_memory)?;
        reconstruct_plan(complement, dp, cached_contractions, plan, peak_memory)?;
        let tensor1 = cached_contractions
            .get(&submask)
            .ok_or_else(|| anyhow!("Missing cached tensor for submask {}", submask))?;
        let tensor2 = cached_contractions
            .get(&complement)
            .ok_or_else(|| anyhow!("Missing cached tensor for complement {}", complement))?;
        let pairwise_spec = compute_pairwise_spec(tensor1, tensor2)?;
        let output_shape = compute_pairwise_output_shape(&pairwise_spec, tensor1, tensor2)?;
        let stats = vec![tensor1.stats.clone(), tensor2.stats.clone()];
        let cost = estimate_flops(&pairwise_spec, &stats)?;
        let memory = output_shape.iter().product::<usize>() * 8;
        *peak_memory = (*peak_memory).max(memory);
        let mut input_indices = Vec::new();
        for i in 0..64 {
            if (submask & (1 << i)) != 0 {
                input_indices.push(i);
            }
        }
        for i in 0..64 {
            if (complement & (1 << i)) != 0 {
                input_indices.push(i);
            }
        }
        let node = PlanNode {
            inputs: input_indices,
            output_spec: ContractionSpec::new(
                vec![tensor1.indices.clone(), tensor2.indices.clone()],
                pairwise_spec.output.clone(),
            ),
            cost,
            memory,
            repr: ReprHint::Auto,
        };
        plan.nodes.push(node);
        Ok(())
    }
    let mut peak_memory = 0;
    reconstruct_plan(
        full_mask,
        &dp,
        &cached_contractions,
        &mut plan,
        &mut peak_memory,
    )?;
    plan.estimated_memory = peak_memory;
    Ok(plan)
}
/// Beam search planner for contraction order
///
/// Maintains k best partial plans at each step, exploring more options than greedy
/// while remaining tractable for larger networks.
///
/// # Algorithm
///
/// 1. Start with k=beam_width initial contractions (best pairs)
/// 2. At each step, expand each candidate by trying all possible next contractions
/// 3. Keep only the k best candidates based on total cost
/// 4. Repeat until all candidates have single tensor
/// 5. Return the best complete plan
///
/// # Complexity
///
/// O(n² * beam_width * n) = O(n³ * beam_width) where n is number of inputs
/// Practical limit: beam_width = 3-10, n ≤ 100 tensors
///
/// # Trade-offs
///
/// - beam_width = 1: equivalent to greedy
/// - beam_width = ∞: explores all possibilities (exponential)
/// - beam_width = 3-10: good balance of quality and speed
pub fn beam_search_planner(
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    _hints: &PlanHints,
    beam_width: usize,
) -> Result<Plan> {
    if spec.num_inputs() != shapes.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match shapes ({})",
            spec.num_inputs(),
            shapes.len()
        ));
    }
    let n = spec.num_inputs();
    if n == 0 {
        return Err(anyhow!("No inputs to contract"));
    }
    if n == 1 {
        return Ok(Plan {
            nodes: vec![],
            order: vec![],
            estimated_flops: 0.0,
            estimated_memory: shapes[0].iter().product::<usize>() * 8,
        });
    }
    let initial_intermediates: Vec<IntermediateTensor> = spec
        .inputs
        .iter()
        .zip(shapes.iter())
        .enumerate()
        .map(|(i, (indices, shape))| {
            IntermediateTensor::from_input(indices.clone(), shape.clone(), i)
        })
        .collect();
    #[derive(Clone)]
    struct Candidate {
        intermediates: Vec<IntermediateTensor>,
        plan: Plan,
        total_flops: f64,
        peak_memory: usize,
    }
    let mut beam: Vec<Candidate> = vec![Candidate {
        intermediates: initial_intermediates,
        plan: Plan::new(),
        total_flops: 0.0,
        peak_memory: 0,
    }];
    while beam.iter().any(|c| c.intermediates.len() > 1) {
        let mut next_beam: Vec<Candidate> = Vec::new();
        for candidate in &beam {
            if candidate.intermediates.len() <= 1 {
                next_beam.push(candidate.clone());
                continue;
            }
            for i in 0..candidate.intermediates.len() {
                for j in (i + 1)..candidate.intermediates.len() {
                    let a = &candidate.intermediates[i];
                    let b = &candidate.intermediates[j];
                    if let Ok(pairwise_spec) = compute_pairwise_spec(a, b) {
                        let stats = vec![a.stats.clone(), b.stats.clone()];
                        if let Ok(cost) = estimate_flops(&pairwise_spec, &stats) {
                            if let Ok(output_shape) =
                                compute_pairwise_output_shape(&pairwise_spec, a, b)
                            {
                                let mut new_candidate = candidate.clone();
                                let input_indices = vec![
                                    a.original_idx.unwrap_or(1000 + i),
                                    b.original_idx.unwrap_or(1000 + j),
                                ];
                                let node = PlanNode {
                                    inputs: input_indices,
                                    output_spec: ContractionSpec::new(
                                        vec![a.indices.clone(), b.indices.clone()],
                                        pairwise_spec.output.clone(),
                                    ),
                                    cost,
                                    memory: output_shape.iter().product::<usize>() * 8,
                                    repr: ReprHint::Auto,
                                };
                                new_candidate.plan.nodes.push(node);
                                new_candidate.total_flops += cost;
                                new_candidate.peak_memory = new_candidate
                                    .peak_memory
                                    .max(output_shape.iter().product::<usize>() * 8);
                                new_candidate.plan.order.push((i, j));
                                let output_stats = TensorStats::dense(output_shape.clone());
                                let intermediate = IntermediateTensor::from_contraction(
                                    pairwise_spec.output.clone(),
                                    output_shape,
                                    output_stats,
                                );
                                let (remove_first, remove_second) =
                                    if i > j { (i, j) } else { (j, i) };
                                new_candidate.intermediates.remove(remove_first);
                                new_candidate.intermediates.remove(remove_second);
                                new_candidate.intermediates.push(intermediate);
                                next_beam.push(new_candidate);
                            }
                        }
                    }
                }
            }
        }
        if next_beam.is_empty() {
            return Err(anyhow!("No valid contractions found in beam search"));
        }
        next_beam.sort_by(|a, b| a.total_flops.partial_cmp(&b.total_flops).unwrap());
        next_beam.truncate(beam_width);
        beam = next_beam;
    }
    let best = beam
        .into_iter()
        .min_by(|a, b| a.total_flops.partial_cmp(&b.total_flops).unwrap())
        .ok_or_else(|| anyhow!("No complete plan found"))?;
    let mut final_plan = best.plan;
    final_plan.estimated_flops = best.total_flops;
    final_plan.estimated_memory = best.peak_memory;
    Ok(final_plan)
}
use std::f64::consts::E;
/// Simulated annealing planner for contraction order
///
/// Uses stochastic search with temperature-based acceptance to escape local minima.
/// Good for large tensor networks where DP is infeasible and greedy may be suboptimal.
///
/// Uses SciRS2-Core's professional-grade RNG with fixed seed (12345) for reproducibility.
///
/// # Algorithm
///
/// 1. Start with greedy plan as initial solution
/// 2. Generate neighbor by swapping two random contractions (using SciRS2 RNG)
/// 3. Accept if better, or with probability exp(-ΔE/T) if worse
/// 4. Decrease temperature gradually
/// 5. Repeat for max_iterations
/// 6. Return best plan found
///
/// # Complexity
///
/// O(max_iterations * n) where n is number of inputs
/// Practical: max_iterations = 1000-10000, works for any n
///
/// # Parameters
///
/// - initial_temp: Starting temperature (higher = more exploration)
/// - cooling_rate: Temperature decay (0.9-0.99 typical)
/// - max_iterations: Number of iterations to run
pub fn simulated_annealing_planner(
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    hints: &PlanHints,
    initial_temp: f64,
    cooling_rate: f64,
    max_iterations: usize,
) -> Result<Plan> {
    let mut current_plan = greedy_planner(spec, shapes, hints)?;
    let mut current_cost = current_plan.estimated_flops;
    let mut best_plan = current_plan.clone();
    let mut best_cost = current_cost;
    let mut temperature = initial_temp;

    // Use SciRS2-Core's RNG with fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(12345);
    for iteration in 0..max_iterations {
        if current_plan.nodes.len() < 2 {
            break;
        }
        let i = rng.gen_range(0..current_plan.nodes.len());
        let j = rng.gen_range(0..current_plan.nodes.len());
        if i == j {
            continue;
        }
        let mut neighbor_plan = current_plan.clone();
        neighbor_plan.nodes.swap(i, j);
        let neighbor_cost = neighbor_plan.nodes.iter().map(|n| n.cost).sum::<f64>();
        let delta = neighbor_cost - current_cost;
        let accept = if delta < 0.0 {
            true
        } else {
            let prob = E.powf(-delta / temperature);
            rng.random::<f64>() < prob
        };
        if accept {
            current_plan = neighbor_plan;
            current_cost = neighbor_cost;
            if current_cost < best_cost {
                best_plan = current_plan.clone();
                best_cost = current_cost;
            }
        }
        temperature *= cooling_rate;
        if temperature < 1e-10 {
            log::debug!(
                "SA: Early stop at iteration {} (temp={:.2e})",
                iteration,
                temperature
            );
            break;
        }
    }
    best_plan.estimated_flops = best_cost;
    Ok(best_plan)
}

/// Genetic algorithm planner for contraction order
///
/// Uses evolutionary optimization to find high-quality contraction sequences.
/// Excellent for large tensor networks (20+ tensors) where DP is infeasible
/// and beam search may miss good solutions.
///
/// Uses SciRS2-Core's professional-grade RNG with fixed seed (42) for reproducibility.
///
/// # Algorithm
///
/// 1. Initialize population of random contraction orders (using SciRS2 RNG)
/// 2. Evaluate fitness (inverse cost) for each individual
/// 3. Select best individuals for reproduction (tournament selection with SciRS2 RNG)
/// 4. Create offspring via crossover (order-preserving with SciRS2 RNG)
/// 5. Apply mutation (swap operations) for diversity
/// 6. Replace worst individuals with offspring (elitism)
/// 7. Repeat for max_generations
/// 8. Return best plan found
///
/// # Complexity
///
/// O(generations * population_size * n³) where n is number of inputs
/// Practical: population_size = 50-200, generations = 50-200
///
/// # Parameters
///
/// - population_size: Number of candidate plans (higher = better quality, slower)
/// - max_generations: Number of evolution iterations
/// - mutation_rate: Probability of mutation (0.0-1.0, typical: 0.1-0.3)
/// - elitism_count: Number of best individuals to keep unchanged
pub fn genetic_algorithm_planner(
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    hints: &PlanHints,
    population_size: usize,
    max_generations: usize,
    mutation_rate: f64,
    elitism_count: usize,
) -> Result<Plan> {
    let n = spec.num_inputs();

    if n <= 1 {
        return greedy_planner(spec, shapes, hints);
    }

    // Initialize population with random permutations + greedy seed
    let mut population: Vec<Plan> = Vec::with_capacity(population_size);

    // Add greedy solution as seed
    population.push(greedy_planner(spec, shapes, hints)?);

    // Use SciRS2-Core's RNG with fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Fill rest with random valid plans
    while population.len() < population_size {
        // Generate random contraction order by shuffling greedy plan
        let mut plan = population[0].clone();

        // Fisher-Yates shuffle of nodes
        for i in (1..plan.nodes.len()).rev() {
            let j = rng.gen_range(0..=i);
            plan.nodes.swap(i, j);
        }

        // Recompute cost
        plan.estimated_flops = plan.nodes.iter().map(|n| n.cost).sum();
        population.push(plan);
    }

    let mut best_plan = population[0].clone();
    let mut best_cost = best_plan.estimated_flops;

    // Evolution loop
    for _generation in 0..max_generations {
        // Sort population by fitness (lower cost = better)
        population.sort_by(|a, b| a.estimated_flops.partial_cmp(&b.estimated_flops).unwrap());

        // Update best
        if population[0].estimated_flops < best_cost {
            best_plan = population[0].clone();
            best_cost = best_plan.estimated_flops;
        }

        // Create next generation
        let mut offspring: Vec<Plan> = Vec::new();

        // Elitism: keep best individuals
        for item in population.iter().take(elitism_count.min(population.len())) {
            offspring.push(item.clone());
        }

        // Generate offspring through crossover and mutation
        while offspring.len() < population_size {
            // Tournament selection (size 3)
            let parent1_idx = (0..3)
                .map(|_| rng.gen_range(0..population.len()))
                .min_by(|&a, &b| {
                    population[a]
                        .estimated_flops
                        .partial_cmp(&population[b].estimated_flops)
                        .unwrap()
                })
                .unwrap();

            let parent2_idx = (0..3)
                .map(|_| rng.gen_range(0..population.len()))
                .min_by(|&a, &b| {
                    population[a]
                        .estimated_flops
                        .partial_cmp(&population[b].estimated_flops)
                        .unwrap()
                })
                .unwrap();

            // Order crossover (OX): preserve relative order from both parents
            let mut child = population[parent1_idx].clone();
            if child.nodes.len() > 1 {
                let crossover_point = rng
                    .gen_range(1..child.nodes.len())
                    .max(1)
                    .min(child.nodes.len() - 1);

                // Take some nodes from parent2
                if parent2_idx != parent1_idx
                    && population[parent2_idx].nodes.len() > crossover_point
                {
                    for i in 0..crossover_point {
                        if i < population[parent2_idx].nodes.len() {
                            child.nodes[i] = population[parent2_idx].nodes[i].clone();
                        }
                    }
                }
            }

            // Mutation: swap random adjacent nodes
            if rng.random::<f64>() < mutation_rate && child.nodes.len() > 1 {
                let swap_idx = rng.gen_range(0..child.nodes.len() - 1);
                child.nodes.swap(swap_idx, swap_idx + 1);
            }

            // Recompute cost
            child.estimated_flops = child.nodes.iter().map(|n| n.cost).sum();

            offspring.push(child);
        }

        population = offspring;
    }

    Ok(best_plan)
}

/// Refine a plan using local search
///
/// Takes an existing plan and attempts to improve it by trying local modifications:
/// - Swapping adjacent contractions
/// - Trying alternative contraction orders for subsets
///
/// # Algorithm
///
/// 1. Start with input plan
/// 2. Try all adjacent swaps
/// 3. Accept first improvement found
/// 4. Repeat until no improvement for max_no_improve iterations
///
/// # Use Cases
///
/// - Post-process greedy/beam search plans
/// - Fine-tune DP plans with additional constraints
/// - Polish plans before execution
pub fn refine_plan(
    original_plan: &Plan,
    _spec: &EinsumSpec,
    _shapes: &[Vec<usize>],
    _hints: &PlanHints,
    max_iterations: usize,
) -> Result<Plan> {
    let mut current_plan = original_plan.clone();
    let mut current_cost = current_plan.estimated_flops;
    let mut no_improve_count = 0;
    for _iteration in 0..max_iterations {
        let mut improved = false;
        for i in 0..current_plan.nodes.len().saturating_sub(1) {
            let mut neighbor_plan = current_plan.clone();
            neighbor_plan.nodes.swap(i, i + 1);
            let neighbor_cost = neighbor_plan.nodes.iter().map(|n| n.cost).sum::<f64>();
            if neighbor_cost < current_cost {
                current_plan = neighbor_plan;
                current_cost = neighbor_cost;
                improved = true;
                no_improve_count = 0;
                break;
            }
        }
        if !improved {
            no_improve_count += 1;
            if no_improve_count >= 10 {
                break;
            }
        }
    }
    current_plan.estimated_flops = current_cost;
    Ok(current_plan)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Planner;
    use crate::order::{
        AdaptivePlanner, BeamSearchPlanner, DPPlanner, GreedyPlanner, SimulatedAnnealingPlanner,
    };
    #[test]
    fn test_greedy_planner_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();
        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
        assert_eq!(plan.order.len(), 1);
    }
    #[test]
    fn test_greedy_planner_three_tensors() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15], vec![15, 20]];
        let hints = PlanHints::default();
        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 2);
        assert!(plan.estimated_flops > 0.0);
        assert_eq!(plan.order.len(), 2);
    }
    #[test]
    fn test_greedy_planner_large_chain() {
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40], vec![40, 50]];
        let hints = PlanHints::default();
        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 3);
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }
    #[test]
    fn test_greedy_planner_error_shape_mismatch() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20]];
        let hints = PlanHints::default();
        let result = greedy_planner(&spec, &shapes, &hints);
        assert!(result.is_err());
    }
    #[test]
    fn test_compute_pairwise_spec() {
        let a = IntermediateTensor::from_input("ij".to_string(), vec![10, 20], 0);
        let b = IntermediateTensor::from_input("jk".to_string(), vec![20, 30], 1);
        let spec = compute_pairwise_spec(&a, &b).unwrap();
        assert!(spec.output.contains('i'));
        assert!(!spec.output.contains('j'));
        assert!(spec.output.contains('k'));
        assert_eq!(spec.output.len(), 2);
    }
    #[test]
    fn test_compute_pairwise_output_shape() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let a = IntermediateTensor::from_input("ij".to_string(), vec![10, 20], 0);
        let b = IntermediateTensor::from_input("jk".to_string(), vec![20, 30], 1);
        let shape = compute_pairwise_output_shape(&spec, &a, &b).unwrap();
        assert_eq!(shape, vec![10, 30]);
    }
    #[test]
    fn test_dp_planner_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();
        let plan = dp_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }
    #[test]
    fn test_dp_planner_three_tensors() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15], vec![15, 20]];
        let hints = PlanHints::default();
        let plan = dp_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 2);
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }
    #[test]
    fn test_dp_planner_four_tensors() {
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40], vec![40, 50]];
        let hints = PlanHints::default();
        let plan = dp_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 3);
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }
    #[test]
    fn test_dp_planner_single_tensor() {
        let spec = EinsumSpec::parse("ijk->ijk").unwrap();
        let shapes = vec![vec![10, 20, 30]];
        let hints = PlanHints::default();
        let plan = dp_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 0);
        assert_eq!(plan.estimated_flops, 0.0);
    }
    #[test]
    fn test_dp_planner_vs_greedy_small() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![100, 10], vec![10, 100], vec![100, 10]];
        let hints = PlanHints::default();
        let dp_plan = dp_planner(&spec, &shapes, &hints).unwrap();
        let greedy_plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(dp_plan.nodes.len(), 2);
        assert_eq!(greedy_plan.nodes.len(), 2);
        assert!(dp_plan.estimated_flops <= greedy_plan.estimated_flops * 1.01);
    }
    #[test]
    fn test_dp_planner_star_contraction() {
        let spec = EinsumSpec::parse("ia,ib,ic->iabc").unwrap();
        let shapes = vec![vec![10, 5], vec![10, 6], vec![10, 7]];
        let hints = PlanHints::default();
        let plan = dp_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 2);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_dp_planner_fallback_large() {
        let indices = "abcdefghijklmnopqrstu";
        let mut input_specs = Vec::new();
        let mut shapes = Vec::new();
        for c in indices.chars() {
            input_specs.push(c.to_string());
            shapes.push(vec![2]);
        }
        let spec_str = format!("{}->{}", input_specs.join(","), input_specs[0]);
        let spec = EinsumSpec::parse(&spec_str).unwrap();
        let hints = PlanHints::default();
        let result = dp_planner(&spec, &shapes, &hints);
        assert!(result.is_ok());
    }
    #[test]
    fn test_dp_planner_inner_product() {
        let spec = EinsumSpec::parse("ij,ij->").unwrap();
        let shapes = vec![vec![10, 20], vec![10, 20]];
        let hints = PlanHints::default();
        let plan = dp_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_dp_planner_outer_product() {
        let spec = EinsumSpec::parse("i,j,k->ijk").unwrap();
        let shapes = vec![vec![10], vec![20], vec![30]];
        let hints = PlanHints::default();
        let plan = dp_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 2);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_dp_planner_five_tensors() {
        let spec = EinsumSpec::parse("ab,bc,cd,de,ea->abcde").unwrap();
        let shapes = vec![vec![5, 6], vec![6, 7], vec![7, 8], vec![8, 9], vec![9, 5]];
        let hints = PlanHints::default();
        let plan = dp_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 4);
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }
    #[test]
    fn test_greedy_planner_struct() {
        let planner = GreedyPlanner::new();
        let hints = PlanHints::default();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_greedy_planner_default() {
        let planner = GreedyPlanner;
        let hints = PlanHints::default();
        let plan = planner
            .make_plan(
                "ij,jk,kl->il",
                &[vec![5, 10], vec![10, 15], vec![15, 20]],
                &hints,
            )
            .unwrap();
        assert_eq!(plan.nodes.len(), 2);
    }
    #[test]
    fn test_dp_planner_struct() {
        let planner = DPPlanner::new();
        let hints = PlanHints::default();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_dp_planner_default() {
        let planner = DPPlanner;
        let hints = PlanHints::default();
        let plan = planner
            .make_plan(
                "ij,jk,kl->il",
                &[vec![5, 10], vec![10, 15], vec![15, 20]],
                &hints,
            )
            .unwrap();
        assert_eq!(plan.nodes.len(), 2);
    }
    #[test]
    fn test_planner_trait_polymorphism() {
        fn test_planner(planner: &dyn Planner) {
            let hints = PlanHints::default();
            let plan = planner
                .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
                .unwrap();
            assert_eq!(plan.nodes.len(), 1);
        }
        test_planner(&GreedyPlanner::new());
        test_planner(&DPPlanner::new());
    }
    use proptest::prelude::*;
    proptest! {
        #[doc = " Property: Plans should always succeed for valid matmul specs"] #[test]
        fn prop_greedy_planner_matmul_always_succeeds(shared_dim in 1..= 50usize, m in 1
        ..= 30usize, n in 1..= 30usize,) { let spec = EinsumSpec::parse("ij,jk->ik")
        .unwrap(); let shapes = vec![vec![m, shared_dim], vec![shared_dim, n]]; let hints
        = PlanHints::default(); let result = greedy_planner(& spec, & shapes, & hints);
        prop_assert!(result.is_ok()); let plan = result.unwrap(); prop_assert_eq!(plan
        .nodes.len(), 1); prop_assert!(plan.estimated_flops > 0.0); prop_assert!(plan
        .estimated_memory > 0); } #[doc =
        " Property: DP planner should find equal or better cost than greedy"] #[test] fn
        prop_dp_cost_less_or_equal_greedy(dims in prop::collection::vec(2..= 10usize, 3
        ..= 3)) { let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap(); let shapes =
        vec![vec![dims[0], dims[1]], vec![dims[1], dims[2]], vec![dims[2], dims[0]],];
        let hints = PlanHints::default(); let greedy_result = greedy_planner(& spec, &
        shapes, & hints); let dp_result = dp_planner(& spec, & shapes, & hints); if
        greedy_result.is_ok() && dp_result.is_ok() { let greedy_plan = greedy_result
        .unwrap(); let dp_plan = dp_result.unwrap(); prop_assert!(dp_plan.estimated_flops
        <= greedy_plan.estimated_flops * 1.01, "DP cost {} should be ≤ greedy cost {}",
        dp_plan.estimated_flops, greedy_plan.estimated_flops); } } #[doc =
        " Property: Number of contraction steps should be n-1 for n inputs"] #[test] fn
        prop_contraction_steps_count(n_inputs in 2..= 6usize) { let mut input_specs =
        Vec::new(); let mut shapes = Vec::new(); let indices : Vec < char > =
        "abcdefghij".chars().collect(); for i in 0..n_inputs { let idx1 = indices[i]; let
        idx2 = if i < n_inputs - 1 { indices[i + 1] } else { indices[0] }; input_specs
        .push(format!("{}{}", idx1, idx2)); shapes.push(vec![5, 6]); } let output =
        format!("{}{}", indices[0], indices[n_inputs - 1]); let spec_str =
        format!("{}->{}", input_specs.join(","), output); if let Ok(spec) =
        EinsumSpec::parse(& spec_str) { let hints = PlanHints::default(); if let Ok(plan)
        = greedy_planner(& spec, & shapes, & hints) { prop_assert_eq!(plan.nodes.len(),
        n_inputs - 1, "Expected {} contraction steps for {} inputs", n_inputs - 1,
        n_inputs); } } } #[doc = " Property: Plans should be deterministic"] #[test] fn
        prop_plans_are_deterministic(m in 5..= 20usize, n in 5..= 20usize, k in 5..=
        20usize,) { let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap(); let shapes =
        vec![vec![m, n], vec![n, k], vec![k, m]]; let hints = PlanHints::default(); let
        plan1 = greedy_planner(& spec, & shapes, & hints).unwrap(); let plan2 =
        greedy_planner(& spec, & shapes, & hints).unwrap(); prop_assert_eq!(plan1.nodes
        .len(), plan2.nodes.len()); prop_assert_eq!(plan1.estimated_flops, plan2
        .estimated_flops); prop_assert_eq!(plan1.estimated_memory, plan2
        .estimated_memory); } #[doc =
        " Property: Peak memory should be at least as large as output"] #[test] fn
        prop_peak_memory_exceeds_output(m in 10..= 30usize, n in 10..= 30usize,) { let
        spec = EinsumSpec::parse("ij,jk->ik").unwrap(); let shapes = vec![vec![m, n],
        vec![n, m]]; let hints = PlanHints::default(); let plan = greedy_planner(& spec,
        & shapes, & hints).unwrap(); let output_size = m * m * 8; prop_assert!(plan
        .estimated_memory >= output_size, "Peak memory {} should be >= output size {}",
        plan.estimated_memory, output_size); } #[doc =
        " Property: Empty plans for single tensors"] #[test] fn
        prop_single_tensor_no_contraction(rank in 1..= 4usize, size in 5..= 20usize) {
        let indices : String = "ijkl".chars().take(rank).collect(); let spec_str =
        format!("{}->{}", indices, indices); if let Ok(spec) = EinsumSpec::parse(&
        spec_str) { let shapes = vec![vec![size; rank]]; let hints =
        PlanHints::default(); let plan = greedy_planner(& spec, & shapes, & hints)
        .unwrap(); prop_assert_eq!(plan.nodes.len(), 0); prop_assert_eq!(plan
        .estimated_flops, 0.0); } } #[doc =
        " Property: All nodes should have positive cost"] #[test] fn
        prop_all_nodes_positive_cost(dims in prop::collection::vec(5..= 15usize, 4..= 4))
        { let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap(); let shapes =
        vec![vec![dims[0], dims[1]], vec![dims[1], dims[2]], vec![dims[2], dims[3]],
        vec![dims[3], dims[0]],]; let hints = PlanHints::default(); let plan =
        greedy_planner(& spec, & shapes, & hints).unwrap(); for node in & plan.nodes {
        prop_assert!(node.cost > 0.0,
        "All contraction costs should be positive, found {}", node.cost); } }

        #[doc = " Property: Extremely large dimensions should not panic"]
        #[test]
        fn prop_large_dimensions_no_panic(
            m in 1000..=5000usize,
            n in 1000..=5000usize,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![m, n], vec![n, m]];
            let hints = PlanHints::default();
            let result = greedy_planner(&spec, &shapes, &hints);
            prop_assert!(result.is_ok());
            let plan = result.unwrap();
            prop_assert!(plan.estimated_flops > 0.0);
            prop_assert!(plan.estimated_memory > 0);
        }

        #[doc = " Property: Very small dimensions should work correctly"]
        #[test]
        fn prop_minimal_dimensions(
            m in 1..=3usize,
            n in 1..=3usize,
            k in 1..=3usize,
        ) {
            let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
            let shapes = vec![vec![m, n], vec![n, k], vec![k, m]];
            let hints = PlanHints::default();
            let result = greedy_planner(&spec, &shapes, &hints);
            prop_assert!(result.is_ok());
            let plan = result.unwrap();
            prop_assert_eq!(plan.nodes.len(), 2);
        }

        #[doc = " Property: Beam search should never be worse than beam width 1"]
        #[test]
        fn prop_beam_search_quality_improves(
            dims in prop::collection::vec(5..=15usize, 3..=3),
        ) {
            let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
            let shapes = vec![
                vec![dims[0], dims[1]],
                vec![dims[1], dims[2]],
                vec![dims[2], dims[0]],
            ];
            let hints = PlanHints::default();

            let beam1 = beam_search_planner(&spec, &shapes, &hints, 1).unwrap();
            let beam5 = beam_search_planner(&spec, &shapes, &hints, 5).unwrap();

            // Beam width 5 should find equal or better solution than beam width 1
            prop_assert!(
                beam5.estimated_flops <= beam1.estimated_flops * 1.01,
                "Wider beam should find equal or better plan"
            );
        }

        #[doc = " Property: Adaptive planner should always succeed"]
        #[test]
        fn prop_adaptive_planner_robustness(
            n_tensors in 2..=10usize,
            dim in 5..=20usize,
        ) {
            // Create chain contraction
            let mut input_specs = Vec::new();
            let mut shapes = Vec::new();
            let indices: Vec<char> = "abcdefghijklmnop".chars().collect();

            for i in 0..n_tensors {
                let idx1 = indices[i];
                let idx2 = indices[i + 1];
                input_specs.push(format!("{}{}", idx1, idx2));
                shapes.push(vec![dim, dim]);
            }

            let output = format!("{}{}", indices[0], indices[n_tensors]);
            let spec_str = format!("{}->{}", input_specs.join(","), output);

            if EinsumSpec::parse(&spec_str).is_ok() {
                let hints = PlanHints::default();
                let adaptive = AdaptivePlanner::new();
                let result = adaptive.make_plan(&spec_str, &shapes, &hints);
                prop_assert!(result.is_ok(), "Adaptive planner should handle any valid input");
            }
        }
    }
    #[test]
    fn test_beam_search_planner_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();
        let plan = beam_search_planner(&spec, &shapes, &hints, 3).unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_beam_search_planner_chain() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 10]];
        let hints = PlanHints::default();
        let plan = beam_search_planner(&spec, &shapes, &hints, 5).unwrap();
        assert_eq!(plan.nodes.len(), 2);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_beam_search_planner_struct() {
        let planner = BeamSearchPlanner::with_beam_width(3);
        let hints = PlanHints::default();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
    }
    #[test]
    fn test_beam_search_planner_default() {
        let planner = BeamSearchPlanner::new();
        let hints = PlanHints::default();
        let plan = planner
            .make_plan(
                "ij,jk,kl->il",
                &[vec![5, 10], vec![10, 15], vec![15, 5]],
                &hints,
            )
            .unwrap();
        assert_eq!(plan.nodes.len(), 2);
    }
    #[test]
    fn test_beam_search_single_tensor() {
        let spec = EinsumSpec::parse("ij->ij").unwrap();
        let shapes = vec![vec![10, 20]];
        let hints = PlanHints::default();
        let plan = beam_search_planner(&spec, &shapes, &hints, 3).unwrap();
        assert_eq!(plan.nodes.len(), 0);
        assert_eq!(plan.estimated_flops, 0.0);
    }
    #[test]
    fn test_beam_width_one_matches_greedy() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 10]];
        let hints = PlanHints::default();
        let beam_plan = beam_search_planner(&spec, &shapes, &hints, 1).unwrap();
        let greedy_plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(beam_plan.nodes.len(), greedy_plan.nodes.len());
    }
    #[test]
    fn test_simulated_annealing_planner_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();
        let plan = simulated_annealing_planner(&spec, &shapes, &hints, 100.0, 0.95, 100).unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_simulated_annealing_planner_chain() {
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40], vec![40, 10]];
        let hints = PlanHints::default();
        let plan = simulated_annealing_planner(&spec, &shapes, &hints, 1000.0, 0.95, 500).unwrap();
        assert_eq!(plan.nodes.len(), 3);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_simulated_annealing_planner_struct() {
        let planner = SimulatedAnnealingPlanner::new();
        let hints = PlanHints::default();
        let plan = planner
            .make_plan(
                "ij,jk,kl->il",
                &[vec![10, 20], vec![20, 30], vec![30, 10]],
                &hints,
            )
            .unwrap();
        assert_eq!(plan.nodes.len(), 2);
    }
    #[test]
    fn test_simulated_annealing_planner_custom_params() {
        let planner = SimulatedAnnealingPlanner::with_params(500.0, 0.98, 200);
        let hints = PlanHints::default();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
    }
    #[test]
    fn test_simulated_annealing_deterministic_with_fixed_seed() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 10]];
        let hints = PlanHints::default();
        let plan1 = simulated_annealing_planner(&spec, &shapes, &hints, 100.0, 0.95, 50).unwrap();
        let plan2 = simulated_annealing_planner(&spec, &shapes, &hints, 100.0, 0.95, 50).unwrap();
        assert_eq!(plan1.nodes.len(), plan2.nodes.len());
    }
    #[test]
    fn test_refine_plan_improves_or_maintains() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![10, 100], vec![100, 10], vec![10, 10]];
        let hints = PlanHints::default();
        let original_plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let original_cost = original_plan.estimated_flops;
        let refined_plan = refine_plan(&original_plan, &spec, &shapes, &hints, 100).unwrap();
        let refined_cost = refined_plan.estimated_flops;
        assert!(
            refined_cost <= original_cost,
            "Refined cost {} should be <= original cost {}",
            refined_cost,
            original_cost
        );
    }
    #[test]
    fn test_refine_plan_single_node() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();
        let original_plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let refined_plan = refine_plan(&original_plan, &spec, &shapes, &hints, 10).unwrap();
        assert_eq!(refined_plan.nodes.len(), original_plan.nodes.len());
    }
    #[test]
    fn test_planner_trait_polymorphism_all_planners() {
        fn test_planner(planner: &dyn Planner, expected_nodes: usize) {
            let hints = PlanHints::default();
            let plan = planner
                .make_plan(
                    "ij,jk,kl->il",
                    &[vec![10, 20], vec![20, 30], vec![30, 10]],
                    &hints,
                )
                .unwrap();
            assert_eq!(plan.nodes.len(), expected_nodes);
        }
        test_planner(&GreedyPlanner::new(), 2);
        test_planner(&DPPlanner::new(), 2);
        test_planner(&BeamSearchPlanner::new(), 2);
        test_planner(&SimulatedAnnealingPlanner::new(), 2);
    }
    #[test]
    fn test_beam_search_quality_vs_greedy() {
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![100, 10], vec![10, 100], vec![100, 10], vec![10, 100]];
        let hints = PlanHints::default();
        let greedy_plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let beam_plan = beam_search_planner(&spec, &shapes, &hints, 10).unwrap();
        assert!(
            beam_plan.estimated_flops <= greedy_plan.estimated_flops * 1.1,
            "Beam search cost {} should be close to greedy cost {}",
            beam_plan.estimated_flops,
            greedy_plan.estimated_flops
        );
    }
    #[test]
    fn test_adaptive_planner_small_network() {
        let planner = AdaptivePlanner::new();
        let hints = PlanHints::default();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_adaptive_planner_medium_network() {
        let planner = AdaptivePlanner::new();
        let hints = PlanHints::default();
        let plan = planner
            .make_plan(
                "ij,jk,kl,lm,mn->in",
                &[
                    vec![10, 20],
                    vec![20, 30],
                    vec![30, 40],
                    vec![40, 50],
                    vec![50, 10],
                ],
                &hints,
            )
            .unwrap();
        assert_eq!(plan.nodes.len(), 4);
    }
    #[test]
    fn test_adaptive_planner_quality_low() {
        let planner = AdaptivePlanner::with_quality("low");
        let hints = PlanHints::default();
        let plan = planner
            .make_plan(
                "ij,jk,kl->il",
                &[vec![10, 20], vec![20, 30], vec![30, 10]],
                &hints,
            )
            .unwrap();
        assert_eq!(plan.nodes.len(), 2);
    }
    #[test]
    fn test_adaptive_planner_quality_high() {
        let planner = AdaptivePlanner::with_quality("high");
        let hints = PlanHints::default();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
    }
    #[test]
    fn test_adaptive_planner_with_budget() {
        let planner = AdaptivePlanner::with_budget("medium", 50);
        let hints = PlanHints::default();
        let plan = planner
            .make_plan(
                "ij,jk,kl,lm->im",
                &[vec![10, 20], vec![20, 30], vec![30, 40], vec![40, 10]],
                &hints,
            )
            .unwrap();
        assert_eq!(plan.nodes.len(), 3);
    }
    #[test]
    fn test_adaptive_planner_large_network() {
        let planner = AdaptivePlanner::new();
        let hints = PlanHints::default();
        let mut spec_parts = Vec::new();
        let mut shapes = Vec::new();
        let indices: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        for i in 0..25 {
            let idx1 = indices[i];
            let idx2 = indices[i + 1];
            spec_parts.push(format!("{}{}", idx1, idx2));
            shapes.push(vec![5, 5]);
        }
        let spec_str = format!("{}->az", spec_parts.join(","));
        if let Ok(plan) = planner.make_plan(&spec_str, &shapes, &hints) {
            assert_eq!(plan.nodes.len(), 24);
        }
    }
    #[test]
    fn test_adaptive_planner_default() {
        let planner = AdaptivePlanner::default();
        let hints = PlanHints::default();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
    }
    #[test]
    fn test_adaptive_planner_trait() {
        fn test_planner(planner: &dyn Planner) {
            let hints = PlanHints::default();
            let plan = planner
                .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
                .unwrap();
            assert_eq!(plan.nodes.len(), 1);
        }
        test_planner(&AdaptivePlanner::new());
    }
    #[test]
    fn test_all_planners_polymorphism() {
        fn test_all(planner: &dyn Planner, spec: &str, shapes: &[Vec<usize>]) {
            let hints = PlanHints::default();
            let result = planner.make_plan(spec, shapes, &hints);
            assert!(result.is_ok());
        }
        let spec = "ij,jk,kl->il";
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 10]];
        test_all(&GreedyPlanner::new(), spec, &shapes);
        test_all(&DPPlanner::new(), spec, &shapes);
        test_all(&BeamSearchPlanner::new(), spec, &shapes);
        test_all(&SimulatedAnnealingPlanner::new(), spec, &shapes);
        test_all(&AdaptivePlanner::new(), spec, &shapes);
    }
    #[test]
    fn test_genetic_algorithm_planner_simple() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();
        let plan = genetic_algorithm_planner(&spec, &shapes, &hints, 20, 10, 0.2, 2).unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_genetic_algorithm_planner_chain() {
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40], vec![40, 10]];
        let hints = PlanHints::default();
        let plan = genetic_algorithm_planner(&spec, &shapes, &hints, 30, 20, 0.15, 3).unwrap();
        assert_eq!(plan.nodes.len(), 3);
        assert!(plan.estimated_flops > 0.0);
    }
    #[test]
    fn test_genetic_algorithm_quality() {
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![100, 10], vec![10, 100], vec![100, 10], vec![10, 100]];
        let hints = PlanHints::default();
        let greedy_plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let ga_plan = genetic_algorithm_planner(&spec, &shapes, &hints, 50, 30, 0.2, 5).unwrap();
        assert!(
            ga_plan.estimated_flops <= greedy_plan.estimated_flops * 1.1,
            "GA cost {} should be competitive with greedy cost {}",
            ga_plan.estimated_flops,
            greedy_plan.estimated_flops
        );
    }
    #[test]
    fn test_genetic_algorithm_reproducible() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 10]];
        let hints = PlanHints::default();
        let plan1 = genetic_algorithm_planner(&spec, &shapes, &hints, 20, 10, 0.2, 2).unwrap();
        let plan2 = genetic_algorithm_planner(&spec, &shapes, &hints, 20, 10, 0.2, 2).unwrap();
        assert_eq!(plan1.estimated_flops, plan2.estimated_flops);
    }
    #[test]
    fn test_genetic_algorithm_single_tensor() {
        let spec = EinsumSpec::parse("ij->ij").unwrap();
        let shapes = vec![vec![10, 20]];
        let hints = PlanHints::default();
        let plan = genetic_algorithm_planner(&spec, &shapes, &hints, 20, 10, 0.2, 2).unwrap();
        assert_eq!(plan.nodes.len(), 0);
        assert_eq!(plan.estimated_flops, 0.0);
    }
}
