//! Order-driven plan construction shared by the stochastic planners.
//!
//! `build_plan_from_order` turns a fixed, executor-consumable contraction
//! `order` into a full [`Plan`], and `randomized_greedy_order` samples diverse
//! but always-valid contraction trees.  Both are built on the corrected
//! `compute_pairwise_spec` surviving-index rule, so every plan they produce has
//! per-step subscripts that keep exactly the indices still needed (including
//! batch indices) and a `order` field the executor can run verbatim.

use crate::api::{ContractionSpec, Plan, PlanHints, PlanNode};
use crate::cost::{estimate_flops, estimate_output_sparsity, TensorStats};
use crate::parser::EinsumSpec;
use crate::repr::{select_representation, ReprConfig};
use anyhow::{anyhow, Result};
use scirs2_core::random::StdRng;

use super::functions::{compute_pairwise_output_shape, compute_pairwise_spec};
use super::types::IntermediateTensor;

/// Materialise the input operands as [`IntermediateTensor`]s, honouring any
/// per-input sparsity hints (mirrors the setup shared by every planner).
fn initial_intermediates(
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    hints: &PlanHints,
) -> Vec<IntermediateTensor> {
    spec.inputs
        .iter()
        .zip(shapes.iter())
        .enumerate()
        .map(|(i, (indices, shape))| {
            let sparsity = hints.sparsity_hints.get(&i).copied().unwrap_or(0.0);
            let density = (1.0 - sparsity).clamp(0.0, 1.0);
            let stats = TensorStats::with_density(shape.clone(), density);
            IntermediateTensor::from_input_with_stats(indices.clone(), shape.clone(), i, stats)
        })
        .collect()
}

/// Build a full [`Plan`] from a fixed positional contraction `order`.
///
/// Replays the executor's remove-two-push-one discipline: step `k`'s pair
/// `(i, j)` indexes the *current* working list of intermediates, the two named
/// operands are removed, and their contraction is pushed to the end.  Each step's
/// surviving indices come from `compute_pairwise_spec`, which keeps exactly the
/// indices still needed downstream (the final output or a not-yet-consumed
/// operand) — so the resulting per-node `output_spec` values are correct, unlike
/// the pre-fix behaviour that silently summed over shared batch indices.
///
/// Returns an error if `order` is not a valid sequence over the inputs (a step
/// naming an out-of-range or repeated position, or an order that fails to reduce
/// the network to a single tensor).
///
/// # Complexity
///
/// `O(n²)` per step for the surviving-index/shape work, `O(n³)` overall for the
/// `n - 1` steps of an `n`-operand network.
pub(super) fn build_plan_from_order(
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    hints: &PlanHints,
    order: &[(usize, usize)],
) -> Result<Plan> {
    if spec.num_inputs() != shapes.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match shapes ({})",
            spec.num_inputs(),
            shapes.len()
        ));
    }
    let mut intermediates = initial_intermediates(spec, shapes, hints);
    let mut plan = Plan::new();
    let mut total_flops = 0.0;
    let mut peak_memory = 0usize;

    for (step_idx, &(i, j)) in order.iter().enumerate() {
        let len = intermediates.len();
        if i == j || i >= len || j >= len {
            return Err(anyhow!(
                "build_plan_from_order: step {} names invalid pair ({}, {}) for {} operands",
                step_idx,
                i,
                j,
                len
            ));
        }

        let remaining: Vec<&str> = intermediates
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != i && *k != j)
            .map(|(_, t)| t.indices.as_str())
            .collect();
        let a = &intermediates[i];
        let b = &intermediates[j];
        let pairwise_spec =
            compute_pairwise_spec(&a.indices, &b.indices, &remaining, &spec.output)?;
        let output_shape = compute_pairwise_output_shape(&pairwise_spec, a, b)?;
        let stats = vec![a.stats.clone(), b.stats.clone()];
        let cost = estimate_flops(&pairwise_spec, &stats)?;
        let output_density = match estimate_output_sparsity(&pairwise_spec, &stats) {
            Ok(sparsity) => (1.0 - sparsity).clamp(0.0, 1.0),
            Err(_) => 1.0,
        };
        let output_stats = TensorStats::with_density(output_shape.clone(), output_density);
        let output_repr = select_representation(&output_stats, &ReprConfig::default());
        let memory = output_shape.iter().product::<usize>() * 8;
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
            memory,
            repr: output_repr,
        };
        plan.nodes.push(node);
        total_flops += cost;
        peak_memory = peak_memory.max(memory);

        let intermediate = IntermediateTensor::from_contraction(
            pairwise_spec.output.clone(),
            output_shape,
            output_stats,
        );
        let (hi, lo) = if i > j { (i, j) } else { (j, i) };
        intermediates.remove(hi);
        intermediates.remove(lo);
        intermediates.push(intermediate);
    }

    if intermediates.len() != 1 {
        return Err(anyhow!(
            "build_plan_from_order: order left {} operands (expected exactly 1)",
            intermediates.len()
        ));
    }

    plan.estimated_flops = total_flops;
    plan.estimated_memory = peak_memory;
    plan.order = order.to_vec();
    Ok(plan)
}

/// Sample a *valid* contraction order by a randomised greedy walk.
///
/// Uses the same remove-two-push-one discipline as `greedy_planner`, but at each
/// step it picks uniformly at random among the `explore` cheapest feasible pairs
/// rather than strictly the cheapest one.  With `explore == 1` this reduces to the
/// deterministic greedy order.  Because every emitted pair is a legal contraction
/// of the current working list, the returned order is always executable and can
/// be fed straight to `build_plan_from_order`.
///
/// The SA and GA planners use this to draw diverse contraction *trees* (which
/// genuinely change the total FLOP cost), instead of permuting a fixed tree
/// (which does not).
pub(super) fn randomized_greedy_order(
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    hints: &PlanHints,
    rng: &mut StdRng,
    explore: usize,
) -> Result<Vec<(usize, usize)>> {
    if spec.num_inputs() != shapes.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match shapes ({})",
            spec.num_inputs(),
            shapes.len()
        ));
    }
    let mut intermediates = initial_intermediates(spec, shapes, hints);
    let mut order: Vec<(usize, usize)> = Vec::with_capacity(intermediates.len().saturating_sub(1));

    while intermediates.len() > 1 {
        // Every feasible pair with its single-step contraction cost.
        let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..intermediates.len() {
            for j in (i + 1)..intermediates.len() {
                let remaining: Vec<&str> = intermediates
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != i && *k != j)
                    .map(|(_, t)| t.indices.as_str())
                    .collect();
                if let Ok(pairwise_spec) = compute_pairwise_spec(
                    &intermediates[i].indices,
                    &intermediates[j].indices,
                    &remaining,
                    &spec.output,
                ) {
                    // Defer fully-dead contractions (scalar intermediate) to the
                    // final step, matching the deterministic greedy planner: the
                    // einsum representation cannot chain an empty operand.
                    if pairwise_spec.output.is_empty() && intermediates.len() > 2 {
                        continue;
                    }
                    let stats = vec![
                        intermediates[i].stats.clone(),
                        intermediates[j].stats.clone(),
                    ];
                    if let Ok(cost) = estimate_flops(&pairwise_spec, &stats) {
                        candidates.push((cost, i, j));
                    }
                }
            }
        }
        if candidates.is_empty() {
            return Err(anyhow!(
                "randomized_greedy_order: no valid contraction among {} operands",
                intermediates.len()
            ));
        }
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let window = explore.max(1).min(candidates.len());
        let pick = rng.random_range(0..window);
        let (_, i, j) = candidates[pick];
        order.push((i, j));

        // Merge the chosen pair to advance the working list.
        let remaining: Vec<&str> = intermediates
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != i && *k != j)
            .map(|(_, t)| t.indices.as_str())
            .collect();
        let merged = {
            let a = &intermediates[i];
            let b = &intermediates[j];
            let pairwise_spec =
                compute_pairwise_spec(&a.indices, &b.indices, &remaining, &spec.output)?;
            let output_shape = compute_pairwise_output_shape(&pairwise_spec, a, b)?;
            let output_density =
                match estimate_output_sparsity(&pairwise_spec, &[a.stats.clone(), b.stats.clone()])
                {
                    Ok(sparsity) => (1.0 - sparsity).clamp(0.0, 1.0),
                    Err(_) => 1.0,
                };
            let output_stats = TensorStats::with_density(output_shape.clone(), output_density);
            IntermediateTensor::from_contraction(pairwise_spec.output, output_shape, output_stats)
        };
        let (hi, lo) = if i > j { (i, j) } else { (j, i) };
        intermediates.remove(hi);
        intermediates.remove(lo);
        intermediates.push(merged);
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Planner;
    use crate::order::{
        beam_search_planner, dp_planner, genetic_algorithm_planner, greedy_planner,
        simulated_annealing_planner, AdaptivePlanner, BeamSearchPlanner, DPPlanner,
        GeneticAlgorithmPlanner, GreedyPlanner, SimulatedAnnealingPlanner,
    };
    use crate::parser::compute_output_shape;
    use std::collections::HashMap;

    // ===================== Bug 1: compute_pairwise_spec =====================

    #[test]
    fn pairwise_keeps_batch_index_via_final_output() {
        // Single-step batch matmul: `b` is shared but is in the final output,
        // so it must survive; `j` is shared and not needed → contracted.
        let spec = compute_pairwise_spec("bij", "bjk", &[], "bik").unwrap();
        assert_eq!(spec.output, "bik");
    }

    #[test]
    fn pairwise_keeps_batch_index_via_remaining_operand() {
        // First step of `bij,bjk,bkl->bil`: `b` is shared by the pair AND appears
        // in the not-yet-consumed operand `bkl` and the output — it must survive.
        // `j` is shared but dead → contracted; `k` lives on in `bkl` → survives.
        let spec = compute_pairwise_spec("bij", "bjk", &["bkl"], "bil").unwrap();
        assert_eq!(spec.output, "bik");
    }

    #[test]
    fn pairwise_final_step_honours_requested_order() {
        // `ij,jk->ki`: survivors {i,k}; requested order is "ki", not "ik".
        let spec = compute_pairwise_spec("ij", "jk", &[], "ki").unwrap();
        assert_eq!(spec.output, "ki");
    }

    #[test]
    fn pairwise_final_step_permuted_three_operand() {
        // Second/final step of `ij,jk,kl->li` after `ij·jk = ik`:
        // contract `kl` with `ik`; survivors {l,i}; requested "li".
        let spec = compute_pairwise_spec("kl", "ik", &[], "li").unwrap();
        assert_eq!(spec.output, "li");
    }

    #[test]
    fn pairwise_contracts_shared_and_unshared_dead_indices() {
        // Output is just "i": the shared `j` and the unshared `k` are both dead
        // and must be contracted away.
        let spec = compute_pairwise_spec("ij", "jk", &[], "i").unwrap();
        assert_eq!(spec.output, "i");
    }

    #[test]
    fn pairwise_intermediate_step_uses_first_appearance_order() {
        // Non-final step (a later operand `kl` remains): no reorder to output,
        // survivors emitted in first-appearance order across a then b.
        let spec = compute_pairwise_spec("bij", "bjk", &["kl"], "bil").unwrap();
        // b (output/remaining? b not in "kl" but in output) survives, i survives
        // (output), j dead, k survives (in remaining "kl"). First appearance: b,i,k.
        assert_eq!(spec.output, "bik");
    }

    // ============ Bug 1: PlanNode::output_spec through the planners ============

    #[test]
    fn greedy_batch_chain_preserves_batch_in_every_node() {
        let spec = EinsumSpec::parse("bij,bjk,bkl->bil").unwrap();
        let shapes = vec![vec![4, 5, 6], vec![4, 6, 7], vec![4, 7, 8]];
        let hints = PlanHints::default();
        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 2);
        // The batch index `b` is live throughout, so it must appear in every
        // intermediate subscript — the old code dropped it and summed over it.
        for node in &plan.nodes {
            assert!(
                node.output_spec.output_spec.contains('b'),
                "batch index dropped from a step output: {:?}",
                node.output_spec.output_spec
            );
        }
        // Final node yields the requested output exactly.
        let last = plan.nodes.last().unwrap();
        assert_eq!(last.output_spec.output_spec, "bil");
    }

    #[test]
    fn greedy_permuted_output_matches_request() {
        let spec = EinsumSpec::parse("ij,jk,kl->li").unwrap();
        let shapes = vec![vec![3, 4], vec![4, 5], vec![5, 6]];
        let hints = PlanHints::default();
        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let last = plan.nodes.last().unwrap();
        assert_eq!(last.output_spec.output_spec, "li");
    }

    #[test]
    fn greedy_permuted_matmul_output() {
        let spec = EinsumSpec::parse("ij,jk->ki").unwrap();
        let shapes = vec![vec![3, 4], vec![4, 5]];
        let hints = PlanHints::default();
        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert_eq!(plan.nodes[0].output_spec.output_spec, "ki");
    }

    // ======================= Round-trip infrastructure =======================

    /// Execute `order` step by step through the *same* pairwise-spec logic the
    /// planners use, returning the final tensor's labels and shape. Errors if the
    /// order is not a valid remove-two-push-one sequence over the inputs.
    fn simulate_order(
        spec: &EinsumSpec,
        shapes: &[Vec<usize>],
        order: &[(usize, usize)],
    ) -> Result<(String, Vec<usize>)> {
        let mut labels: Vec<String> = spec.inputs.clone();
        let mut shape_of: Vec<Vec<usize>> = shapes.to_vec();
        for (step, &(i, j)) in order.iter().enumerate() {
            let len = labels.len();
            if i == j || i >= len || j >= len {
                return Err(anyhow!(
                    "simulate_order: step {} invalid pair ({}, {}) for {} operands",
                    step,
                    i,
                    j,
                    len
                ));
            }
            let remaining: Vec<&str> = labels
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != i && *k != j)
                .map(|(_, l)| l.as_str())
                .collect();
            let pairwise_spec =
                compute_pairwise_spec(&labels[i], &labels[j], &remaining, &spec.output)?;
            let mut dim: HashMap<char, usize> = HashMap::new();
            for (c, &s) in labels[i].chars().zip(shape_of[i].iter()) {
                dim.insert(c, s);
            }
            for (c, &s) in labels[j].chars().zip(shape_of[j].iter()) {
                dim.insert(c, s);
            }
            let out_shape: Vec<usize> = pairwise_spec
                .output
                .chars()
                .map(|c| *dim.get(&c).unwrap_or(&1))
                .collect();
            let out_labels = pairwise_spec.output.clone();
            let (hi, lo) = if i > j { (i, j) } else { (j, i) };
            labels.remove(hi);
            shape_of.remove(hi);
            labels.remove(lo);
            shape_of.remove(lo);
            labels.push(out_labels);
            shape_of.push(out_shape);
        }
        if labels.len() != 1 || shape_of.len() != 1 {
            return Err(anyhow!(
                "simulate_order: {} operands left (expected 1)",
                labels.len()
            ));
        }
        Ok((labels.pop().unwrap(), shape_of.pop().unwrap()))
    }

    /// Assert a plan's `order` executes down to a single tensor matching the spec.
    fn assert_roundtrip(planner: &str, spec_str: &str, shapes: &[Vec<usize>], plan: &Plan) {
        let spec = EinsumSpec::parse(spec_str).unwrap();
        assert_eq!(
            plan.order.len(),
            spec.num_inputs().saturating_sub(1),
            "{planner}: order for `{spec_str}` should have n-1 steps"
        );
        let (final_labels, final_shape) = simulate_order(&spec, shapes, &plan.order)
            .unwrap_or_else(|e| panic!("{planner}: order for `{spec_str}` is not executable: {e}"));
        assert_eq!(
            final_labels, spec.output,
            "{planner}: `{spec_str}` reduced to labels {final_labels}, expected {}",
            spec.output
        );
        let expected = compute_output_shape(&spec, shapes).unwrap();
        assert_eq!(
            final_shape, expected,
            "{planner}: `{spec_str}` final shape mismatch"
        );
    }

    /// The specs (and matching shapes) exercised by every planner's round trip.
    /// Includes batch indices and permuted outputs — the two Bug 1 hazards.
    fn roundtrip_cases() -> Vec<(&'static str, Vec<Vec<usize>>)> {
        vec![
            ("ij,jk->ik", vec![vec![3, 4], vec![4, 5]]),
            ("ij,jk->ki", vec![vec![3, 4], vec![4, 5]]),
            ("ij,jk,kl->il", vec![vec![3, 4], vec![4, 5], vec![5, 6]]),
            ("ij,jk,kl->li", vec![vec![3, 4], vec![4, 5], vec![5, 6]]),
            ("bij,bjk->bik", vec![vec![2, 3, 4], vec![2, 4, 5]]),
            (
                "bij,bjk,bkl->bil",
                vec![vec![2, 3, 4], vec![2, 4, 5], vec![2, 5, 6]],
            ),
            (
                "ij,jk,kl,lm->im",
                vec![vec![3, 4], vec![4, 5], vec![5, 6], vec![6, 7]],
            ),
        ]
    }

    #[test]
    fn greedy_order_roundtrips() {
        let hints = PlanHints::default();
        for (spec_str, shapes) in roundtrip_cases() {
            let spec = EinsumSpec::parse(spec_str).unwrap();
            let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
            assert_roundtrip("greedy", spec_str, &shapes, &plan);
        }
    }

    #[test]
    fn beam_order_roundtrips() {
        let hints = PlanHints::default();
        for (spec_str, shapes) in roundtrip_cases() {
            let spec = EinsumSpec::parse(spec_str).unwrap();
            let plan = beam_search_planner(&spec, &shapes, &hints, 5).unwrap();
            assert_roundtrip("beam", spec_str, &shapes, &plan);
        }
    }

    #[test]
    fn dp_order_roundtrips() {
        let hints = PlanHints::default();
        for (spec_str, shapes) in roundtrip_cases() {
            let spec = EinsumSpec::parse(spec_str).unwrap();
            let plan = dp_planner(&spec, &shapes, &hints).unwrap();
            // DP previously left `order` empty — this asserts it is now populated
            // AND executable.
            assert!(
                !plan.order.is_empty() || spec.num_inputs() <= 1,
                "dp: order must be populated for `{spec_str}`"
            );
            assert_roundtrip("dp", spec_str, &shapes, &plan);
        }
    }

    #[test]
    fn sa_order_roundtrips() {
        let hints = PlanHints::default();
        for (spec_str, shapes) in roundtrip_cases() {
            let spec = EinsumSpec::parse(spec_str).unwrap();
            let plan =
                simulated_annealing_planner(&spec, &shapes, &hints, 500.0, 0.95, 200).unwrap();
            assert_roundtrip("sa", spec_str, &shapes, &plan);
        }
    }

    #[test]
    fn ga_order_roundtrips() {
        let hints = PlanHints::default();
        for (spec_str, shapes) in roundtrip_cases() {
            let spec = EinsumSpec::parse(spec_str).unwrap();
            let plan = genetic_algorithm_planner(&spec, &shapes, &hints, 24, 15, 0.3, 3).unwrap();
            assert_roundtrip("ga", spec_str, &shapes, &plan);
        }
    }

    #[test]
    fn adaptive_order_roundtrips() {
        let hints = PlanHints::default();
        let planner = AdaptivePlanner::new();
        for (spec_str, shapes) in roundtrip_cases() {
            let plan = planner.make_plan(spec_str, &shapes, &hints).unwrap();
            assert_roundtrip("adaptive", spec_str, &shapes, &plan);
        }
    }

    #[test]
    fn struct_planners_roundtrip_batch_chain() {
        // Cross-check the trait-based planners on the batch chain that was the
        // headline Bug 1 failure.
        let spec_str = "bij,bjk,bkl->bil";
        let shapes = vec![vec![2, 3, 4], vec![2, 4, 5], vec![2, 5, 6]];
        let hints = PlanHints::default();

        let planners: Vec<(&str, Box<dyn Planner>)> = vec![
            ("GreedyPlanner", Box::new(GreedyPlanner::new())),
            ("DPPlanner", Box::new(DPPlanner::new())),
            (
                "BeamSearchPlanner",
                Box::new(BeamSearchPlanner::with_beam_width(4)),
            ),
            (
                "SimulatedAnnealingPlanner",
                Box::new(SimulatedAnnealingPlanner::with_params(500.0, 0.95, 100)),
            ),
            (
                "GeneticAlgorithmPlanner",
                Box::new(GeneticAlgorithmPlanner::with_params(24, 15, 0.3, 3)),
            ),
        ];
        for (name, planner) in planners {
            let plan = planner.make_plan(spec_str, &shapes, &hints).unwrap();
            assert_roundtrip(name, spec_str, &shapes, &plan);
        }
    }

    // ================= build_plan_from_order / order helpers =================

    #[test]
    fn build_plan_from_order_matches_greedy_shape() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40]];
        let hints = PlanHints::default();
        let greedy = greedy_planner(&spec, &shapes, &hints).unwrap();
        // Rebuilding from greedy's own order must reproduce greedy exactly.
        let rebuilt = build_plan_from_order(&spec, &shapes, &hints, &greedy.order).unwrap();
        assert_eq!(rebuilt.nodes.len(), greedy.nodes.len());
        assert_eq!(rebuilt.estimated_flops, greedy.estimated_flops);
        assert_eq!(rebuilt.order, greedy.order);
        assert_roundtrip("build_from_order", "ij,jk,kl->il", &shapes, &rebuilt);
    }

    #[test]
    fn build_plan_from_order_rejects_invalid_order() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40]];
        let hints = PlanHints::default();
        // (0, 5) is out of range at the first step.
        let bad = vec![(0usize, 5usize), (0, 1)];
        assert!(build_plan_from_order(&spec, &shapes, &hints, &bad).is_err());
        // A single step cannot reduce three operands to one.
        let short = vec![(0usize, 1usize)];
        assert!(build_plan_from_order(&spec, &shapes, &hints, &short).is_err());
    }

    #[test]
    fn randomized_greedy_order_is_always_executable() {
        use scirs2_core::random::SeedableRng;
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![4, 5], vec![5, 6], vec![6, 7], vec![7, 8]];
        let hints = PlanHints::default();
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..32 {
            let order = randomized_greedy_order(&spec, &shapes, &hints, &mut rng, 3).unwrap();
            assert_eq!(order.len(), 3);
            let plan = build_plan_from_order(&spec, &shapes, &hints, &order).unwrap();
            assert_roundtrip("randomized_greedy", "ij,jk,kl,lm->im", &shapes, &plan);
        }
    }

    #[test]
    fn randomized_greedy_explore_one_is_deterministic_greedy() {
        use scirs2_core::random::SeedableRng;
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![100, 10], vec![10, 100], vec![100, 10]];
        let hints = PlanHints::default();
        let mut rng = StdRng::seed_from_u64(1);
        let rnd = randomized_greedy_order(&spec, &shapes, &hints, &mut rng, 1).unwrap();
        let greedy = greedy_planner(&spec, &shapes, &hints).unwrap();
        // explore == 1 always takes the cheapest pair, i.e. the greedy choice.
        let rnd_cost = build_plan_from_order(&spec, &shapes, &hints, &rnd)
            .unwrap()
            .estimated_flops;
        assert_eq!(rnd_cost, greedy.estimated_flops);
    }

    #[test]
    fn sa_and_ga_never_worse_than_greedy() {
        // Both metaheuristics seed with greedy and only ever keep a strictly
        // cheaper order, so they must be <= greedy (allowing float slack).
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![100, 10], vec![10, 100], vec![100, 10], vec![10, 100]];
        let hints = PlanHints::default();
        let greedy = greedy_planner(&spec, &shapes, &hints).unwrap();
        let sa = simulated_annealing_planner(&spec, &shapes, &hints, 1000.0, 0.95, 300).unwrap();
        let ga = genetic_algorithm_planner(&spec, &shapes, &hints, 40, 25, 0.3, 4).unwrap();
        assert!(sa.estimated_flops <= greedy.estimated_flops * 1.0001);
        assert!(ga.estimated_flops <= greedy.estimated_flops * 1.0001);
    }
}
