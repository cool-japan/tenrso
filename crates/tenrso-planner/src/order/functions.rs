//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
use crate::api::ReprHint;
use crate::api::{ContractionSpec, Plan, PlanHints, PlanNode};
use crate::cost::{estimate_flops, estimate_output_sparsity, TensorStats};
use crate::parser::EinsumSpec;
use crate::repr::{select_representation, ReprConfig};
use anyhow::{anyhow, Result};
use scirs2_core::random::{SeedableRng, StdRng};
use std::collections::HashMap;

use super::plan_ops::{build_plan_from_order, randomized_greedy_order};
use super::types::IntermediateTensor;
/// Compute the surviving-index einsum spec for one pairwise contraction step.
///
/// An index of operand `a` or `b` **survives** the step iff it is still needed
/// afterwards — that is, it appears in the final einsum output `final_output`, or
/// in one of `remaining_indices` (the index strings of the operands that are
/// *not* consumed by this step).  Every other index — whether it is *shared* by
/// the pair or appears in only one operand — is contracted (summed) here, because
/// this is its last live appearance across {both operands, all remaining
/// operands, the final output}.
///
/// Surviving indices are emitted in order of first appearance across `a` then
/// `b`, with duplicates dropped (a repeated index inside one operand collapses to
/// a single diagonal axis).  On the **final** step — signalled by an empty
/// `remaining_indices` — the survivors are exactly the requested output indices,
/// so they are emitted in `final_output`'s order rather than first-appearance
/// order.  This restores the caller's requested layout for specs such as
/// `"ij,jk,kl->li"`.
///
/// # Why the extra context is needed
///
/// The previous implementation dropped every *shared* index unconditionally and
/// then sorted the survivors alphabetically.  Both are wrong: dropping a shared
/// index silently sums over a batch index that is still needed downstream (e.g.
/// `b` in `"bij,bjk,bkl->bil"` produced a right-shaped but wrong-valued result),
/// and sorting discards the requested output order.  Deciding correctly requires
/// knowing which indices are still *live*, hence `remaining_indices` and
/// `final_output`.  This mirrors the executor's `step_output_labels`.
pub(super) fn compute_pairwise_spec(
    a_indices: &str,
    b_indices: &str,
    remaining_indices: &[&str],
    final_output: &str,
) -> Result<EinsumSpec> {
    let mut output = String::new();
    for c in a_indices.chars().chain(b_indices.chars()) {
        if output.contains(c) {
            continue;
        }
        let needed_later =
            final_output.contains(c) || remaining_indices.iter().any(|l| l.contains(c));
        if needed_later {
            output.push(c);
        }
    }
    if remaining_indices.is_empty() {
        // Final step: the survivors are exactly the requested output indices, so
        // honour the caller's requested order instead of first-appearance order.
        let survivors: std::collections::HashSet<char> = output.chars().collect();
        let requested: std::collections::HashSet<char> = final_output.chars().collect();
        if survivors == requested {
            output = final_output.to_string();
        }
    }
    let spec_str = format!("{},{}->{}", a_indices, b_indices, output);
    EinsumSpec::parse(&spec_str)
}
/// Compute the output shape for a pairwise contraction
pub(super) fn compute_pairwise_output_shape(
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
pub fn greedy_planner(spec: &EinsumSpec, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
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
            let sparsity = hints.sparsity_hints.get(&i).copied().unwrap_or(0.0);
            let density = (1.0 - sparsity).clamp(0.0, 1.0);
            let stats = TensorStats::with_density(shape.clone(), density);
            IntermediateTensor::from_input_with_stats(indices.clone(), shape.clone(), i, stats)
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
                // Operands still live *after* contracting this pair: every other
                // intermediate. Their indices (plus the final output) decide which
                // of the pair's indices survive the step.
                let remaining: Vec<&str> = intermediates
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != i && *k != j)
                    .map(|(_, t)| t.indices.as_str())
                    .collect();
                let a = &intermediates[i];
                let b = &intermediates[j];
                if let Ok(pairwise_spec) =
                    compute_pairwise_spec(&a.indices, &b.indices, &remaining, &spec.output)
                {
                    // A step that contracts *all* of a pair's indices yields a
                    // scalar. The planner represents intermediates as einsum
                    // subscripts, which cannot express an empty operand, so a
                    // scalar may only appear as the *final* result. Deferring such
                    // a fully-dead contraction is always possible and never
                    // changes the result — only the order.
                    if pairwise_spec.output.is_empty() && intermediates.len() > 2 {
                        continue;
                    }
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
        let (pairwise_spec, output_shape) = match (best_spec, best_shape) {
            (Some(spec), Some(shape)) => (spec, shape),
            _ => return Err(anyhow!("No valid contraction found")),
        };
        let (i, j) = best_pair;
        let a = &intermediates[i];
        let b = &intermediates[j];
        let input_indices = vec![
            a.original_idx.unwrap_or(1000 + i),
            b.original_idx.unwrap_or(1000 + j),
        ];
        let output_density = {
            let a_stats = a.stats.clone();
            let b_stats = b.stats.clone();
            match estimate_output_sparsity(&pairwise_spec, &[a_stats, b_stats]) {
                Ok(sparsity) => (1.0 - sparsity).clamp(0.0, 1.0),
                Err(_) => 1.0,
            }
        };
        let output_stats = TensorStats::with_density(output_shape.clone(), output_density);
        let output_repr = select_representation(&output_stats, &ReprConfig::default());
        let node = PlanNode {
            inputs: input_indices.clone(),
            output_spec: ContractionSpec::new(
                vec![a.indices.clone(), b.indices.clone()],
                pairwise_spec.output.clone(),
            ),
            cost: best_cost,
            memory: output_shape.iter().product::<usize>() * 8,
            repr: output_repr,
        };
        plan.nodes.push(node);
        total_flops += best_cost;
        peak_memory = peak_memory.max(output_shape.iter().product::<usize>() * 8);
        contraction_order.push(best_pair);
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
            let sparsity = hints.sparsity_hints.get(&i).copied().unwrap_or(0.0);
            let density = (1.0 - sparsity).clamp(0.0, 1.0);
            let stats = TensorStats::with_density(shape.clone(), density);
            IntermediateTensor::from_input_with_stats(indices.clone(), shape.clone(), i, stats)
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
        // Indices still needed *outside* this subnetwork: those of any original
        // input whose bit is not in `mask`. An index internal to `mask` (and not
        // in the final output) is contracted when `mask` is formed; a shared
        // index that also appears outside `mask` survives.
        let remaining_for_mask: Vec<&str> = (0..n)
            .filter(|b| (mask & (1 << b)) == 0)
            .map(|b| intermediates[b].indices.as_str())
            .collect();
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
                        if let Ok(pairwise_spec) = compute_pairwise_spec(
                            &tensor1.indices,
                            &tensor2.indices,
                            &remaining_for_mask,
                            &spec.output,
                        ) {
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
            // No representable partition of this subset: every split would create
            // a scalar intermediate the planner's einsum representation cannot
            // chain into a later step (e.g. a subset of purely contracted
            // vectors). Leave the subset infeasible (`dp[mask]` stays `None`); any
            // superset routes around it, and if the full network turns out to be
            // unreachable the final lookup reports it.
            continue;
        }
        dp[mask] = Some((best_cost, best_partition));
        let submask = best_partition;
        let complement = mask ^ submask;
        if let (Some(tensor1), Some(tensor2)) = (
            cached_contractions.get(&submask),
            cached_contractions.get(&complement),
        ) {
            if let Ok(pairwise_spec) = compute_pairwise_spec(
                &tensor1.indices,
                &tensor2.indices,
                &remaining_for_mask,
                &spec.output,
            ) {
                if let Ok(output_shape) =
                    compute_pairwise_output_shape(&pairwise_spec, tensor1, tensor2)
                {
                    let output_density = {
                        let s1 = tensor1.stats.clone();
                        let s2 = tensor2.stats.clone();
                        match estimate_output_sparsity(&pairwise_spec, &[s1, s2]) {
                            Ok(sparsity) => (1.0 - sparsity).clamp(0.0, 1.0),
                            Err(_) => 1.0,
                        }
                    };
                    let output_stats =
                        TensorStats::with_density(output_shape.clone(), output_density);
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
    /// Walk the optimal DP tree in post-order, emitting one [`PlanNode`] per merge
    /// and recording each merge as the `(submask, complement)` bitmask pair so the
    /// caller can turn the tree into an executor-consumable positional order.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_plan(
        mask: usize,
        dp: &[Option<(f64, usize)>],
        cached_contractions: &HashMap<usize, IntermediateTensor>,
        plan: &mut Plan,
        peak_memory: &mut usize,
        merges: &mut Vec<(usize, usize)>,
        intermediates: &[IntermediateTensor],
        final_output: &str,
    ) -> Result<()> {
        if mask.count_ones() == 1 {
            return Ok(());
        }
        let (_, best_partition) =
            dp[mask].ok_or_else(|| anyhow!("Missing DP state for mask {}", mask))?;
        let submask = best_partition;
        let complement = mask ^ submask;
        reconstruct_plan(
            submask,
            dp,
            cached_contractions,
            plan,
            peak_memory,
            merges,
            intermediates,
            final_output,
        )?;
        reconstruct_plan(
            complement,
            dp,
            cached_contractions,
            plan,
            peak_memory,
            merges,
            intermediates,
            final_output,
        )?;
        let n = intermediates.len();
        let remaining_for_mask: Vec<&str> = (0..n)
            .filter(|b| (mask & (1 << b)) == 0)
            .map(|b| intermediates[b].indices.as_str())
            .collect();
        let tensor1 = cached_contractions
            .get(&submask)
            .ok_or_else(|| anyhow!("Missing cached tensor for submask {}", submask))?;
        let tensor2 = cached_contractions
            .get(&complement)
            .ok_or_else(|| anyhow!("Missing cached tensor for complement {}", complement))?;
        let pairwise_spec = compute_pairwise_spec(
            &tensor1.indices,
            &tensor2.indices,
            &remaining_for_mask,
            final_output,
        )?;
        let output_shape = compute_pairwise_output_shape(&pairwise_spec, tensor1, tensor2)?;
        let stats = vec![tensor1.stats.clone(), tensor2.stats.clone()];
        let cost = estimate_flops(&pairwise_spec, &stats)?;
        let memory = output_shape.iter().product::<usize>() * 8;
        *peak_memory = (*peak_memory).max(memory);
        let output_density = match estimate_output_sparsity(&pairwise_spec, &stats) {
            Ok(sparsity) => (1.0 - sparsity).clamp(0.0, 1.0),
            Err(_) => 1.0,
        };
        let output_stats = TensorStats::with_density(output_shape.clone(), output_density);
        let output_repr = select_representation(&output_stats, &ReprConfig::default());
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
            repr: output_repr,
        };
        plan.nodes.push(node);
        merges.push((submask, complement));
        Ok(())
    }
    let mut peak_memory = 0;
    let mut merges: Vec<(usize, usize)> = Vec::with_capacity(n.saturating_sub(1));
    reconstruct_plan(
        full_mask,
        &dp,
        &cached_contractions,
        &mut plan,
        &mut peak_memory,
        &mut merges,
        &intermediates,
        &spec.output,
    )?;
    plan.estimated_memory = peak_memory;
    // Turn the post-order merge tree into a positional `(i, j)` order over a
    // shrinking working list, exactly matching the executor's remove-two-push-one
    // discipline: each token is a subset bitmask, merges consume two present
    // tokens and push their union. Post-order guarantees both operands of a merge
    // are already single tokens when it is processed.
    let mut working: Vec<usize> = (0..n).map(|i| 1usize << i).collect();
    let mut order: Vec<(usize, usize)> = Vec::with_capacity(merges.len());
    for &(submask, complement) in &merges {
        let pos_a = working
            .iter()
            .position(|&token| token == submask)
            .ok_or_else(|| anyhow!("DP order: submask {} not present in working list", submask))?;
        let pos_b = working
            .iter()
            .position(|&token| token == complement)
            .ok_or_else(|| {
                anyhow!(
                    "DP order: complement {} not present in working list",
                    complement
                )
            })?;
        order.push((pos_a, pos_b));
        let (hi, lo) = if pos_a > pos_b {
            (pos_a, pos_b)
        } else {
            (pos_b, pos_a)
        };
        working.remove(hi);
        working.remove(lo);
        working.push(submask | complement);
    }
    plan.order = order;
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
    hints: &PlanHints,
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
            let sparsity = hints.sparsity_hints.get(&i).copied().unwrap_or(0.0);
            let density = (1.0 - sparsity).clamp(0.0, 1.0);
            let stats = TensorStats::with_density(shape.clone(), density);
            IntermediateTensor::from_input_with_stats(indices.clone(), shape.clone(), i, stats)
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
                    let remaining: Vec<&str> = candidate
                        .intermediates
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != i && *k != j)
                        .map(|(_, t)| t.indices.as_str())
                        .collect();
                    let a = &candidate.intermediates[i];
                    let b = &candidate.intermediates[j];
                    if let Ok(pairwise_spec) =
                        compute_pairwise_spec(&a.indices, &b.indices, &remaining, &spec.output)
                    {
                        // Defer fully-dead contractions (empty intermediate) to the
                        // final step: the planner cannot chain a scalar operand.
                        if pairwise_spec.output.is_empty() && candidate.intermediates.len() > 2 {
                            continue;
                        }
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
                                let output_density = {
                                    let sa = a.stats.clone();
                                    let sb = b.stats.clone();
                                    match estimate_output_sparsity(&pairwise_spec, &[sa, sb]) {
                                        Ok(sparsity) => (1.0 - sparsity).clamp(0.0, 1.0),
                                        Err(_) => 1.0,
                                    }
                                };
                                let output_stats =
                                    TensorStats::with_density(output_shape.clone(), output_density);
                                let output_repr =
                                    select_representation(&output_stats, &ReprConfig::default());
                                let node = PlanNode {
                                    inputs: input_indices,
                                    output_spec: ContractionSpec::new(
                                        vec![a.indices.clone(), b.indices.clone()],
                                        pairwise_spec.output.clone(),
                                    ),
                                    cost,
                                    memory: output_shape.iter().product::<usize>() * 8,
                                    repr: output_repr,
                                };
                                new_candidate.plan.nodes.push(node);
                                new_candidate.total_flops += cost;
                                new_candidate.peak_memory = new_candidate
                                    .peak_memory
                                    .max(output_shape.iter().product::<usize>() * 8);
                                new_candidate.plan.order.push((i, j));
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
        next_beam.sort_by(|a, b| {
            a.total_flops
                .partial_cmp(&b.total_flops)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        next_beam.truncate(beam_width);
        beam = next_beam;
    }
    let best = beam
        .into_iter()
        .min_by(|a, b| {
            a.total_flops
                .partial_cmp(&b.total_flops)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| anyhow!("No complete plan found"))?;
    let mut final_plan = best.plan;
    final_plan.estimated_flops = best.total_flops;
    final_plan.estimated_memory = best.peak_memory;
    Ok(final_plan)
}
use std::f64::consts::E;
/// Simulated annealing planner for contraction order
///
/// Uses stochastic search with temperature-based acceptance to escape local
/// minima. Good for large tensor networks where DP is infeasible and greedy may
/// be suboptimal.
///
/// Uses SciRS2-Core's professional-grade RNG with fixed seed (12345) for
/// reproducibility.
///
/// # Algorithm
///
/// The search space is the set of **valid contraction trees**, each represented
/// by an executable positional order (`Vec<(usize, usize)>`). The total FLOP cost
/// of a tree is order-invariant to the *linearisation* of independent steps, so
/// SA moves between *different trees*, not between permutations of a fixed tree:
///
/// 1. Start from the deterministic greedy order (also the guaranteed-valid
///    fallback).
/// 2. Propose a neighbour by sampling a fresh randomised-greedy tree
///    (`randomized_greedy_order`).
/// 3. Score both with the corrected pairwise cost model via
///    `build_plan_from_order`.
/// 4. Accept if cheaper, else with probability `exp(-ΔE / T)`.
/// 5. Cool the temperature and track the best order seen.
/// 6. Rebuild and return the plan for the best order — its `order` field is
///    always populated and executable.
///
/// # Complexity
///
/// `O(max_iterations · n³)` (each proposal is a full randomised-greedy walk).
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
    let greedy = greedy_planner(spec, shapes, hints)?;
    // Fewer than two steps: there is exactly one contraction tree, so there is
    // nothing to anneal — return the greedy plan (already has a valid `order`).
    if greedy.nodes.len() < 2 {
        return Ok(greedy);
    }

    // Use SciRS2-Core's RNG with fixed seed for reproducibility.
    let mut rng = StdRng::seed_from_u64(12345);
    let mut current_order = greedy.order.clone();
    let mut current_cost = greedy.estimated_flops;
    let mut best_order = current_order.clone();
    let mut best_cost = current_cost;
    let mut temperature = initial_temp;

    for iteration in 0..max_iterations {
        let candidate_order = match randomized_greedy_order(spec, shapes, hints, &mut rng, 3) {
            Ok(order) => order,
            Err(_) => continue,
        };
        let candidate_cost = match build_plan_from_order(spec, shapes, hints, &candidate_order) {
            Ok(plan) => plan.estimated_flops,
            Err(_) => continue,
        };
        let delta = candidate_cost - current_cost;
        let accept = if delta < 0.0 {
            true
        } else {
            let prob = E.powf(-delta / temperature.max(1e-12));
            rng.random_f64() < prob
        };
        if accept {
            current_order = candidate_order;
            current_cost = candidate_cost;
            if current_cost < best_cost {
                best_order = current_order.clone();
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

    // Rebuild the winning tree into a full, executable plan.
    build_plan_from_order(spec, shapes, hints, &best_order)
}

/// Genetic algorithm planner for contraction order
///
/// Uses evolutionary optimization to find high-quality contraction sequences.
/// Excellent for large tensor networks (20+ tensors) where DP is infeasible
/// and beam search may miss good solutions.
///
/// Uses SciRS2-Core's professional-grade RNG with fixed seed (42) for
/// reproducibility.
///
/// # Algorithm
///
/// Individuals are **valid contraction trees**, each carried as an executable
/// positional order (`Vec<(usize, usize)>`) together with its true FLOP cost
/// (evaluated with the corrected pairwise cost model via
/// `build_plan_from_order`):
///
/// 1. Seed the population with the deterministic greedy tree, then fill it with
///    diverse randomised-greedy trees (`randomized_greedy_order`).
/// 2. Each generation, sort by cost, keep the best `elitism_count` unchanged,
///    and breed the rest by tournament selection (size 3).
/// 3. Reproduction is mutation-based: with probability `mutation_rate` a child is
///    a freshly sampled randomised-greedy tree, otherwise it clones its parent.
///    (Positional-order crossover is *not* closed under validity for contraction
///    trees — an arbitrary splice of two orders rarely remains a legal
///    remove-two-push-one sequence — so this is a `(μ + λ)` evolutionary strategy
///    without crossover, which keeps every individual executable by construction.)
/// 4. Return the plan rebuilt from the best order found; its `order` field is
///    always populated and executable.
///
/// # Complexity
///
/// `O(generations · population_size · n³)` where n is number of inputs.
/// Practical: population_size = 50-200, generations = 50-200.
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

    let greedy = greedy_planner(spec, shapes, hints)?;
    // A single contraction step admits exactly one tree; nothing to evolve.
    if greedy.nodes.len() < 2 {
        return Ok(greedy);
    }

    // An individual is a valid order paired with its true total cost.
    type Individual = (Vec<(usize, usize)>, f64);
    let pop_target = population_size.max(1);
    let cost_of = |order: &[(usize, usize)]| -> Option<f64> {
        build_plan_from_order(spec, shapes, hints, order)
            .ok()
            .map(|plan| plan.estimated_flops)
    };

    // Use SciRS2-Core's RNG with fixed seed for reproducibility.
    let mut rng = StdRng::seed_from_u64(42);

    // Seed with the deterministic greedy tree, then add randomised-greedy trees.
    let mut population: Vec<Individual> = Vec::with_capacity(pop_target);
    population.push((greedy.order.clone(), greedy.estimated_flops));
    // Bound the sampling attempts so a pathological spec cannot spin forever;
    // every feasible spec yields a valid randomised-greedy tree (greedy already
    // succeeded above), so this cap is only a safety net.
    let mut attempts = 0usize;
    let max_attempts = pop_target.saturating_mul(8).max(16);
    while population.len() < pop_target && attempts < max_attempts {
        attempts += 1;
        if let Ok(order) = randomized_greedy_order(spec, shapes, hints, &mut rng, 4) {
            if let Some(cost) = cost_of(&order) {
                population.push((order, cost));
            }
        }
    }

    let mut best: Individual = population
        .iter()
        .cloned()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((greedy.order.clone(), greedy.estimated_flops));

    for _generation in 0..max_generations {
        population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if population[0].1 < best.1 {
            best = population[0].clone();
        }

        let mut offspring: Vec<Individual> = Vec::with_capacity(pop_target);
        // Elitism: carry the best individuals forward unchanged.
        for item in population.iter().take(elitism_count.min(population.len())) {
            offspring.push(item.clone());
        }

        while offspring.len() < pop_target {
            // Tournament selection (size 3). `population` is non-empty.
            let parent_idx = (0..3)
                .map(|_| rng.random_range(0..population.len()))
                .min_by(|&a, &b| {
                    population[a]
                        .1
                        .partial_cmp(&population[b].1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);

            // Mutation-based reproduction: resample a fresh tree, or clone parent.
            let child = if rng.random_f64() < mutation_rate {
                match randomized_greedy_order(spec, shapes, hints, &mut rng, 4) {
                    Ok(order) => match cost_of(&order) {
                        Some(cost) => (order, cost),
                        None => population[parent_idx].clone(),
                    },
                    Err(_) => population[parent_idx].clone(),
                }
            } else {
                population[parent_idx].clone()
            };
            offspring.push(child);
        }

        population = offspring;
    }

    if let Some(final_best) = population
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    {
        if final_best.1 < best.1 {
            best = final_best.clone();
        }
    }

    build_plan_from_order(spec, shapes, hints, &best.0)
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
        // Final matmul step (no operands remain): j is contracted, output is "ik".
        let spec = compute_pairwise_spec("ij", "jk", &[], "ik").unwrap();
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
        #![proptest_config(ProptestConfig { cases: 10, ..ProptestConfig::default() })]
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
    // NOTE: `refine_plan`'s tests now live in `order::refine`, alongside the
    // implementation. The two tests that used to sit here only asserted
    // `refined <= original` and an unchanged node count — both of which the old
    // no-op implementation satisfied trivially. Their strengthened successors are
    // `refine_improves_or_maintains_a_greedy_plan` and
    // `refine_single_step_plan_is_a_fixed_point`.
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

    // ----- repr hint tests -----

    #[test]
    fn test_repr_hint_sparse_input() {
        use crate::api::Planner;
        use crate::order::GreedyPlanner;
        // Two very sparse inputs (95% sparse) with large shapes → output should be Sparse
        let mut hints = PlanHints::default();
        hints.sparsity_hints.insert(0, 0.95);
        hints.sparsity_hints.insert(1, 0.95);
        let planner = GreedyPlanner::new();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![1000, 1000], vec![1000, 1000]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
        // With 95% sparse inputs the output density ≈ 0.05*0.05 = 0.0025 (99.75% sparse),
        // and size = 1_000_000 >= min_sparse_size(10_000) so repr must be Sparse.
        assert_eq!(
            plan.nodes[0].repr,
            ReprHint::Sparse,
            "Expected Sparse repr for highly-sparse inputs, got {:?}",
            plan.nodes[0].repr
        );
    }

    #[test]
    fn test_repr_hint_dense_default() {
        use crate::api::Planner;
        use crate::order::GreedyPlanner;
        // No sparsity hints → all dense → output should be Dense
        let hints = PlanHints::default();
        let planner = GreedyPlanner::new();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert_eq!(
            plan.nodes[0].repr,
            ReprHint::Dense,
            "Expected Dense repr for fully-dense inputs, got {:?}",
            plan.nodes[0].repr
        );
    }

    #[test]
    fn test_repr_hint_lowrank_candidate() {
        use crate::api::Planner;
        use crate::order::GreedyPlanner;
        // A very rectangular output shape triggers LowRank:
        // input 0: "ij" shape [10000, 10] — rectangular, min/max ratio = 0.001 < 0.3
        // input 1: "jk" shape [10, 10000]
        // output "ik" shape [10000, 10000] — actually square, not rectangular.
        // Instead use shapes that give a rectangular output:
        // "ij,j->i" with shape [10000, 5] gives output shape [10000] — 1D, not ≥2D.
        // Use "ij,jk->ik" where output is [10000, 10], ratio 10/10000 = 0.001 < 0.3
        let hints = PlanHints::default();
        let planner = GreedyPlanner::new();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10_000, 10], vec![10, 10]], &hints)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
        // Output shape is [10000, 10]: min_dim=10, max_dim=10000, ratio=0.001 < 0.3
        // size = 100000 >= 10000 → LowRank
        assert_eq!(
            plan.nodes[0].repr,
            ReprHint::LowRank,
            "Expected LowRank repr for rectangular output, got {:?}",
            plan.nodes[0].repr
        );
    }
}
