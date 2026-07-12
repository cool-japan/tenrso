//! Local-search refinement of an existing contraction plan.
//!
//! [`refine_plan`] improves a plan by searching over **contraction trees** — the
//! shape of the pairwise contraction tree is what actually drives cost — and
//! rescoring every candidate by rebuilding it through
//! [`build_plan_from_order`](super::plan_ops::build_plan_from_order).  The plan
//! it returns therefore has `nodes`, `order` and `estimated_flops` that are
//! mutually consistent and executable by construction.

use crate::api::{Plan, PlanHints};
use crate::parser::EinsumSpec;
use anyhow::{anyhow, Result};

use super::functions::greedy_planner;
use super::plan_ops::build_plan_from_order;

/// A rooted binary contraction tree over the einsum inputs.
///
/// Node ids `0..n_inputs` are the leaves (the input operands).  Internal node
/// `n_inputs + t` is `children[t]`, the pair contracted at that node.  The tree
/// — not the linear order — is what determines cost: two linearisations of the
/// same tree contract the same pairs and so cost the same, whereas a different
/// tree contracts different pairs and genuinely changes the FLOP count.  This is
/// exactly the distinction the previous `refine_plan` missed.
#[derive(Clone, Debug)]
struct ContractionTree {
    /// Number of leaves (einsum input operands).
    n_inputs: usize,
    /// `children[t]` are the two node ids merged by internal node `n_inputs + t`.
    children: Vec<(usize, usize)>,
}

impl ContractionTree {
    /// Recover the tree from an executable positional `order`.
    ///
    /// Replays the executor's remove-two-push-one discipline: step `k` names two
    /// positions in the *current* working list, which resolve to stable node ids.
    ///
    /// Returns an error if `order` is not a valid sequence over `n_inputs`
    /// operands (out-of-range or repeated position, or an order that fails to
    /// reduce the network to a single tensor).
    fn from_order(n_inputs: usize, order: &[(usize, usize)]) -> Result<Self> {
        let mut working: Vec<usize> = (0..n_inputs).collect();
        let mut children = Vec::with_capacity(order.len());
        for (step, &(i, j)) in order.iter().enumerate() {
            let len = working.len();
            if i == j || i >= len || j >= len {
                return Err(anyhow!(
                    "ContractionTree::from_order: step {} names invalid pair ({}, {}) for {} operands",
                    step,
                    i,
                    j,
                    len
                ));
            }
            let (a, b) = (working[i], working[j]);
            let id = n_inputs + children.len();
            children.push((a, b));
            let (hi, lo) = if i > j { (i, j) } else { (j, i) };
            working.remove(hi);
            working.remove(lo);
            working.push(id);
        }
        if working.len() != 1 {
            return Err(anyhow!(
                "ContractionTree::from_order: order left {} operands (expected exactly 1)",
                working.len()
            ));
        }
        Ok(Self { n_inputs, children })
    }

    /// Linearise the tree back into an executable positional `order`.
    ///
    /// Emits internal nodes in a deterministic topological order — repeatedly the
    /// lowest-numbered internal node whose two children are both present in the
    /// working list — translating stable node ids into the positions the executor
    /// expects.  Deterministic, so refinement is reproducible.
    ///
    /// # Complexity
    ///
    /// `O(n²)` for an `n`-leaf tree.
    fn to_order(&self) -> Result<Vec<(usize, usize)>> {
        let mut working: Vec<usize> = (0..self.n_inputs).collect();
        let mut emitted = vec![false; self.children.len()];
        let mut order = Vec::with_capacity(self.children.len());
        for _ in 0..self.children.len() {
            let mut progressed = false;
            for (t, &(a, b)) in self.children.iter().enumerate() {
                if emitted[t] {
                    continue;
                }
                let (Some(i), Some(j)) = (
                    working.iter().position(|&x| x == a),
                    working.iter().position(|&x| x == b),
                ) else {
                    continue;
                };
                order.push((i, j));
                let (hi, lo) = if i > j { (i, j) } else { (j, i) };
                working.remove(hi);
                working.remove(lo);
                working.push(self.n_inputs + t);
                emitted[t] = true;
                progressed = true;
                break;
            }
            if !progressed {
                return Err(anyhow!(
                    "ContractionTree::to_order: tree is not linearisable (dangling or cyclic node)"
                ));
            }
        }
        if working.len() != 1 {
            return Err(anyhow!(
                "ContractionTree::to_order: tree left {} operands (expected exactly 1)",
                working.len()
            ));
        }
        Ok(order)
    }

    /// The internal-node slot backing node `id`, or `None` if `id` is a leaf.
    fn internal_slot(&self, id: usize) -> Option<usize> {
        id.checked_sub(self.n_inputs)
            .filter(|t| *t < self.children.len())
    }

    /// Every neighbour of this tree under **nearest-neighbour interchange** (NNI).
    ///
    /// For each internal node `P = other ∘ inner` whose child `inner = c ∘ d` is
    /// itself internal, NNI re-associates the grandchildren, yielding the two
    /// rotations
    ///
    /// ```text
    /// other ∘ (c ∘ d)   ⟶   (other ∘ c) ∘ d       and       (other ∘ d) ∘ c
    /// ```
    ///
    /// Both children of `P` are tried as `inner`, so each internal node
    /// contributes up to four neighbours.  Every move preserves the leaf set of
    /// `P`'s subtree and keeps `inner` a child of `P`, so the result is always a
    /// well-formed binary tree over the same inputs — no move can produce an
    /// invalid contraction.  NNI is the standard local move on binary trees and
    /// is precisely the associativity rewrite that turns, say, `A·(B·C)` into
    /// `(A·B)·C` — the rewrite that actually changes a matrix chain's cost.
    ///
    /// # Complexity
    ///
    /// `O(n)` neighbours for an `n`-leaf tree, each `O(n)` to construct.
    fn nni_neighbors(&self) -> Vec<ContractionTree> {
        let mut neighbors = Vec::new();
        for p in 0..self.children.len() {
            let (left, right) = self.children[p];
            for (other, inner) in [(left, right), (right, left)] {
                let Some(q) = self.internal_slot(inner) else {
                    continue;
                };
                let (c, d) = self.children[q];
                for (keep, promote) in [(c, d), (d, c)] {
                    let mut candidate = self.clone();
                    // inner (slot q) now merges `other` with `keep`; P (slot p)
                    // merges that result with the promoted grandchild.
                    candidate.children[q] = (other, keep);
                    candidate.children[p] = (inner, promote);
                    neighbors.push(candidate);
                }
            }
        }
        neighbors
    }
}

/// Can every step of `plan` be handed to the einsum executor?
///
/// A non-final step whose output subscript is empty has contracted its operands
/// down to a scalar, and `EinsumSpec` cannot express an empty subscript, so such
/// an intermediate cannot be chained into a later step.  The stochastic planners
/// apply the same guard when sampling orders (see `randomized_greedy_order`);
/// refinement applies it when accepting one.
fn is_chainable(plan: &Plan) -> bool {
    let last = plan.nodes.len().saturating_sub(1);
    plan.nodes
        .iter()
        .enumerate()
        .all(|(step, node)| step == last || !node.output_spec.output_spec.is_empty())
}

/// Refine a plan by local search over the contraction tree.
///
/// Takes an existing plan and improves the **contraction order** — the sequence
/// of pairwise contractions the executor actually runs — by steepest-descent
/// local search over nearest-neighbour interchanges of the contraction tree.
///
/// # Algorithm
///
/// 1. Recover the contraction tree from `original_plan.order` and rebuild it via
///    [`build_plan_from_order`], so the starting cost is the order's *true* cost
///    under the current cost model rather than whatever the caller happened to
///    store in `estimated_flops`.
/// 2. Enumerate the tree's NNI neighbours (see [`ContractionTree::nni_neighbors`]),
///    rebuild a full plan for each, and read its real `estimated_flops`.
/// 3. Move to the strictly cheapest improving neighbour; stop at a local optimum
///    or after `max_iterations` descents.
///
/// Because only strict improvements are accepted, refinement is **monotone
/// non-worsening**: the returned plan never costs more than the input order does.
/// The returned plan comes straight from [`build_plan_from_order`], so its
/// `nodes`, `order` and `estimated_flops` are mutually consistent and executable.
///
/// # Cost model
///
/// `hints` (including per-input sparsity) feeds the same cost model the other
/// planners use, so refinement is sparsity-aware wherever they are.
///
/// # Fallback
///
/// If `original_plan.order` is missing or not executable for `spec`/`shapes` — it
/// has no meaningful cost to be measured against — refinement seeds from
/// [`greedy_planner`] instead. Monotonicity is stated against a *valid* input
/// order.
///
/// # Complexity
///
/// `O(max_iterations · n⁴)` for an `n`-operand network: `O(n)` neighbours per
/// descent, each rebuilt in `O(n³)`.
///
/// # Use Cases
///
/// - Post-process greedy/beam-search plans
/// - Polish the output of a metaheuristic before execution
///
/// # Errors
///
/// Returns an error if `spec` and `shapes` disagree on the operand count, or if
/// no valid plan can be built at all.
pub fn refine_plan(
    original_plan: &Plan,
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    hints: &PlanHints,
    max_iterations: usize,
) -> Result<Plan> {
    if spec.num_inputs() != shapes.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match shapes ({})",
            spec.num_inputs(),
            shapes.len()
        ));
    }

    // Score the incoming order honestly: rebuild it rather than trusting the
    // `estimated_flops` the caller supplied (which may be stale or inconsistent
    // with `nodes`). If it is not a usable order, start from greedy instead.
    let mut best_plan = match build_plan_from_order(spec, shapes, hints, &original_plan.order) {
        Ok(plan) if is_chainable(&plan) => plan,
        _ => greedy_planner(spec, shapes, hints)?,
    };
    let mut best_tree = ContractionTree::from_order(spec.num_inputs(), &best_plan.order)?;

    for _ in 0..max_iterations {
        let mut improvement: Option<(ContractionTree, Plan)> = None;
        for neighbor in best_tree.nni_neighbors() {
            let Ok(order) = neighbor.to_order() else {
                continue;
            };
            let Ok(plan) = build_plan_from_order(spec, shapes, hints, &order) else {
                continue;
            };
            if !is_chainable(&plan) {
                continue;
            }
            let incumbent = improvement
                .as_ref()
                .map_or(best_plan.estimated_flops, |(_, p)| p.estimated_flops);
            if plan.estimated_flops < incumbent {
                improvement = Some((neighbor, plan));
            }
        }
        // Steepest descent: no strictly-cheaper neighbour ⇒ local optimum.
        let Some((tree, plan)) = improvement else {
            break;
        };
        best_tree = tree;
        best_plan = plan;
    }

    Ok(best_plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::compute_output_shape;
    use std::collections::HashMap;

    /// A 4-operand chain whose cost is dominated by the contraction order.
    ///
    /// With `i = m = k = 2` and `j = l = 100`, contracting `B·C` first blows up
    /// to a 100x100 intermediate, while the left-to-right tree never leaves the
    /// small dimensions.
    fn lopsided_chain() -> (EinsumSpec, Vec<Vec<usize>>) {
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").expect("spec parses");
        let shapes = vec![vec![2, 100], vec![100, 2], vec![2, 100], vec![100, 2]];
        (spec, shapes)
    }

    /// The deliberately bad order for [`lopsided_chain`]: contract `B·C` (the
    /// 100x100 blow-up) first, then fold in `A`, then `D`.
    fn bad_order() -> Vec<(usize, usize)> {
        vec![(1, 2), (0, 2), (0, 1)]
    }

    /// Rebuild `order` into a plan (the honest cost of that order).
    fn plan_for(spec: &EinsumSpec, shapes: &[Vec<usize>], order: &[(usize, usize)]) -> Plan {
        build_plan_from_order(spec, shapes, &PlanHints::default(), order)
            .expect("order is executable")
    }

    // ========================= the headline regression =========================

    #[test]
    fn refine_strictly_improves_a_deliberately_bad_order() {
        let (spec, shapes) = lopsided_chain();
        let hints = PlanHints::default();
        let bad_plan = plan_for(&spec, &shapes, &bad_order());
        let bad_cost = bad_plan.estimated_flops;

        let refined = refine_plan(&bad_plan, &spec, &shapes, &hints, 100).expect("refine succeeds");

        // The old implementation permuted `nodes` and scored the permutation by
        // `sum(node.cost)` — invariant under permutation — so it could only ever
        // return the input cost. A strict decrease is impossible for it.
        assert!(
            refined.estimated_flops < bad_cost,
            "refinement must strictly improve a bad order: {} -> {}",
            bad_cost,
            refined.estimated_flops
        );
        // And it must actually pick a *different* tree, not just re-price one.
        assert_ne!(
            refined.order,
            bad_order(),
            "refined plan still runs the bad order"
        );
    }

    #[test]
    fn refine_reaches_the_optimal_tree_on_the_lopsided_chain() {
        let (spec, shapes) = lopsided_chain();
        let hints = PlanHints::default();
        let bad_plan = plan_for(&spec, &shapes, &bad_order());
        let refined = refine_plan(&bad_plan, &spec, &shapes, &hints, 100).expect("refine succeeds");

        // Brute-force the true optimum over every contraction tree, so the test
        // pins an absolute target rather than "somewhat better".
        let optimal = brute_force_best_cost(&spec, &shapes, &hints);
        assert!(
            (refined.estimated_flops - optimal).abs() <= optimal * 1e-9,
            "refined cost {} should reach the brute-force optimum {}",
            refined.estimated_flops,
            optimal
        );
    }

    #[test]
    fn refine_never_worsens_any_starting_order() {
        // Monotone non-worsening over *every* valid tree of the chain, not just
        // the one bad order.
        let (spec, shapes) = lopsided_chain();
        let hints = PlanHints::default();
        for order in all_valid_orders(spec.num_inputs()) {
            let Ok(start) = build_plan_from_order(&spec, &shapes, &hints, &order) else {
                continue;
            };
            if !is_chainable(&start) {
                continue;
            }
            let refined =
                refine_plan(&start, &spec, &shapes, &hints, 100).expect("refine succeeds");
            assert!(
                refined.estimated_flops <= start.estimated_flops,
                "refinement worsened {:?}: {} -> {}",
                order,
                start.estimated_flops,
                refined.estimated_flops
            );
        }
    }

    // ===================== nodes / order / flops consistency =====================

    /// The invariant the old implementation broke: `nodes`, `order` and
    /// `estimated_flops` must all describe the *same* contraction.
    fn assert_plan_self_consistent(spec: &EinsumSpec, plan: &Plan) {
        let n = spec.num_inputs();
        assert_eq!(
            plan.order.len(),
            n - 1,
            "order must have n-1 steps, got {:?}",
            plan.order
        );
        assert_eq!(
            plan.nodes.len(),
            plan.order.len(),
            "one node per contraction step"
        );

        // estimated_flops is exactly the sum of the per-step costs.
        let summed: f64 = plan.nodes.iter().map(|node| node.cost).sum();
        assert!(
            (plan.estimated_flops - summed).abs() <= summed.abs() * 1e-9,
            "estimated_flops {} disagrees with sum(node.cost) {}",
            plan.estimated_flops,
            summed
        );

        // `order` is executable, and each node's recorded input subscripts are
        // the operands that `order` actually presents at that step.
        let mut labels: Vec<String> = spec.inputs.clone();
        for (step, &(i, j)) in plan.order.iter().enumerate() {
            let len = labels.len();
            assert!(
                i != j && i < len && j < len,
                "step {step} names invalid pair ({i}, {j}) for {len} operands"
            );
            let node = &plan.nodes[step];
            assert_eq!(
                node.output_spec.input_specs,
                vec![labels[i].clone(), labels[j].clone()],
                "step {step}: node input_specs disagree with the operands `order` selects"
            );
            let produced = node.output_spec.output_spec.clone();
            let (hi, lo) = if i > j { (i, j) } else { (j, i) };
            labels.remove(hi);
            labels.remove(lo);
            labels.push(produced);
        }
        assert_eq!(labels.len(), 1, "order must reduce to a single tensor");
        assert_eq!(
            labels[0], spec.output,
            "the final step must produce the requested output subscript"
        );
    }

    #[test]
    fn refined_plan_is_self_consistent() {
        let (spec, shapes) = lopsided_chain();
        let hints = PlanHints::default();
        let bad_plan = plan_for(&spec, &shapes, &bad_order());
        let refined = refine_plan(&bad_plan, &spec, &shapes, &hints, 100).expect("refine succeeds");
        assert_plan_self_consistent(&spec, &refined);
    }

    #[test]
    fn refine_does_not_corrupt_a_plan_whose_stored_cost_is_stale() {
        // `Plan`'s fields are all public, so a caller can hand in a plan whose
        // `estimated_flops` disagrees with `sum(node.cost)` — a stale or
        // externally-supplied estimate.
        //
        // The old implementation seeded `current_cost` from `estimated_flops` but
        // scored neighbours as `sum(node.cost)`. With a stale (here: inflated)
        // estimate, the very first adjacent swap "improved", so it permuted
        // `nodes` — which encodes the contraction tree's dependency order — while
        // leaving `order` untouched, leaving the two mutually inconsistent.
        //
        // Refinement must be driven by the order's *true* rebuilt cost, so a
        // stale input estimate can never induce a bogus move.
        let spec = EinsumSpec::parse("ij,jk,kl->il").expect("spec parses");
        let shapes = vec![vec![10, 100], vec![100, 10], vec![10, 10]];
        let hints = PlanHints::default();

        let mut stale = greedy_planner(&spec, &shapes, &hints).expect("greedy plans");
        let honest_cost: f64 = stale.nodes.iter().map(|node| node.cost).sum();
        let honest_order = stale.order.clone();
        stale.estimated_flops = 1.0e18; // wildly inconsistent with `nodes`

        let refined = refine_plan(&stale, &spec, &shapes, &hints, 100).expect("refine succeeds");

        // Nodes and order must still describe the same contraction.
        assert_plan_self_consistent(&spec, &refined);
        // The stale estimate must not have bought a phantom "improvement": the
        // order is already optimal here, so it must survive untouched, priced
        // honestly rather than at the fictional 1e18.
        assert_eq!(
            refined.order, honest_order,
            "a stale cost estimate must not change the contraction order"
        );
        assert!(
            (refined.estimated_flops - honest_cost).abs() <= honest_cost * 1e-9,
            "refined cost {} should be the honest cost {}, not the stale 1e18",
            refined.estimated_flops,
            honest_cost
        );
    }

    #[test]
    fn refined_plan_is_self_consistent_across_specs() {
        let hints = PlanHints::default();
        let cases: Vec<(&str, Vec<Vec<usize>>)> = vec![
            ("ij,jk->ik", vec![vec![3, 4], vec![4, 5]]),
            ("ij,jk->ki", vec![vec![3, 4], vec![4, 5]]),
            (
                "ij,jk,kl->il",
                vec![vec![10, 100], vec![100, 10], vec![10, 10]],
            ),
            ("ij,jk,kl->li", vec![vec![3, 4], vec![4, 5], vec![5, 6]]),
            (
                "bij,bjk,bkl->bil",
                vec![vec![2, 3, 4], vec![2, 4, 5], vec![2, 5, 6]],
            ),
            (
                "ij,jk,kl,lm->im",
                vec![vec![2, 100], vec![100, 2], vec![2, 100], vec![100, 2]],
            ),
        ];
        for (spec_str, shapes) in cases {
            let spec = EinsumSpec::parse(spec_str).expect("spec parses");
            let start = greedy_planner(&spec, &shapes, &hints).expect("greedy plans");
            let refined = refine_plan(&start, &spec, &shapes, &hints, 50).expect("refine succeeds");
            assert_plan_self_consistent(&spec, &refined);
            assert!(
                refined.estimated_flops <= start.estimated_flops * (1.0 + 1e-9),
                "{spec_str}: refinement worsened greedy"
            );
            // The refined shape must still be the requested one.
            let expected = compute_output_shape(&spec, &shapes).expect("output shape");
            assert_eq!(execute_ref(&spec, &shapes, &refined).shape, expected);
        }
    }

    // ===================== semantics: same numbers, lower cost =====================

    /// A dense row-major tensor labelled by an einsum subscript. Deliberately a
    /// from-scratch reference implementation: it is the independent oracle the
    /// planner's own machinery is checked against.
    #[derive(Clone)]
    struct RefTensor {
        labels: String,
        shape: Vec<usize>,
        data: Vec<f64>,
    }

    impl RefTensor {
        /// Row-major offset of the element selected by `assignment`.
        ///
        /// Summing `stride * assignment[label]` over the tensor's own axes reads
        /// the diagonal when a label repeats, which is the correct einsum
        /// semantics for a repeated subscript.
        fn offset(&self, assignment: &HashMap<char, usize>) -> usize {
            let labels: Vec<char> = self.labels.chars().collect();
            let mut offset = 0;
            let mut stride = 1;
            for (axis, label) in labels.iter().enumerate().rev() {
                let index = *assignment.get(label).expect("label is assigned");
                offset += index * stride;
                stride *= self.shape[axis];
            }
            offset
        }
    }

    /// Odometer over every assignment of `labels`, invoking `visit` on each.
    fn for_each_assignment(
        labels: &str,
        dims: &HashMap<char, usize>,
        visit: &mut impl FnMut(&HashMap<char, usize>),
    ) {
        let labels: Vec<char> = labels.chars().collect();
        let extents: Vec<usize> = labels
            .iter()
            .map(|c| *dims.get(c).expect("label has an extent"))
            .collect();
        let total: usize = extents.iter().product();
        let mut assignment: HashMap<char, usize> = HashMap::new();
        for flat in 0..total {
            let mut rest = flat;
            for (axis, &extent) in extents.iter().enumerate().rev() {
                assignment.insert(labels[axis], rest % extent);
                rest /= extent;
            }
            visit(&assignment);
        }
    }

    /// Naive einsum over an arbitrary number of labelled operands.
    fn einsum_ref(
        operands: &[RefTensor],
        out_labels: &str,
        dims: &HashMap<char, usize>,
    ) -> RefTensor {
        let mut union = String::new();
        for operand in operands {
            for c in operand.labels.chars() {
                if !union.contains(c) {
                    union.push(c);
                }
            }
        }
        for c in out_labels.chars() {
            if !union.contains(c) {
                union.push(c);
            }
        }
        let out_shape: Vec<usize> = out_labels
            .chars()
            .map(|c| *dims.get(&c).expect("label has an extent"))
            .collect();
        let mut out = RefTensor {
            labels: out_labels.to_string(),
            shape: out_shape.clone(),
            data: vec![0.0; out_shape.iter().product::<usize>().max(1)],
        };
        for_each_assignment(&union, dims, &mut |assignment| {
            let term: f64 = operands
                .iter()
                .map(|operand| operand.data[operand.offset(assignment)])
                .product();
            let slot = out.offset(assignment);
            out.data[slot] += term;
        });
        out
    }

    /// Run a plan the way the executor does: drive `plan.order`, and contract each
    /// step with the subscripts the plan's *own* node records. A plan whose nodes
    /// and order disagree cannot produce the right numbers here.
    fn execute_ref(spec: &EinsumSpec, shapes: &[Vec<usize>], plan: &Plan) -> RefTensor {
        let dims = dim_map(spec, shapes);
        let mut operands: Vec<RefTensor> = spec
            .inputs
            .iter()
            .zip(shapes.iter())
            .map(|(labels, shape)| RefTensor {
                labels: labels.clone(),
                shape: shape.clone(),
                data: fill(shape),
            })
            .collect();
        for (step, &(i, j)) in plan.order.iter().enumerate() {
            let node = &plan.nodes[step];
            let pair = [operands[i].clone(), operands[j].clone()];
            let merged = einsum_ref(&pair, &node.output_spec.output_spec, &dims);
            let (hi, lo) = if i > j { (i, j) } else { (j, i) };
            operands.remove(hi);
            operands.remove(lo);
            operands.push(merged);
        }
        assert_eq!(operands.len(), 1, "plan order must reduce to one tensor");
        operands.pop().expect("one operand remains")
    }

    /// Extent of every index in the spec.
    fn dim_map(spec: &EinsumSpec, shapes: &[Vec<usize>]) -> HashMap<char, usize> {
        let mut dims = HashMap::new();
        for (labels, shape) in spec.inputs.iter().zip(shapes.iter()) {
            for (c, &extent) in labels.chars().zip(shape.iter()) {
                dims.insert(c, extent);
            }
        }
        dims
    }

    /// Deterministic, non-degenerate operand data (distinct values, mixed signs),
    /// so an incorrect contraction cannot coincidentally match.
    fn fill(shape: &[usize]) -> Vec<f64> {
        let total: usize = shape.iter().product();
        (0..total)
            .map(|k| ((k * 37 % 23) as f64 - 11.0) / 7.0)
            .collect()
    }

    /// The small twin of [`lopsided_chain`]: same topology and the same
    /// order-sensitivity, but with extents tiny enough to contract naively.
    fn small_lopsided_chain() -> (EinsumSpec, Vec<Vec<usize>>) {
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").expect("spec parses");
        let shapes = vec![vec![2, 5], vec![5, 2], vec![2, 5], vec![5, 2]];
        (spec, shapes)
    }

    #[test]
    fn refined_plan_preserves_semantics() {
        let (spec, shapes) = small_lopsided_chain();
        let hints = PlanHints::default();
        let bad_plan = plan_for(&spec, &shapes, &bad_order());
        let refined = refine_plan(&bad_plan, &spec, &shapes, &hints, 100).expect("refine succeeds");

        // Cost must genuinely fall...
        assert!(
            refined.estimated_flops < bad_plan.estimated_flops,
            "cost did not fall: {} -> {}",
            bad_plan.estimated_flops,
            refined.estimated_flops
        );

        // ...while the numbers must not move at all. Ground truth is a single
        // all-at-once naive einsum, independent of any plan.
        let dims = dim_map(&spec, &shapes);
        let operands: Vec<RefTensor> = spec
            .inputs
            .iter()
            .zip(shapes.iter())
            .map(|(labels, shape)| RefTensor {
                labels: labels.clone(),
                shape: shape.clone(),
                data: fill(shape),
            })
            .collect();
        let truth = einsum_ref(&operands, &spec.output, &dims);
        let before = execute_ref(&spec, &shapes, &bad_plan);
        let after = execute_ref(&spec, &shapes, &refined);

        assert_eq!(after.labels, spec.output);
        assert_eq!(after.shape, truth.shape);
        assert_eq!(before.shape, truth.shape);
        assert!(
            truth.data.iter().any(|v| v.abs() > 1e-6),
            "test data is degenerate; the comparison would be vacuous"
        );
        for (k, ((&a, &b), &t)) in after
            .data
            .iter()
            .zip(before.data.iter())
            .zip(truth.data.iter())
            .enumerate()
        {
            assert!(
                (a - t).abs() < 1e-9,
                "refined plan element {k} = {a}, expected {t}"
            );
            assert!(
                (b - t).abs() < 1e-9,
                "unrefined plan element {k} = {b}, expected {t}"
            );
        }
    }

    #[test]
    fn refined_plan_preserves_semantics_with_batch_and_permuted_output() {
        // Batch indices and a permuted output are the two subscript hazards; a
        // refined order must respect both.
        let hints = PlanHints::default();
        let cases: Vec<(&str, Vec<Vec<usize>>)> = vec![
            (
                "bij,bjk,bkl->bil",
                vec![vec![2, 3, 2], vec![2, 2, 4], vec![2, 4, 2]],
            ),
            ("ij,jk,kl->li", vec![vec![2, 5], vec![5, 2], vec![2, 4]]),
        ];
        for (spec_str, shapes) in cases {
            let spec = EinsumSpec::parse(spec_str).expect("spec parses");
            let dims = dim_map(&spec, &shapes);
            let operands: Vec<RefTensor> = spec
                .inputs
                .iter()
                .zip(shapes.iter())
                .map(|(labels, shape)| RefTensor {
                    labels: labels.clone(),
                    shape: shape.clone(),
                    data: fill(shape),
                })
                .collect();
            let truth = einsum_ref(&operands, &spec.output, &dims);

            for order in all_valid_orders(spec.num_inputs()) {
                let Ok(start) = build_plan_from_order(&spec, &shapes, &hints, &order) else {
                    continue;
                };
                if !is_chainable(&start) {
                    continue;
                }
                let refined =
                    refine_plan(&start, &spec, &shapes, &hints, 50).expect("refine succeeds");
                assert_plan_self_consistent(&spec, &refined);
                let got = execute_ref(&spec, &shapes, &refined);
                assert_eq!(got.labels, spec.output, "{spec_str}: wrong output labels");
                assert_eq!(got.shape, truth.shape, "{spec_str}: wrong output shape");
                for (k, (&a, &t)) in got.data.iter().zip(truth.data.iter()).enumerate() {
                    assert!(
                        (a - t).abs() < 1e-9,
                        "{spec_str} from order {order:?}: element {k} = {a}, expected {t}"
                    );
                }
            }
        }
    }

    // ============================ tree machinery ============================

    /// Every valid positional order over `n` operands (each step picks an ordered
    /// pair of distinct positions in the shrinking working list).
    fn all_valid_orders(n: usize) -> Vec<Vec<(usize, usize)>> {
        fn walk(
            remaining: usize,
            prefix: &mut Vec<(usize, usize)>,
            out: &mut Vec<Vec<(usize, usize)>>,
        ) {
            if remaining == 1 {
                out.push(prefix.clone());
                return;
            }
            for i in 0..remaining {
                for j in 0..remaining {
                    if i == j {
                        continue;
                    }
                    prefix.push((i, j));
                    walk(remaining - 1, prefix, out);
                    prefix.pop();
                }
            }
        }
        let mut out = Vec::new();
        walk(n, &mut Vec::new(), &mut out);
        out
    }

    /// Cheapest cost over *every* contraction order — the ground-truth optimum.
    fn brute_force_best_cost(spec: &EinsumSpec, shapes: &[Vec<usize>], hints: &PlanHints) -> f64 {
        let mut best = f64::INFINITY;
        for order in all_valid_orders(spec.num_inputs()) {
            let Ok(plan) = build_plan_from_order(spec, shapes, hints, &order) else {
                continue;
            };
            if !is_chainable(&plan) {
                continue;
            }
            best = best.min(plan.estimated_flops);
        }
        assert!(best.is_finite(), "no valid order found");
        best
    }

    #[test]
    fn tree_roundtrips_through_order() {
        // from_order ∘ to_order must be the identity on cost: the same tree, so
        // the same contraction, however it is linearised.
        let (spec, shapes) = lopsided_chain();
        let hints = PlanHints::default();
        for order in all_valid_orders(spec.num_inputs()) {
            let Ok(plan) = build_plan_from_order(&spec, &shapes, &hints, &order) else {
                continue;
            };
            let tree = ContractionTree::from_order(spec.num_inputs(), &order).expect("tree");
            let relinearised = tree.to_order().expect("linearisable");
            let rebuilt = build_plan_from_order(&spec, &shapes, &hints, &relinearised)
                .expect("relinearised order is executable");
            assert!(
                (rebuilt.estimated_flops - plan.estimated_flops).abs()
                    <= plan.estimated_flops * 1e-9,
                "relinearising {order:?} changed cost: {} -> {}",
                plan.estimated_flops,
                rebuilt.estimated_flops
            );
        }
    }

    #[test]
    fn nni_neighbors_are_all_valid_trees() {
        let (spec, shapes) = lopsided_chain();
        let hints = PlanHints::default();
        let tree = ContractionTree::from_order(spec.num_inputs(), &bad_order()).expect("tree");
        let neighbors = tree.nni_neighbors();
        assert!(!neighbors.is_empty(), "a 4-leaf tree has NNI neighbours");
        for neighbor in &neighbors {
            let order = neighbor.to_order().expect("neighbour is linearisable");
            let plan = build_plan_from_order(&spec, &shapes, &hints, &order)
                .expect("neighbour order is executable");
            assert_plan_self_consistent(&spec, &plan);
        }
    }

    #[test]
    fn from_order_rejects_invalid_orders() {
        assert!(ContractionTree::from_order(3, &[(0, 5), (0, 1)]).is_err());
        assert!(ContractionTree::from_order(3, &[(0, 0), (0, 1)]).is_err());
        // Too few steps to reduce three operands to one.
        assert!(ContractionTree::from_order(3, &[(0, 1)]).is_err());
    }

    // ============================ edge cases ============================

    #[test]
    fn refine_recovers_from_an_unusable_input_order() {
        // A plan whose `order` is empty (as some hand-built plans are) has no
        // order to refine; refinement must still return a usable, consistent plan
        // rather than propagating the emptiness.
        let (spec, shapes) = lopsided_chain();
        let hints = PlanHints::default();
        let mut broken = Plan::new();
        broken.estimated_flops = 1.0; // deliberately inconsistent with `nodes`
        let refined = refine_plan(&broken, &spec, &shapes, &hints, 50).expect("refine succeeds");
        assert_plan_self_consistent(&spec, &refined);
        assert!(refined.estimated_flops > 0.0);
    }

    #[test]
    fn refine_single_step_plan_is_a_fixed_point() {
        // Two operands admit exactly one contraction tree: nothing to refine.
        let spec = EinsumSpec::parse("ij,jk->ik").expect("spec parses");
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();
        let original = greedy_planner(&spec, &shapes, &hints).expect("greedy plans");
        let refined = refine_plan(&original, &spec, &shapes, &hints, 10).expect("refine succeeds");
        assert_eq!(refined.nodes.len(), original.nodes.len());
        assert_eq!(refined.order, original.order);
        assert_eq!(refined.estimated_flops, original.estimated_flops);
        assert_plan_self_consistent(&spec, &refined);
    }

    #[test]
    fn refine_improves_or_maintains_a_greedy_plan() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").expect("spec parses");
        let shapes = vec![vec![10, 100], vec![100, 10], vec![10, 10]];
        let hints = PlanHints::default();
        let original = greedy_planner(&spec, &shapes, &hints).expect("greedy plans");
        let refined = refine_plan(&original, &spec, &shapes, &hints, 100).expect("refine succeeds");
        assert!(
            refined.estimated_flops <= original.estimated_flops,
            "Refined cost {} should be <= original cost {}",
            refined.estimated_flops,
            original.estimated_flops
        );
        assert_plan_self_consistent(&spec, &refined);
    }

    #[test]
    fn refine_rejects_shape_mismatch() {
        let spec = EinsumSpec::parse("ij,jk->ik").expect("spec parses");
        let hints = PlanHints::default();
        let plan = Plan::new();
        assert!(refine_plan(&plan, &spec, &[vec![10, 20]], &hints, 10).is_err());
    }

    #[test]
    fn refine_with_zero_iterations_returns_a_consistent_plan() {
        // `max_iterations == 0` performs no descent, but the returned plan must
        // still be internally consistent (the old code returned the input `nodes`
        // with a cost field that need not match them).
        let (spec, shapes) = lopsided_chain();
        let hints = PlanHints::default();
        let bad_plan = plan_for(&spec, &shapes, &bad_order());
        let refined = refine_plan(&bad_plan, &spec, &shapes, &hints, 0).expect("refine succeeds");
        assert_plan_self_consistent(&spec, &refined);
        assert_eq!(refined.order, bad_order());
    }
}
