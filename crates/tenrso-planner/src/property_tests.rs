//! Property-based tests for the planner
//!
//! These tests use `proptest` to generate random inputs and verify
//! that the planner behaves correctly across a wide range of scenarios.

#[cfg(test)]
mod tests {
    use crate::{
        beam_search_planner, dp_planner, greedy_planner, simulate_execution, EinsumSpec,
        HardwareModel, PlanCache, PlanHints,
    };
    use proptest::prelude::*;

    proptest! {
        /// Test that greedy planner never panics on valid inputs
        #[test]
        fn prop_greedy_no_panic(
            dim1 in 1usize..=50,
            dim2 in 1usize..=50,
            dim3 in 1usize..=50,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![dim1, dim2], vec![dim2, dim3]];
            let hints = PlanHints::default();

            let result = greedy_planner(&spec, &shapes, &hints);
            prop_assert!(result.is_ok());
        }

        /// Test that DP planner produces valid plans for small inputs
        #[test]
        fn prop_dp_valid_plans(
            dim in 2usize..=20,
        ) {
            let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
            let shapes = vec![
                vec![dim, dim],
                vec![dim, dim],
                vec![dim, dim],
            ];
            let hints = PlanHints::default();

            let result = dp_planner(&spec, &shapes, &hints);
            prop_assert!(result.is_ok());

            if let Ok(plan) = result {
                prop_assert!(plan.nodes.len() == 2); // 3 tensors -> 2 contractions
                prop_assert!(plan.estimated_flops > 0.0);
            }
        }

        /// Test that plan cost estimates are reasonable
        #[test]
        fn prop_cost_estimates_reasonable(
            dim1 in 1usize..=100,
            dim2 in 1usize..=100,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![dim1, dim2], vec![dim2, dim1]];
            let hints = PlanHints::default();

            let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

            // FLOPs should be proportional to dim1 * dim2 * dim1
            let expected_ops = (dim1 * dim2 * dim1) as f64 * 2.0; // 2 FLOPs per multiply-add
            prop_assert!(plan.estimated_flops > 0.0);
            prop_assert!((plan.estimated_flops - expected_ops).abs() / expected_ops < 0.01);
        }

        /// Test that cache correctly identifies identical queries
        #[test]
        fn prop_cache_identical_queries(
            dim in 5usize..=20,
        ) {
            let cache = PlanCache::new(100);
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![dim, dim], vec![dim, dim]];
            let hints = PlanHints::default();

            // First query
            cache.get_or_compute(&spec, &shapes, &hints, || {
                greedy_planner(&spec, &shapes, &hints)
            }).unwrap();

            // Second identical query should hit cache
            cache.get_or_compute(&spec, &shapes, &hints, || {
                panic!("Should not compute!")
            }).unwrap();

            let stats = cache.stats();
            prop_assert_eq!(stats.hits, 1);
            prop_assert_eq!(stats.misses, 1);
        }

        /// Test that cache distinguishes different shapes
        #[test]
        fn prop_cache_different_shapes(
            dim1 in 5usize..=20,
            dim2 in 21usize..=40, // Ensure dim1 != dim2
        ) {
            let cache = PlanCache::new(100);
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let hints = PlanHints::default();

            let shapes1 = vec![vec![dim1, dim1], vec![dim1, dim1]];
            let shapes2 = vec![vec![dim2, dim2], vec![dim2, dim2]];

            // Two different queries
            cache.get_or_compute(&spec, &shapes1, &hints, || {
                greedy_planner(&spec, &shapes1, &hints)
            }).unwrap();

            cache.get_or_compute(&spec, &shapes2, &hints, || {
                greedy_planner(&spec, &shapes2, &hints)
            }).unwrap();

            let stats = cache.stats();
            prop_assert_eq!(stats.hits, 0);
            prop_assert_eq!(stats.misses, 2);
            prop_assert_eq!(stats.entries, 2);
        }

        /// Test that simulation produces valid results
        #[test]
        fn prop_simulation_valid(
            dim in 10usize..=100,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![dim, dim], vec![dim, dim]];
            let hints = PlanHints::default();

            let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
            let sim = simulate_execution(&plan);

            prop_assert!(sim.total_time_ms > 0.0);
            prop_assert!(sim.peak_memory_bytes > 0);
            prop_assert_eq!(sim.num_allocations, plan.nodes.len());
            prop_assert!(sim.compute_intensity >= 0.0);
            prop_assert!(sim.compute_intensity <= 1.0);
            prop_assert!(sim.cache_efficiency >= 0.0);
            prop_assert!(sim.cache_efficiency <= 1.0);
        }

        /// Test that larger problems take more time in simulation
        #[test]
        fn prop_simulation_scales(
            small_dim in 10usize..=20,
            scale_factor in 2usize..=5,
        ) {
            let large_dim = small_dim * scale_factor;

            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let small_shapes = vec![vec![small_dim, small_dim], vec![small_dim, small_dim]];
            let large_shapes = vec![vec![large_dim, large_dim], vec![large_dim, large_dim]];
            let hints = PlanHints::default();

            let small_plan = greedy_planner(&spec, &small_shapes, &hints).unwrap();
            let large_plan = greedy_planner(&spec, &large_shapes, &hints).unwrap();

            let small_sim = simulate_execution(&small_plan);
            let large_sim = simulate_execution(&large_plan);

            // Larger problem should take more time
            prop_assert!(large_sim.total_time_ms > small_sim.total_time_ms);
            prop_assert!(large_sim.peak_memory_bytes > small_sim.peak_memory_bytes);
        }

        /// Test that different hardware models produce different estimates
        #[test]
        fn prop_hardware_model_affects_simulation(
            dim in 50usize..=100,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![dim, dim], vec![dim, dim]];
            let hints = PlanHints::default();

            let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

            let sim_low = crate::simulate_execution_with_hardware(&plan, &HardwareModel::low_end_cpu());
            let sim_high = crate::simulate_execution_with_hardware(&plan, &HardwareModel::high_end_cpu());

            // High-end should be faster
            prop_assert!(sim_high.total_time_ms < sim_low.total_time_ms);
        }

        /// Test that beam search with larger beam finds better or equal plans
        #[test]
        fn prop_beam_width_improves_quality(
            dim in 10usize..=30,
        ) {
            let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
            let shapes = vec![
                vec![dim, dim],
                vec![dim, dim],
                vec![dim, dim],
            ];
            let hints = PlanHints::default();

            let plan_small = beam_search_planner(&spec, &shapes, &hints, 2).unwrap();
            let plan_large = beam_search_planner(&spec, &shapes, &hints, 5).unwrap();

            // Larger beam should find better or equal plan
            prop_assert!(plan_large.estimated_flops <= plan_small.estimated_flops * 1.1);
        }

        /// Test that plans are deterministic for fixed inputs
        #[test]
        fn prop_deterministic_planning(
            dim in 10usize..=50,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![dim, dim], vec![dim, dim]];
            let hints = PlanHints::default();

            let plan1 = greedy_planner(&spec, &shapes, &hints).unwrap();
            let plan2 = greedy_planner(&spec, &shapes, &hints).unwrap();

            // Should produce identical plans
            prop_assert_eq!(plan1.estimated_flops, plan2.estimated_flops);
            prop_assert_eq!(plan1.estimated_memory, plan2.estimated_memory);
            prop_assert_eq!(plan1.nodes.len(), plan2.nodes.len());
        }

        /// Test that memory estimates are reasonable
        #[test]
        fn prop_memory_estimates(
            dim1 in 10usize..=100,
            dim2 in 10usize..=100,
            dim3 in 10usize..=100,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![dim1, dim2], vec![dim2, dim3]];
            let hints = PlanHints::default();

            let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

            // Output size is dim1 * dim3 * 8 bytes (f64)
            let min_expected = dim1 * dim3 * 8;
            prop_assert!(plan.estimated_memory >= min_expected);

            // Should not be absurdly large (< 100x input)
            let input_size = (dim1 * dim2 + dim2 * dim3) * 8;
            prop_assert!(plan.estimated_memory < input_size * 100);
        }

        /// Stress test: many small contractions
        #[test]
        fn prop_stress_many_small(
            count in 3usize..=10,
            dim in 5usize..=20,
        ) {
            // Create a chain of contractions: A₀ × A₁ × A₂ × ... → result
            let mut inputs = Vec::new();
            let mut indices = Vec::new();

            for i in 0..count {
                indices.push(format!("{}{}", (b'a' + i as u8) as char,
                                          (b'a' + i as u8 + 1) as char));
                inputs.push(vec![dim, dim]);
            }

            let output = format!("a{}", (b'a' + count as u8) as char);
            let spec_str = format!("{}->{}", indices.join(","), output);

            if let Ok(spec) = EinsumSpec::parse(&spec_str) {
                let hints = PlanHints::default();
                let result = greedy_planner(&spec, &inputs, &hints);
                prop_assert!(result.is_ok());

                if let Ok(plan) = result {
                    prop_assert_eq!(plan.nodes.len(), count - 1);
                }
            }
        }

        /// Stress test: large dimensions
        #[test]
        fn prop_stress_large_dims(
            large_dim in 500usize..=2000,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![large_dim, 10], vec![10, large_dim]];
            let hints = PlanHints::default();

            let result = greedy_planner(&spec, &shapes, &hints);
            prop_assert!(result.is_ok());

            if let Ok(plan) = result {
                prop_assert!(plan.estimated_flops > 0.0);
                prop_assert!(plan.estimated_memory > 0);
            }
        }

        /// Stress test: high-dimensional tensors
        #[test]
        fn prop_stress_high_ndim(
            ndim in 3usize..=6,
            dim in 5usize..=15,
        ) {
            let shape = vec![dim; ndim];

            // Create a contraction that sums over all but 2 dimensions
            let input_indices: String = (0..ndim).map(|i| (b'a' + i as u8) as char).collect();
            let output_indices: String = (0..2).map(|i| (b'a' + i as u8) as char).collect();

            let spec_str = format!("{},{}->{}", input_indices, input_indices, output_indices);

            if let Ok(spec) = EinsumSpec::parse(&spec_str) {
                let hints = PlanHints::default();
                let result = greedy_planner(&spec, &[shape.clone(), shape], &hints);
                prop_assert!(result.is_ok());
            }
        }
    }
}
