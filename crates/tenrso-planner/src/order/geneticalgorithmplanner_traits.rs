//! Planner trait implementation for GeneticAlgorithmPlanner

use crate::api::{Plan, PlanHints, Planner};
use crate::parser::EinsumSpec;
use anyhow::Result;

use super::functions::genetic_algorithm_planner;
use super::types::GeneticAlgorithmPlanner;

impl Planner for GeneticAlgorithmPlanner {
    fn make_plan(&self, spec: &str, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
        let spec = EinsumSpec::parse(spec)?;
        genetic_algorithm_planner(
            &spec,
            shapes,
            hints,
            self.population_size,
            self.max_generations,
            self.mutation_rate,
            self.elitism_count,
        )
    }
}

impl Default for GeneticAlgorithmPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ga_planner_struct() {
        let planner = GeneticAlgorithmPlanner::new();
        let hints = PlanHints::default();
        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();

        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }

    #[test]
    fn test_ga_planner_custom_params() {
        let planner = GeneticAlgorithmPlanner::with_params(50, 50, 0.3, 3);
        assert_eq!(planner.population_size, 50);
        assert_eq!(planner.max_generations, 50);
        assert_eq!(planner.mutation_rate, 0.3);
        assert_eq!(planner.elitism_count, 3);
    }

    #[test]
    fn test_ga_planner_fast() {
        let planner = GeneticAlgorithmPlanner::fast();
        assert_eq!(planner.population_size, 50);
        assert_eq!(planner.max_generations, 50);
    }

    #[test]
    fn test_ga_planner_high_quality() {
        let planner = GeneticAlgorithmPlanner::high_quality();
        assert_eq!(planner.population_size, 200);
        assert_eq!(planner.max_generations, 200);
    }

    #[test]
    fn test_ga_planner_chain() {
        let planner = GeneticAlgorithmPlanner::new();
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
    fn test_ga_planner_default() {
        let planner1 = GeneticAlgorithmPlanner::new();
        let planner2 = GeneticAlgorithmPlanner::default();

        assert_eq!(planner1.population_size, planner2.population_size);
        assert_eq!(planner1.max_generations, planner2.max_generations);
        assert_eq!(planner1.mutation_rate, planner2.mutation_rate);
        assert_eq!(planner1.elitism_count, planner2.elitism_count);
    }
}
