//! # AdaptivePlanner - Trait Implementations
//!
//! This module contains trait implementations for `AdaptivePlanner`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Planner`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::api::{Plan, PlanHints, Planner};
use crate::parser::EinsumSpec;
use anyhow::Result;

use super::functions::{
    beam_search_planner, dp_planner, genetic_algorithm_planner, greedy_planner,
    simulated_annealing_planner,
};
use super::types::{AdaptivePlanner, PlanningAlgorithm};

impl Default for AdaptivePlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl Planner for AdaptivePlanner {
    fn make_plan(&self, spec: &str, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
        let parsed_spec = EinsumSpec::parse(spec)?;
        let algorithm = self.select_algorithm(&parsed_spec, shapes);
        match algorithm {
            PlanningAlgorithm::Greedy => greedy_planner(&parsed_spec, shapes, hints),
            PlanningAlgorithm::DP => dp_planner(&parsed_spec, shapes, hints),
            PlanningAlgorithm::BeamSearch(width) => {
                beam_search_planner(&parsed_spec, shapes, hints, width)
            }
            PlanningAlgorithm::SimulatedAnnealing => {
                simulated_annealing_planner(&parsed_spec, shapes, hints, 1000.0, 0.95, 1000)
            }
            PlanningAlgorithm::GeneticAlgorithm => {
                // Use default GA parameters: pop=100, gen=100, mut=0.2, elite=5
                genetic_algorithm_planner(&parsed_spec, shapes, hints, 100, 100, 0.2, 5)
            }
        }
    }
}
