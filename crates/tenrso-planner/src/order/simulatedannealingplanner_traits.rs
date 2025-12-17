//! # SimulatedAnnealingPlanner - Trait Implementations
//!
//! This module contains trait implementations for `SimulatedAnnealingPlanner`.
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

use super::functions::simulated_annealing_planner;
use super::types::SimulatedAnnealingPlanner;

impl Default for SimulatedAnnealingPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl Planner for SimulatedAnnealingPlanner {
    fn make_plan(&self, spec: &str, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
        let parsed_spec = EinsumSpec::parse(spec)?;
        simulated_annealing_planner(
            &parsed_spec,
            shapes,
            hints,
            self.initial_temp,
            self.cooling_rate,
            self.max_iterations,
        )
    }
}
