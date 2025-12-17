//! # GreedyPlanner - Trait Implementations
//!
//! This module contains trait implementations for `GreedyPlanner`.
//!
//! ## Implemented Traits
//!
//! - `Planner`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::api::{Plan, PlanHints, Planner};
use crate::parser::EinsumSpec;
use anyhow::Result;

use super::functions::greedy_planner;
use super::types::GreedyPlanner;

impl Planner for GreedyPlanner {
    fn make_plan(&self, spec: &str, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
        let parsed_spec = EinsumSpec::parse(spec)?;
        greedy_planner(&parsed_spec, shapes, hints)
    }
}
