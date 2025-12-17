//! # BeamSearchPlanner - Trait Implementations
//!
//! This module contains trait implementations for `BeamSearchPlanner`.
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

use super::functions::beam_search_planner;
use super::types::BeamSearchPlanner;

impl Default for BeamSearchPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl Planner for BeamSearchPlanner {
    fn make_plan(&self, spec: &str, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
        let parsed_spec = EinsumSpec::parse(spec)?;
        beam_search_planner(&parsed_spec, shapes, hints, self.beam_width)
    }
}
