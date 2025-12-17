//! # DPPlanner - Trait Implementations
//!
//! This module contains trait implementations for `DPPlanner`.
//!
//! ## Implemented Traits
//!
//! - `Planner`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::api::{Plan, PlanHints, Planner};
use crate::parser::EinsumSpec;
use anyhow::Result;

use super::functions::dp_planner;
use super::types::DPPlanner;

impl Planner for DPPlanner {
    fn make_plan(&self, spec: &str, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
        let parsed_spec = EinsumSpec::parse(spec)?;
        dp_planner(&parsed_spec, shapes, hints)
    }
}
