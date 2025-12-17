//! Public planner API
//!
//! Provides the main interface for tensor contraction planning.

use anyhow::Result;
use std::collections::HashMap;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A planned tensor contraction operation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Plan {
    /// Sequence of contraction steps
    pub nodes: Vec<PlanNode>,
    /// Estimated total FLOPs
    pub estimated_flops: f64,
    /// Estimated peak memory (bytes)
    pub estimated_memory: usize,
    /// Contraction order (input indices to contract)
    pub order: Vec<(usize, usize)>,
}

impl Plan {
    /// Create a new empty plan
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            estimated_flops: 0.0,
            estimated_memory: 0,
            order: Vec::new(),
        }
    }

    /// Get the contraction order
    pub fn order(&self) -> &[(usize, usize)] {
        &self.order
    }

    /// Get estimated FLOPs
    pub fn estimated_flops(&self) -> f64 {
        self.estimated_flops
    }

    /// Get estimated memory usage
    pub fn estimated_memory(&self) -> usize {
        self.estimated_memory
    }
}

impl Default for Plan {
    fn default() -> Self {
        Self::new()
    }
}

impl Plan {
    /// Compare this plan with another plan
    ///
    /// Returns a [`PlanComparison`] with detailed metrics.
    pub fn compare(&self, other: &Plan) -> PlanComparison {
        PlanComparison {
            flops_ratio: other.estimated_flops / self.estimated_flops.max(1.0),
            flops_difference: other.estimated_flops - self.estimated_flops,
            memory_ratio: other.estimated_memory as f64 / (self.estimated_memory.max(1)) as f64,
            memory_difference: other.estimated_memory as i64 - self.estimated_memory as i64,
            steps_difference: other.nodes.len() as i64 - self.nodes.len() as i64,
        }
    }

    /// Analyze this plan and return detailed statistics
    pub fn analyze(&self) -> PlanAnalysis {
        let mut max_cost_step: f64 = 0.0;
        let mut min_cost_step: f64 = f64::INFINITY;
        let mut total_cost: f64 = 0.0;
        let mut max_memory_step: usize = 0;
        let mut min_memory_step: usize = usize::MAX;

        for node in &self.nodes {
            total_cost += node.cost;
            max_cost_step = max_cost_step.max(node.cost);
            min_cost_step = min_cost_step.min(node.cost);
            max_memory_step = max_memory_step.max(node.memory);
            min_memory_step = min_memory_step.min(node.memory);
        }

        let avg_cost_per_step = if !self.nodes.is_empty() {
            total_cost / self.nodes.len() as f64
        } else {
            0.0
        };

        let avg_memory_per_step = if !self.nodes.is_empty() {
            self.nodes.iter().map(|n| n.memory).sum::<usize>() / self.nodes.len()
        } else {
            0
        };

        PlanAnalysis {
            num_steps: self.nodes.len(),
            total_flops: self.estimated_flops,
            peak_memory: self.estimated_memory,
            avg_cost_per_step,
            max_cost_step,
            min_cost_step: if min_cost_step == f64::INFINITY {
                0.0
            } else {
                min_cost_step
            },
            avg_memory_per_step,
            max_memory_step,
            min_memory_step: if min_memory_step == usize::MAX {
                0
            } else {
                min_memory_step
            },
        }
    }

    /// Check if this plan is better than another based on a metric
    pub fn is_better_than(&self, other: &Plan, metric: PlanMetric) -> bool {
        match metric {
            PlanMetric::FLOPs => self.estimated_flops < other.estimated_flops,
            PlanMetric::Memory => self.estimated_memory < other.estimated_memory,
            PlanMetric::Steps => self.nodes.len() < other.nodes.len(),
            PlanMetric::Combined => {
                // Weighted combination: 70% FLOPs, 30% memory
                let self_score = self.estimated_flops * 0.7 + (self.estimated_memory as f64) * 0.3;
                let other_score =
                    other.estimated_flops * 0.7 + (other.estimated_memory as f64) * 0.3;
                self_score < other_score
            }
        }
    }
}

/// Comparison between two plans
#[derive(Debug, Clone, PartialEq)]
pub struct PlanComparison {
    /// Ratio of FLOPs (other / self)
    pub flops_ratio: f64,
    /// Absolute difference in FLOPs (other - self)
    pub flops_difference: f64,
    /// Ratio of memory usage (other / self)
    pub memory_ratio: f64,
    /// Absolute difference in memory (other - self)
    pub memory_difference: i64,
    /// Difference in number of steps (other - self)
    pub steps_difference: i64,
}

impl PlanComparison {
    /// Get improvement percentage for FLOPs
    ///
    /// Positive value means the second plan is worse (uses more FLOPs)
    /// Negative value means the second plan is better (uses fewer FLOPs)
    pub fn flops_improvement_percent(&self) -> f64 {
        (self.flops_ratio - 1.0) * 100.0
    }

    /// Get improvement percentage for memory
    pub fn memory_improvement_percent(&self) -> f64 {
        (self.memory_ratio - 1.0) * 100.0
    }
}

/// Detailed analysis of a plan
#[derive(Debug, Clone, PartialEq)]
pub struct PlanAnalysis {
    /// Number of contraction steps
    pub num_steps: usize,
    /// Total estimated FLOPs
    pub total_flops: f64,
    /// Peak memory usage
    pub peak_memory: usize,
    /// Average cost per step
    pub avg_cost_per_step: f64,
    /// Maximum cost in a single step
    pub max_cost_step: f64,
    /// Minimum cost in a single step
    pub min_cost_step: f64,
    /// Average memory per step
    pub avg_memory_per_step: usize,
    /// Maximum memory in a single step
    pub max_memory_step: usize,
    /// Minimum memory in a single step
    pub min_memory_step: usize,
}

/// Metrics for comparing plans
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanMetric {
    /// Compare by FLOPs
    FLOPs,
    /// Compare by peak memory
    Memory,
    /// Compare by number of steps
    Steps,
    /// Combined metric (weighted FLOPs + memory)
    Combined,
}

/// A single contraction step in the plan
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PlanNode {
    /// Indices of inputs to contract (temporary indices if intermediate)
    pub inputs: Vec<usize>,
    /// Output specification
    pub output_spec: ContractionSpec,
    /// Estimated cost (FLOPs) for this contraction
    pub cost: f64,
    /// Intermediate result size (bytes)
    pub memory: usize,
    /// Representation hint (dense/sparse)
    pub repr: ReprHint,
}

/// Specification for a single contraction
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ContractionSpec {
    /// Input subscript indices (e.g., ["ijk", "jkl"])
    pub input_specs: Vec<String>,
    /// Output subscript indices (e.g., "il")
    pub output_spec: String,
}

impl ContractionSpec {
    /// Create a new contraction specification
    pub fn new(input_specs: Vec<String>, output_spec: String) -> Self {
        Self {
            input_specs,
            output_spec,
        }
    }
}

/// Representation hint for a tensor operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ReprHint {
    /// Dense representation
    Dense,
    /// Sparse representation (COO/CSR/CSC)
    Sparse,
    /// Low-rank representation (CP/Tucker)
    LowRank,
    /// Let planner decide
    Auto,
}

/// Hints for the planner
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PlanHints {
    /// Prefer memory efficiency over speed
    pub minimize_memory: bool,
    /// Allow out-of-core execution
    pub allow_ooc: bool,
    /// Target device (CPU only for v0)
    pub device: Device,
    /// Memory budget (bytes), None = unlimited
    pub memory_budget: Option<usize>,
    /// Sparsity thresholds for each input
    pub sparsity_hints: HashMap<usize, f64>,
}

impl Default for PlanHints {
    fn default() -> Self {
        Self {
            minimize_memory: false,
            allow_ooc: false,
            device: Device::Cpu,
            memory_budget: None,
            sparsity_hints: HashMap::new(),
        }
    }
}

/// Target device for execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Device {
    /// CPU execution
    Cpu,
    /// GPU execution (future)
    Gpu,
}

/// Main planner trait
pub trait Planner {
    /// Create a contraction plan from an einsum specification
    ///
    /// # Arguments
    ///
    /// * `spec` - Einstein summation specification (e.g., "ijk,jkl->il")
    /// * `shapes` - Shapes of input tensors
    /// * `hints` - Planning hints and preferences
    ///
    /// # Returns
    ///
    /// A `Plan` containing the contraction sequence and cost estimates
    fn make_plan(&self, spec: &str, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan>;
}

// Display implementations for pretty-printing

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Contraction Plan:")?;
        writeln!(f, "  Steps: {}", self.nodes.len())?;
        writeln!(f, "  Estimated FLOPs: {:.2e}", self.estimated_flops)?;
        writeln!(
            f,
            "  Peak Memory: {} bytes ({:.2} MB)",
            self.estimated_memory,
            self.estimated_memory as f64 / 1_048_576.0
        )?;
        writeln!(f)?;

        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(f, "Step {}:", i + 1)?;
            writeln!(f, "  {}", node)?;
        }

        Ok(())
    }
}

impl fmt::Display for PlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Contract ")?;
        for (i, input_spec) in self.output_spec.input_specs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "'{}'", input_spec)?;
        }
        writeln!(f, " -> '{}'", self.output_spec.output_spec)?;
        writeln!(f, "    Cost: {:.2e} FLOPs", self.cost)?;
        writeln!(f, "    Memory: {} bytes", self.memory)?;
        write!(f, "    Representation: {:?}", self.repr)?;

        Ok(())
    }
}

impl fmt::Display for ReprHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReprHint::Dense => write!(f, "Dense"),
            ReprHint::Sparse => write!(f, "Sparse"),
            ReprHint::LowRank => write!(f, "LowRank"),
            ReprHint::Auto => write!(f, "Auto"),
        }
    }
}

impl Plan {
    /// Export the plan as a Graphviz DOT format string for visualization
    ///
    /// The resulting string can be saved to a `.dot` file and rendered with:
    /// ```bash
    /// dot -Tpng plan.dot -o plan.png
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};
    ///
    /// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    /// let shapes = vec![vec![10, 20], vec![20, 30]];
    /// let hints = PlanHints::default();
    /// let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
    ///
    /// let dot = plan.to_dot();
    /// assert!(dot.contains("digraph"));
    /// ```
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph ContractionPlan {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, style=rounded];\n\n");

        // Add metadata
        dot.push_str(&format!(
            "  label=\"Contraction Plan\\nSteps: {}\\nFLOPs: {:.2e}\\nMemory: {:.2} MB\";\n",
            self.nodes.len(),
            self.estimated_flops,
            self.estimated_memory as f64 / 1_048_576.0
        ));
        dot.push_str("  labelloc=t;\n\n");

        // Add nodes
        for (i, node) in self.nodes.iter().enumerate() {
            let inputs_str = node.output_spec.input_specs.join(", ");
            let label = format!(
                "Step {}\\n{}\\n→ {}\\nCost: {:.2e}\\nRepr: {:?}",
                i + 1,
                inputs_str,
                node.output_spec.output_spec,
                node.cost,
                node.repr
            );
            dot.push_str(&format!("  step{} [label=\"{}\"];\n", i, label));
        }

        // Add edges (sequential flow)
        for i in 0..self.nodes.len().saturating_sub(1) {
            dot.push_str(&format!("  step{} -> step{};\n", i, i + 1));
        }

        dot.push_str("}\n");
        dot
    }

    /// Validate the plan for correctness
    ///
    /// Checks:
    /// - All contraction steps have valid specifications
    /// - No self-contractions
    /// - Cost and memory estimates are non-negative
    ///
    /// # Returns
    ///
    /// `Ok(())` if the plan is valid, otherwise an error describing the issue
    ///
    /// # Example
    ///
    /// ```
    /// use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};
    ///
    /// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
    /// let shapes = vec![vec![10, 20], vec![20, 30]];
    /// let hints = PlanHints::default();
    /// let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
    ///
    /// assert!(plan.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<()> {
        // Check for empty plan
        if self.nodes.is_empty() && self.estimated_flops > 0.0 {
            anyhow::bail!("Plan has no steps but non-zero FLOPs estimate");
        }

        // Validate each node
        for (i, node) in self.nodes.iter().enumerate() {
            // Check for empty input specs
            if node.output_spec.input_specs.is_empty() {
                anyhow::bail!("Step {}: No input specifications", i + 1);
            }

            // Check for self-contraction (same input spec twice)
            if node.output_spec.input_specs.len() == 2
                && node.output_spec.input_specs[0] == node.output_spec.input_specs[1]
            {
                anyhow::bail!(
                    "Step {}: Self-contraction detected (same input twice)",
                    i + 1
                );
            }

            // Check cost is non-negative
            if node.cost < 0.0 {
                anyhow::bail!("Step {}: Negative cost {}", i + 1, node.cost);
            }

            // Check cost is finite
            if !node.cost.is_finite() {
                anyhow::bail!("Step {}: Non-finite cost {}", i + 1, node.cost);
            }

            // Check output spec is not empty
            if node.output_spec.output_spec.is_empty() {
                anyhow::bail!("Step {}: Empty output specification", i + 1);
            }

            // Check all characters in specs are valid
            for input_spec in &node.output_spec.input_specs {
                if !input_spec.chars().all(|c| c.is_ascii_lowercase()) {
                    anyhow::bail!(
                        "Step {}: Invalid characters in input spec '{}'",
                        i + 1,
                        input_spec
                    );
                }
            }

            if !node
                .output_spec
                .output_spec
                .chars()
                .all(|c| c.is_ascii_lowercase())
            {
                anyhow::bail!(
                    "Step {}: Invalid characters in output spec '{}'",
                    i + 1,
                    node.output_spec.output_spec
                );
            }
        }

        // Check total FLOPs is non-negative
        if self.estimated_flops < 0.0 {
            anyhow::bail!("Plan has negative total FLOPs: {}", self.estimated_flops);
        }

        // Check total FLOPs is finite
        if !self.estimated_flops.is_finite() {
            anyhow::bail!("Plan has non-finite total FLOPs: {}", self.estimated_flops);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_creation() {
        let plan = Plan::new();
        assert_eq!(plan.nodes.len(), 0);
        assert_eq!(plan.estimated_flops, 0.0);
        assert_eq!(plan.estimated_memory, 0);
    }

    #[test]
    fn test_plan_hints_default() {
        let hints = PlanHints::default();
        assert!(!hints.minimize_memory);
        assert!(!hints.allow_ooc);
        assert_eq!(hints.device, Device::Cpu);
        assert!(hints.memory_budget.is_none());
    }

    #[test]
    fn test_contraction_spec() {
        let spec =
            ContractionSpec::new(vec!["ijk".to_string(), "jkl".to_string()], "il".to_string());
        assert_eq!(spec.input_specs.len(), 2);
        assert_eq!(spec.output_spec, "il");
    }

    #[test]
    fn test_plan_compare() {
        let mut plan1 = Plan::new();
        plan1.estimated_flops = 1000.0;
        plan1.estimated_memory = 5000;

        let mut plan2 = Plan::new();
        plan2.estimated_flops = 1500.0;
        plan2.estimated_memory = 6000;

        let comparison = plan1.compare(&plan2);

        assert_eq!(comparison.flops_ratio, 1.5);
        assert_eq!(comparison.flops_difference, 500.0);
        assert_eq!(comparison.memory_difference, 1000);
        assert_eq!(comparison.flops_improvement_percent(), 50.0);
    }

    #[test]
    fn test_plan_analyze() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["ij".to_string(), "jk".to_string()],
                "ik".to_string(),
            ),
            cost: 100.0,
            memory: 1000,
            repr: ReprHint::Dense,
        });
        plan.nodes.push(PlanNode {
            inputs: vec![2, 3],
            output_spec: ContractionSpec::new(
                vec!["kl".to_string(), "lm".to_string()],
                "km".to_string(),
            ),
            cost: 200.0,
            memory: 2000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 300.0;
        plan.estimated_memory = 2000;

        let analysis = plan.analyze();

        assert_eq!(analysis.num_steps, 2);
        assert_eq!(analysis.total_flops, 300.0);
        assert_eq!(analysis.peak_memory, 2000);
        assert_eq!(analysis.avg_cost_per_step, 150.0);
        assert_eq!(analysis.max_cost_step, 200.0);
        assert_eq!(analysis.min_cost_step, 100.0);
        assert_eq!(analysis.avg_memory_per_step, 1500);
        assert_eq!(analysis.max_memory_step, 2000);
        assert_eq!(analysis.min_memory_step, 1000);
    }

    #[test]
    fn test_plan_is_better_than_flops() {
        let mut plan1 = Plan::new();
        plan1.estimated_flops = 1000.0;
        plan1.estimated_memory = 5000;

        let mut plan2 = Plan::new();
        plan2.estimated_flops = 1500.0;
        plan2.estimated_memory = 4000;

        assert!(plan1.is_better_than(&plan2, PlanMetric::FLOPs));
        assert!(!plan1.is_better_than(&plan2, PlanMetric::Memory));
    }

    #[test]
    fn test_plan_is_better_than_combined() {
        let mut plan1 = Plan::new();
        plan1.estimated_flops = 1000.0;
        plan1.estimated_memory = 5000;

        let mut plan2 = Plan::new();
        plan2.estimated_flops = 1100.0;
        plan2.estimated_memory = 3000;

        // Plan1 score: 1000*0.7 + 5000*0.3 = 700 + 1500 = 2200
        // Plan2 score: 1100*0.7 + 3000*0.3 = 770 + 900 = 1670
        // Plan2 is better (lower score)
        assert!(!plan1.is_better_than(&plan2, PlanMetric::Combined));
    }

    #[test]
    fn test_plan_analyze_empty() {
        let plan = Plan::new();
        let analysis = plan.analyze();

        assert_eq!(analysis.num_steps, 0);
        assert_eq!(analysis.total_flops, 0.0);
        assert_eq!(analysis.avg_cost_per_step, 0.0);
        assert_eq!(analysis.min_cost_step, 0.0);
        assert_eq!(analysis.min_memory_step, 0);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_plan_serialization() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["ij".to_string(), "jk".to_string()],
                "ik".to_string(),
            ),
            cost: 12000.0,
            memory: 240000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 12000.0;
        plan.estimated_memory = 240000;

        // Serialize to JSON
        let json = serde_json::to_string(&plan).unwrap();
        assert!(json.contains("estimated_flops"));
        assert!(json.contains("12000"));

        // Deserialize from JSON
        let deserialized: Plan = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.nodes.len(), 1);
        assert_eq!(deserialized.estimated_flops, 12000.0);
        assert_eq!(deserialized.estimated_memory, 240000);
        assert_eq!(deserialized.nodes[0].cost, 12000.0);
        assert_eq!(deserialized.nodes[0].memory, 240000);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_plan_hints_serialization() {
        let hints = PlanHints {
            minimize_memory: true,
            memory_budget: Some(1_000_000),
            device: Device::Cpu,
            ..Default::default()
        };

        // Serialize to JSON
        let json = serde_json::to_string(&hints).unwrap();
        assert!(json.contains("minimize_memory"));
        assert!(json.contains("true"));

        // Deserialize from JSON
        let deserialized: PlanHints = serde_json::from_str(&json).unwrap();

        assert!(deserialized.minimize_memory);
        assert_eq!(deserialized.memory_budget, Some(1_000_000));
        assert_eq!(deserialized.device, Device::Cpu);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_repr_hint_serialization() {
        let hints = vec![
            ReprHint::Dense,
            ReprHint::Sparse,
            ReprHint::LowRank,
            ReprHint::Auto,
        ];

        for hint in hints {
            let json = serde_json::to_string(&hint).unwrap();
            let deserialized: ReprHint = serde_json::from_str(&json).unwrap();
            assert_eq!(hint, deserialized);
        }
    }

    #[test]
    fn test_plan_display() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["ij".to_string(), "jk".to_string()],
                "ik".to_string(),
            ),
            cost: 12000.0,
            memory: 240000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 12000.0;
        plan.estimated_memory = 240000;

        let display_str = format!("{}", plan);
        assert!(display_str.contains("Contraction Plan"));
        assert!(display_str.contains("Steps: 1"));
        assert!(display_str.contains("1.20e4"));
        assert!(display_str.contains("240000"));
        assert!(display_str.contains("Step 1:"));
        assert!(display_str.contains("'ij'"));
        assert!(display_str.contains("'jk'"));
        assert!(display_str.contains("'ik'"));
    }

    #[test]
    fn test_plan_node_display() {
        let node = PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["ijk".to_string(), "jkl".to_string()],
                "il".to_string(),
            ),
            cost: 5000.0,
            memory: 10000,
            repr: ReprHint::Sparse,
        };

        let display_str = format!("{}", node);
        assert!(display_str.contains("Contract"));
        assert!(display_str.contains("'ijk'"));
        assert!(display_str.contains("'jkl'"));
        assert!(display_str.contains("'il'"));
        assert!(display_str.contains("5.00e3"));
        assert!(display_str.contains("10000"));
        assert!(display_str.contains("Sparse"));
    }

    #[test]
    fn test_repr_hint_display() {
        assert_eq!(format!("{}", ReprHint::Dense), "Dense");
        assert_eq!(format!("{}", ReprHint::Sparse), "Sparse");
        assert_eq!(format!("{}", ReprHint::LowRank), "LowRank");
        assert_eq!(format!("{}", ReprHint::Auto), "Auto");
    }

    #[test]
    fn test_plan_to_dot() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["ij".to_string(), "jk".to_string()],
                "ik".to_string(),
            ),
            cost: 1000.0,
            memory: 5000,
            repr: ReprHint::Dense,
        });
        plan.nodes.push(PlanNode {
            inputs: vec![2, 3],
            output_spec: ContractionSpec::new(
                vec!["ik".to_string(), "kl".to_string()],
                "il".to_string(),
            ),
            cost: 2000.0,
            memory: 10000,
            repr: ReprHint::Sparse,
        });
        plan.estimated_flops = 3000.0;
        plan.estimated_memory = 10000;

        let dot = plan.to_dot();

        // Check structure
        assert!(dot.starts_with("digraph ContractionPlan"));
        assert!(dot.contains("rankdir=TB"));
        assert!(dot.contains("shape=box"));

        // Check metadata
        assert!(dot.contains("Steps: 2"));
        assert!(dot.contains("3.00e3"));

        // Check nodes
        assert!(dot.contains("step0"));
        assert!(dot.contains("step1"));
        assert!(dot.contains("ij, jk"));
        assert!(dot.contains("ik, kl"));
        assert!(dot.contains("→ ik"));
        assert!(dot.contains("→ il"));

        // Check edges
        assert!(dot.contains("step0 -> step1"));

        // Check it ends properly
        assert!(dot.trim().ends_with("}"));
    }

    #[test]
    fn test_plan_to_dot_empty() {
        let plan = Plan::new();
        let dot = plan.to_dot();

        assert!(dot.contains("digraph ContractionPlan"));
        assert!(dot.contains("Steps: 0"));
        assert!(dot.contains("0.00e0"));
    }

    #[test]
    fn test_plan_validate_valid() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["ij".to_string(), "jk".to_string()],
                "ik".to_string(),
            ),
            cost: 1000.0,
            memory: 5000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 1000.0;
        plan.estimated_memory = 5000;

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_plan_validate_empty() {
        let plan = Plan::new();
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_plan_validate_negative_cost() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["ij".to_string(), "jk".to_string()],
                "ik".to_string(),
            ),
            cost: -1000.0, // Invalid
            memory: 5000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 1000.0;

        let result = plan.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Negative cost"));
    }

    #[test]
    fn test_plan_validate_infinite_cost() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["ij".to_string(), "jk".to_string()],
                "ik".to_string(),
            ),
            cost: f64::INFINITY, // Invalid
            memory: 5000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 1000.0;

        let result = plan.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Non-finite cost"));
    }

    #[test]
    fn test_plan_validate_empty_input_specs() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![],
            output_spec: ContractionSpec::new(
                vec![], // Invalid - no inputs
                "ik".to_string(),
            ),
            cost: 1000.0,
            memory: 5000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 1000.0;

        let result = plan.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No input specifications"));
    }

    #[test]
    fn test_plan_validate_invalid_characters() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["IJ".to_string(), "jk".to_string()], // Invalid - uppercase
                "ik".to_string(),
            ),
            cost: 1000.0,
            memory: 5000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 1000.0;

        let result = plan.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid characters"));
    }

    #[test]
    fn test_plan_validate_empty_output_spec() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 1],
            output_spec: ContractionSpec::new(
                vec!["ij".to_string(), "jk".to_string()],
                "".to_string(), // Invalid - empty output
            ),
            cost: 1000.0,
            memory: 5000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 1000.0;

        let result = plan.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Empty output specification"));
    }

    #[test]
    fn test_plan_validate_self_contraction() {
        let mut plan = Plan::new();
        plan.nodes.push(PlanNode {
            inputs: vec![0, 0], // Same input twice
            output_spec: ContractionSpec::new(
                vec!["ij".to_string(), "ij".to_string()], // Self-contraction
                "ij".to_string(),
            ),
            cost: 1000.0,
            memory: 5000,
            repr: ReprHint::Dense,
        });
        plan.estimated_flops = 1000.0;

        let result = plan.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Self-contraction"));
    }
}
