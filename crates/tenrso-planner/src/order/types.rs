//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// Import for doc links (unused in code, used in doc comments)
#[allow(unused_imports)]
use crate::api::Planner;
use crate::cost::TensorStats;
use crate::parser::EinsumSpec;

/// Represents an intermediate tensor in the contraction sequence
#[derive(Debug, Clone)]
pub(super) struct IntermediateTensor {
    /// Indices for this tensor (e.g., "ijk")
    pub(super) indices: String,
    /// Shape of this tensor
    pub(super) shape: Vec<usize>,
    /// Original input index (if this is an original input)
    pub(super) original_idx: Option<usize>,
    /// Statistics for cost estimation
    pub(super) stats: TensorStats,
}
impl IntermediateTensor {
    /// Create from an original input
    pub(super) fn from_input(indices: String, shape: Vec<usize>, original_idx: usize) -> Self {
        let stats = TensorStats::dense(shape.clone());
        Self {
            indices,
            shape,
            original_idx: Some(original_idx),
            stats,
        }
    }
    /// Create an intermediate tensor from a contraction
    pub(super) fn from_contraction(indices: String, shape: Vec<usize>, stats: TensorStats) -> Self {
        Self {
            indices,
            shape,
            original_idx: None,
            stats,
        }
    }
}
/// Adaptive planner that automatically selects the best algorithm
///
/// Analyzes problem characteristics and chooses the optimal planning strategy:
/// - **Small networks (n ≤ 8)**: Use DP for optimal solution
/// - **Medium networks (8 < n ≤ 20)**: Use Beam Search with adaptive width
/// - **Large networks (n > 20)**: Use Greedy or SA based on quality requirements
///
/// # Decision Criteria
///
/// 1. **Problem size**: Number of tensors to contract
/// 2. **Tensor dimensions**: Large tensors favor greedy (faster planning)
/// 3. **Quality requirement**: From quality preference (low/medium/high)
/// 4. **Time budget**: Available planning time
///
/// # Example
///
/// ```
/// use tenrso_planner::{AdaptivePlanner, Planner, PlanHints};
///
/// let planner = AdaptivePlanner::new();
/// let hints = PlanHints::default();
///
/// // Automatically selects best algorithm
/// let plan = planner.make_plan(
///     "ij,jk,kl->il",
///     &[vec![10, 20], vec![20, 30], vec![30, 10]],
///     &hints
/// ).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AdaptivePlanner {
    /// Quality preference: "low", "medium", "high"
    quality_preference: String,
    /// Maximum planning time in milliseconds (0 = no limit)
    max_planning_time_ms: u64,
}
impl AdaptivePlanner {
    /// Create a new adaptive planner with default settings (medium quality)
    pub fn new() -> Self {
        Self {
            quality_preference: "medium".to_string(),
            max_planning_time_ms: 0,
        }
    }
    /// Create an adaptive planner with custom quality preference
    ///
    /// # Arguments
    ///
    /// * `quality` - "low" (fast), "medium" (balanced), "high" (best quality)
    pub fn with_quality(quality: &str) -> Self {
        Self {
            quality_preference: quality.to_string(),
            max_planning_time_ms: 0,
        }
    }
    /// Create an adaptive planner with time budget
    ///
    /// # Arguments
    ///
    /// * `quality` - Quality preference
    /// * `max_time_ms` - Maximum planning time in milliseconds
    pub fn with_budget(quality: &str, max_time_ms: u64) -> Self {
        Self {
            quality_preference: quality.to_string(),
            max_planning_time_ms: max_time_ms,
        }
    }
    /// Select the best planner for the given problem
    pub(super) fn select_algorithm(
        &self,
        spec: &EinsumSpec,
        shapes: &[Vec<usize>],
    ) -> PlanningAlgorithm {
        let n = spec.num_inputs();
        if n <= 1 {
            return PlanningAlgorithm::Greedy;
        }
        let total_elements: usize = shapes.iter().map(|s| s.iter().product::<usize>()).sum();
        let avg_tensor_size = total_elements / n;
        match n {
            _ if n <= 5 => PlanningAlgorithm::DP,
            _ if n <= 8 => {
                if self.quality_preference == "high" {
                    PlanningAlgorithm::DP
                } else {
                    PlanningAlgorithm::BeamSearch(5)
                }
            }
            _ if n <= 15 => {
                let beam_width = match self.quality_preference.as_str() {
                    "low" => 3,
                    "medium" => 5,
                    "high" => 10,
                    _ => 5,
                };
                if self.max_planning_time_ms > 0 && self.max_planning_time_ms < 100 {
                    PlanningAlgorithm::Greedy
                } else {
                    PlanningAlgorithm::BeamSearch(beam_width)
                }
            }
            _ if n <= 20 => {
                if self.quality_preference == "high" && avg_tensor_size < 1_000_000 {
                    PlanningAlgorithm::BeamSearch(5)
                } else {
                    PlanningAlgorithm::Greedy
                }
            }
            _ => {
                // For very large networks (> 20), choose between GA, SA, or Greedy
                if self.quality_preference == "high" {
                    if self.max_planning_time_ms > 5000 || self.max_planning_time_ms == 0 {
                        // Enough time for GA
                        PlanningAlgorithm::GeneticAlgorithm
                    } else if self.max_planning_time_ms > 1000 {
                        // Medium time: use SA
                        PlanningAlgorithm::SimulatedAnnealing
                    } else {
                        PlanningAlgorithm::Greedy
                    }
                } else {
                    PlanningAlgorithm::Greedy
                }
            }
        }
    }
}
/// Greedy contraction order planner (struct-based)
///
/// Implements the [`Planner`] trait using a greedy heuristic.
///
/// # Example
///
/// ```
/// use tenrso_planner::{GreedyPlanner, Planner, PlanHints};
///
/// let planner = GreedyPlanner::new();
/// let hints = PlanHints::default();
/// let plan = planner.make_plan(
///     "ij,jk->ik",
///     &[vec![10, 20], vec![20, 30]],
///     &hints
/// ).unwrap();
///
/// assert_eq!(plan.nodes.len(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GreedyPlanner;
impl GreedyPlanner {
    /// Create a new greedy planner
    pub fn new() -> Self {
        Self
    }
}
/// Simulated annealing planner (struct-based)
///
/// Implements the [`Planner`] trait using simulated annealing.
///
/// # Example
///
/// ```
/// use tenrso_planner::{SimulatedAnnealingPlanner, Planner, PlanHints};
///
/// let planner = SimulatedAnnealingPlanner::new();
/// let hints = PlanHints::default();
/// let plan = planner.make_plan(
///     "ij,jk,kl,lm->im",
///     &[vec![10, 20], vec![20, 30], vec![30, 40], vec![40, 10]],
///     &hints
/// ).unwrap();
///
/// assert_eq!(plan.nodes.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingPlanner {
    pub(super) initial_temp: f64,
    pub(super) cooling_rate: f64,
    pub(super) max_iterations: usize,
}
impl SimulatedAnnealingPlanner {
    /// Create a new SA planner with default parameters
    pub fn new() -> Self {
        Self {
            initial_temp: 1000.0,
            cooling_rate: 0.95,
            max_iterations: 1000,
        }
    }
    /// Create an SA planner with custom parameters
    pub fn with_params(initial_temp: f64, cooling_rate: f64, max_iterations: usize) -> Self {
        Self {
            initial_temp,
            cooling_rate,
            max_iterations,
        }
    }
}
/// Beam search planner (struct-based)
///
/// Implements the [`Planner`] trait using beam search with configurable beam width.
///
/// # Example
///
/// ```
/// use tenrso_planner::{BeamSearchPlanner, Planner, PlanHints};
///
/// let planner = BeamSearchPlanner::with_beam_width(5);
/// let hints = PlanHints::default();
/// let plan = planner.make_plan(
///     "ij,jk,kl->il",
///     &[vec![10, 20], vec![20, 30], vec![30, 10]],
///     &hints
/// ).unwrap();
///
/// assert_eq!(plan.nodes.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct BeamSearchPlanner {
    pub(super) beam_width: usize,
}
impl BeamSearchPlanner {
    /// Create a new beam search planner with default beam width (5)
    pub fn new() -> Self {
        Self { beam_width: 5 }
    }
    /// Create a beam search planner with custom beam width
    pub fn with_beam_width(beam_width: usize) -> Self {
        Self { beam_width }
    }
}
/// Dynamic programming contraction order planner (struct-based)
///
/// Implements the [`Planner`] trait using optimal DP algorithm.
/// Automatically falls back to greedy for large networks (n > 20).
///
/// # Example
///
/// ```
/// use tenrso_planner::{DPPlanner, Planner, PlanHints};
///
/// let planner = DPPlanner::new();
/// let hints = PlanHints::default();
/// let plan = planner.make_plan(
///     "ij,jk,kl->il",
///     &[vec![10, 20], vec![20, 30], vec![30, 10]],
///     &hints
/// ).unwrap();
///
/// assert_eq!(plan.nodes.len(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct DPPlanner;
impl DPPlanner {
    /// Create a new DP planner
    pub fn new() -> Self {
        Self
    }
}

/// Genetic algorithm planner (struct-based)
///
/// Implements the [`Planner`] trait using a genetic algorithm for large-scale optimization.
///
/// Uses evolutionary search with:
/// - Population-based exploration (50-200 individuals)
/// - Tournament selection (k=3)
/// - Order-preserving crossover
/// - Adjacent swap mutation
/// - Elitism to preserve best solutions
///
/// **Best for:** Large tensor networks (> 20 tensors) where DP is infeasible
///
/// # Complexity
///
/// Time: O(generations × population × n³)
/// Space: O(population × n)
///
/// # Example
///
/// ```
/// use tenrso_planner::{GeneticAlgorithmPlanner, Planner, PlanHints};
///
/// let planner = GeneticAlgorithmPlanner::new();
/// let hints = PlanHints::default();
/// let plan = planner.make_plan(
///     "ij,jk,kl,lm->im",
///     &[vec![10, 20], vec![20, 30], vec![30, 40], vec![40, 10]],
///     &hints
/// ).unwrap();
///
/// assert_eq!(plan.nodes.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct GeneticAlgorithmPlanner {
    pub(super) population_size: usize,
    pub(super) max_generations: usize,
    pub(super) mutation_rate: f64,
    pub(super) elitism_count: usize,
}

impl GeneticAlgorithmPlanner {
    /// Create a new GA planner with default parameters
    ///
    /// Default configuration:
    /// - Population size: 100
    /// - Generations: 100
    /// - Mutation rate: 0.2
    /// - Elitism: 5 best individuals
    pub fn new() -> Self {
        Self {
            population_size: 100,
            max_generations: 100,
            mutation_rate: 0.2,
            elitism_count: 5,
        }
    }

    /// Create a GA planner with custom parameters
    ///
    /// # Arguments
    ///
    /// * `population_size` - Number of candidate plans (50-200 typical)
    /// * `max_generations` - Number of evolution iterations (50-200 typical)
    /// * `mutation_rate` - Probability of mutation (0.1-0.3 typical)
    /// * `elitism_count` - Number of best individuals to preserve (2-10 typical)
    pub fn with_params(
        population_size: usize,
        max_generations: usize,
        mutation_rate: f64,
        elitism_count: usize,
    ) -> Self {
        Self {
            population_size,
            max_generations,
            mutation_rate,
            elitism_count,
        }
    }

    /// Create a fast GA planner for quick experimentation
    pub fn fast() -> Self {
        Self {
            population_size: 50,
            max_generations: 50,
            mutation_rate: 0.2,
            elitism_count: 3,
        }
    }

    /// Create a high-quality GA planner for production use
    pub fn high_quality() -> Self {
        Self {
            population_size: 200,
            max_generations: 200,
            mutation_rate: 0.15,
            elitism_count: 10,
        }
    }
}

/// Enum representing available planning algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PlanningAlgorithm {
    Greedy,
    DP,
    BeamSearch(usize),
    SimulatedAnnealing,
    GeneticAlgorithm,
}
