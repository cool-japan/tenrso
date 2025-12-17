//! Plan execution simulation and cost estimation
//!
//! This module provides tools to simulate the execution of a contraction plan
//! without actually performing the computations. This is useful for:
//! - Estimating runtime before execution
//! - Debugging plan quality issues
//! - Understanding memory usage patterns
//! - Validating plan correctness
//!
//! # Examples
//!
//! ```
//! use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints, simulate_execution};
//!
//! let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
//! let shapes = vec![vec![100, 200], vec![200, 300], vec![300, 100]];
//! let hints = PlanHints::default();
//!
//! let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
//! let simulation = simulate_execution(&plan);
//!
//! println!("Total estimated runtime: {:.2}ms", simulation.total_time_ms);
//! println!("Peak memory usage: {} MB", simulation.peak_memory_bytes / 1_000_000);
//! println!("Number of allocations: {}", simulation.num_allocations);
//! ```

use crate::api::{Plan, PlanNode};

/// Hardware type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareType {
    /// CPU-based execution
    CPU,
    /// GPU-based execution
    GPU,
}

/// Hardware model for cost estimation
#[derive(Debug, Clone)]
pub struct HardwareModel {
    /// Hardware type (CPU or GPU)
    pub hardware_type: HardwareType,
    /// FLOPs per second (e.g., 1e12 for 1 TFLOPs)
    pub flops_per_second: f64,
    /// Memory bandwidth in bytes/second (e.g., 100e9 for 100 GB/s)
    pub memory_bandwidth: f64,
    /// Allocation overhead in nanoseconds
    pub allocation_overhead_ns: f64,
    /// Cache sizes (L1, L2, L3 for CPU; L1, L2 for GPU) in bytes
    pub cache_sizes: Vec<usize>,
    /// GPU-specific: Kernel launch overhead in microseconds
    pub kernel_launch_overhead_us: f64,
    /// GPU-specific: PCIe transfer overhead (latency + bandwidth)
    pub pcie_bandwidth: Option<f64>, // bytes/second
    /// GPU-specific: Has Tensor Cores (for mixed precision acceleration)
    pub has_tensor_cores: bool,
    /// GPU-specific: Tensor Core FLOPs (higher than regular FLOPs)
    pub tensor_core_flops: Option<f64>,
    /// GPU-specific: Shared memory size (per SM/block)
    pub shared_memory_size: Option<usize>,
}

impl Default for HardwareModel {
    fn default() -> Self {
        Self {
            hardware_type: HardwareType::CPU,
            // Default: Modern CPU (e.g., Intel i9-12900K)
            flops_per_second: 1e12,        // ~1 TFLOPs
            memory_bandwidth: 50e9,        // ~50 GB/s
            allocation_overhead_ns: 100.0, // ~100ns per allocation
            cache_sizes: vec![
                32 * 1024,        // L1: 32 KB
                512 * 1024,       // L2: 512 KB
                16 * 1024 * 1024, // L3: 16 MB
            ],
            kernel_launch_overhead_us: 0.0,
            pcie_bandwidth: None,
            has_tensor_cores: false,
            tensor_core_flops: None,
            shared_memory_size: None,
        }
    }
}

impl HardwareModel {
    /// Create a hardware model for a high-end CPU
    pub fn high_end_cpu() -> Self {
        Self {
            hardware_type: HardwareType::CPU,
            flops_per_second: 2e12,       // ~2 TFLOPs
            memory_bandwidth: 100e9,      // ~100 GB/s
            allocation_overhead_ns: 50.0, // ~50ns
            cache_sizes: vec![
                48 * 1024,        // L1: 48 KB
                1024 * 1024,      // L2: 1 MB
                32 * 1024 * 1024, // L3: 32 MB
            ],
            kernel_launch_overhead_us: 0.0,
            pcie_bandwidth: None,
            has_tensor_cores: false,
            tensor_core_flops: None,
            shared_memory_size: None,
        }
    }

    /// Create a hardware model for a low-end CPU
    pub fn low_end_cpu() -> Self {
        Self {
            hardware_type: HardwareType::CPU,
            flops_per_second: 0.5e12,      // ~0.5 TFLOPs
            memory_bandwidth: 20e9,        // ~20 GB/s
            allocation_overhead_ns: 200.0, // ~200ns
            cache_sizes: vec![
                16 * 1024,       // L1: 16 KB
                256 * 1024,      // L2: 256 KB
                8 * 1024 * 1024, // L3: 8 MB
            ],
            kernel_launch_overhead_us: 0.0,
            pcie_bandwidth: None,
            has_tensor_cores: false,
            tensor_core_flops: None,
            shared_memory_size: None,
        }
    }

    /// Create a hardware model for a generic GPU (defaults to Ampere-class)
    pub fn gpu() -> Self {
        Self::gpu_ampere()
    }

    /// NVIDIA Pascal GPU (e.g., GTX 1080 Ti, P100)
    ///
    /// - Compute: 11-12 TFLOPs (FP32)
    /// - Memory: 11 GB GDDR5X (484 GB/s)
    /// - Architecture: Pascal (2016)
    pub fn gpu_pascal() -> Self {
        Self {
            hardware_type: HardwareType::GPU,
            flops_per_second: 11e12,        // ~11 TFLOPs FP32
            memory_bandwidth: 484e9,        // ~484 GB/s
            allocation_overhead_ns: 2000.0, // ~2µs GPU malloc
            cache_sizes: vec![
                128 * 1024,      // L1: 128 KB per SM
                4 * 1024 * 1024, // L2: 4 MB
            ],
            kernel_launch_overhead_us: 5.0, // ~5µs kernel launch
            pcie_bandwidth: Some(16e9),     // PCIe 3.0 x16: ~16 GB/s
            has_tensor_cores: false,
            tensor_core_flops: None,
            shared_memory_size: Some(96 * 1024), // 96 KB per SM
        }
    }

    /// NVIDIA Volta GPU (e.g., V100)
    ///
    /// - Compute: 14 TFLOPs (FP32), 112 TFLOPs (Tensor Cores FP16)
    /// - Memory: 16/32 GB HBM2 (900 GB/s)
    /// - Architecture: Volta (2017)
    /// - First gen with Tensor Cores
    pub fn gpu_volta() -> Self {
        Self {
            hardware_type: HardwareType::GPU,
            flops_per_second: 14e12, // ~14 TFLOPs FP32
            memory_bandwidth: 900e9, // ~900 GB/s HBM2
            allocation_overhead_ns: 2000.0,
            cache_sizes: vec![
                128 * 1024,      // L1: 128 KB per SM
                6 * 1024 * 1024, // L2: 6 MB
            ],
            kernel_launch_overhead_us: 4.0,
            pcie_bandwidth: Some(16e9), // PCIe 3.0 x16
            has_tensor_cores: true,
            tensor_core_flops: Some(112e12), // 112 TFLOPs FP16 mixed precision
            shared_memory_size: Some(96 * 1024),
        }
    }

    /// NVIDIA Turing GPU (e.g., RTX 2080 Ti)
    ///
    /// - Compute: 13 TFLOPs (FP32), 110 TFLOPs (Tensor Cores FP16)
    /// - Memory: 11 GB GDDR6 (616 GB/s)
    /// - Architecture: Turing (2018)
    /// - 2nd gen Tensor Cores
    pub fn gpu_turing() -> Self {
        Self {
            hardware_type: HardwareType::GPU,
            flops_per_second: 13e12,
            memory_bandwidth: 616e9,
            allocation_overhead_ns: 1800.0,
            cache_sizes: vec![
                128 * 1024,
                5 * 1024 * 1024, // L2: 5.5 MB
            ],
            kernel_launch_overhead_us: 3.5,
            pcie_bandwidth: Some(16e9),
            has_tensor_cores: true,
            tensor_core_flops: Some(110e12),
            shared_memory_size: Some(96 * 1024),
        }
    }

    /// NVIDIA Ampere GPU (e.g., RTX 3080, A100)
    ///
    /// - Compute: 30 TFLOPs (FP32, RTX 3080), 19.5 TFLOPs (A100)
    /// - Memory: 10 GB GDDR6X (760 GB/s RTX 3080), 40/80 GB HBM2 (1555 GB/s A100)
    /// - Architecture: Ampere (2020)
    /// - 3rd gen Tensor Cores
    pub fn gpu_ampere() -> Self {
        Self {
            hardware_type: HardwareType::GPU,
            flops_per_second: 30e12, // RTX 3080
            memory_bandwidth: 760e9, // ~760 GB/s
            allocation_overhead_ns: 1500.0,
            cache_sizes: vec![
                192 * 1024,      // L1: 192 KB per SM
                6 * 1024 * 1024, // L2: 6 MB
            ],
            kernel_launch_overhead_us: 3.0,
            pcie_bandwidth: Some(32e9), // PCIe 4.0 x16: ~32 GB/s
            has_tensor_cores: true,
            tensor_core_flops: Some(238e12), // 238 TFLOPs FP16 (RTX 3080)
            shared_memory_size: Some(164 * 1024), // 164 KB per SM (configurable)
        }
    }

    /// NVIDIA Hopper GPU (e.g., H100)
    ///
    /// - Compute: 60 TFLOPs (FP32), 1000 TFLOPs (Tensor Cores FP16)
    /// - Memory: 80 GB HBM3 (3000 GB/s)
    /// - Architecture: Hopper (2022)
    /// - 4th gen Tensor Cores with FP8 support
    pub fn gpu_hopper() -> Self {
        Self {
            hardware_type: HardwareType::GPU,
            flops_per_second: 60e12,  // ~60 TFLOPs FP32
            memory_bandwidth: 3000e9, // ~3 TB/s HBM3
            allocation_overhead_ns: 1000.0,
            cache_sizes: vec![
                256 * 1024,       // L1: 256 KB per SM
                50 * 1024 * 1024, // L2: 50 MB
            ],
            kernel_launch_overhead_us: 2.0,
            pcie_bandwidth: Some(64e9), // PCIe 5.0 x16: ~64 GB/s
            has_tensor_cores: true,
            tensor_core_flops: Some(1000e12), // 1 PFLOPs FP16/BF16
            shared_memory_size: Some(228 * 1024),
        }
    }

    /// AMD CDNA2 GPU (e.g., MI250X)
    ///
    /// - Compute: 48 TFLOPs (FP32), 383 TFLOPs (Matrix Cores FP16)
    /// - Memory: 128 GB HBM2e (3200 GB/s)
    /// - Architecture: CDNA2 (2021)
    pub fn gpu_amd_cdna2() -> Self {
        Self {
            hardware_type: HardwareType::GPU,
            flops_per_second: 48e12,
            memory_bandwidth: 3200e9, // ~3.2 TB/s
            allocation_overhead_ns: 1500.0,
            cache_sizes: vec![
                128 * 1024,      // L1: 128 KB per CU
                8 * 1024 * 1024, // L2: 8 MB
            ],
            kernel_launch_overhead_us: 3.0,
            pcie_bandwidth: Some(64e9),          // PCIe 4.0 x16
            has_tensor_cores: true,              // Matrix Cores (AMD equivalent)
            tensor_core_flops: Some(383e12),     // 383 TFLOPs FP16
            shared_memory_size: Some(64 * 1024), // LDS: 64 KB per CU
        }
    }

    /// Check if this hardware model represents a GPU
    pub fn is_gpu(&self) -> bool {
        matches!(self.hardware_type, HardwareType::GPU)
    }

    /// Check if this hardware model has Tensor Cores (or AMD Matrix Cores)
    pub fn supports_tensor_cores(&self) -> bool {
        self.has_tensor_cores
    }

    /// Get effective FLOPs considering Tensor Cores for mixed precision workloads
    ///
    /// If Tensor Cores are available and `use_tensor_cores` is true,
    /// returns the higher Tensor Core throughput. Otherwise returns regular FLOPs.
    pub fn effective_flops(&self, use_tensor_cores: bool) -> f64 {
        if use_tensor_cores && self.has_tensor_cores {
            self.tensor_core_flops.unwrap_or(self.flops_per_second)
        } else {
            self.flops_per_second
        }
    }
}

/// Simulation result for a single contraction step
#[derive(Debug, Clone)]
pub struct StepSimulation {
    /// Step index in the plan
    pub step: usize,
    /// Estimated computation time (ms)
    pub compute_time_ms: f64,
    /// Estimated memory transfer time (ms)
    pub memory_time_ms: f64,
    /// Allocation overhead (ms)
    pub allocation_time_ms: f64,
    /// GPU-specific: Kernel launch overhead (ms)
    pub kernel_launch_time_ms: f64,
    /// GPU-specific: PCIe transfer time (ms) for CPU-GPU data movement
    pub pcie_transfer_time_ms: f64,
    /// Total time for this step (ms)
    pub total_time_ms: f64,
    /// Memory allocated for this step (bytes)
    pub memory_allocated: usize,
    /// Is this operation cache-friendly?
    pub cache_friendly: bool,
    /// Used Tensor Cores for acceleration?
    pub used_tensor_cores: bool,
}

/// Complete simulation result for a plan
#[derive(Debug, Clone)]
pub struct PlanSimulation {
    /// Per-step simulation results
    pub steps: Vec<StepSimulation>,
    /// Total estimated execution time (ms)
    pub total_time_ms: f64,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Total number of allocations
    pub num_allocations: usize,
    /// Compute-bound vs memory-bound ratio (0-1, higher = more compute-bound)
    pub compute_intensity: f64,
    /// Overall cache efficiency (0-1, higher = better)
    pub cache_efficiency: f64,
}

impl PlanSimulation {
    /// Get the critical path (slowest step)
    pub fn critical_step(&self) -> Option<&StepSimulation> {
        self.steps
            .iter()
            .max_by(|a, b| a.total_time_ms.partial_cmp(&b.total_time_ms).unwrap())
    }

    /// Get the total computation time (excluding memory transfers)
    pub fn total_compute_time_ms(&self) -> f64 {
        self.steps.iter().map(|s| s.compute_time_ms).sum()
    }

    /// Get the total memory transfer time
    pub fn total_memory_time_ms(&self) -> f64 {
        self.steps.iter().map(|s| s.memory_time_ms).sum()
    }

    /// Get the total allocation overhead
    pub fn total_allocation_time_ms(&self) -> f64 {
        self.steps.iter().map(|s| s.allocation_time_ms).sum()
    }

    /// Check if the plan is compute-bound (>0.5) or memory-bound (<0.5)
    pub fn is_compute_bound(&self) -> bool {
        self.compute_intensity > 0.5
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Total: {:.2}ms | Compute: {:.2}ms ({:.1}%) | Memory: {:.2}ms ({:.1}%) | Peak Mem: {} MB | Steps: {}",
            self.total_time_ms,
            self.total_compute_time_ms(),
            (self.total_compute_time_ms() / self.total_time_ms) * 100.0,
            self.total_memory_time_ms(),
            (self.total_memory_time_ms() / self.total_time_ms) * 100.0,
            self.peak_memory_bytes / 1_000_000,
            self.steps.len()
        )
    }
}

/// Simulate the execution of a contraction plan
///
/// This function estimates the runtime and memory characteristics of executing
/// a plan without actually performing the computations.
///
/// # Arguments
///
/// * `plan` - The contraction plan to simulate
///
/// # Returns
///
/// A `PlanSimulation` with detailed cost estimates
///
/// # Examples
///
/// ```
/// use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints, simulate_execution};
///
/// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
/// let shapes = vec![vec![100, 200], vec![200, 300]];
/// let hints = PlanHints::default();
///
/// let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
/// let sim = simulate_execution(&plan);
///
/// println!("Estimated runtime: {:.2}ms", sim.total_time_ms);
/// println!("Peak memory: {} bytes", sim.peak_memory_bytes);
/// ```
pub fn simulate_execution(plan: &Plan) -> PlanSimulation {
    simulate_execution_with_hardware(plan, &HardwareModel::default())
}

/// Simulate plan execution with a specific hardware model
///
/// # Examples
///
/// ```
/// use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};
/// use tenrso_planner::{simulate_execution_with_hardware, HardwareModel};
///
/// let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
/// let shapes = vec![vec![100, 200], vec![200, 300]];
/// let hints = PlanHints::default();
///
/// let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
///
/// // Simulate on a high-end CPU
/// let sim_cpu = simulate_execution_with_hardware(&plan, &HardwareModel::high_end_cpu());
///
/// // Simulate on a GPU
/// let sim_gpu = simulate_execution_with_hardware(&plan, &HardwareModel::gpu());
///
/// println!("CPU runtime: {:.2}ms", sim_cpu.total_time_ms);
/// println!("GPU runtime: {:.2}ms", sim_gpu.total_time_ms);
/// println!("Speedup: {:.2}x", sim_cpu.total_time_ms / sim_gpu.total_time_ms);
/// ```
pub fn simulate_execution_with_hardware(plan: &Plan, hardware: &HardwareModel) -> PlanSimulation {
    let mut steps = Vec::new();
    let mut current_memory = 0;
    let mut peak_memory = 0;
    let mut num_allocations = 0;

    for node in plan.nodes.iter() {
        let step_sim = simulate_step(node, hardware);
        current_memory += node.memory;
        peak_memory = peak_memory.max(current_memory);
        num_allocations += 1;

        steps.push(step_sim);
    }

    let total_time_ms: f64 = steps.iter().map(|s| s.total_time_ms).sum();
    let total_compute_ms: f64 = steps.iter().map(|s| s.compute_time_ms).sum();

    let compute_intensity = if total_time_ms > 0.0 {
        total_compute_ms / total_time_ms
    } else {
        0.0
    };

    let cache_friendly_steps = steps.iter().filter(|s| s.cache_friendly).count();
    let cache_efficiency = if !steps.is_empty() {
        cache_friendly_steps as f64 / steps.len() as f64
    } else {
        0.0
    };

    PlanSimulation {
        steps,
        total_time_ms,
        peak_memory_bytes: peak_memory,
        num_allocations,
        compute_intensity,
        cache_efficiency,
    }
}

/// Simulate a single contraction step
fn simulate_step(node: &PlanNode, hardware: &HardwareModel) -> StepSimulation {
    // Determine if we can use Tensor Cores
    // Heuristic: Use Tensor Cores for large matrix multiplications (> 1M elements)
    let used_tensor_cores = hardware.supports_tensor_cores() && node.memory > 1_000_000;
    let effective_flops = hardware.effective_flops(used_tensor_cores);

    // Estimate compute time (FLOPs / effective FLOPs per second)
    let compute_time_s = node.cost / effective_flops;
    let compute_time_ms = compute_time_s * 1000.0;

    // Estimate memory transfer time (bytes / bandwidth)
    // Assume we need to read inputs and write output
    let bytes_transferred = node.memory * 2; // Read + write
    let memory_time_s = bytes_transferred as f64 / hardware.memory_bandwidth;
    let memory_time_ms = memory_time_s * 1000.0;

    // Allocation overhead
    let allocation_time_ms = hardware.allocation_overhead_ns / 1_000_000.0;

    // GPU-specific: Kernel launch overhead
    let kernel_launch_time_ms = if hardware.is_gpu() {
        hardware.kernel_launch_overhead_us / 1000.0 // Convert µs to ms
    } else {
        0.0
    };

    // GPU-specific: PCIe transfer time (CPU ↔ GPU)
    let pcie_transfer_time_ms = if hardware.is_gpu() && hardware.pcie_bandwidth.is_some() {
        let pcie_bw = hardware.pcie_bandwidth.unwrap();
        // Assume data needs to be transferred to GPU and results back to CPU
        let pcie_transfer_s = (bytes_transferred as f64) / pcie_bw;
        pcie_transfer_s * 1000.0
    } else {
        0.0
    };

    // Check if operation fits in cache (largest cache level)
    let largest_cache = hardware.cache_sizes.iter().max().copied().unwrap_or(0);
    let cache_friendly = node.memory <= largest_cache;

    // Apply cache penalty if not cache-friendly
    let cache_penalty = if cache_friendly { 1.0 } else { 1.5 };

    // For GPUs, shared memory can mitigate cache misses
    let effective_cache_penalty = if hardware.is_gpu() && hardware.shared_memory_size.is_some() {
        cache_penalty * 0.8 // GPUs handle memory better with shared memory
    } else {
        cache_penalty
    };

    let total_time_ms = compute_time_ms
        + (memory_time_ms * effective_cache_penalty)
        + allocation_time_ms
        + kernel_launch_time_ms
        + pcie_transfer_time_ms;

    StepSimulation {
        step: 0, // Will be set by caller
        compute_time_ms,
        memory_time_ms: memory_time_ms * effective_cache_penalty,
        allocation_time_ms,
        kernel_launch_time_ms,
        pcie_transfer_time_ms,
        total_time_ms,
        memory_allocated: node.memory,
        cache_friendly,
        used_tensor_cores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{greedy_planner, EinsumSpec, PlanHints};

    #[test]
    fn test_simulate_execution_basic() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let sim = simulate_execution(&plan);

        assert_eq!(sim.steps.len(), plan.nodes.len());
        assert!(sim.total_time_ms > 0.0);
        assert!(sim.peak_memory_bytes > 0);
        assert_eq!(sim.num_allocations, plan.nodes.len());
    }

    #[test]
    fn test_simulate_with_hardware_models() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        // Use large matrices where GPU excels (amortizes overhead)
        let shapes = vec![vec![1000, 2000], vec![2000, 3000]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        let sim_default = simulate_execution(&plan);
        let sim_high_end = simulate_execution_with_hardware(&plan, &HardwareModel::high_end_cpu());
        let sim_low_end = simulate_execution_with_hardware(&plan, &HardwareModel::low_end_cpu());
        let sim_gpu = simulate_execution_with_hardware(&plan, &HardwareModel::gpu());

        // High-end CPU should be faster than default
        assert!(sim_high_end.total_time_ms < sim_default.total_time_ms);

        // Low-end CPU should be slower than default
        assert!(sim_low_end.total_time_ms > sim_default.total_time_ms);

        // For large problems, GPU should be faster than high-end CPU
        // (amortizes PCIe and kernel launch overhead)
        assert!(sim_gpu.total_time_ms < sim_high_end.total_time_ms);
    }

    #[test]
    fn test_critical_step() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 10]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let sim = simulate_execution(&plan);

        let critical = sim.critical_step();
        assert!(critical.is_some());

        // Critical step should be the slowest
        if let Some(crit) = critical {
            for step in &sim.steps {
                assert!(crit.total_time_ms >= step.total_time_ms);
            }
        }
    }

    #[test]
    fn test_compute_intensity() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![1000, 2000], vec![2000, 3000]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let sim = simulate_execution(&plan);

        // Large matrix multiplication should be compute-bound
        assert!(sim.compute_intensity > 0.0);
        assert!(sim.compute_intensity <= 1.0);
    }

    #[test]
    fn test_simulation_summary() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let sim = simulate_execution(&plan);

        let summary = sim.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Total:"));
        assert!(summary.contains("Compute:"));
        assert!(summary.contains("Memory:"));
    }

    #[test]
    fn test_hardware_model_presets() {
        let high_end = HardwareModel::high_end_cpu();
        let low_end = HardwareModel::low_end_cpu();
        let gpu = HardwareModel::gpu();

        assert!(high_end.flops_per_second > low_end.flops_per_second);
        assert!(high_end.memory_bandwidth > low_end.memory_bandwidth);
        assert!(gpu.flops_per_second > high_end.flops_per_second);
    }

    #[test]
    fn test_cache_efficiency() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 10], vec![10, 10]]; // Small, cache-friendly
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();
        let sim = simulate_execution(&plan);

        // Small operations should have good cache efficiency
        assert!(sim.cache_efficiency >= 0.0);
        assert!(sim.cache_efficiency <= 1.0);
    }

    // ========== GPU Backend Tests ==========

    #[test]
    fn test_gpu_pascal_model() {
        let gpu = HardwareModel::gpu_pascal();
        assert_eq!(gpu.hardware_type, HardwareType::GPU);
        assert!(gpu.is_gpu());
        assert!(!gpu.supports_tensor_cores());
        assert!(gpu.flops_per_second > 10e12);
        assert!(gpu.pcie_bandwidth.is_some());
    }

    #[test]
    fn test_gpu_volta_model() {
        let gpu = HardwareModel::gpu_volta();
        assert!(gpu.is_gpu());
        assert!(gpu.supports_tensor_cores());
        assert!(gpu.has_tensor_cores);
        assert!(gpu.tensor_core_flops.is_some());
        assert!(gpu.tensor_core_flops.unwrap() > gpu.flops_per_second);
    }

    #[test]
    fn test_gpu_turing_model() {
        let gpu = HardwareModel::gpu_turing();
        assert!(gpu.supports_tensor_cores());
        assert!(gpu.shared_memory_size.is_some());
    }

    #[test]
    fn test_gpu_ampere_model() {
        let gpu = HardwareModel::gpu_ampere();
        assert!(gpu.supports_tensor_cores());
        assert!(gpu.flops_per_second > 20e12);
        assert!(gpu.memory_bandwidth > 700e9);
    }

    #[test]
    fn test_gpu_hopper_model() {
        let gpu = HardwareModel::gpu_hopper();
        assert!(gpu.supports_tensor_cores());
        assert!(gpu.flops_per_second > 50e12);
        assert!(gpu.memory_bandwidth > 2500e9);
        assert!(gpu.tensor_core_flops.unwrap() > 900e12); // Nearly 1 PFLOPs
    }

    #[test]
    fn test_gpu_amd_cdna2_model() {
        let gpu = HardwareModel::gpu_amd_cdna2();
        assert!(gpu.is_gpu());
        assert!(gpu.supports_tensor_cores()); // Matrix Cores
        assert!(gpu.memory_bandwidth > 3000e9); // >3 TB/s
    }

    #[test]
    fn test_gpu_vs_cpu_simulation() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![1000, 2000], vec![2000, 3000]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        let sim_cpu = simulate_execution_with_hardware(&plan, &HardwareModel::default());
        let sim_gpu = simulate_execution_with_hardware(&plan, &HardwareModel::gpu_ampere());

        // GPU should be significantly faster for large matmul
        assert!(sim_gpu.total_time_ms < sim_cpu.total_time_ms);
    }

    #[test]
    fn test_tensor_core_acceleration() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![2000, 2000], vec![2000, 2000]]; // Large enough for Tensor Cores
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        let volta = HardwareModel::gpu_volta();
        let sim = simulate_execution_with_hardware(&plan, &volta);

        // Check if Tensor Cores were used
        let tensor_core_used = sim.steps.iter().any(|s| s.used_tensor_cores);
        assert!(
            tensor_core_used,
            "Tensor Cores should be used for large operations"
        );
    }

    #[test]
    fn test_pcie_transfer_overhead() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![100, 200], vec![200, 300]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        let gpu = HardwareModel::gpu_ampere();
        let sim = simulate_execution_with_hardware(&plan, &gpu);

        // PCIe transfer time should be non-zero for GPU
        let total_pcie = sim
            .steps
            .iter()
            .map(|s| s.pcie_transfer_time_ms)
            .sum::<f64>();
        assert!(
            total_pcie > 0.0,
            "PCIe transfer should contribute to GPU execution time"
        );
    }

    #[test]
    fn test_kernel_launch_overhead() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        let gpu = HardwareModel::gpu_ampere();
        let sim = simulate_execution_with_hardware(&plan, &gpu);

        // Kernel launch overhead should be present
        let total_launch = sim
            .steps
            .iter()
            .map(|s| s.kernel_launch_time_ms)
            .sum::<f64>();
        assert!(
            total_launch > 0.0,
            "Kernel launch overhead should be non-zero for GPU"
        );
    }

    #[test]
    fn test_effective_flops_with_tensor_cores() {
        let gpu = HardwareModel::gpu_volta();

        let regular_flops = gpu.effective_flops(false);
        let tensor_core_flops = gpu.effective_flops(true);

        assert_eq!(regular_flops, gpu.flops_per_second);
        assert_eq!(tensor_core_flops, gpu.tensor_core_flops.unwrap());
        assert!(tensor_core_flops > regular_flops);
    }

    #[test]
    fn test_cpu_has_no_gpu_features() {
        let cpu = HardwareModel::default();

        assert!(!cpu.is_gpu());
        assert!(!cpu.supports_tensor_cores());
        assert_eq!(cpu.hardware_type, HardwareType::CPU);
        assert!(cpu.pcie_bandwidth.is_none());
        assert!(cpu.tensor_core_flops.is_none());
    }

    #[test]
    fn test_gpu_generations_progression() {
        let pascal = HardwareModel::gpu_pascal();
        let volta = HardwareModel::gpu_volta();
        let turing = HardwareModel::gpu_turing();
        let ampere = HardwareModel::gpu_ampere();
        let hopper = HardwareModel::gpu_hopper();

        // Memory bandwidth should generally increase
        assert!(volta.memory_bandwidth > pascal.memory_bandwidth);
        assert!(hopper.memory_bandwidth > ampere.memory_bandwidth);

        // Tensor Cores introduced in Volta
        assert!(!pascal.supports_tensor_cores());
        assert!(volta.supports_tensor_cores());
        assert!(turing.supports_tensor_cores());
        assert!(ampere.supports_tensor_cores());
        assert!(hopper.supports_tensor_cores());

        // FLOPs should generally increase (though architecture changes vary)
        assert!(hopper.flops_per_second > volta.flops_per_second);
    }

    #[test]
    fn test_multi_gpu_comparison() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![500, 1000], vec![1000, 1500], vec![1500, 500]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        let nvidia_hopper = HardwareModel::gpu_hopper();
        let amd_cdna2 = HardwareModel::gpu_amd_cdna2();

        let sim_nvidia = simulate_execution_with_hardware(&plan, &nvidia_hopper);
        let sim_amd = simulate_execution_with_hardware(&plan, &amd_cdna2);

        // Both should complete successfully
        assert!(sim_nvidia.total_time_ms > 0.0);
        assert!(sim_amd.total_time_ms > 0.0);

        // Hopper should be faster due to higher compute and memory bandwidth
        assert!(sim_nvidia.total_time_ms < sim_amd.total_time_ms);
    }

    #[test]
    fn test_gpu_shared_memory_benefit() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        // Size that doesn't fit in L2 but could use shared memory
        let shapes = vec![vec![200, 300], vec![300, 400]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        let gpu = HardwareModel::gpu_ampere();
        let sim = simulate_execution_with_hardware(&plan, &gpu);

        // Simulation should complete
        assert!(sim.total_time_ms > 0.0);
        assert!(gpu.shared_memory_size.is_some());
    }
}
