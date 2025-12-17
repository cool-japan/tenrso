//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::hints::ExecHints;
use crate::ops::execute_dense_contraction;
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use std::collections::HashMap;
use tenrso_core::{DenseND, TensorHandle};
use tenrso_planner::{greedy_planner, EinsumSpec, Plan, PlanHints};

// Re-export ScatterMode from advanced_indexing
pub use super::advanced_indexing::ScatterMode;

/// Reduction operation types
#[derive(Clone, Debug)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
    Prod,
    All,
    Any,
    ArgMax,
    ArgMin,
}
/// Binary element-wise operation types (operations on two tensors)
#[derive(Clone, Debug)]
pub enum BinaryOp {
    /// Element-wise addition: x + y
    Add,
    /// Element-wise subtraction: x - y
    Sub,
    /// Element-wise multiplication: x * y
    Mul,
    /// Element-wise division: x / y
    Div,
    /// Element-wise power: x^y
    Pow,
    /// Element-wise maximum: max(x, y)
    Maximum,
    /// Element-wise minimum: min(x, y)
    Minimum,
}
/// Memory pool for tensor allocation reuse
///
/// Tracks allocated tensors by their shape signature for reuse.
/// This reduces allocation overhead for repeated operations.
///
/// # Type Safety
///
/// The pool is generic over type T, ensuring type-safe buffer reuse.
/// T must implement `bytemuck::Pod` (Plain Old Data) and `bytemuck::Zeroable`
/// for safe memory operations.
///
/// # Statistics
///
/// The pool tracks:
/// - **hits**: Number of successful buffer reuses
/// - **misses**: Number of new allocations
/// - **total_bytes_pooled**: Total bytes currently in pools
/// - **unique_shapes**: Number of distinct shape signatures
pub(crate) struct MemoryPool<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    /// Map from shape signature to available buffer pool
    /// Shape signature is a string like "2x3x4" for a [2, 3, 4] tensor
    pools: HashMap<String, Vec<Vec<T>>>,
    /// Statistics for monitoring
    hits: usize,
    misses: usize,
    total_allocations: usize,
    total_releases: usize,
    enabled: bool,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

/// Memory pool statistics
#[derive(Debug, Clone, PartialEq)]
pub struct PoolStats {
    /// Number of buffer reuses (cache hits)
    pub hits: usize,
    /// Number of new allocations (cache misses)
    pub misses: usize,
    /// Total number of allocation requests
    pub total_allocations: usize,
    /// Total number of buffer releases
    pub total_releases: usize,
    /// Cache hit rate (hits / total)
    pub hit_rate: f64,
    /// Number of unique shape signatures in pool
    pub unique_shapes: usize,
    /// Total bytes currently pooled
    pub total_bytes_pooled: usize,
    /// Total number of buffers currently pooled
    pub total_buffers_pooled: usize,
}

impl<T> MemoryPool<T>
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    /// Create a new memory pool
    pub(crate) fn new() -> Self {
        Self {
            pools: HashMap::new(),
            hits: 0,
            misses: 0,
            total_allocations: 0,
            total_releases: 0,
            enabled: true,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a disabled memory pool (no actual pooling)
    pub(crate) fn disabled() -> Self {
        Self {
            pools: HashMap::new(),
            hits: 0,
            misses: 0,
            total_allocations: 0,
            total_releases: 0,
            enabled: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Enable or disable the memory pool
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            // Clear pools when disabling
            self.pools.clear();
        }
    }

    /// Check if pooling is enabled
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get a buffer for the given shape, reusing if available
    ///
    /// **Phase 2 Status**: Now using type-safe generic buffer pooling.
    ///
    /// Returns a Vec<T> with the specified total size (product of shape dimensions).
    /// If a matching buffer is available in the pool, it will be reused (cache hit).
    /// Otherwise, a new buffer is allocated (cache miss).
    #[allow(dead_code)]
    pub(crate) fn acquire(&mut self, shape: &[usize]) -> Vec<T> {
        self.total_allocations += 1;

        let total_size: usize = shape.iter().product();

        if !self.enabled {
            self.misses += 1;
            return vec![T::zeroed(); total_size];
        }

        let signature = Self::shape_signature(shape);

        if let Some(pool) = self.pools.get_mut(&signature) {
            if let Some(mut buffer) = pool.pop() {
                self.hits += 1;
                // Resize buffer to match requested size
                buffer.resize(total_size, T::zeroed());
                return buffer;
            }
        }

        self.misses += 1;
        vec![T::zeroed(); total_size]
    }

    /// Return a buffer to the pool for reuse
    ///
    /// **Phase 2 Status**: Now using type-safe generic buffer pooling.
    ///
    /// Adds the buffer back to the pool for the given shape signature.
    /// Pools are limited to MAX_POOL_SIZE buffers per shape to prevent unbounded growth.
    #[allow(dead_code)]
    pub(crate) fn release(&mut self, shape: &[usize], buffer: Vec<T>) {
        self.total_releases += 1;

        if !self.enabled {
            // Don't pool if disabled - buffer will be dropped
            return;
        }

        let signature = Self::shape_signature(shape);
        let pool = self.pools.entry(signature).or_default();

        const MAX_POOL_SIZE: usize = 16;
        if pool.len() < MAX_POOL_SIZE {
            pool.push(buffer);
        }
        // If pool is full, buffer is dropped
    }

    /// Create a shape signature for hashing
    ///
    /// Converts a shape like `[2, 3, 4]` to a string like `"2x3x4"`.
    /// Used as a key for the buffer pool HashMap.
    #[allow(dead_code)]
    pub(crate) fn shape_signature(shape: &[usize]) -> String {
        shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }

    /// Get pool statistics (deprecated - use detailed_stats)
    pub(crate) fn stats(&self) -> (usize, usize, f64) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        (self.hits, self.misses, hit_rate)
    }

    /// Get detailed pool statistics
    pub(crate) fn detailed_stats(&self) -> PoolStats {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };

        let unique_shapes = self.pools.len();
        let mut total_bytes_pooled = 0;
        let mut total_buffers_pooled = 0;

        let elem_size = std::mem::size_of::<T>();

        for pool in self.pools.values() {
            total_buffers_pooled += pool.len();
            for buffer in pool {
                total_bytes_pooled += buffer.len() * elem_size;
            }
        }

        PoolStats {
            hits: self.hits,
            misses: self.misses,
            total_allocations: self.total_allocations,
            total_releases: self.total_releases,
            hit_rate,
            unique_shapes,
            total_bytes_pooled,
            total_buffers_pooled,
        }
    }

    /// Clear all pooled buffers
    pub(crate) fn clear(&mut self) {
        self.pools.clear();
        self.hits = 0;
        self.misses = 0;
        self.total_allocations = 0;
        self.total_releases = 0;
    }

    /// Get the number of unique shapes in the pool
    pub(crate) fn num_shapes(&self) -> usize {
        self.pools.len()
    }

    /// Get the total number of buffers in the pool
    pub(crate) fn num_buffers(&self) -> usize {
        self.pools.values().map(|v| v.len()).sum()
    }
}
/// CPU executor implementation with memory pooling and parallel execution
pub struct CpuExecutor {
    /// Memory pool for f32 tensors
    ///
    /// **Phase 2 Status**: Type-safe buffer pooling now operational.
    memory_pool_f32: MemoryPool<f32>,
    /// Memory pool for f64 tensors
    ///
    /// **Phase 2 Status**: Type-safe buffer pooling now operational.
    memory_pool_f64: MemoryPool<f64>,
    /// Number of threads to use (0 = auto-detect)
    pub num_threads: usize,
    /// Enable parallel execution for large tensors
    pub enable_parallel: bool,
    /// Enable SIMD-optimized element-wise operations
    pub enable_simd: bool,
    /// Enable tiled/blocked reductions for large tensors
    pub enable_tiled_reductions: bool,
    /// Enable vectorized broadcasting optimizations
    pub enable_vectorized_broadcast: bool,
    /// Enable memory pooling
    pub enable_memory_pool: bool,
}
impl CpuExecutor {
    /// Create a new CPU executor with default settings
    /// All optimizations are enabled by default
    pub fn new() -> Self {
        Self {
            memory_pool_f32: MemoryPool::new(),
            memory_pool_f64: MemoryPool::new(),
            num_threads: 0,
            enable_parallel: true,
            enable_simd: true,
            enable_tiled_reductions: true,
            enable_vectorized_broadcast: true,
            enable_memory_pool: true,
        }
    }
    /// Create a CPU executor with custom thread count
    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            memory_pool_f32: MemoryPool::new(),
            memory_pool_f64: MemoryPool::new(),
            num_threads,
            enable_parallel: true,
            enable_simd: true,
            enable_tiled_reductions: true,
            enable_vectorized_broadcast: true,
            enable_memory_pool: true,
        }
    }
    /// Create a CPU executor with parallel execution disabled
    pub fn serial() -> Self {
        Self {
            memory_pool_f32: MemoryPool::new(),
            memory_pool_f64: MemoryPool::new(),
            num_threads: 1,
            enable_parallel: false,
            enable_simd: false,
            enable_tiled_reductions: false,
            enable_vectorized_broadcast: false,
            enable_memory_pool: false,
        }
    }

    /// Create a CPU executor with all optimizations disabled (for debugging/testing)
    pub fn unoptimized() -> Self {
        Self {
            memory_pool_f32: MemoryPool::disabled(),
            memory_pool_f64: MemoryPool::disabled(),
            num_threads: 1,
            enable_parallel: false,
            enable_simd: false,
            enable_tiled_reductions: false,
            enable_vectorized_broadcast: false,
            enable_memory_pool: false,
        }
    }

    /// Configure SIMD optimization
    pub fn with_simd(mut self, enabled: bool) -> Self {
        self.enable_simd = enabled;
        self
    }

    /// Configure tiled reductions
    pub fn with_tiled_reductions(mut self, enabled: bool) -> Self {
        self.enable_tiled_reductions = enabled;
        self
    }

    /// Configure vectorized broadcasting
    pub fn with_vectorized_broadcast(mut self, enabled: bool) -> Self {
        self.enable_vectorized_broadcast = enabled;
        self
    }

    /// Configure memory pooling
    pub fn with_memory_pool(mut self, enabled: bool) -> Self {
        self.enable_memory_pool = enabled;
        self.memory_pool_f32.set_enabled(enabled);
        self.memory_pool_f64.set_enabled(enabled);
        self
    }

    /// Get memory pool statistics for f32 tensors (hits, misses, hit_rate)
    ///
    /// **Deprecated**: Use `get_pool_stats_f32()` for detailed statistics.
    pub fn pool_stats(&self) -> (usize, usize, f64) {
        self.memory_pool_f32.stats()
    }

    /// Get detailed memory pool statistics for f32 tensors
    ///
    /// Returns comprehensive statistics about f32 memory pool usage including:
    /// - Hit/miss counts and rate
    /// - Total allocations and releases
    /// - Number of unique shapes and buffers
    /// - Total bytes currently pooled
    pub fn get_pool_stats(&self) -> PoolStats {
        self.memory_pool_f32.detailed_stats()
    }

    /// Get detailed memory pool statistics for f32 tensors
    pub fn get_pool_stats_f32(&self) -> PoolStats {
        self.memory_pool_f32.detailed_stats()
    }

    /// Get detailed memory pool statistics for f64 tensors
    pub fn get_pool_stats_f64(&self) -> PoolStats {
        self.memory_pool_f64.detailed_stats()
    }

    /// Clear all memory pools
    ///
    /// Releases all pooled buffers and resets statistics for both f32 and f64 pools.
    pub fn clear_pool(&mut self) {
        self.memory_pool_f32.clear();
        self.memory_pool_f64.clear();
    }

    /// Check if memory pooling is enabled
    pub fn is_pool_enabled(&self) -> bool {
        self.enable_memory_pool
            && self.memory_pool_f32.is_enabled()
            && self.memory_pool_f64.is_enabled()
    }

    /// Enable or disable memory pooling at runtime
    pub fn set_pool_enabled(&mut self, enabled: bool) {
        self.enable_memory_pool = enabled;
        self.memory_pool_f32.set_enabled(enabled);
        self.memory_pool_f64.set_enabled(enabled);
    }

    /// Get the number of unique shapes in f32 pool
    pub fn pool_num_shapes(&self) -> usize {
        self.memory_pool_f32.num_shapes()
    }

    /// Get the number of unique shapes in f32 pool
    pub fn pool_num_shapes_f32(&self) -> usize {
        self.memory_pool_f32.num_shapes()
    }

    /// Get the number of unique shapes in f64 pool
    pub fn pool_num_shapes_f64(&self) -> usize {
        self.memory_pool_f64.num_shapes()
    }

    /// Get the total number of buffers in f32 pool
    pub fn pool_num_buffers(&self) -> usize {
        self.memory_pool_f32.num_buffers()
    }

    /// Get the total number of buffers in f32 pool
    pub fn pool_num_buffers_f32(&self) -> usize {
        self.memory_pool_f32.num_buffers()
    }

    /// Get the total number of buffers in f64 pool
    pub fn pool_num_buffers_f64(&self) -> usize {
        self.memory_pool_f64.num_buffers()
    }

    /// Acquire a buffer from the f32 pool
    ///
    /// # Phase 2 Memory Pool API
    ///
    /// This is the public API for manually acquiring buffers from the f32 pool.
    /// Useful for custom tensor operations and benchmarking.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut executor = CpuExecutor::new();
    /// let buffer = executor.acquire_f32(&[64, 64]);
    /// // Use buffer...
    /// executor.release_f32(&[64, 64], buffer);
    /// ```
    pub fn acquire_f32(&mut self, shape: &[usize]) -> Vec<f32> {
        if self.enable_memory_pool {
            self.memory_pool_f32.acquire(shape)
        } else {
            vec![0.0; shape.iter().product()]
        }
    }

    /// Release a buffer to the f32 pool
    ///
    /// # Phase 2 Memory Pool API
    ///
    /// Returns a buffer to the pool for reuse. The buffer will be available
    /// for future `acquire_f32` calls with the same shape.
    pub fn release_f32(&mut self, shape: &[usize], buffer: Vec<f32>) {
        if self.enable_memory_pool {
            self.memory_pool_f32.release(shape, buffer);
        }
    }

    /// Acquire a buffer from the f64 pool
    ///
    /// # Phase 2 Memory Pool API
    ///
    /// This is the public API for manually acquiring buffers from the f64 pool.
    /// Useful for custom tensor operations and benchmarking.
    pub fn acquire_f64(&mut self, shape: &[usize]) -> Vec<f64> {
        if self.enable_memory_pool {
            self.memory_pool_f64.acquire(shape)
        } else {
            vec![0.0; shape.iter().product()]
        }
    }

    /// Release a buffer to the f64 pool
    ///
    /// # Phase 2 Memory Pool API
    ///
    /// Returns a buffer to the pool for reuse. The buffer will be available
    /// for future `acquire_f64` calls with the same shape.
    pub fn release_f64(&mut self, shape: &[usize], buffer: Vec<f64>) {
        if self.enable_memory_pool {
            self.memory_pool_f64.release(shape, buffer);
        }
    }

    // ========================================================================
    // Generic Pooling Helpers (Phase 5: Automatic Pooling Integration)
    // ========================================================================

    /// Acquire a pooled buffer with automatic type dispatch
    ///
    /// This is an internal helper that automatically selects the appropriate
    /// pool based on the type T. Only f32 and f64 are pooled; other types
    /// allocate directly.
    ///
    /// **Phase 5 Status**: Automatic pooling for all operations.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn acquire_pooled_generic<T>(&mut self, shape: &[usize]) -> Vec<T>
    where
        T: Clone + std::default::Default + 'static,
    {
        if !self.enable_memory_pool {
            return vec![T::default(); shape.iter().product()];
        }

        // Use type introspection to dispatch to the correct pool
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // Safe: We've verified T is f32
            let buffer_f32 = self.memory_pool_f32.acquire(shape);
            // SAFETY: We checked that T == f32, so transmuting Vec<f32> to Vec<T> is safe
            unsafe { std::mem::transmute::<Vec<f32>, Vec<T>>(buffer_f32) }
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            // Safe: We've verified T is f64
            let buffer_f64 = self.memory_pool_f64.acquire(shape);
            // SAFETY: We checked that T == f64, so transmuting Vec<f64> to Vec<T> is safe
            unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(buffer_f64) }
        } else {
            // For other types, allocate directly (no pooling)
            vec![T::default(); shape.iter().product()]
        }
    }

    /// Release a pooled buffer with automatic type dispatch
    ///
    /// This is an internal helper that automatically returns buffers to the
    /// appropriate pool based on the type T.
    ///
    /// **Phase 5 Status**: Automatic pooling for all operations.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn release_pooled_generic<T>(&mut self, shape: &[usize], buffer: Vec<T>)
    where
        T: Clone + std::default::Default + 'static,
    {
        if !self.enable_memory_pool {
            return;
        }

        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: We checked that T == f32, so transmuting Vec<T> to Vec<f32> is safe
            let buffer_f32: Vec<f32> = unsafe { std::mem::transmute::<Vec<T>, Vec<f32>>(buffer) };
            self.memory_pool_f32.release(shape, buffer_f32);
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: We checked that T == f64, so transmuting Vec<T> to Vec<f64> is safe
            let buffer_f64: Vec<f64> = unsafe { std::mem::transmute::<Vec<T>, Vec<f64>>(buffer) };
            self.memory_pool_f64.release(shape, buffer_f64);
        }
        // For other types, buffer is dropped (no pooling)
    }

    /// Execute a computation with a pooled buffer (RAII pattern)
    ///
    /// This helper automatically acquires and releases a pooled buffer,
    /// ensuring the buffer is returned to the pool even if an error occurs.
    ///
    /// **Phase 5 Status**: Automatic pooling for all operations.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn with_pooled_buffer<T, F, R>(&mut self, shape: &[usize], f: F) -> Result<R>
    where
        T: Clone + std::default::Default + 'static,
        F: FnOnce(Vec<T>) -> Result<R>,
    {
        let buffer = self.acquire_pooled_generic::<T>(shape);
        let result = f(buffer.clone());
        self.release_pooled_generic::<T>(shape, buffer);
        result
    }

    // ========================================================================
    // End Generic Pooling Helpers
    // ========================================================================

    /// Execute einsum with planner integration
    pub(crate) fn execute_einsum_with_planner<T>(
        &mut self,
        spec: &EinsumSpec,
        inputs: &[DenseND<T>],
        _hints: &ExecHints,
    ) -> Result<DenseND<T>>
    where
        T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
    {
        let shapes: Vec<Vec<usize>> = inputs.iter().map(|t| t.shape().to_vec()).collect();
        let plan_hints = PlanHints::default();
        let plan = greedy_planner(spec, &shapes, &plan_hints)?;
        if inputs.len() == 2 {
            return execute_dense_contraction(spec, &inputs[0], &inputs[1]);
        }
        self.execute_plan(&plan, inputs)
    }
    /// Execute binary operation with full NumPy-style broadcasting support
    pub(crate) fn binary_op_with_broadcast<T>(
        &mut self,
        op: BinaryOp,
        x: &DenseND<T>,
        y: &DenseND<T>,
    ) -> Result<TensorHandle<T>>
    where
        T: Clone
            + Num
            + std::ops::AddAssign
            + std::default::Default
            + Float
            + FromPrimitive
            + 'static,
    {
        let x_shape = x.shape();
        let y_shape = y.shape();
        let output_shape = self.broadcast_shapes(x_shape, y_shape)?;
        let x_is_scalar = x_shape.is_empty() || (x_shape.len() == 1 && x_shape[0] == 1);
        let y_is_scalar = y_shape.is_empty() || (y_shape.len() == 1 && y_shape[0] == 1);
        if x_is_scalar {
            let x_val = if x_shape.is_empty() {
                x.view()[[]]
            } else {
                x.view()[[0]]
            };
            let result_data = match op {
                BinaryOp::Add => y.view().mapv(|y_val| x_val + y_val),
                BinaryOp::Sub => y.view().mapv(|y_val| x_val - y_val),
                BinaryOp::Mul => y.view().mapv(|y_val| x_val * y_val),
                BinaryOp::Div => y.view().mapv(|y_val| x_val / y_val),
                BinaryOp::Pow => y.view().mapv(|y_val| x_val.powf(y_val)),
                BinaryOp::Maximum => y
                    .view()
                    .mapv(|y_val| if x_val > y_val { x_val } else { y_val }),
                BinaryOp::Minimum => y
                    .view()
                    .mapv(|y_val| if x_val < y_val { x_val } else { y_val }),
            };
            return Ok(TensorHandle::from_dense_auto(DenseND::from_array(
                result_data,
            )));
        }
        if y_is_scalar {
            let y_val = if y_shape.is_empty() {
                y.view()[[]]
            } else {
                y.view()[[0]]
            };
            let result_data = match op {
                BinaryOp::Add => x.view().mapv(|x_val| x_val + y_val),
                BinaryOp::Sub => x.view().mapv(|x_val| x_val - y_val),
                BinaryOp::Mul => x.view().mapv(|x_val| x_val * y_val),
                BinaryOp::Div => x.view().mapv(|x_val| x_val / y_val),
                BinaryOp::Pow => x.view().mapv(|x_val| x_val.powf(y_val)),
                BinaryOp::Maximum => x
                    .view()
                    .mapv(|x_val| if x_val > y_val { x_val } else { y_val }),
                BinaryOp::Minimum => x
                    .view()
                    .mapv(|x_val| if x_val < y_val { x_val } else { y_val }),
            };
            return Ok(TensorHandle::from_dense_auto(DenseND::from_array(
                result_data,
            )));
        }
        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let output_size: usize = output_shape.iter().product();

        // Use pooled buffer for output allocation (Phase 5: Automatic Pooling)
        let mut output_data = self.acquire_pooled_generic::<T>(&output_shape);
        output_data.clear(); // Ensure buffer starts empty

        for flat_idx in 0..output_size {
            let out_idx = self.flat_to_multidim(flat_idx, &output_shape);
            let x_idx = self.broadcast_index(&out_idx, x_shape, &output_shape);
            let y_idx = self.broadcast_index(&out_idx, y_shape, &output_shape);
            let x_val = x.view()[x_idx.as_slice()];
            let y_val = y.view()[y_idx.as_slice()];
            let result_val = match op {
                BinaryOp::Add => x_val + y_val,
                BinaryOp::Sub => x_val - y_val,
                BinaryOp::Mul => x_val * y_val,
                BinaryOp::Div => x_val / y_val,
                BinaryOp::Pow => x_val.powf(y_val),
                BinaryOp::Maximum => {
                    if x_val > y_val {
                        x_val
                    } else {
                        y_val
                    }
                }
                BinaryOp::Minimum => {
                    if x_val < y_val {
                        x_val
                    } else {
                        y_val
                    }
                }
            };
            output_data.push(result_val);
        }

        // Create result array and release buffer back to pool
        let result_array = Array::from_shape_vec(IxDyn(&output_shape), output_data.clone())
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        self.release_pooled_generic::<T>(&output_shape, output_data);
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    /// Convert flat index to multi-dimensional index
    pub(crate) fn flat_to_multidim(&self, flat_idx: usize, shape: &[usize]) -> Vec<usize> {
        let mut idx = Vec::with_capacity(shape.len());
        let mut remaining = flat_idx;
        for &dim_size in shape.iter().rev() {
            idx.push(remaining % dim_size);
            remaining /= dim_size;
        }
        idx.reverse();
        idx
    }
    /// Convert multi-dimensional index to flat index
    pub(crate) fn multidim_to_flat(&self, idx: &[usize], shape: &[usize]) -> usize {
        let mut flat_idx = 0;
        let mut multiplier = 1;
        for i in (0..shape.len()).rev() {
            flat_idx += idx[i] * multiplier;
            multiplier *= shape[i];
        }
        flat_idx
    }
    /// Map output index to input index with broadcasting
    fn broadcast_index(
        &self,
        out_idx: &[usize],
        in_shape: &[usize],
        out_shape: &[usize],
    ) -> Vec<usize> {
        let mut in_idx = Vec::with_capacity(in_shape.len());
        let ndim_diff = out_shape.len() - in_shape.len();
        for (i, &in_dim) in in_shape.iter().enumerate() {
            let out_i = i + ndim_diff;
            if in_dim == 1 {
                in_idx.push(0);
            } else {
                in_idx.push(out_idx[out_i]);
            }
        }
        in_idx
    }
    /// Compute broadcast shape for two shapes
    fn broadcast_shapes(&self, x_shape: &[usize], y_shape: &[usize]) -> Result<Vec<usize>> {
        let max_ndim = x_shape.len().max(y_shape.len());
        let mut result_shape = Vec::with_capacity(max_ndim);
        for i in 0..max_ndim {
            let x_dim = if i < x_shape.len() {
                x_shape[x_shape.len() - 1 - i]
            } else {
                1
            };
            let y_dim = if i < y_shape.len() {
                y_shape[y_shape.len() - 1 - i]
            } else {
                1
            };
            if x_dim == y_dim || x_dim == 1 || y_dim == 1 {
                result_shape.push(x_dim.max(y_dim));
            } else {
                return Err(anyhow!(
                    "Shapes {:?} and {:?} are not broadcast-compatible at dimension {}",
                    x_shape,
                    y_shape,
                    i
                ));
            }
        }
        result_shape.reverse();
        Ok(result_shape)
    }
    /// Execute a multi-step contraction plan
    fn execute_plan<T>(&mut self, plan: &Plan, inputs: &[DenseND<T>]) -> Result<DenseND<T>>
    where
        T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
    {
        let mut intermediates: Vec<DenseND<T>> = inputs.to_vec();
        for (step_idx, &(i, j)) in plan.order.iter().enumerate() {
            if i >= intermediates.len() || j >= intermediates.len() {
                return Err(anyhow!(
                    "Step {}: Invalid indices ({}, {}) for {} intermediates",
                    step_idx,
                    i,
                    j,
                    intermediates.len()
                ));
            }
            let node = &plan.nodes[step_idx];
            let (tensor_a, tensor_b) = if i < j {
                let b = intermediates.remove(j);
                let a = intermediates.remove(i);
                (a, b)
            } else {
                let a = intermediates.remove(i);
                let b = intermediates.remove(j);
                (a, b)
            };
            let spec_str = format!(
                "{},{}->{}",
                node.output_spec.input_specs[0],
                node.output_spec.input_specs[1],
                node.output_spec.output_spec
            );
            let step_spec = EinsumSpec::parse(&spec_str)?;
            let result = execute_dense_contraction(&step_spec, &tensor_a, &tensor_b)?;
            intermediates.push(result);
        }
        if intermediates.len() != 1 {
            return Err(anyhow!(
                "Expected 1 final tensor, got {}",
                intermediates.len()
            ));
        }
        Ok(intermediates.into_iter().next().unwrap())
    }
    /// Helper: Compute determinant of a 2D matrix using LU decomposition
    pub(crate) fn compute_determinant_2d<T2>(
        &self,
        matrix: &scirs2_core::ndarray_ext::Array2<T2>,
    ) -> Result<T2>
    where
        T2: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
    {
        let n = matrix.nrows();
        if n == 0 {
            return Ok(T2::one());
        }
        if n == 1 {
            return Ok(matrix[[0, 0]]);
        }
        if n == 2 {
            let a = matrix[[0, 0]];
            let b = matrix[[0, 1]];
            let c = matrix[[1, 0]];
            let d = matrix[[1, 1]];
            return Ok(a * d - b * c);
        }
        let mut a = matrix.clone();
        let mut det = T2::one();
        let mut sign = T2::one();
        for i in 0..n {
            let mut pivot = i;
            let mut max_val = a[[i, i]].abs();
            for k in (i + 1)..n {
                let val = a[[k, i]].abs();
                if val > max_val {
                    max_val = val;
                    pivot = k;
                }
            }
            if max_val < T2::from_f64(1e-10).unwrap() {
                return Ok(T2::zero());
            }
            if pivot != i {
                for j in 0..n {
                    let temp = a[[i, j]];
                    a[[i, j]] = a[[pivot, j]];
                    a[[pivot, j]] = temp;
                }
                sign = -sign;
            }
            det = det * a[[i, i]];
            for k in (i + 1)..n {
                let factor = a[[k, i]] / a[[i, i]];
                for j in i..n {
                    a[[k, j]] = a[[k, j]] - factor * a[[i, j]];
                }
            }
        }
        Ok(sign * det)
    }
    /// Helper: Compute inverse of a 2D matrix using Gauss-Jordan elimination
    pub(crate) fn compute_inverse_2d<T2>(
        &self,
        matrix: &scirs2_core::ndarray_ext::Array2<T2>,
    ) -> Result<scirs2_core::ndarray_ext::Array2<T2>>
    where
        T2: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
    {
        use scirs2_core::ndarray_ext::Array2;
        let n = matrix.nrows();
        if n == 0 {
            return Err(anyhow!("Cannot invert empty matrix"));
        }
        let mut aug = Array2::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = matrix[[i, j]];
            }
            aug[[i, n + i]] = T2::one();
        }
        for i in 0..n {
            let mut pivot = i;
            let mut max_val = aug[[i, i]].abs();
            for k in (i + 1)..n {
                let val = aug[[k, i]].abs();
                if val > max_val {
                    max_val = val;
                    pivot = k;
                }
            }
            if max_val < T2::from_f64(1e-10).unwrap() {
                return Err(anyhow!("Matrix is singular and cannot be inverted"));
            }
            if pivot != i {
                for j in 0..(2 * n) {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[pivot, j]];
                    aug[[pivot, j]] = temp;
                }
            }
            let pivot_val = aug[[i, i]];
            for j in 0..(2 * n) {
                aug[[i, j]] = aug[[i, j]] / pivot_val;
            }
            for k in 0..n {
                if k != i {
                    let factor = aug[[k, i]];
                    for j in 0..(2 * n) {
                        aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
                    }
                }
            }
        }
        let mut inv = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = aug[[i, n + j]];
            }
        }
        Ok(inv)
    }
    /// Helper: Solve linear system Ax = b using LU decomposition
    pub(crate) fn solve_2d_1d<T2>(
        &self,
        a: &scirs2_core::ndarray_ext::Array2<T2>,
        b: &scirs2_core::ndarray_ext::Array1<T2>,
    ) -> Result<scirs2_core::ndarray_ext::Array1<T2>>
    where
        T2: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
    {
        use scirs2_core::ndarray_ext::Array1;
        let n = a.nrows();
        if n != b.len() {
            return Err(anyhow!("Dimension mismatch in solve"));
        }
        let mut a_work = a.clone();
        let mut b_work = b.clone();
        for i in 0..n {
            let mut pivot = i;
            let mut max_val = a_work[[i, i]].abs();
            for k in (i + 1)..n {
                let val = a_work[[k, i]].abs();
                if val > max_val {
                    max_val = val;
                    pivot = k;
                }
            }
            if max_val < T2::from_f64(1e-10).unwrap() {
                return Err(anyhow!("Matrix is singular, cannot solve"));
            }
            if pivot != i {
                for j in 0..n {
                    let temp = a_work[[i, j]];
                    a_work[[i, j]] = a_work[[pivot, j]];
                    a_work[[pivot, j]] = temp;
                }
                let temp = b_work[i];
                b_work[i] = b_work[pivot];
                b_work[pivot] = temp;
            }
            for k in (i + 1)..n {
                let factor = a_work[[k, i]] / a_work[[i, i]];
                for j in i..n {
                    a_work[[k, j]] = a_work[[k, j]] - factor * a_work[[i, j]];
                }
                b_work[k] = b_work[k] - factor * b_work[i];
            }
        }
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = b_work[i];
            for j in (i + 1)..n {
                sum = sum - a_work[[i, j]] * x[j];
            }
            x[i] = sum / a_work[[i, i]];
        }
        Ok(x)
    }
}
/// Element-wise operation types
#[derive(Clone, Debug)]
pub enum ElemOp {
    /// Negation: -x
    Neg,
    /// Absolute value: |x|
    Abs,
    /// Exponential: e^x
    Exp,
    /// Natural logarithm: ln(x)
    Log,
    /// Sine: sin(x)
    Sin,
    /// Cosine: cos(x)
    Cos,
    /// Square root: sqrt(x)
    Sqrt,
    /// Power of 2: x^2
    Sqr,
    /// Reciprocal: 1/x
    Recip,
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    /// Sigmoid: 1 / (1 + e^(-x))
    Sigmoid,
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Gaussian Error Linear Unit: x * Φ(x) where Φ is the CDF of standard normal
    /// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    Gelu,
    /// Exponential Linear Unit: x if x > 0, else e^x - 1
    Elu,
    /// Scaled Exponential Linear Unit: scale * (x if x > 0, else alpha * (e^x - 1))
    /// where scale ≈ 1.0507, alpha ≈ 1.67326
    Selu,
    /// Softplus: ln(1 + e^x)
    Softplus,
    /// Sign function: -1 if x < 0, 0 if x == 0, 1 if x > 0
    Sign,
}
