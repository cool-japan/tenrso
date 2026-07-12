//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::hints::ExecHints;
use crate::ops::{execute_dense_contraction_accelerated, execute_unary_einsum};
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use std::collections::HashMap;
use tenrso_core::{DenseND, TensorHandle};
use tenrso_planner::{greedy_planner, EinsumSpec, PlanHints};

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
///
/// # Which knobs are live
///
/// Every configuration field on this struct changes the behaviour of a real call
/// path. There are no decorative flags:
///
/// | knob | what it actually changes |
/// |------|--------------------------|
/// | [`Self::enable_parallel`] | rayon dispatch in the element-wise / scalar / binary / reduction paths |
/// | [`Self::enable_simd`] | the AVX2 kernels in [`super::simd_ops`] for `exp`-family ops |
/// | [`Self::enable_blocked_reductions`] | the multi-accumulator reduction kernel in [`super::blocked_reductions`] |
/// | [`Self::enable_memory_pool`] | buffer reuse in the pooled allocation helpers |
/// | thread count (via [`Self::with_threads`]) | the private rayon pool every parallel region is installed into |
pub struct CpuExecutor {
    /// Memory pool for f32 tensors
    ///
    /// **Phase 2 Status**: Type-safe buffer pooling now operational.
    memory_pool_f32: MemoryPool<f32>,
    /// Memory pool for f64 tensors
    ///
    /// **Phase 2 Status**: Type-safe buffer pooling now operational.
    memory_pool_f64: MemoryPool<f64>,
    /// Requested thread count; 0 means "use rayon's ambient global pool".
    ///
    /// Private on purpose: the count and [`Self::thread_pool`] must not be able to
    /// disagree. Read it with [`Self::num_threads`], set it with
    /// [`Self::with_threads`], and ask what it actually resolves to with
    /// [`Self::effective_num_threads`].
    num_threads: usize,
    /// The executor's own rayon pool, when a specific thread count was requested.
    ///
    /// `None` = run on rayon's ambient global pool. Every parallel region in this
    /// crate's element-wise, scalar, binary and reduction paths goes through
    /// [`Self::install`], so this is what actually bounds their parallelism.
    thread_pool: Option<rayon::ThreadPool>,
    /// Enable parallel execution for large tensors
    pub enable_parallel: bool,
    /// Enable the AVX2 element-wise kernels for `exp`/`log`-family operations
    ///
    /// Only those ops have a SIMD kernel; see [`super::simd_ops`] for why the
    /// bandwidth-bound ops deliberately do not.
    pub enable_simd: bool,
    /// Enable the blocked (multi-accumulator) full-reduction kernel
    ///
    /// When off, reductions use the naive left-to-right accumulation order.
    /// Both orders are deterministic; see [`super::blocked_reductions`].
    pub enable_blocked_reductions: bool,
    /// Enable memory pooling
    pub enable_memory_pool: bool,
}
impl CpuExecutor {
    /// Create a new CPU executor with default settings
    ///
    /// All optimizations are enabled, and parallel regions run on rayon's ambient
    /// global pool. Use [`Self::with_threads`] to bound the thread count instead.
    pub fn new() -> Self {
        Self {
            memory_pool_f32: MemoryPool::new(),
            memory_pool_f64: MemoryPool::new(),
            num_threads: 0,
            thread_pool: None,
            enable_parallel: true,
            enable_simd: true,
            enable_blocked_reductions: true,
            enable_memory_pool: true,
        }
    }

    /// Create a CPU executor that runs its parallel regions on a private rayon
    /// pool of exactly `num_threads` threads.
    ///
    /// This is a real bound, not a hint: the pool is constructed here and every
    /// parallel region in the element-wise, scalar, binary and reduction paths is
    /// `install`ed into it, so `rayon::current_num_threads()` inside them reports
    /// `num_threads` and no more than `num_threads` workers ever run.
    /// [`Self::effective_num_threads`] reports the count that will actually be used.
    ///
    /// `num_threads == 0` means "use rayon's ambient global pool" (auto-detect),
    /// which is what [`Self::new`] does.
    ///
    /// # Errors
    ///
    /// Returns an error if the OS refuses to spawn the worker threads. It does not
    /// silently fall back to the ambient pool — a thread bound that is quietly
    /// ignored is worse than a loud failure.
    ///
    /// # Note
    ///
    /// This bounds the executor's own data-parallel regions. The einsum/GEMM
    /// contraction path in [`crate::ops`] is not routed through this pool (its
    /// generic bounds do not carry `Send`), and still uses the ambient pool.
    pub fn with_threads(num_threads: usize) -> Result<Self> {
        let thread_pool = if num_threads == 0 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .map_err(|e| {
                        anyhow!("failed to build a rayon pool of {num_threads} threads: {e}")
                    })?,
            )
        };

        Ok(Self {
            memory_pool_f32: MemoryPool::new(),
            memory_pool_f64: MemoryPool::new(),
            num_threads,
            thread_pool,
            enable_parallel: true,
            enable_simd: true,
            enable_blocked_reductions: true,
            enable_memory_pool: true,
        })
    }

    /// Create a CPU executor with parallel execution disabled
    pub fn serial() -> Self {
        Self {
            memory_pool_f32: MemoryPool::new(),
            memory_pool_f64: MemoryPool::new(),
            num_threads: 1,
            thread_pool: None,
            enable_parallel: false,
            enable_simd: false,
            enable_blocked_reductions: false,
            enable_memory_pool: false,
        }
    }

    /// Create a CPU executor with all optimizations disabled (for debugging/testing)
    pub fn unoptimized() -> Self {
        Self {
            memory_pool_f32: MemoryPool::disabled(),
            memory_pool_f64: MemoryPool::disabled(),
            num_threads: 1,
            thread_pool: None,
            enable_parallel: false,
            enable_simd: false,
            enable_blocked_reductions: false,
            enable_memory_pool: false,
        }
    }

    /// The thread count this executor was configured with (0 = ambient pool).
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// The number of threads this executor's parallel regions will actually use.
    ///
    /// For a private pool that is its size; for the ambient pool it is whatever
    /// rayon's global pool currently has. This is the value a caller can hold the
    /// executor to, and the one the tests assert on.
    pub fn effective_num_threads(&self) -> usize {
        match &self.thread_pool {
            Some(pool) => pool.current_num_threads(),
            None => rayon::current_num_threads(),
        }
    }

    /// Run `f` inside this executor's thread pool.
    ///
    /// With a private pool this is `ThreadPool::install`, which makes every rayon
    /// operation nested inside `f` — including `rayon::current_num_threads()` —
    /// see that pool and no other. With no private pool it just calls `f`, leaving
    /// the work on rayon's ambient global pool.
    pub(crate) fn install<R, F>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match &self.thread_pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// Configure the AVX2 element-wise kernels
    pub fn with_simd(mut self, enabled: bool) -> Self {
        self.enable_simd = enabled;
        self
    }

    /// Configure the blocked full-reduction kernel
    pub fn with_blocked_reductions(mut self, enabled: bool) -> Self {
        self.enable_blocked_reductions = enabled;
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
            // TypeId check above proves T is exactly f32.
            let mut buffer_f32 = self.memory_pool_f32.acquire(shape);
            let len = buffer_f32.len();
            let cap = buffer_f32.capacity();
            let ptr = buffer_f32.as_mut_ptr() as *mut T;
            std::mem::forget(buffer_f32);
            // SAFETY: TypeId::of::<T>() == TypeId::of::<f32>() was verified above, so T and
            // f32 are the same type with identical size, alignment, and bit representation.
            // ptr, len, and cap come from a valid Vec<f32> allocation; forget() prevents
            // double-free. Violating this: if T were a different type that happened to match
            // via TypeId spoofing (impossible in safe Rust), the reinterpretation would be UB.
            unsafe { Vec::from_raw_parts(ptr, len, cap) }
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            // TypeId check above proves T is exactly f64.
            let mut buffer_f64 = self.memory_pool_f64.acquire(shape);
            let len = buffer_f64.len();
            let cap = buffer_f64.capacity();
            let ptr = buffer_f64.as_mut_ptr() as *mut T;
            std::mem::forget(buffer_f64);
            // SAFETY: TypeId::of::<T>() == TypeId::of::<f64>() was verified above, so T and
            // f64 are the same type with identical size, alignment, and bit representation.
            // ptr, len, and cap come from a valid Vec<f64> allocation; forget() prevents
            // double-free. Violating this: same argument as f32 branch above.
            unsafe { Vec::from_raw_parts(ptr, len, cap) }
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
            // TypeId check above proves T is exactly f32.
            let mut buffer = buffer;
            let len = buffer.len();
            let cap = buffer.capacity();
            let ptr = buffer.as_mut_ptr() as *mut f32;
            std::mem::forget(buffer);
            // SAFETY: TypeId::of::<T>() == TypeId::of::<f32>() was verified above, so T and
            // f32 are the same type. ptr, len, cap come from a valid Vec<T> allocation;
            // forget() prevents double-free. The resulting Vec<f32> is returned to the pool
            // which allocated it, so allocator identity is preserved.
            let buffer_f32: Vec<f32> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
            self.memory_pool_f32.release(shape, buffer_f32);
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            // TypeId check above proves T is exactly f64.
            let mut buffer = buffer;
            let len = buffer.len();
            let cap = buffer.capacity();
            let ptr = buffer.as_mut_ptr() as *mut f64;
            std::mem::forget(buffer);
            // SAFETY: TypeId::of::<T>() == TypeId::of::<f64>() was verified above, so T and
            // f64 are the same type. ptr, len, cap come from a valid Vec<T> allocation;
            // forget() prevents double-free. The resulting Vec<f64> is returned to the pool
            // which allocated it, so allocator identity is preserved.
            let buffer_f64: Vec<f64> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
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

    /// Execute einsum with planner integration.
    ///
    /// # Arity dispatch
    ///
    /// | operands | path |
    /// |----------|------|
    /// | 1 | [`execute_unary_einsum`] — permute / diagonal / trace / reduce |
    /// | 2 | [`execute_dense_contraction_accelerated`] — one batched GEMM |
    /// | ≥3 | [`greedy_planner`] chooses the *order*; [`Self::execute_plan`] executes it |
    ///
    /// The single-operand case is short-circuited **before** planning, and
    /// deliberately so: a [`Plan`] models a sequence of *pairwise* contractions
    /// (`order: Vec<(usize, usize)>`, `ContractionSpec` with two input
    /// subscripts).  A one-operand einsum has no pair to schedule and no order
    /// to search over — the planner correctly returns an empty plan, and an
    /// empty plan executed over one input is a passthrough.  Routing `"ij->ji"`
    /// through the planner therefore *cannot* be made correct without changing
    /// the planner's data model to something it does not mean; the operation is
    /// a gather, so it belongs in the kernel layer next to the contraction
    /// kernel, which is where [`crate::ops::execute_unary_einsum`] lives.
    pub(crate) fn execute_einsum_with_planner<T>(
        &mut self,
        spec: &EinsumSpec,
        inputs: &[DenseND<T>],
        _hints: &ExecHints,
    ) -> Result<DenseND<T>>
    where
        T: Clone
            + Num
            + std::ops::AddAssign
            + std::default::Default
            + Float
            + FromPrimitive
            + 'static,
    {
        if spec.num_inputs() != inputs.len() {
            return Err(anyhow!(
                "Spec expects {} inputs, got {}",
                spec.num_inputs(),
                inputs.len()
            ));
        }

        match inputs {
            [] => Err(anyhow!("einsum requires at least one input tensor")),
            [a] => execute_unary_einsum(spec, a),
            [a, b] => execute_dense_contraction_accelerated(spec, a, b),
            _ => {
                let shapes: Vec<Vec<usize>> = inputs.iter().map(|t| t.shape().to_vec()).collect();
                let order = contraction_order(spec, &shapes);
                self.execute_plan(spec, &order, inputs)
            }
        }
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
        // Stretch both operands to the output shape with stride-0 views, then walk
        // them together.
        //
        // `broadcast` costs one `Dim` allocation per operand and moves no data: a
        // broadcast axis just gets stride 0, so the "repeated" element is re-read
        // from the same address and stays hot in L1.
        //
        // What this replaced walked the *flat output index* and rebuilt each
        // operand's subscripts from it — and `flat_to_multidim` and
        // `broadcast_index` each allocate a `Vec`, **per element**. An 8.4M-element
        // add therefore did ~25M heap allocations before touching any arithmetic.
        // Measured on (512,1,64) + (512,256,64), median of 5 on a contended box:
        // **1010 ms before, 62 ms after — 16x.**
        //
        // `Zip` iterates a stride-0 broadcast operand at memory speed (the swept
        // axes just re-read the same address); a bare `.iter()` on such a view,
        // by contrast, pays per-element index arithmetic and is ~7x slower here.
        // The output buffer comes from the pool (Phase 5 automatic pooling), and
        // `Zip` writes it in place through an `ArrayViewMut`, so the fill neither
        // allocates nor picks an order that could disagree between operands.
        use scirs2_core::ndarray_ext::{Array, ArrayViewMut, IxDyn, Zip};

        let out_dim = IxDyn(&output_shape);
        let x_b = x.as_array().broadcast(out_dim.clone()).ok_or_else(|| {
            anyhow!(
                "cannot broadcast {:?} to {:?}",
                x_shape.to_vec(),
                output_shape
            )
        })?;
        let y_b = y.as_array().broadcast(out_dim).ok_or_else(|| {
            anyhow!(
                "cannot broadcast {:?} to {:?}",
                y_shape.to_vec(),
                output_shape
            )
        })?;

        // Pooled scratch, sized to the output; reused on the next op of this shape.
        let mut output_data = self.acquire_pooled_generic::<T>(&output_shape);
        {
            let mut out_view = ArrayViewMut::from_shape(IxDyn(&output_shape), &mut output_data)
                .map_err(|e| anyhow!("Failed to view output buffer: {}", e))?;
            macro_rules! fill {
                ($f:expr) => {
                    Zip::from(&mut out_view)
                        .and(&x_b)
                        .and(&y_b)
                        .for_each(|o, &a, &b| *o = $f(a, b))
                };
            }
            match op {
                BinaryOp::Add => fill!(|a: T, b: T| a + b),
                BinaryOp::Sub => fill!(|a: T, b: T| a - b),
                BinaryOp::Mul => fill!(|a: T, b: T| a * b),
                BinaryOp::Div => fill!(|a: T, b: T| a / b),
                BinaryOp::Pow => fill!(|a: T, b: T| a.powf(b)),
                BinaryOp::Maximum => fill!(|a: T, b: T| if a > b { a } else { b }),
                BinaryOp::Minimum => fill!(|a: T, b: T| if a < b { a } else { b }),
            }
        }

        // Hand a copy to the result and return the buffer to the pool, so the next
        // op of this shape is a pool hit.
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
    /// Execute a multi-operand contraction in the given pairwise `order`.
    ///
    /// # Why the plan's step specs are re-derived here
    ///
    /// The planner's job is to choose the *order* — which pair to contract next,
    /// by cost.  The *semantics* of each step (which indices survive it) are not
    /// a free choice: an index must survive a step iff it is still needed, i.e.
    /// it appears in the final output **or** in an operand that has not been
    /// consumed yet.  `greedy_planner`'s `compute_pairwise_spec` decides this
    /// from the two operands alone: it drops every index the pair *shares*.
    /// That is wrong for a shared index which is also needed later — most
    /// visibly a batch index, so `"bij,bjk,bkl->bil"` would sum over `b` in the
    /// first step and silently return a wrong-valued (but right-shaped!) tensor.
    /// It also emits its output indices in alphabetical order, which need not be
    /// the caller's requested order.
    ///
    /// So the executor takes only `order` from the plan and derives each step's
    /// subscripts itself, then fixes up the final index order with a unary
    /// einsum.  This keeps the fix inside the layer that owns correctness while
    /// leaving cost-based ordering to the planner.
    fn execute_plan<T>(
        &mut self,
        spec: &EinsumSpec,
        order: &[(usize, usize)],
        inputs: &[DenseND<T>],
    ) -> Result<DenseND<T>>
    where
        T: Clone
            + Num
            + std::ops::AddAssign
            + std::default::Default
            + Float
            + FromPrimitive
            + 'static,
    {
        let mut labels: Vec<String> = spec.inputs.clone();
        let mut tensors: Vec<DenseND<T>> = inputs.to_vec();

        for (step_idx, &(i, j)) in order.iter().enumerate() {
            if i == j || i >= tensors.len() || j >= tensors.len() {
                return Err(anyhow!(
                    "Step {}: invalid contraction pair ({}, {}) for {} operands",
                    step_idx,
                    i,
                    j,
                    tensors.len()
                ));
            }

            // Remove the higher index first so the lower one stays valid.
            let (hi, lo) = if i > j { (i, j) } else { (j, i) };
            let tensor_hi = tensors.remove(hi);
            let labels_hi = labels.remove(hi);
            let tensor_lo = tensors.remove(lo);
            let labels_lo = labels.remove(lo);

            // `i` is always operand A and `j` operand B, whichever came first.
            let (labels_a, tensor_a, labels_b, tensor_b) = if i == hi {
                (labels_hi, tensor_hi, labels_lo, tensor_lo)
            } else {
                (labels_lo, tensor_lo, labels_hi, tensor_hi)
            };

            let out_labels = step_output_labels(&labels_a, &labels_b, &labels, &spec.output);
            let result = contract_pair(&labels_a, &tensor_a, &labels_b, &tensor_b, &out_labels)?;

            labels.push(out_labels);
            tensors.push(result);
        }

        if tensors.len() != 1 || labels.len() != 1 {
            return Err(anyhow!(
                "Expected 1 final tensor, got {} (the contraction order left the network \
                 unreduced)",
                tensors.len()
            ));
        }
        let result = tensors
            .pop()
            .ok_or_else(|| anyhow!("BUG: expected exactly 1 intermediate after guard check"))?;
        let final_labels = labels
            .pop()
            .ok_or_else(|| anyhow!("BUG: expected exactly 1 label set after guard check"))?;

        if final_labels == spec.output {
            return Ok(result);
        }
        if final_labels.is_empty() {
            return Err(anyhow!(
                "BUG: the contraction reduced to a scalar but the spec requests output '{}'",
                spec.output
            ));
        }

        // The steps kept exactly the output's index set, but not necessarily in
        // the requested order (`"ij,jk,kl->li"`).  One unary gather fixes it.
        let reorder = EinsumSpec::parse(&format!("{}->{}", final_labels, spec.output))?;
        execute_unary_einsum(&reorder, &result)
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
            if max_val
                < T2::from_f64(1e-10)
                    .expect("FromPrimitive invariant: 1e-10 representable in float type")
            {
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
            if max_val
                < T2::from_f64(1e-10)
                    .expect("FromPrimitive invariant: 1e-10 representable in float type")
            {
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
            if max_val
                < T2::from_f64(1e-10)
                    .expect("FromPrimitive invariant: 1e-10 representable in float type")
            {
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
// ────────────────────────── multi-operand contraction ───────────────────────

/// Choose the pairwise contraction order for a ≥3-operand einsum.
///
/// Asks [`greedy_planner`] for a cost-based order and validates it against the
/// remove-two-push-one discipline [`CpuExecutor::execute_plan`] uses.  If the
/// planner fails (it rejects a few valid specs — e.g. any pairwise step whose
/// operands share *all* their indices, which makes its intermediate subscript
/// empty and unparseable) or returns an order that is not executable, fall back
/// to a left-to-right chain, which is always a valid order.  Order only affects
/// cost, never the result, so a fallback can never make an answer wrong — it can
/// only make it slower than the planner would have.
fn contraction_order(spec: &EinsumSpec, shapes: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let n = shapes.len();
    let sequential =
        || -> Vec<(usize, usize)> { (0..n.saturating_sub(1)).map(|_| (0, 1)).collect() };

    match greedy_planner(spec, shapes, &PlanHints::default()) {
        Ok(plan) if is_executable_order(&plan.order, n) => plan.order,
        _ => sequential(),
    }
}

/// Is `order` a valid sequence of pairwise contractions over `n` operands?
///
/// Each step removes two tensors and pushes one, so the operand count drops by
/// one per step and there must be exactly `n - 1` steps, each naming two
/// distinct in-range positions.
fn is_executable_order(order: &[(usize, usize)], n: usize) -> bool {
    if n == 0 || order.len() != n - 1 {
        return false;
    }
    let mut remaining = n;
    for &(i, j) in order {
        if i == j || i >= remaining || j >= remaining {
            return false;
        }
        remaining -= 1;
    }
    remaining == 1
}

/// The index subscript of one contraction step's output.
///
/// An index survives the step iff it is still needed afterwards: it appears in
/// the final output, or in an operand that has not been consumed yet.  Every
/// other index of the pair is contracted (shared) or summed out (unshared) by
/// this step — which is exactly what the pairwise engine does with them.
///
/// Emitted in order of first appearance in `labels_a` then `labels_b`, with
/// duplicates dropped (a repeated index inside one operand is its diagonal, and
/// contributes a single output axis).
fn step_output_labels(
    labels_a: &str,
    labels_b: &str,
    remaining: &[String],
    final_output: &str,
) -> String {
    let mut out = String::new();
    for c in labels_a.chars().chain(labels_b.chars()) {
        if out.contains(c) {
            continue;
        }
        let needed_later = final_output.contains(c) || remaining.iter().any(|l| l.contains(c));
        if needed_later {
            out.push(c);
        }
    }
    out
}

/// Contract one pair of labelled tensors into `out_labels`.
///
/// Delegates to the pairwise engine, except when an operand is a rank-0 scalar —
/// which a previous step can legitimately produce (`"ij,ij,kl->kl"` contracts
/// the first pair to a scalar).  `EinsumSpec` cannot express an empty subscript,
/// so that degenerate case is handled directly: scale, then reduce.
fn contract_pair<T>(
    labels_a: &str,
    tensor_a: &DenseND<T>,
    labels_b: &str,
    tensor_b: &DenseND<T>,
    out_labels: &str,
) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    if !labels_a.is_empty() && !labels_b.is_empty() {
        let spec = EinsumSpec::parse(&format!("{},{}->{}", labels_a, labels_b, out_labels))?;
        return execute_dense_contraction_accelerated(&spec, tensor_a, tensor_b);
    }

    // At least one operand is a scalar: the contraction degenerates to a scaling
    // (plus whatever reduction `out_labels` still asks for).
    let (scalar, other, other_labels) = if labels_a.is_empty() {
        (tensor_a, tensor_b, labels_b)
    } else {
        (tensor_b, tensor_a, labels_a)
    };
    let scale = first_element(scalar, "scalar einsum operand")?;

    if other_labels.is_empty() {
        let value = first_element(other, "scalar einsum operand")?;
        return DenseND::from_vec(vec![scale * value], &[]);
    }

    let scaled = DenseND::from_array(other.as_array().mapv(|v| v * scale));
    let spec = EinsumSpec::parse(&format!("{}->{}", other_labels, out_labels))?;
    execute_unary_einsum(&spec, &scaled)
}

/// The single element of a rank-0 tensor.
fn first_element<T: Copy + Num>(tensor: &DenseND<T>, context: &str) -> Result<T> {
    tensor
        .as_array()
        .iter()
        .next()
        .copied()
        .ok_or_else(|| anyhow!("{} is empty", context))
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
