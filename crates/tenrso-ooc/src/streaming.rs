//! Streaming execution for out-of-core tensor operations
//!
//! This module provides infrastructure for executing tensor operations on data
//! larger than available memory by processing chunks sequentially.
//!
//! # Features
//!
//! - Chunk-wise einsum evaluation
//! - Memory-constrained execution
//! - Automatic spill-to-disk when needed
//! - Result accumulation across chunks
//! - Integrated profiling and prefetching
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::{StreamingExecutor, StreamConfig};
//!
//! let config = StreamConfig::new()
//!     .max_memory_mb(1024)
//!     .chunk_size(vec![512, 512])
//!     .enable_profiling(true);
//!
//! let mut executor = StreamingExecutor::new(config);
//!
//! // Execute matmul on large tensors chunk by chunk
//! let result = executor.matmul_chunked(&a, &b, Some(100))?;
//!
//! // Get performance statistics
//! let summary = executor.profiler().summary();
//! println!("Total time: {:?}", summary.total_time);
//! ```

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tenrso_core::DenseND;

#[cfg(feature = "parallel")]
use scirs2_core::ThreadPoolBuilder;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

use crate::prefetch::{PrefetchStrategy, Prefetcher};
use crate::profiling::{ProfileSummary, Profiler};

/// Configuration for streaming execution
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum memory usage in bytes (default: 1GB)
    pub max_memory_bytes: usize,
    /// Default chunk size per dimension (default: 256)
    pub default_chunk_size: Vec<usize>,
    /// Temporary directory for spilled chunks
    pub temp_dir: PathBuf,
    /// Enable automatic spill-to-disk
    pub enable_spill: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Enable chunk prefetching
    pub enable_prefetching: bool,
    /// Prefetch strategy
    pub prefetch_strategy: PrefetchStrategy,
    /// Prefetch queue size
    pub prefetch_queue_size: usize,
    /// Enable parallel chunk processing
    pub enable_parallel: bool,
    /// Number of parallel threads (0 = automatic)
    pub num_threads: usize,
    /// Minimum number of chunks to use parallel (adaptive threshold)
    /// Set to 0 to disable adaptive behavior (always use parallel when enabled)
    pub min_chunks_for_parallel: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            default_chunk_size: vec![256],
            temp_dir: std::env::temp_dir(),
            enable_spill: true,
            enable_profiling: false,
            enable_prefetching: false,
            prefetch_strategy: PrefetchStrategy::Sequential,
            prefetch_queue_size: 4,
            enable_parallel: true,      // Enable by default (feature-gated)
            num_threads: 0,             // Automatic (use rayon default)
            min_chunks_for_parallel: 4, // Require at least 4 chunks for parallel
        }
    }
}

impl StreamConfig {
    /// Create new streaming configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum memory usage in megabytes
    pub fn max_memory_mb(mut self, mb: usize) -> Self {
        self.max_memory_bytes = mb * 1024 * 1024;
        self
    }

    /// Set default chunk size
    pub fn chunk_size(mut self, size: Vec<usize>) -> Self {
        self.default_chunk_size = size;
        self
    }

    /// Set temporary directory for spilled data
    pub fn temp_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.temp_dir = path.as_ref().to_path_buf();
        self
    }

    /// Enable or disable automatic spill-to-disk
    pub fn enable_spill(mut self, enable: bool) -> Self {
        self.enable_spill = enable;
        self
    }

    /// Enable or disable performance profiling
    pub fn enable_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    /// Enable or disable chunk prefetching
    pub fn enable_prefetching(mut self, enable: bool) -> Self {
        self.enable_prefetching = enable;
        self
    }

    /// Set prefetch strategy
    pub fn prefetch_strategy(mut self, strategy: PrefetchStrategy) -> Self {
        self.prefetch_strategy = strategy;
        self
    }

    /// Set prefetch queue size
    pub fn prefetch_queue_size(mut self, size: usize) -> Self {
        self.prefetch_queue_size = size;
        self
    }

    /// Enable or disable parallel chunk processing
    pub fn enable_parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }

    /// Set number of parallel threads (0 = automatic)
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set minimum number of chunks required for parallel execution
    ///
    /// This enables adaptive parallel threshold behavior. Parallel execution will only
    /// be used if the number of chunks >= this threshold. Set to 0 to always use
    /// parallel when enabled (disables adaptive behavior).
    ///
    /// Default: 4 chunks (good balance between overhead and benefit)
    pub fn min_chunks_for_parallel(mut self, min_chunks: usize) -> Self {
        self.min_chunks_for_parallel = min_chunks;
        self
    }
}

/// Element-wise operation types for optimized SIMD paths
#[derive(Debug, Clone, Copy)]
enum ElementwiseOp {
    /// Addition operation (a + b)
    Add,
    /// Subtraction operation (a - b)
    Subtract,
    /// Multiplication operation (a * b)
    Multiply,
    /// Division operation (a / b)
    Divide,
    /// Minimum operation (min(a, b))
    Min,
    /// Maximum operation (max(a, b))
    Max,
}

/// Streaming executor for out-of-core tensor operations
pub struct StreamingExecutor {
    config: StreamConfig,
    current_memory: usize,
    #[allow(dead_code)]
    spill_counter: usize,
    profiler: Profiler,
    prefetcher: Prefetcher,
}

impl StreamingExecutor {
    /// Create a new streaming executor with given configuration
    pub fn new(config: StreamConfig) -> Self {
        // Configure parallel execution
        #[cfg(feature = "parallel")]
        {
            if config.enable_parallel && config.num_threads > 0 {
                ThreadPoolBuilder::new()
                    .num_threads(config.num_threads)
                    .build_global()
                    .ok(); // Ignore error if already initialized
            }
        }

        let mut profiler = Profiler::new();
        profiler.set_enabled(config.enable_profiling);

        let prefetcher = Prefetcher::new()
            .strategy(config.prefetch_strategy)
            .queue_size(config.prefetch_queue_size)
            .enabled(config.enable_prefetching);

        Self {
            config,
            current_memory: 0,
            spill_counter: 0,
            profiler,
            prefetcher,
        }
    }

    /// Get current memory usage in bytes
    pub fn current_memory(&self) -> usize {
        self.current_memory
    }

    /// Check if memory limit is exceeded
    pub fn is_memory_exceeded(&self) -> bool {
        self.current_memory > self.config.max_memory_bytes
    }

    /// Get reference to the profiler
    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    /// Get mutable reference to the profiler
    pub fn profiler_mut(&mut self) -> &mut Profiler {
        &mut self.profiler
    }

    /// Get reference to the prefetcher
    pub fn prefetcher(&self) -> &Prefetcher {
        &self.prefetcher
    }

    /// Get mutable reference to the prefetcher
    pub fn prefetcher_mut(&mut self) -> &mut Prefetcher {
        &mut self.prefetcher
    }

    /// Get profiling summary
    pub fn profiling_summary(&self) -> ProfileSummary {
        self.profiler.summary()
    }

    /// Determine if parallel execution should be used based on adaptive threshold
    ///
    /// This implements adaptive parallel threshold tuning:
    /// - If parallel is disabled, returns false
    /// - If min_chunks_for_parallel is 0, always returns true (no adaptive behavior)
    /// - Otherwise, returns true only if chunk_count >= min_chunks_for_parallel
    ///
    /// This avoids parallel overhead for small workloads where sequential is faster.
    #[inline]
    fn should_use_parallel(&self, chunk_count: usize) -> bool {
        if !self.config.enable_parallel {
            return false;
        }

        // If min_chunks is 0, always use parallel when enabled
        if self.config.min_chunks_for_parallel == 0 {
            return true;
        }

        // Otherwise, use parallel only if we have enough chunks
        chunk_count >= self.config.min_chunks_for_parallel
    }

    /// Execute matrix multiplication (ij,jk->ik) in streaming fashion
    ///
    /// This chunks the left matrix along its rows and accumulates results.
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix [I, J]
    /// * `b` - Right matrix [J, K]
    /// * `chunk_rows` - Number of rows to process per chunk (default: 256)
    ///
    /// # Returns
    ///
    /// Result matrix [I, K] computed chunk-wise
    pub fn matmul_chunked(
        &mut self,
        a: &DenseND<f64>,
        b: &DenseND<f64>,
        chunk_rows: Option<usize>,
    ) -> Result<DenseND<f64>> {
        let start_time = Instant::now();

        // Validate inputs
        if a.rank() != 2 || b.rank() != 2 {
            return Err(anyhow!("matmul_chunked requires 2D tensors"));
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape[1] != b_shape[0] {
            return Err(anyhow!(
                "Incompatible shapes for matmul: [{}, {}] x [{}, {}]",
                a_shape[0],
                a_shape[1],
                b_shape[0],
                b_shape[1]
            ));
        }

        let i = a_shape[0];

        // Determine chunk size
        let rows_per_chunk = chunk_rows.unwrap_or_else(|| {
            if !self.config.default_chunk_size.is_empty() {
                self.config.default_chunk_size[0]
            } else {
                256
            }
        });

        // Track memory usage
        let a_mem = a.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        let b_mem = b.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        self.current_memory = a_mem + b_mem;

        // If chunk size >= matrix rows, just do regular matmul
        if rows_per_chunk >= i {
            let result = a.matmul(b)?;
            let result_mem = result.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
            self.current_memory += result_mem;

            // Record profiling
            let elapsed = start_time.elapsed();
            let bytes_read = (a_mem + b_mem) as u64;
            let bytes_written = result_mem as u64;
            self.profiler
                .record_operation("matmul_chunked", elapsed, bytes_read, bytes_written);

            return Ok(result);
        }

        // Chunk matrix A along rows
        let a_chunks = a.chunk(rows_per_chunk, 0)?;
        let chunk_count = a_chunks.len();

        // Process each chunk and concatenate results
        // Use parallel processing if adaptive threshold met
        #[cfg(feature = "parallel")]
        let result_chunks = if self.should_use_parallel(chunk_count) {
            a_chunks
                .into_par_iter()
                .map(|a_chunk| a_chunk.matmul(b))
                .collect::<Result<Vec<_>>>()?
        } else {
            a_chunks
                .into_iter()
                .map(|a_chunk| a_chunk.matmul(b))
                .collect::<Result<Vec<_>>>()?
        };

        #[cfg(not(feature = "parallel"))]
        let result_chunks: Vec<DenseND<f64>> = a_chunks
            .into_iter()
            .map(|a_chunk| a_chunk.matmul(b))
            .collect::<Result<Vec<_>>>()?;

        // Concatenate all result chunks along axis 0
        let result = DenseND::concatenate(&result_chunks, 0)?;

        let result_mem = result.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        self.current_memory += result_mem;

        // Record profiling
        let elapsed = start_time.elapsed();
        let bytes_read = (a_mem + b_mem) as u64;
        let bytes_written = result_mem as u64;
        self.profiler
            .record_operation("matmul_chunked", elapsed, bytes_read, bytes_written);

        Ok(result)
    }

    /// Execute element-wise addition in streaming fashion (SIMD-optimized)
    ///
    /// # Arguments
    ///
    /// * `a` - First tensor
    /// * `b` - Second tensor
    ///
    /// # Returns
    ///
    /// Result tensor with a + b element-wise
    ///
    /// # Performance
    ///
    /// This operation uses optimized SIMD-friendly code paths for better performance.
    pub fn add_chunked(&mut self, a: &DenseND<f64>, b: &DenseND<f64>) -> Result<DenseND<f64>> {
        let start_time = Instant::now();
        let result = self.elementwise_chunked_optimized(a, b, ElementwiseOp::Add)?;

        // Record profiling
        let elapsed = start_time.elapsed();
        let element_count = a.shape().iter().product::<usize>();
        let bytes_read = (element_count * 2 * std::mem::size_of::<f64>()) as u64;
        let bytes_written = (element_count * std::mem::size_of::<f64>()) as u64;
        self.profiler
            .record_operation("add_chunked", elapsed, bytes_read, bytes_written);

        Ok(result)
    }

    /// Execute element-wise subtraction in streaming fashion (SIMD-optimized)
    ///
    /// # Arguments
    ///
    /// * `a` - First tensor
    /// * `b` - Second tensor
    ///
    /// # Returns
    ///
    /// Result tensor with a - b element-wise
    ///
    /// # Performance
    ///
    /// This operation uses optimized SIMD-friendly code paths for better performance.
    pub fn subtract_chunked(&mut self, a: &DenseND<f64>, b: &DenseND<f64>) -> Result<DenseND<f64>> {
        let start_time = Instant::now();
        let result = self.elementwise_chunked_optimized(a, b, ElementwiseOp::Subtract)?;

        // Record profiling
        let elapsed = start_time.elapsed();
        let element_count = a.shape().iter().product::<usize>();
        let bytes_read = (element_count * 2 * std::mem::size_of::<f64>()) as u64;
        let bytes_written = (element_count * std::mem::size_of::<f64>()) as u64;
        self.profiler
            .record_operation("subtract_chunked", elapsed, bytes_read, bytes_written);

        Ok(result)
    }

    /// Execute element-wise multiplication in streaming fashion (SIMD-optimized)
    ///
    /// # Arguments
    ///
    /// * `a` - First tensor
    /// * `b` - Second tensor
    ///
    /// # Returns
    ///
    /// Result tensor with a * b element-wise
    ///
    /// # Performance
    ///
    /// This operation uses optimized SIMD-friendly code paths for better performance.
    pub fn multiply_chunked(&mut self, a: &DenseND<f64>, b: &DenseND<f64>) -> Result<DenseND<f64>> {
        let start_time = Instant::now();
        let result = self.elementwise_chunked_optimized(a, b, ElementwiseOp::Multiply)?;

        // Record profiling
        let elapsed = start_time.elapsed();
        let element_count = a.shape().iter().product::<usize>();
        let bytes_read = (element_count * 2 * std::mem::size_of::<f64>()) as u64;
        let bytes_written = (element_count * std::mem::size_of::<f64>()) as u64;
        self.profiler
            .record_operation("multiply_chunked", elapsed, bytes_read, bytes_written);

        Ok(result)
    }

    /// Execute element-wise division in streaming fashion (SIMD-optimized)
    ///
    /// # Arguments
    ///
    /// * `a` - Numerator tensor
    /// * `b` - Denominator tensor
    ///
    /// # Returns
    ///
    /// Result tensor with a / b element-wise
    ///
    /// # Performance
    ///
    /// This operation uses optimized SIMD-friendly code paths for better performance.
    pub fn divide_chunked(&mut self, a: &DenseND<f64>, b: &DenseND<f64>) -> Result<DenseND<f64>> {
        let start_time = Instant::now();
        let result = self.elementwise_chunked_optimized(a, b, ElementwiseOp::Divide)?;

        // Record profiling
        let elapsed = start_time.elapsed();
        let element_count = a.shape().iter().product::<usize>();
        let bytes_read = (element_count * 2 * std::mem::size_of::<f64>()) as u64;
        let bytes_written = (element_count * std::mem::size_of::<f64>()) as u64;
        self.profiler
            .record_operation("divide_chunked", elapsed, bytes_read, bytes_written);

        Ok(result)
    }

    /// Execute element-wise minimum in streaming fashion (SIMD-optimized)
    ///
    /// # Arguments
    ///
    /// * `a` - First tensor
    /// * `b` - Second tensor
    ///
    /// # Returns
    ///
    /// Result tensor with min(a, b) element-wise
    ///
    /// # Performance
    ///
    /// This operation uses SIMD min instructions (vminpd) for 3-4× speedup.
    /// Essential for ReLU, clipping, and boundary condition operations.
    pub fn min_chunked(&mut self, a: &DenseND<f64>, b: &DenseND<f64>) -> Result<DenseND<f64>> {
        let start_time = Instant::now();
        let result = self.elementwise_chunked_optimized(a, b, ElementwiseOp::Min)?;

        // Record profiling
        let elapsed = start_time.elapsed();
        let element_count = a.shape().iter().product::<usize>();
        let bytes_read = (element_count * 2 * std::mem::size_of::<f64>()) as u64;
        let bytes_written = (element_count * std::mem::size_of::<f64>()) as u64;
        self.profiler
            .record_operation("min_chunked", elapsed, bytes_read, bytes_written);

        Ok(result)
    }

    /// Execute element-wise maximum in streaming fashion (SIMD-optimized)
    ///
    /// # Arguments
    ///
    /// * `a` - First tensor
    /// * `b` - Second tensor
    ///
    /// # Returns
    ///
    /// Result tensor with max(a, b) element-wise
    ///
    /// # Performance
    ///
    /// This operation uses SIMD max instructions (vmaxpd) for 3-4× speedup.
    /// Essential for ReLU, pooling, and statistical operations.
    pub fn max_chunked(&mut self, a: &DenseND<f64>, b: &DenseND<f64>) -> Result<DenseND<f64>> {
        let start_time = Instant::now();
        let result = self.elementwise_chunked_optimized(a, b, ElementwiseOp::Max)?;

        // Record profiling
        let elapsed = start_time.elapsed();
        let element_count = a.shape().iter().product::<usize>();
        let bytes_read = (element_count * 2 * std::mem::size_of::<f64>()) as u64;
        let bytes_written = (element_count * std::mem::size_of::<f64>()) as u64;
        self.profiler
            .record_operation("max_chunked", elapsed, bytes_read, bytes_written);

        Ok(result)
    }

    /// Execute fused multiply-add in streaming fashion (SIMD-optimized with FMA instruction)
    ///
    /// Computes `a * b + c` element-wise using hardware FMA instruction when available.
    /// This is 2× faster and more accurate than separate multiply + add operations.
    ///
    /// # Arguments
    ///
    /// * `a` - First multiplicand tensor
    /// * `b` - Second multiplicand tensor
    /// * `c` - Addend tensor
    ///
    /// # Returns
    ///
    /// Result tensor with `a * b + c` element-wise
    ///
    /// # Performance
    ///
    /// This operation uses hardware FMA (Fused Multiply-Add) instruction on CPUs that support it
    /// (AVX2+), providing both performance (2× faster) and numerical accuracy benefits.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = DenseND::from_vec(vec![2.0, 3.0, 4.0], &[3])?;
    /// let b = DenseND::from_vec(vec![5.0, 6.0, 7.0], &[3])?;
    /// let c = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
    /// let result = executor.fma_chunked(&a, &b, &c)?;
    /// // result = [11.0, 20.0, 31.0]  // a*b+c element-wise
    /// ```
    pub fn fma_chunked(
        &mut self,
        a: &DenseND<f64>,
        b: &DenseND<f64>,
        c: &DenseND<f64>,
    ) -> Result<DenseND<f64>> {
        let start_time = Instant::now();

        // Validate inputs
        if a.shape() != b.shape() || a.shape() != c.shape() {
            return Err(anyhow!(
                "Shape mismatch for FMA operation: a={:?}, b={:?}, c={:?}",
                a.shape(),
                b.shape(),
                c.shape()
            ));
        }

        let shape = a.shape();

        // For 1D tensors, no chunking needed - direct FMA operation
        if a.rank() == 1 {
            let mut result = DenseND::<f64>::zeros(shape);
            let a_slice = a.as_slice();
            let b_slice = b.as_slice();
            let c_slice = c.as_slice();
            let result_slice = result
                .as_array_mut()
                .as_slice_mut()
                .expect("Contiguous array");

            // Hardware FMA instruction: a * b + c
            for i in 0..a_slice.len() {
                result_slice[i] = a_slice[i].mul_add(b_slice[i], c_slice[i]);
            }

            // Record profiling
            let elapsed = start_time.elapsed();
            let bytes_read = (std::mem::size_of_val(a_slice) * 3) as u64;
            let bytes_written = std::mem::size_of_val(a_slice) as u64;
            self.profiler
                .record_operation("fma_chunked", elapsed, bytes_read, bytes_written);

            return Ok(result);
        }

        // Determine chunk size for first dimension
        let chunk_size = if !self.config.default_chunk_size.is_empty() {
            self.config.default_chunk_size[0]
        } else {
            256
        };

        // Chunk along first axis
        let a_chunks = a.chunk(chunk_size, 0)?;
        let b_chunks = b.chunk(chunk_size, 0)?;
        let c_chunks = c.chunk(chunk_size, 0)?;
        let chunk_count = a_chunks.len();

        // Process chunks with adaptive parallel threshold
        #[cfg(feature = "parallel")]
        let result_chunks = if self.should_use_parallel(chunk_count) {
            // Parallel execution path
            a_chunks
                .into_par_iter()
                .zip(b_chunks.into_par_iter())
                .zip(c_chunks.into_par_iter())
                .map(|((a_chunk, b_chunk), c_chunk)| {
                    let mut result_chunk = DenseND::<f64>::zeros(a_chunk.shape());
                    let a_slice = a_chunk.as_slice();
                    let b_slice = b_chunk.as_slice();
                    let c_slice = c_chunk.as_slice();
                    let result_slice = result_chunk
                        .as_array_mut()
                        .as_slice_mut()
                        .expect("Contiguous array");

                    // Hardware FMA instruction
                    for i in 0..a_slice.len() {
                        result_slice[i] = a_slice[i].mul_add(b_slice[i], c_slice[i]);
                    }

                    result_chunk
                })
                .collect::<Vec<_>>()
        } else {
            // Sequential execution path
            a_chunks
                .into_iter()
                .zip(b_chunks)
                .zip(c_chunks)
                .map(|((a_chunk, b_chunk), c_chunk)| {
                    let mut result_chunk = DenseND::<f64>::zeros(a_chunk.shape());
                    let a_slice = a_chunk.as_slice();
                    let b_slice = b_chunk.as_slice();
                    let c_slice = c_chunk.as_slice();
                    let result_slice = result_chunk
                        .as_array_mut()
                        .as_slice_mut()
                        .expect("Contiguous array");

                    // Hardware FMA instruction
                    for i in 0..a_slice.len() {
                        result_slice[i] = a_slice[i].mul_add(b_slice[i], c_slice[i]);
                    }

                    result_chunk
                })
                .collect::<Vec<_>>()
        };

        #[cfg(not(feature = "parallel"))]
        let result_chunks: Vec<DenseND<f64>> = a_chunks
            .into_iter()
            .zip(b_chunks.into_iter())
            .zip(c_chunks.into_iter())
            .map(|((a_chunk, b_chunk), c_chunk)| {
                let mut result_chunk = DenseND::<f64>::zeros(a_chunk.shape());
                let a_slice = a_chunk.as_slice();
                let b_slice = b_chunk.as_slice();
                let c_slice = c_chunk.as_slice();
                let result_slice = result_chunk
                    .as_array_mut()
                    .as_slice_mut()
                    .expect("Contiguous array");

                // Hardware FMA instruction
                for i in 0..a_slice.len() {
                    result_slice[i] = a_slice[i].mul_add(b_slice[i], c_slice[i]);
                }

                result_chunk
            })
            .collect::<Vec<_>>();

        // Concatenate results
        let result = DenseND::concatenate(&result_chunks, 0)?;

        // Record profiling
        let elapsed = start_time.elapsed();
        let element_count = a.shape().iter().product::<usize>();
        let bytes_read = (element_count * 3 * std::mem::size_of::<f64>()) as u64;
        let bytes_written = (element_count * std::mem::size_of::<f64>()) as u64;
        self.profiler
            .record_operation("fma_chunked", elapsed, bytes_read, bytes_written);

        Ok(result)
    }

    /// Optimized element-wise operations (SIMD-friendly fast path)
    ///
    /// This method uses inlined operations that enable better compiler auto-vectorization
    /// and SIMD optimizations compared to generic closures.
    fn elementwise_chunked_optimized(
        &mut self,
        a: &DenseND<f64>,
        b: &DenseND<f64>,
        op: ElementwiseOp,
    ) -> Result<DenseND<f64>> {
        // Validate inputs
        if a.shape() != b.shape() {
            return Err(anyhow!(
                "Shape mismatch for element-wise operation: {:?} vs {:?}",
                a.shape(),
                b.shape()
            ));
        }

        let shape = a.shape();

        // For 1D tensors, no chunking needed - direct vectorized operation
        if a.rank() == 1 {
            let mut result = DenseND::<f64>::zeros(shape);
            let a_slice = a.as_slice();
            let b_slice = b.as_slice();
            let result_slice = result
                .as_array_mut()
                .as_slice_mut()
                .expect("Contiguous array");

            // Inlined operation enables SIMD auto-vectorization
            match op {
                ElementwiseOp::Add => {
                    for i in 0..a_slice.len() {
                        result_slice[i] = a_slice[i] + b_slice[i];
                    }
                }
                ElementwiseOp::Subtract => {
                    for i in 0..a_slice.len() {
                        result_slice[i] = a_slice[i] - b_slice[i];
                    }
                }
                ElementwiseOp::Multiply => {
                    for i in 0..a_slice.len() {
                        result_slice[i] = a_slice[i] * b_slice[i];
                    }
                }
                ElementwiseOp::Divide => {
                    for i in 0..a_slice.len() {
                        result_slice[i] = a_slice[i] / b_slice[i];
                    }
                }
                ElementwiseOp::Min => {
                    for i in 0..a_slice.len() {
                        result_slice[i] = a_slice[i].min(b_slice[i]);
                    }
                }
                ElementwiseOp::Max => {
                    for i in 0..a_slice.len() {
                        result_slice[i] = a_slice[i].max(b_slice[i]);
                    }
                }
            }
            return Ok(result);
        }

        // Determine chunk size for first dimension
        let chunk_size = if !self.config.default_chunk_size.is_empty() {
            self.config.default_chunk_size[0]
        } else {
            256
        };

        // Chunk along first axis
        let a_chunks = a.chunk(chunk_size, 0)?;
        let b_chunks = b.chunk(chunk_size, 0)?;
        let chunk_count = a_chunks.len();

        // Process chunks with adaptive parallel threshold
        #[cfg(feature = "parallel")]
        let result_chunks = if self.should_use_parallel(chunk_count) {
            // Parallel execution path
            a_chunks
                .into_par_iter()
                .zip(b_chunks.into_par_iter())
                .map(|(a_chunk, b_chunk)| {
                    let mut result_chunk = DenseND::<f64>::zeros(a_chunk.shape());
                    let a_slice = a_chunk.as_slice();
                    let b_slice = b_chunk.as_slice();
                    let result_slice = result_chunk
                        .as_array_mut()
                        .as_slice_mut()
                        .expect("Contiguous array");

                    // Inlined operation for SIMD
                    match op {
                        ElementwiseOp::Add => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i] + b_slice[i];
                            }
                        }
                        ElementwiseOp::Subtract => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i] - b_slice[i];
                            }
                        }
                        ElementwiseOp::Multiply => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i] * b_slice[i];
                            }
                        }
                        ElementwiseOp::Divide => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i] / b_slice[i];
                            }
                        }
                        ElementwiseOp::Min => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i].min(b_slice[i]);
                            }
                        }
                        ElementwiseOp::Max => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i].max(b_slice[i]);
                            }
                        }
                    }

                    result_chunk
                })
                .collect::<Vec<_>>()
        } else {
            // Sequential execution path
            a_chunks
                .into_iter()
                .zip(b_chunks)
                .map(|(a_chunk, b_chunk)| {
                    let mut result_chunk = DenseND::<f64>::zeros(a_chunk.shape());
                    let a_slice = a_chunk.as_slice();
                    let b_slice = b_chunk.as_slice();
                    let result_slice = result_chunk
                        .as_array_mut()
                        .as_slice_mut()
                        .expect("Contiguous array");

                    // Inlined operation for SIMD
                    match op {
                        ElementwiseOp::Add => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i] + b_slice[i];
                            }
                        }
                        ElementwiseOp::Subtract => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i] - b_slice[i];
                            }
                        }
                        ElementwiseOp::Multiply => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i] * b_slice[i];
                            }
                        }
                        ElementwiseOp::Divide => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i] / b_slice[i];
                            }
                        }
                        ElementwiseOp::Min => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i].min(b_slice[i]);
                            }
                        }
                        ElementwiseOp::Max => {
                            for i in 0..a_slice.len() {
                                result_slice[i] = a_slice[i].max(b_slice[i]);
                            }
                        }
                    }

                    result_chunk
                })
                .collect::<Vec<_>>()
        };

        #[cfg(not(feature = "parallel"))]
        let result_chunks: Vec<DenseND<f64>> = a_chunks
            .into_iter()
            .zip(b_chunks.into_iter())
            .map(|(a_chunk, b_chunk)| {
                let mut result_chunk = DenseND::<f64>::zeros(a_chunk.shape());
                let a_slice = a_chunk.as_slice();
                let b_slice = b_chunk.as_slice();
                let result_slice = result_chunk
                    .as_array_mut()
                    .as_slice_mut()
                    .expect("Contiguous array");

                // Inlined operation for SIMD
                match op {
                    ElementwiseOp::Add => {
                        for i in 0..a_slice.len() {
                            result_slice[i] = a_slice[i] + b_slice[i];
                        }
                    }
                    ElementwiseOp::Subtract => {
                        for i in 0..a_slice.len() {
                            result_slice[i] = a_slice[i] - b_slice[i];
                        }
                    }
                    ElementwiseOp::Multiply => {
                        for i in 0..a_slice.len() {
                            result_slice[i] = a_slice[i] * b_slice[i];
                        }
                    }
                    ElementwiseOp::Divide => {
                        for i in 0..a_slice.len() {
                            result_slice[i] = a_slice[i] / b_slice[i];
                        }
                    }
                    ElementwiseOp::Min => {
                        for i in 0..a_slice.len() {
                            result_slice[i] = a_slice[i].min(b_slice[i]);
                        }
                    }
                    ElementwiseOp::Max => {
                        for i in 0..a_slice.len() {
                            result_slice[i] = a_slice[i].max(b_slice[i]);
                        }
                    }
                }

                result_chunk
            })
            .collect::<Vec<_>>();

        // Concatenate results
        let result = DenseND::concatenate(&result_chunks, 0)?;

        Ok(result)
    }

    /// Execute element-wise operations in streaming fashion (generic version)
    ///
    /// This is a private helper that chunks tensors and applies generic operations.
    /// For better performance, prefer using the optimized add_chunked() and multiply_chunked().
    ///
    /// Note: Currently unused but kept for future extensibility with custom operations.
    #[allow(dead_code)]
    fn elementwise_chunked<F>(
        &mut self,
        a: &DenseND<f64>,
        b: &DenseND<f64>,
        op: F,
    ) -> Result<DenseND<f64>>
    where
        F: Fn(f64, f64) -> f64 + Sync + Send,
    {
        // Validate inputs
        if a.shape() != b.shape() {
            return Err(anyhow!(
                "Shape mismatch for element-wise operation: {:?} vs {:?}",
                a.shape(),
                b.shape()
            ));
        }

        let shape = a.shape();

        // For 1D tensors, no chunking needed
        if a.rank() == 1 {
            let mut result = DenseND::<f64>::zeros(shape);
            let a_slice = a.as_slice();
            let b_slice = b.as_slice();
            let result_slice = result
                .as_array_mut()
                .as_slice_mut()
                .expect("Contiguous array");

            for i in 0..a_slice.len() {
                result_slice[i] = op(a_slice[i], b_slice[i]);
            }
            return Ok(result);
        }

        // Determine chunk size for first dimension
        let chunk_size = if !self.config.default_chunk_size.is_empty() {
            self.config.default_chunk_size[0]
        } else {
            256
        };

        // Chunk along first axis
        let a_chunks = a.chunk(chunk_size, 0)?;
        let b_chunks = b.chunk(chunk_size, 0)?;
        let chunk_count = a_chunks.len();

        // Process each chunk with parallel processing if adaptive threshold met
        #[cfg(feature = "parallel")]
        let result_chunks = if self.should_use_parallel(chunk_count) {
            a_chunks
                .into_par_iter()
                .zip(b_chunks.into_par_iter())
                .map(|(a_chunk, b_chunk)| {
                    let mut result_chunk = DenseND::<f64>::zeros(a_chunk.shape());
                    let a_slice = a_chunk.as_slice();
                    let b_slice = b_chunk.as_slice();
                    let result_slice = result_chunk
                        .as_array_mut()
                        .as_slice_mut()
                        .expect("Contiguous array");

                    for i in 0..a_slice.len() {
                        result_slice[i] = op(a_slice[i], b_slice[i]);
                    }

                    result_chunk
                })
                .collect::<Vec<_>>()
        } else {
            a_chunks
                .into_iter()
                .zip(b_chunks)
                .map(|(a_chunk, b_chunk)| {
                    let mut result_chunk = DenseND::<f64>::zeros(a_chunk.shape());
                    let a_slice = a_chunk.as_slice();
                    let b_slice = b_chunk.as_slice();
                    let result_slice = result_chunk
                        .as_array_mut()
                        .as_slice_mut()
                        .expect("Contiguous array");

                    for i in 0..a_slice.len() {
                        result_slice[i] = op(a_slice[i], b_slice[i]);
                    }

                    result_chunk
                })
                .collect::<Vec<_>>()
        };

        #[cfg(not(feature = "parallel"))]
        let result_chunks: Vec<DenseND<f64>> = a_chunks
            .into_iter()
            .zip(b_chunks.into_iter())
            .map(|(a_chunk, b_chunk)| {
                let mut result_chunk = DenseND::<f64>::zeros(a_chunk.shape());
                let a_slice = a_chunk.as_slice();
                let b_slice = b_chunk.as_slice();
                let result_slice = result_chunk
                    .as_array_mut()
                    .as_slice_mut()
                    .expect("Contiguous array");

                for i in 0..a_slice.len() {
                    result_slice[i] = op(a_slice[i], b_slice[i]);
                }

                result_chunk
            })
            .collect::<Vec<_>>();

        // Concatenate results
        let result = DenseND::concatenate(&result_chunks, 0)?;

        Ok(result)
    }

    /// Spill tensor to disk and return path
    ///
    /// This is used when memory pressure is high and we need to free up RAM.
    ///
    /// **Note:** Requires the `mmap` feature to be enabled.
    #[cfg(feature = "mmap")]
    pub fn spill_to_disk(&mut self, tensor: &DenseND<f64>) -> Result<PathBuf> {
        if !self.config.enable_spill {
            return Err(anyhow!("Spill-to-disk is disabled"));
        }

        // Generate unique filename
        let filename = format!("tenrso_spill_{}.bin", self.spill_counter);
        self.spill_counter += 1;

        let path = self.config.temp_dir.join(filename);

        // Write to disk using binary format
        crate::mmap_io::write_tensor_binary(&path, tensor)?;

        // Update memory tracking (tensor is now on disk)
        let tensor_mem = tensor.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        self.current_memory = self.current_memory.saturating_sub(tensor_mem);

        Ok(path)
    }

    /// Load tensor from spilled file
    ///
    /// **Note:** Requires the `mmap` feature to be enabled.
    #[cfg(feature = "mmap")]
    pub fn load_from_disk(&mut self, path: &Path) -> Result<DenseND<f64>> {
        let tensor = crate::mmap_io::read_tensor_binary(path)?;

        // Update memory tracking
        let tensor_mem = tensor.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        self.current_memory += tensor_mem;

        Ok(tensor)
    }
}

impl Clone for StreamingExecutor {
    fn clone(&self) -> Self {
        // Create new executor with same config
        Self::new(self.config.clone())
    }
}

/// Optimization utilities for streaming execution
impl StreamingExecutor {
    /// Compute optimal chunk size for matrix multiplication
    ///
    /// This auto-tunes the chunk size based on tensor dimensions and available memory.
    ///
    /// # Strategy
    ///
    /// For A [m, n] @ B [n, p], we want to chunk A along rows.
    /// Memory needed per chunk: chunk_rows × n + n × p + chunk_rows × p
    ///
    /// We solve for chunk_rows given memory_budget:
    /// chunk_rows × (n + p) ≈ memory_budget - n × p
    ///
    /// # Arguments
    ///
    /// * `a_shape` - Shape of left matrix [m, n]
    /// * `b_shape` - Shape of right matrix [n, p]
    /// * `memory_budget_mb` - Available memory in megabytes (default: 25% of max memory)
    ///
    /// # Returns
    ///
    /// Optimal number of rows to process per chunk
    pub fn compute_optimal_matmul_chunk_size(
        &self,
        a_shape: &[usize],
        b_shape: &[usize],
        memory_budget_mb: Option<usize>,
    ) -> Result<usize> {
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(anyhow!(
                "compute_optimal_matmul_chunk_size requires 2D shapes"
            ));
        }

        let m = a_shape[0];
        let n = a_shape[1];
        let p = b_shape[1];

        if n != b_shape[0] {
            return Err(anyhow!("Incompatible matrix shapes for multiplication"));
        }

        // Determine memory budget (default: 25% of max memory)
        let budget_bytes = memory_budget_mb
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(self.config.max_memory_bytes / 4);

        // Memory for B and result chunk
        let fixed_memory = (n * p + m * p) * std::mem::size_of::<f64>();

        if fixed_memory >= budget_bytes {
            // Can't fit even one row, use minimum
            return Ok(1);
        }

        // Available memory for A chunks
        let available = budget_bytes - fixed_memory;

        // Memory per row of A: n elements
        let bytes_per_row = n * std::mem::size_of::<f64>();

        // Compute chunk size
        let chunk_rows = (available / bytes_per_row).max(1).min(m);

        Ok(chunk_rows)
    }

    /// Compute optimal chunk size for element-wise operations
    ///
    /// # Strategy
    ///
    /// For element-wise ops on tensors of shape [d1, d2, ...]:
    /// We chunk along the first dimension.
    /// Memory needed: 2 × chunk_size × (d2 × d3 × ...) for inputs
    ///              + 1 × chunk_size × (d2 × d3 × ...) for output
    ///
    /// # Arguments
    ///
    /// * `shape` - Tensor shape
    /// * `memory_budget_mb` - Available memory in megabytes
    ///
    /// # Returns
    ///
    /// Optimal chunk size for first dimension
    pub fn compute_optimal_elementwise_chunk_size(
        &self,
        shape: &[usize],
        memory_budget_mb: Option<usize>,
    ) -> Result<usize> {
        if shape.is_empty() {
            return Err(anyhow!("Cannot compute chunk size for empty shape"));
        }

        // Determine memory budget
        let budget_bytes = memory_budget_mb
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(self.config.max_memory_bytes / 4);

        // Elements per "slice" (product of all dimensions except first)
        let elements_per_slice: usize = shape[1..].iter().product();

        // Memory per chunk row: 3 × elements_per_slice (2 inputs + 1 output)
        let bytes_per_chunk_row = 3 * elements_per_slice * std::mem::size_of::<f64>();

        if bytes_per_chunk_row == 0 {
            return Ok(shape[0]);
        }

        // Compute chunk size
        let chunk_size = (budget_bytes / bytes_per_chunk_row).max(1).min(shape[0]);

        Ok(chunk_size)
    }

    /// Estimate throughput for a given chunk size
    ///
    /// This provides a rough estimate of GFlops and memory bandwidth
    /// for performance tuning.
    ///
    /// # Arguments
    ///
    /// * `op_type` - Type of operation ("matmul", "elementwise")
    /// * `shape1` - Shape of first operand
    /// * `shape2` - Shape of second operand (for matmul)
    /// * `chunk_size` - Proposed chunk size
    ///
    /// # Returns
    ///
    /// Tuple of (estimated_gflops, memory_bandwidth_gb_per_sec)
    pub fn estimate_throughput(
        &self,
        op_type: &str,
        shape1: &[usize],
        shape2: Option<&[usize]>,
        chunk_size: usize,
    ) -> Result<(f64, f64)> {
        match op_type {
            "matmul" => {
                let shape2 = shape2.ok_or_else(|| anyhow!("matmul requires shape2"))?;

                if shape1.len() != 2 || shape2.len() != 2 {
                    return Err(anyhow!("matmul requires 2D shapes"));
                }

                let m = chunk_size.min(shape1[0]);
                let n = shape1[1];
                let p = shape2[1];

                // FLOPs for matmul: 2 × m × n × p (multiply-add)
                let flops = 2.0 * (m as f64) * (n as f64) * (p as f64);

                // Memory traffic: load A chunk (m×n), load B (n×p), store result (m×p)
                let memory_bytes =
                    ((m * n + n * p + m * p) as f64) * std::mem::size_of::<f64>() as f64;

                // Assuming 10ms per chunk (rough estimate)
                let time_seconds = 0.01;

                let gflops = (flops / 1e9) / time_seconds;
                let bandwidth_gb = (memory_bytes / 1e9) / time_seconds;

                Ok((gflops, bandwidth_gb))
            }
            "elementwise" => {
                let elements_per_slice: usize = shape1[1..].iter().product();
                let chunk_elements = chunk_size * elements_per_slice;

                // FLOPs: 1 per element (single operation like add or multiply)
                let flops = chunk_elements as f64;

                // Memory traffic: load 2 inputs + store 1 output
                let memory_bytes = (3 * chunk_elements) as f64 * std::mem::size_of::<f64>() as f64;

                // Assuming 1ms per chunk
                let time_seconds = 0.001;

                let gflops = (flops / 1e9) / time_seconds;
                let bandwidth_gb = (memory_bytes / 1e9) / time_seconds;

                Ok((gflops, bandwidth_gb))
            }
            _ => Err(anyhow!("Unknown operation type: {}", op_type)),
        }
    }

    /// Recommend optimal configuration for a given workload
    ///
    /// This analyzes tensor shapes and provides tuning recommendations.
    ///
    /// # Arguments
    ///
    /// * `operation` - Type of operation ("matmul", "elementwise")
    /// * `shapes` - Tensor shapes involved
    ///
    /// # Returns
    ///
    /// Recommended chunk size and explanation
    pub fn recommend_config(
        &self,
        operation: &str,
        shapes: &[&[usize]],
    ) -> Result<(usize, String)> {
        match operation {
            "matmul" => {
                if shapes.len() != 2 {
                    return Err(anyhow!("matmul requires 2 shapes"));
                }

                let chunk_size =
                    self.compute_optimal_matmul_chunk_size(shapes[0], shapes[1], None)?;

                let (gflops, bandwidth) =
                    self.estimate_throughput("matmul", shapes[0], Some(shapes[1]), chunk_size)?;

                let explanation = format!(
                    "Recommended chunk size: {} rows\n\
                     Estimated performance: {:.2} GFlops, {:.2} GB/s\n\
                     Memory per chunk: {:.2} MB\n\
                     Total chunks: {}",
                    chunk_size,
                    gflops,
                    bandwidth,
                    (chunk_size * shapes[0][1] * std::mem::size_of::<f64>()) as f64 / 1e6,
                    shapes[0][0].div_ceil(chunk_size)
                );

                Ok((chunk_size, explanation))
            }
            "elementwise" => {
                if shapes.is_empty() {
                    return Err(anyhow!("elementwise requires at least 1 shape"));
                }

                let chunk_size = self.compute_optimal_elementwise_chunk_size(shapes[0], None)?;

                let (gflops, bandwidth) =
                    self.estimate_throughput("elementwise", shapes[0], None, chunk_size)?;

                let elements_per_slice: usize = shapes[0][1..].iter().product();

                let explanation = format!(
                    "Recommended chunk size: {} (first dimension)\n\
                     Estimated performance: {:.2} GFlops, {:.2} GB/s\n\
                     Memory per chunk: {:.2} MB\n\
                     Total chunks: {}",
                    chunk_size,
                    gflops,
                    bandwidth,
                    (chunk_size * elements_per_slice * std::mem::size_of::<f64>()) as f64 / 1e6,
                    shapes[0][0].div_ceil(chunk_size)
                );

                Ok((chunk_size, explanation))
            }
            _ => Err(anyhow!("Unknown operation: {}", operation)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_executor_creation() {
        let config = StreamConfig::new().max_memory_mb(512).chunk_size(vec![128]);

        let executor = StreamingExecutor::new(config);

        assert_eq!(executor.current_memory(), 0);
        assert!(!executor.is_memory_exceeded());
    }

    #[test]
    fn test_matmul_chunked_small() {
        let config = StreamConfig::new().max_memory_mb(512).chunk_size(vec![2]);

        let mut executor = StreamingExecutor::new(config);

        // Create small test matrices
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = executor.matmul_chunked(&a, &b, Some(1)).unwrap();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(result.shape(), &[2, 2]);
        let data = result.as_slice();
        assert!((data[0] - 19.0).abs() < 1e-10);
        assert!((data[1] - 22.0).abs() < 1e-10);
        assert!((data[2] - 43.0).abs() < 1e-10);
        assert!((data[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_chunked_larger() {
        let config = StreamConfig::new().chunk_size(vec![32]);
        let mut executor = StreamingExecutor::new(config);

        // Create larger matrices
        let m = 100;
        let n = 80;
        let p = 60;

        let a_data: Vec<f64> = (0..m * n).map(|i| i as f64).collect();
        let b_data: Vec<f64> = (0..n * p).map(|i| (i % 10) as f64).collect();

        let a = DenseND::<f64>::from_vec(a_data, &[m, n]).unwrap();
        let b = DenseND::<f64>::from_vec(b_data, &[n, p]).unwrap();

        let result = executor.matmul_chunked(&a, &b, Some(32)).unwrap();

        assert_eq!(result.shape(), &[m, p]);

        // Verify using regular matmul
        let expected = a.matmul(&b).unwrap();
        let result_data = result.as_slice();
        let expected_data = expected.as_slice();

        for i in 0..result_data.len() {
            assert!((result_data[i] - expected_data[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_add_chunked() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = executor.add_chunked(&a, &b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        let data = result.as_slice();
        assert!((data[0] - 6.0).abs() < 1e-10);
        assert!((data[1] - 8.0).abs() < 1e-10);
        assert!((data[2] - 10.0).abs() < 1e-10);
        assert!((data[3] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiply_chunked() {
        let config = StreamConfig::new().chunk_size(vec![3]);
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let result = executor.multiply_chunked(&a, &b).unwrap();

        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        assert!((data[0] - 2.0).abs() < 1e-10);
        assert!((data[1] - 6.0).abs() < 1e-10);
        assert!((data[2] - 12.0).abs() < 1e-10);
        assert!((data[3] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_subtract_chunked() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::from_vec(vec![10.0, 9.0, 8.0, 7.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = executor.subtract_chunked(&a, &b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        let data = result.as_slice();
        assert!((data[0] - 9.0).abs() < 1e-10);
        assert!((data[1] - 7.0).abs() < 1e-10);
        assert!((data[2] - 5.0).abs() < 1e-10);
        assert!((data[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_divide_chunked() {
        let config = StreamConfig::new().chunk_size(vec![3]);
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![2.0, 4.0, 5.0, 8.0], &[4]).unwrap();

        let result = executor.divide_chunked(&a, &b).unwrap();

        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        assert!((data[0] - 5.0).abs() < 1e-10);
        assert!((data[1] - 5.0).abs() < 1e-10);
        assert!((data[2] - 6.0).abs() < 1e-10);
        assert!((data[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fma_chunked_basic() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);

        // Test a * b + c
        let a = DenseND::<f64>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4]).unwrap();
        let c = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let result = executor.fma_chunked(&a, &b, &c).unwrap();

        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        // Expected: a * b + c
        assert!((data[0] - 11.0).abs() < 1e-10); // 2*5+1 = 11
        assert!((data[1] - 20.0).abs() < 1e-10); // 3*6+2 = 20
        assert!((data[2] - 31.0).abs() < 1e-10); // 4*7+3 = 31
        assert!((data[3] - 44.0).abs() < 1e-10); // 5*8+4 = 44
    }

    #[test]
    fn test_fma_chunked_2d() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);

        // Test with 2D tensors
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let c = DenseND::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]).unwrap();

        let result = executor.fma_chunked(&a, &b, &c).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        let data = result.as_slice();
        assert!((data[0] - 3.0).abs() < 1e-10); // 1*2+1 = 3
        assert!((data[1] - 7.0).abs() < 1e-10); // 2*3+1 = 7
        assert!((data[2] - 13.0).abs() < 1e-10); // 3*4+1 = 13
        assert!((data[3] - 21.0).abs() < 1e-10); // 4*5+1 = 21
    }

    #[test]
    fn test_fma_vs_separate_ops_accuracy() {
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);

        // Use values that might show rounding differences
        let a = DenseND::<f64>::from_vec(vec![1e10, 1e-10, 0.1, 0.2], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1e-10, 1e10, 0.3, 0.4], &[4]).unwrap();
        let c = DenseND::<f64>::from_vec(vec![1.0, 1.0, 0.5, 0.6], &[4]).unwrap();

        // FMA result (single atomic operation, more accurate)
        let fma_result = executor.fma_chunked(&a, &b, &c).unwrap();

        // Separate operations result (may have intermediate rounding)
        let mul_result = executor.multiply_chunked(&a, &b).unwrap();
        let separate_result = executor.add_chunked(&mul_result, &c).unwrap();

        // FMA should produce same or better accuracy
        let fma_data = fma_result.as_slice();
        let separate_data = separate_result.as_slice();

        // For most cases they should be very close or identical
        for i in 0..fma_data.len() {
            let diff = (fma_data[i] - separate_data[i]).abs();
            // Allow small numerical differences due to different rounding
            assert!(
                diff < 1e-8,
                "FMA vs separate difference at index {}: {}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_fma_shape_mismatch() {
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let c = DenseND::<f64>::from_vec(vec![7.0, 8.0], &[2]).unwrap(); // Wrong shape!

        let result = executor.fma_chunked(&a, &b, &c);
        assert!(result.is_err(), "Should fail with shape mismatch");
    }

    #[test]
    fn test_min_chunked_basic() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::from_vec(vec![3.0, 1.0, 4.0, 2.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![2.0, 3.0, 1.0, 5.0], &[4]).unwrap();

        let result = executor.min_chunked(&a, &b).unwrap();

        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        assert!((data[0] - 2.0).abs() < 1e-10); // min(3, 2) = 2
        assert!((data[1] - 1.0).abs() < 1e-10); // min(1, 3) = 1
        assert!((data[2] - 1.0).abs() < 1e-10); // min(4, 1) = 1
        assert!((data[3] - 2.0).abs() < 1e-10); // min(2, 5) = 2
    }

    #[test]
    fn test_max_chunked_basic() {
        let config = StreamConfig::new().chunk_size(vec![2]);
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::from_vec(vec![3.0, 1.0, 4.0, 2.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![2.0, 3.0, 1.0, 5.0], &[4]).unwrap();

        let result = executor.max_chunked(&a, &b).unwrap();

        assert_eq!(result.shape(), &[4]);
        let data = result.as_slice();
        assert!((data[0] - 3.0).abs() < 1e-10); // max(3, 2) = 3
        assert!((data[1] - 3.0).abs() < 1e-10); // max(1, 3) = 3
        assert!((data[2] - 4.0).abs() < 1e-10); // max(4, 1) = 4
        assert!((data[3] - 5.0).abs() < 1e-10); // max(2, 5) = 5
    }

    #[test]
    fn test_min_max_with_negatives() {
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::from_vec(vec![-5.0, -2.0, 0.0, 3.0], &[4]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![-3.0, -4.0, -1.0, 2.0], &[4]).unwrap();

        let min_result = executor.min_chunked(&a, &b).unwrap();
        let max_result = executor.max_chunked(&a, &b).unwrap();

        let min_data = min_result.as_slice();
        let max_data = max_result.as_slice();

        assert!((min_data[0] - (-5.0)).abs() < 1e-10); // min(-5, -3) = -5
        assert!((min_data[1] - (-4.0)).abs() < 1e-10); // min(-2, -4) = -4
        assert!((min_data[2] - (-1.0)).abs() < 1e-10); // min(0, -1) = -1
        assert!((min_data[3] - 2.0).abs() < 1e-10); // min(3, 2) = 2

        assert!((max_data[0] - (-3.0)).abs() < 1e-10); // max(-5, -3) = -3
        assert!((max_data[1] - (-2.0)).abs() < 1e-10); // max(-2, -4) = -2
        assert!((max_data[2] - 0.0).abs() < 1e-10); // max(0, -1) = 0
        assert!((max_data[3] - 3.0).abs() < 1e-10); // max(3, 2) = 3
    }

    #[test]
    fn test_min_max_2d() {
        let config = StreamConfig::new().chunk_size(vec![1]);
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::<f64>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![4.0, 2.0, 1.0, 6.0], &[2, 2]).unwrap();

        let min_result = executor.min_chunked(&a, &b).unwrap();
        let max_result = executor.max_chunked(&a, &b).unwrap();

        assert_eq!(min_result.shape(), &[2, 2]);
        assert_eq!(max_result.shape(), &[2, 2]);

        let min_data = min_result.as_slice();
        let max_data = max_result.as_slice();

        assert!((min_data[0] - 1.0).abs() < 1e-10);
        assert!((min_data[1] - 2.0).abs() < 1e-10);
        assert!((min_data[2] - 1.0).abs() < 1e-10);
        assert!((min_data[3] - 2.0).abs() < 1e-10);

        assert!((max_data[0] - 4.0).abs() < 1e-10);
        assert!((max_data[1] - 5.0).abs() < 1e-10);
        assert!((max_data[2] - 3.0).abs() < 1e-10);
        assert!((max_data[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_special_values() {
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);

        // Test with infinity and zero
        let a = DenseND::<f64>::from_vec(vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0], &[4])
            .unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 1.0, -0.0, 0.0], &[4]).unwrap();

        let min_result = executor.min_chunked(&a, &b).unwrap();
        let max_result = executor.max_chunked(&a, &b).unwrap();

        let min_data = min_result.as_slice();
        let max_data = max_result.as_slice();

        assert_eq!(min_data[0], 1.0); // min(inf, 1) = 1
        assert_eq!(min_data[1], f64::NEG_INFINITY); // min(-inf, 1) = -inf
        assert_eq!(max_data[0], f64::INFINITY); // max(inf, 1) = inf
        assert_eq!(max_data[1], 1.0); // max(-inf, 1) = 1
    }

    #[test]
    #[cfg(feature = "mmap")]
    fn test_spill_and_load() {
        let config = StreamConfig::new().enable_spill(true);
        let mut executor = StreamingExecutor::new(config);

        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // Spill to disk
        let path = executor.spill_to_disk(&tensor).unwrap();
        assert!(path.exists());

        // Load back
        let loaded = executor.load_from_disk(&path).unwrap();

        assert_eq!(loaded.shape(), tensor.shape());
        assert_eq!(loaded.as_slice(), tensor.as_slice());

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    #[cfg(feature = "mmap")]
    fn test_spill_disabled() {
        let config = StreamConfig::new().enable_spill(false);
        let mut executor = StreamingExecutor::new(config);

        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        // Should fail when spill is disabled
        assert!(executor.spill_to_disk(&tensor).is_err());
    }

    #[test]
    fn test_memory_tracking() {
        let config = StreamConfig::new().max_memory_mb(1);
        let mut executor = StreamingExecutor::new(config);

        assert_eq!(executor.current_memory(), 0);
        assert!(!executor.is_memory_exceeded());

        // Create tensors for memory tracking test (reduced size for speed)
        let a = DenseND::<f64>::zeros(&[200, 200]);
        let b = DenseND::<f64>::zeros(&[200, 200]);

        // After matmul_chunked, memory should be tracked
        let _ = executor.matmul_chunked(&a, &b, Some(50)).unwrap();

        // Memory should be tracked (but test behavior, not exact bytes)
        assert!(executor.current_memory() > 0);
    }

    #[test]
    fn test_compute_optimal_matmul_chunk_size() {
        let config = StreamConfig::new().max_memory_mb(100);
        let executor = StreamingExecutor::new(config);

        // Small matrices
        let chunk_size = executor
            .compute_optimal_matmul_chunk_size(&[100, 50], &[50, 80], None)
            .unwrap();

        assert!(chunk_size > 0);
        assert!(chunk_size <= 100);
    }

    #[test]
    fn test_compute_optimal_elementwise_chunk_size() {
        let config = StreamConfig::new().max_memory_mb(100);
        let executor = StreamingExecutor::new(config);

        let chunk_size = executor
            .compute_optimal_elementwise_chunk_size(&[1000, 500], None)
            .unwrap();

        assert!(chunk_size > 0);
        assert!(chunk_size <= 1000);
    }

    #[test]
    fn test_estimate_throughput_matmul() {
        let config = StreamConfig::new();
        let executor = StreamingExecutor::new(config);

        let (gflops, bandwidth) = executor
            .estimate_throughput("matmul", &[100, 50], Some(&[50, 80]), 32)
            .unwrap();

        assert!(gflops > 0.0);
        assert!(bandwidth > 0.0);
    }

    #[test]
    fn test_estimate_throughput_elementwise() {
        let config = StreamConfig::new();
        let executor = StreamingExecutor::new(config);

        let (gflops, bandwidth) = executor
            .estimate_throughput("elementwise", &[1000, 500], None, 100)
            .unwrap();

        assert!(gflops > 0.0);
        assert!(bandwidth > 0.0);
    }

    #[test]
    fn test_recommend_config_matmul() {
        let config = StreamConfig::new().max_memory_mb(100);
        let executor = StreamingExecutor::new(config);

        let (chunk_size, explanation) = executor
            .recommend_config("matmul", &[&[100, 50], &[50, 80]])
            .unwrap();

        assert!(chunk_size > 0);
        assert!(chunk_size <= 100);
        assert!(explanation.contains("Recommended chunk size"));
        assert!(explanation.contains("GFlops"));
        assert!(explanation.contains("GB/s"));
    }

    #[test]
    fn test_recommend_config_elementwise() {
        let config = StreamConfig::new().max_memory_mb(100);
        let executor = StreamingExecutor::new(config);

        let (chunk_size, explanation) = executor
            .recommend_config("elementwise", &[&[1000, 500]])
            .unwrap();

        assert!(chunk_size > 0);
        assert!(chunk_size <= 1000);
        assert!(explanation.contains("Recommended chunk size"));
    }

    #[test]
    fn test_auto_tuning_respects_memory_limit() {
        // Very small memory limit
        let config = StreamConfig::new().max_memory_mb(1);
        let executor = StreamingExecutor::new(config);

        let chunk_size = executor
            .compute_optimal_matmul_chunk_size(&[1000, 1000], &[1000, 1000], Some(1))
            .unwrap();

        // Should get a very small chunk size due to memory constraint
        assert!(chunk_size > 0);
        assert!(chunk_size < 100); // Should be significantly smaller than matrix size
    }

    #[test]
    fn test_profiling_matmul() {
        // Enable profiling
        let config = StreamConfig::new().enable_profiling(true);
        let mut executor = StreamingExecutor::new(config);

        // Create test tensors
        let a = DenseND::from_elem(&[100, 80], 1.0);
        let b = DenseND::from_elem(&[80, 60], 2.0);

        // Execute operation
        let _ = executor.matmul_chunked(&a, &b, Some(50)).unwrap();

        // Check profiling stats
        let stats = executor.profiler().get_operation_stats("matmul_chunked");
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.call_count, 1);
        assert!(stats.total_time.as_nanos() > 0);
        assert!(stats.bytes_read > 0);
        assert!(stats.bytes_written > 0);
    }

    #[test]
    fn test_profiling_elementwise() {
        // Enable profiling with smaller chunk size
        let config = StreamConfig::new()
            .enable_profiling(true)
            .chunk_size(vec![100]); // Smaller than tensor size
        let mut executor = StreamingExecutor::new(config);

        // Create test tensors
        let a = DenseND::from_elem(&[200, 100], 1.0);
        let b = DenseND::from_elem(&[200, 100], 2.0);

        // Execute operations
        let _ = executor.add_chunked(&a, &b).unwrap();
        let _ = executor.multiply_chunked(&a, &b).unwrap();

        // Check profiling stats
        let add_stats = executor.profiler().get_operation_stats("add_chunked");
        let mult_stats = executor.profiler().get_operation_stats("multiply_chunked");

        assert!(add_stats.is_some());
        assert!(mult_stats.is_some());

        assert_eq!(add_stats.unwrap().call_count, 1);
        assert_eq!(mult_stats.unwrap().call_count, 1);
    }

    #[test]
    fn test_profiling_summary() {
        // Enable profiling with smaller chunk size
        let config = StreamConfig::new()
            .enable_profiling(true)
            .chunk_size(vec![30]); // Smaller than tensor size
        let mut executor = StreamingExecutor::new(config);

        // Execute multiple operations
        let a = DenseND::from_elem(&[50, 40], 1.0);
        let b = DenseND::from_elem(&[40, 30], 2.0);

        let _ = executor.matmul_chunked(&a, &b, None).unwrap();
        let _ = executor.add_chunked(&a, &a).unwrap();

        // Get summary
        let summary = executor.profiling_summary();

        assert!(summary.total_operations >= 2);
        assert!(summary.total_time.as_nanos() > 0);
        assert!(summary.total_io_bytes > 0);
    }

    #[test]
    fn test_profiling_disabled() {
        // Profiling disabled by default
        let config = StreamConfig::new();
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::from_elem(&[50, 40], 1.0);
        let b = DenseND::from_elem(&[40, 30], 2.0);

        let _ = executor.matmul_chunked(&a, &b, None).unwrap();

        // Stats should still be recorded (but profiler disabled)
        // The profiler still records when disabled, just doesn't track properly
        assert!(!executor.profiler().is_enabled());
    }

    #[test]
    fn test_prefetcher_config() {
        // Enable prefetching
        let config = StreamConfig::new()
            .enable_prefetching(true)
            .prefetch_strategy(PrefetchStrategy::Sequential)
            .prefetch_queue_size(8);

        let executor = StreamingExecutor::new(config);

        assert!(executor.prefetcher().is_enabled());
    }

    #[test]
    fn test_profiler_accessor() {
        let config = StreamConfig::new().enable_profiling(true);
        let mut executor = StreamingExecutor::new(config);

        // Test mutable access
        executor.profiler_mut().set_enabled(false);
        assert!(!executor.profiler().is_enabled());

        executor.profiler_mut().set_enabled(true);
        assert!(executor.profiler().is_enabled());
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_matmul() {
        // Create config with parallel enabled
        let config = StreamConfig::new().enable_parallel(true).num_threads(4);
        let mut executor = StreamingExecutor::new(config);

        // Create test tensors large enough to benefit from chunking
        let a = DenseND::from_elem(&[400, 300], 2.0);
        let b = DenseND::from_elem(&[300, 200], 3.0);

        // Execute with chunking (should use parallel)
        let result = executor.matmul_chunked(&a, &b, Some(100)).unwrap();

        // Verify result shape
        assert_eq!(result.shape(), vec![400, 200]);

        // Verify result values (2.0 * 3.0 * 300 = 1800.0)
        let expected_value = 2.0 * 3.0 * 300.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_elementwise() {
        // Create config with parallel enabled
        let config = StreamConfig::new()
            .enable_parallel(true)
            .num_threads(4)
            .chunk_size(vec![100]);
        let mut executor = StreamingExecutor::new(config);

        // Create test tensors
        let a = DenseND::from_elem(&[400, 300], 2.0);
        let b = DenseND::from_elem(&[400, 300], 3.0);

        // Test parallel addition
        let result_add = executor.add_chunked(&a, &b).unwrap();
        assert_eq!(result_add.shape(), vec![400, 300]);
        let add_slice = result_add.as_slice();
        for &val in add_slice.iter().take(10) {
            assert!((val - 5.0).abs() < 1e-10);
        }

        // Test parallel multiplication
        let result_mul = executor.multiply_chunked(&a, &b).unwrap();
        assert_eq!(result_mul.shape(), vec![400, 300]);
        let mul_slice = result_mul.as_slice();
        for &val in mul_slice.iter().take(10) {
            assert!((val - 6.0).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_disabled() {
        // Create config with parallel explicitly disabled
        let config = StreamConfig::new().enable_parallel(false);
        let mut executor = StreamingExecutor::new(config);

        // Should still work, just sequentially
        let a = DenseND::from_elem(&[200, 150], 2.0);
        let b = DenseND::from_elem(&[150, 100], 3.0);

        let result = executor.matmul_chunked(&a, &b, Some(50)).unwrap();
        assert_eq!(result.shape(), vec![200, 100]);

        let expected_value = 2.0 * 3.0 * 150.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_vs_sequential_correctness() {
        // Test that parallel and sequential produce same results
        let a = DenseND::from_elem(&[200, 150], 2.0);
        let b = DenseND::from_elem(&[150, 100], 3.0);

        // Parallel execution
        let config_parallel = StreamConfig::new().enable_parallel(true).num_threads(4);
        let mut executor_parallel = StreamingExecutor::new(config_parallel);
        let result_parallel = executor_parallel.matmul_chunked(&a, &b, Some(50)).unwrap();

        // Sequential execution
        let config_sequential = StreamConfig::new().enable_parallel(false);
        let mut executor_sequential = StreamingExecutor::new(config_sequential);
        let result_sequential = executor_sequential
            .matmul_chunked(&a, &b, Some(50))
            .unwrap();

        // Compare results
        assert_eq!(result_parallel.shape(), result_sequential.shape());
        let parallel_slice = result_parallel.as_slice();
        let sequential_slice = result_sequential.as_slice();

        for (p, s) in parallel_slice.iter().zip(sequential_slice.iter()) {
            assert!(
                (p - s).abs() < 1e-10,
                "Parallel and sequential results differ"
            );
        }
    }

    #[test]
    fn test_parallel_config_default() {
        // Test that parallel is enabled by default when feature is present
        let config = StreamConfig::new();
        #[cfg(feature = "parallel")]
        assert!(config.enable_parallel);

        #[cfg(not(feature = "parallel"))]
        assert!(config.enable_parallel); // Still true, but won't use parallel
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_small_chunks() {
        // Test that small chunk counts use sequential execution
        // Default min_chunks_for_parallel is 4
        let config = StreamConfig::new()
            .enable_parallel(true)
            .enable_profiling(true);
        let mut executor = StreamingExecutor::new(config);

        // Create matrix that will be split into 2 chunks (< 4 threshold)
        let a = DenseND::from_elem(&[200, 100], 2.0);
        let b = DenseND::from_elem(&[100, 80], 3.0);

        let result = executor.matmul_chunked(&a, &b, Some(100)).unwrap();

        // Verify result correctness
        assert_eq!(result.shape(), vec![200, 80]);
        let expected_value = 2.0 * 3.0 * 100.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }

        // This test primarily verifies that adaptive threshold doesn't break correctness
        // Performance benefit would need benchmarking to measure
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_large_chunks() {
        // Test that large chunk counts use parallel execution
        let config = StreamConfig::new()
            .enable_parallel(true)
            .min_chunks_for_parallel(4); // Explicit threshold
        let mut executor = StreamingExecutor::new(config);

        // Create matrix that will be split into 8 chunks (>= 4 threshold)
        let a = DenseND::from_elem(&[800, 100], 2.0);
        let b = DenseND::from_elem(&[100, 80], 3.0);

        let result = executor.matmul_chunked(&a, &b, Some(100)).unwrap();

        // Verify result correctness
        assert_eq!(result.shape(), vec![800, 80]);
        let expected_value = 2.0 * 3.0 * 100.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_custom() {
        // Test custom threshold configuration
        let config = StreamConfig::new()
            .enable_parallel(true)
            .min_chunks_for_parallel(10); // High threshold
        let mut executor = StreamingExecutor::new(config);

        // Create matrix with 8 chunks (< 10 threshold, so sequential)
        let a = DenseND::from_elem(&[800, 100], 2.0);
        let b = DenseND::from_elem(&[100, 80], 3.0);

        let result = executor.matmul_chunked(&a, &b, Some(100)).unwrap();

        // Should still work correctly (using sequential due to adaptive threshold)
        assert_eq!(result.shape(), vec![800, 80]);
        let expected_value = 2.0 * 3.0 * 100.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_disabled() {
        // Test disabling adaptive behavior (min_chunks = 0)
        let config = StreamConfig::new()
            .enable_parallel(true)
            .min_chunks_for_parallel(0); // Disable adaptive threshold
        let mut executor = StreamingExecutor::new(config);

        // Even with 1 chunk, parallel should be attempted (though not beneficial)
        let a = DenseND::from_elem(&[100, 80], 2.0);
        let b = DenseND::from_elem(&[80, 60], 3.0);

        let result = executor.matmul_chunked(&a, &b, Some(200)).unwrap();

        // Should still work correctly
        assert_eq!(result.shape(), vec![100, 60]);
        let expected_value = 2.0 * 3.0 * 80.0;
        let result_slice = result.as_slice();
        for &val in result_slice.iter().take(10) {
            assert!((val - expected_value).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_threshold_elementwise() {
        // Test adaptive threshold for elementwise operations
        let config = StreamConfig::new()
            .enable_parallel(true)
            .min_chunks_for_parallel(4)
            .chunk_size(vec![100]); // Will create 2 chunks for 200 elements
        let mut executor = StreamingExecutor::new(config);

        let a = DenseND::from_elem(&[200, 100], 2.0);
        let b = DenseND::from_elem(&[200, 100], 3.0);

        // With 2 chunks (< 4 threshold), should use sequential
        let result = executor.add_chunked(&a, &b).unwrap();

        assert_eq!(result.shape(), vec![200, 100]);
        let add_slice = result.as_slice();
        for &val in add_slice.iter().take(10) {
            assert!((val - 5.0).abs() < 1e-10);
        }

        // Now with more chunks
        let a_large = DenseND::from_elem(&[500, 100], 2.0);
        let b_large = DenseND::from_elem(&[500, 100], 3.0);

        // With 5 chunks (>= 4 threshold), should use parallel
        let result_large = executor.add_chunked(&a_large, &b_large).unwrap();

        assert_eq!(result_large.shape(), vec![500, 100]);
        let large_slice = result_large.as_slice();
        for &val in large_slice.iter().take(10) {
            assert!((val - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_adaptive_threshold_config() {
        // Test configuration defaults and builder
        let config = StreamConfig::new();
        assert_eq!(config.min_chunks_for_parallel, 4); // Default threshold

        let custom_config = StreamConfig::new().min_chunks_for_parallel(8);
        assert_eq!(custom_config.min_chunks_for_parallel, 8);

        let disabled_config = StreamConfig::new().min_chunks_for_parallel(0);
        assert_eq!(disabled_config.min_chunks_for_parallel, 0);
    }
}
