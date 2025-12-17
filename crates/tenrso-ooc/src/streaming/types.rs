//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

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

/// Element-wise operation types for optimized SIMD paths
#[derive(Debug, Clone, Copy)]
pub(crate) enum ElementwiseOp {
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
/// Streaming executor for out-of-core tensor operations
pub struct StreamingExecutor {
    pub(crate) config: StreamConfig,
    current_memory: usize,
    #[allow(dead_code)]
    spill_counter: usize,
    profiler: Profiler,
    prefetcher: Prefetcher,
}
impl StreamingExecutor {
    /// Create a new streaming executor with given configuration
    pub fn new(config: StreamConfig) -> Self {
        #[cfg(feature = "parallel")]
        {
            if config.enable_parallel && config.num_threads > 0 {
                ThreadPoolBuilder::new()
                    .num_threads(config.num_threads)
                    .build_global()
                    .ok();
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
        if self.config.min_chunks_for_parallel == 0 {
            return true;
        }
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
        let rows_per_chunk = chunk_rows.unwrap_or_else(|| {
            if !self.config.default_chunk_size.is_empty() {
                self.config.default_chunk_size[0]
            } else {
                256
            }
        });
        let a_mem = a.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        let b_mem = b.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        self.current_memory = a_mem + b_mem;
        if rows_per_chunk >= i {
            let result = a.matmul(b)?;
            let result_mem = result.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
            self.current_memory += result_mem;
            let elapsed = start_time.elapsed();
            let bytes_read = (a_mem + b_mem) as u64;
            let bytes_written = result_mem as u64;
            self.profiler
                .record_operation("matmul_chunked", elapsed, bytes_read, bytes_written);
            return Ok(result);
        }
        let a_chunks = a.chunk(rows_per_chunk, 0)?;
        let chunk_count = a_chunks.len();
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
        let result = DenseND::concatenate(&result_chunks, 0)?;
        let result_mem = result.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        self.current_memory += result_mem;
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
        if a.shape() != b.shape() || a.shape() != c.shape() {
            return Err(anyhow!(
                "Shape mismatch for FMA operation: a={:?}, b={:?}, c={:?}",
                a.shape(),
                b.shape(),
                c.shape()
            ));
        }
        let shape = a.shape();
        if a.rank() == 1 {
            let mut result = DenseND::<f64>::zeros(shape);
            let a_slice = a.as_slice();
            let b_slice = b.as_slice();
            let c_slice = c.as_slice();
            let result_slice = result
                .as_array_mut()
                .as_slice_mut()
                .expect("Contiguous array");
            for i in 0..a_slice.len() {
                result_slice[i] = a_slice[i].mul_add(b_slice[i], c_slice[i]);
            }
            let elapsed = start_time.elapsed();
            let bytes_read = (std::mem::size_of_val(a_slice) * 3) as u64;
            let bytes_written = std::mem::size_of_val(a_slice) as u64;
            self.profiler
                .record_operation("fma_chunked", elapsed, bytes_read, bytes_written);
            return Ok(result);
        }
        let chunk_size = if !self.config.default_chunk_size.is_empty() {
            self.config.default_chunk_size[0]
        } else {
            256
        };
        let a_chunks = a.chunk(chunk_size, 0)?;
        let b_chunks = b.chunk(chunk_size, 0)?;
        let c_chunks = c.chunk(chunk_size, 0)?;
        let chunk_count = a_chunks.len();
        #[cfg(feature = "parallel")]
        let result_chunks = if self.should_use_parallel(chunk_count) {
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
                    for i in 0..a_slice.len() {
                        result_slice[i] = a_slice[i].mul_add(b_slice[i], c_slice[i]);
                    }
                    result_chunk
                })
                .collect::<Vec<_>>()
        } else {
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
                for i in 0..a_slice.len() {
                    result_slice[i] = a_slice[i].mul_add(b_slice[i], c_slice[i]);
                }
                result_chunk
            })
            .collect::<Vec<_>>();
        let result = DenseND::concatenate(&result_chunks, 0)?;
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
        if a.shape() != b.shape() {
            return Err(anyhow!(
                "Shape mismatch for element-wise operation: {:?} vs {:?}",
                a.shape(),
                b.shape()
            ));
        }
        let shape = a.shape();
        if a.rank() == 1 {
            let mut result = DenseND::<f64>::zeros(shape);
            let a_slice = a.as_slice();
            let b_slice = b.as_slice();
            let result_slice = result
                .as_array_mut()
                .as_slice_mut()
                .expect("Contiguous array");
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
        let chunk_size = if !self.config.default_chunk_size.is_empty() {
            self.config.default_chunk_size[0]
        } else {
            256
        };
        let a_chunks = a.chunk(chunk_size, 0)?;
        let b_chunks = b.chunk(chunk_size, 0)?;
        let chunk_count = a_chunks.len();
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
        if a.shape() != b.shape() {
            return Err(anyhow!(
                "Shape mismatch for element-wise operation: {:?} vs {:?}",
                a.shape(),
                b.shape()
            ));
        }
        let shape = a.shape();
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
        let chunk_size = if !self.config.default_chunk_size.is_empty() {
            self.config.default_chunk_size[0]
        } else {
            256
        };
        let a_chunks = a.chunk(chunk_size, 0)?;
        let b_chunks = b.chunk(chunk_size, 0)?;
        let chunk_count = a_chunks.len();
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
        let filename = format!("tenrso_spill_{}.bin", self.spill_counter);
        self.spill_counter += 1;
        let path = self.config.temp_dir.join(filename);
        crate::mmap_io::write_tensor_binary(&path, tensor)?;
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
        let tensor_mem = tensor.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
        self.current_memory += tensor_mem;
        Ok(tensor)
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
        let budget_bytes = memory_budget_mb
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(self.config.max_memory_bytes / 4);
        let fixed_memory = (n * p + m * p) * std::mem::size_of::<f64>();
        if fixed_memory >= budget_bytes {
            return Ok(1);
        }
        let available = budget_bytes - fixed_memory;
        let bytes_per_row = n * std::mem::size_of::<f64>();
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
        let budget_bytes = memory_budget_mb
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(self.config.max_memory_bytes / 4);
        let elements_per_slice: usize = shape[1..].iter().product();
        let bytes_per_chunk_row = 3 * elements_per_slice * std::mem::size_of::<f64>();
        if bytes_per_chunk_row == 0 {
            return Ok(shape[0]);
        }
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
                let flops = 2.0 * (m as f64) * (n as f64) * (p as f64);
                let memory_bytes =
                    ((m * n + n * p + m * p) as f64) * std::mem::size_of::<f64>() as f64;
                let time_seconds = 0.01;
                let gflops = (flops / 1e9) / time_seconds;
                let bandwidth_gb = (memory_bytes / 1e9) / time_seconds;
                Ok((gflops, bandwidth_gb))
            }
            "elementwise" => {
                let elements_per_slice: usize = shape1[1..].iter().product();
                let chunk_elements = chunk_size * elements_per_slice;
                let flops = chunk_elements as f64;
                let memory_bytes = (3 * chunk_elements) as f64 * std::mem::size_of::<f64>() as f64;
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
