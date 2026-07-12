//! Out-of-core (streaming, chunked) MTTKRP.
//!
//! MTTKRP is the inner kernel of CP-ALS. For a tensor `X ∈ R^{I_0 × … × I_{N-1}}` and
//! factor matrices `A_k ∈ R^{I_k × R}`, the mode-`n` MTTKRP is the `I_n × R` matrix
//!
//! ```text
//! M_n[i_n, r] = Σ_{i_j : j ≠ n}  X[i_0, …, i_{N-1}] · Π_{j ≠ n} A_j[i_j, r]
//! ```
//!
//! which [`tenrso_kernels::mttkrp`](fn@tenrso_kernels::mttkrp) computes in-core as `unfold_n(X) · KR(A_{j≠n})`.
//!
//! # Why it streams
//!
//! The defining sum ranges over the *elements* of `X`, and each element contributes to
//! exactly one row of `M_n`. So MTTKRP is **additive over any partition of the index
//! space**: cut `X` into disjoint sub-boxes `B_1, …, B_C` and
//!
//! ```text
//! M_n = Σ_c  M_n(B_c)      (each B_c's contribution landing in its own row range)
//! ```
//!
//! The accumulator is `I_n × R` — *independent of the number of elements in X*. That is
//! the whole trick: peak memory is set by the factors, the accumulator, and however many
//! chunks we choose to hold at once, never by `|X|`.
//!
//! # Global index offsets (the one thing that must be exactly right)
//!
//! A chunk covering the box `[s_k, e_k)` is handed to the kernel as a *local* tensor with
//! axis `k` running `0 .. e_k - s_k`. Its local index `j_k` means global index
//! `s_k + j_k`. Two corrections are therefore mandatory, and both are applied in
//! [`chunk_partials`] / [`MttkrpAccumulator::accumulate_chunk`]:
//!
//! 1. **Input side.** The chunk must be multiplied by the factor rows belonging to *its*
//!    global slice, so factor `A_k` is row-sliced to `A_k[s_k .. e_k, :]` before the
//!    kernel sees it. Feeding the full `A_k` (or, worse, `A_k[0 .. c_k, :]`) multiplies
//!    the chunk by the wrong rows and yields a plausible, silently wrong answer.
//! 2. **Output side.** The kernel returns a `(e_n - s_n) × R` block whose row `j`
//!    is global row `s_n + j`, so it is accumulated into `M_n[s_n .. e_n, :]`, not into
//!    `M_n[0 .. c_n, :]`.
//!
//! With both corrections, the chunk's partial is *exactly* the restriction of the defining
//! sum to that box — no approximation, no boundary term. Ragged edge chunks need no
//! special case at all: they are just boxes with a smaller extent.
//!
//! # One pass, every mode
//!
//! Each element of `X` contributes to *every* mode's accumulator, so a full CP-ALS sweep
//! does not need `N` passes over the data. [`StreamingMttkrp::mttkrp_all_modes`] keeps `N`
//! accumulators alive (still tiny: `Σ_n I_n · R`) and, per chunk, computes all `N` mode
//! partials with [`tenrso_kernels::mttkrp_all_modes`] — the dimension-tree kernel, which
//! shares partial contractions across modes. A sweep therefore costs **one** disk pass
//! (instead of `N`) and `≈ 2·nnz·R` multiply-adds (instead of `N·nnz·R`).
//!
//! # Determinism and accumulation order
//!
//! The result is a floating-point sum, so its low bits depend on the summation order. The
//! order is fixed and documented:
//!
//! * Chunks are visited in **ascending linear chunk index** — the row-major order over the
//!   chunk grid that [`ChunkSpec::iter`] yields.
//! * Chunk partials are reduced into the accumulator in that same ascending order,
//!   **never in completion order**. When chunks are processed in parallel, a window of
//!   partials is computed concurrently and then folded in ascending index order.
//! * The per-chunk kernel is always the *serial* kernel.
//!
//! Consequences, all of which are asserted in the test-suite:
//!
//! * The result is **bit-identical** regardless of thread count, window size, memory
//!   budget, and the order in which chunk reads happen to complete.
//! * The result is **not** bit-identical to the in-core kernel, nor across different chunk
//!   sizes — those change the grouping of the sum. They agree to floating-point tolerance
//!   (relative Frobenius error ~1e-14 in practice for f64; chunking is, if anything,
//!   slightly *more* accurate, being a partially pairwise summation).
//!
//! # Memory bound
//!
//! Peak working set is bounded, by construction, by
//!
//! ```text
//! accumulators      Σ_{n ∈ modes} I_n · R · 8
//! + W · chunk       W · (∏_k min(chunk_size_k, I_k)) · 8
//! + W · partials    W · (Σ_{n ∈ modes} min(chunk_size_n, I_n) · R · 8)
//! ≤ config.max_memory_bytes
//! ```
//!
//! where the window `W` is *derived from* the budget (see [`MttkrpStreamPlan`]) — the
//! budget is the input, the schedule is the output. Chunk residency is additionally
//! policed at run time by a [`MemoryManager`] whose limit is set to the chunk share of the
//! budget, so an over-run would surface as a loud error, not as swapping.
//!
//! Factor matrices are borrowed from the caller and are *not* counted in the budget; they
//! are the caller's fixed `Σ_k I_k · R · 8` cost, unavoidable for any MTTKRP.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{s, Array2, ArrayView2};
use tenrso_core::DenseND;

use crate::chunk_source::ChunkSource;
use crate::chunking::{ChunkIndex, ChunkSpec};
use crate::memory::{AccessPattern, MemoryManager, SpillPolicy};
use crate::profiling::Profiler;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

/// Bytes per `f64` element.
const F64_BYTES: usize = std::mem::size_of::<f64>();

/// Configuration for a streaming MTTKRP run.
///
/// The only knob that really matters is [`max_memory_bytes`](field@Self::max_memory_bytes):
/// it is a hard bound on the streaming working set, and everything else (how many chunks
/// are in flight, how much parallelism is available) is derived from it.
#[derive(Debug, Clone)]
pub struct MttkrpStreamConfig {
    /// Hard bound on the streaming working set: accumulators + in-flight chunks +
    /// in-flight partials. Default: 256 MiB.
    ///
    /// Does **not** include the caller's factor matrices.
    pub max_memory_bytes: usize,
    /// Upper bound on chunks held in flight, regardless of how much budget is left.
    /// Default: `8 × num_cpus`.
    pub max_window_chunks: usize,
    /// Process the chunks of a window concurrently (requires the `parallel` feature).
    /// Has no effect on the numerical result — the reduction order is fixed. Default:
    /// `true` when the `parallel` feature is on.
    pub parallel: bool,
    /// Record per-phase timings into the executor's [`Profiler`]. Default: `false`.
    pub enable_profiling: bool,
    /// Directory the chunk [`MemoryManager`] would use if it ever had to spill. A
    /// streaming MTTKRP never spills (chunks are consume-once and released immediately),
    /// so in practice nothing is written here. Default: [`std::env::temp_dir`].
    pub temp_dir: PathBuf,
    /// Spill policy handed to the chunk [`MemoryManager`]. Default: [`SpillPolicy::LRU`].
    pub spill_policy: SpillPolicy,
}

impl Default for MttkrpStreamConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 256 * 1024 * 1024,
            max_window_chunks: 8 * num_cpus::get().max(1),
            parallel: cfg!(feature = "parallel"),
            enable_profiling: false,
            temp_dir: std::env::temp_dir(),
            spill_policy: SpillPolicy::LRU,
        }
    }
}

impl MttkrpStreamConfig {
    /// Create a configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hard working-set bound, in bytes.
    pub fn max_memory_bytes(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Set the hard working-set bound, in mebibytes.
    pub fn max_memory_mb(mut self, mb: usize) -> Self {
        self.max_memory_bytes = mb * 1024 * 1024;
        self
    }

    /// Cap the number of chunks held in flight.
    pub fn max_window_chunks(mut self, chunks: usize) -> Self {
        self.max_window_chunks = chunks.max(1);
        self
    }

    /// Enable or disable concurrent processing of a window's chunks.
    pub fn parallel(mut self, enable: bool) -> Self {
        self.parallel = enable;
        self
    }

    /// Enable or disable per-phase profiling.
    pub fn enable_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    /// Set the directory used by the chunk memory manager for (never-taken) spills.
    pub fn temp_dir<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.temp_dir = path.into();
        self
    }

    /// Set the spill policy of the chunk memory manager.
    pub fn spill_policy(mut self, policy: SpillPolicy) -> Self {
        self.spill_policy = policy;
        self
    }
}

/// The bounded-memory execution schedule derived from a budget.
///
/// This is the *proof object* for the memory claim: every resident byte of a streaming
/// MTTKRP is accounted for by one of these fields, and
/// [`peak_working_set_bytes`](Self::peak_working_set_bytes) is `≤ max_memory_bytes` by
/// construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MttkrpStreamPlan {
    /// Chunks held in flight simultaneously.
    pub window_chunks: usize,
    /// Total chunks in the grid.
    pub total_chunks: usize,
    /// Bytes of the largest possible chunk (the interior, non-ragged one).
    pub chunk_bytes: usize,
    /// Bytes of one chunk's partial results, summed over the requested modes.
    pub partial_bytes_per_chunk: usize,
    /// Bytes of the mode accumulators, summed over the requested modes.
    pub accumulator_bytes: usize,
    /// The budget this plan was derived from.
    pub budget_bytes: usize,
}

impl MttkrpStreamPlan {
    /// Derive the schedule for `spec`/`modes`/`cp_rank` under `config`'s budget.
    ///
    /// # Errors
    ///
    /// Returns an error if the budget cannot even hold the accumulators plus a single
    /// chunk — the irreducible minimum. The message states the minimum required.
    pub fn new(
        spec: &ChunkSpec,
        cp_rank: usize,
        modes: &[usize],
        config: &MttkrpStreamConfig,
    ) -> Result<Self> {
        let shape = spec.tensor_shape();
        let cs = spec.chunk_size();

        // A chunk's extent along axis k is at most min(chunk_size_k, I_k): a chunk size
        // larger than the dimension is clamped, never padded.
        let chunk_elems: usize = cs
            .iter()
            .zip(shape.iter())
            .map(|(&c, &d)| c.min(d))
            .product();
        let chunk_bytes = chunk_elems * F64_BYTES;

        let partial_bytes_per_chunk: usize = modes
            .iter()
            .map(|&n| cs[n].min(shape[n]) * cp_rank * F64_BYTES)
            .sum();

        let accumulator_bytes: usize = modes.iter().map(|&n| shape[n] * cp_rank * F64_BYTES).sum();

        let per_chunk = chunk_bytes + partial_bytes_per_chunk;
        let minimum = accumulator_bytes + per_chunk;
        if config.max_memory_bytes < minimum {
            return Err(anyhow!(
                "Memory budget of {} bytes is too small for streaming MTTKRP: the \
                 accumulators need {} bytes and a single chunk of {:?} needs {} bytes \
                 (data) + {} bytes (partials) = {} bytes minimum. Either raise the budget \
                 to >= {} bytes or use a smaller chunk size.",
                config.max_memory_bytes,
                accumulator_bytes,
                cs,
                chunk_bytes,
                partial_bytes_per_chunk,
                per_chunk,
                minimum
            ));
        }

        let available = config.max_memory_bytes - accumulator_bytes;
        let total_chunks = spec.total_chunks();
        let window_chunks = (available / per_chunk)
            .min(config.max_window_chunks)
            .min(total_chunks)
            .max(1);

        Ok(Self {
            window_chunks,
            total_chunks,
            chunk_bytes,
            partial_bytes_per_chunk,
            accumulator_bytes,
            budget_bytes: config.max_memory_bytes,
        })
    }

    /// Bound on chunk *data* bytes resident at once — the limit given to the
    /// [`MemoryManager`].
    pub fn chunk_budget_bytes(&self) -> usize {
        self.window_chunks * self.chunk_bytes
    }

    /// Worst-case resident bytes of the whole streaming pass.
    ///
    /// Invariant: `peak_working_set_bytes() <= budget_bytes`.
    pub fn peak_working_set_bytes(&self) -> usize {
        self.accumulator_bytes
            + self.window_chunks * (self.chunk_bytes + self.partial_bytes_per_chunk)
    }

    /// Number of windows the pass will take.
    pub fn num_windows(&self) -> usize {
        self.total_chunks.div_ceil(self.window_chunks)
    }
}

/// Statistics of a completed streaming MTTKRP run.
#[derive(Debug, Clone)]
pub struct MttkrpStreamStats {
    /// The schedule that was executed.
    pub plan: MttkrpStreamPlan,
    /// Chunks read and accumulated.
    pub chunks_processed: usize,
    /// Tensor bytes pulled from the source.
    pub bytes_read: usize,
    /// High-water mark of chunk-data residency, as observed by the [`MemoryManager`].
    pub peak_chunk_bytes: usize,
    /// Modes that were accumulated, ascending.
    pub modes: Vec<usize>,
    /// Backing store the chunks came from.
    pub source_name: &'static str,
    /// Wall-clock time reading chunks.
    pub io_time: Duration,
    /// Wall-clock time computing chunk partials.
    pub compute_time: Duration,
    /// Wall-clock time folding partials into the accumulators.
    pub reduce_time: Duration,
    /// Total wall-clock time.
    pub total_time: Duration,
}

impl MttkrpStreamStats {
    /// Observed peak working set: accumulators + peak chunk residency + in-flight partials.
    ///
    /// This is the number to compare against the configured budget; it uses the
    /// *measured* chunk high-water mark rather than the planned bound.
    pub fn peak_working_set_bytes(&self) -> usize {
        self.plan.accumulator_bytes
            + self.peak_chunk_bytes
            + self.plan.window_chunks * self.plan.partial_bytes_per_chunk
    }

    /// Effective read bandwidth in MiB/s (bytes read / I/O time).
    pub fn read_bandwidth_mib_s(&self) -> f64 {
        let secs = self.io_time.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        (self.bytes_read as f64 / (1024.0 * 1024.0)) / secs
    }
}

/// The `I_n × R` accumulators of a streaming MTTKRP, one per requested mode.
///
/// Fold chunk partials in with [`accumulate_chunk`](Self::accumulate_chunk); the caller is
/// responsible for doing so in ascending chunk order (that is what makes the result
/// reproducible).
#[derive(Debug, Clone)]
pub struct MttkrpAccumulator {
    shape: Vec<usize>,
    cp_rank: usize,
    modes: Vec<usize>,
    acc: Vec<Array2<f64>>,
    chunks_accumulated: usize,
}

impl MttkrpAccumulator {
    /// Allocate zeroed `I_n × R` accumulators for each mode in `modes`.
    ///
    /// # Errors
    ///
    /// Returns an error if `modes` is empty, contains a duplicate, or names an axis that
    /// does not exist; or if `cp_rank` is zero.
    pub fn new(shape: &[usize], cp_rank: usize, modes: &[usize]) -> Result<Self> {
        if cp_rank == 0 {
            return Err(anyhow!("CP rank must be >= 1"));
        }
        if modes.is_empty() {
            return Err(anyhow!("At least one mode must be requested"));
        }
        let mut sorted = modes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != modes.len() {
            return Err(anyhow!("Duplicate mode in {:?}", modes));
        }
        if let Some(&bad) = sorted.iter().find(|&&n| n >= shape.len()) {
            return Err(anyhow!(
                "Mode {} out of bounds for tensor of rank {}",
                bad,
                shape.len()
            ));
        }

        let acc = sorted
            .iter()
            .map(|&n| Array2::<f64>::zeros((shape[n], cp_rank)))
            .collect();

        Ok(Self {
            shape: shape.to_vec(),
            cp_rank,
            modes: sorted,
            acc,
            chunks_accumulated: 0,
        })
    }

    /// The modes being accumulated, ascending.
    pub fn modes(&self) -> &[usize] {
        &self.modes
    }

    /// CP rank (columns of every accumulator).
    pub fn cp_rank(&self) -> usize {
        self.cp_rank
    }

    /// Number of chunk partials folded in so far.
    pub fn chunks_accumulated(&self) -> usize {
        self.chunks_accumulated
    }

    /// Resident bytes of the accumulators.
    pub fn bytes(&self) -> usize {
        self.modes
            .iter()
            .map(|&n| self.shape[n] * self.cp_rank * F64_BYTES)
            .sum()
    }

    /// Fold one chunk's partials into the accumulators.
    ///
    /// `start` is the chunk's **global** origin; `partials[i]` is the partial for
    /// `self.modes()[i]` and has `end[modes[i]] - start[modes[i]]` rows. Row `j` of
    /// `partials[i]` lands on global row `start[modes[i]] + j` — this is the output-side
    /// index offset, and it is the reason `start` must be threaded all the way here.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of partials, their column count, or their row count
    /// disagrees with the chunk geometry, or if a partial would write past the end of an
    /// accumulator.
    pub fn accumulate_chunk(&mut self, start: &[usize], partials: &[Array2<f64>]) -> Result<()> {
        if start.len() != self.shape.len() {
            return Err(anyhow!(
                "Chunk origin rank {} does not match tensor rank {}",
                start.len(),
                self.shape.len()
            ));
        }
        if partials.len() != self.modes.len() {
            return Err(anyhow!(
                "Expected {} partials (one per mode), got {}",
                self.modes.len(),
                partials.len()
            ));
        }

        for (i, &mode) in self.modes.iter().enumerate() {
            let partial = &partials[i];
            let rows = partial.nrows();
            if partial.ncols() != self.cp_rank {
                return Err(anyhow!(
                    "Partial for mode {} has {} columns, expected CP rank {}",
                    mode,
                    partial.ncols(),
                    self.cp_rank
                ));
            }
            let offset = start[mode];
            if offset + rows > self.shape[mode] {
                return Err(anyhow!(
                    "Partial for mode {} covers global rows {}..{}, past dimension {}",
                    mode,
                    offset,
                    offset + rows,
                    self.shape[mode]
                ));
            }

            // Output-side global offset: local row j -> global row start[mode] + j.
            let mut dst = self.acc[i].slice_mut(s![offset..offset + rows, ..]);
            dst.zip_mut_with(partial, |a, &b| *a += b);
        }

        self.chunks_accumulated += 1;
        Ok(())
    }

    /// Borrow the accumulator of `mode`.
    pub fn get(&self, mode: usize) -> Option<&Array2<f64>> {
        self.modes
            .iter()
            .position(|&n| n == mode)
            .map(|i| &self.acc[i])
    }

    /// Consume the accumulator, returning one `I_n × R` matrix per mode, ascending.
    pub fn into_results(self) -> Vec<Array2<f64>> {
        self.acc
    }
}

/// Compute one chunk's MTTKRP partials for the requested modes.
///
/// This is where the **input-side** global index offset lives: factor `A_k` is row-sliced
/// to `A_k[start_k .. end_k, :]` so that the chunk is multiplied by the factor rows of its
/// own global slice. The per-chunk math itself is delegated to `tenrso-kernels` — this
/// crate does not reimplement MTTKRP.
///
/// When every mode is requested (`modes.len() == N`, `N >= 2`), the dimension-tree kernel
/// [`tenrso_kernels::mttkrp_all_modes`] computes all `N` partials with shared intermediate
/// contractions (`≈ 2·nnz·R` multiply-adds instead of `N·nnz·R`). Otherwise the modes are
/// computed one at a time with [`tenrso_kernels::mttkrp`](fn@tenrso_kernels::mttkrp).
///
/// The *serial* kernels are used deliberately: parallelism in this crate is across chunks,
/// which keeps the reduction order — and therefore the result, bit for bit — independent
/// of the thread count.
///
/// # Errors
///
/// Propagates any kernel error (shape/rank mismatch).
pub fn chunk_partials(
    block: &DenseND<f64>,
    factors: &[ArrayView2<f64>],
    start: &[usize],
    end: &[usize],
    modes: &[usize],
) -> Result<Vec<Array2<f64>>> {
    let order = factors.len();

    // Input-side global offset: the chunk's axis-k slice must meet A_k's rows [s_k, e_k).
    let sub_factors: Vec<ArrayView2<f64>> = (0..order)
        .map(|k| factors[k].slice(s![start[k]..end[k], ..]))
        .collect();

    let view = block.as_array().view();

    if modes.len() == order && order >= 2 {
        // Full sweep: share partial contractions across all modes.
        tenrso_kernels::mttkrp_all_modes(&view, &sub_factors)
    } else {
        modes
            .iter()
            .map(|&mode| tenrso_kernels::mttkrp(&view, &sub_factors, mode))
            .collect()
    }
}

/// Streaming MTTKRP executor.
///
/// Owns the memory manager (back-pressure), the profiler, and the run statistics. Reusable
/// across calls: a CP-ALS driver builds one and calls
/// [`mttkrp_all_modes`](Self::mttkrp_all_modes) once per ALS iteration.
pub struct StreamingMttkrp {
    config: MttkrpStreamConfig,
    profiler: Profiler,
    last_stats: Option<MttkrpStreamStats>,
}

impl StreamingMttkrp {
    /// Create an executor with the given configuration.
    pub fn new(config: MttkrpStreamConfig) -> Self {
        let mut profiler = Profiler::new();
        profiler.set_enabled(config.enable_profiling);
        Self {
            config,
            profiler,
            last_stats: None,
        }
    }

    /// The configuration in force.
    pub fn config(&self) -> &MttkrpStreamConfig {
        &self.config
    }

    /// The profiler (populated only when `enable_profiling` is set).
    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    /// Statistics of the most recent run, if any.
    pub fn stats(&self) -> Option<&MttkrpStreamStats> {
        self.last_stats.as_ref()
    }

    /// Streaming MTTKRP for a single mode.
    ///
    /// Result is numerically equal (to floating-point tolerance) to
    /// `tenrso_kernels::mttkrp(&x.view(), factors, mode)` for the tensor the source backs.
    ///
    /// # Errors
    ///
    /// See [`Self::mttkrp_modes`].
    pub fn mttkrp(
        &mut self,
        source: &dyn ChunkSource,
        spec: &ChunkSpec,
        factors: &[ArrayView2<f64>],
        mode: usize,
    ) -> Result<Array2<f64>> {
        let mut results = self.mttkrp_modes(source, spec, factors, &[mode])?;
        results
            .pop()
            .ok_or_else(|| anyhow!("internal: streaming MTTKRP produced no result"))
    }

    /// Streaming MTTKRP for **every** mode, in a single pass over the data.
    ///
    /// Returns `result[n] == mttkrp(X, factors, n)` for `n in 0..N`. This is what a CP-ALS
    /// sweep wants: one disk pass instead of `N`.
    ///
    /// # Errors
    ///
    /// See [`Self::mttkrp_modes`].
    pub fn mttkrp_all_modes(
        &mut self,
        source: &dyn ChunkSource,
        spec: &ChunkSpec,
        factors: &[ArrayView2<f64>],
    ) -> Result<Vec<Array2<f64>>> {
        let modes: Vec<usize> = (0..factors.len()).collect();
        self.mttkrp_modes(source, spec, factors, &modes)
    }

    /// Streaming MTTKRP for an arbitrary set of modes, in a single pass over the data.
    ///
    /// Results are returned in **ascending mode order** (the request order is irrelevant;
    /// duplicates are rejected).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - the tensor order is `< 2`, or the factor count does not match it;
    /// - a factor's row count disagrees with the corresponding tensor dimension, or the
    ///   factors disagree about the CP rank;
    /// - `spec` does not describe the source's shape, or the source stores a different
    ///   grid ([`ChunkSource::native_chunk_spec`]);
    /// - a mode is out of bounds or repeated;
    /// - the memory budget cannot hold the accumulators plus one chunk;
    /// - the source fails to serve a chunk.
    pub fn mttkrp_modes(
        &mut self,
        source: &dyn ChunkSource,
        spec: &ChunkSpec,
        factors: &[ArrayView2<f64>],
        modes: &[usize],
    ) -> Result<Vec<Array2<f64>>> {
        let started = Instant::now();
        let shape = source.shape().to_vec();

        validate_inputs(&shape, spec, factors, modes)?;
        if let Some(native) = source.native_chunk_spec() {
            if native != spec {
                return Err(anyhow!(
                    "Source {} stores a fixed chunk grid (chunk size {:?}) and cannot serve \
                     boxes of the requested grid (chunk size {:?}); stream it with its own \
                     ChunkSpec (see ChunkSource::native_chunk_spec)",
                    source.source_name(),
                    native.chunk_size(),
                    spec.chunk_size()
                ));
            }
        }

        let cp_rank = factors[0].ncols();
        let plan = MttkrpStreamPlan::new(spec, cp_rank, modes, &self.config)?;

        let mut accumulator = MttkrpAccumulator::new(&shape, cp_rank, modes)?;
        let sorted_modes = accumulator.modes().to_vec();

        // The chunk memory manager polices chunk residency against the chunk share of the
        // budget. auto_spill is off: chunks are consume-once, so the correct response to
        // pressure is to consume them, never to write them back out to disk.
        let mut memory = MemoryManager::new()
            .max_memory_bytes(plan.chunk_budget_bytes())
            .spill_policy(self.config.spill_policy)
            .temp_dir(&self.config.temp_dir)
            .auto_spill(false)
            .pressure_threshold(1.0);

        let mut io_time = Duration::ZERO;
        let mut compute_time = Duration::ZERO;
        let mut reduce_time = Duration::ZERO;
        let mut bytes_read = 0usize;
        let mut chunks_processed = 0usize;

        let mut window_start = 0usize;
        while window_start < plan.total_chunks {
            let window_end = (window_start + plan.window_chunks).min(plan.total_chunks);

            // --- Plan the window: ascending linear chunk index, always. ---
            let boxes: Vec<ChunkBox> = (window_start..window_end)
                .map(|linear| {
                    let idx = ChunkIndex::from_linear(linear, spec.num_chunks());
                    let (start, end) = spec.chunk_bounds(&idx);
                    ChunkBox { linear, start, end }
                })
                .collect();

            // --- Load: back-pressure is enforced here, by the window size itself. ---
            let io_started = Instant::now();
            let blocks = self.load_window(source, &boxes)?;
            io_time += io_started.elapsed();

            for (b, block) in boxes.iter().zip(blocks.into_iter()) {
                bytes_read += block.len() * F64_BYTES;
                if !memory.can_fit(block.len() * F64_BYTES) {
                    return Err(anyhow!(
                        "internal: window of {} chunks overran its {}-byte chunk budget at \
                         chunk {} — the plan is inconsistent",
                        plan.window_chunks,
                        plan.chunk_budget_bytes(),
                        b.linear
                    ));
                }
                memory.register_chunk(&chunk_id(b.linear), block, AccessPattern::ReadOnce)?;
            }

            // --- Compute: per-chunk partials, independent, order-free. ---
            let compute_started = Instant::now();
            let partials = self.compute_window(&memory, &boxes, factors, &sorted_modes)?;
            compute_time += compute_started.elapsed();

            // --- Reduce: ascending chunk index, NEVER completion order. ---
            let reduce_started = Instant::now();
            for (b, part) in boxes.iter().zip(partials.iter()) {
                accumulator.accumulate_chunk(&b.start, part)?;
            }
            reduce_time += reduce_started.elapsed();

            // --- Release: consume-once chunks are dropped, not spilled. ---
            for b in &boxes {
                memory.release_chunk(&chunk_id(b.linear))?;
            }

            chunks_processed += boxes.len();
            window_start = window_end;
        }

        let total_time = started.elapsed();

        if self.config.enable_profiling {
            self.profiler
                .record_operation("mttkrp_stream_io", io_time, bytes_read as u64, 0);
            self.profiler
                .record_operation("mttkrp_stream_compute", compute_time, 0, 0);
            self.profiler
                .record_operation("mttkrp_stream_reduce", reduce_time, 0, 0);
        }

        self.last_stats = Some(MttkrpStreamStats {
            plan,
            chunks_processed,
            bytes_read,
            peak_chunk_bytes: memory.peak_memory(),
            modes: sorted_modes,
            source_name: source.source_name(),
            io_time,
            compute_time,
            reduce_time,
            total_time,
        });

        debug_assert_eq!(accumulator.chunks_accumulated(), chunks_processed);
        Ok(accumulator.into_results())
    }

    /// Fetch a window's chunks, in ascending chunk order.
    ///
    /// Reads may run concurrently (they are pure functions of the box), which overlaps I/O
    /// on any source that supports it; the returned vector is always in ascending order, so
    /// the downstream reduction order is unaffected.
    fn load_window(
        &self,
        source: &dyn ChunkSource,
        boxes: &[ChunkBox],
    ) -> Result<Vec<DenseND<f64>>> {
        #[cfg(feature = "parallel")]
        if self.config.parallel && boxes.len() > 1 {
            return boxes
                .par_iter()
                .map(|b| source.read_chunk(&b.start, &b.end))
                .collect::<Result<Vec<_>>>();
        }

        boxes
            .iter()
            .map(|b| source.read_chunk(&b.start, &b.end))
            .collect()
    }

    /// Compute the partials of every chunk in the window.
    ///
    /// Chunks are independent, so this may run concurrently; the result vector is in
    /// ascending chunk order regardless.
    fn compute_window(
        &self,
        memory: &MemoryManager,
        boxes: &[ChunkBox],
        factors: &[ArrayView2<f64>],
        modes: &[usize],
    ) -> Result<Vec<Vec<Array2<f64>>>> {
        let compute_one = |b: &ChunkBox| -> Result<Vec<Array2<f64>>> {
            let block = memory.get_chunk(&chunk_id(b.linear)).ok_or_else(|| {
                anyhow!(
                    "internal: chunk {} vanished from the memory manager mid-window",
                    b.linear
                )
            })?;
            chunk_partials(block, factors, &b.start, &b.end, modes)
        };

        #[cfg(feature = "parallel")]
        if self.config.parallel && boxes.len() > 1 {
            return boxes
                .par_iter()
                .map(compute_one)
                .collect::<Result<Vec<_>>>();
        }

        boxes.iter().map(compute_one).collect()
    }
}

/// One chunk's global bounding box plus its position in the deterministic chunk order.
#[derive(Debug, Clone)]
struct ChunkBox {
    /// Linear index in the chunk grid; also the reduction order key.
    linear: usize,
    /// Global origin (inclusive).
    start: Vec<usize>,
    /// Global end (exclusive).
    end: Vec<usize>,
}

/// Stable identifier of a chunk within the memory manager.
fn chunk_id(linear: usize) -> String {
    format!("mttkrp_chunk_{}", linear)
}

/// Validate tensor/spec/factor/mode agreement before any I/O happens.
fn validate_inputs(
    shape: &[usize],
    spec: &ChunkSpec,
    factors: &[ArrayView2<f64>],
    modes: &[usize],
) -> Result<()> {
    let order = shape.len();
    if order < 2 {
        return Err(anyhow!(
            "Streaming MTTKRP requires a tensor of order >= 2, got {}",
            order
        ));
    }
    if spec.tensor_shape() != shape {
        return Err(anyhow!(
            "ChunkSpec describes shape {:?} but the source holds shape {:?}",
            spec.tensor_shape(),
            shape
        ));
    }
    if factors.len() != order {
        return Err(anyhow!(
            "Expected {} factor matrices (tensor order), got {}",
            order,
            factors.len()
        ));
    }
    let cp_rank = factors[0].ncols();
    if cp_rank == 0 {
        return Err(anyhow!("CP rank must be >= 1"));
    }
    for (k, f) in factors.iter().enumerate() {
        if f.nrows() != shape[k] {
            return Err(anyhow!(
                "Factor {} has {} rows, expected {} (tensor dimension {})",
                k,
                f.nrows(),
                shape[k],
                k
            ));
        }
        if f.ncols() != cp_rank {
            return Err(anyhow!(
                "Factor {} has {} columns, expected CP rank {}",
                k,
                f.ncols(),
                cp_rank
            ));
        }
    }
    if modes.is_empty() {
        return Err(anyhow!("At least one mode must be requested"));
    }
    let mut seen = vec![false; order];
    for &m in modes {
        if m >= order {
            return Err(anyhow!(
                "Mode {} out of bounds for tensor of order {}",
                m,
                order
            ));
        }
        if seen[m] {
            return Err(anyhow!("Mode {} requested more than once", m));
        }
        seen[m] = true;
    }
    Ok(())
}

/// Streaming MTTKRP for one mode (convenience wrapper over [`StreamingMttkrp`]).
///
/// # Errors
///
/// See [`StreamingMttkrp::mttkrp_modes`].
pub fn streaming_mttkrp(
    source: &dyn ChunkSource,
    spec: &ChunkSpec,
    factors: &[ArrayView2<f64>],
    mode: usize,
    config: MttkrpStreamConfig,
) -> Result<Array2<f64>> {
    StreamingMttkrp::new(config).mttkrp(source, spec, factors, mode)
}

/// Streaming MTTKRP for every mode in a single pass (convenience wrapper).
///
/// Returns the `N` results together with the run statistics.
///
/// # Errors
///
/// See [`StreamingMttkrp::mttkrp_modes`].
pub fn streaming_mttkrp_all_modes(
    source: &dyn ChunkSource,
    spec: &ChunkSpec,
    factors: &[ArrayView2<f64>],
    config: MttkrpStreamConfig,
) -> Result<(Vec<Array2<f64>>, MttkrpStreamStats)> {
    let mut executor = StreamingMttkrp::new(config);
    let results = executor.mttkrp_all_modes(source, spec, factors)?;
    let stats = executor
        .stats()
        .cloned()
        .ok_or_else(|| anyhow!("internal: streaming MTTKRP recorded no statistics"))?;
    Ok((results, stats))
}

#[cfg(test)]
// `for mode in 0..N { ... mttkrp(x, views, mode) ... }` uses `mode` as a semantic value
// (the mode fed to the kernel), not merely as an index, so `enumerate` would not read more
// clearly here.
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::chunk_source::DenseChunkSource;
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::ndarray_ext::IxDyn;

    fn deterministic_tensor(shape: &[usize]) -> DenseND<f64> {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n)
            .map(|i| ((i * 37 % 101) as f64) / 101.0 - 0.5)
            .collect();
        DenseND::from_vec(data, shape).expect("shape/data agree by construction")
    }

    fn deterministic_factors(shape: &[usize], cp_rank: usize) -> Vec<Array2<f64>> {
        shape
            .iter()
            .enumerate()
            .map(|(k, &dim)| {
                Array2::from_shape_fn((dim, cp_rank), |(i, r)| {
                    (((i * 13 + r * 7 + k * 5) % 23) as f64) / 23.0 - 0.5
                })
            })
            .collect()
    }

    fn rel_error(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let num: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        let den: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
        if den == 0.0 {
            num
        } else {
            num / den
        }
    }

    #[test]
    fn test_plan_respects_budget() {
        let spec = ChunkSpec::tile_size(&[8, 9, 10], &[4, 5, 5]).unwrap();
        let config = MttkrpStreamConfig::new().max_memory_bytes(64 * 1024);
        let plan = MttkrpStreamPlan::new(&spec, 3, &[0, 1, 2], &config).unwrap();

        assert!(plan.window_chunks >= 1);
        assert!(plan.peak_working_set_bytes() <= config.max_memory_bytes);
        assert_eq!(plan.total_chunks, spec.total_chunks());
    }

    #[test]
    fn test_plan_rejects_impossible_budget() {
        let spec = ChunkSpec::tile_size(&[8, 9, 10], &[8, 9, 10]).unwrap();
        // One chunk is the whole 720-element tensor = 5760 bytes; 1 KiB cannot hold it.
        let config = MttkrpStreamConfig::new().max_memory_bytes(1024);
        let err = MttkrpStreamPlan::new(&spec, 4, &[0], &config).unwrap_err();
        assert!(
            err.to_string().contains("too small"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_accumulator_applies_output_offset() {
        let shape = [4, 3];
        let mut acc = MttkrpAccumulator::new(&shape, 2, &[0]).unwrap();
        // A chunk whose mode-0 slice is global rows 2..4.
        let partial = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        acc.accumulate_chunk(&[2, 0], std::slice::from_ref(&partial))
            .unwrap();

        let m = acc.get(0).unwrap();
        assert_eq!(m[[0, 0]], 0.0);
        assert_eq!(m[[1, 0]], 0.0);
        assert_eq!(m[[2, 0]], 1.0);
        assert_eq!(m[[2, 1]], 2.0);
        assert_eq!(m[[3, 0]], 3.0);
        assert_eq!(m[[3, 1]], 4.0);
    }

    #[test]
    fn test_accumulator_rejects_overrun() {
        let mut acc = MttkrpAccumulator::new(&[4, 3], 2, &[0]).unwrap();
        let partial = Array2::<f64>::zeros((3, 2));
        // rows 2..5 does not fit in a dimension of 4.
        assert!(acc.accumulate_chunk(&[2, 0], &[partial]).is_err());
    }

    #[test]
    fn test_chunk_partials_use_sliced_factor_rows() {
        // The whole point: a chunk at global origin (1, 0) must be multiplied by A_0's
        // rows 1.., not rows 0.. . Construct a case where the two differ.
        let tensor = deterministic_tensor(&[3, 2]);
        let factors = deterministic_factors(&[3, 2], 2);
        let views: Vec<ArrayView2<f64>> = factors.iter().map(|f| f.view()).collect();

        let block = DenseND::from_vec(
            tensor.try_as_slice().unwrap()[2..6].to_vec(), // rows 1..3
            &[2, 2],
        )
        .unwrap();

        let partials = chunk_partials(&block, &views, &[1, 0], &[3, 2], &[1]).unwrap();

        // Reference: restrict the defining sum to the box by hand.
        let mut expected = Array2::<f64>::zeros((2, 2));
        let t = tensor.as_array();
        for i0 in 1..3 {
            for i1 in 0..2 {
                for r in 0..2 {
                    expected[[i1, r]] += t[IxDyn(&[i0, i1])] * factors[0][[i0, r]];
                }
            }
        }
        assert!(rel_error(&partials[0], &expected) < 1e-14);
    }

    #[test]
    fn test_streaming_matches_in_core_all_modes() {
        let shape = vec![5, 3, 4];
        let tensor = deterministic_tensor(&shape);
        let factors = deterministic_factors(&shape, 3);
        let views: Vec<ArrayView2<f64>> = factors.iter().map(|f| f.view()).collect();

        let source = DenseChunkSource::new(tensor.clone()).unwrap();
        // Ragged on every axis.
        let spec = ChunkSpec::tile_size(&shape, &[2, 2, 3]).unwrap();
        let config = MttkrpStreamConfig::new().max_memory_bytes(1 << 20);

        let (streamed, stats) = streaming_mttkrp_all_modes(&source, &spec, &views, config).unwrap();
        assert_eq!(stats.chunks_processed, spec.total_chunks());

        let x = tensor.as_array().view();
        for mode in 0..shape.len() {
            let reference = tenrso_kernels::mttkrp(&x, &views, mode).unwrap();
            assert!(
                rel_error(&streamed[mode], &reference) < 1e-13,
                "mode {} mismatch: {:e}",
                mode,
                rel_error(&streamed[mode], &reference)
            );
        }
    }

    #[test]
    fn test_window_size_does_not_change_result_bitwise() {
        let shape = vec![6, 5, 4];
        let tensor = deterministic_tensor(&shape);
        let factors = deterministic_factors(&shape, 2);
        let views: Vec<ArrayView2<f64>> = factors.iter().map(|f| f.view()).collect();
        let source = DenseChunkSource::new(tensor).unwrap();
        let spec = ChunkSpec::tile_size(&shape, &[4, 3, 3]).unwrap();

        let one_at_a_time = StreamingMttkrp::new(
            MttkrpStreamConfig::new()
                .max_memory_bytes(1 << 20)
                .max_window_chunks(1)
                .parallel(false),
        )
        .mttkrp_all_modes(&source, &spec, &views)
        .unwrap();

        let wide = StreamingMttkrp::new(
            MttkrpStreamConfig::new()
                .max_memory_bytes(1 << 20)
                .max_window_chunks(64)
                .parallel(true),
        )
        .mttkrp_all_modes(&source, &spec, &views)
        .unwrap();

        for mode in 0..shape.len() {
            assert_eq!(
                one_at_a_time[mode], wide[mode],
                "mode {} is not bit-identical across window sizes",
                mode
            );
        }
    }

    #[test]
    fn test_rejects_grid_mismatch_with_source_shape() {
        let shape = vec![4, 4, 4];
        let tensor = deterministic_tensor(&shape);
        let factors = deterministic_factors(&shape, 2);
        let views: Vec<ArrayView2<f64>> = factors.iter().map(|f| f.view()).collect();
        let source = DenseChunkSource::new(tensor).unwrap();

        let wrong = ChunkSpec::tile_size(&[4, 4, 5], &[2, 2, 2]).unwrap();
        let err = StreamingMttkrp::new(MttkrpStreamConfig::new())
            .mttkrp_all_modes(&source, &wrong, &views)
            .unwrap_err();
        assert!(err.to_string().contains("ChunkSpec describes shape"));
    }

    #[test]
    fn test_rejects_bad_factors_and_modes() {
        let shape = vec![4, 3];
        let tensor = deterministic_tensor(&shape);
        let source = DenseChunkSource::new(tensor).unwrap();
        let spec = ChunkSpec::tile_size(&shape, &[2, 2]).unwrap();

        // Wrong number of rows in factor 1.
        let bad = [Array2::<f64>::zeros((4, 2)), Array2::<f64>::zeros((5, 2))];
        let bad_views: Vec<ArrayView2<f64>> = bad.iter().map(|f| f.view()).collect();
        assert!(StreamingMttkrp::new(MttkrpStreamConfig::new())
            .mttkrp(&source, &spec, &bad_views, 0)
            .is_err());

        // Repeated mode.
        let good = deterministic_factors(&shape, 2);
        let good_views: Vec<ArrayView2<f64>> = good.iter().map(|f| f.view()).collect();
        assert!(StreamingMttkrp::new(MttkrpStreamConfig::new())
            .mttkrp_modes(&source, &spec, &good_views, &[0, 0])
            .is_err());

        // Out-of-bounds mode.
        assert!(StreamingMttkrp::new(MttkrpStreamConfig::new())
            .mttkrp(&source, &spec, &good_views, 2)
            .is_err());
    }

    #[test]
    fn test_single_element_chunks() {
        // The most extreme grid: every element is its own chunk. Exercises the degenerate
        // 1x1x1 sub-tensor path through the kernels.
        let shape = vec![3, 2, 2];
        let tensor = deterministic_tensor(&shape);
        let factors = deterministic_factors(&shape, 2);
        let views: Vec<ArrayView2<f64>> = factors.iter().map(|f| f.view()).collect();
        let source = DenseChunkSource::new(tensor.clone()).unwrap();
        let spec = ChunkSpec::tile_size(&shape, &[1, 1, 1]).unwrap();

        let streamed = StreamingMttkrp::new(MttkrpStreamConfig::new())
            .mttkrp_all_modes(&source, &spec, &views)
            .unwrap();

        let x = tensor.as_array().view();
        for mode in 0..shape.len() {
            let reference = tenrso_kernels::mttkrp(&x, &views, mode).unwrap();
            assert!(rel_error(&streamed[mode], &reference) < 1e-13);
        }
    }

    #[test]
    fn test_stats_peak_within_budget() {
        let shape = vec![10, 8, 6];
        let tensor = deterministic_tensor(&shape);
        let factors = deterministic_factors(&shape, 4);
        let views: Vec<ArrayView2<f64>> = factors.iter().map(|f| f.view()).collect();
        let source = DenseChunkSource::new(tensor).unwrap();
        let spec = ChunkSpec::tile_size(&shape, &[3, 3, 3]).unwrap();

        // Tensor is 480 * 8 = 3840 bytes; budget is deliberately below that.
        let budget = 3000;
        let config = MttkrpStreamConfig::new().max_memory_bytes(budget);
        let (_results, stats) = streaming_mttkrp_all_modes(&source, &spec, &views, config).unwrap();

        assert!(stats.plan.peak_working_set_bytes() <= budget);
        assert!(stats.peak_working_set_bytes() <= budget);
        assert!(stats.peak_chunk_bytes <= stats.plan.chunk_budget_bytes());
        assert_eq!(stats.chunks_processed, spec.total_chunks());
    }

    #[test]
    fn test_all_modes_matches_per_mode_calls() {
        let shape = vec![4, 5, 3, 2];
        let tensor = deterministic_tensor(&shape);
        let factors = deterministic_factors(&shape, 3);
        let views: Vec<ArrayView2<f64>> = factors.iter().map(|f| f.view()).collect();
        let source = DenseChunkSource::new(tensor).unwrap();
        let spec = ChunkSpec::tile_size(&shape, &[3, 2, 2, 1]).unwrap();
        let config = MttkrpStreamConfig::new();

        let mut exec = StreamingMttkrp::new(config.clone());
        let all = exec.mttkrp_all_modes(&source, &spec, &views).unwrap();

        for mode in 0..shape.len() {
            let single = StreamingMttkrp::new(config.clone())
                .mttkrp(&source, &spec, &views, mode)
                .unwrap();
            assert!(
                rel_error(&all[mode], &single) < 1e-13,
                "mode {}: one-pass all-modes disagrees with the single-mode pass",
                mode
            );
        }
    }

    #[test]
    fn test_zero_tensor_gives_zero_result() {
        let shape = vec![3, 4, 2];
        let tensor = DenseND::from_array(Array::zeros(IxDyn(&shape)));
        let factors = deterministic_factors(&shape, 2);
        let views: Vec<ArrayView2<f64>> = factors.iter().map(|f| f.view()).collect();
        let source = DenseChunkSource::new(tensor).unwrap();
        let spec = ChunkSpec::tile_size(&shape, &[2, 3, 1]).unwrap();

        let out = StreamingMttkrp::new(MttkrpStreamConfig::new())
            .mttkrp_all_modes(&source, &spec, &views)
            .unwrap();
        for m in &out {
            assert!(m.iter().all(|&v| v == 0.0));
        }
    }
}
