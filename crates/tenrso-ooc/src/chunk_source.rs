//! Chunk sources: uniform, random-access reads of dense sub-boxes of a tensor.
//!
//! A [`ChunkSource`] is the read side of out-of-core execution. It answers exactly one
//! question — *give me the dense sub-box `[start_k, end_k)` of the tensor* — and hides
//! whether that box comes from RAM, from a memory-mapped file, or from an Arrow IPC
//! chunk store on disk. Streaming kernels (see [`crate::mttkrp_stream`]) are written
//! once against this trait and then run unchanged over any backing store.
//!
//! # The one piece of index math that matters
//!
//! Every source hands back a **local** tensor whose axis `k` runs `0..(end_k - start_k)`.
//! The caller is responsible for mapping those local indices back to global ones. The
//! offset vector `start` is therefore returned to the caller alongside the data (it is
//! the caller's `start` argument), and *every* consumer must use it. Dropping the offset
//! is silent and catastrophic: the numbers still come out, they are just the wrong ones.
//!
//! # Layout contract
//!
//! All sources speak row-major (C-order): axis `N-1` is contiguous. Sub-box extraction
//! copies `∏_{k<N-1} c_k` contiguous runs of `c_{N-1}` elements each, where
//! `c_k = end_k - start_k`. That is the minimum possible number of `memcpy` calls for a
//! strided box in C order.

use anyhow::{anyhow, Result};
use tenrso_core::DenseND;

use crate::chunking::{ChunkIndex, ChunkSpec};

/// Row-major (C-order) strides for `shape`, in elements.
///
/// `strides[N-1] == 1`, `strides[k] == ∏_{j>k} shape[j]`.
fn c_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1usize; n];
    for k in (0..n.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

/// Validate a sub-box request against a tensor shape.
///
/// # Errors
///
/// Returns an error if the ranks disagree, the box is empty, or it leaves the tensor.
pub fn validate_subbox(shape: &[usize], start: &[usize], end: &[usize]) -> Result<()> {
    if shape.is_empty() {
        return Err(anyhow!("Sub-box extraction requires rank >= 1"));
    }
    if start.len() != shape.len() || end.len() != shape.len() {
        return Err(anyhow!(
            "Sub-box rank mismatch: tensor rank {}, start rank {}, end rank {}",
            shape.len(),
            start.len(),
            end.len()
        ));
    }
    for k in 0..shape.len() {
        if start[k] >= end[k] {
            return Err(anyhow!(
                "Empty sub-box on axis {}: start {} >= end {}",
                k,
                start[k],
                end[k]
            ));
        }
        if end[k] > shape[k] {
            return Err(anyhow!(
                "Sub-box out of bounds on axis {}: end {} > dim {}",
                k,
                end[k],
                shape[k]
            ));
        }
    }
    Ok(())
}

/// Copy the dense sub-box `[start_k, end_k)` out of a C-contiguous element buffer.
///
/// `src` must hold `∏ shape` elements in row-major order.
///
/// # Errors
///
/// Returns an error if the box is invalid (see [`validate_subbox`]) or `src` is the
/// wrong length.
///
/// # Complexity
///
/// `O(∏_k (end_k - start_k))` element copies, issued as `∏_{k<N-1} c_k` contiguous runs.
pub fn copy_subbox(
    src: &[f64],
    shape: &[usize],
    start: &[usize],
    end: &[usize],
) -> Result<Vec<f64>> {
    validate_subbox(shape, start, end)?;

    let expected: usize = shape.iter().product();
    if src.len() != expected {
        return Err(anyhow!(
            "Buffer length {} does not match shape {:?} ({} elements)",
            src.len(),
            shape,
            expected
        ));
    }

    let n = shape.len();
    let strides = c_strides(shape);
    let box_shape: Vec<usize> = start.iter().zip(end.iter()).map(|(s, e)| e - s).collect();
    let total: usize = box_shape.iter().product();

    // Axis N-1 is contiguous in `src`, so each run is a single memcpy.
    let run_len = box_shape[n - 1];
    let num_runs = total / run_len;

    let mut out = Vec::with_capacity(total);
    // Odometer over the leading (non-contiguous) axes, rightmost varying fastest.
    let mut coord = vec![0usize; n - 1];

    for _ in 0..num_runs {
        // Global offset of the first element of this run.
        let mut base = start[n - 1];
        for (k, &c) in coord.iter().enumerate() {
            base += (start[k] + c) * strides[k];
        }
        out.extend_from_slice(&src[base..base + run_len]);

        for k in (0..n - 1).rev() {
            coord[k] += 1;
            if coord[k] < box_shape[k] {
                break;
            }
            coord[k] = 0;
        }
    }

    debug_assert_eq!(out.len(), total);
    Ok(out)
}

/// A random-access provider of dense tensor sub-boxes.
///
/// Implementors must be `Send + Sync` so that a window of chunks can be fetched and
/// consumed in parallel. Reads take `&self`; any interior mutability (a file cursor,
/// say) must be synchronized by the implementor.
pub trait ChunkSource: Send + Sync {
    /// Shape of the full (global) tensor.
    fn shape(&self) -> &[usize];

    /// Read the dense sub-box `[start_k, end_k)` into a freshly-allocated tensor.
    ///
    /// The returned tensor has shape `[end_k - start_k]` and row-major layout. Its local
    /// index `j_k` corresponds to global index `start_k + j_k`.
    ///
    /// # Errors
    ///
    /// Returns an error if the box is invalid or the backing store cannot serve it.
    fn read_chunk(&self, start: &[usize], end: &[usize]) -> Result<DenseND<f64>>;

    /// The block grid the store was *materialized* with, if it has one.
    ///
    /// Sources that can serve an arbitrary sub-box (RAM, mmap) return `None`. Sources
    /// that store pre-cut blocks (an Arrow chunk store) return `Some(spec)`; a streaming
    /// kernel must then iterate that exact grid, and will refuse a mismatched one rather
    /// than silently re-cutting boxes it cannot serve.
    fn native_chunk_spec(&self) -> Option<&ChunkSpec> {
        None
    }

    /// Human-readable name of the backing store, for stats and diagnostics.
    fn source_name(&self) -> &'static str;

    /// Total number of elements in the full tensor.
    fn num_elements(&self) -> usize {
        self.shape().iter().product()
    }
}

/// A [`ChunkSource`] over a tensor that is already fully resident in RAM.
///
/// Useful as the reference implementation, for benchmarking streaming overhead against
/// the in-core kernel, and for callers whose tensor happens to fit but who still want
/// the bounded-working-set execution schedule.
#[derive(Debug, Clone)]
pub struct DenseChunkSource {
    tensor: DenseND<f64>,
    shape: Vec<usize>,
}

impl DenseChunkSource {
    /// Wrap an in-memory tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not C-contiguous (sub-box extraction relies on
    /// row-major strides).
    pub fn new(tensor: DenseND<f64>) -> Result<Self> {
        if tensor.try_as_slice().is_none() {
            return Err(anyhow!(
                "DenseChunkSource requires a C-contiguous tensor; got a non-standard layout"
            ));
        }
        let shape = tensor.shape().to_vec();
        Ok(Self { tensor, shape })
    }

    /// Borrow the underlying tensor.
    pub fn tensor(&self) -> &DenseND<f64> {
        &self.tensor
    }
}

impl ChunkSource for DenseChunkSource {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn read_chunk(&self, start: &[usize], end: &[usize]) -> Result<DenseND<f64>> {
        let src = self
            .tensor
            .try_as_slice()
            .ok_or_else(|| anyhow!("DenseChunkSource tensor is not contiguous"))?;
        let data = copy_subbox(src, &self.shape, start, end)?;
        let box_shape: Vec<usize> = start.iter().zip(end.iter()).map(|(s, e)| e - s).collect();
        DenseND::from_vec(data, &box_shape)
    }

    fn source_name(&self) -> &'static str {
        "dense-memory"
    }
}

/// A [`ChunkSource`] over a memory-mapped tensor file (TenRSo binary format).
///
/// This is the workhorse out-of-core source: the file is mapped once, and each
/// `read_chunk` copies only the requested box out of the mapping. Pages outside the
/// working set are never faulted in, and the ones that are can be evicted by the kernel
/// under pressure, so the *anonymous* (non-reclaimable) footprint of a full streaming
/// pass is the working set, not the tensor.
///
/// Create the file with [`crate::mmap_io::write_tensor_binary`].
#[cfg(feature = "mmap")]
#[derive(Debug)]
pub struct MmapChunkSource {
    mmap: crate::mmap_io::MmapTensor<f64>,
}

#[cfg(feature = "mmap")]
impl MmapChunkSource {
    /// Memory-map a tensor file written by [`crate::mmap_io::write_tensor_binary`].
    ///
    /// # Errors
    ///
    /// Returns an error if the file is missing or is not a valid TenRSo tensor file.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Ok(Self {
            mmap: crate::mmap_io::MmapTensor::<f64>::open(path)?,
        })
    }
}

#[cfg(feature = "mmap")]
impl ChunkSource for MmapChunkSource {
    fn shape(&self) -> &[usize] {
        self.mmap.shape()
    }

    fn read_chunk(&self, start: &[usize], end: &[usize]) -> Result<DenseND<f64>> {
        let data = copy_subbox(self.mmap.as_slice(), self.mmap.shape(), start, end)?;
        let box_shape: Vec<usize> = start.iter().zip(end.iter()).map(|(s, e)| e - s).collect();
        DenseND::from_vec(data, &box_shape)
    }

    fn source_name(&self) -> &'static str {
        "mmap-file"
    }
}

/// Sidecar metadata describing how a tensor was cut into an Arrow chunk store.
#[cfg(feature = "arrow")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ArrowStoreMeta {
    /// Shape of the full tensor.
    tensor_shape: Vec<usize>,
    /// Tile size the tensor was cut with.
    chunk_size: Vec<usize>,
}

/// A [`ChunkSource`] over an Arrow IPC file holding one record batch per chunk.
///
/// The tensor is cut with a [`ChunkSpec`] and each block is written as its own Arrow
/// record batch, in ascending linear chunk order (the same order
/// [`ChunkSpec::iter`] yields). The Arrow IPC *file* footer indexes every batch, so a
/// chunk is fetched by seeking straight to its batch — no scan, and only that batch is
/// ever decoded into memory.
///
/// The grid itself is persisted in a small JSON sidecar next to the data file, so
/// [`ArrowChunkStore::open`] recovers the store without the caller having to remember
/// how it was cut.
///
/// Because blocks are materialized, this source reports a
/// [`native_chunk_spec`](ChunkSource::native_chunk_spec): it can only serve boxes that
/// lie on its grid.
#[cfg(feature = "arrow")]
pub struct ArrowChunkStore {
    reader: std::sync::Mutex<crate::arrow_io::ArrowReader>,
    spec: ChunkSpec,
    shape: Vec<usize>,
}

#[cfg(feature = "arrow")]
impl std::fmt::Debug for ArrowChunkStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowChunkStore")
            .field("shape", &self.shape)
            .field("chunk_size", &self.spec.chunk_size())
            .field("num_chunks", &self.spec.total_chunks())
            .finish()
    }
}

#[cfg(feature = "arrow")]
impl ArrowChunkStore {
    /// Path of the JSON sidecar that accompanies `path`.
    fn meta_path(path: &std::path::Path) -> std::path::PathBuf {
        let mut s = path.as_os_str().to_os_string();
        s.push(".chunks.json");
        std::path::PathBuf::from(s)
    }

    /// Cut `tensor` with `spec` and write it as an Arrow chunk store at `path`.
    ///
    /// Writes `spec.total_chunks()` record batches in ascending linear chunk order plus a
    /// `<path>.chunks.json` sidecar. Peak extra memory is one chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if `spec` does not describe `tensor`'s shape, or on any I/O
    /// failure.
    pub fn create<P: AsRef<std::path::Path>>(
        path: P,
        tensor: &DenseND<f64>,
        spec: &ChunkSpec,
    ) -> Result<()> {
        if spec.tensor_shape() != tensor.shape() {
            return Err(anyhow!(
                "ChunkSpec shape {:?} does not match tensor shape {:?}",
                spec.tensor_shape(),
                tensor.shape()
            ));
        }
        let src = tensor
            .try_as_slice()
            .ok_or_else(|| anyhow!("ArrowChunkStore::create requires a C-contiguous tensor"))?;

        let path = path.as_ref();
        let mut writer = crate::arrow_io::ArrowWriter::new(path)?;

        // Ascending linear chunk order — the order `ChunkSpec::iter` yields and the order
        // batch indices are assigned in, so batch i == chunk i by construction.
        for chunk_idx in spec.iter() {
            let (start, end) = spec.chunk_bounds(&chunk_idx);
            let data = copy_subbox(src, spec.tensor_shape(), &start, &end)?;
            let block_shape = spec.chunk_shape(&chunk_idx);
            let block = DenseND::from_vec(data, &block_shape)?;
            writer.write(&block)?;
        }
        writer.finish()?;

        let meta = ArrowStoreMeta {
            tensor_shape: spec.tensor_shape().to_vec(),
            chunk_size: spec.chunk_size().to_vec(),
        };
        std::fs::write(Self::meta_path(path), serde_json::to_vec_pretty(&meta)?)?;

        Ok(())
    }

    /// Open an Arrow chunk store previously written by [`ArrowChunkStore::create`].
    ///
    /// # Errors
    ///
    /// Returns an error if the data file or its sidecar is missing or inconsistent (e.g.
    /// the batch count disagrees with the recorded grid).
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let meta_bytes = std::fs::read(Self::meta_path(path))?;
        let meta: ArrowStoreMeta = serde_json::from_slice(&meta_bytes)?;
        let spec = ChunkSpec::tile_size(&meta.tensor_shape, &meta.chunk_size)?;

        let reader = crate::arrow_io::ArrowReader::open(path)?;
        let batches = reader.num_batches();
        if batches != spec.total_chunks() {
            return Err(anyhow!(
                "Arrow chunk store is inconsistent: {} batches but grid has {} chunks",
                batches,
                spec.total_chunks()
            ));
        }

        let shape = meta.tensor_shape.clone();
        Ok(Self {
            reader: std::sync::Mutex::new(reader),
            spec,
            shape,
        })
    }

    /// The block grid this store was written with.
    pub fn spec(&self) -> &ChunkSpec {
        &self.spec
    }

    /// Map a sub-box onto the store's grid, erroring if it is not a stored block.
    fn chunk_index_of(&self, start: &[usize], end: &[usize]) -> Result<ChunkIndex> {
        validate_subbox(&self.shape, start, end)?;

        let cs = self.spec.chunk_size();
        let mut coords = Vec::with_capacity(self.shape.len());
        for k in 0..self.shape.len() {
            if !start[k].is_multiple_of(cs[k]) {
                return Err(anyhow!(
                    "Sub-box does not lie on the store's grid: start[{}] = {} is not a multiple of chunk size {}",
                    k,
                    start[k],
                    cs[k]
                ));
            }
            let coord = start[k] / cs[k];
            let expected_end = (start[k] + cs[k]).min(self.shape[k]);
            if end[k] != expected_end {
                return Err(anyhow!(
                    "Sub-box does not lie on the store's grid: end[{}] = {}, stored block ends at {}",
                    k,
                    end[k],
                    expected_end
                ));
            }
            coords.push(coord);
        }
        Ok(ChunkIndex::new(coords))
    }
}

#[cfg(feature = "arrow")]
impl ChunkSource for ArrowChunkStore {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn read_chunk(&self, start: &[usize], end: &[usize]) -> Result<DenseND<f64>> {
        let chunk_idx = self.chunk_index_of(start, end)?;
        let linear = chunk_idx.to_linear(self.spec.num_chunks());

        let mut reader = self
            .reader
            .lock()
            .map_err(|_| anyhow!("ArrowChunkStore reader mutex was poisoned"))?;
        let block = reader.read_batch(linear)?;

        let expected = self.spec.chunk_shape(&chunk_idx);
        if block.shape() != expected.as_slice() {
            return Err(anyhow!(
                "Arrow chunk store corruption: batch {} has shape {:?}, grid expects {:?}",
                linear,
                block.shape(),
                expected
            ));
        }
        Ok(block)
    }

    fn native_chunk_spec(&self) -> Option<&ChunkSpec> {
        Some(&self.spec)
    }

    fn source_name(&self) -> &'static str {
        "arrow-chunk-store"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference sub-box extraction: literal index arithmetic, no run optimization.
    fn naive_subbox(src: &[f64], shape: &[usize], start: &[usize], end: &[usize]) -> Vec<f64> {
        let strides = c_strides(shape);
        let box_shape: Vec<usize> = start.iter().zip(end.iter()).map(|(s, e)| e - s).collect();
        let total: usize = box_shape.iter().product();
        let n = shape.len();

        let mut out = Vec::with_capacity(total);
        let mut local = vec![0usize; n];
        for _ in 0..total {
            let mut off = 0usize;
            for k in 0..n {
                off += (start[k] + local[k]) * strides[k];
            }
            out.push(src[off]);
            for k in (0..n).rev() {
                local[k] += 1;
                if local[k] < box_shape[k] {
                    break;
                }
                local[k] = 0;
            }
        }
        out
    }

    #[test]
    fn test_c_strides() {
        assert_eq!(c_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(c_strides(&[5]), vec![1]);
        assert_eq!(c_strides(&[7, 1]), vec![1, 1]);
    }

    #[test]
    fn test_copy_subbox_matches_naive_3d() {
        let shape = vec![4, 5, 6];
        let n: usize = shape.iter().product();
        let src: Vec<f64> = (0..n).map(|v| v as f64).collect();

        for (start, end) in &[
            (vec![0, 0, 0], vec![4, 5, 6]),
            (vec![1, 2, 3], vec![3, 4, 5]),
            (vec![3, 4, 5], vec![4, 5, 6]), // ragged corner: 1x1x1
            (vec![2, 0, 1], vec![4, 5, 4]),
        ] {
            let fast = copy_subbox(&src, &shape, start, end).unwrap();
            let slow = naive_subbox(&src, &shape, start, end);
            assert_eq!(fast, slow, "mismatch for box {:?}..{:?}", start, end);
        }
    }

    #[test]
    fn test_copy_subbox_matches_naive_4d_asymmetric() {
        let shape = vec![3, 7, 2, 5];
        let n: usize = shape.iter().product();
        let src: Vec<f64> = (0..n).map(|v| (v as f64) * 0.5 - 3.0).collect();

        let start = vec![1, 3, 0, 2];
        let end = vec![3, 7, 2, 5];
        let fast = copy_subbox(&src, &shape, &start, &end).unwrap();
        let slow = naive_subbox(&src, &shape, &start, &end);
        assert_eq!(fast, slow);
    }

    #[test]
    fn test_copy_subbox_rank1() {
        let src: Vec<f64> = (0..10).map(|v| v as f64).collect();
        let got = copy_subbox(&src, &[10], &[3], &[7]).unwrap();
        assert_eq!(got, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_copy_subbox_errors() {
        let src = vec![0.0; 6];
        // Out of bounds.
        assert!(copy_subbox(&src, &[2, 3], &[0, 0], &[3, 3]).is_err());
        // Empty box.
        assert!(copy_subbox(&src, &[2, 3], &[1, 0], &[1, 3]).is_err());
        // Rank mismatch.
        assert!(copy_subbox(&src, &[2, 3], &[0], &[2]).is_err());
        // Wrong buffer length.
        assert!(copy_subbox(&src, &[2, 4], &[0, 0], &[2, 4]).is_err());
    }

    // `lin`/`k` are genuine mixed-radix indices into several parallel arrays here, not a
    // single-array walk `enumerate` could replace.
    #[allow(clippy::needless_range_loop)]
    #[test]
    fn test_dense_chunk_source_covers_tensor() {
        let shape = vec![5, 4, 3];
        let n: usize = shape.iter().product();
        let tensor = DenseND::from_vec((0..n).map(|v| v as f64).collect(), &shape).unwrap();
        let source = DenseChunkSource::new(tensor).unwrap();
        assert_eq!(source.shape(), shape.as_slice());
        assert_eq!(source.num_elements(), n);
        assert!(source.native_chunk_spec().is_none());

        // Reassemble the tensor from a ragged grid and compare to the original.
        let spec = ChunkSpec::tile_size(&shape, &[2, 3, 2]).unwrap();
        let mut seen = vec![f64::NAN; n];
        let strides = c_strides(&shape);
        for chunk_idx in spec.iter() {
            let (start, end) = spec.chunk_bounds(&chunk_idx);
            let block = source.read_chunk(&start, &end).unwrap();
            let bshape = spec.chunk_shape(&chunk_idx);
            assert_eq!(block.shape(), bshape.as_slice());

            let bstrides = c_strides(&bshape);
            let bdata = block.try_as_slice().unwrap();
            let btotal: usize = bshape.iter().product();
            for lin in 0..btotal {
                let mut off = 0usize;
                let mut rem = lin;
                for k in 0..shape.len() {
                    let c = rem / bstrides[k];
                    rem %= bstrides[k];
                    off += (start[k] + c) * strides[k];
                }
                seen[off] = bdata[lin];
            }
        }
        for (i, &v) in seen.iter().enumerate() {
            assert_eq!(v, i as f64, "element {} was not covered exactly once", i);
        }
    }
}
