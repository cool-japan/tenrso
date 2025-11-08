//! Tensor chunking strategies
//!
//! This module provides deterministic chunking for large tensors, enabling
//! out-of-core processing by breaking tensors into manageable pieces.
//!
//! # Features
//!
//! - Deterministic chunk specifications
//! - Tile-based and dimension-based chunking
//! - Chunk iteration with predictable ordering
//! - Chunk index algebra for coordinate transformations
//!
//! # Example
//!
//! ```ignore
//! use tenrso_ooc::chunking::{ChunkSpec, ChunkIterator};
//!
//! // Define chunking: split [1000, 1000] tensor into [100, 100] tiles
//! let spec = ChunkSpec::tile_size(&[1000, 1000], &[100, 100])?;
//!
//! // Iterate over all chunks
//! for chunk_idx in spec.iter() {
//!     let (start, end) = spec.chunk_bounds(&chunk_idx);
//!     // Process chunk at start..end
//! }
//! ```

use anyhow::{anyhow, Result};

/// Chunk specification defining how to split a tensor
///
/// Provides deterministic chunking strategy for out-of-core processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpec {
    /// Shape of the full tensor
    tensor_shape: Vec<usize>,
    /// Size of each chunk (tile size per dimension)
    chunk_size: Vec<usize>,
    /// Number of chunks per dimension
    num_chunks: Vec<usize>,
}

impl ChunkSpec {
    /// Create a chunk specification with fixed tile size
    ///
    /// # Arguments
    ///
    /// * `tensor_shape` - Shape of the full tensor
    /// * `chunk_size` - Desired size of each chunk
    ///
    /// # Returns
    ///
    /// A new `ChunkSpec` instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Shapes have different ranks
    /// - Any dimension has zero size
    pub fn tile_size(tensor_shape: &[usize], chunk_size: &[usize]) -> Result<Self> {
        if tensor_shape.len() != chunk_size.len() {
            return Err(anyhow!(
                "Shape rank mismatch: tensor rank {}, chunk rank {}",
                tensor_shape.len(),
                chunk_size.len()
            ));
        }

        if tensor_shape.contains(&0) {
            return Err(anyhow!("Tensor shape cannot have zero dimensions"));
        }

        if chunk_size.contains(&0) {
            return Err(anyhow!("Chunk size cannot have zero dimensions"));
        }

        // Calculate number of chunks per dimension (ceiling division)
        let num_chunks: Vec<usize> = tensor_shape
            .iter()
            .zip(chunk_size.iter())
            .map(|(&t, &c)| t.div_ceil(c))
            .collect();

        Ok(Self {
            tensor_shape: tensor_shape.to_vec(),
            chunk_size: chunk_size.to_vec(),
            num_chunks,
        })
    }

    /// Create a chunk specification with fixed number of chunks
    ///
    /// # Arguments
    ///
    /// * `tensor_shape` - Shape of the full tensor
    /// * `num_chunks` - Desired number of chunks per dimension
    ///
    /// # Returns
    ///
    /// A new `ChunkSpec` instance
    ///
    /// # Errors
    ///
    /// Returns an error if shapes have different ranks
    pub fn with_num_chunks(tensor_shape: &[usize], num_chunks: &[usize]) -> Result<Self> {
        if tensor_shape.len() != num_chunks.len() {
            return Err(anyhow!(
                "Shape rank mismatch: tensor rank {}, num_chunks rank {}",
                tensor_shape.len(),
                num_chunks.len()
            ));
        }

        if num_chunks.contains(&0) {
            return Err(anyhow!("Number of chunks cannot be zero"));
        }

        // Calculate chunk size per dimension (ceiling division)
        let chunk_size: Vec<usize> = tensor_shape
            .iter()
            .zip(num_chunks.iter())
            .map(|(&t, &n)| t.div_ceil(n))
            .collect();

        Ok(Self {
            tensor_shape: tensor_shape.to_vec(),
            chunk_size,
            num_chunks: num_chunks.to_vec(),
        })
    }

    /// Get the shape of the full tensor
    pub fn tensor_shape(&self) -> &[usize] {
        &self.tensor_shape
    }

    /// Get the chunk size (tile size)
    pub fn chunk_size(&self) -> &[usize] {
        &self.chunk_size
    }

    /// Get the number of chunks per dimension
    pub fn num_chunks(&self) -> &[usize] {
        &self.num_chunks
    }

    /// Get the total number of chunks
    pub fn total_chunks(&self) -> usize {
        self.num_chunks.iter().product()
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.tensor_shape.len()
    }

    /// Get the bounds (start, end) for a specific chunk
    ///
    /// # Arguments
    ///
    /// * `chunk_idx` - The chunk index
    ///
    /// # Returns
    ///
    /// Tuple of (start, end) coordinates for each dimension
    pub fn chunk_bounds(&self, chunk_idx: &ChunkIndex) -> (Vec<usize>, Vec<usize>) {
        assert_eq!(chunk_idx.coords.len(), self.rank());

        let mut start = Vec::with_capacity(self.rank());
        let mut end = Vec::with_capacity(self.rank());

        for i in 0..self.rank() {
            let chunk_coord = chunk_idx.coords[i];
            let s = chunk_coord * self.chunk_size[i];
            let e = (s + self.chunk_size[i]).min(self.tensor_shape[i]);
            start.push(s);
            end.push(e);
        }

        (start, end)
    }

    /// Get the shape of a specific chunk
    ///
    /// Note: Edge chunks may be smaller than chunk_size
    pub fn chunk_shape(&self, chunk_idx: &ChunkIndex) -> Vec<usize> {
        let (start, end) = self.chunk_bounds(chunk_idx);
        start.iter().zip(end.iter()).map(|(s, e)| e - s).collect()
    }

    /// Check if a chunk index is valid
    pub fn is_valid_chunk(&self, chunk_idx: &ChunkIndex) -> bool {
        if chunk_idx.coords.len() != self.rank() {
            return false;
        }

        chunk_idx
            .coords
            .iter()
            .zip(self.num_chunks.iter())
            .all(|(&coord, &num)| coord < num)
    }

    /// Create a chunk iterator
    pub fn iter(&self) -> ChunkIterator<'_> {
        ChunkIterator::new(self)
    }
}

/// Index representing a specific chunk's position
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChunkIndex {
    /// Coordinates in chunk space (not element space)
    pub coords: Vec<usize>,
}

impl ChunkIndex {
    /// Create a new chunk index
    pub fn new(coords: Vec<usize>) -> Self {
        Self { coords }
    }

    /// Create a chunk index from a linear index
    ///
    /// # Arguments
    ///
    /// * `linear_idx` - Linear index into chunk array
    /// * `num_chunks` - Number of chunks per dimension
    ///
    /// # Returns
    ///
    /// The corresponding chunk index
    pub fn from_linear(linear_idx: usize, num_chunks: &[usize]) -> Self {
        let mut coords = Vec::with_capacity(num_chunks.len());
        let mut remaining = linear_idx;

        // Row-major order: rightmost index varies fastest
        for i in 0..num_chunks.len() {
            let stride: usize = num_chunks[i + 1..].iter().product();
            coords.push(remaining / stride);
            remaining %= stride;
        }

        Self { coords }
    }

    /// Convert chunk index to linear index
    ///
    /// # Arguments
    ///
    /// * `num_chunks` - Number of chunks per dimension
    ///
    /// # Returns
    ///
    /// Linear index in row-major order
    pub fn to_linear(&self, num_chunks: &[usize]) -> usize {
        assert_eq!(self.coords.len(), num_chunks.len());

        let mut linear = 0;
        let mut stride = 1;

        for i in (0..self.coords.len()).rev() {
            linear += self.coords[i] * stride;
            stride *= num_chunks[i];
        }

        linear
    }
}

/// Iterator over chunks in deterministic order
pub struct ChunkIterator<'a> {
    spec: &'a ChunkSpec,
    current_linear: usize,
    total_chunks: usize,
}

impl<'a> ChunkIterator<'a> {
    /// Create a new chunk iterator
    pub fn new(spec: &'a ChunkSpec) -> Self {
        Self {
            spec,
            current_linear: 0,
            total_chunks: spec.total_chunks(),
        }
    }
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = ChunkIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_linear >= self.total_chunks {
            return None;
        }

        let chunk_idx = ChunkIndex::from_linear(self.current_linear, self.spec.num_chunks());
        self.current_linear += 1;

        Some(chunk_idx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_chunks - self.current_linear;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for ChunkIterator<'a> {
    fn len(&self) -> usize {
        self.total_chunks - self.current_linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_spec_tile_size() {
        let spec = ChunkSpec::tile_size(&[100, 100], &[25, 25]).unwrap();

        assert_eq!(spec.tensor_shape(), &[100, 100]);
        assert_eq!(spec.chunk_size(), &[25, 25]);
        assert_eq!(spec.num_chunks(), &[4, 4]);
        assert_eq!(spec.total_chunks(), 16);
        assert_eq!(spec.rank(), 2);
    }

    #[test]
    fn test_chunk_spec_num_chunks() {
        let spec = ChunkSpec::with_num_chunks(&[100, 100], &[4, 4]).unwrap();

        assert_eq!(spec.tensor_shape(), &[100, 100]);
        assert_eq!(spec.chunk_size(), &[25, 25]);
        assert_eq!(spec.num_chunks(), &[4, 4]);
        assert_eq!(spec.total_chunks(), 16);
    }

    #[test]
    fn test_chunk_spec_uneven_division() {
        let spec = ChunkSpec::tile_size(&[100, 100], &[30, 30]).unwrap();

        assert_eq!(spec.num_chunks(), &[4, 4]); // Ceiling division
        assert_eq!(spec.total_chunks(), 16);

        // Last chunk in each dimension should be smaller
        let last_chunk = ChunkIndex::new(vec![3, 3]);
        let shape = spec.chunk_shape(&last_chunk);
        assert_eq!(shape, vec![10, 10]); // 100 - 3*30 = 10
    }

    #[test]
    fn test_chunk_bounds() {
        let spec = ChunkSpec::tile_size(&[100, 100], &[25, 25]).unwrap();

        // First chunk
        let chunk_idx = ChunkIndex::new(vec![0, 0]);
        let (start, end) = spec.chunk_bounds(&chunk_idx);
        assert_eq!(start, vec![0, 0]);
        assert_eq!(end, vec![25, 25]);

        // Middle chunk
        let chunk_idx = ChunkIndex::new(vec![1, 2]);
        let (start, end) = spec.chunk_bounds(&chunk_idx);
        assert_eq!(start, vec![25, 50]);
        assert_eq!(end, vec![50, 75]);

        // Last chunk
        let chunk_idx = ChunkIndex::new(vec![3, 3]);
        let (start, end) = spec.chunk_bounds(&chunk_idx);
        assert_eq!(start, vec![75, 75]);
        assert_eq!(end, vec![100, 100]);
    }

    #[test]
    fn test_chunk_shape() {
        let spec = ChunkSpec::tile_size(&[100, 100], &[30, 30]).unwrap();

        // Regular chunk
        let chunk_idx = ChunkIndex::new(vec![0, 0]);
        assert_eq!(spec.chunk_shape(&chunk_idx), vec![30, 30]);

        // Edge chunk
        let chunk_idx = ChunkIndex::new(vec![3, 3]);
        assert_eq!(spec.chunk_shape(&chunk_idx), vec![10, 10]);
    }

    #[test]
    fn test_chunk_index_linear_conversion() {
        let num_chunks = vec![4, 4];

        // Test round-trip conversion
        for linear in 0..16 {
            let chunk_idx = ChunkIndex::from_linear(linear, &num_chunks);
            let back = chunk_idx.to_linear(&num_chunks);
            assert_eq!(back, linear);
        }

        // Test specific cases
        let chunk_idx = ChunkIndex::from_linear(0, &num_chunks);
        assert_eq!(chunk_idx.coords, vec![0, 0]);

        let chunk_idx = ChunkIndex::from_linear(5, &num_chunks);
        assert_eq!(chunk_idx.coords, vec![1, 1]);

        let chunk_idx = ChunkIndex::from_linear(15, &num_chunks);
        assert_eq!(chunk_idx.coords, vec![3, 3]);
    }

    #[test]
    fn test_chunk_iterator() {
        let spec = ChunkSpec::tile_size(&[100, 100], &[50, 50]).unwrap();

        let chunks: Vec<ChunkIndex> = spec.iter().collect();

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].coords, vec![0, 0]);
        assert_eq!(chunks[1].coords, vec![0, 1]);
        assert_eq!(chunks[2].coords, vec![1, 0]);
        assert_eq!(chunks[3].coords, vec![1, 1]);
    }

    #[test]
    fn test_chunk_iterator_3d() {
        let spec = ChunkSpec::tile_size(&[10, 10, 10], &[5, 5, 5]).unwrap();

        let chunks: Vec<ChunkIndex> = spec.iter().collect();

        assert_eq!(chunks.len(), 8);
        assert_eq!(spec.total_chunks(), 8);

        // Verify deterministic ordering
        assert_eq!(chunks[0].coords, vec![0, 0, 0]);
        assert_eq!(chunks[7].coords, vec![1, 1, 1]);
    }

    #[test]
    fn test_is_valid_chunk() {
        let spec = ChunkSpec::tile_size(&[100, 100], &[25, 25]).unwrap();

        assert!(spec.is_valid_chunk(&ChunkIndex::new(vec![0, 0])));
        assert!(spec.is_valid_chunk(&ChunkIndex::new(vec![3, 3])));
        assert!(!spec.is_valid_chunk(&ChunkIndex::new(vec![4, 0])));
        assert!(!spec.is_valid_chunk(&ChunkIndex::new(vec![0, 4])));
        assert!(!spec.is_valid_chunk(&ChunkIndex::new(vec![0, 0, 0]))); // Wrong rank
    }

    #[test]
    fn test_chunk_spec_errors() {
        // Rank mismatch
        assert!(ChunkSpec::tile_size(&[100, 100], &[25, 25, 25]).is_err());

        // Zero dimensions
        assert!(ChunkSpec::tile_size(&[0, 100], &[25, 25]).is_err());
        assert!(ChunkSpec::tile_size(&[100, 100], &[0, 25]).is_err());
    }

    #[test]
    fn test_iterator_size_hint() {
        let spec = ChunkSpec::tile_size(&[100, 100], &[50, 50]).unwrap();
        let mut iter = spec.iter();

        assert_eq!(iter.len(), 4);
        assert_eq!(iter.size_hint(), (4, Some(4)));

        iter.next();
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }
}
