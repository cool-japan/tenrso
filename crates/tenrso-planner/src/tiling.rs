//! Tiling and blocking strategies
//!
//! Provides cache-aware tiling for tensor operations to improve memory locality
//! and enable out-of-core execution.

use anyhow::{anyhow, Result};

/// Cache configuration for tiling decisions
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// L1 cache size in bytes (default: 32 KB)
    pub l1_size: usize,

    /// L2 cache size in bytes (default: 256 KB)
    pub l2_size: usize,

    /// L3 cache size in bytes (default: 8 MB)
    pub l3_size: usize,

    /// Cache line size in bytes (default: 64 bytes)
    pub cache_line_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 32 * 1024,       // 32 KB
            l2_size: 256 * 1024,      // 256 KB
            l3_size: 8 * 1024 * 1024, // 8 MB
            cache_line_size: 64,
        }
    }
}

/// Tiling strategy specification
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileSpec {
    /// Tile sizes for each dimension
    pub sizes: Vec<usize>,

    /// Total number of tiles
    pub num_tiles: usize,

    /// Memory per tile (bytes)
    pub memory_per_tile: usize,
}

impl TileSpec {
    /// Create a new tile specification
    pub fn new(sizes: Vec<usize>, bytes_per_element: usize) -> Self {
        let memory_per_tile = sizes.iter().product::<usize>() * bytes_per_element;
        let num_tiles = 1; // Computed by caller

        Self {
            sizes,
            num_tiles,
            memory_per_tile,
        }
    }

    /// Check if tiles fit in a given cache level
    pub fn fits_in_cache(&self, cache_size: usize) -> bool {
        self.memory_per_tile <= cache_size
    }
}

/// Compute optimal tile sizes for a tensor operation
///
/// # Arguments
///
/// * `shape` - Shape of the tensor to tile
/// * `bytes_per_element` - Size of each element (e.g., 8 for f64)
/// * `config` - Cache configuration
/// * `target_cache` - Target cache level ("L1", "L2", or "L3")
///
/// # Returns
///
/// Tile specification with optimal sizes
///
/// # Algorithm
///
/// 1. Determine target cache size based on target_cache
/// 2. Compute total tensor memory
/// 3. If tensor fits in cache, no tiling needed
/// 4. Otherwise, compute tile sizes to fit in target cache
/// 5. Prefer tiling larger dimensions first for better locality
pub fn compute_tile_sizes(
    shape: &[usize],
    bytes_per_element: usize,
    config: &CacheConfig,
    target_cache: &str,
) -> Result<TileSpec> {
    if shape.is_empty() {
        return Err(anyhow!("Cannot tile empty shape"));
    }

    let target_size = match target_cache {
        "L1" => config.l1_size,
        "L2" => config.l2_size,
        "L3" => config.l3_size,
        _ => return Err(anyhow!("Invalid target cache: {}", target_cache)),
    };

    let total_memory = shape.iter().product::<usize>() * bytes_per_element;

    // If entire tensor fits in cache, no tiling needed
    if total_memory <= target_size {
        return Ok(TileSpec::new(shape.to_vec(), bytes_per_element));
    }

    // Compute tile sizes to fit in target cache
    let max_elements = target_size / bytes_per_element;

    // Strategy: Keep smaller dimensions whole, tile larger dimensions
    let mut tile_sizes = shape.to_vec();
    let mut current_size: usize = tile_sizes.iter().product();

    // Find dimensions to tile (largest first)
    let mut dims_with_sizes: Vec<(usize, usize)> = tile_sizes.iter().copied().enumerate().collect();
    dims_with_sizes.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

    for (dim_idx, _) in dims_with_sizes {
        if current_size <= max_elements {
            break;
        }

        // Reduce this dimension to fit
        let reduction_needed = (current_size as f64 / max_elements as f64).ceil() as usize;
        let new_size = (tile_sizes[dim_idx] as f64 / reduction_needed as f64).ceil() as usize;

        current_size = current_size / tile_sizes[dim_idx] * new_size;
        tile_sizes[dim_idx] = new_size.max(1);
    }

    Ok(TileSpec::new(tile_sizes, bytes_per_element))
}

/// Compute tile sizes based on memory budget
///
/// # Arguments
///
/// * `shape` - Shape of the tensor to tile
/// * `bytes_per_element` - Size of each element
/// * `memory_budget` - Maximum memory budget in bytes
///
/// # Returns
///
/// Tile specification that fits in memory budget
pub fn compute_tile_sizes_budget(
    shape: &[usize],
    bytes_per_element: usize,
    memory_budget: usize,
) -> Result<TileSpec> {
    if shape.is_empty() {
        return Err(anyhow!("Cannot tile empty shape"));
    }

    let total_memory = shape.iter().product::<usize>() * bytes_per_element;

    // If entire tensor fits in budget, no tiling needed
    if total_memory <= memory_budget {
        return Ok(TileSpec::new(shape.to_vec(), bytes_per_element));
    }

    let max_elements = memory_budget / bytes_per_element;
    if max_elements == 0 {
        return Err(anyhow!("Memory budget too small for even one element"));
    }

    // Compute tile sizes
    let mut tile_sizes = shape.to_vec();
    let mut current_size: usize = tile_sizes.iter().product();

    // Tile largest dimensions first
    let mut dims_with_sizes: Vec<(usize, usize)> = tile_sizes.iter().copied().enumerate().collect();
    dims_with_sizes.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

    for (dim_idx, _) in dims_with_sizes {
        if current_size <= max_elements {
            break;
        }

        let reduction_needed = (current_size as f64 / max_elements as f64).ceil() as usize;
        let new_size = (tile_sizes[dim_idx] as f64 / reduction_needed as f64).ceil() as usize;

        current_size = current_size / tile_sizes[dim_idx] * new_size;
        tile_sizes[dim_idx] = new_size.max(1);
    }

    Ok(TileSpec::new(tile_sizes, bytes_per_element))
}

/// Compute optimal block size for matrix multiplication
///
/// For C = A × B where A is m×k and B is k×n:
/// - Block size should fit 2 blocks of A and B in cache
/// - Minimize cache misses by blocking for temporal locality
///
/// # Arguments
///
/// * `m` - Number of rows in A
/// * `k` - Shared dimension
/// * `n` - Number of columns in B
/// * `bytes_per_element` - Size of each element
/// * `config` - Cache configuration
///
/// # Returns
///
/// (block_m, block_k, block_n) - Block sizes for m, k, n dimensions
pub fn compute_matmul_block_sizes(
    m: usize,
    k: usize,
    n: usize,
    bytes_per_element: usize,
    config: &CacheConfig,
) -> (usize, usize, usize) {
    // Target L2 cache (allows for A, B, C blocks)
    let target_size = config.l2_size / 3; // Divide by 3 for A, B, C

    // Compute maximum block size
    // Memory for blocks: block_m * block_k + block_k * block_n + block_m * block_n
    // Simplification: assume square blocks (block_m ≈ block_k ≈ block_n ≈ B)
    // Memory: 3 * B^2 * bytes_per_element ≤ target_size
    let max_block_elems = target_size / (3 * bytes_per_element);
    let max_block = (max_block_elems as f64).sqrt().floor() as usize;

    // Ensure block size is reasonable
    let block_m = m.min(max_block.max(32));
    let block_k = k.min(max_block.max(32));
    let block_n = n.min(max_block.max(32));

    (block_m, block_k, block_n)
}

/// Estimate number of cache misses for a tile configuration
///
/// This is a simplified model assuming LRU cache replacement
///
/// # Arguments
///
/// * `tile_spec` - Tile specification
/// * `access_pattern` - "sequential" or "random"
/// * `config` - Cache configuration
///
/// # Returns
///
/// Estimated number of cache misses per tile
pub fn estimate_cache_misses(
    tile_spec: &TileSpec,
    access_pattern: &str,
    config: &CacheConfig,
) -> f64 {
    let cache_lines_needed =
        (tile_spec.memory_per_tile as f64 / config.cache_line_size as f64).ceil();

    match access_pattern {
        "sequential" => {
            // Sequential access: mostly hits after initial cold misses
            cache_lines_needed
        }
        "random" => {
            // Random access: mostly misses
            cache_lines_needed * 0.9
        }
        _ => cache_lines_needed * 0.5, // Unknown pattern
    }
}

// ============================================================================
// Multi-Level (Nested) Tiling
// ============================================================================

/// Multi-level tile specification
///
/// Represents a hierarchical tiling strategy with tiles for each cache level.
/// Enables optimal cache utilization across L1/L2/L3 hierarchy.
///
/// # Example
///
/// For a 1024×1024 matrix:
/// - L1 tiles: 32×32 (fits in L1 cache)
/// - L2 tiles: 128×128 (composed of 4×4 L1 tiles)
/// - L3 tiles: 512×512 (composed of 4×4 L2 tiles)
#[derive(Debug, Clone)]
pub struct MultiLevelTileSpec {
    /// L1 cache tile sizes
    pub l1_tiles: Vec<usize>,

    /// L2 cache tile sizes
    pub l2_tiles: Vec<usize>,

    /// L3 cache tile sizes
    pub l3_tiles: Vec<usize>,

    /// Tensor shape
    pub shape: Vec<usize>,

    /// Bytes per element
    pub bytes_per_element: usize,
}

impl MultiLevelTileSpec {
    /// Compute multi-level tiling for a tensor
    ///
    /// # Algorithm
    ///
    /// 1. Compute L1 tiles (smallest, innermost)
    /// 2. Compute L2 tiles as multiples of L1 tiles
    /// 3. Compute L3 tiles as multiples of L2 tiles
    /// 4. Ensure each level fits in respective cache
    ///
    /// # Returns
    ///
    /// Multi-level tile specification with hierarchical sizes
    pub fn compute(
        shape: &[usize],
        bytes_per_element: usize,
        config: &CacheConfig,
    ) -> Result<Self> {
        if shape.is_empty() {
            return Err(anyhow!("Cannot tile empty shape"));
        }

        // Compute L1 tiles (innermost, smallest)
        let l1_tiles = compute_tile_sizes_for_cache(shape, bytes_per_element, config.l1_size)?;

        // Compute L2 tiles (medium, multiples of L1)
        let l2_tiles = compute_tile_sizes_for_cache(shape, bytes_per_element, config.l2_size)?;

        // Ensure L2 >= L1
        let l2_tiles: Vec<usize> = l2_tiles
            .into_iter()
            .zip(l1_tiles.iter())
            .map(|(l2, &l1)| l2.max(l1))
            .collect();

        // Compute L3 tiles (outermost, largest)
        let l3_tiles = compute_tile_sizes_for_cache(shape, bytes_per_element, config.l3_size)?;

        // Ensure L3 >= L2 >= L1
        let l3_tiles: Vec<usize> = l3_tiles
            .into_iter()
            .zip(l2_tiles.iter())
            .map(|(l3, &l2)| l3.max(l2))
            .collect();

        Ok(Self {
            l1_tiles,
            l2_tiles,
            l3_tiles,
            shape: shape.to_vec(),
            bytes_per_element,
        })
    }

    /// Get the number of L1 tiles within an L2 tile for dimension i
    pub fn l1_tiles_per_l2(&self, dim: usize) -> usize {
        if dim >= self.l1_tiles.len() {
            return 1;
        }
        self.l2_tiles[dim].div_ceil(self.l1_tiles[dim])
    }

    /// Get the number of L2 tiles within an L3 tile for dimension i
    pub fn l2_tiles_per_l3(&self, dim: usize) -> usize {
        if dim >= self.l2_tiles.len() {
            return 1;
        }
        self.l3_tiles[dim].div_ceil(self.l2_tiles[dim])
    }

    /// Total number of L1 tiles across entire tensor
    pub fn total_l1_tiles(&self) -> usize {
        self.shape
            .iter()
            .zip(self.l1_tiles.iter())
            .map(|(&s, &t)| s.div_ceil(t))
            .product()
    }

    /// Total number of L2 tiles across entire tensor
    pub fn total_l2_tiles(&self) -> usize {
        self.shape
            .iter()
            .zip(self.l2_tiles.iter())
            .map(|(&s, &t)| s.div_ceil(t))
            .product()
    }

    /// Total number of L3 tiles across entire tensor
    pub fn total_l3_tiles(&self) -> usize {
        self.shape
            .iter()
            .zip(self.l3_tiles.iter())
            .map(|(&s, &t)| s.div_ceil(t))
            .product()
    }
}

/// Helper: Compute tile sizes for a specific cache size
fn compute_tile_sizes_for_cache(
    shape: &[usize],
    bytes_per_element: usize,
    cache_size: usize,
) -> Result<Vec<usize>> {
    let total_bytes = shape.iter().product::<usize>() * bytes_per_element;

    // If fits in cache, no tiling needed
    if total_bytes <= cache_size {
        return Ok(shape.to_vec());
    }

    // Compute tile sizes
    let mut tile_sizes = shape.to_vec();
    let target_elems = cache_size / bytes_per_element;

    // Scale down uniformly to fit in cache
    let scale_factor = (target_elems as f64 / shape.iter().product::<usize>() as f64)
        .powf(1.0 / shape.len() as f64);

    for (i, &dim) in shape.iter().enumerate() {
        let scaled = (dim as f64 * scale_factor).floor() as usize;
        tile_sizes[i] = scaled.max(1).min(dim);
    }

    // Verify total size
    let tile_bytes = tile_sizes.iter().product::<usize>() * bytes_per_element;
    if tile_bytes > cache_size {
        // Further reduce if still too large
        for size in &mut tile_sizes {
            *size = (*size / 2).max(1);
        }
    }

    Ok(tile_sizes)
}

/// Type alias for multi-level matmul block sizes
///
/// Format: ((l1_m, l1_k, l1_n), (l2_m, l2_k, l2_n), (l3_m, l3_k, l3_n))
pub type MatmulMultilevelBlocks = (
    (usize, usize, usize),
    (usize, usize, usize),
    (usize, usize, usize),
);

/// Compute multi-level blocking for matrix multiplication
///
/// Optimizes for 3-level cache hierarchy with nested blocking.
///
/// # Algorithm
///
/// 1. L1 blocking: micro-kernels (e.g., 8×8 or 16×16)
/// 2. L2 blocking: panel operations (e.g., 64×64 or 128×128)
/// 3. L3 blocking: outer blocks (e.g., 256×256 or 512×512)
///
/// # Returns
///
/// ((l1_m, l1_k, l1_n), (l2_m, l2_k, l2_n), (l3_m, l3_k, l3_n))
pub fn compute_matmul_multilevel_blocks(
    m: usize,
    k: usize,
    n: usize,
    bytes_per_element: usize,
    config: &CacheConfig,
) -> MatmulMultilevelBlocks {
    // L1 blocks (micro-kernels)
    let l1_target = config.l1_size / (3 * bytes_per_element);
    let l1_block = ((l1_target as f64).cbrt().floor() as usize).clamp(4, 32);
    let l1_m = m.min(l1_block);
    let l1_k = k.min(l1_block);
    let l1_n = n.min(l1_block);

    // L2 blocks (panels)
    let l2_target = config.l2_size / (3 * bytes_per_element);
    let l2_block = ((l2_target as f64).cbrt().floor() as usize).clamp(32, 256);
    let l2_m = m.min(l2_block).max(l1_m);
    let l2_k = k.min(l2_block).max(l1_k);
    let l2_n = n.min(l2_block).max(l1_n);

    // L3 blocks (outer blocks)
    let l3_target = config.l3_size / (3 * bytes_per_element);
    let l3_block = ((l3_target as f64).cbrt().floor() as usize).clamp(128, 1024);
    let l3_m = m.min(l3_block).max(l2_m);
    let l3_k = k.min(l3_block).max(l2_k);
    let l3_n = n.min(l3_block).max(l2_n);

    ((l1_m, l1_k, l1_n), (l2_m, l2_k, l2_n), (l3_m, l3_k, l3_n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();

        assert_eq!(config.l1_size, 32 * 1024);
        assert_eq!(config.l2_size, 256 * 1024);
        assert_eq!(config.l3_size, 8 * 1024 * 1024);
        assert_eq!(config.cache_line_size, 64);
    }

    #[test]
    fn test_compute_tile_sizes_no_tiling() {
        // Small tensor that fits in L1
        let shape = vec![10, 10];
        let config = CacheConfig::default();

        let tile = compute_tile_sizes(&shape, 8, &config, "L1").unwrap();

        // Should not need tiling
        assert_eq!(tile.sizes, vec![10, 10]);
        assert!(tile.fits_in_cache(config.l1_size));
    }

    #[test]
    fn test_compute_tile_sizes_with_tiling() {
        // Large tensor that needs tiling
        let shape = vec![1000, 1000];
        let config = CacheConfig::default();

        let tile = compute_tile_sizes(&shape, 8, &config, "L2").unwrap();

        // Should be tiled to fit in L2
        assert!(tile.sizes[0] < 1000 || tile.sizes[1] < 1000);
        assert!(tile.fits_in_cache(config.l2_size));
    }

    #[test]
    fn test_compute_tile_sizes_3d() {
        // 3D tensor
        let shape = vec![100, 100, 100];
        let config = CacheConfig::default();

        let tile = compute_tile_sizes(&shape, 8, &config, "L3").unwrap();

        // Should be tiled to fit in L3
        assert!(tile.fits_in_cache(config.l3_size));
    }

    #[test]
    fn test_compute_tile_sizes_budget() {
        let shape = vec![1000, 1000];
        let budget = 100_000; // 100 KB

        let tile = compute_tile_sizes_budget(&shape, 8, budget).unwrap();

        assert!(tile.memory_per_tile <= budget);
    }

    #[test]
    fn test_compute_tile_sizes_budget_fits() {
        let shape = vec![10, 10];
        let budget = 10_000; // More than enough

        let tile = compute_tile_sizes_budget(&shape, 8, budget).unwrap();

        // Should not need tiling
        assert_eq!(tile.sizes, vec![10, 10]);
    }

    #[test]
    fn test_compute_matmul_block_sizes() {
        let config = CacheConfig::default();

        let (bm, bk, bn) = compute_matmul_block_sizes(1000, 1000, 1000, 8, &config);

        // Block sizes should be reasonable
        assert!((32..=1000).contains(&bm));
        assert!((32..=1000).contains(&bk));
        assert!((32..=1000).contains(&bn));

        // Should fit in cache
        let memory = (bm * bk + bk * bn + bm * bn) * 8;
        assert!(memory <= config.l2_size);
    }

    #[test]
    fn test_compute_matmul_block_sizes_small() {
        let config = CacheConfig::default();

        // Small matrices
        let (bm, bk, bn) = compute_matmul_block_sizes(50, 50, 50, 8, &config);

        // Should not need blocking
        assert_eq!(bm, 50);
        assert_eq!(bk, 50);
        assert_eq!(bn, 50);
    }

    #[test]
    fn test_tile_spec_fits_in_cache() {
        let tile = TileSpec::new(vec![100, 100], 8);

        assert!(tile.fits_in_cache(100_000)); // Fits
        assert!(!tile.fits_in_cache(10_000)); // Doesn't fit
    }

    #[test]
    fn test_estimate_cache_misses_sequential() {
        let tile = TileSpec::new(vec![100, 100], 8);
        let config = CacheConfig::default();

        let misses = estimate_cache_misses(&tile, "sequential", &config);

        // Sequential should have fewer misses
        assert!(misses > 0.0);
        assert!(misses < 100_000.0);
    }

    #[test]
    fn test_estimate_cache_misses_random() {
        let tile = TileSpec::new(vec![100, 100], 8);
        let config = CacheConfig::default();

        let misses = estimate_cache_misses(&tile, "random", &config);

        // Random should have more misses
        assert!(misses > 0.0);
    }

    #[test]
    fn test_compute_tile_sizes_empty_shape() {
        let result = compute_tile_sizes(&[], 8, &CacheConfig::default(), "L1");
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_tile_sizes_invalid_cache() {
        let result = compute_tile_sizes(&[10, 10], 8, &CacheConfig::default(), "L4");
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_tile_sizes_budget_too_small() {
        let result = compute_tile_sizes_budget(&[1000, 1000], 8, 1);
        assert!(result.is_err());
    }

    // ============================================================================
    // Tests for Multi-Level Tiling
    // ============================================================================

    #[test]
    fn test_multilevel_tiling_small_tensor() {
        // Small tensor that fits in L1
        let shape = vec![16, 16];
        let config = CacheConfig::default();

        let spec = MultiLevelTileSpec::compute(&shape, 8, &config).unwrap();

        // Should not need tiling
        assert_eq!(spec.l1_tiles, vec![16, 16]);
        assert_eq!(spec.l2_tiles, vec![16, 16]);
        assert_eq!(spec.l3_tiles, vec![16, 16]);
    }

    #[test]
    fn test_multilevel_tiling_large_tensor() {
        // Large tensor requiring multi-level tiling
        let shape = vec![2048, 2048];
        let config = CacheConfig::default();

        let spec = MultiLevelTileSpec::compute(&shape, 8, &config).unwrap();

        // L1 tiles should be smallest
        let l1_size = spec.l1_tiles.iter().product::<usize>() * 8;
        assert!(l1_size <= config.l1_size);

        // L2 tiles should be >= L1 tiles
        for i in 0..spec.l1_tiles.len() {
            assert!(spec.l2_tiles[i] >= spec.l1_tiles[i]);
        }

        // L3 tiles should be >= L2 tiles
        for i in 0..spec.l2_tiles.len() {
            assert!(spec.l3_tiles[i] >= spec.l2_tiles[i]);
        }

        // L2 tiles should fit in L2 cache
        let l2_size = spec.l2_tiles.iter().product::<usize>() * 8;
        assert!(l2_size <= config.l2_size * 2); // Allow some tolerance

        // L3 tiles should fit in L3 cache
        let l3_size = spec.l3_tiles.iter().product::<usize>() * 8;
        assert!(l3_size <= config.l3_size * 2); // Allow some tolerance
    }

    #[test]
    fn test_multilevel_tiling_3d_tensor() {
        let shape = vec![256, 256, 256];
        let config = CacheConfig::default();

        let spec = MultiLevelTileSpec::compute(&shape, 8, &config).unwrap();

        assert_eq!(spec.l1_tiles.len(), 3);
        assert_eq!(spec.l2_tiles.len(), 3);
        assert_eq!(spec.l3_tiles.len(), 3);

        // Verify hierarchy
        for i in 0..3 {
            assert!(spec.l1_tiles[i] <= spec.l2_tiles[i]);
            assert!(spec.l2_tiles[i] <= spec.l3_tiles[i]);
        }
    }

    #[test]
    fn test_multilevel_tiles_per_level() {
        let shape = vec![1024, 1024];
        let config = CacheConfig::default();

        let spec = MultiLevelTileSpec::compute(&shape, 8, &config).unwrap();

        // Test tiles per level calculations
        for dim in 0..2 {
            let l1_per_l2 = spec.l1_tiles_per_l2(dim);
            let l2_per_l3 = spec.l2_tiles_per_l3(dim);

            assert!(l1_per_l2 >= 1);
            assert!(l2_per_l3 >= 1);
        }
    }

    #[test]
    fn test_multilevel_total_tiles() {
        let shape = vec![512, 512];
        let config = CacheConfig::default();

        let spec = MultiLevelTileSpec::compute(&shape, 8, &config).unwrap();

        let total_l1 = spec.total_l1_tiles();
        let total_l2 = spec.total_l2_tiles();
        let total_l3 = spec.total_l3_tiles();

        // More L1 tiles than L2 tiles than L3 tiles
        assert!(total_l1 >= total_l2);
        assert!(total_l2 >= total_l3);
        assert!(total_l3 >= 1);
    }

    #[test]
    fn test_multilevel_tiling_empty_shape() {
        let config = CacheConfig::default();
        let result = MultiLevelTileSpec::compute(&[], 8, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_multilevel_blocks_small() {
        let config = CacheConfig::default();
        let (l1, l2, l3) = compute_matmul_multilevel_blocks(64, 64, 64, 8, &config);

        // Verify hierarchy
        assert!(l1.0 <= l2.0);
        assert!(l1.1 <= l2.1);
        assert!(l1.2 <= l2.2);

        assert!(l2.0 <= l3.0);
        assert!(l2.1 <= l3.1);
        assert!(l2.2 <= l3.2);

        // For small matrices, may not need multiple levels
        assert!(l1.0 > 0 && l1.1 > 0 && l1.2 > 0);
    }

    #[test]
    fn test_matmul_multilevel_blocks_large() {
        let config = CacheConfig::default();
        let (l1, l2, l3) = compute_matmul_multilevel_blocks(2048, 2048, 2048, 8, &config);

        // L1 blocks should be small (micro-kernels)
        assert!(l1.0 <= 32);
        assert!(l1.1 <= 32);
        assert!(l1.2 <= 32);

        // L2 blocks should be medium (panels)
        assert!(l2.0 >= 32 && l2.0 <= 256);
        assert!(l2.1 >= 32 && l2.1 <= 256);
        assert!(l2.2 >= 32 && l2.2 <= 256);

        // L3 blocks should be large (outer blocks)
        assert!(l3.0 >= 128);
        assert!(l3.1 >= 128);
        assert!(l3.2 >= 128);

        // Verify strict hierarchy
        assert!(l1.0 <= l2.0 && l2.0 <= l3.0);
        assert!(l1.1 <= l2.1 && l2.1 <= l3.1);
        assert!(l1.2 <= l2.2 && l2.2 <= l3.2);
    }

    #[test]
    fn test_matmul_multilevel_blocks_rectangular() {
        let config = CacheConfig::default();
        let (l1, l2, l3) = compute_matmul_multilevel_blocks(100, 2000, 500, 8, &config);

        // Should handle rectangular matrices
        assert!(l1.0 > 0 && l1.1 > 0 && l1.2 > 0);
        assert!(l2.0 > 0 && l2.1 > 0 && l2.2 > 0);
        assert!(l3.0 > 0 && l3.1 > 0 && l3.2 > 0);

        // Hierarchy should still hold
        assert!(l1.0 <= l2.0 && l2.0 <= l3.0);
        assert!(l1.1 <= l2.1 && l2.1 <= l3.1);
        assert!(l1.2 <= l2.2 && l2.2 <= l3.2);

        // Should not exceed matrix dimensions
        assert!(l3.0 <= 100);
        assert!(l3.1 <= 2000);
        assert!(l3.2 <= 500);
    }
}
