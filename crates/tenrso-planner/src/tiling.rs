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
}
