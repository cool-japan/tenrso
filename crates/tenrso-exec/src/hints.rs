//! Execution hints and configuration

/// Mask specification — a flat boolean mask over tensor elements.
///
/// The mask is stored as a flat `Vec<bool>` in row-major (C) order together
/// with the matching shape so that callers can reconstruct the multi-dimensional
/// structure without pulling in a tensor dependency here.
#[derive(Clone, Debug)]
pub struct MaskPack {
    /// Boolean mask values in row-major order.
    pub mask: Option<Vec<bool>>,
    /// Shape of the mask tensor (same rank as the target tensor).
    pub shape: Vec<usize>,
}

impl MaskPack {
    /// Create a new mask pack.
    pub fn new(mask: Vec<bool>, shape: Vec<usize>) -> Self {
        Self {
            mask: Some(mask),
            shape,
        }
    }

    /// Create an empty (no-op) mask pack.
    pub fn empty() -> Self {
        Self {
            mask: None,
            shape: Vec::new(),
        }
    }
}

/// Subset specification — a list of flat indices to select from a tensor.
///
/// Indices are stored in the order they should be processed; callers are
/// responsible for interpreting them relative to a specific axis or
/// flattened layout.
#[derive(Clone, Debug)]
pub struct SubsetSpec {
    /// Flat indices selecting a subset of tensor elements (row-major order).
    pub indices: Option<Vec<usize>>,
}

impl SubsetSpec {
    /// Create a new subset specification from a list of indices.
    pub fn new(indices: Vec<usize>) -> Self {
        Self {
            indices: Some(indices),
        }
    }

    /// Create an empty (no-op) subset specification.
    pub fn empty() -> Self {
        Self { indices: None }
    }
}

/// Execution hints for controlling tensor operations
#[derive(Clone, Debug, Default)]
pub struct ExecHints {
    /// Optional mask for masked operations
    pub mask: Option<MaskPack>,
    /// Optional subset specification
    pub subset: Option<SubsetSpec>,
    /// Prefer sparse representation
    pub prefer_sparse: bool,
    /// Prefer low-rank representation
    pub prefer_lowrank: bool,
    /// Tile size in KB
    pub tile_kb: Option<usize>,
}

impl ExecHints {
    /// Create new execution hints with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set sparse preference
    pub fn with_sparse(mut self, prefer: bool) -> Self {
        self.prefer_sparse = prefer;
        self
    }

    /// Set low-rank preference
    pub fn with_lowrank(mut self, prefer: bool) -> Self {
        self.prefer_lowrank = prefer;
        self
    }

    /// Set tile size
    pub fn with_tile_kb(mut self, kb: usize) -> Self {
        self.tile_kb = Some(kb);
        self
    }

    /// Set a boolean mask (flat row-major) for masked einsum routing.
    ///
    /// When combined with `prefer_sparse = true`, the executor routes through
    /// the sparse masked einsum path, computing only the output positions
    /// indicated by the mask.
    pub fn with_mask(mut self, mask: Vec<bool>, shape: Vec<usize>) -> Self {
        self.mask = Some(MaskPack::new(mask, shape));
        self
    }
}
