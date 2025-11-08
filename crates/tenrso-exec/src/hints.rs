//! Execution hints and configuration

/// Mask specification (placeholder)
#[derive(Clone, Debug)]
pub struct MaskPack {
    // TODO: Implement mask storage
}

/// Subset specification (placeholder)
#[derive(Clone, Debug)]
pub struct SubsetSpec {
    // TODO: Implement subset/index-list storage
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
}
