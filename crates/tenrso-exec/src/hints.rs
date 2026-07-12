//! Execution hints and configuration.
//!
//! An [`ExecHints`] value is a *request*, but never a silent one: every field on
//! it is read by the executor and changes what runs.  A hint that the engine
//! cannot honour is either rejected with an error or absent from this struct —
//! there is deliberately no knob here that is accepted and then ignored.
//!
//! # Selecting part of the output
//!
//! Two fields describe "compute only these output positions", and they are two
//! spellings of one capability — the `tenrso_sparse::masked_einsum` engine.  For
//! an output of `N` cells of which `s` are selected:
//!
//! | field                         | representation                     | memory | use when                   |
//! |-------------------------------|------------------------------------|--------|----------------------------|
//! | [`mask`](ExecHints::mask)     | dense `Vec<bool>` over the output  | `O(N)` | the selection is dense-ish |
//! | [`subset`](ExecHints::subset) | flat indices of the selected cells | `O(s)` | the selection is sparse    |
//!
//! Both require [`prefer_sparse`](ExecHints::prefer_sparse) — that flag is what
//! switches `einsum_ex` from the dense GEMM engine to the masked engine.  Setting
//! both `mask` and `subset` at once is an error rather than a silent precedence
//! rule.

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

/// Subset specification — the flat indices of the output cells to compute.
///
/// This is the sparse spelling of [`MaskPack`]: instead of one `bool` per output
/// element it stores only the selected positions, so a selection of `s` cells out
/// of an `N`-element output costs `O(s)` rather than `O(N)`.  Indices are
/// row-major (C-order) offsets into an output tensor of shape [`shape`](Self::shape),
/// which is what makes them unambiguous — a bare index list is not interpretable
/// without the extents it indexes into.
///
/// Order and duplicates do not matter: the executor converts the list to a
/// `tenrso_sparse::mask::Mask`, which is a *set* of positions.
#[derive(Clone, Debug)]
pub struct SubsetSpec {
    /// Flat row-major indices selecting the output cells to compute.
    pub indices: Option<Vec<usize>>,
    /// Shape of the output tensor the indices refer to.
    pub shape: Vec<usize>,
}

impl SubsetSpec {
    /// Create a new subset specification from flat row-major indices and the
    /// shape of the output tensor they index into.
    pub fn new(indices: Vec<usize>, shape: Vec<usize>) -> Self {
        Self {
            indices: Some(indices),
            shape,
        }
    }

    /// Create an empty (no-op) subset specification.
    pub fn empty() -> Self {
        Self {
            indices: None,
            shape: Vec::new(),
        }
    }
}

/// Execution hints for controlling tensor operations.
///
/// Every field here is honoured; see the [module documentation](self) for how
/// `mask` and `subset` relate.
#[derive(Clone, Debug, Default)]
pub struct ExecHints {
    /// Compute only the output cells selected by this boolean mask.
    ///
    /// Honoured together with [`prefer_sparse`](Self::prefer_sparse): the pair
    /// routes `einsum_ex` through the `tenrso_sparse` masked einsum engine.
    pub mask: Option<MaskPack>,
    /// Compute only the output cells at these flat indices.
    ///
    /// The sparse spelling of [`mask`](Self::mask), honoured on exactly the same
    /// terms and through the same engine.  Supplying both is an error.
    pub subset: Option<SubsetSpec>,
    /// Route through the sparse engine when the operation supports one.
    ///
    /// On its own this changes nothing for a dense einsum — there is no sparse
    /// kernel to route to until a [`mask`](Self::mask) or a
    /// [`subset`](Self::subset) says *which* output cells are wanted.  With one
    /// of those present it selects the masked einsum engine.
    pub prefer_sparse: bool,
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

    /// Set a boolean mask (flat row-major) for masked einsum routing.
    ///
    /// When combined with `prefer_sparse = true`, the executor routes through
    /// the sparse masked einsum path, computing only the output positions
    /// indicated by the mask.
    pub fn with_mask(mut self, mask: Vec<bool>, shape: Vec<usize>) -> Self {
        self.mask = Some(MaskPack::new(mask, shape));
        self
    }

    /// Select the output cells to compute by flat row-major index.
    ///
    /// The sparse counterpart of [`with_mask`](Self::with_mask): identical
    /// semantics and the same engine, but `O(|selection|)` memory instead of
    /// `O(|output|)`.  Also requires `prefer_sparse = true`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_exec::ExecHints;
    ///
    /// // Only the two diagonal cells of a 2×2 output are computed.
    /// let hints = ExecHints::new()
    ///     .with_sparse(true)
    ///     .with_subset(vec![0, 3], vec![2, 2]);
    /// assert_eq!(hints.subset.expect("subset set").indices, Some(vec![0, 3]));
    /// ```
    pub fn with_subset(mut self, indices: Vec<usize>, shape: Vec<usize>) -> Self {
        self.subset = Some(SubsetSpec::new(indices, shape));
        self
    }
}
