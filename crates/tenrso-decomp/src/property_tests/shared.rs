//! Shared configuration for the decomposition property tests.

use proptest::prelude::ProptestConfig;

/// Proptest configuration for the decomposition property tests.
///
/// Tensor decompositions are far too expensive for proptest's default 256 cases;
/// three well-chosen cases per property keep each test under ~30s while still
/// exercising a range of shapes and ranks.
pub(super) fn proptest_config() -> ProptestConfig {
    ProptestConfig {
        cases: 3,
        max_local_rejects: 1000,
        max_global_rejects: 10000,
        ..ProptestConfig::default()
    }
}
