//! Property-based tests for tensor decompositions.
//!
//! These tests use proptest to verify mathematical properties that should hold
//! for all tensor decompositions, grouped by decomposition family. They were
//! split out of a single 2003-line `property_tests.rs` (over the 2000-line
//! per-file limit); the test bodies are unchanged.

mod shared;

mod completion;
mod constraints;
mod convergence;
mod cp;
mod cross;
mod randomized;
mod tt;
mod tt_matvec;
mod tucker;
