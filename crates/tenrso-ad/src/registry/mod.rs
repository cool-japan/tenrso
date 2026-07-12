//! Framework-agnostic operation registry
//!
//! This module provides the *pluggable* op registry that external AD frameworks
//! (Tensorlogic, or any other tape/graph engine) use to discover TenRSo
//! operations together with their gradient rules.
//!
//! # Why a registry
//!
//! [`crate::graph`] contains a *closed* `Operation` enum with a hard-coded
//! backward `match`. That is fine for TenRSo's own tape, but an external engine
//! cannot extend it, and it cannot ask "which ops do you support?".
//!
//! The registry solves both problems:
//!
//! * ops are addressed **by name** (`"einsum"`, `"relu"`, `"cp_reconstruct"`, …),
//! * every op exposes both a **forward** and a **VJP** (backward) implementation
//!   behind one object-safe trait ([`OpRule`]),
//! * new ops can be added at runtime with [`OpRegistry::register`].
//!
//! # Relationship with [`crate::hooks`]
//!
//! The registry does not duplicate [`crate::hooks::AdOperation`] /
//! [`crate::hooks::AdContext`] — it *feeds* them.
//! [`OpRegistry::make_operation`] binds a rule to concrete inputs and yields a
//! [`RegisteredOp`], which implements [`crate::hooks::AdOperation`] and can be
//! pushed straight onto any [`crate::hooks::AdContext`] tape.
//!
//! ```rust,ignore
//! use tenrso_ad::hooks::{AdContext, GenericAdAdapter};
//! use tenrso_ad::registry::{OpParams, OpRegistry};
//!
//! let registry = OpRegistry::<f64>::with_builtins();
//! let op = registry.make_operation("einsum", vec![a, b], OpParams::einsum("ij,jk->ik"))?;
//!
//! let mut tape = GenericAdAdapter::<f64>::new();
//! let id = tape.register_operation(Box::new(op));
//! ```
//!
//! # Gradient provenance
//!
//! The rules do not re-derive gradients that already exist in this crate:
//!
//! | rule | forward | VJP |
//! |------|---------|-----|
//! | [`EinsumRule`] | [`tenrso_exec`] contraction / planner | [`crate::vjp::EinsumVjp`] (binary), adjoint einsum (n-ary) |
//! | [`ElementwiseUnaryRule`] | element map | [`crate::vjp::ElementwiseUnaryVjp`] |
//! | [`ElementwiseBinaryRule`] | element map | [`crate::vjp::ElementwiseBinaryVjp`] |
//! | [`CpReconstructionRule`] | Khatri-Rao contraction | [`crate::grad::CpReconstructionGrad`] |
//! | [`TuckerReconstructionRule`] | mode-n product chain | [`crate::grad::TuckerReconstructionGrad`] |
//! | [`TtReconstructionRule`] | [`crate::grad::tt_reconstruct`] | [`crate::grad::TtReconstructionGrad`] |
//! | [`ReductionRule`] | value reduction | native (see below) |
//!
//! [`ReductionRule`] is the canonical implementation, not a wrapper around
//! [`crate::vjp::ReductionVjp`] — it is the other way around:
//! `ReductionVjp::with_input`/`with_input_axes` delegate to
//! [`ReductionRule::vjp`] (using a private axis-validation/shape-translation
//! helper) so there is exactly one implementation of the sum/mean/max/min/product
//! math. Every rule in this module is verified against
//! [`crate::gradcheck::check_gradient`] in `tests/registry_gradcheck.rs`;
//! `ReductionVjp`'s delegation is separately gradchecked in `src/vjp.rs`.

use anyhow::{anyhow, bail, Result};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use tenrso_core::DenseND;

use crate::hooks::AdOperation;

pub mod decomposition;
pub mod einsum;
pub mod elementwise;
pub mod reduction;

pub use decomposition::{CpReconstructionRule, TtReconstructionRule, TuckerReconstructionRule};
pub use einsum::EinsumRule;
pub use elementwise::{BinaryOpKind, ElementwiseBinaryRule, ElementwiseUnaryRule, UnaryOpKind};
pub use reduction::{ReduceKind, ReductionRule};

/// Scalar element type usable by registry rules.
///
/// This is the intersection of what the existing TenRSo gradient rules need:
/// * [`Float`] — the decomposition gradients ([`crate::grad`]),
/// * `AddAssign + Default` — [`tenrso_exec::ops::execute_dense_contraction`],
/// * [`FromPrimitive`] — the n-ary einsum planner path,
/// * `Send + Sync + Debug + 'static` — object-safe storage in the registry.
///
/// It is blanket-implemented, so `f32` and `f64` satisfy it automatically.
pub trait AdScalar:
    Float + FromPrimitive + Default + std::ops::AddAssign + Send + Sync + Debug + 'static
{
}

impl<T> AdScalar for T where
    T: Float + FromPrimitive + Default + std::ops::AddAssign + Send + Sync + Debug + 'static
{
}

/// Number of inputs an operation accepts.
///
/// Decomposition reconstructions are variadic (one input per factor / core),
/// so a plain `usize` cannot describe them faithfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    /// Exactly `n` inputs.
    Exact(usize),
    /// At least `n` inputs (variadic tail).
    AtLeast(usize),
}

impl Arity {
    /// Does this arity accept `n` inputs?
    pub fn accepts(&self, n: usize) -> bool {
        match *self {
            Arity::Exact(k) => n == k,
            Arity::AtLeast(k) => n >= k,
        }
    }

    /// Minimum number of inputs accepted.
    pub fn min_inputs(&self) -> usize {
        match *self {
            Arity::Exact(k) | Arity::AtLeast(k) => k,
        }
    }
}

impl std::fmt::Display for Arity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Arity::Exact(k) => write!(f, "exactly {k}"),
            Arity::AtLeast(k) => write!(f, "at least {k}"),
        }
    }
}

/// Static (non-tensor) parameters of an operation invocation.
///
/// Some ops are parameterized by data that is not a tensor: einsum carries a
/// contraction spec, reductions carry the axes they collapse. Keeping those in
/// a separate value (instead of baking them into the rule object) means a single
/// registered rule instance serves every spec / axis set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpParams {
    /// Einsum contraction specification, e.g. `"ij,jk->ik"`.
    pub spec: Option<String>,
    /// Axes for reduction operations. Empty means "no axis argument given".
    pub axes: Vec<usize>,
}

impl OpParams {
    /// Parameters for an op that takes none.
    pub fn none() -> Self {
        Self::default()
    }

    /// Parameters carrying an einsum spec.
    pub fn einsum(spec: impl Into<String>) -> Self {
        Self {
            spec: Some(spec.into()),
            axes: Vec::new(),
        }
    }

    /// Parameters carrying reduction axes.
    pub fn axes(axes: impl Into<Vec<usize>>) -> Self {
        Self {
            spec: None,
            axes: axes.into(),
        }
    }

    /// Borrow the einsum spec, or fail with a descriptive error.
    pub fn require_spec(&self, op_name: &str) -> Result<&str> {
        self.spec
            .as_deref()
            .ok_or_else(|| anyhow!("operation '{op_name}' requires an einsum spec in OpParams"))
    }
}

/// A differentiable operation: forward evaluation plus its vector-Jacobian product.
///
/// Rules are stateless with respect to a call (all call data arrives through
/// `inputs` / `params`), which is what makes them shareable across threads and
/// storable in an [`OpRegistry`].
pub trait OpRule<T>: Debug + Send + Sync
where
    T: AdScalar,
{
    /// Unique registry key, e.g. `"einsum"`.
    fn name(&self) -> &str;

    /// How many inputs this op accepts.
    fn arity(&self) -> Arity;

    /// Forward evaluation. Produces exactly one output tensor.
    fn forward(&self, inputs: &[DenseND<T>], params: &OpParams) -> Result<DenseND<T>>;

    /// Vector-Jacobian product.
    ///
    /// Given `output_grad` = ∂L/∂output, returns ∂L/∂input for every input, in
    /// input order. The returned vector always has `inputs.len()` entries and
    /// each entry has the shape of the corresponding input.
    fn vjp(
        &self,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
        params: &OpParams,
    ) -> Result<Vec<DenseND<T>>>;

    /// Validate the number of inputs against [`OpRule::arity`].
    fn check_arity(&self, n_inputs: usize) -> Result<()> {
        if !self.arity().accepts(n_inputs) {
            bail!(
                "operation '{}' expects {} input(s), got {}",
                self.name(),
                self.arity(),
                n_inputs
            );
        }
        Ok(())
    }
}

/// Registry of named operation rules.
///
/// Cloning a registry is cheap: the rules are behind [`Arc`].
#[derive(Debug)]
pub struct OpRegistry<T>
where
    T: AdScalar,
{
    rules: HashMap<String, Arc<dyn OpRule<T>>>,
}

impl<T> Clone for OpRegistry<T>
where
    T: AdScalar,
{
    fn clone(&self) -> Self {
        Self {
            rules: self.rules.clone(),
        }
    }
}

impl<T> Default for OpRegistry<T>
where
    T: AdScalar,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OpRegistry<T>
where
    T: AdScalar,
{
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with every built-in TenRSo rule.
    pub fn with_builtins() -> Self {
        let mut rules: HashMap<String, Arc<dyn OpRule<T>>> = HashMap::new();
        for rule in builtin_rules::<T>() {
            rules.insert(rule.name().to_string(), Arc::from(rule));
        }
        Self { rules }
    }

    /// Add the built-in rules to an existing registry.
    ///
    /// # Errors
    ///
    /// Fails if a built-in name collides with an already registered rule.
    pub fn register_builtins(&mut self) -> Result<()> {
        for rule in builtin_rules::<T>() {
            self.register(rule)?;
        }
        Ok(())
    }

    /// Register a rule.
    ///
    /// # Errors
    ///
    /// Fails if a rule with the same name is already registered — registration
    /// never silently shadows an existing op.
    pub fn register(&mut self, rule: Box<dyn OpRule<T>>) -> Result<()> {
        let name = rule.name().to_string();
        if name.is_empty() {
            bail!("cannot register an operation with an empty name");
        }
        if self.rules.contains_key(&name) {
            bail!("operation '{name}' is already registered");
        }
        self.rules.insert(name, Arc::from(rule));
        Ok(())
    }

    /// Replace (or insert) a rule, returning the previous one if any.
    pub fn replace(&mut self, rule: Box<dyn OpRule<T>>) -> Option<Arc<dyn OpRule<T>>> {
        let name = rule.name().to_string();
        self.rules.insert(name, Arc::from(rule))
    }

    /// Remove a rule.
    pub fn unregister(&mut self, name: &str) -> Option<Arc<dyn OpRule<T>>> {
        self.rules.remove(name)
    }

    /// Look up a rule by name.
    pub fn get(&self, name: &str) -> Option<&dyn OpRule<T>> {
        self.rules.get(name).map(|rule| rule.as_ref())
    }

    /// Look up a rule by name, returning a shared handle.
    pub fn get_arc(&self, name: &str) -> Option<Arc<dyn OpRule<T>>> {
        self.rules.get(name).cloned()
    }

    /// Look up a rule by name, failing with a descriptive error.
    pub fn require(&self, name: &str) -> Result<&dyn OpRule<T>> {
        self.get(name)
            .ok_or_else(|| anyhow!("operation '{name}' is not registered"))
    }

    /// Is this op registered?
    pub fn contains(&self, name: &str) -> bool {
        self.rules.contains_key(name)
    }

    /// Number of registered ops.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Is the registry empty?
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// All registered op names, sorted (deterministic order).
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.rules.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Run the forward pass of a registered op.
    pub fn forward(
        &self,
        name: &str,
        inputs: &[DenseND<T>],
        params: &OpParams,
    ) -> Result<DenseND<T>> {
        let rule = self.require(name)?;
        rule.check_arity(inputs.len())?;
        rule.forward(inputs, params)
    }

    /// Run the VJP of a registered op.
    pub fn vjp(
        &self,
        name: &str,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
        params: &OpParams,
    ) -> Result<Vec<DenseND<T>>> {
        let rule = self.require(name)?;
        rule.check_arity(inputs.len())?;
        let grads = rule.vjp(inputs, output_grad, params)?;
        if grads.len() != inputs.len() {
            bail!(
                "operation '{name}' returned {} gradients for {} inputs",
                grads.len(),
                inputs.len()
            );
        }
        for (idx, (grad, input)) in grads.iter().zip(inputs.iter()).enumerate() {
            if grad.shape() != input.shape() {
                bail!(
                    "operation '{name}': gradient {idx} has shape {:?}, expected {:?}",
                    grad.shape(),
                    input.shape()
                );
            }
        }
        Ok(grads)
    }

    /// Bind a registered rule to concrete inputs, producing a tape-ready
    /// [`AdOperation`].
    pub fn make_operation(
        &self,
        name: &str,
        inputs: Vec<DenseND<T>>,
        params: OpParams,
    ) -> Result<RegisteredOp<T>> {
        let rule = self
            .get_arc(name)
            .ok_or_else(|| anyhow!("operation '{name}' is not registered"))?;
        rule.check_arity(inputs.len())?;
        Ok(RegisteredOp {
            rule,
            inputs,
            params,
        })
    }
}

/// Every rule TenRSo ships with.
fn builtin_rules<T: AdScalar>() -> Vec<Box<dyn OpRule<T>>> {
    let mut rules: Vec<Box<dyn OpRule<T>>> = Vec::new();

    rules.push(Box::new(EinsumRule::new()));

    for kind in UnaryOpKind::ALL {
        rules.push(Box::new(ElementwiseUnaryRule::new(kind)));
    }
    for kind in BinaryOpKind::ALL {
        rules.push(Box::new(ElementwiseBinaryRule::new(kind)));
    }
    for kind in ReduceKind::ALL {
        rules.push(Box::new(ReductionRule::new(kind)));
    }

    rules.push(Box::new(CpReconstructionRule::new()));
    rules.push(Box::new(TuckerReconstructionRule::new()));
    rules.push(Box::new(TtReconstructionRule::new()));

    rules
}

/// A registry rule bound to concrete inputs.
///
/// This is the bridge between the registry and the existing
/// [`AdOperation`] / [`crate::hooks::AdContext`] tape traits.
#[derive(Debug)]
pub struct RegisteredOp<T>
where
    T: AdScalar,
{
    rule: Arc<dyn OpRule<T>>,
    inputs: Vec<DenseND<T>>,
    params: OpParams,
}

impl<T> RegisteredOp<T>
where
    T: AdScalar,
{
    /// Bind a rule to inputs directly (without going through a registry).
    pub fn new(
        rule: Arc<dyn OpRule<T>>,
        inputs: Vec<DenseND<T>>,
        params: OpParams,
    ) -> Result<Self> {
        rule.check_arity(inputs.len())?;
        Ok(Self {
            rule,
            inputs,
            params,
        })
    }

    /// The saved forward inputs.
    pub fn inputs(&self) -> &[DenseND<T>] {
        &self.inputs
    }

    /// The static parameters of this invocation.
    pub fn params(&self) -> &OpParams {
        &self.params
    }

    /// The underlying rule.
    pub fn rule(&self) -> &dyn OpRule<T> {
        self.rule.as_ref()
    }
}

impl<T> AdOperation<T> for RegisteredOp<T>
where
    T: AdScalar,
{
    fn forward(&self) -> Result<Vec<DenseND<T>>> {
        Ok(vec![self.rule.forward(&self.inputs, &self.params)?])
    }

    fn backward(&self, output_grads: &[DenseND<T>]) -> Result<Vec<DenseND<T>>> {
        if output_grads.len() != 1 {
            bail!(
                "operation '{}' has 1 output, got {} output gradients",
                self.rule.name(),
                output_grads.len()
            );
        }
        self.rule.vjp(&self.inputs, &output_grads[0], &self.params)
    }

    fn name(&self) -> &str {
        self.rule.name()
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// Shared helpers used by the rule implementations
// ---------------------------------------------------------------------------

/// Row-major strides for a shape.
pub(crate) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    strides
}

/// Expand a row-major linear index into a multi-index for `shape`.
pub(crate) fn linear_to_multi(linear: usize, shape: &[usize], out: &mut [usize]) {
    let mut remaining = linear;
    for dim in (0..shape.len()).rev() {
        let size = shape[dim];
        if size == 0 {
            out[dim] = 0;
            continue;
        }
        out[dim] = remaining % size;
        remaining /= size;
    }
}

/// View a rank-2 [`DenseND`] as a matrix.
pub(crate) fn as_matrix<T: AdScalar>(
    tensor: &DenseND<T>,
    what: &str,
) -> Result<scirs2_core::ndarray_ext::Array2<T>> {
    if tensor.rank() != 2 {
        bail!(
            "{what} must be a matrix (rank 2), got shape {:?}",
            tensor.shape()
        );
    }
    tensor
        .as_array()
        .to_owned()
        .into_dimensionality::<scirs2_core::ndarray_ext::Ix2>()
        .map_err(|e| anyhow!("{what}: failed to view as matrix: {e}"))
}

/// Wrap a matrix back into a [`DenseND`].
pub(crate) fn from_matrix<T: AdScalar>(matrix: scirs2_core::ndarray_ext::Array2<T>) -> DenseND<T> {
    DenseND::from_array(matrix.into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_builtins_are_registered() {
        let registry = OpRegistry::<f64>::with_builtins();

        for expected in [
            "einsum",
            "relu",
            "sigmoid",
            "add",
            "mul",
            "reduce_sum",
            "reduce_mean",
            "reduce_max",
            "reduce_min",
            "cp_reconstruct",
            "tucker_reconstruct",
            "tt_reconstruct",
        ] {
            assert!(
                registry.contains(expected),
                "builtin '{expected}' should be registered"
            );
        }

        assert_eq!(registry.len(), registry.names().len());
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_registry_rejects_duplicates() {
        let mut registry = OpRegistry::<f64>::with_builtins();
        let err = registry
            .register(Box::new(ElementwiseUnaryRule::new(UnaryOpKind::Relu)))
            .unwrap_err();
        assert!(err.to_string().contains("already registered"));
    }

    #[test]
    fn test_registry_register_builtins_twice_fails() {
        let mut registry = OpRegistry::<f64>::with_builtins();
        assert!(registry.register_builtins().is_err());

        let mut empty = OpRegistry::<f64>::new();
        assert!(empty.register_builtins().is_ok());
        assert_eq!(empty.len(), OpRegistry::<f64>::with_builtins().len());
    }

    #[test]
    fn test_registry_unknown_op() {
        let registry = OpRegistry::<f64>::with_builtins();
        assert!(registry.get("does_not_exist").is_none());
        let err = registry.require("does_not_exist").unwrap_err();
        assert!(err.to_string().contains("not registered"));
    }

    #[test]
    fn test_arity_semantics() {
        assert!(Arity::Exact(2).accepts(2));
        assert!(!Arity::Exact(2).accepts(3));
        assert!(Arity::AtLeast(2).accepts(5));
        assert!(!Arity::AtLeast(2).accepts(1));
        assert_eq!(Arity::AtLeast(3).min_inputs(), 3);
    }

    #[test]
    fn test_registered_op_implements_ad_operation() {
        use crate::hooks::AdOperation;

        let registry = OpRegistry::<f64>::with_builtins();
        let x = DenseND::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();

        let op = registry
            .make_operation("relu", vec![x.clone()], OpParams::none())
            .unwrap();

        assert_eq!(op.name(), "relu");
        assert_eq!(op.num_inputs(), 1);
        assert_eq!(op.num_outputs(), 1);

        let outputs = op.forward().unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(*outputs[0].get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*outputs[0].get(&[0, 1]).unwrap(), 0.0);

        let grads = op.backward(&[DenseND::ones(&[2, 2])]).unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(*grads[0].get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*grads[0].get(&[0, 1]).unwrap(), 0.0);
    }

    #[test]
    fn test_registry_arity_is_enforced() {
        let registry = OpRegistry::<f64>::with_builtins();
        let x = DenseND::<f64>::ones(&[2, 2]);
        let err = registry
            .forward("relu", &[x.clone(), x.clone()], &OpParams::none())
            .unwrap_err();
        assert!(err.to_string().contains("expects"));
    }

    #[test]
    fn test_row_major_helpers() {
        let shape = [2usize, 3, 4];
        assert_eq!(row_major_strides(&shape), vec![12, 4, 1]);

        let mut multi = [0usize; 3];
        linear_to_multi(13, &shape, &mut multi);
        // 13 = 1*12 + 0*4 + 1
        assert_eq!(multi, [1, 0, 1]);
    }
}
