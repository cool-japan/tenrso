//! Tensorlogic bridge (feature `tensorlogic`, default-OFF).
//!
//! This module implements Tensorlogic's execution traits
//! ([`tensorlogic_infer::TlExecutor`] and [`tensorlogic_infer::TlAutodiff`])
//! on top of TenRSo tensors and
//! the TenRSo operation registry ([`crate::registry`]).
//!
//! ```text
//!   tensorlogic_ir::EinsumGraph
//!             │
//!             ▼
//!   TenrsoTlExecutor<T>  ── TlExecutor ──► einsum / elem_op / elem_op_binary / reduce
//!             │                                   │
//!             │                                   ▼
//!             │                        tenrso_ad::registry::OpRegistry<T>
//!             │                          (forward + VJP for every op)
//!             ▼
//!         GradientTape<T>  ◄── TlAutodiff::backward
//! ```
//!
//! # What this bridge covers
//!
//! * `TlExecutor::einsum` — 1, 2 and n-ary contractions. Two operands go through
//!   [`tenrso_exec::ops::execute_dense_contraction`]; three or more go through
//!   [`tenrso_exec::einsum_ex`] (planner-selected contraction order); a single
//!   operand is evaluated by the registry's native permute/diagonal/sum einsum.
//! * `TlExecutor::elem_op` — `Relu`, `Sigmoid`, `OneMinus`.
//! * `TlExecutor::elem_op_binary` — every arithmetic, comparison and fuzzy-logic
//!   `ElemOp` (`Add`, `Subtract`, `Multiply`, `Divide`, `Min`, `Max`, `Eq`, `Lt`,
//!   `Gt`, `Lte`, `Gte`, `OrMax`, `OrProbSum`, `Nand`, `Nor`, `Xor`), including
//!   the rank-0 scalar broadcast the reference backend performs.
//! * `TlExecutor::reduce` — `Sum`, `Mean`, `Max`, `Min`, `Product` over any axis set.
//! * `TlAutodiff::forward` — evaluates an `EinsumGraph` against the tensors bound
//!   with [`TenrsoTlExecutor::bind_tensor`], recording a forward tape.
//! * `TlAutodiff::backward` — reverse-mode sweep over the same graph using the
//!   registry's VJP rules, accumulating a gradient for **every** graph tensor
//!   (not just the inputs). Scalar-broadcast operands get their gradient summed
//!   back to rank 0.
//!
//! # What this bridge does NOT cover (honest limits)
//!
//! * **Graph inputs must be bound.** `TlAutodiff::forward` takes no tensors, so
//!   the executor resolves `graph.tensors[i]` by name against its own binding
//!   map (exact name, then the base name before `[`, then a `const_<value>`
//!   literal). An unbound name is a hard [`ExecutorError::TensorNotFound`] — the
//!   bridge never invents data.
//! * **Multi-output nodes** are rejected (`UnsupportedOperation`): every
//!   registry rule produces exactly one output.
//! * **Ops outside the registry** (e.g. the temporal `next`/`until` operators the
//!   Tensorlogic SciRS2 backend supports) are rejected with
//!   `UnsupportedOperation`. In particular, backward never falls back to
//!   "pass the gradient through", which would silently produce a wrong gradient.
//! * **Graphs with several outputs**: `TlAutodiff::forward` returns
//!   `graph.outputs[0]` (the trait's return type is a single tensor) and
//!   `backward` seeds the cotangent on that same tensor.
//! * **Sparse / quantized / device tensors** from `tensorlogic-infer` are not
//!   handled; `Self::Tensor` is a dense [`DenseND`].
//!
//! # Tensor conversion
//!
//! Tensorlogic's SciRS2 backend uses `Scirs2Tensor = ArrayD<f64>`, not
//! [`DenseND`]. [`dense_to_tl`] / [`tl_to_dense`] (and the `f32` variants)
//! perform the real conversion: element cast plus shape preservation.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array, ArrayD, IxDyn, Zip};
use std::collections::HashMap;
use tenrso_core::DenseND;
use tensorlogic_infer::{ElemOp, ExecutorError, ReduceOp, TlAutodiff, TlExecutor};
use tensorlogic_ir::{EinsumGraph, OpType};

use crate::registry::{AdScalar, OpParams, OpRegistry};

// ---------------------------------------------------------------------------
// Tensor conversion
// ---------------------------------------------------------------------------

/// Convert a TenRSo tensor into Tensorlogic's dense tensor type (`ArrayD<f64>`).
///
/// # Errors
///
/// Fails if an element cannot be represented as `f64`.
pub fn dense_to_tl<T: AdScalar>(tensor: &DenseND<T>) -> Result<ArrayD<f64>> {
    let mut values = Vec::with_capacity(tensor.len());
    for value in tensor.as_array().iter() {
        values.push(
            value
                .to_f64()
                .ok_or_else(|| anyhow!("tensor element cannot be represented as f64"))?,
        );
    }
    Array::from_shape_vec(IxDyn(tensor.shape()), values)
        .map_err(|e| anyhow!("failed to build a Tensorlogic tensor: {e}"))
}

/// Convert a Tensorlogic dense tensor (`ArrayD<f64>`) into a TenRSo tensor.
///
/// # Errors
///
/// Fails if an element cannot be represented in `T`.
pub fn tl_to_dense<T: AdScalar>(tensor: &ArrayD<f64>) -> Result<DenseND<T>> {
    let mut values = Vec::with_capacity(tensor.len());
    for &value in tensor.iter() {
        values
            .push(T::from(value).ok_or_else(|| {
                anyhow!("value {value} cannot be represented in the target type")
            })?);
    }
    DenseND::from_vec(values, tensor.shape())
}

/// Convert a TenRSo tensor into Tensorlogic's `f32` dense tensor type
/// (`Scirs2Tensor32 = ArrayD<f32>`).
///
/// # Errors
///
/// Fails if an element cannot be represented as `f32`.
pub fn dense_to_tl_f32<T: AdScalar>(tensor: &DenseND<T>) -> Result<ArrayD<f32>> {
    let mut values = Vec::with_capacity(tensor.len());
    for value in tensor.as_array().iter() {
        values.push(
            value
                .to_f32()
                .ok_or_else(|| anyhow!("tensor element cannot be represented as f32"))?,
        );
    }
    Array::from_shape_vec(IxDyn(tensor.shape()), values)
        .map_err(|e| anyhow!("failed to build a Tensorlogic tensor: {e}"))
}

/// Convert a Tensorlogic `f32` dense tensor into a TenRSo tensor.
///
/// # Errors
///
/// Fails if an element cannot be represented in `T`.
pub fn tl_f32_to_dense<T: AdScalar>(tensor: &ArrayD<f32>) -> Result<DenseND<T>> {
    let mut values = Vec::with_capacity(tensor.len());
    for &value in tensor.iter() {
        values
            .push(T::from(value).ok_or_else(|| {
                anyhow!("value {value} cannot be represented in the target type")
            })?);
    }
    DenseND::from_vec(values, tensor.shape())
}

// ---------------------------------------------------------------------------
// Op-name mapping (mirrors the Tensorlogic reference backend)
// ---------------------------------------------------------------------------

/// Registry name of a Tensorlogic unary `ElemOp`.
fn unary_registry_name(op: ElemOp) -> Result<&'static str, ExecutorError> {
    match op {
        ElemOp::Relu => Ok("relu"),
        ElemOp::Sigmoid => Ok("sigmoid"),
        ElemOp::OneMinus => Ok("one_minus"),
        other => Err(ExecutorError::UnsupportedOperation(format!(
            "{other:?} is a binary operation; call elem_op_binary instead"
        ))),
    }
}

/// Registry name of a Tensorlogic binary `ElemOp`.
fn binary_registry_name(op: ElemOp) -> Result<&'static str, ExecutorError> {
    match op {
        ElemOp::Add => Ok("add"),
        ElemOp::Subtract => Ok("sub"),
        ElemOp::Multiply => Ok("mul"),
        ElemOp::Divide => Ok("div"),
        ElemOp::Min => Ok("min"),
        ElemOp::Max => Ok("max"),
        ElemOp::Eq => Ok("eq"),
        ElemOp::Lt => Ok("lt"),
        ElemOp::Gt => Ok("gt"),
        ElemOp::Lte => Ok("lte"),
        ElemOp::Gte => Ok("gte"),
        ElemOp::OrMax => Ok("or_max"),
        ElemOp::OrProbSum => Ok("or_prob_sum"),
        ElemOp::Nand => Ok("nand"),
        ElemOp::Nor => Ok("nor"),
        ElemOp::Xor => Ok("xor"),
        other => Err(ExecutorError::UnsupportedOperation(format!(
            "{other:?} is a unary operation; call elem_op instead"
        ))),
    }
}

/// Registry name of a Tensorlogic `ReduceOp`.
fn reduce_registry_name(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "reduce_sum",
        ReduceOp::Mean => "reduce_mean",
        ReduceOp::Max => "reduce_max",
        ReduceOp::Min => "reduce_min",
        ReduceOp::Product => "reduce_product",
    }
}

/// Parse an `ElemUnary` node's op string (same spellings as the reference backend).
fn parse_unary_op(name: &str) -> Result<ElemOp, ExecutorError> {
    match name.to_lowercase().as_str() {
        "relu" => Ok(ElemOp::Relu),
        "sigmoid" => Ok(ElemOp::Sigmoid),
        "oneminus" | "one_minus" => Ok(ElemOp::OneMinus),
        other => Err(ExecutorError::UnsupportedOperation(format!(
            "unary operation '{other}' is not implemented by the TenRSo bridge"
        ))),
    }
}

/// Parse an `ElemBinary` node's op string (same spellings as the reference backend).
fn parse_binary_op(name: &str) -> Result<ElemOp, ExecutorError> {
    match name.to_lowercase().as_str() {
        "add" => Ok(ElemOp::Add),
        "subtract" | "sub" => Ok(ElemOp::Subtract),
        "multiply" | "mul" => Ok(ElemOp::Multiply),
        "divide" | "div" => Ok(ElemOp::Divide),
        "min" => Ok(ElemOp::Min),
        "max" => Ok(ElemOp::Max),
        "eq" | "equal" => Ok(ElemOp::Eq),
        "lt" | "lessthan" => Ok(ElemOp::Lt),
        "gt" | "greaterthan" => Ok(ElemOp::Gt),
        "lte" | "lessthanorequal" => Ok(ElemOp::Lte),
        "gte" | "greaterthanorequal" => Ok(ElemOp::Gte),
        "or_max" | "ormax" => Ok(ElemOp::OrMax),
        "or_prob_sum" | "orprobsum" | "or_probabilistic" => Ok(ElemOp::OrProbSum),
        "nand" => Ok(ElemOp::Nand),
        "nor" => Ok(ElemOp::Nor),
        "xor" => Ok(ElemOp::Xor),
        other => Err(ExecutorError::UnsupportedOperation(format!(
            "binary operation '{other}' is not implemented by the TenRSo bridge"
        ))),
    }
}

/// Parse a `Reduce` node's op string.
fn parse_reduce_op(name: &str) -> Result<ReduceOp, ExecutorError> {
    match name.to_lowercase().as_str() {
        "sum" => Ok(ReduceOp::Sum),
        "mean" => Ok(ReduceOp::Mean),
        "max" => Ok(ReduceOp::Max),
        "min" => Ok(ReduceOp::Min),
        "product" | "prod" => Ok(ReduceOp::Product),
        other => Err(ExecutorError::UnsupportedOperation(format!(
            "reduction '{other}' is not implemented by the TenRSo bridge"
        ))),
    }
}

/// Map a registry (`anyhow`) failure onto a Tensorlogic executor error.
fn exec_error(context: &str, error: anyhow::Error) -> ExecutorError {
    ExecutorError::InvalidInput(format!("{context}: {error:#}"))
}

// ---------------------------------------------------------------------------
// Broadcasting (scalar operands only, matching the Tensorlogic contract)
// ---------------------------------------------------------------------------

/// Expand a rank-0 operand to `shape`.
fn expand_scalar<T: AdScalar>(
    scalar: &DenseND<T>,
    shape: &[usize],
) -> Result<DenseND<T>, ExecutorError> {
    let value =
        *scalar.as_array().iter().next().ok_or_else(|| {
            ExecutorError::EmptyInput("scalar operand has no element".to_string())
        })?;
    Ok(DenseND::from_elem(shape, value))
}

/// Align two binary operands, broadcasting a rank-0 operand if needed.
///
/// Returns the aligned pair plus, for each side, whether it was broadcast (so the
/// backward pass knows it must sum the gradient back down to a scalar).
fn align_binary<T: AdScalar>(
    x: &DenseND<T>,
    y: &DenseND<T>,
) -> Result<(DenseND<T>, DenseND<T>, bool, bool), ExecutorError> {
    if x.shape() == y.shape() {
        return Ok((x.clone(), y.clone(), false, false));
    }
    if x.rank() == 0 {
        return Ok((expand_scalar(x, y.shape())?, y.clone(), true, false));
    }
    if y.rank() == 0 {
        return Ok((x.clone(), expand_scalar(y, x.shape())?, false, true));
    }
    Err(ExecutorError::ShapeMismatch(format!(
        "binary operands have incompatible shapes {:?} and {:?} (only a rank-0 operand is \
         broadcast)",
        x.shape(),
        y.shape()
    )))
}

/// Sum a gradient back onto a rank-0 operand.
fn sum_to_scalar<T: AdScalar>(grad: &DenseND<T>) -> DenseND<T> {
    let mut total = T::zero();
    for value in grad.as_array().iter() {
        total += *value;
    }
    DenseND::from_elem(&[], total)
}

/// Accumulate `grad` into `slot`.
fn accumulate<T: AdScalar>(
    slot: &mut Option<DenseND<T>>,
    grad: DenseND<T>,
) -> Result<(), ExecutorError> {
    match slot {
        None => {
            *slot = Some(grad);
            Ok(())
        }
        Some(existing) => {
            if existing.shape() != grad.shape() {
                return Err(ExecutorError::ShapeMismatch(format!(
                    "cannot accumulate a gradient of shape {:?} onto {:?}",
                    grad.shape(),
                    existing.shape()
                )));
            }
            Zip::from(existing.as_array_mut())
                .and(grad.as_array())
                .for_each(|acc, &value| {
                    *acc += value;
                });
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Tapes
// ---------------------------------------------------------------------------

/// Values recorded during [`TlAutodiff::forward`].
#[derive(Debug, Clone)]
pub struct ForwardTape<T>
where
    T: AdScalar,
{
    /// Value of every graph tensor (`None` if the node producing it never ran).
    pub tensors: Vec<Option<DenseND<T>>>,
    /// The inputs each node consumed, in node order.
    pub node_inputs: Vec<Vec<DenseND<T>>>,
}

/// Gradients produced by [`TlAutodiff::backward`], one slot per graph tensor.
#[derive(Debug, Clone)]
pub struct GradientTape<T>
where
    T: AdScalar,
{
    names: Vec<String>,
    gradients: Vec<Option<DenseND<T>>>,
}

impl<T> GradientTape<T>
where
    T: AdScalar,
{
    /// Gradient of the graph tensor at `index`.
    pub fn grad_by_index(&self, index: usize) -> Option<&DenseND<T>> {
        self.gradients.get(index).and_then(Option::as_ref)
    }

    /// Gradient of the graph tensor called `name`.
    pub fn grad(&self, name: &str) -> Option<&DenseND<T>> {
        let index = self.names.iter().position(|candidate| candidate == name)?;
        self.grad_by_index(index)
    }

    /// Names of the graph tensors, in graph order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Number of tensors that received a gradient.
    pub fn len(&self) -> usize {
        self.gradients.iter().filter(|g| g.is_some()).count()
    }

    /// Did any tensor receive a gradient?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// A Tensorlogic executor backed by TenRSo tensors and the TenRSo op registry.
#[derive(Debug)]
pub struct TenrsoTlExecutor<T>
where
    T: AdScalar,
{
    registry: OpRegistry<T>,
    tensors: HashMap<String, DenseND<T>>,
    tape: Option<ForwardTape<T>>,
}

impl<T> Default for TenrsoTlExecutor<T>
where
    T: AdScalar,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TenrsoTlExecutor<T>
where
    T: AdScalar,
{
    /// Create an executor whose registry holds every built-in TenRSo op.
    pub fn new() -> Self {
        Self::with_registry(OpRegistry::with_builtins())
    }

    /// Create an executor over a caller-supplied registry (e.g. one carrying
    /// extra, application-specific rules).
    pub fn with_registry(registry: OpRegistry<T>) -> Self {
        Self {
            registry,
            tensors: HashMap::new(),
            tape: None,
        }
    }

    /// Bind a named tensor for graph execution.
    pub fn bind_tensor(&mut self, name: impl Into<String>, tensor: DenseND<T>) {
        self.tensors.insert(name.into(), tensor);
    }

    /// Look up a bound tensor.
    pub fn tensor(&self, name: &str) -> Option<&DenseND<T>> {
        self.tensors.get(name)
    }

    /// Drop every bound tensor.
    pub fn clear_tensors(&mut self) {
        self.tensors.clear();
    }

    /// The op registry backing this executor.
    pub fn registry(&self) -> &OpRegistry<T> {
        &self.registry
    }

    /// Mutable access to the registry (to add custom ops).
    pub fn registry_mut(&mut self) -> &mut OpRegistry<T> {
        &mut self.registry
    }

    /// The forward tape recorded by the last [`TlAutodiff::forward`] call.
    pub fn forward_tape(&self) -> Option<&ForwardTape<T>> {
        self.tape.as_ref()
    }

    /// Resolve a graph tensor name to a bound tensor or a `const_<value>` literal.
    fn resolve(&self, name: &str) -> Option<DenseND<T>> {
        if let Some(tensor) = self.tensors.get(name) {
            return Some(tensor.clone());
        }

        // Tensorlogic graphs may carry index annotations, e.g. `age[a]`.
        let base = name.split('[').next().unwrap_or(name);
        if let Some(tensor) = self.tensors.get(base) {
            return Some(tensor.clone());
        }

        // Constant literals, e.g. `const_0.5`.
        let literal = base.strip_prefix("const_")?;
        let value = literal.parse::<f64>().ok()?;
        let value = T::from(value)?;
        Some(DenseND::from_elem(&[], value))
    }

    /// Single-output tensor index of a node.
    fn node_output(
        node: &tensorlogic_ir::EinsumNode,
        index: usize,
    ) -> Result<usize, ExecutorError> {
        match node.outputs.len() {
            1 => Ok(node.outputs[0]),
            n => Err(ExecutorError::UnsupportedOperation(format!(
                "node {index} ({}) has {n} outputs; the TenRSo bridge implements single-output \
                 operations only",
                node.operation_description()
            ))),
        }
    }

    /// Evaluate one graph node.
    fn eval_node(
        &mut self,
        node: &tensorlogic_ir::EinsumNode,
        inputs: &[DenseND<T>],
    ) -> Result<DenseND<T>, ExecutorError> {
        match &node.op {
            OpType::Einsum { spec } => self.einsum(spec, inputs),
            OpType::ElemUnary { op } => {
                if inputs.len() != 1 {
                    return Err(ExecutorError::InvalidInput(format!(
                        "unary op '{op}' expects 1 input, got {}",
                        inputs.len()
                    )));
                }
                let elem_op = parse_unary_op(op)?;
                self.elem_op(elem_op, &inputs[0])
            }
            OpType::ElemBinary { op } => {
                if inputs.len() != 2 {
                    return Err(ExecutorError::InvalidInput(format!(
                        "binary op '{op}' expects 2 inputs, got {}",
                        inputs.len()
                    )));
                }
                let elem_op = parse_binary_op(op)?;
                self.elem_op_binary(elem_op, &inputs[0], &inputs[1])
            }
            OpType::Reduce { op, axes } => {
                if inputs.len() != 1 {
                    return Err(ExecutorError::InvalidInput(format!(
                        "reduction '{op}' expects 1 input, got {}",
                        inputs.len()
                    )));
                }
                let reduce_op = parse_reduce_op(op)?;
                self.reduce(reduce_op, &inputs[0], axes)
            }
        }
    }

    /// VJP of one graph node: gradients w.r.t. its inputs, in input order.
    fn node_vjp(
        &self,
        node: &tensorlogic_ir::EinsumNode,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
    ) -> Result<Vec<DenseND<T>>, ExecutorError> {
        match &node.op {
            OpType::Einsum { spec } => self
                .registry
                .vjp("einsum", inputs, output_grad, &OpParams::einsum(spec))
                .map_err(|e| exec_error(&format!("einsum '{spec}' backward"), e)),

            OpType::ElemUnary { op } => {
                let name = unary_registry_name(parse_unary_op(op)?)?;
                self.registry
                    .vjp(name, inputs, output_grad, &OpParams::none())
                    .map_err(|e| exec_error(&format!("'{op}' backward"), e))
            }

            OpType::ElemBinary { op } => {
                let name = binary_registry_name(parse_binary_op(op)?)?;
                if inputs.len() != 2 {
                    return Err(ExecutorError::InvalidInput(format!(
                        "binary op '{op}' expects 2 inputs, got {}",
                        inputs.len()
                    )));
                }

                let (x, y, x_broadcast, y_broadcast) = align_binary(&inputs[0], &inputs[1])?;
                let grads = self
                    .registry
                    .vjp(name, &[x, y], output_grad, &OpParams::none())
                    .map_err(|e| exec_error(&format!("'{op}' backward"), e))?;

                // Undo the scalar broadcast performed in the forward pass.
                let mut grads = grads;
                if y_broadcast {
                    grads[1] = sum_to_scalar(&grads[1]);
                }
                if x_broadcast {
                    grads[0] = sum_to_scalar(&grads[0]);
                }
                Ok(grads)
            }

            OpType::Reduce { op, axes } => {
                let name = reduce_registry_name(parse_reduce_op(op)?);
                self.registry
                    .vjp(name, inputs, output_grad, &OpParams::axes(axes.clone()))
                    .map_err(|e| exec_error(&format!("'{op}' backward"), e))
            }
        }
    }
}

impl<T> TlExecutor for TenrsoTlExecutor<T>
where
    T: AdScalar,
{
    type Tensor = DenseND<T>;
    type Error = ExecutorError;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error> {
        if inputs.is_empty() {
            return Err(ExecutorError::EmptyInput(
                "einsum requires at least one operand".to_string(),
            ));
        }
        self.registry
            .forward("einsum", inputs, &OpParams::einsum(spec))
            .map_err(|e| ExecutorError::InvalidEinsumSpec(format!("einsum '{spec}': {e:#}")))
    }

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        let name = unary_registry_name(op)?;
        self.registry
            .forward(name, std::slice::from_ref(x), &OpParams::none())
            .map_err(|e| exec_error(&format!("{op:?}"), e))
    }

    fn elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        let name = binary_registry_name(op)?;
        let (x, y, _, _) = align_binary(x, y)?;
        self.registry
            .forward(name, &[x, y], &OpParams::none())
            .map_err(|e| exec_error(&format!("{op:?}"), e))
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &Self::Tensor,
        axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error> {
        let name = reduce_registry_name(op);
        self.registry
            .forward(
                name,
                std::slice::from_ref(x),
                &OpParams::axes(axes.to_vec()),
            )
            .map_err(|e| exec_error(&format!("{op:?}"), e))
    }
}

impl<T> TlAutodiff for TenrsoTlExecutor<T>
where
    T: AdScalar,
{
    type Tape = GradientTape<T>;

    fn forward(&mut self, graph: &EinsumGraph) -> Result<Self::Tensor, Self::Error> {
        if graph.nodes.is_empty() {
            return Err(ExecutorError::GraphValidationError(
                "graph has no nodes".to_string(),
            ));
        }
        if graph.outputs.is_empty() {
            return Err(ExecutorError::GraphValidationError(
                "graph declares no output tensor".to_string(),
            ));
        }

        let mut computed: Vec<Option<DenseND<T>>> = vec![None; graph.tensors.len()];
        for (index, name) in graph.tensors.iter().enumerate() {
            computed[index] = self.resolve(name);
        }

        let mut node_inputs: Vec<Vec<DenseND<T>>> = Vec::with_capacity(graph.nodes.len());

        for (node_index, node) in graph.nodes.iter().enumerate() {
            let output_index = Self::node_output(node, node_index)?;

            let mut inputs = Vec::with_capacity(node.inputs.len());
            for &tensor_index in &node.inputs {
                let tensor = computed
                    .get(tensor_index)
                    .and_then(Option::as_ref)
                    .cloned()
                    .ok_or_else(|| {
                        ExecutorError::TensorNotFound(format!(
                            "tensor '{}' (index {tensor_index}) is not bound and is not produced \
                             by an earlier node; bind it with bind_tensor()",
                            graph
                                .tensors
                                .get(tensor_index)
                                .map(String::as_str)
                                .unwrap_or("<out of range>")
                        ))
                    })?;
                inputs.push(tensor);
            }

            node_inputs.push(inputs.clone());
            let result = self.eval_node(node, &inputs)?;

            if output_index >= computed.len() {
                return Err(ExecutorError::GraphValidationError(format!(
                    "node {node_index} writes tensor {output_index}, which is out of range"
                )));
            }
            computed[output_index] = Some(result);
        }

        let output_index = graph.outputs[0];
        let output = computed
            .get(output_index)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or_else(|| {
                ExecutorError::TensorNotFound(format!(
                    "graph output (tensor index {output_index}) was never computed"
                ))
            })?;

        self.tape = Some(ForwardTape {
            tensors: computed,
            node_inputs,
        });

        Ok(output)
    }

    fn backward(
        &mut self,
        graph: &EinsumGraph,
        loss_grad: &Self::Tensor,
    ) -> Result<Self::Tape, Self::Error> {
        let tape = self.tape.clone().ok_or_else(|| {
            ExecutorError::InvalidInput(
                "backward() requires a forward() pass on the same graph first".to_string(),
            )
        })?;

        if tape.node_inputs.len() != graph.nodes.len() || tape.tensors.len() != graph.tensors.len()
        {
            return Err(ExecutorError::GraphValidationError(
                "the recorded forward tape does not match this graph; re-run forward()".to_string(),
            ));
        }
        if graph.outputs.is_empty() {
            return Err(ExecutorError::GraphValidationError(
                "graph declares no output tensor".to_string(),
            ));
        }

        let output_index = graph.outputs[0];
        let output_value = tape
            .tensors
            .get(output_index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                ExecutorError::TensorNotFound(format!(
                    "graph output (tensor index {output_index}) is missing from the forward tape"
                ))
            })?;

        if output_value.shape() != loss_grad.shape() {
            return Err(ExecutorError::ShapeMismatch(format!(
                "the cotangent has shape {:?} but the graph output has shape {:?}",
                loss_grad.shape(),
                output_value.shape()
            )));
        }

        let mut gradients: Vec<Option<DenseND<T>>> = vec![None; graph.tensors.len()];
        gradients[output_index] = Some(loss_grad.clone());

        for (node_index, node) in graph.nodes.iter().enumerate().rev() {
            let node_output = Self::node_output(node, node_index)?;

            let output_grad = match &gradients[node_output] {
                Some(grad) => grad.clone(),
                // This node does not lie on the path to the seeded output.
                None => continue,
            };

            let inputs = &tape.node_inputs[node_index];
            let input_grads = self.node_vjp(node, inputs, &output_grad)?;

            if input_grads.len() != node.inputs.len() {
                return Err(ExecutorError::InvalidInput(format!(
                    "node {node_index} ({}) produced {} gradients for {} inputs",
                    node.operation_description(),
                    input_grads.len(),
                    node.inputs.len()
                )));
            }

            for (&tensor_index, grad) in node.inputs.iter().zip(input_grads.into_iter()) {
                accumulate(&mut gradients[tensor_index], grad)?;
            }
        }

        Ok(GradientTape {
            names: graph.tensors.clone(),
            gradients,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::EinsumNode;

    fn matrix(values: Vec<f64>, shape: &[usize]) -> DenseND<f64> {
        DenseND::from_vec(values, shape).expect("valid tensor")
    }

    #[test]
    fn test_conversion_roundtrip() {
        let tensor = matrix(vec![1.5, -2.5, 3.0, 4.25, 5.0, 6.0], &[2, 3]);

        let tl = dense_to_tl(&tensor).expect("to tensorlogic");
        assert_eq!(tl.shape(), &[2, 3]);
        assert_eq!(tl[[1, 2]], 6.0);

        let back: DenseND<f64> = tl_to_dense(&tl).expect("from tensorlogic");
        assert_eq!(back.shape(), tensor.shape());
        assert_eq!(back.as_slice(), tensor.as_slice());

        let tl32 = dense_to_tl_f32(&tensor).expect("to tensorlogic f32");
        let back32: DenseND<f32> = tl_f32_to_dense(&tl32).expect("from tensorlogic f32");
        assert_eq!(back32.shape(), tensor.shape());
        assert!((back32.as_slice()[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_tl_executor_einsum() {
        let mut exec = TenrsoTlExecutor::<f64>::new();
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = matrix(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let c = exec.einsum("ij,jk->ik", &[a, b]).expect("einsum");
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_tl_executor_elem_and_reduce() {
        let mut exec = TenrsoTlExecutor::<f64>::new();
        let x = matrix(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]);

        let relu = exec.elem_op(ElemOp::Relu, &x).expect("relu");
        assert_eq!(relu.as_slice(), &[0.0, 2.0, 0.0, 4.0]);

        let y = matrix(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let sum = exec
            .elem_op_binary(ElemOp::Add, &x, &y)
            .expect("elementwise add");
        assert_eq!(sum.as_slice(), &[0.0, 3.0, -2.0, 5.0]);

        // Rank-0 broadcast, exactly like the reference backend.
        let scalar = DenseND::from_elem(&[], 2.0);
        let scaled = exec
            .elem_op_binary(ElemOp::Multiply, &x, &scalar)
            .expect("scalar broadcast");
        assert_eq!(scaled.as_slice(), &[-2.0, 4.0, -6.0, 8.0]);

        let reduced = exec
            .reduce(ReduceOp::Sum, &x, &[1])
            .expect("row-wise reduction");
        assert_eq!(reduced.as_slice(), &[1.0, 1.0]);
    }

    #[test]
    fn test_tl_executor_rejects_unknown_ops() {
        let mut exec = TenrsoTlExecutor::<f64>::new();
        let x = matrix(vec![1.0], &[1]);

        // A binary op handed to elem_op must not silently do something else.
        let err = exec.elem_op(ElemOp::Add, &x).unwrap_err();
        assert!(matches!(err, ExecutorError::UnsupportedOperation(_)));

        let err = exec.elem_op_binary(ElemOp::Relu, &x, &x).unwrap_err();
        assert!(matches!(err, ExecutorError::UnsupportedOperation(_)));
    }

    /// `y = relu(A @ B)`, then `loss = sum(y)`.
    fn matmul_relu_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("a");
        let b = graph.add_tensor("b");
        let ab = graph.add_tensor("ab");
        let y = graph.add_tensor("y");

        graph
            .nodes
            .push(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![ab]));
        graph.nodes.push(EinsumNode::elem_unary("relu", ab, y));
        graph.inputs = vec![a, b];
        graph.outputs = vec![y];
        graph
    }

    #[test]
    fn test_autodiff_forward_and_backward() {
        let graph = matmul_relu_graph();
        let mut exec = TenrsoTlExecutor::<f64>::new();

        let a = matrix(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = matrix(vec![1.0, -1.0, -1.0, 1.0], &[2, 2]);
        exec.bind_tensor("a", a.clone());
        exec.bind_tensor("b", b.clone());

        // A @ B = [[-1, 1], [-1, 1]]  ->  relu -> [[0, 1], [0, 1]]
        let y = exec.forward(&graph).expect("forward");
        assert_eq!(y.as_slice(), &[0.0, 1.0, 0.0, 1.0]);

        let seed = DenseND::<f64>::ones(&[2, 2]);
        let tape = exec.backward(&graph, &seed).expect("backward");

        // d(relu)/d(ab) masks the negative column, so grad_ab = [[0, 1], [0, 1]].
        // grad_a = grad_ab @ B^T = [[-1, 1], [-1, 1]]
        let grad_a = tape.grad("a").expect("grad for a");
        assert_eq!(grad_a.as_slice(), &[-1.0, 1.0, -1.0, 1.0]);

        // grad_b = A^T @ grad_ab = [[0, 4], [0, 6]]
        let grad_b = tape.grad("b").expect("grad for b");
        assert_eq!(grad_b.as_slice(), &[0.0, 4.0, 0.0, 6.0]);

        assert_eq!(tape.names().len(), 4);
        assert!(!tape.is_empty());
    }

    #[test]
    fn test_autodiff_backward_matches_finite_differences() {
        use crate::gradcheck::{check_gradient, GradCheckConfig};

        let graph = matmul_relu_graph();
        let b = matrix(vec![0.5, -1.25, 2.0, 0.75], &[2, 2]);
        let seed = DenseND::from_vec(vec![1.0, 0.5, -0.25, 2.0], &[2, 2]).expect("seed");

        let forward = |a: &DenseND<f64>| -> anyhow::Result<DenseND<f64>> {
            let mut exec = TenrsoTlExecutor::<f64>::new();
            exec.bind_tensor("a", a.clone());
            exec.bind_tensor("b", b.clone());
            exec.forward(&graph).map_err(|e| anyhow!("{e}"))
        };

        let backward = |a: &DenseND<f64>, grad: &DenseND<f64>| -> anyhow::Result<DenseND<f64>> {
            let mut exec = TenrsoTlExecutor::<f64>::new();
            exec.bind_tensor("a", a.clone());
            exec.bind_tensor("b", b.clone());
            exec.forward(&graph).map_err(|e| anyhow!("{e}"))?;
            let tape = exec.backward(&graph, grad).map_err(|e| anyhow!("{e}"))?;
            tape.grad("a")
                .cloned()
                .ok_or_else(|| anyhow!("no gradient for 'a'"))
        };

        // Values chosen away from the relu kink so the derivative is defined.
        let a = matrix(vec![1.0, 2.0, -1.5, 3.0], &[2, 2]);
        let result = check_gradient(forward, backward, &a, &seed, &GradCheckConfig::default())
            .expect("gradcheck");

        assert!(
            result.passed,
            "Tensorlogic graph gradient failed gradcheck: max_abs={:.3e} max_rel={:.3e}",
            result.max_abs_diff, result.max_rel_diff
        );
    }

    #[test]
    fn test_backward_without_forward_is_an_error() {
        let graph = matmul_relu_graph();
        let mut exec = TenrsoTlExecutor::<f64>::new();
        let seed = DenseND::<f64>::ones(&[2, 2]);

        let err = exec.backward(&graph, &seed).unwrap_err();
        assert!(matches!(err, ExecutorError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn test_unbound_tensor_is_an_error_not_a_guess() {
        let graph = matmul_relu_graph();
        let mut exec = TenrsoTlExecutor::<f64>::new();
        exec.bind_tensor("a", matrix(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        // 'b' deliberately left unbound.

        let err = exec.forward(&graph).unwrap_err();
        assert!(matches!(err, ExecutorError::TensorNotFound(_)), "{err:?}");
    }

    #[test]
    fn test_const_literal_binding() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("x");
        let c = graph.add_tensor("const_2.5");
        let y = graph.add_tensor("y");
        graph.nodes.push(EinsumNode::elem_binary("mul", x, c, y));
        graph.inputs = vec![x];
        graph.outputs = vec![y];

        let mut exec = TenrsoTlExecutor::<f64>::new();
        exec.bind_tensor("x", matrix(vec![1.0, 2.0], &[2]));

        let y_value = exec.forward(&graph).expect("forward");
        assert_eq!(y_value.as_slice(), &[2.5, 5.0]);

        let tape = exec
            .backward(&graph, &DenseND::<f64>::ones(&[2]))
            .expect("backward");
        assert_eq!(tape.grad("x").expect("grad x").as_slice(), &[2.5, 2.5]);

        // The scalar constant's gradient is summed back to rank 0: 1*1 + 1*2 = 3.
        let grad_c = tape.grad("const_2.5").expect("grad const");
        assert!(grad_c.shape().is_empty());
        assert_eq!(grad_c.as_slice(), &[3.0]);
    }

    #[test]
    fn test_reduction_graph_backward() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("x");
        let y = graph.add_tensor("y");
        graph.nodes.push(EinsumNode::reduce("max", vec![1], x, y));
        graph.inputs = vec![x];
        graph.outputs = vec![y];

        let mut exec = TenrsoTlExecutor::<f64>::new();
        exec.bind_tensor("x", matrix(vec![1.0, 5.0, 3.0, 2.0, 9.0, 4.0], &[2, 3]));

        let y_value = exec.forward(&graph).expect("forward");
        assert_eq!(y_value.as_slice(), &[5.0, 9.0]);

        let tape = exec
            .backward(&graph, &matrix(vec![1.0, 10.0], &[2]))
            .expect("backward");
        // Gradient flows only to the arg-max of each row.
        assert_eq!(
            tape.grad("x").expect("grad x").as_slice(),
            &[0.0, 1.0, 0.0, 0.0, 10.0, 0.0]
        );
    }

    #[test]
    fn test_unsupported_node_op_is_rejected() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("x");
        let y = graph.add_tensor("y");
        graph.nodes.push(EinsumNode::elem_unary("next", x, y));
        graph.inputs = vec![x];
        graph.outputs = vec![y];

        let mut exec = TenrsoTlExecutor::<f64>::new();
        exec.bind_tensor("x", matrix(vec![1.0, 2.0], &[2]));

        let err = exec.forward(&graph).unwrap_err();
        assert!(
            matches!(err, ExecutorError::UnsupportedOperation(_)),
            "{err:?}"
        );
    }
}
