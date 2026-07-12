//! Graph-based automatic differentiation with dynamic computation graphs.
//!
//! This module provides a PyTorch-style computation graph and tape-based automatic
//! differentiation system. Unlike the explicit VJP approach, this system automatically
//! records operations and constructs the computation graph during the forward pass.
//!
//! # Features
//!
//! - **Automatic graph construction**: Operations are automatically recorded during forward pass
//! - **Dynamic control flow**: Supports conditionals, loops, and runtime-dependent operations
//! - **Efficient backward pass**: Graph traversal in topological order with gradient accumulation
//! - **Memory management**: Automatic cleanup of intermediate values
//! - **Graph optimization**: Dead code elimination and operation fusion
//!
//! # Example
//!
//! ```rust,ignore
//! use tenrso_ad::graph::{ComputationGraph, Variable};
//! use scirs2_core::ndarray_ext::array;
//!
//! // Create computation graph
//! let mut graph = ComputationGraph::new();
//!
//! // Create variables (with gradient tracking enabled)
//! let x = graph.variable(array![2.0, 3.0], true)?;
//! let y = graph.variable(array![4.0, 5.0], true)?;
//!
//! // Forward pass - operations are automatically recorded
//! let z = graph.add(&x, &y)?;
//! let w = graph.mul(&z, &x)?;
//! let loss = graph.sum(&w)?;
//!
//! // Backward pass - compute all gradients
//! graph.backward(&loss)?;
//!
//! // Access gradients
//! let grad_x = graph.gradient(&x)?;
//! let grad_y = graph.gradient(&y)?;
//! ```

use anyhow::{anyhow, Context, Result};
use scirs2_core::ndarray_ext::{ArrayD, Axis, Ix2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

/// Acquire a mutex guard, transparently recovering from poisoning.
///
/// The internal mutexes in this module only protect data structures that are
/// refreshed per-operation (the node map, next-id counter, recording flag), so a
/// poisoned mutex carries no invariant that would make continued use unsafe.
/// Returning the inner guard instead of panicking keeps the whole graph API
/// panic-free even if a worker thread previously panicked while holding a lock.
#[inline]
fn lock_mutex<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    match m.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Unique identifier for a node in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Operation type in the computation graph
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// Input variable (leaf node)
    Input,
    /// Addition: z = x + y
    Add { lhs: NodeId, rhs: NodeId },
    /// Subtraction: z = x - y
    Sub { lhs: NodeId, rhs: NodeId },
    /// Multiplication: z = x * y (element-wise)
    Mul { lhs: NodeId, rhs: NodeId },
    /// Division: z = x / y (element-wise)
    Div { lhs: NodeId, rhs: NodeId },
    /// Matrix multiplication: z = x @ y
    MatMul { lhs: NodeId, rhs: NodeId },
    /// Negation: z = -x
    Neg { input: NodeId },
    /// Exponential: z = exp(x)
    Exp { input: NodeId },
    /// Natural logarithm: z = log(x)
    Log { input: NodeId },
    /// Power: z = x^n
    Pow { input: NodeId, exponent: f64 },
    /// Sum reduction: z = sum(x, axis)
    Sum { input: NodeId, axis: Option<usize> },
    /// Mean reduction: z = mean(x, axis)
    Mean { input: NodeId, axis: Option<usize> },
    /// Reshape: z = reshape(x, new_shape)
    Reshape {
        input: NodeId,
        old_shape: Vec<usize>,
    },
    /// Transpose: z = transpose(x, axes)
    Transpose { input: NodeId, axes: Vec<usize> },
    /// Broadcast: z = broadcast(x, target_shape)
    Broadcast {
        input: NodeId,
        original_shape: Vec<usize>,
    },
    /// ReLU activation: z = max(0, x)
    ReLU { input: NodeId },
    /// Sigmoid activation: z = 1 / (1 + exp(-x))
    Sigmoid { input: NodeId },
    /// Tanh activation: z = tanh(x)
    Tanh { input: NodeId },
    /// Slice operation: z = x[slice]
    Slice {
        input: NodeId,
        ranges: Vec<(usize, usize)>,
    },
}

/// Node in the computation graph
#[derive(Clone)]
struct GraphNode<T> {
    /// Unique identifier
    #[allow(dead_code)]
    id: NodeId,
    /// Operation that produced this node
    operation: Operation,
    /// Current value (Some during forward pass, None after backward to save memory)
    value: Option<ArrayD<T>>,
    /// Accumulated gradient
    gradient: Option<ArrayD<T>>,
    /// Whether to track gradients for this node
    requires_grad: bool,
    /// Parent nodes (inputs to this operation)
    parents: Vec<NodeId>,
    /// Child nodes (operations that use this node)
    children: Vec<NodeId>,
}

impl<T: Float> GraphNode<T> {
    fn new(
        id: NodeId,
        operation: Operation,
        value: ArrayD<T>,
        requires_grad: bool,
        parents: Vec<NodeId>,
    ) -> Self {
        Self {
            id,
            operation,
            value: Some(value),
            gradient: None,
            requires_grad,
            parents,
            children: Vec::new(),
        }
    }

    /// Initialize gradient accumulator if needed
    fn init_gradient(&mut self, shape: &[usize]) -> Result<()> {
        if self.requires_grad && self.gradient.is_none() {
            self.gradient = Some(ArrayD::zeros(IxDyn(shape)));
        }
        Ok(())
    }

    /// Accumulate gradient
    fn accumulate_gradient(&mut self, grad: ArrayD<T>) -> Result<()> {
        if !self.requires_grad {
            return Ok(());
        }

        if let Some(ref mut current_grad) = self.gradient {
            // Add to existing gradient
            *current_grad = &*current_grad + &grad;
        } else {
            // Initialize with this gradient
            self.gradient = Some(grad);
        }
        Ok(())
    }
}

/// Variable reference in the computation graph
#[derive(Debug, Clone, Copy)]
pub struct Variable {
    id: NodeId,
}

impl Variable {
    fn new(id: NodeId) -> Self {
        Self { id }
    }

    /// Get the node ID
    pub fn id(&self) -> NodeId {
        self.id
    }
}

/// Computation graph for tape-based automatic differentiation
pub struct ComputationGraph<T: Float + ScalarOperand + FromPrimitive> {
    /// All nodes in the graph
    nodes: Arc<Mutex<HashMap<NodeId, GraphNode<T>>>>,
    /// Next available node ID
    next_id: Arc<Mutex<usize>>,
    /// Whether to record operations (training mode)
    recording: Arc<Mutex<bool>>,
}

impl<T: Float + ScalarOperand + FromPrimitive> Default for ComputationGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + ScalarOperand + FromPrimitive> ComputationGraph<T> {
    /// Create a new computation graph
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
            recording: Arc::new(Mutex::new(true)),
        }
    }

    /// Enable gradient recording (training mode)
    pub fn train(&self) {
        *lock_mutex(&self.recording) = true;
    }

    /// Disable gradient recording (inference mode)
    pub fn eval(&self) {
        *lock_mutex(&self.recording) = false;
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        *lock_mutex(&self.recording)
    }

    /// Clear all nodes and reset the graph
    pub fn clear(&self) {
        lock_mutex(&self.nodes).clear();
        *lock_mutex(&self.next_id) = 0;
    }

    /// Get next available node ID
    fn allocate_id(&self) -> NodeId {
        let mut next_id = lock_mutex(&self.next_id);
        let id = NodeId(*next_id);
        *next_id += 1;
        id
    }

    /// Create a variable (input node)
    pub fn variable(&self, value: ArrayD<T>, requires_grad: bool) -> Result<Variable> {
        let id = self.allocate_id();
        let node = GraphNode::new(id, Operation::Input, value, requires_grad, vec![]);

        lock_mutex(&self.nodes).insert(id, node);
        Ok(Variable::new(id))
    }

    /// Create a constant (non-differentiable input)
    pub fn constant(&self, value: ArrayD<T>) -> Result<Variable> {
        self.variable(value, false)
    }

    /// Add an operation node to the graph
    fn add_node(
        &self,
        operation: Operation,
        value: ArrayD<T>,
        parents: Vec<NodeId>,
    ) -> Result<Variable> {
        let id = self.allocate_id();

        // Check if any parent requires gradients
        let requires_grad = if *lock_mutex(&self.recording) {
            let nodes = lock_mutex(&self.nodes);
            parents
                .iter()
                .any(|&parent_id| nodes.get(&parent_id).is_some_and(|n| n.requires_grad))
        } else {
            false
        };

        let node = GraphNode::new(id, operation, value, requires_grad, parents.clone());

        // Update parent nodes to add this as a child
        {
            let mut nodes = lock_mutex(&self.nodes);
            for parent_id in &parents {
                if let Some(parent) = nodes.get_mut(parent_id) {
                    parent.children.push(id);
                }
            }
            nodes.insert(id, node);
        }

        Ok(Variable::new(id))
    }

    /// Get the value of a variable
    pub fn value(&self, var: &Variable) -> Result<ArrayD<T>> {
        let nodes = lock_mutex(&self.nodes);
        let node = nodes
            .get(&var.id)
            .ok_or_else(|| anyhow!("Variable not found in graph"))?;
        node.value
            .clone()
            .ok_or_else(|| anyhow!("Value has been freed from memory"))
    }

    /// Get the gradient of a variable
    pub fn gradient(&self, var: &Variable) -> Result<ArrayD<T>> {
        let nodes = lock_mutex(&self.nodes);
        let node = nodes
            .get(&var.id)
            .ok_or_else(|| anyhow!("Variable not found in graph"))?;
        node.gradient
            .clone()
            .ok_or_else(|| anyhow!("No gradient available for this variable"))
    }

    /// Check if a variable has a gradient
    pub fn has_gradient(&self, var: &Variable) -> bool {
        let nodes = lock_mutex(&self.nodes);
        nodes
            .get(&var.id)
            .is_some_and(|node| node.gradient.is_some())
    }

    /// Zero all gradients in the graph
    pub fn zero_grad(&self) {
        let mut nodes = lock_mutex(&self.nodes);
        for node in nodes.values_mut() {
            node.gradient = None;
        }
    }

    // ===== Operations =====

    /// Addition: z = x + y
    pub fn add(&self, lhs: &Variable, rhs: &Variable) -> Result<Variable> {
        let lhs_val = self.value(lhs)?;
        let rhs_val = self.value(rhs)?;
        let result = &lhs_val + &rhs_val;

        self.add_node(
            Operation::Add {
                lhs: lhs.id,
                rhs: rhs.id,
            },
            result,
            vec![lhs.id, rhs.id],
        )
    }

    /// Subtraction: z = x - y
    pub fn sub(&self, lhs: &Variable, rhs: &Variable) -> Result<Variable> {
        let lhs_val = self.value(lhs)?;
        let rhs_val = self.value(rhs)?;
        let result = &lhs_val - &rhs_val;

        self.add_node(
            Operation::Sub {
                lhs: lhs.id,
                rhs: rhs.id,
            },
            result,
            vec![lhs.id, rhs.id],
        )
    }

    /// Element-wise multiplication: z = x * y
    pub fn mul(&self, lhs: &Variable, rhs: &Variable) -> Result<Variable> {
        let lhs_val = self.value(lhs)?;
        let rhs_val = self.value(rhs)?;
        let result = &lhs_val * &rhs_val;

        self.add_node(
            Operation::Mul {
                lhs: lhs.id,
                rhs: rhs.id,
            },
            result,
            vec![lhs.id, rhs.id],
        )
    }

    /// Element-wise division: z = x / y
    pub fn div(&self, lhs: &Variable, rhs: &Variable) -> Result<Variable> {
        let lhs_val = self.value(lhs)?;
        let rhs_val = self.value(rhs)?;
        let result = &lhs_val / &rhs_val;

        self.add_node(
            Operation::Div {
                lhs: lhs.id,
                rhs: rhs.id,
            },
            result,
            vec![lhs.id, rhs.id],
        )
    }

    /// Matrix multiplication: z = x @ y
    pub fn matmul(&self, lhs: &Variable, rhs: &Variable) -> Result<Variable> {
        let lhs_val = self.value(lhs)?;
        let rhs_val = self.value(rhs)?;

        // For simplicity, only support 2D matrices for now
        if lhs_val.ndim() != 2 || rhs_val.ndim() != 2 {
            return Err(anyhow!(
                "MatMul only supports 2D matrices, got shapes {:?} and {:?}",
                lhs_val.shape(),
                rhs_val.shape()
            ));
        }

        let lhs_2d = lhs_val.view().into_dimensionality::<Ix2>()?;
        let rhs_2d = rhs_val.view().into_dimensionality::<Ix2>()?;
        let result_2d = lhs_2d.dot(&rhs_2d);
        let result = result_2d.into_dyn();

        self.add_node(
            Operation::MatMul {
                lhs: lhs.id,
                rhs: rhs.id,
            },
            result,
            vec![lhs.id, rhs.id],
        )
    }

    /// Negation: z = -x
    pub fn neg(&self, input: &Variable) -> Result<Variable> {
        let input_val = self.value(input)?;
        let result = input_val.mapv(|x| -x);

        self.add_node(Operation::Neg { input: input.id }, result, vec![input.id])
    }

    /// Exponential: z = exp(x)
    pub fn exp(&self, input: &Variable) -> Result<Variable> {
        let input_val = self.value(input)?;
        let result = input_val.mapv(|x| x.exp());

        self.add_node(Operation::Exp { input: input.id }, result, vec![input.id])
    }

    /// Natural logarithm: z = log(x)
    pub fn log(&self, input: &Variable) -> Result<Variable> {
        let input_val = self.value(input)?;
        let result = input_val.mapv(|x| x.ln());

        self.add_node(Operation::Log { input: input.id }, result, vec![input.id])
    }

    /// Power: z = x^n
    pub fn pow(&self, input: &Variable, exponent: f64) -> Result<Variable> {
        let input_val = self.value(input)?;
        let exp_t = T::from(exponent).ok_or_else(|| anyhow!("Failed to convert exponent"))?;
        let result = input_val.mapv(|x| x.powf(exp_t));

        self.add_node(
            Operation::Pow {
                input: input.id,
                exponent,
            },
            result,
            vec![input.id],
        )
    }

    /// Sum reduction: z = sum(x, axis)
    pub fn sum(&self, input: &Variable) -> Result<Variable> {
        self.sum_axis(input, None)
    }

    /// Sum along specific axis
    pub fn sum_axis(&self, input: &Variable, axis: Option<usize>) -> Result<Variable> {
        let input_val = self.value(input)?;

        let result = if let Some(ax) = axis {
            input_val.sum_axis(Axis(ax))
        } else {
            let sum_scalar = input_val.iter().fold(T::zero(), |acc, &x| acc + x);
            ArrayD::from_elem(IxDyn(&[]), sum_scalar)
        };

        self.add_node(
            Operation::Sum {
                input: input.id,
                axis,
            },
            result,
            vec![input.id],
        )
    }

    /// Mean reduction: z = mean(x, axis)
    pub fn mean(&self, input: &Variable) -> Result<Variable> {
        self.mean_axis(input, None)
    }

    /// Mean along specific axis
    pub fn mean_axis(&self, input: &Variable, axis: Option<usize>) -> Result<Variable> {
        let input_val = self.value(input)?;

        let result = if let Some(ax) = axis {
            input_val
                .mean_axis(Axis(ax))
                .ok_or_else(|| anyhow!("Mean computation failed"))?
        } else {
            let sum_scalar = input_val.iter().fold(T::zero(), |acc, &x| acc + x);
            let n = T::from(input_val.len()).ok_or_else(|| anyhow!("Failed to convert length"))?;
            let mean_scalar = sum_scalar / n;
            ArrayD::from_elem(IxDyn(&[]), mean_scalar)
        };

        self.add_node(
            Operation::Mean {
                input: input.id,
                axis,
            },
            result,
            vec![input.id],
        )
    }

    /// ReLU activation: z = max(0, x)
    pub fn relu(&self, input: &Variable) -> Result<Variable> {
        let input_val = self.value(input)?;
        let result = input_val.mapv(|x| if x > T::zero() { x } else { T::zero() });

        self.add_node(Operation::ReLU { input: input.id }, result, vec![input.id])
    }

    /// Sigmoid activation: z = 1 / (1 + exp(-x))
    pub fn sigmoid(&self, input: &Variable) -> Result<Variable> {
        let input_val = self.value(input)?;
        let result = input_val.mapv(|x| T::one() / (T::one() + (-x).exp()));

        self.add_node(
            Operation::Sigmoid { input: input.id },
            result,
            vec![input.id],
        )
    }

    /// Tanh activation: z = tanh(x)
    pub fn tanh(&self, input: &Variable) -> Result<Variable> {
        let input_val = self.value(input)?;
        let result = input_val.mapv(|x| x.tanh());

        self.add_node(Operation::Tanh { input: input.id }, result, vec![input.id])
    }

    /// Reshape: z = reshape(x, new_shape)
    pub fn reshape(&self, input: &Variable, new_shape: &[usize]) -> Result<Variable> {
        let input_val = self.value(input)?;
        let old_shape = input_val.shape().to_vec();
        let result = input_val
            .to_shape(IxDyn(new_shape))
            .context("Reshape failed")?
            .to_owned();

        self.add_node(
            Operation::Reshape {
                input: input.id,
                old_shape,
            },
            result,
            vec![input.id],
        )
    }

    /// Transpose (permute axes): z = permute(x, axes)
    ///
    /// Permutes the tensor axes according to `axes`, which must be a permutation
    /// of `0..x.ndim()`. The backward pass applies the inverse permutation.
    ///
    /// # Complexity
    ///
    /// O(n) for the copy; O(ndim) for axis validation.
    pub fn transpose(&self, input: &Variable, axes: Vec<usize>) -> Result<Variable> {
        let input_val = self.value(input)?;
        let ndim = input_val.ndim();
        if axes.len() != ndim {
            return Err(anyhow!(
                "transpose: axes length {} != tensor ndim {}",
                axes.len(),
                ndim
            ));
        }
        let mut seen = vec![false; ndim];
        for &ax in &axes {
            if ax >= ndim {
                return Err(anyhow!(
                    "transpose: axis {} out of range for ndim {}",
                    ax,
                    ndim
                ));
            }
            if seen[ax] {
                return Err(anyhow!("transpose: duplicate axis {}", ax));
            }
            seen[ax] = true;
        }
        let result = input_val.view().permuted_axes(IxDyn(&axes)).to_owned();
        self.add_node(
            Operation::Transpose {
                input: input.id,
                axes,
            },
            result,
            vec![input.id],
        )
    }

    /// Broadcast: z = broadcast(x, target_shape)
    ///
    /// Broadcasts the input tensor to `target_shape` using NumPy broadcasting
    /// semantics — trailing dimensions are matched, leading singleton dimensions
    /// are expanded, and missing leading dimensions are implicitly 1.
    /// The original shape is stored in the graph for the backward pass.
    ///
    /// # Complexity
    ///
    /// O(n) for the copy where n = product(target_shape).
    pub fn broadcast(&self, input: &Variable, target_shape: &[usize]) -> Result<Variable> {
        let input_val = self.value(input)?;
        let original_shape = input_val.shape().to_vec();
        let result = input_val
            .broadcast(IxDyn(target_shape))
            .ok_or_else(|| {
                anyhow!(
                    "broadcast: cannot broadcast {:?} to {:?}",
                    original_shape,
                    target_shape
                )
            })?
            .to_owned();
        self.add_node(
            Operation::Broadcast {
                input: input.id,
                original_shape,
            },
            result,
            vec![input.id],
        )
    }

    /// Slice: z = x\[ranges\]
    ///
    /// Extracts a sub-tensor using per-axis half-open `(start, end)` ranges.
    /// All axes must satisfy `0 ≤ start ≤ end ≤ dim_size`. The backward pass
    /// scatters the output gradient back into a zeros tensor with the input shape.
    ///
    /// # Complexity
    ///
    /// O(n) where n = product(end_i − start_i).
    pub fn slice_nd(&self, input: &Variable, ranges: Vec<(usize, usize)>) -> Result<Variable> {
        let input_val = self.value(input)?;
        let ndim = input_val.ndim();
        if ranges.len() != ndim {
            return Err(anyhow!(
                "slice_nd: ranges length {} != tensor ndim {}",
                ranges.len(),
                ndim
            ));
        }
        for (ax, &(start, end)) in ranges.iter().enumerate() {
            let dim = input_val.shape()[ax];
            if end > dim || start > end {
                return Err(anyhow!(
                    "slice_nd: invalid range [{}, {}) for axis {} of size {}",
                    start,
                    end,
                    ax,
                    dim
                ));
            }
        }
        let new_shape: Vec<usize> = ranges.iter().map(|&(s, e)| e - s).collect();
        let flat_size: usize = new_shape.iter().product::<usize>();
        let mut result_flat: Vec<T> = Vec::with_capacity(flat_size);
        for out_flat in 0..flat_size {
            let mut out_idx = vec![0usize; ndim];
            let mut remaining = out_flat;
            for d in (0..ndim).rev() {
                if new_shape[d] > 0 {
                    out_idx[d] = remaining % new_shape[d];
                    remaining /= new_shape[d];
                }
            }
            let in_idx: Vec<usize> = out_idx
                .iter()
                .enumerate()
                .map(|(ax, &i)| ranges[ax].0 + i)
                .collect();
            result_flat.push(input_val[in_idx.as_slice()]);
        }
        let result = ArrayD::from_shape_vec(IxDyn(&new_shape), result_flat)
            .context("slice_nd: failed to construct output array")?;
        self.add_node(
            Operation::Slice {
                input: input.id,
                ranges,
            },
            result,
            vec![input.id],
        )
    }

    // ===== Backward Pass =====

    /// Perform backward pass from the given output node
    pub fn backward(&self, output: &Variable) -> Result<()> {
        let mut nodes = lock_mutex(&self.nodes);

        // Check that output is a scalar
        let output_node = nodes
            .get(&output.id)
            .ok_or_else(|| anyhow!("Output variable not found"))?;
        let output_shape = output_node
            .value
            .as_ref()
            .ok_or_else(|| anyhow!("Output value not available"))?
            .shape();

        if !output_shape.is_empty() && output_shape.iter().product::<usize>() != 1 {
            return Err(anyhow!(
                "Backward can only be called on scalar outputs, got shape {:?}",
                output_shape
            ));
        }

        // Initialize output gradient to 1
        let output_grad = ArrayD::from_elem(IxDyn(output_shape), T::one());
        nodes
            .get_mut(&output.id)
            .ok_or_else(|| anyhow!("Output variable not found"))?
            .gradient = Some(output_grad);

        // Compute topological order
        let topo_order = self.topological_sort_locked(&nodes, output.id)?;

        // Backward pass in reverse topological order
        for &node_id in topo_order.iter().rev() {
            let node = nodes
                .get(&node_id)
                .ok_or_else(|| anyhow!("Node {} not found", node_id))?;

            if !node.requires_grad {
                continue;
            }

            let grad_output = node
                .gradient
                .clone()
                .ok_or_else(|| anyhow!("No gradient for node {}", node_id))?;

            // Compute gradients for parent nodes based on operation type
            let parent_grads =
                self.compute_backward_locked(&nodes, &node.operation, &grad_output)?;

            // Accumulate gradients to parent nodes
            for (parent_id, parent_grad) in parent_grads {
                let parent = nodes
                    .get_mut(&parent_id)
                    .ok_or_else(|| anyhow!("Parent node {} not found", parent_id))?;

                if parent.requires_grad {
                    let parent_shape = parent
                        .value
                        .as_ref()
                        .ok_or_else(|| anyhow!("Parent value not available"))?
                        .shape()
                        .to_vec();
                    parent.init_gradient(&parent_shape)?;
                    parent.accumulate_gradient(parent_grad)?;
                }
            }
        }

        Ok(())
    }

    /// Topological sort starting from the given node (assumes nodes is already locked)
    fn topological_sort_locked(
        &self,
        nodes: &HashMap<NodeId, GraphNode<T>>,
        start: NodeId,
    ) -> Result<Vec<NodeId>> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        while let Some(node_id) = stack.pop() {
            if visited.contains(&node_id) {
                continue;
            }

            let node = nodes
                .get(&node_id)
                .ok_or_else(|| anyhow!("Node {} not found during topological sort", node_id))?;

            // Add parents to stack first
            let mut all_parents_visited = true;
            for &parent_id in &node.parents {
                if !visited.contains(&parent_id) {
                    stack.push(node_id); // Re-add current node
                    stack.push(parent_id); // Visit parent first
                    all_parents_visited = false;
                    break;
                }
            }

            if all_parents_visited {
                visited.insert(node_id);
                order.push(node_id);
            }
        }

        Ok(order)
    }

    /// Compute gradients for parent nodes (assumes nodes is already locked)
    fn compute_backward_locked(
        &self,
        nodes: &HashMap<NodeId, GraphNode<T>>,
        operation: &Operation,
        grad_output: &ArrayD<T>,
    ) -> Result<Vec<(NodeId, ArrayD<T>)>> {
        match operation {
            Operation::Input => Ok(vec![]),

            Operation::Add { lhs, rhs } => {
                // d/dx (x + y) = 1, d/dy (x + y) = 1
                Ok(vec![
                    (*lhs, grad_output.clone()),
                    (*rhs, grad_output.clone()),
                ])
            }

            Operation::Sub { lhs, rhs } => {
                // d/dx (x - y) = 1, d/dy (x - y) = -1
                let grad_rhs = grad_output.mapv(|x| -x);
                Ok(vec![(*lhs, grad_output.clone()), (*rhs, grad_rhs)])
            }

            Operation::Mul { lhs, rhs } => {
                // d/dx (x * y) = y, d/dy (x * y) = x
                let lhs_val = nodes
                    .get(lhs)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("LHS value not available for Mul backward"))?;
                let rhs_val = nodes
                    .get(rhs)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("RHS value not available for Mul backward"))?;

                let grad_lhs = grad_output * rhs_val;
                let grad_rhs = grad_output * lhs_val;
                Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
            }

            Operation::Div { lhs, rhs } => {
                // d/dx (x / y) = 1/y, d/dy (x / y) = -x/y^2
                let lhs_val = nodes
                    .get(lhs)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("LHS value not available for Div backward"))?;
                let rhs_val = nodes
                    .get(rhs)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("RHS value not available for Div backward"))?;

                let grad_lhs = grad_output / rhs_val;
                let grad_rhs = -(grad_output * lhs_val) / (rhs_val * rhs_val);
                Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
            }

            Operation::MatMul { lhs, rhs } => {
                // d/dx (x @ y) = grad_out @ y^T, d/dy (x @ y) = x^T @ grad_out
                let lhs_val = nodes
                    .get(lhs)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("LHS value not available for MatMul backward"))?;
                let rhs_val = nodes
                    .get(rhs)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("RHS value not available for MatMul backward"))?;

                let grad_2d = grad_output.view().into_dimensionality::<Ix2>()?;
                let lhs_2d = lhs_val.view().into_dimensionality::<Ix2>()?;
                let rhs_2d = rhs_val.view().into_dimensionality::<Ix2>()?;

                let grad_lhs_2d = grad_2d.dot(&rhs_2d.t());
                let grad_rhs_2d = lhs_2d.t().dot(&grad_2d);

                Ok(vec![
                    (*lhs, grad_lhs_2d.into_dyn()),
                    (*rhs, grad_rhs_2d.into_dyn()),
                ])
            }

            Operation::Neg { input } => {
                // d/dx (-x) = -1
                let grad_input = grad_output.mapv(|x| -x);
                Ok(vec![(*input, grad_input)])
            }

            Operation::Exp { input } => {
                // d/dx exp(x) = exp(x)
                let input_val = nodes
                    .get(input)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("Input value not available for Exp backward"))?;
                let grad_input = grad_output * &input_val.mapv(|x| x.exp());
                Ok(vec![(*input, grad_input)])
            }

            Operation::Log { input } => {
                // d/dx log(x) = 1/x
                let input_val = nodes
                    .get(input)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("Input value not available for Log backward"))?;
                let grad_input = grad_output / input_val;
                Ok(vec![(*input, grad_input)])
            }

            Operation::Pow { input, exponent } => {
                // d/dx x^n = n * x^(n-1)
                let input_val = nodes
                    .get(input)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("Input value not available for Pow backward"))?;

                let n = T::from(*exponent).ok_or_else(|| anyhow!("Failed to convert exponent"))?;
                let n_minus_1 = T::from(exponent - 1.0)
                    .ok_or_else(|| anyhow!("Failed to convert exponent-1"))?;

                let grad_input = grad_output * &(input_val.mapv(|x| n * x.powf(n_minus_1)));
                Ok(vec![(*input, grad_input)])
            }

            Operation::Sum { input, axis } => {
                // Broadcast gradient back to input shape
                let input_val = nodes
                    .get(input)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("Input value not available for Sum backward"))?;
                let input_shape = input_val.shape();

                let grad_input = match *axis {
                    None => {
                        // Full reduction - broadcast scalar to full shape
                        ArrayD::from_elem(IxDyn(input_shape), grad_output[[]])
                    }
                    Some(ax) => {
                        // Partial reduction - add dimension back
                        let mut new_shape = grad_output.shape().to_vec();
                        new_shape.insert(ax, 1);
                        let reshaped = grad_output
                            .clone()
                            .to_shape(IxDyn(&new_shape))
                            .context("Reshape failed in Sum backward")?
                            .to_owned();
                        reshaped
                            .broadcast(IxDyn(input_shape))
                            .ok_or_else(|| anyhow!("Broadcast failed in Sum backward"))?
                            .to_owned()
                    }
                };

                Ok(vec![(*input, grad_input)])
            }

            Operation::Mean { input, axis } => {
                // Similar to sum, but divide by the number of elements
                let input_val = nodes
                    .get(input)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("Input value not available for Mean backward"))?;
                let input_shape = input_val.shape();

                let (grad_input, n_elements) = match *axis {
                    None => {
                        let n = input_val.len();
                        let grad = ArrayD::from_elem(IxDyn(input_shape), grad_output[[]]);
                        (grad, n)
                    }
                    Some(ax) => {
                        let n = input_shape[ax];
                        let mut new_shape = grad_output.shape().to_vec();
                        new_shape.insert(ax, 1);
                        let reshaped = grad_output
                            .clone()
                            .to_shape(IxDyn(&new_shape))
                            .context("Reshape failed in Mean backward")?
                            .to_owned();
                        let grad = reshaped
                            .broadcast(IxDyn(input_shape))
                            .ok_or_else(|| anyhow!("Broadcast failed in Mean backward"))?
                            .to_owned();
                        (grad, n)
                    }
                };

                let divisor =
                    T::from(n_elements).ok_or_else(|| anyhow!("Failed to convert n_elements"))?;
                let grad_input_scaled = grad_input / divisor;

                Ok(vec![(*input, grad_input_scaled)])
            }

            Operation::ReLU { input } => {
                // d/dx ReLU(x) = 1 if x > 0, else 0
                let input_val = nodes
                    .get(input)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("Input value not available for ReLU backward"))?;

                let mask = input_val.mapv(|x| if x > T::zero() { T::one() } else { T::zero() });
                let grad_input = grad_output * &mask;
                Ok(vec![(*input, grad_input)])
            }

            Operation::Sigmoid { input } => {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                let input_val = nodes
                    .get(input)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("Input value not available for Sigmoid backward"))?;

                let sigmoid_val = input_val.mapv(|x| T::one() / (T::one() + (-x).exp()));
                let grad_factor = &sigmoid_val * &sigmoid_val.mapv(|s| T::one() - s);
                let grad_input = grad_output * &grad_factor;
                Ok(vec![(*input, grad_input)])
            }

            Operation::Tanh { input } => {
                // d/dx tanh(x) = 1 - tanh(x)^2
                let input_val = nodes
                    .get(input)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("Input value not available for Tanh backward"))?;

                let tanh_val = input_val.mapv(|x| x.tanh());
                let grad_factor = tanh_val.mapv(|t| T::one() - t * t);
                let grad_input = grad_output * &grad_factor;
                Ok(vec![(*input, grad_input)])
            }

            Operation::Reshape { input, old_shape } => {
                // Gradient has the same shape as output, reshape back to input shape
                let grad_input = grad_output
                    .clone()
                    .to_shape(IxDyn(old_shape))
                    .context("Reshape backward failed")?
                    .to_owned();
                Ok(vec![(*input, grad_input)])
            }

            Operation::Transpose { input, axes } => {
                // grad_x = permute(grad_out, inverse permutation of axes)
                //
                // Forward: y[axes[0], axes[1], ...] = x, so backward
                // scatters grad_y back along the original axes by reversing the permutation.
                let ndim = axes.len();
                let mut inv_axes = vec![0usize; ndim];
                for (i, &ax) in axes.iter().enumerate() {
                    inv_axes[ax] = i;
                }
                let grad_input = grad_output
                    .view()
                    .permuted_axes(IxDyn(&inv_axes))
                    .to_owned();
                Ok(vec![(*input, grad_input)])
            }

            Operation::Broadcast {
                input,
                original_shape,
            } => {
                // Backward: sum over all axes that were added or expanded during broadcast.
                //
                // Algorithm:
                // 1. Pad original_shape on the left with 1s to match grad_output.ndim().
                // 2. For each axis where padded_orig[ax] == 1 and out_shape[ax] > 1:
                //    sum over that axis (keepdim: remove then re-insert as size 1).
                // 3. Reshape to original_shape (removes leading padded-1 dimensions).
                let ndim_out = grad_output.ndim();
                let ndim_in = original_shape.len();
                let pad_left = ndim_out.saturating_sub(ndim_in);
                let padded_orig: Vec<usize> = std::iter::repeat_n(1, pad_left)
                    .chain(original_shape.iter().copied())
                    .collect();
                let out_shape = grad_output.shape().to_vec();
                let mut grad = grad_output.clone();
                for ax in 0..ndim_out {
                    if padded_orig[ax] == 1 && out_shape[ax] > 1 {
                        let summed = grad.sum_axis(Axis(ax));
                        let mut new_shape = summed.shape().to_vec();
                        new_shape.insert(ax, 1);
                        grad = summed
                            .to_shape(IxDyn(&new_shape))
                            .context("Broadcast backward keepdim reshape failed")?
                            .to_owned();
                    }
                }
                grad = grad
                    .to_shape(IxDyn(original_shape))
                    .context("Broadcast backward final reshape failed")?
                    .to_owned();
                Ok(vec![(*input, grad)])
            }

            Operation::Slice { input, ranges } => {
                // Backward: scatter grad_output into zeros of input shape.
                //
                // For each element at output position out_idx,
                // the corresponding input position is ranges[ax].0 + out_idx[ax] per axis.
                let input_val = nodes
                    .get(input)
                    .and_then(|n| n.value.as_ref())
                    .ok_or_else(|| anyhow!("Input value not available for Slice backward"))?;
                let input_shape = input_val.shape().to_vec();
                let slice_shape: Vec<usize> = ranges.iter().map(|&(s, e)| e - s).collect();
                let ndim = input_shape.len();
                let flat_size: usize = slice_shape.iter().product::<usize>();
                let mut grad_input = ArrayD::<T>::zeros(IxDyn(&input_shape));
                for out_flat in 0..flat_size {
                    let mut out_idx = vec![0usize; ndim];
                    let mut remaining = out_flat;
                    for d in (0..ndim).rev() {
                        if slice_shape[d] > 0 {
                            out_idx[d] = remaining % slice_shape[d];
                            remaining /= slice_shape[d];
                        }
                    }
                    let in_idx: Vec<usize> = out_idx
                        .iter()
                        .enumerate()
                        .map(|(ax, &i)| ranges[ax].0 + i)
                        .collect();
                    let g = grad_output[out_idx.as_slice()];
                    grad_input[in_idx.as_slice()] = grad_input[in_idx.as_slice()] + g;
                }
                Ok(vec![(*input, grad_input)])
            }
        }
    }

    /// Get statistics about the computation graph
    pub fn stats(&self) -> GraphStats {
        let nodes = lock_mutex(&self.nodes);
        let num_nodes = nodes.len();
        let num_edges: usize = nodes.values().map(|n| n.children.len()).sum();

        let mut ops_count: HashMap<String, usize> = HashMap::new();
        for node in nodes.values() {
            let op_name = format!("{:?}", node.operation)
                .split(' ')
                .next()
                .unwrap_or("Unknown")
                .to_string();
            *ops_count.entry(op_name).or_insert(0) += 1;
        }

        let num_requires_grad = nodes.values().filter(|n| n.requires_grad).count();

        GraphStats {
            num_nodes,
            num_edges,
            num_requires_grad,
            ops_count,
        }
    }

    // ===== Helpers for graph optimization passes =====
    //
    // These `pub(crate)` helpers expose internals so optimization passes in
    // `graph_optimizer` can inspect and mutate the graph while keeping
    // `GraphNode` private.

    /// Snapshot all `(NodeId, Operation)` pairs for iteration by passes.
    pub(crate) fn snapshot_ops(&self) -> Vec<(NodeId, Operation)> {
        let nodes = lock_mutex(&self.nodes);
        nodes
            .iter()
            .map(|(id, node)| (*id, node.operation.clone()))
            .collect()
    }

    /// Return every `NodeId` currently present in the graph.
    pub(crate) fn all_node_ids(&self) -> Vec<NodeId> {
        let nodes = lock_mutex(&self.nodes);
        nodes.keys().copied().collect()
    }

    /// Total number of nodes.
    pub(crate) fn num_nodes(&self) -> usize {
        let nodes = lock_mutex(&self.nodes);
        nodes.len()
    }

    /// Return `true` when the node is a compile-time constant: `Operation::Input`
    /// with `requires_grad == false`.
    pub(crate) fn is_constant(&self, id: NodeId) -> bool {
        let nodes = lock_mutex(&self.nodes);
        nodes
            .get(&id)
            .map(|n| matches!(n.operation, Operation::Input) && !n.requires_grad)
            .unwrap_or(false)
    }

    /// Return `true` if the node has `requires_grad == true`.
    pub(crate) fn node_requires_grad(&self, id: NodeId) -> bool {
        let nodes = lock_mutex(&self.nodes);
        nodes.get(&id).is_some_and(|n| n.requires_grad)
    }

    /// Element count of the stored value, if available.
    pub(crate) fn node_element_count(&self, id: NodeId) -> Option<usize> {
        let nodes = lock_mutex(&self.nodes);
        nodes
            .get(&id)
            .and_then(|n| n.value.as_ref().map(|v| v.len()))
    }

    /// Parent node IDs of a node in their original order.
    pub(crate) fn node_parents(&self, id: NodeId) -> Option<Vec<NodeId>> {
        let nodes = lock_mutex(&self.nodes);
        nodes.get(&id).map(|n| n.parents.clone())
    }

    /// Operation of a node.
    pub(crate) fn node_operation(&self, id: NodeId) -> Option<Operation> {
        let nodes = lock_mutex(&self.nodes);
        nodes.get(&id).map(|n| n.operation.clone())
    }

    /// Reclassify an existing node as a compile-time constant.
    ///
    /// Preserves the cached `value` (the result of eager evaluation that
    /// happened during graph construction) but rewrites the operation to
    /// `Operation::Input`, clears `requires_grad`, and removes the node from
    /// its former parents' `children` lists so DCE can later collect them.
    ///
    /// This is the core primitive used by the constant-folding pass.
    pub(crate) fn reclassify_as_constant(&self, id: NodeId) -> Result<()> {
        let mut nodes = lock_mutex(&self.nodes);
        let old_parents = {
            let node = nodes
                .get_mut(&id)
                .ok_or_else(|| anyhow!("Node {} not found for reclassify", id))?;
            let parents = std::mem::take(&mut node.parents);
            node.operation = Operation::Input;
            node.requires_grad = false;
            node.gradient = None;
            parents
        };
        for parent_id in old_parents {
            if let Some(parent) = nodes.get_mut(&parent_id) {
                parent.children.retain(|&c| c != id);
            }
        }
        Ok(())
    }

    /// Redirect every reference to `from` to point at `to`.
    ///
    /// For CSE: rewrites every `Operation` field and every `parents` entry
    /// that names `from`, replacing it with `to`. Also migrates `from`'s
    /// children list into `to`'s children list (deduplicated). The `from`
    /// node itself is left in place; the subsequent DCE pass removes it if
    /// it is no longer reachable from any output.
    pub(crate) fn redirect_consumers(&self, from: NodeId, to: NodeId) -> Result<()> {
        if from == to {
            return Ok(());
        }
        let mut nodes = lock_mutex(&self.nodes);

        // Consumers: every node that lists `from` in its parents.
        let consumer_ids: Vec<NodeId> = nodes
            .iter()
            .filter(|(_, node)| node.parents.contains(&from))
            .map(|(id, _)| *id)
            .collect();

        for consumer_id in &consumer_ids {
            if let Some(consumer) = nodes.get_mut(consumer_id) {
                for p in consumer.parents.iter_mut() {
                    if *p == from {
                        *p = to;
                    }
                }
                rewrite_op_node_id(&mut consumer.operation, from, to);
            }
        }

        // Migrate children of `from` onto `to`.
        let moved_children: Vec<NodeId> = {
            if let Some(from_node) = nodes.get_mut(&from) {
                std::mem::take(&mut from_node.children)
            } else {
                Vec::new()
            }
        };
        if !moved_children.is_empty() {
            if let Some(to_node) = nodes.get_mut(&to) {
                for child in moved_children {
                    if !to_node.children.contains(&child) {
                        to_node.children.push(child);
                    }
                }
            }
        }

        Ok(())
    }

    /// Remove a set of nodes from the graph (used by DCE).
    ///
    /// Also cleans up any dangling references in the remaining nodes'
    /// `children` lists so the graph stays internally consistent.
    pub(crate) fn remove_nodes(&self, to_remove: &HashSet<NodeId>) -> usize {
        if to_remove.is_empty() {
            return 0;
        }
        let mut nodes = lock_mutex(&self.nodes);
        let mut removed = 0usize;
        for id in to_remove {
            if nodes.remove(id).is_some() {
                removed += 1;
            }
        }
        for node in nodes.values_mut() {
            node.children.retain(|c| !to_remove.contains(c));
            node.parents.retain(|p| !to_remove.contains(p));
        }
        removed
    }

    /// Collect every node id reachable from `roots` through the `parents`
    /// edges (i.e. the live set w.r.t. a chosen set of outputs).
    pub(crate) fn reachable_from(&self, roots: &[NodeId]) -> HashSet<NodeId> {
        let nodes = lock_mutex(&self.nodes);
        let mut live = HashSet::new();
        let mut stack: Vec<NodeId> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if !live.insert(id) {
                continue;
            }
            if let Some(node) = nodes.get(&id) {
                for &p in &node.parents {
                    if !live.contains(&p) {
                        stack.push(p);
                    }
                }
            }
        }
        live
    }
}

/// Rewrite every `NodeId` field inside an `Operation` value, replacing any
/// occurrence of `from` with `to`.
fn rewrite_op_node_id(op: &mut Operation, from: NodeId, to: NodeId) {
    let replace = |n: &mut NodeId| {
        if *n == from {
            *n = to;
        }
    };
    match op {
        Operation::Input => {}
        Operation::Add { lhs, rhs }
        | Operation::Sub { lhs, rhs }
        | Operation::Mul { lhs, rhs }
        | Operation::Div { lhs, rhs }
        | Operation::MatMul { lhs, rhs } => {
            replace(lhs);
            replace(rhs);
        }
        Operation::Neg { input }
        | Operation::Exp { input }
        | Operation::Log { input }
        | Operation::Pow { input, .. }
        | Operation::Sum { input, .. }
        | Operation::Mean { input, .. }
        | Operation::Reshape { input, .. }
        | Operation::Transpose { input, .. }
        | Operation::Broadcast { input, .. }
        | Operation::ReLU { input }
        | Operation::Sigmoid { input }
        | Operation::Tanh { input }
        | Operation::Slice { input, .. } => {
            replace(input);
        }
    }
}

/// Statistics about the computation graph
#[derive(Debug)]
pub struct GraphStats {
    /// Total number of nodes
    pub num_nodes: usize,
    /// Total number of edges
    pub num_edges: usize,
    /// Number of nodes requiring gradients
    pub num_requires_grad: usize,
    /// Count of each operation type
    pub ops_count: HashMap<String, usize>,
}

impl fmt::Display for GraphStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Computation Graph Statistics:")?;
        writeln!(f, "  Nodes: {}", self.num_nodes)?;
        writeln!(f, "  Edges: {}", self.num_edges)?;
        writeln!(f, "  Requires Grad: {}", self.num_requires_grad)?;
        writeln!(f, "  Operations:")?;
        for (op, count) in &self.ops_count {
            writeln!(f, "    {}: {}", op, count)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;
    use scirs2_core::ndarray_ext::{ArrayD, IxDyn};

    #[test]
    fn test_basic_addition() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![2.0, 3.0].into_dyn(), true)?;
        let y = graph.variable(array![4.0, 5.0].into_dyn(), true)?;
        let z = graph.add(&x, &y)?;

        let z_val = graph.value(&z)?;
        assert_eq!(z_val[[0]], 6.0);
        assert_eq!(z_val[[1]], 8.0);

        graph.backward(&graph.sum(&z)?)?;
        let grad_x = graph.gradient(&x)?;
        let grad_y = graph.gradient(&y)?;

        assert_eq!(grad_x[[0]], 1.0);
        assert_eq!(grad_x[[1]], 1.0);
        assert_eq!(grad_y[[0]], 1.0);
        assert_eq!(grad_y[[1]], 1.0);

        Ok(())
    }

    #[test]
    fn test_multiplication_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![2.0, 3.0].into_dyn(), true)?;
        let y = graph.variable(array![4.0, 5.0].into_dyn(), true)?;
        let z = graph.mul(&x, &y)?;

        graph.backward(&graph.sum(&z)?)?;
        let grad_x = graph.gradient(&x)?;
        let grad_y = graph.gradient(&y)?;

        // d/dx (x*y) = y
        assert_eq!(grad_x[[0]], 4.0);
        assert_eq!(grad_x[[1]], 5.0);
        // d/dy (x*y) = x
        assert_eq!(grad_y[[0]], 2.0);
        assert_eq!(grad_y[[1]], 3.0);

        Ok(())
    }

    #[test]
    fn test_matmul_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let a = graph.variable(array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), true)?;
        let b = graph.variable(array![[5.0, 6.0], [7.0, 8.0]].into_dyn(), true)?;
        let c = graph.matmul(&a, &b)?;

        graph.backward(&graph.sum(&c)?)?;
        let grad_a = graph.gradient(&a)?;
        let grad_b = graph.gradient(&b)?;

        // Verify shapes
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_b.shape(), &[2, 2]);

        // d/dA (A @ B).sum() = ones @ B^T = B.sum(axis=0) repeated
        // d/dB (A @ B).sum() = A^T @ ones = A.sum(axis=1) repeated
        assert_eq!(grad_a[[0, 0]], 11.0); // 5 + 6
        assert_eq!(grad_a[[0, 1]], 15.0); // 7 + 8
        assert_eq!(grad_b[[0, 0]], 4.0); // 1 + 3
        assert_eq!(grad_b[[1, 0]], 6.0); // 2 + 4

        Ok(())
    }

    #[test]
    fn test_chain_rule() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        // f(x) = (x + 1) * 2
        let x = graph.variable(array![3.0].into_dyn(), true)?;
        let one = graph.constant(array![1.0].into_dyn())?;
        let two = graph.constant(array![2.0].into_dyn())?;

        let x_plus_1 = graph.add(&x, &one)?;
        let y = graph.mul(&x_plus_1, &two)?;

        graph.backward(&y)?;
        let grad_x = graph.gradient(&x)?;

        // df/dx = 2
        assert_eq!(grad_x[[0]], 2.0);

        Ok(())
    }

    #[test]
    fn test_exp_log_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![1.0, 2.0].into_dyn(), true)?;
        let exp_x = graph.exp(&x)?;
        let log_exp_x = graph.log(&exp_x)?;

        graph.backward(&graph.sum(&log_exp_x)?)?;
        let grad_x = graph.gradient(&x)?;

        // d/dx log(exp(x)) = 1
        assert!((grad_x[[0]] - 1.0).abs() < 1e-6);
        assert!((grad_x[[1]] - 1.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_relu_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![-1.0, 0.0, 1.0, 2.0].into_dyn(), true)?;
        let y = graph.relu(&x)?;

        let y_val = graph.value(&y)?;
        assert_eq!(y_val[[0]], 0.0);
        assert_eq!(y_val[[1]], 0.0);
        assert_eq!(y_val[[2]], 1.0);
        assert_eq!(y_val[[3]], 2.0);

        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;

        assert_eq!(grad_x[[0]], 0.0); // x < 0
        assert_eq!(grad_x[[1]], 0.0); // x = 0
        assert_eq!(grad_x[[2]], 1.0); // x > 0
        assert_eq!(grad_x[[3]], 1.0); // x > 0

        Ok(())
    }

    #[test]
    fn test_sigmoid_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![0.0].into_dyn(), true)?;
        let y = graph.sigmoid(&x)?;

        let y_val = graph.value(&y)?;
        assert!((y_val[[0]] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5

        graph.backward(&y)?;
        let grad_x = graph.gradient(&x)?;

        // d/dx sigmoid(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        assert!((grad_x[[0]] - 0.25).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_mean_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![2.0, 4.0, 6.0, 8.0].into_dyn(), true)?;
        let y = graph.mean(&x)?;

        let y_val = graph.value(&y)?;
        assert_eq!(y_val[[]], 5.0); // (2+4+6+8)/4 = 5

        graph.backward(&y)?;
        let grad_x = graph.gradient(&x)?;

        // d/dx mean(x) = 1/n for each element
        for i in 0..4 {
            assert_eq!(grad_x[[i]], 0.25);
        }

        Ok(())
    }

    #[test]
    fn test_reshape_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), true)?;
        let y = graph.reshape(&x, &[4])?;

        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;

        assert_eq!(grad_x.shape(), &[2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(grad_x[[i, j]], 1.0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_zero_grad() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![1.0, 2.0].into_dyn(), true)?;
        let y = graph.mul(&x, &x)?;

        graph.backward(&graph.sum(&y)?)?;
        assert!(graph.has_gradient(&x));

        graph.zero_grad();
        assert!(!graph.has_gradient(&x));

        Ok(())
    }

    #[test]
    fn test_eval_mode() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        graph.eval(); // Disable gradient tracking

        let x = graph.variable(array![1.0, 2.0].into_dyn(), true)?;
        let y = graph.mul(&x, &x)?;

        // Should not create gradients in eval mode
        let y_val = graph.value(&y)?;
        assert_eq!(y_val[[0]], 1.0);
        assert_eq!(y_val[[1]], 4.0);

        // Switch back to train mode
        graph.train();
        assert!(graph.is_recording());

        Ok(())
    }

    #[test]
    fn test_graph_stats() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![1.0].into_dyn(), true)?;
        let y = graph.variable(array![2.0].into_dyn(), true)?;
        let z = graph.add(&x, &y)?;
        let _ = graph.mul(&z, &x)?;

        let stats = graph.stats();
        assert_eq!(stats.num_nodes, 4); // x, y, z, result
        assert!(stats.num_requires_grad > 0);

        Ok(())
    }

    #[test]
    fn test_pow_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![2.0, 3.0].into_dyn(), true)?;
        let y = graph.pow(&x, 3.0)?; // x^3

        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;

        // d/dx x^3 = 3x^2
        assert_eq!(grad_x[[0]], 12.0); // 3 * 2^2 = 12
        assert_eq!(grad_x[[1]], 27.0); // 3 * 3^2 = 27

        Ok(())
    }

    #[test]
    fn test_tanh_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![0.0].into_dyn(), true)?;
        let y = graph.tanh(&x)?;

        let y_val = graph.value(&y)?;
        assert!((y_val[[0]] - 0.0).abs() < 1e-6); // tanh(0) = 0

        graph.backward(&y)?;
        let grad_x = graph.gradient(&x)?;

        // d/dx tanh(0) = 1 - tanh(0)^2 = 1
        assert!((grad_x[[0]] - 1.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_division_gradient() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(array![6.0].into_dyn(), true)?;
        let y = graph.variable(array![2.0].into_dyn(), true)?;
        let z = graph.div(&x, &y)?; // 6 / 2 = 3

        graph.backward(&z)?;
        let grad_x = graph.gradient(&x)?;
        let grad_y = graph.gradient(&y)?;

        // d/dx (x/y) = 1/y = 1/2 = 0.5
        assert_eq!(grad_x[[0]], 0.5);
        // d/dy (x/y) = -x/y^2 = -6/4 = -1.5
        assert_eq!(grad_y[[0]], -1.5);

        Ok(())
    }

    #[test]
    fn test_transpose_forward_backward() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        // 2D matrix: [[1,2,3],[4,5,6]] shape [2,3]
        let x = graph.variable(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            true,
        )?;
        let y = graph.transpose(&x, vec![1, 0])?; // [2,3] -> [3,2]
        let y_val = graph.value(&y)?;
        assert_eq!(y_val.shape(), &[3, 2]);
        // y[0,0]=1, y[0,1]=4, y[1,0]=2, y[1,1]=5, y[2,0]=3, y[2,1]=6
        assert_eq!(y_val[[0usize, 0]], 1.0);
        assert_eq!(y_val[[0usize, 1]], 4.0);
        assert_eq!(y_val[[1usize, 0]], 2.0);
        assert_eq!(y_val[[2usize, 1]], 6.0);

        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;
        // Transpose backward = inverse-permute = transpose again; all-ones grad_out
        // -> grad_x is all-ones with shape [2,3]
        assert_eq!(grad_x.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(grad_x[[i, j]], 1.0);
            }
        }
        Ok(())
    }

    #[test]
    fn test_transpose_3d_backward() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        // x shape [2,3,4]; permute to [4,2,3]
        let size = 2 * 3 * 4;
        let data: Vec<f64> = (1..=size).map(|i| i as f64).collect();
        let x = graph.variable(
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), data).unwrap(),
            true,
        )?;
        let y = graph.transpose(&x, vec![2, 0, 1])?;
        let y_val = graph.value(&y)?;
        assert_eq!(y_val.shape(), &[4, 2, 3]);
        // y[k,i,j] = x[i,j,k]
        // x[0,0,0] = 1 -> y[0,0,0] = 1
        assert_eq!(y_val[[0usize, 0, 0]], 1.0);
        // x[1,2,3] = 24 -> y[3,1,2] = 24
        assert_eq!(y_val[[3usize, 1, 2]], 24.0);

        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;
        assert_eq!(grad_x.shape(), &[2, 3, 4]);
        // All-ones output gradient -> all-ones input gradient
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(grad_x[[i, j, k]], 1.0, "grad_x[{i},{j},{k}] != 1.0");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_transpose_invalid_axes() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0; 6]).unwrap(),
            true,
        )?;
        // Wrong length
        assert!(graph.transpose(&x, vec![0]).is_err());
        // Out of range axis
        assert!(graph.transpose(&x, vec![0, 5]).is_err());
        // Duplicate axis
        assert!(graph.transpose(&x, vec![0, 0]).is_err());
        Ok(())
    }

    #[test]
    fn test_broadcast_forward_backward() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        // x shape [3], broadcast to [4,3]
        let x = graph.variable(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
            true,
        )?;
        let y = graph.broadcast(&x, &[4, 3])?;
        let y_val = graph.value(&y)?;
        assert_eq!(y_val.shape(), &[4, 3]);
        // Each row of y should equal x
        for row in 0..4 {
            for col in 0..3 {
                assert_eq!(y_val[[row, col]], (col + 1) as f64);
            }
        }

        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;
        // Each element of x was used 4 times (one per row) -> gradient = 4
        assert_eq!(grad_x.shape(), &[3]);
        for j in 0..3 {
            assert_eq!(grad_x[[j]], 4.0, "grad_x[{j}] = {}", grad_x[[j]]);
        }
        Ok(())
    }

    #[test]
    fn test_broadcast_singleton_expansion() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        // x shape [1,3], broadcast to [4,3]
        let x = graph.variable(
            ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
            true,
        )?;
        let y = graph.broadcast(&x, &[4, 3])?;
        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;
        // Shape must be preserved as [1, 3] (keepdim)
        assert_eq!(grad_x.shape(), &[1, 3]);
        for j in 0..3 {
            assert_eq!(grad_x[[0, j]], 4.0);
        }
        Ok(())
    }

    #[test]
    fn test_broadcast_scalar_to_matrix() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        // x scalar (shape [1]), broadcast to [3,4]
        let x = graph.variable(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![5.0]).unwrap(),
            true,
        )?;
        let y = graph.broadcast(&x, &[3, 4])?;
        let y_val = graph.value(&y)?;
        assert_eq!(y_val.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(y_val[[i, j]], 5.0);
            }
        }
        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;
        // Scalar used 12 times -> gradient sum = 12
        assert_eq!(grad_x.shape(), &[1]);
        assert_eq!(grad_x[[0]], 12.0);
        Ok(())
    }

    #[test]
    fn test_slice_nd_forward_backward() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        // x shape [4,5]; slice [1..3, 2..4] -> shape [2,2]
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let x = graph.variable(ArrayD::from_shape_vec(IxDyn(&[4, 5]), data).unwrap(), true)?;
        let y = graph.slice_nd(&x, vec![(1, 3), (2, 4)])?;
        let y_val = graph.value(&y)?;
        assert_eq!(y_val.shape(), &[2, 2]);
        // x[i,j] = i*5 + j
        // y[0,0] = x[1,2] = 7; y[0,1] = x[1,3] = 8
        // y[1,0] = x[2,2] = 12; y[1,1] = x[2,3] = 13
        assert_eq!(y_val[[0usize, 0]], 7.0);
        assert_eq!(y_val[[0usize, 1]], 8.0);
        assert_eq!(y_val[[1usize, 0]], 12.0);
        assert_eq!(y_val[[1usize, 1]], 13.0);

        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;
        assert_eq!(grad_x.shape(), &[4, 5]);
        // Only elements within the slice [1..3, 2..4] should have gradient 1.0
        for i in 0..4 {
            for j in 0..5 {
                let expected = if (1..3).contains(&i) && (2..4).contains(&j) {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(
                    grad_x[[i, j]],
                    expected,
                    "grad_x[{i},{j}] = {} != {expected}",
                    grad_x[[i, j]]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_slice_nd_full_slice() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(
            ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![1.0; 9]).unwrap(),
            true,
        )?;
        // Full slice: [0..3, 0..3] -> same as input
        let y = graph.slice_nd(&x, vec![(0, 3), (0, 3)])?;
        graph.backward(&graph.sum(&y)?)?;
        let grad_x = graph.gradient(&x)?;
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(grad_x[[i, j]], 1.0);
            }
        }
        Ok(())
    }

    #[test]
    fn test_slice_nd_invalid_ranges() -> Result<()> {
        let graph = ComputationGraph::<f64>::new();
        let x = graph.variable(
            ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![1.0; 9]).unwrap(),
            true,
        )?;
        // End out of bounds
        assert!(graph.slice_nd(&x, vec![(0, 4), (0, 3)]).is_err());
        // Start > end
        assert!(graph.slice_nd(&x, vec![(2, 1), (0, 3)]).is_err());
        // Wrong number of ranges
        assert!(graph.slice_nd(&x, vec![(0, 2)]).is_err());
        Ok(())
    }
}
