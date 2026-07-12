//! Element-wise rules (unary and binary).
//!
//! Forward semantics match the Tensorlogic `ElemOp` contract exactly (see
//! `tensorlogic_infer::ops::ElemOp` and the reference `Scirs2Exec` backend), so
//! these rules can back a Tensorlogic executor without any re-interpretation.
//!
//! Gradients are delegated to the existing [`crate::vjp::ElementwiseUnaryVjp`] /
//! [`crate::vjp::ElementwiseBinaryVjp`] contexts; this module only supplies the
//! forward map and the analytic partial derivatives.
//!
//! # Non-differentiable points
//!
//! * `relu`, `abs`, `min`, `max`, `or_max`, `nor` are piecewise linear. At the
//!   kink the sub-gradient is chosen deterministically: `relu'(0) = 0`,
//!   `abs'(0) = 0` (`signum(0) = 0`), and ties in `min`/`max` route the whole
//!   gradient to the **first** operand.
//! * The comparison ops (`eq`, `lt`, `gt`, `lte`, `gte`) are piecewise constant,
//!   so their derivative is exactly zero almost everywhere. They are registered
//!   with a zero gradient — that is the true derivative, not a placeholder.

use anyhow::{bail, Result};
use tenrso_core::DenseND;

use super::{AdScalar, Arity, OpParams, OpRule};
use crate::vjp::{ElementwiseBinaryVjp, ElementwiseUnaryVjp, VjpOp};

/// Element-wise unary operations with an analytic derivative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    /// `-x`
    Neg,
    /// `1 - x` (Tensorlogic `ElemOp::OneMinus`, logical negation on \[0,1\])
    OneMinus,
    /// `max(x, 0)`
    Relu,
    /// `1 / (1 + exp(-x))`
    Sigmoid,
    /// `tanh(x)`
    Tanh,
    /// `exp(x)`
    Exp,
    /// `ln(x)`
    Ln,
    /// `sqrt(x)`
    Sqrt,
    /// `x^2`
    Square,
    /// `|x|`
    Abs,
}

impl UnaryOpKind {
    /// Every unary kind, in registration order.
    pub const ALL: [UnaryOpKind; 10] = [
        UnaryOpKind::Neg,
        UnaryOpKind::OneMinus,
        UnaryOpKind::Relu,
        UnaryOpKind::Sigmoid,
        UnaryOpKind::Tanh,
        UnaryOpKind::Exp,
        UnaryOpKind::Ln,
        UnaryOpKind::Sqrt,
        UnaryOpKind::Square,
        UnaryOpKind::Abs,
    ];

    /// Registry name of this op.
    pub fn name(&self) -> &'static str {
        match self {
            UnaryOpKind::Neg => "neg",
            UnaryOpKind::OneMinus => "one_minus",
            UnaryOpKind::Relu => "relu",
            UnaryOpKind::Sigmoid => "sigmoid",
            UnaryOpKind::Tanh => "tanh",
            UnaryOpKind::Exp => "exp",
            UnaryOpKind::Ln => "ln",
            UnaryOpKind::Sqrt => "sqrt",
            UnaryOpKind::Square => "square",
            UnaryOpKind::Abs => "abs",
        }
    }

    /// Look a kind up by its registry name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    /// Forward map `f(x)`.
    pub fn apply<T: AdScalar>(&self, x: T) -> T {
        match self {
            UnaryOpKind::Neg => -x,
            UnaryOpKind::OneMinus => T::one() - x,
            UnaryOpKind::Relu => {
                if x > T::zero() {
                    x
                } else {
                    T::zero()
                }
            }
            UnaryOpKind::Sigmoid => T::one() / (T::one() + (-x).exp()),
            UnaryOpKind::Tanh => x.tanh(),
            UnaryOpKind::Exp => x.exp(),
            UnaryOpKind::Ln => x.ln(),
            UnaryOpKind::Sqrt => x.sqrt(),
            UnaryOpKind::Square => x * x,
            UnaryOpKind::Abs => x.abs(),
        }
    }

    /// Derivative `f'(x)`.
    pub fn derivative<T: AdScalar>(&self, x: T) -> T {
        match self {
            UnaryOpKind::Neg => -T::one(),
            UnaryOpKind::OneMinus => -T::one(),
            UnaryOpKind::Relu => {
                if x > T::zero() {
                    T::one()
                } else {
                    T::zero()
                }
            }
            UnaryOpKind::Sigmoid => {
                let s = UnaryOpKind::Sigmoid.apply(x);
                s * (T::one() - s)
            }
            UnaryOpKind::Tanh => {
                let t = x.tanh();
                T::one() - t * t
            }
            UnaryOpKind::Exp => x.exp(),
            UnaryOpKind::Ln => T::one() / x,
            UnaryOpKind::Sqrt => {
                let two = T::one() + T::one();
                T::one() / (two * x.sqrt())
            }
            UnaryOpKind::Square => (T::one() + T::one()) * x,
            UnaryOpKind::Abs => {
                if x > T::zero() {
                    T::one()
                } else if x < T::zero() {
                    -T::one()
                } else {
                    T::zero()
                }
            }
        }
    }
}

/// Registry rule for an element-wise unary op.
#[derive(Debug, Clone, Copy)]
pub struct ElementwiseUnaryRule {
    kind: UnaryOpKind,
}

impl ElementwiseUnaryRule {
    /// Create a rule for `kind`.
    pub fn new(kind: UnaryOpKind) -> Self {
        Self { kind }
    }

    /// The op this rule implements.
    pub fn kind(&self) -> UnaryOpKind {
        self.kind
    }
}

impl<T> OpRule<T> for ElementwiseUnaryRule
where
    T: AdScalar,
{
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn arity(&self) -> Arity {
        Arity::Exact(1)
    }

    fn forward(&self, inputs: &[DenseND<T>], _params: &OpParams) -> Result<DenseND<T>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let kind = self.kind;
        Ok(DenseND::from_array(
            inputs[0].as_array().mapv(|v| kind.apply(v)),
        ))
    }

    fn vjp(
        &self,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
        _params: &OpParams,
    ) -> Result<Vec<DenseND<T>>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let kind = self.kind;
        let ctx = ElementwiseUnaryVjp::new(inputs[0].clone(), move |x: &T| kind.derivative(*x));
        ctx.vjp(output_grad)
    }
}

/// Element-wise binary operations with analytic partial derivatives.
///
/// The arithmetic and logical ops mirror `tensorlogic_infer::ops::ElemOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOpKind {
    /// `x + y`
    Add,
    /// `x - y`
    Sub,
    /// `x * y`
    Mul,
    /// `x / y`
    Div,
    /// `min(x, y)`
    Min,
    /// `max(x, y)`
    Max,
    /// `max(x, y)` — fuzzy OR (Tensorlogic `ElemOp::OrMax`)
    OrMax,
    /// `x + y - x*y` — probabilistic OR
    OrProbSum,
    /// `1 - x*y`
    Nand,
    /// `1 - max(x, y)`
    Nor,
    /// `x + y - 2*x*y` — soft XOR
    Xor,
    /// `1.0` if `|x - y| < 1e-10` else `0.0` (piecewise constant)
    Eq,
    /// `1.0` if `x < y` else `0.0` (piecewise constant)
    Lt,
    /// `1.0` if `x > y` else `0.0` (piecewise constant)
    Gt,
    /// `1.0` if `x <= y` else `0.0` (piecewise constant)
    Lte,
    /// `1.0` if `x >= y` else `0.0` (piecewise constant)
    Gte,
}

impl BinaryOpKind {
    /// Every binary kind, in registration order.
    pub const ALL: [BinaryOpKind; 16] = [
        BinaryOpKind::Add,
        BinaryOpKind::Sub,
        BinaryOpKind::Mul,
        BinaryOpKind::Div,
        BinaryOpKind::Min,
        BinaryOpKind::Max,
        BinaryOpKind::OrMax,
        BinaryOpKind::OrProbSum,
        BinaryOpKind::Nand,
        BinaryOpKind::Nor,
        BinaryOpKind::Xor,
        BinaryOpKind::Eq,
        BinaryOpKind::Lt,
        BinaryOpKind::Gt,
        BinaryOpKind::Lte,
        BinaryOpKind::Gte,
    ];

    /// Registry name of this op.
    pub fn name(&self) -> &'static str {
        match self {
            BinaryOpKind::Add => "add",
            BinaryOpKind::Sub => "sub",
            BinaryOpKind::Mul => "mul",
            BinaryOpKind::Div => "div",
            BinaryOpKind::Min => "min",
            BinaryOpKind::Max => "max",
            BinaryOpKind::OrMax => "or_max",
            BinaryOpKind::OrProbSum => "or_prob_sum",
            BinaryOpKind::Nand => "nand",
            BinaryOpKind::Nor => "nor",
            BinaryOpKind::Xor => "xor",
            BinaryOpKind::Eq => "eq",
            BinaryOpKind::Lt => "lt",
            BinaryOpKind::Gt => "gt",
            BinaryOpKind::Lte => "lte",
            BinaryOpKind::Gte => "gte",
        }
    }

    /// Look a kind up by its registry name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    /// Is this a piecewise-constant predicate (zero gradient a.e.)?
    pub fn is_predicate(&self) -> bool {
        matches!(
            self,
            BinaryOpKind::Eq
                | BinaryOpKind::Lt
                | BinaryOpKind::Gt
                | BinaryOpKind::Lte
                | BinaryOpKind::Gte
        )
    }

    /// Equality tolerance used by [`BinaryOpKind::Eq`], mirroring the reference
    /// Tensorlogic backend (`1e-10`).
    fn eq_tolerance<T: AdScalar>() -> T {
        T::from(1e-10_f64).unwrap_or_else(T::epsilon)
    }

    /// Forward map `f(x, y)`.
    pub fn apply<T: AdScalar>(&self, x: T, y: T) -> T {
        let one = T::one();
        let zero = T::zero();
        let indicator = |cond: bool| if cond { one } else { zero };

        match self {
            BinaryOpKind::Add => x + y,
            BinaryOpKind::Sub => x - y,
            BinaryOpKind::Mul => x * y,
            BinaryOpKind::Div => x / y,
            BinaryOpKind::Min => {
                if x <= y {
                    x
                } else {
                    y
                }
            }
            BinaryOpKind::Max | BinaryOpKind::OrMax => {
                if x >= y {
                    x
                } else {
                    y
                }
            }
            BinaryOpKind::OrProbSum => x + y - x * y,
            BinaryOpKind::Nand => one - x * y,
            BinaryOpKind::Nor => {
                let m = if x >= y { x } else { y };
                one - m
            }
            BinaryOpKind::Xor => x + y - (one + one) * x * y,
            BinaryOpKind::Eq => indicator((x - y).abs() < Self::eq_tolerance::<T>()),
            BinaryOpKind::Lt => indicator(x < y),
            BinaryOpKind::Gt => indicator(x > y),
            BinaryOpKind::Lte => indicator(x <= y),
            BinaryOpKind::Gte => indicator(x >= y),
        }
    }

    /// Partial derivative `∂f/∂x`.
    pub fn derivative_x<T: AdScalar>(&self, x: T, y: T) -> T {
        let one = T::one();
        let zero = T::zero();
        match self {
            BinaryOpKind::Add | BinaryOpKind::Sub => one,
            BinaryOpKind::Mul => y,
            BinaryOpKind::Div => one / y,
            BinaryOpKind::Min => {
                if x <= y {
                    one
                } else {
                    zero
                }
            }
            BinaryOpKind::Max | BinaryOpKind::OrMax => {
                if x >= y {
                    one
                } else {
                    zero
                }
            }
            BinaryOpKind::OrProbSum => one - y,
            BinaryOpKind::Nand => -y,
            BinaryOpKind::Nor => {
                if x >= y {
                    -one
                } else {
                    zero
                }
            }
            BinaryOpKind::Xor => one - (one + one) * y,
            // Piecewise-constant predicates: derivative is exactly zero a.e.
            BinaryOpKind::Eq
            | BinaryOpKind::Lt
            | BinaryOpKind::Gt
            | BinaryOpKind::Lte
            | BinaryOpKind::Gte => zero,
        }
    }

    /// Partial derivative `∂f/∂y`.
    pub fn derivative_y<T: AdScalar>(&self, x: T, y: T) -> T {
        let one = T::one();
        let zero = T::zero();
        match self {
            BinaryOpKind::Add => one,
            BinaryOpKind::Sub => -one,
            BinaryOpKind::Mul => x,
            BinaryOpKind::Div => -x / (y * y),
            BinaryOpKind::Min => {
                if x <= y {
                    zero
                } else {
                    one
                }
            }
            BinaryOpKind::Max | BinaryOpKind::OrMax => {
                if x >= y {
                    zero
                } else {
                    one
                }
            }
            BinaryOpKind::OrProbSum => one - x,
            BinaryOpKind::Nand => -x,
            BinaryOpKind::Nor => {
                if x >= y {
                    zero
                } else {
                    -one
                }
            }
            BinaryOpKind::Xor => one - (one + one) * x,
            // Piecewise-constant predicates: derivative is exactly zero a.e.
            BinaryOpKind::Eq
            | BinaryOpKind::Lt
            | BinaryOpKind::Gt
            | BinaryOpKind::Lte
            | BinaryOpKind::Gte => zero,
        }
    }
}

/// Registry rule for an element-wise binary op.
///
/// Both inputs must have identical shapes. Broadcasting is intentionally *not*
/// performed here: the Tensorlogic bridge handles the scalar-broadcast case
/// explicitly (and un-broadcasts the gradient), so the rule itself stays a plain
/// shape-preserving map with an exactly matching VJP.
#[derive(Debug, Clone, Copy)]
pub struct ElementwiseBinaryRule {
    kind: BinaryOpKind,
}

impl ElementwiseBinaryRule {
    /// Create a rule for `kind`.
    pub fn new(kind: BinaryOpKind) -> Self {
        Self { kind }
    }

    /// The op this rule implements.
    pub fn kind(&self) -> BinaryOpKind {
        self.kind
    }
}

impl<T> OpRule<T> for ElementwiseBinaryRule
where
    T: AdScalar,
{
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn arity(&self) -> Arity {
        Arity::Exact(2)
    }

    fn forward(&self, inputs: &[DenseND<T>], _params: &OpParams) -> Result<DenseND<T>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let (x, y) = (&inputs[0], &inputs[1]);
        if x.shape() != y.shape() {
            bail!(
                "'{}' requires identical input shapes, got {:?} and {:?}",
                self.kind.name(),
                x.shape(),
                y.shape()
            );
        }

        let kind = self.kind;
        let mut result = x.as_array().clone();
        scirs2_core::ndarray_ext::Zip::from(&mut result)
            .and(y.as_array())
            .for_each(|r, &y_val| {
                *r = kind.apply(*r, y_val);
            });

        Ok(DenseND::from_array(result))
    }

    fn vjp(
        &self,
        inputs: &[DenseND<T>],
        output_grad: &DenseND<T>,
        _params: &OpParams,
    ) -> Result<Vec<DenseND<T>>> {
        <Self as OpRule<T>>::check_arity(self, inputs.len())?;
        let (x, y) = (&inputs[0], &inputs[1]);
        if x.shape() != y.shape() {
            bail!(
                "'{}' requires identical input shapes, got {:?} and {:?}",
                self.kind.name(),
                x.shape(),
                y.shape()
            );
        }
        if output_grad.shape() != x.shape() {
            bail!(
                "'{}': output gradient shape {:?} does not match input shape {:?}",
                self.kind.name(),
                output_grad.shape(),
                x.shape()
            );
        }

        let kind = self.kind;
        let ctx = ElementwiseBinaryVjp::new(
            x.clone(),
            y.clone(),
            move |a: &T, b: &T| kind.derivative_x(*a, *b),
            move |a: &T, b: &T| kind.derivative_y(*a, *b),
        );
        ctx.vjp(output_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_names_are_unique() {
        let mut names: Vec<&str> = UnaryOpKind::ALL.iter().map(|k| k.name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn test_binary_names_are_unique() {
        let mut names: Vec<&str> = BinaryOpKind::ALL.iter().map(|k| k.name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn test_from_name_roundtrip() {
        for kind in UnaryOpKind::ALL {
            assert_eq!(UnaryOpKind::from_name(kind.name()), Some(kind));
        }
        for kind in BinaryOpKind::ALL {
            assert_eq!(BinaryOpKind::from_name(kind.name()), Some(kind));
        }
        assert_eq!(UnaryOpKind::from_name("nope"), None);
        assert_eq!(BinaryOpKind::from_name("nope"), None);
    }

    #[test]
    fn test_unary_forward_values() {
        let rule = ElementwiseUnaryRule::new(UnaryOpKind::Relu);
        let x: DenseND<f64> = DenseND::from_vec(vec![-1.0, 0.0, 2.0, 3.0], &[4]).unwrap();
        let y = rule.forward(&[x], &OpParams::none()).unwrap();
        assert_eq!(y.as_slice(), &[0.0, 0.0, 2.0, 3.0]);
    }

    #[test]
    fn test_binary_forward_logical_ops() {
        // Values in [0, 1] as used by Tensorlogic fuzzy logic.
        let x: DenseND<f64> = DenseND::from_vec(vec![0.25, 0.75], &[2]).unwrap();
        let y: DenseND<f64> = DenseND::from_vec(vec![0.5, 0.5], &[2]).unwrap();
        let params = OpParams::none();

        let or_prob = ElementwiseBinaryRule::new(BinaryOpKind::OrProbSum)
            .forward(&[x.clone(), y.clone()], &params)
            .unwrap();
        // 0.25 + 0.5 - 0.125 = 0.625
        assert!((or_prob.as_slice()[0] - 0.625).abs() < 1e-12);

        let nand = ElementwiseBinaryRule::new(BinaryOpKind::Nand)
            .forward(&[x.clone(), y.clone()], &params)
            .unwrap();
        assert!((nand.as_slice()[0] - (1.0 - 0.125)).abs() < 1e-12);

        let xor = ElementwiseBinaryRule::new(BinaryOpKind::Xor)
            .forward(&[x.clone(), y.clone()], &params)
            .unwrap();
        // 0.25 + 0.5 - 2*0.125 = 0.5
        assert!((xor.as_slice()[0] - 0.5).abs() < 1e-12);

        let nor = ElementwiseBinaryRule::new(BinaryOpKind::Nor)
            .forward(&[x, y], &params)
            .unwrap();
        // 1 - max(0.25, 0.5) = 0.5
        assert!((nor.as_slice()[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_predicate_gradients_are_zero() {
        let x: DenseND<f64> = DenseND::from_vec(vec![1.0, 5.0], &[2]).unwrap();
        let y: DenseND<f64> = DenseND::from_vec(vec![2.0, 3.0], &[2]).unwrap();
        let grad = DenseND::<f64>::ones(&[2]);

        for kind in [
            BinaryOpKind::Eq,
            BinaryOpKind::Lt,
            BinaryOpKind::Gt,
            BinaryOpKind::Lte,
            BinaryOpKind::Gte,
        ] {
            let rule = ElementwiseBinaryRule::new(kind);
            let grads = rule
                .vjp(&[x.clone(), y.clone()], &grad, &OpParams::none())
                .unwrap();
            assert_eq!(grads.len(), 2);
            assert!(grads[0].as_slice().iter().all(|&v| v == 0.0));
            assert!(grads[1].as_slice().iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_binary_shape_mismatch_errors() {
        let rule = ElementwiseBinaryRule::new(BinaryOpKind::Add);
        let x = DenseND::<f64>::ones(&[2, 2]);
        let y = DenseND::<f64>::ones(&[3]);
        assert!(rule.forward(&[x, y], &OpParams::none()).is_err());
    }
}
