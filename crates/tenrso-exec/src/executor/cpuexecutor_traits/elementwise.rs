//! Element-wise operation helpers for the `TenrsoExecutor` implementation on `CpuExecutor`.
//!
//! Contains free functions supporting: `elem_op`, `binary_op`, `clip`, `modulo`.

use super::super::types::{BinaryOp, CpuExecutor, ElemOp};
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{DenseND, TensorHandle};

pub(super) fn elem_op<T>(
    _executor: &mut CpuExecutor,
    op: ElemOp,
    x: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for elem_op"))?;
    let result_data = match op {
        ElemOp::Neg => dense.view().mapv(|v| -v),
        ElemOp::Abs => dense.view().mapv(|v| v.abs()),
        ElemOp::Exp => dense.view().mapv(|v| v.exp()),
        ElemOp::Log => dense.view().mapv(|v| v.ln()),
        ElemOp::Sin => dense.view().mapv(|v| v.sin()),
        ElemOp::Cos => dense.view().mapv(|v| v.cos()),
        ElemOp::Sqrt => dense.view().mapv(|v| v.sqrt()),
        ElemOp::Sqr => dense.view().mapv(|v| v * v),
        ElemOp::Recip => dense.view().mapv(|v| v.recip()),
        ElemOp::Tanh => dense.view().mapv(|v| v.tanh()),
        ElemOp::Sigmoid => dense.view().mapv(|v| {
            let one = T::one();
            one / (one + (-v).exp())
        }),
        ElemOp::ReLU => dense.view().mapv(|v| {
            let zero = T::zero();
            if v > zero {
                v
            } else {
                zero
            }
        }),
        ElemOp::Gelu => dense.view().mapv(|v| {
            let half = T::from_f64(0.5).unwrap_or_else(T::one);
            let one = T::one();
            let coeff = T::from_f64(0.7978845608028654).unwrap_or_else(T::one);
            let cubic_coeff = T::from_f64(0.044715).unwrap_or_else(T::zero);
            let x_cubed = v * v * v;
            let inner = coeff * (v + cubic_coeff * x_cubed);
            half * v * (one + inner.tanh())
        }),
        ElemOp::Elu => dense.view().mapv(|v| {
            let zero = T::zero();
            let one = T::one();
            if v > zero {
                v
            } else {
                v.exp() - one
            }
        }),
        ElemOp::Selu => dense.view().mapv(|v| {
            let zero = T::zero();
            let one = T::one();
            let scale = T::from_f64(1.050_700_987_355_480_5).unwrap_or_else(T::one);
            let alpha = T::from_f64(1.673_263_242_354_377_2).unwrap_or_else(T::one);
            if v > zero {
                scale * v
            } else {
                scale * alpha * (v.exp() - one)
            }
        }),
        ElemOp::Softplus => dense.view().mapv(|v| {
            let zero = T::zero();
            let one = T::one();
            let abs_v = v.abs();
            let max_part = if v > zero { v } else { zero };
            max_part + (one + (-abs_v).exp()).ln()
        }),
        ElemOp::Sign => dense.view().mapv(|v| {
            let zero = T::zero();
            let one = T::one();
            let neg_one = -one;
            if v > zero {
                one
            } else if v < zero {
                neg_one
            } else {
                zero
            }
        }),
    };
    let result = DenseND::from_array(result_data);
    Ok(TensorHandle::from_dense_auto(result))
}

pub(super) fn binary_op<T>(
    executor: &mut CpuExecutor,
    op: BinaryOp,
    x: &TensorHandle<T>,
    y: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense_x = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for binary_op"))?;
    let dense_y = y
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for binary_op"))?;
    if dense_x.shape() != dense_y.shape() {
        return executor.binary_op_with_broadcast(op, dense_x, dense_y);
    }
    use scirs2_core::ndarray_ext::Zip;
    let result_data = match op {
        BinaryOp::Add => &dense_x.view() + &dense_y.view(),
        BinaryOp::Sub => &dense_x.view() - &dense_y.view(),
        BinaryOp::Mul => &dense_x.view() * &dense_y.view(),
        BinaryOp::Div => &dense_x.view() / &dense_y.view(),
        BinaryOp::Pow => {
            let mut result = dense_x.view().to_owned();
            Zip::from(&mut result)
                .and(&dense_x.view())
                .and(&dense_y.view())
                .for_each(|r, &x_val, &y_val| {
                    *r = x_val.powf(y_val);
                });
            result
        }
        BinaryOp::Maximum => {
            let mut result = dense_x.view().to_owned();
            Zip::from(&mut result)
                .and(&dense_x.view())
                .and(&dense_y.view())
                .for_each(|r, &x_val, &y_val| {
                    *r = if x_val > y_val { x_val } else { y_val };
                });
            result
        }
        BinaryOp::Minimum => {
            let mut result = dense_x.view().to_owned();
            Zip::from(&mut result)
                .and(&dense_x.view())
                .and(&dense_y.view())
                .for_each(|r, &x_val, &y_val| {
                    *r = if x_val < y_val { x_val } else { y_val };
                });
            result
        }
    };
    let result = DenseND::from_array(result_data);
    Ok(TensorHandle::from_dense_auto(result))
}

pub(super) fn clip<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    min_val: T,
    max_val: T,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for clip"))?;
    if min_val > max_val {
        return Err(anyhow!("Invalid clip bounds: min_val > max_val"));
    }
    let result_data = dense.view().mapv(|v| {
        if v < min_val {
            min_val
        } else if v > max_val {
            max_val
        } else {
            v
        }
    });
    let result = DenseND::from_array(result_data);
    Ok(TensorHandle::from_dense_auto(result))
}

pub(super) fn modulo<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    divisor: T,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for modulo"))?;
    if divisor == T::zero() {
        return Err(anyhow!("Division by zero in modulo operation"));
    }
    let result_data = dense.view().mapv(|v| {
        let quot = (v / divisor).floor();
        v - quot * divisor
    });
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_data,
    )))
}
