//! Contraction (einsum) helper for the `TenrsoExecutor` implementation.

use super::super::types::CpuExecutor;
use crate::hints::ExecHints;
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{DenseND, TensorHandle};
use tenrso_planner::EinsumSpec;

pub(super) fn einsum<T>(
    executor: &mut CpuExecutor,
    spec: &str,
    inputs: &[TensorHandle<T>],
    hints: &ExecHints,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let parsed_spec = EinsumSpec::parse(spec)?;
    if parsed_spec.num_inputs() != inputs.len() {
        return Err(anyhow!(
            "Spec expects {} inputs, got {}",
            parsed_spec.num_inputs(),
            inputs.len()
        ));
    }
    let dense_inputs: Vec<&DenseND<T>> = inputs
        .iter()
        .map(|h| {
            h.as_dense()
                .ok_or_else(|| anyhow!("Only dense tensors supported for now"))
        })
        .collect::<Result<Vec<_>>>()?;
    let dense_inputs_owned: Vec<DenseND<T>> = dense_inputs.iter().map(|&t| t.clone()).collect();
    let result = executor.execute_einsum_with_planner(&parsed_spec, &dense_inputs_owned, hints)?;
    Ok(TensorHandle::from_dense_auto(result))
}
