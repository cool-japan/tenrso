# tenrso-planner

Contraction order planning and optimization for TenRSo.

## Overview

`tenrso-planner` implements intelligent planning for tensor contractions:

- **Contraction order search** - Find optimal/near-optimal contraction sequences
- **Representation selection** - Choose dense/sparse/low-rank per operation
- **Tiling strategies** - Cache-aware blocking for large tensors
- **Cost models** - Estimate flops, memory, and communication

## Features

- Heuristic and dynamic programming planners
- Multi-objective optimization (time, memory, accuracy)
- Streaming and out-of-core planning
- Cost estimation with sparsity awareness

## Usage

```toml
[dependencies]
tenrso-planner = "0.1"
```

### Basic Planning (TODO: M4)

```rust
use tenrso_planner::{Planner, PlanHints};

let planner = Planner::new();
let plan = planner.make_plan(
    "ijk,jkl,klm->ilm",  // einsum spec
    &[&tensor_a, &tensor_b, &tensor_c],
    &PlanHints::default(),
)?;

println!("Contraction order: {:?}", plan.order());
println!("Estimated FLOPs: {}", plan.estimated_flops());
```

## API Reference

```rust
pub struct Plan {
    pub nodes: Vec<PlanNode>,
    pub estimated_flops: f64,
    pub estimated_memory: usize,
}

pub trait Planner {
    fn make_plan(
        spec: &str,
        inputs: &[&TensorHandle<f32>],
        hints: &PlanHints,
    ) -> Result<Plan>;
}
```

## Planning Algorithms

- **Greedy**: Fast, good for small networks
- **Dynamic Programming**: Optimal for small inputs
- **Treewidth-based**: Near-optimal heuristic

## License

Apache-2.0
