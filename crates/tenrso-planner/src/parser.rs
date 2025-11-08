//! Einsum specification parser
//!
//! Parses and validates Einstein summation notation strings like "ijk,jkl->il".

use anyhow::{anyhow, bail, Result};
use std::collections::{HashMap, HashSet};

/// Parsed einsum specification
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumSpec {
    /// Input subscripts (e.g., ["ijk", "jkl"])
    pub inputs: Vec<String>,
    /// Output subscripts (e.g., "il")
    pub output: String,
    /// All unique indices that appear
    pub all_indices: HashSet<char>,
    /// Indices that are summed over (contracted)
    pub contracted_indices: HashSet<char>,
    /// Indices in output
    pub output_indices: HashSet<char>,
}

impl EinsumSpec {
    /// Parse an einsum specification string
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_planner::parser::EinsumSpec;
    ///
    /// let spec = EinsumSpec::parse("ijk,jkl->il").unwrap();
    /// assert_eq!(spec.inputs.len(), 2);
    /// assert_eq!(spec.output, "il");
    /// ```
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();

        // Split by "->" to separate inputs from output
        let parts: Vec<&str> = spec.split("->").collect();

        if parts.is_empty() {
            bail!("Empty einsum specification");
        }

        if parts.len() > 2 {
            bail!("Invalid einsum specification: multiple '->' found");
        }

        // Parse inputs (left side)
        let input_str = parts[0].trim();
        if input_str.is_empty() {
            bail!("No input specifications provided");
        }

        let inputs: Vec<String> = input_str.split(',').map(|s| s.trim().to_string()).collect();

        if inputs.is_empty() {
            bail!("No input specifications after splitting by ','");
        }

        // Validate input subscripts
        for (i, input) in inputs.iter().enumerate() {
            if input.is_empty() {
                bail!("Input {} is empty", i);
            }
            if !input.chars().all(|c| c.is_ascii_lowercase()) {
                bail!(
                    "Input {} contains invalid characters (only lowercase a-z allowed)",
                    i
                );
            }
        }

        // Parse output (right side, or infer if not provided)
        let output = if parts.len() == 2 {
            let out = parts[1].trim().to_string();
            if !out.chars().all(|c| c.is_ascii_lowercase()) {
                bail!("Output contains invalid characters (only lowercase a-z allowed)");
            }
            out
        } else {
            // Infer output: all indices that appear in inputs, in order of first appearance
            Self::infer_output(&inputs)
        };

        // Collect all unique indices
        let mut all_indices = HashSet::new();
        for input in &inputs {
            for c in input.chars() {
                all_indices.insert(c);
            }
        }

        // Output indices
        let output_indices: HashSet<char> = output.chars().collect();

        // Validate output indices are subset of input indices
        for c in &output_indices {
            if !all_indices.contains(c) {
                bail!("Output index '{}' does not appear in any input", c);
            }
        }

        // Contracted indices = all indices - output indices
        let contracted_indices: HashSet<char> = all_indices
            .iter()
            .copied()
            .filter(|c| !output_indices.contains(c))
            .collect();

        Ok(Self {
            inputs,
            output,
            all_indices,
            contracted_indices,
            output_indices,
        })
    }

    /// Infer output from inputs (all unique indices in order of appearance)
    fn infer_output(inputs: &[String]) -> String {
        let mut seen = HashSet::new();
        let mut output = String::new();

        for input in inputs {
            for c in input.chars() {
                if seen.insert(c) {
                    output.push(c);
                }
            }
        }

        output
    }

    /// Get the number of inputs
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Check if a given index is contracted
    pub fn is_contracted(&self, index: char) -> bool {
        self.contracted_indices.contains(&index)
    }

    /// Get indices for a specific input
    pub fn get_input_indices(&self, input_idx: usize) -> Option<Vec<char>> {
        self.inputs.get(input_idx).map(|s| s.chars().collect())
    }
}

/// Validate shapes against an einsum specification
pub fn validate_shapes(spec: &EinsumSpec, shapes: &[Vec<usize>]) -> Result<()> {
    if spec.num_inputs() != shapes.len() {
        bail!(
            "Shape count mismatch: spec has {} inputs but {} shapes provided",
            spec.num_inputs(),
            shapes.len()
        );
    }

    // Build dimension map: index -> size
    let mut dim_map: HashMap<char, usize> = HashMap::new();

    for (input_idx, (input_spec, shape)) in spec.inputs.iter().zip(shapes.iter()).enumerate() {
        if input_spec.len() != shape.len() {
            bail!(
                "Input {}: spec has {} indices but shape has {} dimensions",
                input_idx,
                input_spec.len(),
                shape.len()
            );
        }

        for (c, &size) in input_spec.chars().zip(shape.iter()) {
            if let Some(&prev_size) = dim_map.get(&c) {
                if prev_size != size {
                    bail!(
                        "Dimension mismatch for index '{}': was {}, now {}",
                        c,
                        prev_size,
                        size
                    );
                }
            } else {
                dim_map.insert(c, size);
            }
        }
    }

    Ok(())
}

/// Compute output shape from einsum specification and input shapes
pub fn compute_output_shape(spec: &EinsumSpec, shapes: &[Vec<usize>]) -> Result<Vec<usize>> {
    validate_shapes(spec, shapes)?;

    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (input_spec, shape) in spec.inputs.iter().zip(shapes.iter()) {
        for (c, &size) in input_spec.chars().zip(shape.iter()) {
            dim_map.insert(c, size);
        }
    }

    // Build output shape
    let mut output_shape = Vec::new();
    for c in spec.output.chars() {
        let size = dim_map
            .get(&c)
            .ok_or_else(|| anyhow!("Output index '{}' not found in dimension map", c))?;
        output_shape.push(*size);
    }

    Ok(output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        assert_eq!(spec.inputs, vec!["ij", "jk"]);
        assert_eq!(spec.output, "ik");
        assert!(spec.is_contracted('j'));
        assert!(!spec.is_contracted('i'));
        assert!(!spec.is_contracted('k'));
    }

    #[test]
    fn test_parse_batched_matmul() {
        let spec = EinsumSpec::parse("bij,bjk->bik").unwrap();
        assert_eq!(spec.inputs.len(), 2);
        assert_eq!(spec.output, "bik");
        assert!(spec.is_contracted('j'));
        assert_eq!(spec.contracted_indices.len(), 1);
    }

    #[test]
    fn test_parse_trace() {
        let spec = EinsumSpec::parse("ii->").unwrap();
        assert_eq!(spec.inputs, vec!["ii"]);
        assert_eq!(spec.output, "");
        assert!(spec.is_contracted('i'));
    }

    #[test]
    fn test_parse_outer_product() {
        let spec = EinsumSpec::parse("i,j->ij").unwrap();
        assert_eq!(spec.inputs, vec!["i", "j"]);
        assert_eq!(spec.output, "ij");
        assert_eq!(spec.contracted_indices.len(), 0);
    }

    #[test]
    fn test_parse_infer_output() {
        let spec = EinsumSpec::parse("ij,jk").unwrap();
        assert_eq!(spec.output, "ijk");
    }

    #[test]
    fn test_parse_three_tensors() {
        let spec = EinsumSpec::parse("ijk,jkl,klm->ilm").unwrap();
        assert_eq!(spec.inputs.len(), 3);
        assert_eq!(spec.output, "ilm");
        assert!(spec.is_contracted('j'));
        assert!(spec.is_contracted('k'));
        assert!(!spec.is_contracted('i'));
    }

    #[test]
    fn test_parse_invalid_empty() {
        assert!(EinsumSpec::parse("").is_err());
    }

    #[test]
    fn test_parse_invalid_uppercase() {
        assert!(EinsumSpec::parse("iJ,jk->ik").is_err());
    }

    #[test]
    fn test_parse_invalid_output_index() {
        assert!(EinsumSpec::parse("ij,jk->iz").is_err());
    }

    #[test]
    fn test_validate_shapes_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![2, 3], vec![3, 4]];
        assert!(validate_shapes(&spec, &shapes).is_ok());
    }

    #[test]
    fn test_validate_shapes_mismatch() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![2, 3], vec![4, 5]]; // j dimension mismatch
        assert!(validate_shapes(&spec, &shapes).is_err());
    }

    #[test]
    fn test_compute_output_shape_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![2, 3], vec![3, 4]];
        let output = compute_output_shape(&spec, &shapes).unwrap();
        assert_eq!(output, vec![2, 4]);
    }

    #[test]
    fn test_compute_output_shape_batched() {
        let spec = EinsumSpec::parse("bij,bjk->bik").unwrap();
        let shapes = vec![vec![5, 2, 3], vec![5, 3, 4]];
        let output = compute_output_shape(&spec, &shapes).unwrap();
        assert_eq!(output, vec![5, 2, 4]);
    }
}
