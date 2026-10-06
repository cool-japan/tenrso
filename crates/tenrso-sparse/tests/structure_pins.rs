//! Exact outputs of the graph, ordering, reduction and CSF construction
//! routines on small hand-checked inputs.

use tenrso_sparse::graph::{dijkstra, is_bipartite, strongly_connected_components};
use tenrso_sparse::reductions::{max_axis, min_axis};
use tenrso_sparse::reordering::amd;
use tenrso_sparse::{CooTensor, CsrMatrix};

/// An `n × n` CSR matrix with one entry per `(row, col, value)`.
fn csr(n: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    let indices = entries.iter().map(|&(r, c, _)| vec![r, c]).collect();
    let values = entries.iter().map(|&(_, _, v)| v).collect();
    let coo = CooTensor::new(indices, values, vec![n, n]).expect("valid COO");
    CsrMatrix::from_coo(&coo).expect("valid CSR")
}

/// Both directions of every undirected edge, weight 1.
fn undirected(n: usize, edges: &[(usize, usize)]) -> CsrMatrix<f64> {
    let entries: Vec<(usize, usize, f64)> = edges
        .iter()
        .flat_map(|&(a, b)| [(a, b, 1.0), (b, a, 1.0)])
        .collect();
    csr(n, &entries)
}

#[test]
fn tarjan_components_come_out_in_completion_order() {
    // 0 -> 1 -> 2 -> 0, 2 -> 3, 3 <-> 4, and 5 on its own.
    let graph = csr(
        6,
        &[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 3, 1.0),
        ],
    );
    assert_eq!(
        strongly_connected_components(&graph),
        vec![vec![4, 3], vec![2, 1, 0], vec![5]]
    );
}

#[test]
fn bipartite_detection() {
    assert!(is_bipartite(&undirected(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0)]
    )));
    assert!(!is_bipartite(&undirected(3, &[(0, 1), (1, 2), (2, 0)])));
    // Two components, the second an odd cycle.
    assert!(!is_bipartite(&undirected(
        5,
        &[(0, 1), (2, 3), (3, 4), (4, 2)]
    )));
    assert!(is_bipartite(&undirected(3, &[])));
}

#[test]
fn dijkstra_distances() {
    let graph = csr(
        5,
        &[
            (0, 1, 4.0),
            (0, 2, 1.0),
            (2, 1, 2.0),
            (1, 3, 1.0),
            (3, 0, 7.0),
        ],
    );
    assert_eq!(
        dijkstra(&graph, 0),
        vec![Some(0.0), Some(3.0), Some(1.0), Some(4.0), None]
    );
}

#[test]
fn amd_orders_every_vertex_once() {
    // A path 0-1-2-3 and a star centred on 0, each with its diagonal.
    let path = csr(
        4,
        &[
            (0, 0, 2.0),
            (1, 1, 2.0),
            (2, 2, 2.0),
            (3, 3, 2.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
        ],
    );
    assert_eq!(amd(&path).expect("square"), vec![0, 1, 2, 3]);

    let star = undirected(4, &[(0, 1), (0, 2), (0, 3)]);
    assert_eq!(amd(&star).expect("square"), vec![1, 2, 0, 3]);
}

#[test]
fn axis_extrema_fold_in_implicit_zeros_only_for_partial_slices() {
    // Row 0 fully populated with negatives; row 1 holds one positive value.
    let tensor = CooTensor::new(
        vec![vec![0, 0], vec![0, 1], vec![0, 2], vec![1, 0]],
        vec![-1.0, -2.0, -3.0, 5.0],
        vec![2, 3],
    )
    .expect("valid COO");

    let max = max_axis(&tensor, 1).expect("axis in range");
    assert_eq!(max.to_dense().expect("dense").to_vec(), vec![-1.0, 5.0]);
    let min = min_axis(&tensor, 1).expect("axis in range");
    assert_eq!(min.to_dense().expect("dense").to_vec(), vec![-3.0, 0.0]);

    let max0 = max_axis(&tensor, 0).expect("axis in range");
    assert_eq!(
        max0.to_dense().expect("dense").to_vec(),
        vec![5.0, 0.0, 0.0]
    );
    let min0 = min_axis(&tensor, 0).expect("axis in range");
    assert_eq!(
        min0.to_dense().expect("dense").to_vec(),
        vec![-1.0, -2.0, -3.0]
    );
}

#[cfg(feature = "csf")]
#[test]
fn csf_fiber_pointers() {
    use tenrso_sparse::CsfTensor;

    let coo = CooTensor::new(
        vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![1, 1, 1],
            vec![1, 2, 0],
        ],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![2, 3, 2],
    )
    .expect("valid COO");
    let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).expect("valid CSF");
    // Level 0 groups the five entries by mode 0 (3 + 2), level 1 by mode 1
    // within each group (2 + 1 and 1 + 1), level 2 holds one leaf per entry.
    assert_eq!(csf.fptr(0), &[0, 3, 5]);
    assert_eq!(csf.fids(0), &[0, 1]);
    assert_eq!(csf.fptr(1), &[0, 2, 3, 4, 5]);
    assert_eq!(csf.fids(1), &[0, 1, 1, 2]);
    assert_eq!(csf.fptr(2), &[0, 2, 3, 4, 5]);
    assert_eq!(csf.fids(2), &[0, 1, 0, 1, 0]);
    assert_eq!(csf.vals(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
}
