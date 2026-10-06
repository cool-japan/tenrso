//! Shape and reduction edge cases: odd element counts for `random_normal`,
//! reductions over an axis of length 0, and the layout that `flatten` and the
//! `atleast_*` helpers return.

use scirs2_core::ndarray_ext::{Array, Axis};
use tenrso_core::DenseND;

/// `[2, 3]` holding `0..6`, with its axes swapped by `into_permuted`: a `[3, 2]`
/// tensor whose buffer is not in standard layout.
fn permuted() -> DenseND<f64> {
    DenseND::from_vec((0..6).map(f64::from).collect(), &[2, 3])
        .expect("2 * 3 elements")
        .into_permuted(&[1, 0])
        .expect("a permutation of two axes")
}

/// `[0, 1, 2, 3]` as a 1-D tensor whose only axis has a negative stride.
fn reversed_vector() -> DenseND<f64> {
    let mut data = Array::from_vec(vec![3.0, 2.0, 1.0, 0.0]);
    data.invert_axis(Axis(0));
    DenseND::from_array(data.into_dyn())
}

/// The shape, the logical order and whether the result is in standard layout
/// (which `as_slice` needs).
fn summary(t: &DenseND<f64>) -> (Vec<usize>, Vec<f64>, bool) {
    (t.shape().to_vec(), t.to_vec(), t.is_contiguous())
}

#[test]
fn random_normal_fills_every_element_count() {
    for shape in [&[1usize][..], &[3], &[3, 5], &[2, 3], &[4], &[0]] {
        let t = DenseND::<f64>::random_normal(shape, 0.0, 1.0);
        assert_eq!(t.shape(), shape);
        assert_eq!(t.len(), shape.iter().product::<usize>());
    }
}

#[test]
fn softmax_over_an_empty_axis_is_an_error() {
    let matrix = DenseND::<f64>::zeros(&[3, 0]);
    assert!(matrix.softmax(1).is_err());
    assert!(matrix.log_softmax(1).is_err());
}

#[test]
fn softmax_of_an_empty_vector_is_an_error() {
    let vector = DenseND::<f64>::zeros(&[0]);
    assert!(vector.softmax(0).is_err());
    assert!(vector.log_softmax(0).is_err());
}

#[test]
fn softmax_needing_no_maximum_still_succeeds() {
    // Along axis 0 of `[3, 0]`, and along either axis of `[0, 0]`, there is no
    // output position, so no maximum has to be taken.
    let matrix = DenseND::<f64>::zeros(&[3, 0]);
    assert_eq!(
        matrix.softmax(0).expect("no output position").shape(),
        &[3, 0]
    );
    let empty = DenseND::<f64>::zeros(&[0, 0]);
    assert_eq!(
        empty.softmax(1).expect("no output position").shape(),
        &[0, 0]
    );
    assert_eq!(
        empty.log_softmax(0).expect("no output position").shape(),
        &[0, 0]
    );
}

#[test]
fn flatten_matches_reshape_for_every_layout() {
    for t in [permuted(), reversed_vector(), DenseND::from_elem(&[], 5.0)] {
        let flat = t.flatten();
        assert_eq!(
            summary(&flat),
            summary(&t.reshape(&[t.len()]).expect("same count"))
        );
        assert!(flat.is_contiguous());
        assert_eq!(flat.try_as_slice(), Some(&t.to_vec()[..]));
    }
    assert_eq!(
        permuted().flatten().to_vec(),
        vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    );
}

#[test]
fn atleast_helpers_match_reshape_and_stay_standard_layout() {
    let scalar = DenseND::from_elem(&[], 5.0);
    assert_eq!(
        summary(&scalar.atleast_1d()),
        summary(&scalar.reshape(&[1]).expect("one element"))
    );
    assert_eq!(
        summary(&scalar.atleast_2d()),
        summary(&scalar.reshape(&[1, 1]).expect("one element"))
    );
    assert_eq!(
        summary(&scalar.atleast_3d()),
        summary(&scalar.reshape(&[1, 1, 1]).expect("one element"))
    );

    let vector = reversed_vector();
    assert_eq!(
        summary(&vector.atleast_2d()),
        summary(&vector.reshape(&[1, 4]).expect("same count"))
    );
    assert_eq!(
        summary(&vector.atleast_3d()),
        summary(&vector.reshape(&[1, 4, 1]).expect("same count"))
    );
    assert_eq!(vector.atleast_2d().to_vec(), vec![0.0, 1.0, 2.0, 3.0]);

    let matrix = permuted();
    assert_eq!(
        summary(&matrix.atleast_3d()),
        summary(&matrix.reshape(&[3, 2, 1]).expect("same count"))
    );
    for t in [
        scalar.atleast_1d(),
        scalar.atleast_2d(),
        scalar.atleast_3d(),
        vector.atleast_2d(),
        vector.atleast_3d(),
        matrix.atleast_3d(),
    ] {
        assert!(t.is_contiguous());
        assert_eq!(t.as_slice(), &t.to_vec()[..]);
    }
}
