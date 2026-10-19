//! Tests of the LDLT factorizations of the symmetric 3x3 and 6x6 matrices, in `f64` and `f32`.

mod common;

use common::*;
use featherstone_algebra::float::Float;
use featherstone_algebra::linalg::{Mat3, Sym3, Sym6};

/// Random cases per test.
const TRIALS: usize = 200;


fn assert_close(
    what: &str,
    err: f64,
    tol: f64,
)
{
    assert!(err <= tol, "{what} : error {err:e} > {tol:e}");
}

/// LDLT solves of random symmetric positive definite 3x3 and 6x6 matrices `B B^T + 0.1`, with positive pivots.
fn ldlt_solves_match_the_original_matrix<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(7);
    for _ in 0..TRIALS {
        let mut b3 = [[0.0; 3]; 3];
        b3.iter_mut().flatten().for_each(|x| *x = rng.uniform(-1.0, 1.0));
        let mut a3 = mat_mul(&b3, &transpose(&b3));
        (0..3).for_each(|i| a3[i][i] += 0.1);
        let s3 = Sym3::from_mat3(&Mat3(a3)).cast::<T>();
        let x: [f64; 3] = core::array::from_fn(|_| rng.uniform(-1.0, 1.0));
        let rhs = mat_vec(&a3, &x).map(T::from_f64);
        let f3 = s3.ldlt();
        assert!(f3.pivots.iter().all(|d| *d > T::ZERO));
        assert_close("Sym3 solve", rel_err(&arr(f3.solve_array(rhs)), &x), tol);

        let mut b6 = [[0.0; 6]; 6];
        b6.iter_mut().flatten().for_each(|x| *x = rng.uniform(-1.0, 1.0));
        let mut a6 = mat_mul(&b6, &transpose(&b6));
        (0..6).for_each(|i| a6[i][i] += 0.1);
        let mut s6 = Sym6::<T>::zero();
        for (i, row) in a6.iter().enumerate() {
            for (j, x) in row.iter().enumerate().take(i + 1) {
                s6.set(i, j, T::from_f64(*x));
            }
        }
        let x: [f64; 6] = core::array::from_fn(|_| rng.uniform(-1.0, 1.0));
        let rhs = mat_vec(&a6, &x).map(T::from_f64);
        let f6 = s6.ldlt();
        assert!((0..6).all(|i| f6.pivot(i) > T::ZERO));
        assert_close("Sym6 solve", rel_err(&arr(f6.solve_array(rhs)), &x), tol);
    }
}
// The 6x6 test matrices have condition numbers up to about 1e3, hence the larger f32 tolerance.
crate::both!(ldlt_solves_match_the_original_matrix, 1e-11, 2e-3);
