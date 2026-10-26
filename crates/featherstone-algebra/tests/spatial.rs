//! Tests of the spatial vector algebra against explicit 6x6 matrices and closed forms, in `f64` and `f32`.

mod common;

use common::*;
use featherstone_algebra::float::Float;
use featherstone_algebra::linalg::{Mat3, Vec3};
use featherstone_algebra::rotation::{Axis, AxisRotation, AxisX, AxisY, AxisZ};
use featherstone_algebra::spatial::{Force, Motion, RigidBodyInertia, Transform};

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

/// Force transform `X*` of the motion transform `X` : for `X = [E, 0; -E r~, E]`, `X* = [E, -E r~; 0, E]`.
fn star(
    x: &M6,
) -> M6
{
    let mut out = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = x[i][j];
            out[3 + i][3 + j] = x[3 + i][3 + j];
            out[i][3 + j] = x[3 + i][j];
        }
    }
    out
}

/// Motion cross product matrix `crm(m) = [w~, 0; v~, w~]`.
fn crm(
    m: &Motion<f64>,
) -> M6
{
    let w = Mat3::skew(m.ang).0;
    let v = Mat3::skew(m.lin).0;
    let mut out = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = w[i][j];
            out[3 + i][3 + j] = w[i][j];
            out[3 + i][j] = v[i][j];
        }
    }
    out
}

/// Composition, inverse, and each motion and force transform against the explicit 6x6 matrices.
fn transforms_match_the_explicit_6x6_matrices<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(1);
    for _ in 0..TRIALS {
        let a: Transform<T> = rng.transform();
        let b: Transform<T> = rng.transform();
        let xa = to_f64(&a.to_dense());
        let xb = to_f64(&b.to_dense());
        // Composition : `A` then `B` is `X_B X_A`.
        let ab = to_f64(&a.then(&b).to_dense());
        assert_close("composition", max_abs_diff_mat(&ab, &mat_mul(&xb, &xa)), tol);
        // Inverse : `X^-1 X = 1`.
        let inv = to_f64(&a.inverse().to_dense());
        let mut eye = [[0.0; 6]; 6];
        for (i, row) in eye.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        assert_close("inverse", max_abs_diff_mat(&mat_mul(&inv, &xa), &eye), tol);
        // Each operation against its matrix.
        let m: Motion<T> = rng.motion(1.0);
        let f: Force<T> = rng.force(1.0);
        let (mv, fv) = (arr(m.to_array()), arr(f.to_array()));
        let checks = [
            ("X m", arr(a.motion_to_child(m).to_array()), mat_vec(&xa, &mv)),
            ("X^-1 m", arr(a.motion_to_parent(m).to_array()), mat_vec(&inv, &mv)),
            ("X* f", arr(a.force_to_child(f).to_array()), mat_vec(&star(&xa), &fv)),
            ("X^T f", arr(a.force_to_parent(f).to_array()), mat_vec(&transpose(&xa), &fv)),
        ];
        for (what, got, want) in checks {
            assert_close(what, max_abs_diff(&got, &want), tol);
        }
    }
}
crate::both!(transforms_match_the_explicit_6x6_matrices, 1e-14, 2e-6);

/// Power does not depend on the coordinates : `(X m)^T (X* f) = m^T f` and `(X m)^T f = m^T (X^T f)`.
fn transforms_keep_the_power_of_a_force_on_a_motion<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(2);
    for _ in 0..TRIALS {
        let x: Transform<T> = rng.transform();
        let m: Motion<T> = rng.motion(1.0);
        let f: Force<T> = rng.force(1.0);
        let p0 = m.power(f).to_f64();
        assert_close("(X m)^T X* f", (x.motion_to_child(m).power(x.force_to_child(f)).to_f64() - p0).abs(), tol);
        let lhs = x.motion_to_child(m).power(f).to_f64();
        let rhs = m.power(x.force_to_parent(f)).to_f64();
        assert_close("(X m)^T f = m^T X^T f", (lhs - rhs).abs(), tol);
    }
}
crate::both!(transforms_keep_the_power_of_a_force_on_a_motion, 1e-14, 2e-6);

/// Inertia transform `X^T I X` against the product of explicit matrices, for a general and a joint transform.
fn inertia_transforms_match_the_explicit_6x6_matrices<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(3);
    for _ in 0..TRIALS {
        let inertia = rng.inertia::<T>();
        let ia = to_f64(&inertia.to_dense());
        assert_close("to_sym6", max_abs_diff_mat(&to_f64(&inertia.to_sym6().to_dense()), &ia), 0.0);
        let x: Transform<T> = rng.transform();
        let xm = to_f64(&x.to_dense());
        let want = mat_mul(&transpose(&xm), &mat_mul(&ia, &xm));
        let parent = x.inertia_to_parent(&inertia);
        assert_close("X^T I X", max_abs_diff_mat(&to_f64(&parent.to_dense()), &want), tol);
        // `X^T (I (X m)) = (X^T I X) m`.
        let m: Motion<T> = rng.motion(1.0);
        let lhs = x.force_to_parent(inertia.mul_motion(x.motion_to_child(m)));
        let rhs = parent.mul_motion(m);
        assert_close("X^T I X m", max_abs_diff(&arr(lhs.to_array()), &arr(rhs.to_array())), tol);
        // Same with a joint transform (rotation about one axis).
        let q = rng.scalar::<T>(3.0);
        let jx = Transform { rotation: AxisRotation::<T, AxisY>::from_angle(q), origin: x.origin };
        let want = jx.with_mat3_rotation().inertia_to_parent(&inertia);
        let got = jx.inertia_to_parent(&inertia);
        assert_close("joint X^T I X", max_abs_diff_mat(&to_f64(&got.to_dense()), &to_f64(&want.to_dense())), tol);
    }
}
crate::both!(inertia_transforms_match_the_explicit_6x6_matrices, 1e-14, 2e-6);

/// The product of an inertia with the axis of a revolute joint against the column `k` of the explicit 6x6 matrix `I`, because `S = [e_k; 0]` selects that column.
fn inertia_times_an_axis_is_a_column_of_the_6x6_matrix<T: Float>(
    tol: f64,
)
{
    fn check<T: Float, A: Axis>(
        inertia: &RigidBodyInertia<T>,
        tol: f64,
    )
    {
        let dense = to_f64(&inertia.to_dense());
        let want: [f64; 6] = std::array::from_fn(|i| dense[i][A::INDEX]);
        let got = arr(inertia.mul_axis::<A>().to_array());
        assert_close("I S", max_abs_diff(&got, &want), tol);
    }
    let mut rng = Rng::new(4);
    for _ in 0..TRIALS {
        let inertia = rng.inertia::<T>();
        check::<T, AxisX>(&inertia, tol);
        check::<T, AxisY>(&inertia, tol);
        check::<T, AxisZ>(&inertia, tol);
    }
}
crate::both!(inertia_times_an_axis_is_a_column_of_the_6x6_matrix, 1e-14, 2e-6);

/// Motion and force cross products against `crm` and `crf = -crm^T`, and the force at a point.
fn cross_products_match_crm_and_crf<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(5);
    for _ in 0..TRIALS {
        let m: Motion<T> = rng.motion(1.0);
        let n: Motion<T> = rng.motion(1.0);
        let f: Force<T> = rng.force(1.0);
        let c = crm(&Motion::<f64>::from_array(arr(m.to_array())));
        assert_close("crm", max_abs_diff(&arr(m.cross_motion(n).to_array()), &mat_vec(&c, &arr(n.to_array()))), tol);
        let mut crf = transpose(&c);
        crf.iter_mut().flatten().for_each(|x| *x = -*x);
        assert_close("crf", max_abs_diff(&arr(m.cross_force(f).to_array()), &mat_vec(&crf, &arr(f.to_array()))), tol);
        // Force at a point : `[p x F; F]`.
        let p: Vec3<T> = rng.vec3(1.0);
        let fp = Force::from_point_force(p, f.lin);
        assert_eq!(fp.lin, f.lin);
        assert_close("from_point_force", max_abs_diff(&arr(fp.ang.to_array()), &arr(p.cross(f.lin).to_array())), 0.0);
    }
}
crate::both!(cross_products_match_crm_and_crf, 1e-15, 1e-6);
