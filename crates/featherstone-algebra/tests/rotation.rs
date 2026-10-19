//! Tests of the quaternion and of the rotations about the coordinate axes, in `f64` and `f32`.

mod common;

use common::*;
use featherstone_algebra::float::Float;
use featherstone_algebra::linalg::{Mat3, Sym3, Vec3};
use featherstone_algebra::rotation::{Axis, AxisRotation, AxisX, AxisY, AxisZ, Quat, Rotation};

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

/// Coordinate rotation `E = R_k(q)^T` in closed form (spatial_v2 `rx`, `ry`, `rz`).
fn featherstone_rot(
    k: usize,
    q: f64,
) -> [[f64; 3]; 3]
{
    let (s, c) = q.sin_cos();
    match k {
        0 => [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]],
        1 => [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]],
        _ => [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

/// Checks the operations of one coordinate axis `A` at a random angle.
fn check_axis<T: Float, A: Axis>(
    rng: &mut Rng,
    tol: f64,
)
{
    let q = rng.uniform(-4.0, 4.0);
    let rot = AxisRotation::<T, A>::from_angle(T::from_f64(q));
    let e = rot.to_mat3();
    assert_close("E closed form", max_abs_diff_mat(&to_f64(&e.0), &featherstone_rot(A::INDEX, q)), tol);
    let v: Vec3<T> = rng.vec3(1.0);
    let s = Sym3::from_mat3(&rng.rotation::<T>());
    let unit = Vec3::from_array(core::array::from_fn(|i| if i == A::INDEX { T::ONE } else { T::ZERO }));
    let checks = [
        ("E v", rot.to_child(v), e.mul_vec(v)),
        ("E^T v", rot.to_parent(v), e.mul_vec_t(v)),
        ("a x e_k", A::cross_axis(v), v.cross(unit)),
        ("S e_k", A::symmetric_column(&s), s.mul_vec(unit)),
        ("v + x e_k", A::add_component(v, T::ONE), v + unit),
    ];
    for (what, got, want) in checks {
        assert_close(what, max_abs_diff(&arr(got.to_array()), &arr(want.to_array())), tol);
    }
    assert_eq!(A::component(v), v.to_array()[A::INDEX]);
    let got = rot.symmetric_to_parent(&s).to_mat3();
    let want = e.symmetric_to_parent(&s).to_mat3();
    assert_close("E^T S E", max_abs_diff_mat(&to_f64(&got.0), &to_f64(&want.0)), tol);
}

/// Rotations about the x, y, and z axes against closed forms and general matrices.
fn axis_rotations_match_closed_forms_and_general_matrices<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(4);
    for _ in 0..TRIALS {
        check_axis::<T, AxisX>(&mut rng, tol);
        check_axis::<T, AxisY>(&mut rng, tol);
        check_axis::<T, AxisZ>(&mut rng, tol);
    }
}
crate::both!(axis_rotations_match_closed_forms_and_general_matrices, 1e-15, 1e-6);

/// Quaternion product, rotation matrix, rotation of vectors, exponential map (against the Rodrigues formula), and one integration step.
fn quaternion_operations_match_matrices_and_the_rodrigues_formula<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(6);
    for _ in 0..TRIALS {
        let a: Quat<T> = rng.quat();
        let b: Quat<T> = rng.quat();
        let ra = a.to_rotation_matrix();
        // `R(a * b) = R(a) R(b)`.
        let ab = to_f64(&(a * b).to_rotation_matrix().0);
        assert_close("R(a * b)", max_abs_diff_mat(&ab, &to_f64(&ra.mul_mat(&b.to_rotation_matrix()).0)), tol);
        // `R` is orthonormal.
        let rtr = to_f64(&ra.transpose().mul_mat(&ra).0);
        assert_close("R^T R", max_abs_diff_mat(&rtr, &to_f64(&Mat3::<T>::identity().0)), tol);
        let v: Vec3<T> = rng.vec3(1.0);
        assert_close("to_world", max_abs_diff(&arr(a.to_world(v).to_array()), &arr(ra.mul_vec(v).to_array())), tol);
        assert_close("to_body", max_abs_diff(&arr(a.to_body(v).to_array()), &arr(ra.mul_vec_t(v).to_array())), tol);
        // Exponential map against the Rodrigues formula `R = 1 + sin(theta) K + (1 - cos(theta)) K^2`, also for tiny angles.
        let scale = [3.0, 1e-2, 1e-5, 1e-9][rng.next_u64() as usize % 4];
        let phi: Vec3<f64> = rng.vec3(scale);
        let theta = phi.norm();
        let k = Mat3::skew(phi * (1.0 / theta)).0;
        let k2 = mat_mul(&k, &k);
        let mut rodrigues = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rodrigues[i][j] = f64::from(i == j) + theta.sin() * k[i][j] + (1.0 - theta.cos()) * k2[i][j];
            }
        }
        let q = Quat::<T>::from_rotation_vector(phi.cast());
        assert_close("exp map", max_abs_diff_mat(&to_f64(&q.to_rotation_matrix().0), &rodrigues), tol);
        let qn = (q.w * q.w + q.vector().norm_squared()).to_f64();
        assert_close("exp map norm", (qn - 1.0).abs(), tol);
        // One step at body angular velocity `w` for time `h` rotates by `w h` in body coordinates.
        let w: Vec3<T> = rng.vec3(5.0);
        let h = T::from_f64(1e-3);
        let next = a.integrate(w, h).to_rotation_matrix();
        let want = ra.mul_mat(&Quat::from_rotation_vector(w * h).to_rotation_matrix());
        assert_close("integrate", max_abs_diff_mat(&to_f64(&next.0), &to_f64(&want.0)), tol);
    }
}
crate::both!(quaternion_operations_match_matrices_and_the_rodrigues_formula, 1e-14, 2e-6);
