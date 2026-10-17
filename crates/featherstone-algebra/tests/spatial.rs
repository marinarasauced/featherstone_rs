//! Tests of the spatial vector algebra against explicit 6x6 matrices and closed forms, in 'f64' and 'f32'.

mod common;

use common::*;
use robot_algebra::float::Float;
use robot_algebra::linalg::{Sym3, Sym6};
use robot_algebra::spatial::{Axis, AxisRotation, AxisX, AxisY, AxisZ, Force, Mat3, Motion, Quat, Rotation, Vec3, Xform};

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

/// Force transform 'X*' of the motion transform 'X' : for 'X = [E, 0; -E r~, E]', 'X* = [E, -E r~; 0, E]'.
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

/// Coordinate rotation 'E = R_k(q)^T' in closed form (spatial_v2 'rx', 'ry', 'rz').
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

/// Motion cross product matrix 'crm(m) = [w~, 0; v~, w~]'.
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

/// Checks the operations of one coordinate axis 'A' at a random angle.
fn check_axis<T: Float, A: Axis>(
    rng: &mut Rng,
    tol: f64,
)
{
    let q = rng.uniform(-4.0, 4.0);
    let rot = AxisRotation::<T, A>::from_angle(T::from_f64(q));
    let e = rot.matrix();
    assert_close("E closed form", max_abs_diff_mat(&to_f64(&e.0), &featherstone_rot(A::INDEX, q)), tol);
    let v: Vec3<T> = rng.vec3(1.0);
    let s = Sym3::from_mat3(&rng.rotation::<T>());
    let unit = Vec3::from_array(core::array::from_fn(|i| if i == A::INDEX { T::ONE } else { T::ZERO }));
    let checks = [
        ("E v", rot.apply(v), e.mul_vec(v)),
        ("E^T v", rot.apply_t(v), e.tr_mul_vec(v)),
        ("a x e_k", A::cross(v), v.cross(unit)),
        ("S e_k", A::column(&s), s.mul_vec(unit)),
        ("v + x e_k", A::add(v, T::ONE), v + unit),
    ];
    for (what, got, want) in checks {
        assert_close(what, max_abs_diff(&arr(got.to_array()), &arr(want.to_array())), tol);
    }
    assert_eq!(A::get(v), v.to_array()[A::INDEX]);
    let got = rot.congruence_t(&s).to_mat3();
    let want = e.congruence_t(&s).to_mat3();
    assert_close("E^T S E", max_abs_diff_mat(&to_f64(&got.0), &to_f64(&want.0)), tol);
}


// -------------------------------------------------------------------------------
// Tests
// ===============================================================================


/// Composition, inverse, and each motion and force transform against the explicit 6x6 matrices.
fn xform_matrices<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(1);
    for _ in 0..TRIALS {
        let a: Xform<T> = rng.xform();
        let b: Xform<T> = rng.xform();
        let xa = to_f64(&a.to_array());
        let xb = to_f64(&b.to_array());
        // Composition : 'A' then 'B' is 'X_B X_A'.
        let ab = to_f64(&a.then(&b).to_array());
        assert_close("composition", max_abs_diff_mat(&ab, &mat_mul(&xb, &xa)), tol);
        // Inverse : 'X^-1 X = 1'.
        let inv = to_f64(&a.inverse().to_array());
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
            ("X m", arr(a.apply_motion(m).to_array()), mat_vec(&xa, &mv)),
            ("X^-1 m", arr(a.inv_apply_motion(m).to_array()), mat_vec(&inv, &mv)),
            ("X* f", arr(a.apply_force(f).to_array()), mat_vec(&star(&xa), &fv)),
            ("X^T f", arr(a.inv_apply_force(f).to_array()), mat_vec(&transpose(&xa), &fv)),
        ];
        for (what, got, want) in checks {
            assert_close(what, max_abs_diff(&got, &want), tol);
        }
    }
}
crate::both!(xform_matrices, 1e-14, 2e-6);

/// Power does not depend on the coordinates : '(X m)^T (X* f) = m^T f' and '(X m)^T f = m^T (X^T f)'.
fn power_invariance<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(2);
    for _ in 0..TRIALS {
        let x: Xform<T> = rng.xform();
        let m: Motion<T> = rng.motion(1.0);
        let f: Force<T> = rng.force(1.0);
        let p0 = m.dot(f).to_f64();
        assert_close("(X m)^T X* f", (x.apply_motion(m).dot(x.apply_force(f)).to_f64() - p0).abs(), tol);
        let lhs = x.apply_motion(m).dot(f).to_f64();
        let rhs = m.dot(x.inv_apply_force(f)).to_f64();
        assert_close("(X m)^T f = m^T X^T f", (lhs - rhs).abs(), tol);
    }
}
crate::both!(power_invariance, 1e-14, 2e-6);

/// Inertia transform 'X^T I X' against the product of explicit matrices, for a general and a joint transform.
fn inertia_transform<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(3);
    for _ in 0..TRIALS {
        let inertia = rng.inertia::<T>();
        let ia = to_f64(&inertia.to_array());
        assert_close("to_sym6", max_abs_diff_mat(&to_f64(&inertia.to_sym6().to_array()), &ia), 0.0);
        let x: Xform<T> = rng.xform();
        let xm = to_f64(&x.to_array());
        let want = mat_mul(&transpose(&xm), &mat_mul(&ia, &xm));
        let parent = x.inv_transform_inertia(&inertia);
        assert_close("X^T I X", max_abs_diff_mat(&to_f64(&parent.to_array()), &want), tol);
        // 'X^T (I (X m)) = (X^T I X) m'.
        let m: Motion<T> = rng.motion(1.0);
        let lhs = x.inv_apply_force(inertia.mul_motion(x.apply_motion(m)));
        let rhs = parent.mul_motion(m);
        assert_close("X^T I X m", max_abs_diff(&arr(lhs.to_array()), &arr(rhs.to_array())), tol);
        // Same with a joint transform (rotation about one axis).
        let q = rng.scalar::<T>(3.0);
        let jx = Xform { rot: AxisRotation::<T, AxisY>::from_angle(q), r: x.r };
        let want = jx.to_general().inv_transform_inertia(&inertia);
        let got = jx.inv_transform_inertia(&inertia);
        assert_close("joint X^T I X", max_abs_diff_mat(&to_f64(&got.to_array()), &to_f64(&want.to_array())), tol);
    }
}
crate::both!(inertia_transform, 1e-14, 2e-6);

/// Rotations about the x, y, and z axes against closed forms and general matrices.
fn axis_rotations<T: Float>(
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
crate::both!(axis_rotations, 1e-15, 1e-6);

/// Motion and force cross products against 'crm' and 'crf = -crm^T', and the force at a point.
fn cross_products<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(5);
    for _ in 0..TRIALS {
        let m: Motion<T> = rng.motion(1.0);
        let n: Motion<T> = rng.motion(1.0);
        let f: Force<T> = rng.force(1.0);
        let c = crm(&Motion::<f64>::from_array(arr(m.to_array())));
        assert_close("crm", max_abs_diff(&arr(m.cross(n).to_array()), &mat_vec(&c, &arr(n.to_array()))), tol);
        let mut crf = transpose(&c);
        crf.iter_mut().flatten().for_each(|x| *x = -*x);
        assert_close("crf", max_abs_diff(&arr(m.cross_force(f).to_array()), &mat_vec(&crf, &arr(f.to_array()))), tol);
        // Force at a point : '[p x F; F]'.
        let p: Vec3<T> = rng.vec3(1.0);
        let fp = Force::at_point(p, f.lin);
        assert_eq!(fp.lin, f.lin);
        assert_close("at_point", max_abs_diff(&arr(fp.ang.to_array()), &arr(p.cross(f.lin).to_array())), 0.0);
    }
}
crate::both!(cross_products, 1e-15, 1e-6);

/// Quaternion product, rotation matrix, rotation of vectors, exponential map (against the Rodrigues formula), and one integration step.
fn quaternion<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(6);
    for _ in 0..TRIALS {
        let a: Quat<T> = rng.quat();
        let b: Quat<T> = rng.quat();
        let ra = a.to_mat3();
        // 'R(a * b) = R(a) R(b)'.
        let ab = to_f64(&(a * b).to_mat3().0);
        assert_close("R(a * b)", max_abs_diff_mat(&ab, &to_f64(&ra.mul_mat(&b.to_mat3()).0)), tol);
        // 'R' is orthonormal.
        let rtr = to_f64(&ra.transpose().mul_mat(&ra).0);
        assert_close("R^T R", max_abs_diff_mat(&rtr, &to_f64(&Mat3::<T>::identity().0)), tol);
        let v: Vec3<T> = rng.vec3(1.0);
        assert_close("rotate", max_abs_diff(&arr(a.rotate(v).to_array()), &arr(ra.mul_vec(v).to_array())), tol);
        assert_close("rotate_inv", max_abs_diff(&arr(a.rotate_inv(v).to_array()), &arr(ra.tr_mul_vec(v).to_array())), tol);
        // Exponential map against the Rodrigues formula 'R = 1 + sin(theta) K + (1 - cos(theta)) K^2', also for tiny angles.
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
        assert_close("exp map", max_abs_diff_mat(&to_f64(&q.to_mat3().0), &rodrigues), tol);
        let qn = (q.w * q.w + q.vec().norm_squared()).to_f64();
        assert_close("exp map norm", (qn - 1.0).abs(), tol);
        // One step at body angular velocity 'w' for time 'h' rotates by 'w h' in body coordinates.
        let w: Vec3<T> = rng.vec3(5.0);
        let h = T::from_f64(1e-3);
        let next = a.integrate(w, h).to_mat3();
        let want = ra.mul_mat(&Quat::from_rotation_vector(w * h).to_mat3());
        assert_close("integrate", max_abs_diff_mat(&to_f64(&next.0), &to_f64(&want.0)), tol);
    }
}
crate::both!(quaternion, 1e-14, 2e-6);

/// LDLT solves of random symmetric positive definite 3x3 and 6x6 matrices 'B B^T + 0.1', with positive pivots.
fn ldlt<T: Float>(
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
        assert!(f3.d.iter().all(|d| *d > T::ZERO));
        assert_close("Sym3 solve", rel_err(&arr(f3.solve(rhs)), &x), tol);

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
crate::both!(ldlt, 1e-11, 2e-3);
