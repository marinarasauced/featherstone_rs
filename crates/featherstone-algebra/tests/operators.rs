//! Tests of the operators and the casts that every linear type has : `Add`, `Sub`, `Neg`, `Mul<T>`, `AddAssign`, `SubAssign`, and `cast`, in `f64` and `f32`.
//!
//! Each type has the same six operators, so each is checked the same way : against the same operation on its entries.

mod common;

use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use common::*;
use featherstone_algebra::float::{Convert, Float};
use featherstone_algebra::linalg::{Mat3, Sym3, Sym6, Vec3};
use featherstone_algebra::rotation::Quat;
use featherstone_algebra::spatial::{Force, Motion, RigidBodyInertia};

/// Random cases per test.
const TRIALS: usize = 50;

/// Largest absolute difference of two lists of entries.
fn max_diff(
    got: &[f64],
    want: &[f64],
) -> f64
{
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max)
}

fn vec3_entries<T: Float>(x: &Vec3<T>) -> Vec<f64> { x.to_array().iter().map(|v| v.to_f64()).collect() }
fn mat3_entries<T: Float>(x: &Mat3<T>) -> Vec<f64> { x.0.iter().flatten().map(|v| v.to_f64()).collect() }
fn sym3_entries<T: Float>(x: &Sym3<T>) -> Vec<f64> { mat3_entries(&x.to_mat3()) }
fn sym6_entries<T: Float>(x: &Sym6<T>) -> Vec<f64> { x.to_dense().iter().flatten().map(|v| v.to_f64()).collect() }
fn motion_entries<T: Float>(x: &Motion<T>) -> Vec<f64> { x.to_array().iter().map(|v| v.to_f64()).collect() }
fn force_entries<T: Float>(x: &Force<T>) -> Vec<f64> { x.to_array().iter().map(|v| v.to_f64()).collect() }
fn inertia_entries<T: Float>(x: &RigidBodyInertia<T>) -> Vec<f64> { x.to_dense().iter().flatten().map(|v| v.to_f64()).collect() }

/// The six operators on `a` and `b` against the same operations on their entries `entries(x)`.
fn check_operators<T: Float, X>(
    a: X,
    b: X,
    entries: impl Fn(&X) -> Vec<f64>,
    tol: f64,
)
where
    X: Copy + Add<Output = X> + Sub<Output = X> + Neg<Output = X> + Mul<T, Output = X> + AddAssign + SubAssign,
{
    let (ea, eb) = (entries(&a), entries(&b));
    let k = T::from_f64(1.5);
    let by_entry = |f: &dyn Fn(f64, f64) -> f64| -> Vec<f64> { ea.iter().zip(&eb).map(|(x, y)| f(*x, *y)).collect() };
    let mut sum = a;
    sum += b;
    let mut difference = a;
    difference -= b;
    let checks = [
        ("a + b", entries(&(a + b)), by_entry(&|x, y| x + y)),
        ("a - b", entries(&(a - b)), by_entry(&|x, y| x - y)),
        ("-a", entries(&-a), by_entry(&|x, _| -x)),
        ("a * k", entries(&(a * k)), by_entry(&|x, _| x * 1.5)),
        ("a += b", entries(&sum), by_entry(&|x, y| x + y)),
        ("a -= b", entries(&difference), by_entry(&|x, y| x - y)),
    ];
    for (what, got, want) in checks {
        let err = max_diff(&got, &want);
        assert!(err <= tol, "{what} : error {err:e} > {tol:e}");
    }
}

/// `entries(x)` of a value in `f64` and of its cast to `f32`.
fn check_cast(
    exact: &[f64],
    cast: &[f64],
)
{
    let err = max_diff(exact, cast);
    assert!(err <= 1e-6 * exact.iter().fold(1.0_f64, |m, x| m.max(x.abs())), "cast error {err:e}");
}

/// A random symmetric 6x6 matrix.
fn sym6<T: Float>(
    rng: &mut Rng,
) -> Sym6<T>
{
    let mut s = Sym6::zero();
    for i in 0..6 {
        for j in 0..=i {
            s.set(i, j, rng.scalar(1.0));
        }
    }
    s
}

/// Every linear type has the same six operators.
fn linear_types_have_the_same_operators<T: Float>(
    tol: f64,
)
{
    let mut rng = Rng::new(11);
    for _ in 0..TRIALS {
        check_operators::<T, _>(rng.vec3(1.0), rng.vec3(1.0), vec3_entries, tol);
        check_operators::<T, _>(rng.rotation(), rng.rotation(), mat3_entries, tol);
        check_operators::<T, _>(Sym3::from_mat3(&rng.rotation::<T>()), Sym3::from_mat3(&rng.rotation::<T>()), sym3_entries, tol);
        check_operators::<T, _>(sym6::<T>(&mut rng), sym6::<T>(&mut rng), sym6_entries, tol);
        check_operators::<T, _>(rng.motion(1.0), rng.motion(1.0), motion_entries, tol);
        check_operators::<T, _>(rng.force(1.0), rng.force(1.0), force_entries, tol);
        check_operators::<T, _>(rng.inertia(), rng.inertia(), inertia_entries, tol);
    }
}
crate::both!(linear_types_have_the_same_operators, 1e-14, 2e-6);

/// `cast` exists on every type and keeps the value.
#[test]
fn every_type_has_a_cast_that_keeps_the_value() {
    let mut rng = Rng::new(12);
    for _ in 0..TRIALS {
        let v: Vec3<f64> = rng.vec3(1.0);
        check_cast(&vec3_entries(&v), &vec3_entries(&v.cast::<f32>()));
        let m: Mat3<f64> = rng.rotation();
        check_cast(&mat3_entries(&m), &mat3_entries(&m.cast::<f32>()));
        let s = Sym3::from_mat3(&m);
        check_cast(&sym3_entries(&s), &sym3_entries(&s.cast::<f32>()));
        let s6: Sym6<f64> = sym6(&mut rng);
        check_cast(&sym6_entries(&s6), &sym6_entries(&s6.cast::<f32>()));
        let q: Quat<f64> = rng.quat();
        let q32 = q.cast::<f32>();
        check_cast(&[q.w, q.x, q.y, q.z], &[q32.w.to_f64(), q32.x.to_f64(), q32.y.to_f64(), q32.z.to_f64()]);
        let motion: Motion<f64> = rng.motion(1.0);
        check_cast(&motion_entries(&motion), &motion_entries(&motion.cast::<f32>()));
        let force: Force<f64> = rng.force(1.0);
        check_cast(&force_entries(&force), &force_entries(&force.cast::<f32>()));
        let inertia: RigidBodyInertia<f64> = rng.inertia();
        check_cast(&inertia_entries(&inertia), &inertia_entries(&inertia.cast::<f32>()));
        assert_eq!(Convert::cast::<f32>(1.25_f64), 1.25_f32);
    }
}
