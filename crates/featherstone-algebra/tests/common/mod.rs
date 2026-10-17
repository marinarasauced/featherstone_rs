//! Test helpers : a small deterministic random number generator, random spatial quantities, and dense 'f64' matrix operations to compare against.

#![allow(dead_code)]

use featherstone_algebra::float::Float;
use featherstone_algebra::linalg::Sym3;
use featherstone_algebra::spatial::{Force, Mat3, Motion, Quat, RigidInertia, Vec3, Xform};


// ===============================================================================
// Random numbers.
// -------------------------------------------------------------------------------


/// SplitMix64 random number generator : the same sequence for the same seed on every machine.
pub struct Rng(u64);

impl Rng {

    /// Generator with seed 'seed'.
    pub fn new(
        seed: u64,
    ) -> Self
    {
        Self(seed)
    }

    /// Next 64 random bits.
    pub fn next_u64(
        &mut self,
    ) -> u64
    {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in '[lo, hi)'.
    pub fn uniform(
        &mut self,
        lo: f64,
        hi: f64,
    ) -> f64
    {
        lo + (hi - lo) * (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in '[-scale, scale)', in float type 'T'.
    pub fn scalar<T: Float>(
        &mut self,
        scale: f64,
    ) -> T
    {
        T::from_f64(self.uniform(-scale, scale))
    }

    /// Vector with entries uniform in '[-scale, scale)'.
    pub fn vec3<T: Float>(
        &mut self,
        scale: f64,
    ) -> Vec3<T>
    {
        Vec3::new(self.scalar(scale), self.scalar(scale), self.scalar(scale))
    }

    /// Motion vector with entries uniform in '[-scale, scale)'.
    pub fn motion<T: Float>(
        &mut self,
        scale: f64,
    ) -> Motion<T>
    {
        Motion::new(self.vec3(scale), self.vec3(scale))
    }

    /// Force vector with entries uniform in '[-scale, scale)'.
    pub fn force<T: Float>(
        &mut self,
        scale: f64,
    ) -> Force<T>
    {
        Force::new(self.vec3(scale), self.vec3(scale))
    }

    /// Random unit quaternion.
    pub fn quat<T: Float>(
        &mut self,
    ) -> Quat<T>
    {
        Quat::new(self.scalar(1.0), self.scalar(1.0), self.scalar(1.0), self.scalar(1.0)).normalized()
    }

    /// Random rotation matrix.
    pub fn rotation<T: Float>(
        &mut self,
    ) -> Mat3<T>
    {
        self.quat::<T>().to_mat3()
    }

    /// Random transform : random rotation, child origin within 0.5 m.
    pub fn xform<T: Float>(
        &mut self,
    ) -> Xform<T>
    {
        Xform {
            rot: self.rotation(),
            r: self.vec3(0.5),
        }
    }

    /// Random physically valid rigid body inertia : mass in '[0.2, 2]' kg, center of mass within 0.2 m, principal moments in '[1e-4, 1e-2]' kg m^2 that satisfy the triangle inequality.
    pub fn inertia<T: Float>(
        &mut self,
    ) -> RigidInertia<T>
    {
        let mass = self.uniform(0.2, 2.0);
        let mut d = [0.0; 3];
        loop {
            for x in &mut d {
                *x = self.uniform(1e-4, 1e-2);
            }
            if d[0] + d[1] > d[2] && d[1] + d[2] > d[0] && d[0] + d[2] > d[1] {
                break;
            }
        }
        let q: Mat3<f64> = self.rotation();
        let principal = Sym3 { xx: d[0], yy: d[1], zz: d[2], ..Sym3::zero() };
        let i_com = q.mul_mat(&principal.to_mat3()).mul_mat(&q.transpose());
        let com: Vec3<f64> = self.vec3(0.2);
        RigidInertia::from_com(mass, com, Sym3::from_mat3(&i_com)).cast()
    }
}


// ===============================================================================
// Dense matrices in f64.
// -------------------------------------------------------------------------------


/// Dense 6x6 matrix.
pub type M6 = [[f64; 6]; 6];

/// Matrix of any float type as 'f64'.
pub fn to_f64<T: Float, const R: usize, const C: usize>(
    a: &[[T; C]; R],
) -> [[f64; C]; R]
{
    a.map(|row| row.map(|x| x.to_f64()))
}

/// Array of any float type as 'f64'.
pub fn arr<T: Float, const N: usize>(
    a: [T; N],
) -> [f64; N]
{
    a.map(|x| x.to_f64())
}

/// 'A B'.
pub fn mat_mul<const N: usize>(
    a: &[[f64; N]; N],
    b: &[[f64; N]; N],
) -> [[f64; N]; N]
{
    let mut out = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            out[i][j] = (0..N).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// 'A^T'.
pub fn transpose<const N: usize>(
    a: &[[f64; N]; N],
) -> [[f64; N]; N]
{
    let mut out = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            out[i][j] = a[j][i];
        }
    }
    out
}

/// 'A x'.
pub fn mat_vec<const N: usize>(
    a: &[[f64; N]; N],
    x: &[f64; N],
) -> [f64; N]
{
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = (0..N).map(|k| a[i][k] * x[k]).sum();
    }
    out
}

/// Largest absolute difference between two vectors.
pub fn max_abs_diff<const N: usize>(
    a: &[f64],
    b: &[f64; N],
) -> f64
{
    assert_eq!(a.len(), N);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Largest absolute difference between two matrices.
pub fn max_abs_diff_mat<const R: usize, const C: usize>(
    a: &[[f64; C]; R],
    b: &[[f64; C]; R],
) -> f64
{
    a.as_flattened().iter().zip(b.as_flattened()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Euclidean norm.
pub fn norm(
    a: &[f64],
) -> f64
{
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Relative error of 'a' against the reference 'b' : '|a - b| / max(|b|, 1)'.
pub fn rel_err(
    a: &[f64],
    b: &[f64],
) -> f64
{
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    norm(&d) / norm(b).max(1.0)
}

/// Solves 'A x = b' by Gaussian elimination with partial pivoting : an independent reference for the factorizations.
pub fn dense_solve<const N: usize>(
    a: &[[f64; N]; N],
    b: &[f64; N],
) -> [f64; N]
{
    let mut a = *a;
    let mut b = *b;
    for k in 0..N {
        let p = (k..N).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs())).unwrap();
        a.swap(k, p);
        b.swap(k, p);
        for i in k + 1..N {
            let f = a[i][k] / a[k][k];
            for j in k..N {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = [0.0; N];
    for i in (0..N).rev() {
        let s: f64 = (i + 1..N).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    x
}

/// Runs the check 'name::<f64>' and 'name::<f32>' as two tests, each with its tolerance.
#[macro_export]
macro_rules! both {
    ($name:ident, $tol64:expr, $tol32:expr) => {
        mod $name {
            #[test]
            fn f64() {
                super::$name::<f64>($tol64);
            }

            #[test]
            fn f32() {
                super::$name::<f32>($tol32);
            }
        }
    };
}
