//! Symmetric 3x3 and 6x6 matrices and their LDLT factorizations.
//!
//! This module depends only on [`Float`].
//!
//! `Sym3` holds rotational inertias and the 3x3 joint blocks of each leg.
//! `Sym6` holds the base blocks of the joint space inertia matrix and its Schur complement in 3x3 blocks.

use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::float::Float;

/// 3x1 vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3<T> {
    /// `x` component.
    pub x: T,
    /// `y` component.
    pub y: T,
    /// `z` component.
    pub z: T,
}

impl<T: Float> Vec3<T> {

    /// `(x, y, z)`.
    #[inline(always)]
    pub const fn new(
        x: T,
        y: T,
        z: T,
    ) -> Self
    {
        Self { x, y, z }
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(
    ) -> Self
    {
        Self::new(T::ZERO, T::ZERO, T::ZERO)
    }

    /// Cross product `a x b`.
    #[inline(always)]
    pub fn cross(
        self,
        rhs: Self,
    ) -> Self
    {
        Self::new(self.y * rhs.z - self.z * rhs.y, self.z * rhs.x - self.x * rhs.z, self.x * rhs.y - self.y * rhs.x)
    }

    /// Dot product `a . b`.
    #[inline(always)]
    pub fn dot(
        self,
        rhs: Self,
    ) -> T
    {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Norm `|a|`.
    #[inline(always)]
    pub fn norm(
        self,
    ) -> T
    {
        self.norm_squared().sqrt()
    }

    /// Squared norm `|a|^2`.
    #[inline(always)]
    pub fn norm_squared(
        self,
    ) -> T
    {
        self.dot(self)
    }

    /// Cast to the float type `U`.
    #[inline(always)]
    pub fn cast<U: Float>(
        self,
    ) -> Vec3<U>
    {
        Vec3::new(self.x.cast(), self.y.cast(), self.z.cast())
    }

    /// From `[x, y, z]`.
    #[inline(always)]
    pub fn from_array(
        a: [T; 3],
    ) -> Self
    {
        Self::new(a[0], a[1], a[2])
    }

    /// To `[x, y, z]`.
    #[inline(always)]
    pub fn to_array(
        self,
    ) -> [T; 3]
    {
        [self.x, self.y, self.z]
    }
}

impl_linear_ops!(Vec3 { x, y, z });

/// 3x3 matrix, row major.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat3<T>(pub [[T; 3]; 3]);

impl<T: Float> Mat3<T> {

    /// Identity matrix.
    #[inline(always)]
    pub fn identity(
    ) -> Self
    {
        let (o, z) = (T::ONE, T::ZERO);
        Self([[o, z, z], [z, o, z], [z, z, o]])
    }

    /// Cross product matrix of `a` : `skew(a) b = a x b`.
    ///
    /// The matrix `a x` of RBDA Chapter 2 ; `spatial_v2` calls it `skew`.
    #[inline(always)]
    pub fn skew(
        a: Vec3<T>,
    ) -> Self
    {
        let z = T::ZERO;
        Self([[z, -a.z, a.y], [a.z, z, -a.x], [-a.y, a.x, z]])
    }

    /// Zero matrix.
    #[inline(always)]
    pub fn zero(
    ) -> Self
    {
        Self([[T::ZERO; 3]; 3])
    }

    /// Column `j`.
    #[inline(always)]
    pub fn col(
        &self,
        j: usize,
    ) -> Vec3<T>
    {
        Vec3::new(self.0[0][j], self.0[1][j], self.0[2][j])
    }

    /// Entry `(i, j)`.
    #[inline(always)]
    pub fn get(
        &self,
        i: usize,
        j: usize,
    ) -> T
    {
        self.0[i][j]
    }

    /// Row `i`.
    #[inline(always)]
    pub fn row(
        &self,
        i: usize,
    ) -> Vec3<T>
    {
        Vec3::from_array(self.0[i])
    }

    /// Sets entry `(i, j)`.
    #[inline(always)]
    pub fn set(
        &mut self,
        i: usize,
        j: usize,
        x: T,
    )
    {
        self.0[i][j] = x;
    }

    /// `A B`.
    #[inline(always)]
    pub fn mul_mat(
        &self,
        b: &Self,
    ) -> Self
    {
        let mut out = [[T::ZERO; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = self.row(i).dot(b.col(j));
            }
        }
        Self(out)
    }

    /// `A v`.
    #[inline(always)]
    pub fn mul_vec(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(self.row(0).dot(v), self.row(1).dot(v), self.row(2).dot(v))
    }

    /// `A^T v`.
    #[inline(always)]
    pub fn mul_vec_t(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(self.col(0).dot(v), self.col(1).dot(v), self.col(2).dot(v))
    }

    /// `A^T`.
    #[inline(always)]
    pub fn transpose(
        &self,
    ) -> Self
    {
        let a = &self.0;
        Self([[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]])
    }

    /// Cast to the float type `U`.
    #[inline(always)]
    pub fn cast<U: Float>(
        &self,
    ) -> Mat3<U>
    {
        let mut out = [[U::ZERO; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = self.0[i][j].cast();
            }
        }
        Mat3(out)
    }

    /// Entries `f(a)` of each entry `a` of `self`.
    #[inline(always)]
    fn map(
        &self,
        f: impl Fn(T) -> T,
    ) -> Self
    {
        let mut out = [[T::ZERO; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = f(self.0[i][j]);
            }
        }
        Self(out)
    }

    /// Entries `f(a, b)` of each pair of entries `a` of `self` and `b` of `rhs`.
    #[inline(always)]
    fn zip_with(
        &self,
        rhs: &Self,
        f: impl Fn(T, T) -> T,
    ) -> Self
    {
        let mut out = [[T::ZERO; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = f(self.0[i][j], rhs.0[i][j]);
            }
        }
        Self(out)
    }
}

impl<T: Float> Add for Mat3<T> {
    type Output = Self;
    #[inline(always)]
    fn add(
        self,
        rhs: Self,
    ) -> Self
    {
        self.zip_with(&rhs, |a, b| a + b)
    }
}

impl<T: Float> Sub for Mat3<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(
        self,
        rhs: Self,
    ) -> Self
    {
        self.zip_with(&rhs, |a, b| a - b)
    }
}

impl<T: Float> Neg for Mat3<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(
        self,
    ) -> Self
    {
        self.map(|a| -a)
    }
}

impl<T: Float> Mul<T> for Mat3<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(
        self,
        k: T,
    ) -> Self
    {
        self.map(|a| a * k)
    }
}

impl<T: Float> AddAssign for Mat3<T> {
    #[inline(always)]
    fn add_assign(
        &mut self,
        rhs: Self,
    )
    {
        *self = *self + rhs;
    }
}

impl<T: Float> SubAssign for Mat3<T> {
    #[inline(always)]
    fn sub_assign(
        &mut self,
        rhs: Self,
    )
    {
        *self = *self - rhs;
    }
}

/// 3x3 matrix, symmetric.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sym3<T> {
    /// Entry (0, 0).
    pub xx: T,
    /// Entry (1, 1).
    pub yy: T,
    /// Entry (2, 2).
    pub zz: T,
    /// Entries (0, 1) and (1, 0).
    pub xy: T,
    /// Entries (0, 2) and (2, 0).
    pub xz: T,
    /// Entries (1, 2) and (2, 1).
    pub yz: T,
}

impl<T: Float> Sym3<T> {

    /// Zero matrix.
    #[inline(always)]
    pub fn zero(
    ) -> Self
    {
        Self {
            xx: T::ZERO,
            yy: T::ZERO,
            zz: T::ZERO,
            xy: T::ZERO,
            xz: T::ZERO,
            yz: T::ZERO,
        }
    }

    /// Entry `(i, j)`.
    #[inline(always)]
    pub fn get(
        &self,
        i: usize,
        j: usize,
    ) -> T
    {
        match (i.min(j), i.max(j)) {
            (0, 0) => self.xx,
            (1, 1) => self.yy,
            (2, 2) => self.zz,
            (0, 1) => self.xy,
            (0, 2) => self.xz,
            (1, 2) => self.yz,
            _ => panic!("Sym3 index out of range"),
        }
    }

    /// Sets entries `(i, j)` and `(j, i)`.
    pub fn set(
        &mut self,
        i: usize,
        j: usize,
        x: T,
    )
    {
        match (i.min(j), i.max(j)) {
            (0, 0) => self.xx = x,
            (1, 1) => self.yy = x,
            (2, 2) => self.zz = x,
            (0, 1) => self.xy = x,
            (0, 2) => self.xz = x,
            (1, 2) => self.yz = x,
            _ => panic!("Sym3 index out of range"),
        }
    }

    /// LDLT factorization of a positive definite matrix.
    #[inline(always)]
    pub fn ldlt(
        &self,
    ) -> Ldl3<T>
    {
        let d0 = self.xx;
        let d0_inv = d0.recip();
        let l10 = self.xy * d0_inv;
        let l20 = self.xz * d0_inv;
        let d1 = self.yy - l10 * self.xy;
        let d1_inv = d1.recip();
        let l21 = (self.yz - l20 * self.xy) * d1_inv;
        let d2 = self.zz - l20 * self.xz - l21 * (self.yz - l20 * self.xy);
        Ldl3 {
            l10,
            l20,
            l21,
            pivots: [d0, d1, d2],
            pivots_inv: [d0_inv, d1_inv, d2.recip()],
        }
    }

    /// `S x` for the array `x`.
    #[inline(always)]
    pub fn mul_array(
        &self,
        x: [T; 3],
    ) -> [T; 3]
    {
        self.mul_vec(Vec3::from_array(x)).to_array()
    }

    /// `S v`.
    #[inline(always)]
    pub fn mul_vec(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3 {
            x: self.xx * v.x + self.xy * v.y + self.xz * v.z,
            y: self.xy * v.x + self.yy * v.y + self.yz * v.z,
            z: self.xz * v.x + self.yz * v.y + self.zz * v.z,
        }
    }

    /// Cast to the float type `U`.
    #[inline(always)]
    pub fn cast<U: Float>(
        &self,
    ) -> Sym3<U>
    {
        Sym3 {
            xx: self.xx.cast(),
            yy: self.yy.cast(),
            zz: self.zz.cast(),
            xy: self.xy.cast(),
            xz: self.xz.cast(),
            yz: self.yz.cast(),
        }
    }

    /// Symmetric part `(A + A^T) / 2` of the matrix `a`.
    #[inline(always)]
    pub fn from_mat3(
        a: &Mat3<T>,
    ) -> Self
    {
        let a = &a.0;
        Self {
            xx: a[0][0],
            yy: a[1][1],
            zz: a[2][2],
            xy: T::HALF * (a[0][1] + a[1][0]),
            xz: T::HALF * (a[0][2] + a[2][0]),
            yz: T::HALF * (a[1][2] + a[2][1]),
        }
    }

    /// The full matrix.
    #[inline(always)]
    pub fn to_mat3(
        &self,
    ) -> Mat3<T>
    {
        Mat3([[self.xx, self.xy, self.xz], [self.xy, self.yy, self.yz], [self.xz, self.yz, self.zz]])
    }
}

impl_linear_ops!(Sym3 { xx, yy, zz, xy, xz, yz });

/// `H = L D L^T` of a symmetric 3x3 matrix `H`, with unit lower triangular `L` and diagonal `D`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ldl3<T> {
    /// `L[1][0]`.
    pub l10: T,
    /// `L[2][0]`.
    pub l20: T,
    /// `L[2][1]`.
    pub l21: T,
    /// Pivots, the diagonal of `D`.
    pub pivots: [T; 3],
    /// Inverses of the pivots, the diagonal of `D^-1`.
    pub pivots_inv: [T; 3],
}

impl<T: Float> Ldl3<T> {

    /// `H^-1 b`.
    #[inline(always)]
    pub fn solve(
        &self,
        b: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::from_array(self.solve_array(b.to_array()))
    }

    /// `H^-1 b` for the array `b`.
    #[inline(always)]
    pub fn solve_array(
        &self,
        b: [T; 3],
    ) -> [T; 3]
    {
        let y = self.solve_lower(b);
        self.solve_lower_t([y[0] * self.pivots_inv[0], y[1] * self.pivots_inv[1], y[2] * self.pivots_inv[2]])
    }

    /// `L^-1 b`.
    #[inline(always)]
    pub fn solve_lower(
        &self,
        b: [T; 3],
    ) -> [T; 3]
    {
        let y1 = b[1] - self.l10 * b[0];
        [b[0], y1, b[2] - self.l20 * b[0] - self.l21 * y1]
    }

    /// `L^-T z`.
    #[inline(always)]
    pub fn solve_lower_t(
        &self,
        z: [T; 3],
    ) -> [T; 3]
    {
        let x1 = z[1] - self.l21 * z[2];
        [z[0] - self.l10 * x1 - self.l20 * z[2], x1, z[2]]
    }
}

/// Symmetric 6x6 matrix `[P, Q; Q^T, R]` of 3x3 blocks, where `P` is angular and `R` is linear.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sym6<T> {
    /// Upper left block `P` (angular).
    pub angular: Sym3<T>,
    /// Upper right block `Q`, which is the transpose of the lower left block (coupling).
    pub coupling: Mat3<T>,
    /// Lower right block `R` (linear).
    pub linear: Sym3<T>,
}

impl<T: Float> Sym6<T> {

    /// Zero matrix.
    #[inline(always)]
    pub fn zero(
    ) -> Self
    {
        Self {
            angular: Sym3::zero(),
            coupling: Mat3::zero(),
            linear: Sym3::zero(),
        }
    }

    /// Entry `(i, j)`.
    pub fn get(
        &self,
        i: usize,
        j: usize,
    ) -> T
    {
        match (i < 3, j < 3) {
            (true, true) => self.angular.get(i, j),
            (false, false) => self.linear.get(i - 3, j - 3),
            (true, false) => self.coupling.0[i][j - 3],
            (false, true) => self.coupling.0[j][i - 3],
        }
    }

    /// Sets entries `(i, j)` and `(j, i)`.
    pub fn set(
        &mut self,
        i: usize,
        j: usize,
        x: T,
    )
    {
        match (i < 3, j < 3) {
            (true, true) => self.angular.set(i, j, x),
            (false, false) => self.linear.set(i - 3, j - 3, x),
            (true, false) => self.coupling.0[i][j - 3] = x,
            (false, true) => self.coupling.0[j][i - 3] = x,
        }
    }

    /// Block LDLT factorization.
    ///
    /// - `P = L D L^T`.
    /// - `W = P^-1 Q`.
    /// - Schur complement `S = R - Q^T W = L D L^T`.
    #[inline(always)]
    pub fn ldlt(
        &self,
    ) -> Ldl6<T>
    {
        let angular = self.angular.ldlt();
        let (q0, q1, q2) = (self.coupling.col(0), self.coupling.col(1), self.coupling.col(2));
        let (w0, w1, w2) = (angular.solve(q0), angular.solve(q1), angular.solve(q2));
        let r = &self.linear;
        let schur = Sym3 {
            xx: r.xx - q0.dot(w0),
            yy: r.yy - q1.dot(w1),
            zz: r.zz - q2.dot(w2),
            xy: r.xy - q0.dot(w1),
            xz: r.xz - q0.dot(w2),
            yz: r.yz - q1.dot(w2),
        };
        let w = Mat3([[w0.x, w1.x, w2.x], [w0.y, w1.y, w2.y], [w0.z, w1.z, w2.z]]);
        Ldl6 {
            angular,
            coupling: w,
            schur: schur.ldlt(),
        }
    }

    /// `A -= k g g^T` with `g = [a; b]`.
    #[inline(always)]
    pub fn sub_scaled_outer(
        &mut self,
        k: T,
        a: Vec3<T>,
        b: Vec3<T>,
    )
    {
        let (ka, kb) = (a * k, b * k);
        self.angular.xx -= ka.x * a.x;
        self.angular.yy -= ka.y * a.y;
        self.angular.zz -= ka.z * a.z;
        self.angular.xy -= ka.x * a.y;
        self.angular.xz -= ka.x * a.z;
        self.angular.yz -= ka.y * a.z;
        self.linear.xx -= kb.x * b.x;
        self.linear.yy -= kb.y * b.y;
        self.linear.zz -= kb.z * b.z;
        self.linear.xy -= kb.x * b.y;
        self.linear.xz -= kb.x * b.z;
        self.linear.yz -= kb.y * b.z;
        let q = &mut self.coupling.0;
        q[0][0] -= ka.x * b.x;
        q[0][1] -= ka.x * b.y;
        q[0][2] -= ka.x * b.z;
        q[1][0] -= ka.y * b.x;
        q[1][1] -= ka.y * b.y;
        q[1][2] -= ka.y * b.z;
        q[2][0] -= ka.z * b.x;
        q[2][1] -= ka.z * b.y;
        q[2][2] -= ka.z * b.z;
    }

    /// Cast to the float type `U`.
    #[inline(always)]
    pub fn cast<U: Float>(
        &self,
    ) -> Sym6<U>
    {
        Sym6 {
            angular: self.angular.cast(),
            coupling: self.coupling.cast(),
            linear: self.linear.cast(),
        }
    }

    /// The full matrix, row major.
    pub fn to_dense(
        &self,
    ) -> [[T; 6]; 6]
    {
        let mut out = [[T::ZERO; 6]; 6];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = self.get(i, j);
            }
        }
        out
    }
}

impl_linear_ops!(Sym6 { angular, coupling, linear });

/// Block LDLT of a symmetric 6x6 matrix `[P, Q; Q^T, R]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ldl6<T> {
    /// LDLT of `P`.
    pub angular: Ldl3<T>,
    /// `W = P^-1 Q`.
    pub coupling: Mat3<T>,
    /// LDLT of the Schur complement `S = R - Q^T P^-1 Q`.
    pub schur: Ldl3<T>,
}

impl<T: Float> Ldl6<T> {

    /// Pivot `i` of `D` : the pivots of `P` for `i < 3`, then those of `S`.
    pub fn pivot(
        &self,
        i: usize,
    ) -> T
    {
        if i < 3 { self.angular.pivots[i] } else { self.schur.pivots[i - 3] }
    }

    /// `[x; y] = A^-1 [b; c]` : `y = S^-1 (c - W^T b)`, `x = P^-1 b - W y`.
    #[inline(always)]
    pub fn solve(
        &self,
        b: Vec3<T>,
        c: Vec3<T>,
    ) -> (Vec3<T>, Vec3<T>)
    {
        let y = self.schur.solve(c - self.coupling.mul_vec_t(b));
        (self.angular.solve(b) - self.coupling.mul_vec(y), y)
    }

    /// [`solve`](Self::solve) for the 6 numbers `b`.
    pub fn solve_array(
        &self,
        b: [T; 6],
    ) -> [T; 6]
    {
        let (x, y) = self.solve(Vec3::new(b[0], b[1], b[2]), Vec3::new(b[3], b[4], b[5]));
        [x.x, x.y, x.z, y.x, y.y, y.z]
    }
}
