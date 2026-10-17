//! Symmetric 3x3 and 6x6 matrices and their LDLT factorizations.
//!
//! `Sym3` holds rotational inertias and the 3x3 joint blocks of each leg.
//! `Sym6` holds the base blocks of the joint space inertia matrix and its Schur complement in 3x3 blocks.


use core::ops::{Add, AddAssign, Sub};

use crate::float::Float;
use crate::spatial::{Mat3, Vec3};

/// Symmetric 3x3 matrix.
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

    /// Cast to float type `U`.
    pub fn cast<U: Float>(
        &self,
    ) -> Sym3<U>
    {
        let c = |x: T| U::from_f64(x.to_f64());
        Sym3 {
            xx: c(self.xx),
            yy: c(self.yy),
            zz: c(self.zz),
            xy: c(self.xy),
            xz: c(self.xz),
            yz: c(self.yz),
        }
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

    /// `S x`.
    #[inline(always)]
    pub fn mul_array(
        &self,
        x: [T; 3],
    ) -> [T; 3]
    {
        self.mul_vec(Vec3::from_array(x)).to_array()
    }

    /// `k S`.
    #[inline(always)]
    pub fn scale(
        &self,
        k: T,
    ) -> Self
    {
        Self {
            xx: k * self.xx,
            yy: k * self.yy,
            zz: k * self.zz,
            xy: k * self.xy,
            xz: k * self.xz,
            yz: k * self.yz,
        }
    }

    /// Get entry `(i, j)`.
    #[inline(always)]
    pub fn get(
        &self,
        i: usize,
        j: usize,
    ) -> T
    {
        match(i.min(j), i.max(j)) {
            (0, 0) => self.xx,
            (1, 1) => self.yy,
            (2, 2) => self.zz,
            (0, 1) => self.xy,
            (0, 2) => self.xz,
            (1, 2) => self.yz,
            _ => panic!("Sym3 index out of range"),
        }
    }

    /// Set entries `(i, j)` and `(j, i)`.
    pub fn set(
        &mut self,
        i: usize,
        j: usize,
        x: T,
    )
    {
        match(i.min(j), i.max(j)) {
            (0, 0) => self.xx = x,
            (1, 1) => self.yy = x,
            (2, 2) => self.zz = x,
            (0, 1) => self.xy = x,
            (0, 2) => self.xz = x,
            (1, 2) => self.yz = x,
            _ => panic!("Sym3 index out of range"),
        }
    }

    /// Full matrix.
    #[inline(always)]
    pub fn to_mat3(
        &self,
    ) -> Mat3<T>
    {
        Mat3([[self.xx, self.xy, self.xz], [self.xy, self.yy, self.yz], [self.xz, self.yz, self.zz]])
    }

    /// Symmetric part `(A + A^T) / 2` of `a`.
    #[inline(always)]
    pub fn from_mat3(
        a: &Mat3<T>
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

    /// LDLT factorization, positive definite.
    #[inline(always)]
    pub fn ldlt(
        &self,
    ) -> Ldl3<T>
    {
        let d0 = self.xx;
        let dinv0 = d0.recip();
        let l10 = self.xy * dinv0;
        let l20 = self.xz * dinv0;
        let d1 = self.yy - l10 * self.xy;
        let dinv1 = d1.recip();
        let l21 = (self.yz - l20 * self.xy) * dinv1;
        let d2 = self.zz - l20 * self.xz - l21 * (self.yz - l20 * self.xy);
        Ldl3 {
            l10,
            l20,
            l21,
            d: [d0, d1, d2],
            dinv: [dinv0, dinv1, d2.recip()],
        }
    }
}

impl<T: Float> Add for Sym3<T> {
    type Output = Self;
    #[inline(always)]
    fn add(
        self,
        o: Self,
    ) -> Self
    {
        Self {
            xx: self.xx + o.xx,
            yy: self.yy + o.yy,
            zz: self.zz + o.zz,
            xy: self.xy + o.xy,
            xz: self.xz + o.xz,
            yz: self.yz + o.yz,
        }
    }
}

impl<T: Float> AddAssign for Sym3<T> {
    #[inline(always)]
    fn add_assign(
        &mut self,
        o: Self,
    )
    {
        *self = *self + o;
    }
}

impl<T: Float> Sub for Sym3<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(
        self,
        o: Self,
    ) -> Self
    {
        Self {
            xx: self.xx - o.xx,
            yy: self.yy - o.yy,
            zz: self.zz - o.zz,
            xy: self.xy - o.xy,
            xz: self.xz - o.xz,
            yz: self.yz - o.yz,
        }
    }
}

/// `H = L D L^T` of a symmetric 3x3 matrix, with unit lower triangular `L`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ldl3<T> {
    /// `L[1][0]`.
    pub l10: T,
    /// `L[2][0]`.
    pub l20: T,
    /// `L[2][1]`.
    pub l21: T,
    /// Pivots `D`.
    pub d: [T; 3],
    /// `1 / D`.
    pub dinv: [T; 3],
}

impl<T: Float> Ldl3<T> {

    /// `L^-1 b`.
    #[inline(always)]
    pub fn forward(
        &self,
        b: [T; 3],
    ) -> [T; 3]
    {
        let y1 = b[1] - self.l10 * b[0];
        [b[0], y1, b[2] - self.l20 * b[0] - self.l21 * y1]
    }

    /// `L^-T z`.
    #[inline(always)]
    pub fn backward(
        &self,
        z: [T; 3],
    ) -> [T; 3]
    {
        let x1 = z[1] - self.l21 * z[2];
        [z[0] - self.l10 * x1 - self.l20 * z[2], x1, z[2]]
    }

    /// `H^-1 b` for a vector.
    #[inline(always)]
    pub fn solve_vec(
        &self,
        b: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::from_array(self.solve(b.to_array()))
    }

    /// `H^-1 b`.
    #[inline(always)]
    pub fn solve(
        &self,
        b: [T; 3],
    ) -> [T; 3]
    {
        let y = self.forward(b);
        self.backward([y[0] * self.dinv[0], y[1] * self.dinv[1], y[2] * self.dinv[2]])
    }
}

/// Symmetric 6x6 matrix `[P, Q; Q^T, R]`, where `P` is angular and `R` is linear.
/// Every operation is straight-line code on named fields : on the GPU, an array indexed in a loop that is not unrolled goes to local memory.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sym6<T> {
    /// Upper left block.
    pub p: Sym3<T>,
    /// Upper right block (transposed to lower left block).
    pub q: Mat3<T>,
    /// Lower right block.
    pub r: Sym3<T>,
}

impl<T: Float> Sym6<T> {
    
    /// Zero matrix.
    #[inline(always)]
    pub fn zero(
    ) -> Self
    {
        Self {
            p: Sym3::zero(),
            q: Mat3::default(),
            r: Sym3::zero()
        }
    }

    /// Get entry `(i, j)`.
    pub fn get(
        &self,
        i: usize,
        j: usize,
    ) -> T
    {
        match(i < 3, j < 3) {
            (true, true) => self.p.get(i, j),
            (false, false) => self.r.get(i - 3, j - 3),
            (true, false) => self.q.0[i][j - 3],
            (false, true) => self.q.0[j][i - 3],
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
        match(i < 3, j < 3) {
            (true, true) => self.p.set(i, j, x),
            (false, false) => self.r.set(i - 3, j - 3, x),
            (true, false) => self.q.0[i][j - 3] = x,
            (false, true) => self.q.0[j][i - 3] = x,
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
        self.p.xx -= ka.x * a.x;
        self.p.yy -= ka.y * a.y;
        self.p.zz -= ka.z * a.z;
        self.p.xy -= ka.x * a.y;
        self.p.xz -= ka.x * a.z;
        self.p.yz -= ka.y * a.z;
        self.r.xx -= kb.x * b.x;
        self.r.yy -= kb.y * b.y;
        self.r.zz -= kb.z * b.z;
        self.r.xy -= kb.x * b.y;
        self.r.xz -= kb.x * b.z;
        self.r.yz -= kb.y * b.z;
        let q = &mut self.q.0;
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

    /// Full matrix.
    pub fn to_array(
        &self
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

    /// Block LDLT factorization.
    /// `P = L D L^T`.
    /// `W = P^-1 Q`.
    /// Schur complement `S = R - Q^T W = L D L^T`.
    #[inline(always)]
    pub fn ldlt(
        &self
    ) -> Ldl6<T>
    {
        let p = self.p.ldlt();
        let (q0, q1, q2) = (self.q.col(0), self.q.col(1), self.q.col(2));
        let (w0, w1, w2) = (p.solve_vec(q0), p.solve_vec(q1), p.solve_vec(q2));
        let r = &self.r;
        let s = Sym3 {
            xx: r.xx - q0.dot(w0),
            yy: r.yy - q1.dot(w1),
            zz: r.zz - q2.dot(w2),
            xy: r.xy - q0.dot(w1),
            xz: r.xz - q0.dot(w2),
            yz: r.yz - q1.dot(w2),
        };
        let w = Mat3([[w0.x, w1.x, w2.x], [w0.y, w1.y, w2.y], [w0.z, w1.z, w2.z]]);
        Ldl6 {
            p,
            w,
            s: s.ldlt(),
        }
    }
}

/// Block LDLT of a symmetric 6x6 matrix `[P, Q; Q^T, R]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ldl6<T> {
    /// LDLT of `P`.
    pub p: Ldl3<T>,
    /// `W = P^-1 Q`.
    pub w: Mat3<T>,
    /// LDLT of `S = R - Q^T P^-1 Q`.
    pub s: Ldl3<T>,
}

impl<T: Float> Ldl6<T> {

    /// Pivot `i` of `D` with those of `P` and `S`.
    pub fn pivot(
        &self,
        i: usize,
    ) -> T
    {
        if i < 3 { self.p.d[i] } else { self.s.d[i - 3] }
    }

    /// `[x; y] = A^-1 [b; c]` : `y = S^-1 (c - W^T b)`, `x = P^-1 b - W y`.
    #[inline(always)]
    pub fn solve(
        &self,
        b: Vec3<T>,
        c: Vec3<T>,
    ) -> (Vec3<T>, Vec3<T>)
    {
        let y = self.s.solve_vec(c - self.w.tr_mul_vec(b));
        (self.p.solve_vec(b) - self.w.mul_vec(y), y)
    }

    /// [`solve`](Self::solve) on 6 numbers.
    pub fn solve_array(
        &self,
        b: [T; 6],
    ) -> [T; 6]
    {
        let (x, y) = self.solve(Vec3::new(b[0], b[1], b[2]), Vec3::new(b[3], b[4], b[5]));
        [x.x, x.y, x.z, y.x, y.y, y.z]
    }
}
