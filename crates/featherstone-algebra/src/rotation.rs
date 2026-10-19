//! Rotations, including the unit quaternion, the coordinate rotation `E` of a Plucker transform, and the rotation about a coordinate axis.


use core::marker::PhantomData;
use core::ops::Mul;

use crate::float::Float;
use crate::linalg::{Mat3, Sym3, Vec3};

/// Unit quaternion `(w, x, y, z)` for the orientation `R` of a body in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat<T> {
    /// Scalar part `w`.
    pub w: T,
    /// `x` component of the vector part.
    pub x: T,
    /// `y` component of the vector part.
    pub y: T,
    /// `z` component of the vector part.
    pub z: T,
}

impl<T: Float> Quat<T> {

    /// Identity rotation.
    #[inline(always)]
    pub fn identity(
    ) -> Self
    {
        Self::new(T::ONE, T::ZERO, T::ZERO, T::ZERO)
    }

    /// `(w, x, y, z)`.
    #[inline(always)]
    pub const fn new(
        w: T,
        x: T,
        y: T,
        z: T,
    ) -> Self
    {
        Self {
            w,
            x,
            y,
            z,
        }
    }

    /// Vector part `(x, y, z)`.
    #[inline(always)]
    pub fn vector(
        self,
    ) -> Vec3<T>
    {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Conjugate, which is the inverse of a unit quaternion.
    #[inline(always)]
    pub fn conjugate(
        self,
    ) -> Self
    {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Orientation after a step `dt` at the body angular velocity `omega_body`, normalized.
    #[inline(always)]
    pub fn integrate(
        self,
        omega_body: Vec3<T>,
        dt: T,
    ) -> Self
    {
        self.mul(Self::from_rotation_vector(omega_body * dt)).normalized()
    }

    /// The quaternion divided by its norm.
    #[inline(always)]
    pub fn normalized(
        self
    ) -> Self
    {
        let k = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt().recip();
        Self::new(self.w * k, self.x * k, self.y * k, self.z * k)
    }

    /// `R^T v`, the vector `v` from world to body coordinates.
    #[inline(always)]
    pub fn to_body(
        self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        self.conjugate().to_world(v)
    }

    /// `R v`, the vector `v` from body to world coordinates.
    #[inline(always)]
    pub fn to_world(
        self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        let u = self.vector();
        let t = u.cross(v) * T::TWO;
        v + t * self.w + u.cross(t)
    }

    /// Cast to the float type `U`.
    #[inline(always)]
    pub fn cast<U: Float>(
        self,
    ) -> Quat<U>
    {
        Quat::new(self.w.cast(), self.x.cast(), self.y.cast(), self.z.cast())
    }

    /// Exponential map of the unit quaternion of a rotation by angle `|phi|` about the axis `phi / |phi|`.
    #[inline(always)]
    pub fn from_rotation_vector(
        phi: Vec3<T>,
    ) -> Self
    {
        let theta2 = phi.norm_squared();
        let theta = theta2.sqrt();
        let half = T::HALF * theta;
        let (s, c) = half.sin_cos();

        let k = if theta2 > T::EPSILON.sqrt() {
            s / theta
        } else {
            T::HALF - theta2 / T::from_f64(48.0)
        };
        Self::new(c, phi.x * k, phi.y * k, phi.z * k)
    }

    /// Rotation matrix `R` of the unit quaternion from body coordinates to world coordinates.
    #[inline(always)]
    pub fn to_rotation_matrix(
        self,
    ) -> Mat3<T>
    {
        let Self { w, x, y, z } = self;
        let two = T::TWO;
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        Mat3([[T::ONE - two * (yy + zz), two * (xy - wz), two * (xz + wy)], [two * (xy + wz), T::ONE - two * (xx + zz), two * (yz - wx)], [two * (xz - wy), two * (yz + wx), T::ONE - two * (xx + yy)]])
    }
}

impl<T: Float> Mul for Quat<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(
        self,
        b: Self,
    ) -> Self
    {
        let a = self;
        Self::new(
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        )
    }
}

/// Coordinate rotation `E` from parent to child coordinates.
pub trait Rotation<T: Float>:
    Copy
{
    /// `E^T S E`, the symmetric tensor `S` (an inertia) from child to parent coordinates.
    fn symmetric_to_parent(&self, s: &Sym3<T>) -> Sym3<T>;
    /// `E v`, the vector `v` from parent to child coordinates.
    fn to_child(&self, v: Vec3<T>) -> Vec3<T>;
    /// `E^T v`, the vector `v` from child to parent coordinates.
    fn to_parent(&self, v: Vec3<T>) -> Vec3<T>;
    /// `E` as a matrix.
    fn to_mat3(&self) -> Mat3<T>;
}

impl<T: Float> Rotation<T> for Mat3<T> {

    #[inline(always)]
    fn symmetric_to_parent(
        &self,
        s: &Sym3<T>,
    ) -> Sym3<T>
    {
        let se = s.to_mat3().mul_mat(self);
        Sym3::from_mat3(&self.transpose().mul_mat(&se))
    }

    #[inline(always)]
    fn to_child(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        self.mul_vec(v)
    }

    #[inline(always)]
    fn to_parent(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        self.mul_vec_t(v)
    }

    #[inline(always)]
    fn to_mat3(
        &self,
    ) -> Mat3<T>
    {
        *self
    }
}

/// Coordinate axis `e_k`, fixed at compile time.
pub trait Axis:
    Copy + Default + core::fmt::Debug
{
    /// `k`, 0, 1, 2 for x, y, z.
    const INDEX: usize;
    /// Component `v_k` of `v`.
    fn component<T: Float>(v: Vec3<T>) -> T;
    /// Column `k` of the symmetric matrix `s`.
    fn symmetric_column<T: Float>(sin: &Sym3<T>) -> Vec3<T>;
    /// `v + x e_k`, `x` added to the component `k` of `v`.
    fn add_component<T: Float>(v: Vec3<T>, x: T) -> Vec3<T>;
    /// Cross product `a x e_k`.
    fn cross_axis<T: Float>(a: Vec3<T>) -> Vec3<T>;
    /// `E^T S E`, the symmetric tensor `m` from child to parent coordinates.
    fn symmetric_to_parent<T: Float>(cos: T, sin: T, m: &Sym3<T>) -> Sym3<T>;
    /// `E v`, the vector `v` from parent to child coordinates.
    fn to_child<T: Float>(cos: T, sin: T, v: Vec3<T>) -> Vec3<T>;
    /// `E^T v`, the vector `v` from child to parent coordinates.
    fn to_parent<T: Float>(cos: T, sin: T, v: Vec3<T>) -> Vec3<T>;
}

/// The `x` axis, `e_0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxisX;
/// The `y` axis, `e_1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxisY;
/// The `z` axis, `e_2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxisZ;

impl Axis for AxisX {
    const INDEX: usize = 0;

    #[inline(always)]
    fn component<T: Float>(
        v: Vec3<T>,
    ) -> T
    {
        v.x
    }

    #[inline(always)]
    fn symmetric_column<T: Float>(
        s: &Sym3<T>,
    ) -> Vec3<T> {
        Vec3::new(s.xx, s.xy, s.xz)
    }

    #[inline(always)]
    fn add_component<T: Float>(
        v: Vec3<T>,
        x: T,
    ) -> Vec3<T>
    {
        Vec3::new(v.x + x, v.y, v.z)
    }

    #[inline(always)]
    fn cross_axis<T: Float>(
        a: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(T::ZERO, a.z, -a.y)
    }

    #[inline(always)]
    fn symmetric_to_parent<T: Float>(
        cos: T,
        sin: T,
        m: &Sym3<T>,
    ) -> Sym3<T>
    {
        let (cc, ss, cs) = (cos * cos, sin * sin, cos * sin);
        let two_cs_yz = T::TWO * cs * m.yz;
        Sym3 {
            xx: m.xx,
            yy: cc * m.yy - two_cs_yz + ss * m.zz,
            zz: ss * m.yy + two_cs_yz + cc * m.zz,
            xy: cos * m.xy - sin * m.xz,
            xz: sin * m.xy + cos * m.xz,
            yz: cs * (m.yy - m.zz) + (cc - ss) * m.yz,
        }
    }

    #[inline(always)]
    fn to_child<T: Float>(
        cos: T,
        sin: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(v.x, cos * v.y + sin * v.z, cos * v.z - sin * v.y)
    }

    #[inline(always)]
    fn to_parent<T: Float>(
        cos: T,
        sin: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(v.x, cos * v.y - sin * v.z, cos * v.z + sin * v.y)
    }
}

impl Axis for AxisY {
    const INDEX: usize = 1;

    #[inline(always)]
    fn component<T: Float>(
        v: Vec3<T>,
    ) -> T
    {
        v.y
    }

    #[inline(always)]
    fn symmetric_column<T: Float>(
        s: &Sym3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(s.xy, s.yy, s.yz)
    }

    #[inline(always)]
    fn add_component<T: Float>(
        v: Vec3<T>,
        x: T,
    ) -> Vec3<T>
    {
        Vec3::new(v.x, v.y + x, v.z)
    }

    #[inline(always)]
    fn cross_axis<T: Float>(
        a: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(-a.z, T::ZERO, a.x)
    }

    #[inline(always)]
    fn symmetric_to_parent<T: Float>(
        cos: T,
        sin: T,
        m: &Sym3<T>,
    ) -> Sym3<T>
    {
        let (cc, ss, cs) = (cos * cos, sin * sin, cos * sin);
        let two_cs_xz = T::TWO * cs * m.xz;
        Sym3 {
            xx: cc * m.xx + two_cs_xz + ss * m.zz,
            yy: m.yy,
            zz: ss * m.xx - two_cs_xz + cc * m.zz,
            xy: cos * m.xy + sin * m.yz,
            xz: cs * (m.zz - m.xx) + (cc - ss) * m.xz,
            yz: cos * m.yz - sin * m.xy,
        }
    }

    #[inline(always)]
    fn to_child<T: Float>(
        cos: T,
        sin: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(cos * v.x - sin * v.z, v.y, sin * v.x + cos * v.z)
    }

    #[inline(always)]
    fn to_parent<T: Float>(
        cos: T,
        sin: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(cos * v.x + sin * v.z, v.y, cos * v.z - sin * v.x)
    }
}

impl Axis for AxisZ {
    const INDEX: usize = 2;

    #[inline(always)]
    fn component<T: Float>(
        v: Vec3<T>,
    ) -> T
    {
        v.z
    }

    #[inline(always)]
    fn symmetric_column<T: Float>(
        s: &Sym3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(s.xz, s.yz, s.zz)
    }

    #[inline(always)]
    fn add_component<T: Float>(
        v: Vec3<T>,
        x: T,
    ) -> Vec3<T>
    {
        Vec3::new(v.x, v.y, v.z + x)
    }

    #[inline(always)]
    fn cross_axis<T: Float>(
        a: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(a.y, -a.x, T::ZERO)
    }

    #[inline(always)]
    fn symmetric_to_parent<T: Float>(
        cos: T,
        sin: T,
        m: &Sym3<T>,
    ) -> Sym3<T>
    {
        let (cc, ss, cs) = (cos * cos, sin * sin, cos * sin);
        let two_cs_xy = T::TWO * cs * m.xy;
        Sym3 {
            xx: cc * m.xx - two_cs_xy + ss * m.yy,
            yy: ss * m.xx + two_cs_xy + cc * m.yy,
            zz: m.zz,
            xy: cs * (m.xx - m.yy) + (cc - ss) * m.xy,
            xz: cos * m.xz - sin * m.yz,
            yz: sin * m.xz + cos * m.yz,
        }
    }

    #[inline(always)]
    fn to_child<T: Float>(
        cos: T,
        sin: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(cos * v.x + sin * v.y, cos * v.y - sin * v.x, v.z)
    }

    #[inline(always)]
    fn to_parent<T: Float>(
        cos: T,
        sin: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(cos * v.x - sin * v.y, cos * v.y + sin * v.x, v.z)
    }
}

/// Coordinate rotation `E` for a rotation by the angle `q` about the axis `A`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AxisRotation<T, A> {
    /// `cos q`.
    pub cos: T,
    /// `sin q`.
    pub sin: T,
    axis: PhantomData<A>,
}

impl<T: Float, A: Axis> AxisRotation<T, A> {
    
    /// Rotation by the angle `q` in radians.
    #[inline(always)]
    pub fn from_angle(
        q: T,
    ) -> Self
    {
        let (sin, cos) = q.sin_cos();
        Self {
            cos,
            sin,
            axis: PhantomData,
        }
    }
}

impl<T: Float, A: Axis> Rotation<T> for AxisRotation<T, A> {

    #[inline(always)]
    fn symmetric_to_parent(
        &self,
        s: &Sym3<T>,
    ) -> Sym3<T>
    {
        A::symmetric_to_parent(self.cos, self.sin, s)
    }

    #[inline(always)]
    fn to_child(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        A::to_child(self.cos, self.sin, v)
    }

    #[inline(always)]
    fn to_parent(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        A::to_parent(self.cos, self.sin, v)
    }

    fn to_mat3(
        &self,
    ) -> Mat3<T>
    {
        let e = |v| self.to_child(v);
        let (o, z) = (T::ONE, T::ZERO);
        Mat3([[e(Vec3::new(o, z, z)).x, e(Vec3::new(z, o, z)).x, e(Vec3::new(z, z, o)).x], [e(Vec3::new(o, z, z)).y, e(Vec3::new(z, o, z)).y, e(Vec3::new(z, z, o)).y], [e(Vec3::new(o, z, z)).z, e(Vec3::new(z, o, z)).z, e(Vec3::new(z, z, o)).z]])
    }
}
