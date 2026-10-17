//! Spatial Vector Algebra from Featherstone's Rigid Body Dynamics Algorithms (RBDA), Ch. 2 and App. A.
//!
//! Conventions :
//! - Motion vectors are `[w; v]` and force vectors are `[n; f]`, with the angular part first.
//! - A transform `X` from a parent frame to a child frame stores `E`, the rotation from parent to child coordinates, and `r`, the child origin in parent coordinates : `X = rot(E) xlt(r)`.
//! - A rigid body inertia stores the mass, the first moment of mass `h = m c`, and the rotational inertia about the frame origin.
//! - Quaternions are `(w, x, y, z)`, as in MuJoCo.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::float::Float;
pub use crate::linalg::Sym3;
use crate::linalg::Sym6;

/// 3x1 vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3<T> {
    /// Entry (0).
    pub x: T,
    /// Entry (1).
    pub y: T,
    /// Entry (2).
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

    /// `a * b`.
    #[inline(always)]
    pub fn dot(
        self,
        o: Self,
    ) -> T
    {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// `a x b`.
    #[inline(always)]
    pub fn cross(
        self,
        o: Self,
    ) -> Self
    {
        Self::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }

    /// `|a|^2`.
    #[inline(always)]
    pub fn norm_squared(
        self,
    ) -> T
    {
        self.dot(self)
    }

    /// `|a|`.
    #[inline(always)]
    pub fn norm(
        self,
    ) -> T
    {
        self.norm_squared().sqrt()
    }

    /// Cast to float type `U`.
    pub fn cast<U: Float>(
        self,
    ) -> Vec3<U>
    {
        Vec3::new(U::from_f64(self.x.to_f64()), U::from_f64(self.y.to_f64()), U::from_f64(self.z.to_f64()))
    }
}


impl<T: Float> Add for Vec3<T> {
    type Output = Self;
    #[inline(always)]
    fn add(
        self,
        o: Self,
    ) -> Self
    {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl<T: Float> Sub for Vec3<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(
        self,
        o: Self,
    ) -> Self
    {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl<T: Float> Neg for Vec3<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(
        self,
    ) -> Self
    {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl<T: Float> Mul<T> for Vec3<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(
        self,
        k: T,
    ) -> Self
    {
        Self::new(self.x * k, self.y * k, self.z * k)
    }
}

impl<T: Float> AddAssign for Vec3<T> {
    #[inline(always)]
    fn add_assign(
        &mut self,
        o: Self,
    )
    {
        *self = *self + o;
    }
}

impl<T: Float> SubAssign for Vec3<T> {
    #[inline(always)]
    fn sub_assign(
        &mut self,
        o: Self,
    )
    {
        *self = *self - o;
    }
}

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
    #[inline(always)]
    pub fn skew(
        a: Vec3<T>,
    ) -> Self
    {
        let z = T::ZERO;
        Self([[z, -a.z, a.y], [a.z, z, -a.x], [-a.y, a.x, z]])
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

    /// Col `j`.
    #[inline(always)]
    pub fn col(
        &self,
        j: usize,
    ) -> Vec3<T>
    {
        Vec3::new(self.0[0][j], self.0[1][j], self.0[2][j])
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
    pub fn tr_mul_vec(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(self.col(0).dot(v), self.col(1).dot(v), self.col(2).dot(v))
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

    /// `A^T`.
    #[inline(always)]
    pub fn transpose(
        &self
    ) -> Self
    {
        let a = &self.0;
        Self([[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]])
    }
}

/// Unit quaternion (w, x, y, z) representing the rotation from body to world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat<T> {
    /// Scalar.
    pub w: T,
    /// Vector entry (0).
    pub x: T,
    /// Vector entry (1).
    pub y: T,
    /// Vector entry (2).
    pub z: T,
}

impl<T: Float> Quat<T> {

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

    /// Identity.
    #[inline(always)]
    pub fn identity(
    ) -> Self
    {
        Self::new(T::ONE, T::ZERO, T::ZERO, T::ZERO)
    }

    /// Vector part.
    #[inline(always)]
    pub fn vec(
        self,
    ) -> Vec3<T>
    {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Conjugate (inverse of a unit quaternion).
    #[inline(always)]
    pub fn conj(
        self,
    ) -> Self
    {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Unit norm.
    #[inline(always)]
    pub fn normalized(
        self
    ) -> Self
    {
        let k = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt().recip();
        Self::new(self.w * k, self.x * k, self.y * k, self.z * k)
    }

    /// Rotation matrix `R` of a unit quaternion, body to world.
    #[inline(always)]
    pub fn to_mat3(
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

    /// `R v`.
    #[inline(always)]
    pub fn rotate(
        self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        let u = self.vec();
        let t = u.cross(v) * T::TWO;
        v + t * self.w + u.cross(t)
    }

    /// `R^T v`.
    #[inline(always)]
    pub fn rotate_inv(
        self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        self.conj().rotate(v)
    }

    /// Exponential map : unit quaternion of a rotation by angle `|phi|` about the axis `phi / |phi|`.
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

    /// Attitude after a step `h` at body angular velocity `omega_body`.
    #[inline(always)]
    pub fn integrate(
        self,
        omega_body: Vec3<T>,
        h: T,
    ) -> Self
    {
        self.mul(Self::from_rotation_vector(omega_body * h)).normalized()
    }
}

impl<T: Float> Mul for Quat<T> {
    type Output = Self;
    /// Hamilton product.
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

/// Spatial motion vector `[w; v]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Motion<T> {
    /// Angular vector.
    pub ang: Vec3<T>,
    /// Linear vector.
    pub lin: Vec3<T>,
}

/// Spatial force vector `[n; f]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Force<T> {
    /// Moment about the origin.
    pub ang: Vec3<T>,
    /// Force.
    pub lin: Vec3<T>,
}

macro_rules! impl_spatial_vector {
    ($name:ident) => {
        impl<T: Float> $name<T> {

            /// From angular and linear vectors.
            #[inline(always)]
            pub const fn new(
                ang: Vec3<T>,
                lin: Vec3<T>,
            ) -> Self
            {
                Self {
                    ang,
                    lin,
                }
            }

            /// Zero vector.
            #[inline(always)]
            pub fn zero(
            ) -> Self
            {
                Self::new(Vec3::zero(), Vec3::zero())
            }

            /// `[ang; lin]` as 6 numbers.
            #[inline(always)]
            pub fn to_array(
                self,
            ) -> [T; 6]
            {
                [self.ang.x, self.ang.y, self.ang.z, self.lin.x, self.lin.y, self.lin.z]
            }

            /// From `[ang; lin]`.
            #[inline(always)]
            pub fn from_array(
                a: [T; 6],
            ) -> Self
            {
                Self::new(Vec3::new(a[0], a[1], a[2]), Vec3::new(a[3], a[4], a[5]))
            }
        }

        impl<T: Float> Add for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn add(
                self,
                o: Self,
            ) -> Self
            {
                Self::new(self.ang + o.ang, self.lin + o.lin)
            }
        }

        impl<T: Float> Sub for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn sub(
                self,
                o: Self,
            ) -> Self
            {
                Self::new(self.ang - o.ang, self.lin - o.lin)
            }
        }

        impl<T: Float> Neg for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn neg(
                self,
            ) -> Self
            {
                Self::new(-self.ang, -self.lin)
            }
        }

        impl<T: Float> Mul<T> for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn mul(
                self,
                k: T,
            ) -> Self
            {
                Self::new(self.ang * k, self.lin * k)
            }
        }

        impl<T: Float> AddAssign for $name<T> {
            #[inline(always)]
            fn add_assign(
                &mut self,
                o: Self,
            )
            {
                *self = *self + o;
            }
        }

        impl<T: Float> SubAssign for $name<T> {
            #[inline(always)]
            fn sub_assign(
                &mut self,
                o: Self,
            )
            {
                *self = *self - o;
            }
        }
    };
}

impl_spatial_vector!(Motion);
impl_spatial_vector!(Force);

impl<T: Float> Motion<T> {

    /// Motion cross product.
    #[inline(always)]
    pub fn cross(
        self,
        m: Self,
    ) -> Self
    {
        Self::new(self.ang.cross(m.ang), self.ang.cross(m.lin) + self.lin.cross(m.ang))
    }

    /// Force cross product.
    #[inline(always)]
    pub fn cross_force(
        self,
        f: Force<T>,
    ) -> Force<T>
    {
        Force::new(self.ang.cross(f.ang) + self.lin.cross(f.lin), self.ang.cross(f.lin))
    }

    /// Power `m^T f`.
    #[inline(always)]
    pub fn dot(
        self,
        f: Force<T>,
    ) -> T
    {
        self.ang.dot(f.ang) + self.lin.dot(f.lin)
    }
}

impl<T: Float> Force<T> {

    /// Force acting at a point `p`.
    #[inline(always)]
    pub fn at_point(
        p: Vec3<T>,
        force: Vec3<T>,
    ) -> Self
    {
        Self::new(p.cross(force), force)
    }
}

/// Spatial inertia of a rigid body in the coordinates of a frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RigidInertia<T> {
    /// Mass.
    pub mass: T,
    /// First moment of mass `h = m c` for center of mass `c`.
    pub h: Vec3<T>,
    /// Rotational inertia `I` about the frame origin.
    pub ibar: Sym3<T>,
}

impl<T: Float> RigidInertia<T> {

    /// Inertia of a body from its mass, its center of mass `com`, and its rotational inertia `inertia_com` about the center of mass.
    /// The parallel axis theorem moves the rotational inertia to the frame origin.
    pub fn from_com(
        mass: T,
        com: Vec3<T>,
        inertia_com: Sym3<T>,
    ) -> Self
    {
        let parallel = Sym3 {
            xx: com.y * com.y + com.z * com.z,
            yy: com.x * com.x + com.z * com.z,
            zz: com.x * com.x + com.y * com.y,
            xy: -com.x * com.y,
            xz: -com.x * com.z,
            yz: -com.y * com.z,
        };
        Self {
            mass,
            h: com * mass,
            ibar: inertia_com + parallel.scale(mass),
        }
    }

    /// Zero inertia.
    #[inline(always)]
    pub fn zero(
    ) -> Self
    {
        Self {
            mass: T::ZERO,
            h: Vec3::zero(),
            ibar: Sym3::zero(),
        }
    }

    /// Cast to float type `U`.
    pub fn cast<U: Float>(
        &self,
    ) -> RigidInertia<U>
    {
        RigidInertia {
            mass: U::from_f64(self.mass.to_f64()),
            h: self.h.cast(),
            ibar: self.ibar.cast(),
        }
    }

    /// Center of mass `c = h / m`.
    #[inline(always)]
    pub fn com(
        &self,
    ) -> Vec3<T>
    {
        self.h * self.mass.recip()
    }

    /// Product of this inertia with a motion vector `m`.
    /// For a velocity it gives the spatial momentum of the body, and for an acceleration the force that produces it.
    #[inline(always)]
    pub fn mul_motion(
        &self,
        m: Motion<T>,
    ) -> Force<T>
    {
        Force::new(self.ibar.mul_vec(m.ang) + self.h.cross(m.lin), m.lin * self.mass - self.h.cross(m.ang))
    }

    /// The 6x6 matrix as a [`Sym6`].
    #[inline(always)]
    pub fn to_sym6(
        &self,
    ) -> Sym6<T>
    {
        let (z, m) = (T::ZERO, self.mass);
        Sym6 {
            p: self.ibar,
            q: Mat3::skew(self.h),
            r: Sym3 {
                xx: m,
                yy: m,
                zz: m,
                xy: z,
                xz: z,
                yz: z,
            },
        }
    }

    /// The 6x6 matrix in `[ang; lin]` order.
    pub fn to_array(
        &self,
    ) -> [[T; 6]; 6]
    {
        let mut out = [[T::ZERO; 6]; 6];
        let hx = Mat3::skew(self.h);
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.ibar.get(i, j);
                out[i][3 + j] = hx.0[i][j];
                out[3 + i][j] = hx.0[j][i];
            }
            out[3 + i][3 + i] = self.mass;
        }
        out
    }
}

impl<T: Float> Add for RigidInertia<T> {
    type Output = Self;
    #[inline(always)]
    fn add(
        self,
        o: Self,
    ) -> Self
    {
        Self {
            mass: self.mass + o.mass,
            h: self.h + o.h,
            ibar: self.ibar + o.ibar,
        }
    }
}

impl<T: Float> AddAssign for RigidInertia<T> {
    #[inline(always)]
    fn add_assign(
        &mut self,
        o: Self,
    )
    {
        *self = *self + o;
    }
}

/// Coordinate rotation `E` from parent to child.
pub trait Rotation<T: Float>: Copy {
    /// `E v`.
    fn apply(&self, v: Vec3<T>) -> Vec3<T>;
    /// `E^T v`.
    fn apply_t(&self, v: Vec3<T>) -> Vec3<T>;
    /// `E^T S E` of a symmetric `S`.
    fn congruence_t(&self, s: &Sym3<T>) -> Sym3<T>;
    /// `E` as a matrix.
    fn matrix(&self) -> Mat3<T>;
}

impl<T: Float> Rotation<T> for Mat3<T> {

    #[inline(always)]
    fn apply(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        self.mul_vec(v)
    }

    #[inline(always)]
    fn apply_t(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        self.tr_mul_vec(v)
    }

    #[inline(always)]
    fn congruence_t(
        &self,
        s: &Sym3<T>,
    ) -> Sym3<T>
    {
        let se = s.to_mat3().mul_mat(self);
        Sym3::from_mat3(&self.transpose().mul_mat(&se))
    }

    #[inline(always)]
    fn matrix(
        &self,
    ) -> Mat3<T>
    {
        *self
    }
}

/// Coordinate axis `e_k`, fixed at compile time.
/// Each operation acts on the two components that the axis does not fix.
pub trait Axis: Copy + Default + core::fmt::Debug {
    /// `k` : 0, 1, 2 for x, y, z.
    const INDEX: usize;
    /// `E v`.
    fn rotate<T: Float>(c: T, s: T, v: Vec3<T>) -> Vec3<T>;
    /// `E^T v`.
    fn rotate_t<T: Float>(c: T, s: T, v: Vec3<T>) -> Vec3<T>;
    /// `E^T S E`.
    fn congruence_t<T: Float>(c: T, s: T, m: &Sym3<T>) -> Sym3<T>;
    /// Component `v_k`.
    fn get<T: Float>(v: Vec3<T>) -> T;
    /// `v + x e_k`.
    fn add<T: Float>(v: Vec3<T>, x: T) -> Vec3<T>;
    /// `a x e_k`.
    fn cross<T: Float>(a: Vec3<T>) -> Vec3<T>;
    /// Column `k` of `S`.
    fn column<T: Float>(s: &Sym3<T>) -> Vec3<T>;
}

/// Axis (x).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxisX;
/// Axis (y).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxisY;
/// Axis (z).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxisZ;

impl Axis for AxisX {
    const INDEX: usize = 0;
    
    #[inline(always)]
    fn rotate<T: Float>(
        c: T,
        s: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(v.x, c * v.y + s * v.z, c * v.z - s * v.y)
    }
    
    #[inline(always)]
    fn rotate_t<T: Float>(
        c: T,
        s: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(v.x, c * v.y - s * v.z, c * v.z + s * v.y)
    }
    
    #[inline(always)]
    fn congruence_t<T: Float>(
        c: T,
        s: T,
        m: &Sym3<T>,
    ) -> Sym3<T>
    {
        let (cc, ss, cs) = (c * c, s * s, c * s);
        let two_cs_yz = T::TWO * cs * m.yz;
        Sym3 {
            xx: m.xx,
            yy: cc * m.yy - two_cs_yz + ss * m.zz,
            zz: ss * m.yy + two_cs_yz + cc * m.zz,
            xy: c * m.xy - s * m.xz,
            xz: s * m.xy + c * m.xz,
            yz: cs * (m.yy - m.zz) + (cc - ss) * m.yz,
        }
    }
    
    #[inline(always)]
    fn get<T: Float>(
        v: Vec3<T>,
    ) -> T
    {
        v.x
    }
    
    #[inline(always)]
    fn add<T: Float>(
        v: Vec3<T>,
        x: T,
    ) -> Vec3<T>
    {
        Vec3::new(v.x + x, v.y, v.z)
    }
    
    #[inline(always)]
    fn cross<T: Float>(
        a: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(T::ZERO, a.z, -a.y)
    }
    
    #[inline(always)]
    fn column<T: Float>(
        s: &Sym3<T>,
    ) -> Vec3<T> {
        Vec3::new(s.xx, s.xy, s.xz)
    }
}

impl Axis for AxisY {
    const INDEX: usize = 1;

    #[inline(always)]
    fn rotate<T: Float>(
        c: T,
        s: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(c * v.x - s * v.z, v.y, s * v.x + c * v.z)
    }

    #[inline(always)]
    fn rotate_t<T: Float>(
        c: T,
        s: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(c * v.x + s * v.z, v.y, c * v.z - s * v.x)
    }

    #[inline(always)]
    fn congruence_t<T: Float>(
        c: T,
        s: T,
        m: &Sym3<T>,
    ) -> Sym3<T>
    {
        let (cc, ss, cs) = (c * c, s * s, c * s);
        let two_cs_xz = T::TWO * cs * m.xz;
        Sym3 {
            xx: cc * m.xx + two_cs_xz + ss * m.zz,
            yy: m.yy,
            zz: ss * m.xx - two_cs_xz + cc * m.zz,
            xy: c * m.xy + s * m.yz,
            xz: cs * (m.zz - m.xx) + (cc - ss) * m.xz,
            yz: c * m.yz - s * m.xy,
        }
    }

    #[inline(always)]
    fn get<T: Float>(
        v: Vec3<T>,
    ) -> T
    {
        v.y
    }

    #[inline(always)]
    fn add<T: Float>(
        v: Vec3<T>,
        x: T,
    ) -> Vec3<T>
    {
        Vec3::new(v.x, v.y + x, v.z)
    }
    
    #[inline(always)]
    fn cross<T: Float>(
        a: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(-a.z, T::ZERO, a.x)
    }

    #[inline(always)]
    fn column<T: Float>(
        s: &Sym3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(s.xy, s.yy, s.yz)
    }
}

impl Axis for AxisZ {
    const INDEX: usize = 2;

    #[inline(always)]
    fn rotate<T: Float>(
        c: T,
        s: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(c * v.x + s * v.y, c * v.y - s * v.x, v.z)
    }

    #[inline(always)]
    fn rotate_t<T: Float>(
        c: T,
        s: T,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(c * v.x - s * v.y, c * v.y + s * v.x, v.z)
    }

    #[inline(always)]
    fn congruence_t<T: Float>(
        c: T,
        s: T,
        m: &Sym3<T>,
    ) -> Sym3<T>
    {
        let (cc, ss, cs) = (c * c, s * s, c * s);
        let two_cs_xy = T::TWO * cs * m.xy;
        Sym3 {
            xx: cc * m.xx - two_cs_xy + ss * m.yy,
            yy: ss * m.xx + two_cs_xy + cc * m.yy,
            zz: m.zz,
            xy: cs * (m.xx - m.yy) + (cc - ss) * m.xy,
            xz: c * m.xz - s * m.yz,
            yz: s * m.xz + c * m.yz,
        }
    }

    #[inline(always)]
    fn get<T: Float>(
        v: Vec3<T>,
    ) -> T
    {
        v.z
    }

    #[inline(always)]
    fn add<T: Float>(
        v: Vec3<T>,
        x: T,
    ) -> Vec3<T>
    {
        Vec3::new(v.x, v.y, v.z + x)
    }

    #[inline(always)]
    fn cross<T: Float>(
        a: Vec3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(a.y, -a.x, T::ZERO)
    }

    #[inline(always)]
    fn column<T: Float>(
        s: &Sym3<T>,
    ) -> Vec3<T>
    {
        Vec3::new(s.xz, s.yz, s.zz)
    }
}

/// Rotation by angle `q` about axis `A`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AxisRotation<T, A> {
    /// `cos q`.
    pub c: T,
    /// `sin q`.
    pub s: T,
    axis: PhantomData<A>,
}

impl<T: Float, A: Axis> AxisRotation<T, A> {
    
    /// Rotation by `q` in radians.
    #[inline(always)]
    pub fn from_angle(
        q: T,
    ) -> Self
    {
        let (s, c) = q.sin_cos();
        Self {
            c,
            s,
            axis: PhantomData,
        }
    }
}

impl<T: Float, A: Axis> Rotation<T> for AxisRotation<T, A> {

    #[inline(always)]
    fn apply(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        A::rotate(self.c, self.s, v)
    }

    #[inline(always)]
    fn apply_t(
        &self,
        v: Vec3<T>,
    ) -> Vec3<T>
    {
        A::rotate_t(self.c, self.s, v)
    }

    #[inline(always)]
    fn congruence_t(
        &self,
        s: &Sym3<T>,
    ) -> Sym3<T>
    {
        A::congruence_t(self.c, self.s, s)
    }

    fn matrix(
        &self,
    ) -> Mat3<T>
    {
        let e = |v| self.apply(v);
        let (o, z) = (T::ONE, T::ZERO);
        Mat3([[e(Vec3::new(o, z, z)).x, e(Vec3::new(z, o, z)).x, e(Vec3::new(z, z, o)).x], [e(Vec3::new(o, z, z)).y, e(Vec3::new(z, o, z)).y, e(Vec3::new(z, z, o)).y], [e(Vec3::new(o, z, z)).z, e(Vec3::new(z, o, z)).z, e(Vec3::new(z, z, o)).z]])
    }
}

/// Plucker transform `X = rot(E) xlt(r)` from parent to child.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Xform<T, R = Mat3<T>> {
    /// Rotation `E` from parent to child.
    pub rot: R,
    /// Child origin in parent reference frame.
    pub r: Vec3<T>,
}

/// Transform of a revolute joint about axis `A` after a constant offset.
pub type JointXform<T, A> = Xform<T, AxisRotation<T, A>>;

impl<T: Float, R: Rotation<T>> Xform<T, R> {

    /// Motion vector `m` from parent to child coordinates.
    #[inline(always)]
    pub fn apply_motion(
        &self,
        m: Motion<T>,
    ) -> Motion<T>
    {
        Motion::new(self.rot.apply(m.ang), self.rot.apply(m.lin - self.r.cross(m.ang)))
    }

    /// Motion vector `m` from child to parent coordinates : the inverse of `apply_motion`.
    #[inline(always)]
    pub fn inv_apply_motion(
        &self,
        m: Motion<T>,
    ) -> Motion<T>
    {
        let ang = self.rot.apply_t(m.ang);
        Motion::new(ang, self.rot.apply_t(m.lin) + self.r.cross(ang))
    }

    /// Force vector `f` from parent to child coordinates.
    #[inline(always)]
    pub fn apply_force(
        &self,
        f: Force<T>,
    ) -> Force<T>
    {
        Force::new(self.rot.apply(f.ang - self.r.cross(f.lin)), self.rot.apply(f.lin))
    }

    /// Force vector `f` from child to parent coordinates : the inverse of `apply_force`.
    /// The RNEA and CRBA use it to pass a body's force, or a column of its inertia, to its parent.
    #[inline(always)]
    pub fn inv_apply_force(
        &self,
        f: Force<T>,
    ) -> Force<T>
    {
        let lin = self.rot.apply_t(f.lin);
        Force::new(self.rot.apply_t(f.ang) + self.r.cross(lin), lin)
    }

    /// Rigid body inertia from child to parent coordinates.
    #[inline(always)]
    pub fn inv_transform_inertia(
        &self,
        inertia: &RigidInertia<T>
    ) -> RigidInertia<T>
    {
        let m = inertia.mass;
        let r = self.r;
        let y = self.rot.apply_t(inertia.h);
        let z = y + r * (T::HALF * m);
        let rz2 = T::TWO * r.dot(z);
        let rot = self.rot.congruence_t(&inertia.ibar);
        let ibar = Sym3 {
            xx: rot.xx + rz2 - T::TWO * r.x * z.x,
            yy: rot.yy + rz2 - T::TWO * r.y * z.y,
            zz: rot.zz + rz2 - T::TWO * r.z * z.z,
            xy: rot.xy - (z.x * r.y + r.x * z.y),
            xz: rot.xz - (z.x * r.z + r.x * z.z),
            yz: rot.yz - (z.y * r.z + r.y * z.z),
        };
        RigidInertia {
            mass: m,
            h: y + r * m,
            ibar,
        }
    }

    /// The same transform with its rotation stored as a general 3x3 matrix.
    pub fn to_general(
        &self,
    ) -> Xform<T>
    {
        Xform {
            rot: self.rot.matrix(),
            r: self.r
        }
    }

    /// The 6x6 motion transform.
    pub fn to_array(
        &self,
    ) -> [[T; 6]; 6]
    {
        let e = self.rot.matrix();
        let er = e.mul_mat(&Mat3::skew(self.r));
        let mut out = [[T::ZERO; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = e.0[i][j];
                out[3 + i][3 + j] = e.0[i][j];
                out[3 + i][j] = -er.0[i][j];
            }
        }
        out
    }
}

impl<T: Float> Xform<T> {

    /// Identity transform.
    #[inline(always)]
    pub fn identity(
    ) -> Self
    {
        Self {
            rot: Mat3::identity(),
            r: Vec3::zero(),
        }
    }

    /// Composition of two transforms : `self` maps frame A to frame B and `next` maps frame B to frame C, so the result maps frame A to frame C.
    #[inline(always)]
    pub fn then<R: Rotation<T>>(
        &self,
        next: &Xform<T, R>
    ) -> Self
    {
        Self {
            rot: next.rot.matrix().mul_mat(&self.rot),
            r: self.r + self.rot.tr_mul_vec(next.r),
        }
    }

    /// Inverse transform : maps child coordinates back to parent coordinates.
    #[inline(always)]
    pub fn inverse(
        &self,
    ) -> Self
    {
        Self {
            rot: self.rot.transpose(),
            r: -self.rot.mul_vec(self.r),
        }
    }
}
