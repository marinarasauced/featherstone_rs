//! Spatial Vector Algebra from Featherstone's Rigid Body Dynamics Algorithms (RBDA), Ch. 2 and App. A.
//!
//! Conventions :
//! - Motion vectors are `[w; v]` and force vectors are `[n; f]`, with the angular part first.
//! - A transform `X` from a parent frame to a child frame stores `E`, the rotation from parent to child coordinates, and `r`, the child origin in parent coordinates : `X = rot(E) xlt(r)`.
//! - A rigid body inertia stores the mass, the first moment of mass `h = m c`, and the rotational inertia about the frame origin.
//! - Quaternions are `(w, x, y, z)`, as in MuJoCo.

use crate::float::Float;
use crate::linalg::{Mat3, Sym3, Sym6, Vec3};
use crate::rotation::{AxisRotation, Rotation};

/// Spatial motion vector `[w; v]`, a velocity or an acceleration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Motion<T> {
    /// Angular part `w`.
    pub ang: Vec3<T>,
    /// Linear part `v` : the velocity of the point of the body at the frame origin.
    pub lin: Vec3<T>,
}

/// Spatial force vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Force<T> {
    /// Moment about the frame origin.
    pub ang: Vec3<T>,
    /// Force `f`.
    pub lin: Vec3<T>,
}

/// The constructors and conversions that `Motion` and `Force` have in common.
macro_rules! impl_spatial_vector {
    ($name:ident) => {
        impl<T: Float> $name<T> {

            /// From the angular and the linear part.
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

            /// Cast to the float type `U`.
            #[inline(always)]
            pub fn cast<U: Float>(
                self,
            ) -> $name<U>
            {
                $name::new(self.ang.cast(), self.lin.cast())
            }

            /// From `[ang; lin]`.
            #[inline(always)]
            pub fn from_array(
                a: [T; 6],
            ) -> Self
            {
                Self::new(Vec3::new(a[0], a[1], a[2]), Vec3::new(a[3], a[4], a[5]))
            }

            /// `[ang; lin]` as 6 numbers.
            #[inline(always)]
            pub fn to_array(
                self,
            ) -> [T; 6]
            {
                [self.ang.x, self.ang.y, self.ang.z, self.lin.x, self.lin.y, self.lin.z]
            }
        }

        impl_linear_ops!($name { ang, lin });
    };
}

impl_spatial_vector!(Motion);
impl_spatial_vector!(Force);

impl<T: Float> Motion<T> {

    /// Cross product `v x* f` of a motion vector and a force vector.
    #[inline(always)]
    pub fn cross_force(
        self,
        f: Force<T>,
    ) -> Force<T>
    {
        Force::new(self.ang.cross(f.ang) + self.lin.cross(f.lin), self.ang.cross(f.lin))
    }

    /// Cross product `v x m` of two motion vectors.
    #[inline(always)]
    pub fn cross_motion(
        self,
        m: Self,
    ) -> Self
    {
        Self::new(self.ang.cross(m.ang), self.ang.cross(m.lin) + self.lin.cross(m.ang))
    }

    /// Power `m^T f` of a force vector `f` on a motion vector `m`.
    #[inline(always)]
    pub fn power(
        self,
        f: Force<T>,
    ) -> T
    {
        self.ang.dot(f.ang) + self.lin.dot(f.lin)
    }
}

impl<T: Float> Force<T> {

    /// Force `force` acting at the point `p` ; i.e., `[p x force; force]`.
    #[inline(always)]
    pub fn from_point_force(
        p: Vec3<T>,
        force: Vec3<T>,
    ) -> Self
    {
        Self::new(p.cross(force), force)
    }
}

/// Spatial inertia of a rigid body in the coordinates of a frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RigidBodyInertia<T> {
    /// Mass `m`.
    pub mass: T,
    /// First moment of mass `h = m c` for the center of mass `c`.
    pub first_moment: Vec3<T>,
    /// Rotational inertia `I` about the frame origin.
    pub rotational: Sym3<T>,
}

impl<T: Float> RigidBodyInertia<T> {

    /// Zero inertia.
    #[inline(always)]
    pub fn zero(
    ) -> Self
    {
        Self {
            mass: T::ZERO,
            first_moment: Vec3::zero(),
            rotational: Sym3::zero(),
        }
    }

    /// Center of mass `c = h / m`.
    #[inline(always)]
    pub fn com(
        &self,
    ) -> Vec3<T>
    {
        self.first_moment * self.mass.recip()
    }

    /// Product `I m` of this inertia with a motion vector `m`.
    #[inline(always)]
    pub fn mul_motion(
        &self,
        m: Motion<T>,
    ) -> Force<T>
    {
        Force::new(self.rotational.mul_vec(m.ang) + self.first_moment.cross(m.lin), m.lin * self.mass - self.first_moment.cross(m.ang))
    }

    /// Cast to the float type `U`.
    #[inline(always)]
    pub fn cast<U: Float>(
        &self,
    ) -> RigidBodyInertia<U>
    {
        RigidBodyInertia {
            mass: self.mass.cast(),
            first_moment: self.first_moment.cast(),
            rotational: self.rotational.cast(),
        }
    }

    /// Inertia of a body from its mass, its center of mass `com`, and its rotational inertia `inertia_com` about the center of mass.
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
            first_moment: com * mass,
            rotational: inertia_com + parallel * mass,
        }
    }

    /// The 6x6 matrix in `[ang; lin]` order, row major.
    pub fn to_dense(
        &self,
    ) -> [[T; 6]; 6]
    {
        self.to_sym6().to_dense()
    }

    /// The 6x6 matrix `[I, h x; -h x, m 1]` as a [`Sym6`].
    #[inline(always)]
    pub fn to_sym6(
        &self,
    ) -> Sym6<T>
    {
        let (z, m) = (T::ZERO, self.mass);
        Sym6 {
            angular: self.rotational,
            coupling: Mat3::skew(self.first_moment),
            linear: Sym3 {
                xx: m,
                yy: m,
                zz: m,
                xy: z,
                xz: z,
                yz: z,
            },
        }
    }
}

impl_linear_ops!(RigidBodyInertia { mass, first_moment, rotational });

/// Plucker transform `X = rot(E) xlt(r)` from parent to child.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Transform<T, R = Mat3<T>> {
    /// Rotation `E` from parent to child coordinates.
    pub rotation: R,
    /// Child origin `r` in parent coordinates.
    pub origin: Vec3<T>,
}

/// Transform of a revolute joint about the axis `A` after a constant offset.
pub type JointTransform<T, A> = Transform<T, AxisRotation<T, A>>;

impl<T: Float, R: Rotation<T>> Transform<T, R> {

    /// Force vector `f` from parent to child coordinates.
    #[inline(always)]
    pub fn force_to_child(
        &self,
        f: Force<T>,
    ) -> Force<T>
    {
        Force::new(self.rotation.to_child(f.ang - self.origin.cross(f.lin)), self.rotation.to_child(f.lin))
    }

    /// Force vector `f` from child to parent coordinates.
    #[inline(always)]
    pub fn force_to_parent(
        &self,
        f: Force<T>,
    ) -> Force<T>
    {
        let lin = self.rotation.to_parent(f.lin);
        Force::new(self.rotation.to_parent(f.ang) + self.origin.cross(lin), lin)
    }

    /// Rigid body inertia from child to parent coordinates.
    #[inline(always)]
    pub fn inertia_to_parent(
        &self,
        inertia: &RigidBodyInertia<T>,
    ) -> RigidBodyInertia<T>
    {
        let m = inertia.mass;
        let r = self.origin;
        let y = self.rotation.to_parent(inertia.first_moment);
        let z = y + r * (T::HALF * m);
        let rz2 = T::TWO * r.dot(z);
        let rot = self.rotation.symmetric_to_parent(&inertia.rotational);
        let rotational = Sym3 {
            xx: rot.xx + rz2 - T::TWO * r.x * z.x,
            yy: rot.yy + rz2 - T::TWO * r.y * z.y,
            zz: rot.zz + rz2 - T::TWO * r.z * z.z,
            xy: rot.xy - (z.x * r.y + r.x * z.y),
            xz: rot.xz - (z.x * r.z + r.x * z.z),
            yz: rot.yz - (z.y * r.z + r.y * z.z),
        };
        RigidBodyInertia {
            mass: m,
            first_moment: y + r * m,
            rotational,
        }
    }

    /// Motion vector `m` from parent to child coordinates.
    #[inline(always)]
    pub fn motion_to_child(
        &self,
        m: Motion<T>,
    ) -> Motion<T>
    {
        Motion::new(self.rotation.to_child(m.ang), self.rotation.to_child(m.lin - self.origin.cross(m.ang)))
    }

    /// Motion vector `m` from child to parent coordinates.
    #[inline(always)]
    pub fn motion_to_parent(
        &self,
        m: Motion<T>,
    ) -> Motion<T>
    {
        let ang = self.rotation.to_parent(m.ang);
        Motion::new(ang, self.rotation.to_parent(m.lin) + self.origin.cross(ang))
    }

    /// The 6x6 motion transform `[E, 0; -E r x, E]`, row major.
    pub fn to_dense(
        &self,
    ) -> [[T; 6]; 6]
    {
        let e = self.rotation.to_mat3();
        let er = e.mul_mat(&Mat3::skew(self.origin));
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

    /// The same transform with its rotation stored as a general 3x3 matrix.
    pub fn with_mat3_rotation(
        &self,
    ) -> Transform<T>
    {
        Transform {
            rotation: self.rotation.to_mat3(),
            origin: self.origin,
        }
    }
}

impl<T: Float> Transform<T> {

    /// Identity transform.
    pub fn identity(
    ) -> Self
    {
        Self {
            rotation: Mat3::identity(),
            origin: Vec3::zero(),
        }
    }

    /// Inverse transform that maps child coordinates back to parent coordinates.
    pub fn inverse(
        &self,
    ) -> Self
    {
        Self {
            rotation: self.rotation.transpose(),
            origin: -self.rotation.mul_vec(self.origin),
        }
    }

    /// Composition of two transforms.
    pub fn then<R: Rotation<T>>(
        &self,
        next: &Transform<T, R>,
    ) -> Self
    {
        Self {
            rotation: next.rotation.to_mat3().mul_mat(&self.rotation),
            origin: self.origin + self.rotation.mul_vec_t(next.origin),
        }
    }
}
