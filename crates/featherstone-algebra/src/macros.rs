//! Macros shared by the modules of the crate.
//!
//! `Add`, `Sub`, `Neg`, `Mul<T>`, `AddAssign` and `SubAssign` for a struct of named fields that have these operators.

/// Every linear type in the crate (`Vec3`, `Sym3`, `Sym6`, `Motion`, `Force`, `RigidBodyInertia`) uses these same six operators.
macro_rules! impl_linear_ops {
    ($name:ident { $($field:ident),+ $(,)? }) => {
        impl<T: $crate::float::Float> ::core::ops::Add for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn add(
                self,
                rhs: Self,
            ) -> Self
            {
                Self { $($field: self.$field + rhs.$field),+ }
            }
        }

        impl<T: $crate::float::Float> ::core::ops::Sub for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn sub(
                self,
                rhs: Self,
            ) -> Self
            {
                Self { $($field: self.$field - rhs.$field),+ }
            }
        }

        impl<T: $crate::float::Float> ::core::ops::Neg for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn neg(
                self,
            ) -> Self
            {
                Self { $($field: -self.$field),+ }
            }
        }

        impl<T: $crate::float::Float> ::core::ops::Mul<T> for $name<T> {
            type Output = Self;
            #[inline(always)]
            fn mul(
                self,
                k: T,
            ) -> Self
            {
                Self { $($field: self.$field * k),+ }
            }
        }

        impl<T: $crate::float::Float> ::core::ops::AddAssign for $name<T> {
            #[inline(always)]
            fn add_assign(
                &mut self,
                rhs: Self,
            )
            {
                *self = *self + rhs;
            }
        }

        impl<T: $crate::float::Float> ::core::ops::SubAssign for $name<T> {
            #[inline(always)]
            fn sub_assign(
                &mut self,
                rhs: Self,
            )
            {
                *self = *self - rhs;
            }
        }
    };
}
