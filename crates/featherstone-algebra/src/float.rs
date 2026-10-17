//! The float type of every quantity with 'f32' for GPU usage and 'f64' for validation.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// 'f32' or 'f64'.
pub trait Float:
    Copy + Debug + Default + PartialEq + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> + AddAssign + SubAssign + MulAssign
{
    /// '0.0'
    const ZERO: Self;
    /// '1.0'
    const ONE: Self;
    /// '2.0'
    const TWO: Self;
    /// '0.5'
    const HALF: Self;
    /// Machine epsilon.
    const EPSILON: Self;

    /// Value nearest to 'x'.
    fn from_f64(x: f64) -> Self;
    /// Value as 'f64'.
    fn to_f64(self) -> f64;
    /// Square root.
    fn sqrt(self) -> Self;
    /// '(sin x, cos x)'.
    fn sin_cos(self) -> (Self, Self);
    /// 'self * a + b' with one rounding.
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// '1.0 / self'.
    fn recip(self) -> Self;
    /// 'true' if neither infinite nor NaN.
    fn is_finite(self) -> bool;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// Larger of 'self' and 'other' ; 'other' when they do not compare.
    fn max(self, other: Self) -> Self;
    /// Smaller of 'self' and 'other' ; 'other' when they do not compare.
    fn min(self, other: Self) -> Self;
    /// 'self' limited to '[lo, hi]'.
    fn clamp(self, lo: Self, hi: Self) -> Self;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const HALF: Self = 0.5;
            const EPSILON: Self = <$t>::EPSILON;

            #[inline(always)]
            fn from_f64(
                x: f64,
            ) -> Self
            {
                x as $t
            }

            #[inline(always)]
            fn to_f64(
                self,
            ) -> f64
            {
                self as f64
            }

            #[inline(always)]
            fn sqrt(
                self,
            ) -> Self
            {
                <$t>::sqrt(self)
            }

            #[inline(always)]
            fn sin_cos(
                self,
            ) -> (Self, Self)
            {
                <$t>::sin_cos(self)
            }

            #[inline(always)]
            fn mul_add(
                self,
                a: Self,
                b: Self,
            ) -> Self
            {
                <$t>::mul_add(self, a, b)
            }

            #[inline(always)]
            fn abs(
                self
            ) -> Self
            {
                <$t>::abs(self)
            }

            #[inline(always)]
            fn recip(
                self
            ) -> Self
            {
                1.0 / self
            }

            #[inline(always)]
            fn is_finite(
                self
            ) -> bool
            {
                <$t>::is_finite(self)
            }

            #[inline(always)]
            fn tanh(
                self,
            ) -> Self
            {
                <$t>::tanh(self)
            }

            #[inline(always)]
            fn max(
                self,
                other: Self,
            ) -> Self
            {
                if self > other { self } else { other }
            }

            #[inline(always)]
            fn min(
                self,
                other: Self,
            ) -> Self
            {
                if self < other { self } else { other }
            }

            #[inline(always)]
            fn clamp(
                self,
                lo: Self,
                hi: Self,
            ) -> Self
            {
                Float::min(Float::max(self, lo), hi)
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);
