//! The float type of every quantity with `f32` for GPU usage and `f64` for validation.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Constants and basic operators.
pub trait Arithmetic:
    Copy + Add<Output = Self> + AddAssign + Div<Output = Self> + DivAssign + Mul<Output = Self> + DivAssign + Mul<Output = Self> + MulAssign + Neg<Output = Self> + Sub<Output = Self> + SubAssign
{
    /// Machine epsilon.
    const EPSILON: Self;
    /// `0.5`.
    const HALF: Self;
    /// `1.0`.
    const ONE: Self;
    /// `2.0`.
    const TWO: Self;
    /// `0.0`.
    const ZERO: Self;

    /// Absolute value.
    fn abs(self) -> Self;
    /// Operation : multiply, then add (with one rounding).
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Reciprocal.
    fn recip(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
}

/// Conditionals.
pub trait Compare:
    Copy + PartialOrd
{
    /// Clamp between min and max.
    fn clamp(self, min: Self, max: Self) -> Self;
    /// `true` if `self` is not infinite or nan.
    fn is_finite(self) -> bool;
    /// Larger of `self` and `other`.
    fn max(self, other: Self) -> Self;
    /// Smaller of `self` and `other`.
    fn min(self, other: Self) -> Self;
}

/// Conversions between types.
pub trait Convert:
    Copy
{
    ///Value nearest to `x`.
    fn from_f64(x: f64) -> Self;
    /// Value as `f64`.
    fn to_f64(self) -> f64;

    /// Value nearest to `self` in the float type `U`.
    #[inline(always)]
    fn cast<U: Convert>(
        self,
    ) -> U
    {
        U::from_f64(self.to_f64())
    }
}

/// Trigonometric and hyperbolic functions.
pub trait Trigonometry:
    Copy
{
    /// Sine and cosine of `self`.
    fn sin_cos(self) -> (Self, Self);
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
}

/// `f32` or `f64`.
pub trait Float:
    Arithmetic + Compare + Convert + Debug + Default + Trigonometry
{
}

macro_rules! impl_float {
    ($t:ty) => {

        impl Arithmetic for $t {

            const EPSILON: Self = <$t>::EPSILON;
            const HALF: Self = 0.5;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const ZERO: Self = 0.0;

            #[inline(always)]
            fn abs(
                self,
            ) -> Self
            {
                <$t>::abs(self)
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
            fn recip(
                self,
            ) -> Self
            {
                1.0 / self
            }

            #[inline(always)]
            fn sqrt(
                self,
            ) -> Self
            {
                <$t>::sqrt(self)
            }
        }

        impl Compare for $t {

            #[inline(always)]
            fn clamp(
                self,
                lo: Self,
                hi: Self,
            ) -> Self
            {
                Compare::min(Compare::max(self, lo), hi)
            }

            #[inline(always)]
            fn is_finite(
                self,
            ) -> bool
            {
                <$t>::is_finite(self)
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
        }

        impl Convert for $t {

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
        }

        impl Trigonometry for $t {

            #[inline(always)]
            fn sin_cos(
                self,
            ) -> (Self, Self)
            {
                <$t>::sin_cos(self)
            }

            #[inline(always)]
            fn tanh(
                self,
            ) -> Self
            {
                <$t>::tanh(self)
            }
        }

        impl Float for $t {}
    };
}

impl_float!(f32);
impl_float!(f64);
