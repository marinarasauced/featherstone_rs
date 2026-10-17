//! Algebra of the rigid body dynamics as in Featherstone's Rigid Body Dynamics Algorithms (RBDA) Chapter 2 and Appendix A.
//!
//! All functions are generic over the float type, and any that are hot are inline with index loops for cuda-oxide.

#![forbid(unsafe_code)]

pub mod float;
pub mod linalg;
pub mod spatial;
