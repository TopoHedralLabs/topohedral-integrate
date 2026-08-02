//! This module provides methods for performing fixed quadrature rules for one-dimensional and
//! two-dimensional real-valued functions.
//!
//! There are two entry points for the 1D fixed quadrature algorithm:
//!
//! - The function `fixed_quad`, which can be used when one merely wants to compute the
//!   integral of a function over a given interval once and therefore does not wish to store the
//!   quadrature rule itself.
//! - A reusable quadrature, which stores the mapped rule for use with multiple
//!   different functions.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

pub(crate) mod d1;
pub(crate) mod d2;
