//! This module provides methods for performing fixed quadrature rules for one-dimensional and
//! two-dimensional real-valued functions.
//!
//! There are two entry points for the 1D fixed quadrature algorithm:
//!
//! - The function `fixed_quad`, which can be used when one merely wants to compute the
//!   integral of a function over a given interval once and therefore does not wish to store the
//!   quadrature rule itself.
//! - The struct `FixedQuad`, which will store the quadrature rule and can be re-used for
//!   different functions.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

pub mod d1;
pub mod d2;
