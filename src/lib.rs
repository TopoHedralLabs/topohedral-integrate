//! Numerical quadrature rules for one- and two-dimensional real-valued functions.
//!
//! The crate provides fixed Gauss quadrature rules and adaptive algorithms built from pairs of
//! fixed rules. The dimensionality is encoded in the names of the public types and functions.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

mod adaptive;
mod common;
mod fixed;
mod gauss;

pub use adaptive::d1::adaptive_quad as adaptive_quad_1d;
pub use adaptive::d1::AdaptiveQuadOpts as AdaptiveQuadOpts1D;
pub use adaptive::d1::AdaptiveQuadResult as AdaptiveQuadResult1D;
pub use adaptive::d2::adaptive_quad as adaptive_quad_2d;
pub use adaptive::d2::AdaptiveQuadOpts as AdaptiveQuadOpts2D;
pub use adaptive::d2::AdaptiveQuadResult as AdaptiveQuadResult2D;
pub use common::OptionsError;
pub use fixed::d1::fixed_quad as fixed_quad_1d;
pub use fixed::d1::FixedQuad as FixedQuad1D;
pub use fixed::d1::FixedQuadOpts as FixedQuadOpts1D;
pub use fixed::d2::fixed_quad as fixed_quad_2d;
pub use fixed::d2::FixedQuad as FixedQuad2D;
pub use fixed::d2::FixedQuadOpts as FixedQuadOpts2D;
pub use gauss::get_legendre_points;
pub use gauss::get_lobatto_points;
pub use gauss::GaussQuad;
pub use gauss::GaussQuadType;
pub use gauss::GuassQuadSet;

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {
    use ctor::ctor;
    use topohedral_tracing::*;

    #[ctor]
    fn init_logger() {
        init().unwrap();
    }

    #[test]
    fn test_logging() {
        info!("Logging is working!");
    }
}
//}}}
