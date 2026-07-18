//! This crate provides a collection of quadature rules for numerical integration along with
//! algorithms for adaptive integration build on top of these rules.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

mod common;
mod adaptive;
mod fixed;
mod gauss;

pub use common::{OptionsError, OptionsVerify};
pub use adaptive::d1::{adaptive_quad as adaptive_quad_1d, AdaptiveQuadOpts as AdaptiveQuadOpts1D, AdaptiveQuadResult as AdaptiveQuadResult1D};
pub use adaptive::d2::{adaptive_quad as adaptive_quad_2d, AdaptiveQuadOpts as AdaptiveQuadOpts2D, AdaptiveQuadResult as AdaptiveQuadResult2D};
pub use fixed::d1::{fixed_quad as fixed_quad_1d, FixedQuadOpts as FixedQuadOpts1D, FixedQuad as FixedQuad1D};
pub use fixed::d2::{fixed_quad as fixed_quad_2d, FixedQuadOpts as FixedQuadOpts2D, FixedQuad as FixedQuad2D};
pub use gauss::{GaussQuad, GaussQuadType, GuassQuadSet, get_lobatto_points, get_legendre_points};

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
