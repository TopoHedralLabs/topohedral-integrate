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
mod config;
mod fixed;
mod gauss;
mod integration;

pub use adaptive::d1::adaptive_quad as adaptive_quad_1d;
pub use adaptive::d1::AdaptiveQuadrature as AdaptiveQuadrature1D;
pub use adaptive::d1::Builder as AdaptiveQuadratureBuilder1D;
pub use adaptive::d2::adaptive_quad as adaptive_quad_2d;
pub use adaptive::d2::AdaptiveQuadrature as AdaptiveQuadrature2D;
pub use adaptive::d2::Builder as AdaptiveQuadratureBuilder2D;
pub use common::OptionsError;
pub use config::AxisDepths;
pub use config::ConfigError;
pub use config::ConfigIssue;
pub use config::Interval;
pub use config::PointCount;
pub use config::PolynomialDegree;
pub use config::Rectangle;
pub use config::RefinementDepth;
pub use config::RuleAxis;
pub use config::Tolerance;
pub use fixed::d1::fixed_quad as fixed_quad_1d;
pub use fixed::d1::Builder as FixedQuadratureBuilder1D;
pub use fixed::d1::FixedQuad as FixedQuadrature1D;
pub use fixed::d1::FixedQuadOpts as FixedQuadOpts1D;
pub use fixed::d1::Node as FixedNode1D;
pub use fixed::d2::fixed_quad as fixed_quad_2d;
pub use fixed::d2::Builder as FixedQuadratureBuilder2D;
pub use fixed::d2::FixedQuad as FixedQuadrature2D;
pub use fixed::d2::FixedQuadOpts as FixedQuadOpts2D;
pub use fixed::d2::Node as FixedNode2D;
pub use fixed::d2::TensorRule as TensorRule2D;
pub use gauss::legendre_rules;
pub use gauss::lobatto_rules;
pub use gauss::GaussFamily;
pub use gauss::GaussRule;
pub use gauss::GaussRuleSet;
pub use gauss::RuleError;
pub use integration::AdaptiveResult;
pub use integration::EvaluationPoint;
pub use integration::IntegrationError;

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
