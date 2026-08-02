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

//{{{ collection: modules
mod adaptive;
mod common;
mod config;
mod fixed;
mod gauss;
mod integration;
//}}}

//{{{ collection: public exports
pub use adaptive::d1::adaptive_quad as adaptive_quad_1d;
pub use adaptive::d1::AdaptiveQuadrature as AdaptiveQuadrature1d;
pub use adaptive::d1::Builder as AdaptiveQuadratureBuilder1d;
pub use adaptive::d2::adaptive_quad as adaptive_quad_2d;
pub use adaptive::d2::AdaptiveQuadrature as AdaptiveQuadrature2d;
pub use adaptive::d2::Builder as AdaptiveQuadratureBuilder2d;
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
pub use fixed::d1::Builder as FixedQuadratureBuilder1d;
pub use fixed::d1::Node as FixedNode1d;
pub use fixed::d1::Quadrature as FixedQuadrature1d;
pub use fixed::d2::fixed_quad as fixed_quad_2d;
pub use fixed::d2::Builder as FixedQuadratureBuilder2d;
pub use fixed::d2::Node as FixedNode2d;
pub use fixed::d2::Quadrature as FixedQuadrature2d;
pub use fixed::d2::TensorRule as TensorRule2d;
pub use gauss::legendre_rules;
pub use gauss::lobatto_rules;
pub use gauss::GaussFamily;
pub use gauss::GaussRule;
pub use gauss::GaussRuleSet;
pub use gauss::RuleError;
pub use integration::AdaptiveResult;
pub use integration::EvaluationPoint;
pub use integration::IntegrationError;
//}}}
