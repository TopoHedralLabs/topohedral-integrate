//! Numerical quadrature rules for one- and two-dimensional real-valued functions.
//!
//! The crate provides fixed Gauss quadrature rules and adaptive algorithms built from pairs of
//! fixed rules. The dimensionality is encoded in the names of the public types and functions.
//!
//! # Fixed quadrature
//!
//! ```
//! use topohedral_integrate::{
//!     FixedQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let domain = Interval::new(-1.0, 1.0)?;
//! let degree = PolynomialDegree::new(5)?;
//! let rule = GaussRule::for_degree(GaussFamily::Legendre, degree)?;
//! let quadrature = FixedQuadrature1d::builder(domain, rule).build();
//! let integral = quadrature.integrate(|x| x * x)?;
//! assert!((integral - 2.0 / 3.0).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```
//!
//! # Adaptive quadrature
//!
//! ```
//! use topohedral_integrate::{
//!     AdaptiveQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree, Tolerance,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let domain = Interval::new(0.0, 1.0)?;
//! let low = GaussRule::for_degree(GaussFamily::Legendre, PolynomialDegree::new(3)?)?;
//! let high = GaussRule::for_degree(GaussFamily::Legendre, PolynomialDegree::new(9)?)?;
//! let tolerance = Tolerance::new(1e-10, 1e-10)?;
//! let quadrature = AdaptiveQuadrature1d::builder(domain, low, high, tolerance).build()?;
//! let result = quadrature.integrate(|x| x.exp())?;
//! assert!((result.integral() - (std::f64::consts::E - 1.0)).abs() < 1e-9);
//! # Ok(())
//! # }
//! ```
//!
//! The optional `serde` feature serializes validated values, generated rules, nodes, errors, and
//! adaptive results. Deserialization re-applies constructor validation. The optional `trace`
//! feature enables instrumentation without enabling terminal colours.
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
