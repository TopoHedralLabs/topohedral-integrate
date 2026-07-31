//! Errors produced while evaluating integrands.

use thiserror::Error;

/// Coordinates at which an integrand was evaluated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvaluationPoint {
    /// A one-dimensional coordinate.
    OneDimensional(f64),
    /// A two-dimensional coordinate.
    TwoDimensional {
        /// Coordinate on the `u` axis.
        u: f64,
        /// Coordinate on the `v` axis.
        v: f64,
    },
}

/// An error encountered while evaluating an integral.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum IntegrationError {
    /// The integrand returned NaN or infinity.
    #[error("integrand returned non-finite value {value} at {point:?}")]
    NonFiniteIntegrand {
        /// Evaluation coordinates.
        point: EvaluationPoint,
        /// Non-finite value returned by the integrand.
        value: f64,
    },
}
