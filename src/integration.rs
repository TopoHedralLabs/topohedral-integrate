//! Results and errors produced while evaluating integrals.

use crate::{Interval, Rectangle};

use std::fmt;
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

impl fmt::Display for EvaluationPoint {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::OneDimensional(x) => write!(formatter, "x = {x}"),
            Self::TwoDimensional { u, v } => write!(formatter, "(u, v) = ({u}, {v})"),
        }
    }
}

/// Value and diagnostics produced by adaptive integration.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveResult {
    integral: f64,
    error_estimate: f64,
    terminal_region_count: usize,
    evaluation_count: usize,
}

impl AdaptiveResult {
    pub(crate) const fn new(
        integral: f64,
        error_estimate: f64,
        terminal_region_count: usize,
        evaluation_count: usize,
    ) -> Self {
        Self {
            integral,
            error_estimate,
            terminal_region_count,
            evaluation_count,
        }
    }

    /// Returns the high-order approximation to the integral.
    pub const fn integral(&self) -> f64 {
        self.integral
    }

    /// Returns the sum of the terminal regions' low/high differences.
    pub const fn error_estimate(&self) -> f64 {
        self.error_estimate
    }

    /// Returns the number of terminal regions.
    pub const fn terminal_region_count(&self) -> usize {
        self.terminal_region_count
    }

    /// Returns the total number of integrand evaluations.
    pub const fn evaluation_count(&self) -> usize {
        self.evaluation_count
    }
}

/// An error encountered while evaluating an integral.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum IntegrationError {
    /// The integrand returned NaN or infinity.
    #[error("integrand returned non-finite value {value} at {point}")]
    NonFiniteIntegrand {
        /// Evaluation coordinates.
        point: EvaluationPoint,
        /// Non-finite value returned by the integrand.
        value: f64,
    },
    /// The requested global tolerance could not be met within the configured depth limits.
    #[error("maximum refinement depth reached before convergence")]
    MaxDepthReached {
        /// Best result and diagnostics available when refinement stopped.
        partial_result: AdaptiveResult,
    },
    /// A one-dimensional region was too narrow to bisect in floating-point arithmetic.
    #[error("cannot subdivide interval {interval} because its midpoint is not interior")]
    NonProgressingInterval {
        /// Interval whose computed midpoint equalled one of its endpoints.
        interval: Interval,
        /// Best result and diagnostics available before the failed subdivision.
        partial_result: AdaptiveResult,
    },
    /// A two-dimensional region was too narrow to split in floating-point arithmetic.
    #[error(
        "cannot subdivide rectangle {rectangle} because an active-axis midpoint is not interior"
    )]
    NonProgressingRectangle {
        /// Rectangle that could not be subdivided as required.
        rectangle: Rectangle,
        /// Best result and diagnostics available before the failed subdivision.
        partial_result: AdaptiveResult,
    },
}
