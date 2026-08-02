//! Adaptive tensor-product quadrature for two-dimensional real-valued functions.

//{{{ crate imports
use crate::config::{validate_subdivisions, ConfigError, ConfigIssue, RuleAxis};
use crate::fixed::d2::{Quadrature as FixedQuadrature, TensorRule};
use crate::{
    AdaptiveResult, AxisDepths, IntegrationError, Interval, OptionsError, Rectangle, Tolerance,
};
//}}}

//{{{ struct: Region
#[derive(Clone, Copy, Debug)]
struct Region {
    domain: Rectangle,
    u_depth: usize,
    v_depth: usize,
    high_estimate: f64,
    error_estimate: f64,
}
//}}}

//{{{ collection: Builder
/// Consuming builder for two-dimensional adaptive quadrature.
///
/// See the [adaptive-quadrature guide](https://topohedrallabs.github.io/topohedral-integrate/latest/user-guide/adaptive-quadrature/).
#[derive(Clone, Debug, PartialEq)]
pub struct Builder {
    domain: Rectangle,
    low_rule: TensorRule,
    high_rule: TensorRule,
    tolerance: Tolerance,
    max_depth: AxisDepths,
    u_subdivisions: Vec<f64>,
    v_subdivisions: Vec<f64>,
}

impl Builder {
    /// Sets the maximum refinement depth independently on each axis.
    pub const fn max_depth(
        mut self,
        max_depth: AxisDepths,
    ) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Replaces the initial interior subdivision coordinates on both axes.
    ///
    /// # Errors
    ///
    /// Returns all detected subdivision validation issues.
    pub fn subdivisions<U, V>(
        mut self,
        u_points: U,
        v_points: V,
    ) -> Result<Self, ConfigError>
    where
        U: IntoIterator<Item = f64>,
        V: IntoIterator<Item = f64>,
    {
        let u = validate_subdivisions(self.domain.u(), u_points);
        let v = validate_subdivisions(self.domain.v(), v_points);
        match (u, v) {
            (Ok(u), Ok(v)) => {
                self.u_subdivisions = u;
                self.v_subdivisions = v;
                Ok(self)
            }
            (Err(u), Err(v)) => {
                let mut issues = u.issues().to_vec();
                issues.extend_from_slice(v.issues());
                Err(ConfigError::from_issues(issues))
            }
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    /// Validates both axis rule pairs and constructs a reusable adaptive quadrature.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] for every axis whose high rule's actual exactness does not exceed
    /// its low rule's.
    pub fn build(self) -> Result<AdaptiveQuadrature, ConfigError> {
        let mut issues = Vec::new();
        validate_rule_pair(
            self.low_rule.u(),
            self.high_rule.u(),
            RuleAxis::U,
            &mut issues,
        );
        validate_rule_pair(
            self.low_rule.v(),
            self.high_rule.v(),
            RuleAxis::V,
            &mut issues,
        );
        if !issues.is_empty() {
            return Err(ConfigError::from_issues(issues));
        }

        Ok(AdaptiveQuadrature {
            domain: self.domain,
            low_rule: FixedQuadrature::builder(self.domain, self.low_rule).build(),
            high_rule: FixedQuadrature::builder(self.domain, self.high_rule).build(),
            tolerance: self.tolerance,
            max_depth: self.max_depth,
            u_subdivisions: self.u_subdivisions,
            v_subdivisions: self.v_subdivisions,
        })
    }
}
//}}}

//{{{ collection: AdaptiveQuadrature
/// Reusable two-dimensional adaptive tensor-product quadrature configuration.
///
/// See the [two-dimensional adaptive example](https://topohedrallabs.github.io/topohedral-integrate/latest/user-guide/adaptive-quadrature/#two-dimensional-integration).
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveQuadrature {
    domain: Rectangle,
    low_rule: FixedQuadrature,
    high_rule: FixedQuadrature,
    tolerance: Tolerance,
    max_depth: AxisDepths,
    u_subdivisions: Vec<f64>,
    v_subdivisions: Vec<f64>,
}

impl AdaptiveQuadrature {
    /// Starts a consuming builder with a default maximum depth of 32 on both axes.
    pub fn builder(
        domain: Rectangle,
        low_rule: TensorRule,
        high_rule: TensorRule,
        tolerance: Tolerance,
    ) -> Builder {
        Builder {
            domain,
            low_rule,
            high_rule,
            tolerance,
            max_depth: AxisDepths::default(),
            u_subdivisions: Vec::new(),
            v_subdivisions: Vec::new(),
        }
    }

    /// Returns the complete integration domain.
    pub const fn domain(&self) -> Rectangle {
        self.domain
    }

    /// Returns the low-order tensor-product rule.
    pub const fn low_rule(&self) -> &TensorRule {
        self.low_rule.rule()
    }

    /// Returns the high-order tensor-product rule.
    pub const fn high_rule(&self) -> &TensorRule {
        self.high_rule.rule()
    }

    /// Returns the global convergence tolerance.
    pub const fn tolerance(&self) -> Tolerance {
        self.tolerance
    }

    /// Returns the maximum per-axis refinement depths.
    pub const fn max_depth(&self) -> AxisDepths {
        self.max_depth
    }

    /// Returns the validated initial `u`-axis subdivision coordinates.
    pub fn u_subdivision_points(&self) -> &[f64] {
        &self.u_subdivisions
    }

    /// Returns the validated initial `v`-axis subdivision coordinates.
    pub fn v_subdivision_points(&self) -> &[f64] {
        &self.v_subdivisions
    }

    /// Adaptively integrates `f` using a global error estimate.
    ///
    /// Both axes are split while both remain below their limits. Once one axis reaches its
    /// maximum depth, only the other axis is split.
    ///
    /// # Errors
    ///
    /// Returns [`IntegrationError::NonFiniteIntegrand`] if `f` returns NaN or infinity,
    /// [`IntegrationError::MaxDepthReached`] if the tolerance cannot be met within the configured
    /// per-axis depths, or [`IntegrationError::NonProgressingRectangle`] if an active-axis
    /// floating-point midpoint is not strictly interior.
    ///
    /// # Panics
    ///
    /// Panics raised by `f` are not caught and propagate to the caller.
    ///
    /// # Examples
    ///
    /// ```
    /// use topohedral_integrate::{
    ///     AdaptiveQuadrature2d, GaussFamily, GaussRule, PolynomialDegree, Rectangle,
    ///     TensorRule2d, Tolerance,
    /// };
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let low = GaussRule::for_degree(GaussFamily::Legendre, PolynomialDegree::new(3)?)?;
    /// let high = GaussRule::for_degree(GaussFamily::Legendre, PolynomialDegree::new(7)?)?;
    /// let quadrature = AdaptiveQuadrature2d::builder(
    ///     Rectangle::from_bounds(0.0, 1.0, 0.0, 1.0)?,
    ///     TensorRule2d::new(low.clone(), low),
    ///     TensorRule2d::new(high.clone(), high),
    ///     Tolerance::absolute(1e-10)?,
    /// )
    /// .build()?;
    /// let result = quadrature.integrate(|u, v| u * u + v * v)?;
    /// assert!((result.integral() - 2.0 / 3.0).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    pub fn integrate<F>(
        &self,
        mut f: F,
    ) -> Result<AdaptiveResult, IntegrationError>
    where
        F: FnMut(f64, f64) -> f64,
    {
        let evaluations_per_region = self.low_rule.point_count() + self.high_rule.point_count();
        let u_bounds = axis_bounds(self.domain.u(), &self.u_subdivisions);
        let v_bounds = axis_bounds(self.domain.v(), &self.v_subdivisions);
        let mut regions = Vec::with_capacity((u_bounds.len() - 1) * (v_bounds.len() - 1));
        let mut evaluation_count = 0;

        for u in u_bounds.windows(2) {
            for v in v_bounds.windows(2) {
                let domain = Rectangle::new(
                    Interval::new_unchecked(u[0], u[1]),
                    Interval::new_unchecked(v[0], v[1]),
                );
                regions.push(evaluate_region(
                    domain,
                    0,
                    0,
                    &self.low_rule,
                    &self.high_rule,
                    &mut f,
                )?);
                evaluation_count += evaluations_per_region;
            }
        }

        let mut global_integral: f64 = regions.iter().map(|region| region.high_estimate).sum();
        let mut global_error: f64 = regions.iter().map(|region| region.error_estimate).sum();

        loop {
            let result = AdaptiveResult::new(
                global_integral,
                global_error,
                regions.len(),
                evaluation_count,
            );
            if converged(&result, self.tolerance) {
                return Ok(result);
            }

            let Some(index) = regions
                .iter()
                .enumerate()
                .filter(|(_, region)| self.can_split(region))
                .max_by(|(_, left), (_, right)| {
                    left.error_estimate.total_cmp(&right.error_estimate)
                })
                .map(|(index, _)| index)
            else {
                return Err(IntegrationError::MaxDepthReached {
                    partial_result: result,
                });
            };

            let parent = regions[index];
            let split_u = parent.u_depth < self.max_depth.u().value();
            let split_v = parent.v_depth < self.max_depth.v().value();
            let u_midpoint = split_u.then(|| midpoint(parent.domain.u()));
            let v_midpoint = split_v.then(|| midpoint(parent.domain.v()));
            if u_midpoint.is_some_and(|midpoint| {
                midpoint <= parent.domain.u().lower() || midpoint >= parent.domain.u().upper()
            }) || v_midpoint.is_some_and(|midpoint| {
                midpoint <= parent.domain.v().lower() || midpoint >= parent.domain.v().upper()
            }) {
                return Err(IntegrationError::NonProgressingRectangle {
                    rectangle: parent.domain,
                    partial_result: result,
                });
            }

            let child_domains = child_domains(parent.domain, u_midpoint, v_midpoint);
            let u_depth = parent.u_depth + usize::from(split_u);
            let v_depth = parent.v_depth + usize::from(split_v);
            let child_count = child_domains.len();
            let mut children = Vec::with_capacity(child_count);
            for domain in child_domains {
                children.push(evaluate_region(
                    domain,
                    u_depth,
                    v_depth,
                    &self.low_rule,
                    &self.high_rule,
                    &mut f,
                )?);
            }

            let child_integral: f64 = children.iter().map(|child| child.high_estimate).sum();
            let child_error: f64 = children.iter().map(|child| child.error_estimate).sum();
            evaluation_count += child_count * evaluations_per_region;
            global_integral += child_integral - parent.high_estimate;
            global_error = (global_error - parent.error_estimate).max(0.0) + child_error;
            regions.swap_remove(index);
            regions.extend(children);
        }
    }

    fn can_split(
        &self,
        region: &Region,
    ) -> bool {
        region.u_depth < self.max_depth.u().value() || region.v_depth < self.max_depth.v().value()
    }
}
//}}}

//{{{ fun: validate_rule_pair
fn validate_rule_pair(
    low: &crate::GaussRule,
    high: &crate::GaussRule,
    axis: RuleAxis,
    issues: &mut Vec<ConfigIssue>,
) {
    let low = low.exactness().value();
    let high = high.exactness().value();
    if high <= low {
        issues.push(ConfigIssue::NonIncreasingRuleExactness { axis, low, high });
    }
}
//}}}

//{{{ fun: evaluate_region
fn evaluate_region<F>(
    domain: Rectangle,
    u_depth: usize,
    v_depth: usize,
    low_rule: &FixedQuadrature,
    high_rule: &FixedQuadrature,
    f: &mut F,
) -> Result<Region, IntegrationError>
where
    F: FnMut(f64, f64) -> f64,
{
    let low_estimate = low_rule.integrate_over(domain, &mut *f)?;
    let high_estimate = high_rule.integrate_over(domain, &mut *f)?;
    Ok(Region {
        domain,
        u_depth,
        v_depth,
        high_estimate,
        error_estimate: (high_estimate - low_estimate).abs(),
    })
}
//}}}

//{{{ fun: axis_bounds
fn axis_bounds(
    interval: Interval,
    subdivisions: &[f64],
) -> Vec<f64> {
    let mut bounds = Vec::with_capacity(subdivisions.len() + 2);
    bounds.push(interval.lower());
    bounds.extend_from_slice(subdivisions);
    bounds.push(interval.upper());
    bounds
}
//}}}

//{{{ fun: child_domains
fn child_domains(
    domain: Rectangle,
    u_midpoint: Option<f64>,
    v_midpoint: Option<f64>,
) -> Vec<Rectangle> {
    let u_intervals = split_axis(domain.u(), u_midpoint);
    let v_intervals = split_axis(domain.v(), v_midpoint);
    let mut children = Vec::with_capacity(u_intervals.len() * v_intervals.len());
    for u in &u_intervals {
        for v in &v_intervals {
            children.push(Rectangle::new(*u, *v));
        }
    }
    children
}
//}}}

//{{{ fun: split_axis
fn split_axis(
    interval: Interval,
    midpoint: Option<f64>,
) -> Vec<Interval> {
    match midpoint {
        Some(midpoint) => vec![
            Interval::new_unchecked(interval.lower(), midpoint),
            Interval::new_unchecked(midpoint, interval.upper()),
        ],
        None => vec![interval],
    }
}
//}}}

//{{{ fun: converged
fn converged(
    result: &AdaptiveResult,
    tolerance: Tolerance,
) -> bool {
    result.error_estimate()
        <= tolerance.absolute_value() + tolerance.relative_value() * result.integral().abs()
}
//}}}

//{{{ fun: midpoint
fn midpoint(interval: Interval) -> f64 {
    let direct = interval.lower() + 0.5 * (interval.upper() - interval.lower());
    if direct.is_finite() {
        direct
    } else {
        0.5 * interval.lower() + 0.5 * interval.upper()
    }
}
//}}}

//{{{ fun: adaptive_quad
/// Builds a two-dimensional adaptive quadrature and integrates `f` once.
/// See the [one-shot example](https://topohedrallabs.github.io/topohedral-integrate/latest/user-guide/adaptive-quadrature/#one-shot-helpers).
///
/// Use [`AdaptiveQuadrature::builder`] to configure the consuming `builder`. Construct an
/// [`AdaptiveQuadrature`] directly when the same configuration will be reused.
///
/// # Errors
///
/// Returns [`OptionsError`] if either axis rule pair is invalid or adaptive integration fails.
///
/// # Panics
///
/// Panics raised by `f` are not caught and propagate to the caller.
pub fn adaptive_quad<F>(
    f: F,
    builder: Builder,
) -> Result<AdaptiveResult, OptionsError>
where
    F: FnMut(f64, f64) -> f64,
{
    Ok(builder.build()?.integrate(f)?)
}
//}}}
