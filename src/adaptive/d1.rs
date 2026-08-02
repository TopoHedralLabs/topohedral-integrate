//! Adaptive quadrature for one-dimensional real-valued functions.

//{{{ crate imports
use crate::config::{validate_subdivisions, ConfigError, ConfigIssue, RuleAxis};
use crate::fixed::d1::Quadrature as FixedQuadrature;
use crate::{
    AdaptiveResult, GaussRule, IntegrationError, Interval, OptionsError, RefinementDepth, Tolerance,
};
//}}}

//{{{ struct: Region
#[derive(Clone, Copy, Debug)]
struct Region {
    domain: Interval,
    depth: usize,
    high_estimate: f64,
    error_estimate: f64,
}
//}}}

//{{{ collection: Builder
/// Consuming builder for one-dimensional adaptive quadrature.
///
/// See the [adaptive-quadrature guide](https://topohedrallabs.github.io/topohedral-integrate/latest/user-guide/adaptive-quadrature/).
#[derive(Clone, Debug, PartialEq)]
pub struct Builder {
    domain: Interval,
    low_rule: GaussRule,
    high_rule: GaussRule,
    tolerance: Tolerance,
    max_depth: RefinementDepth,
    subdivisions: Vec<f64>,
}

impl Builder {
    /// Sets the maximum number of bisections permitted from each initial region.
    pub const fn max_depth(
        mut self,
        max_depth: RefinementDepth,
    ) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Replaces the initial interior subdivision points.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] unless the points are finite, strictly increasing, unique, and
    /// strictly interior to the integration domain.
    pub fn subdivisions<I>(
        mut self,
        points: I,
    ) -> Result<Self, ConfigError>
    where
        I: IntoIterator<Item = f64>,
    {
        self.subdivisions = validate_subdivisions(self.domain, points)?;
        Ok(self)
    }

    /// Validates the rule pair and constructs a reusable adaptive quadrature.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] unless the high rule's actual exactness exceeds the low rule's.
    pub fn build(self) -> Result<AdaptiveQuadrature, ConfigError> {
        let low_exactness = self.low_rule.exactness().value();
        let high_exactness = self.high_rule.exactness().value();
        if high_exactness <= low_exactness {
            return Err(ConfigError::from_issues(vec![
                ConfigIssue::NonIncreasingRuleExactness {
                    axis: RuleAxis::OneDimensional,
                    low: low_exactness,
                    high: high_exactness,
                },
            ]));
        }

        Ok(AdaptiveQuadrature {
            domain: self.domain,
            low_rule: FixedQuadrature::builder(self.domain, self.low_rule).build(),
            high_rule: FixedQuadrature::builder(self.domain, self.high_rule).build(),
            tolerance: self.tolerance,
            max_depth: self.max_depth,
            subdivisions: self.subdivisions,
        })
    }
}
//}}}

//{{{ collection: AdaptiveQuadrature
/// Reusable one-dimensional adaptive quadrature configuration.
///
/// See the [one-dimensional adaptive example](https://topohedrallabs.github.io/topohedral-integrate/latest/user-guide/adaptive-quadrature/#one-dimensional-integration).
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveQuadrature {
    domain: Interval,
    low_rule: FixedQuadrature,
    high_rule: FixedQuadrature,
    tolerance: Tolerance,
    max_depth: RefinementDepth,
    subdivisions: Vec<f64>,
}

impl AdaptiveQuadrature {
    /// Starts a consuming builder with a default maximum depth of 32.
    pub fn builder(
        domain: Interval,
        low_rule: GaussRule,
        high_rule: GaussRule,
        tolerance: Tolerance,
    ) -> Builder {
        Builder {
            domain,
            low_rule,
            high_rule,
            tolerance,
            max_depth: RefinementDepth::default(),
            subdivisions: Vec::new(),
        }
    }

    /// Returns the complete integration domain.
    pub const fn domain(&self) -> Interval {
        self.domain
    }

    /// Returns the low-order Gaussian rule.
    pub const fn low_rule(&self) -> &GaussRule {
        self.low_rule.rule()
    }

    /// Returns the high-order Gaussian rule.
    pub const fn high_rule(&self) -> &GaussRule {
        self.high_rule.rule()
    }

    /// Returns the global convergence tolerance.
    pub const fn tolerance(&self) -> Tolerance {
        self.tolerance
    }

    /// Returns the maximum refinement depth.
    pub const fn max_depth(&self) -> RefinementDepth {
        self.max_depth
    }

    /// Returns the validated initial subdivision points.
    pub fn subdivision_points(&self) -> &[f64] {
        &self.subdivisions
    }

    /// Adaptively integrates `f` using a global error estimate.
    ///
    /// The returned integral is the sum of high-order estimates. On each iteration, the
    /// splittable terminal region with the largest estimated error is bisected.
    ///
    /// # Errors
    ///
    /// Returns [`IntegrationError::NonFiniteIntegrand`] if `f` returns NaN or infinity,
    /// [`IntegrationError::MaxDepthReached`] if the tolerance cannot be met within the configured
    /// depth, or [`IntegrationError::NonProgressingInterval`] if a floating-point midpoint is not
    /// strictly interior.
    ///
    /// # Panics
    ///
    /// Panics raised by `f` are not caught and propagate to the caller.
    ///
    /// # Examples
    ///
    /// ```
    /// use topohedral_integrate::{
    ///     AdaptiveQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree, Tolerance,
    /// };
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let low = GaussRule::for_degree(GaussFamily::Legendre, PolynomialDegree::new(3)?)?;
    /// let high = GaussRule::for_degree(GaussFamily::Legendre, PolynomialDegree::new(7)?)?;
    /// let quadrature = AdaptiveQuadrature1d::builder(
    ///     Interval::new(-1.0, 1.0)?,
    ///     low,
    ///     high,
    ///     Tolerance::absolute(1e-10)?,
    /// )
    /// .build()?;
    /// let result = quadrature.integrate(|x| x * x)?;
    /// assert!((result.integral() - 2.0 / 3.0).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    pub fn integrate<F>(
        &self,
        mut f: F,
    ) -> Result<AdaptiveResult, IntegrationError>
    where
        F: FnMut(f64) -> f64,
    {
        let evaluations_per_region = self.low_rule.point_count() + self.high_rule.point_count();
        let mut evaluation_count = 0;
        let mut regions = Vec::with_capacity(self.subdivisions.len() + 1);
        let mut lower = self.domain.lower();

        for upper in self
            .subdivisions
            .iter()
            .copied()
            .chain(std::iter::once(self.domain.upper()))
        {
            let domain = Interval::new_unchecked(lower, upper);
            regions.push(evaluate_region(
                domain,
                0,
                &self.low_rule,
                &self.high_rule,
                &mut f,
            )?);
            evaluation_count += evaluations_per_region;
            lower = upper;
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
                .filter(|(_, region)| region.depth < self.max_depth.value())
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
            let midpoint = midpoint(parent.domain);
            if midpoint <= parent.domain.lower() || midpoint >= parent.domain.upper() {
                return Err(IntegrationError::NonProgressingInterval {
                    interval: parent.domain,
                    partial_result: result,
                });
            }

            let child_depth = parent.depth + 1;
            let left = evaluate_region(
                Interval::new_unchecked(parent.domain.lower(), midpoint),
                child_depth,
                &self.low_rule,
                &self.high_rule,
                &mut f,
            )?;
            let right = evaluate_region(
                Interval::new_unchecked(midpoint, parent.domain.upper()),
                child_depth,
                &self.low_rule,
                &self.high_rule,
                &mut f,
            )?;
            evaluation_count += 2 * evaluations_per_region;
            global_integral += left.high_estimate + right.high_estimate - parent.high_estimate;
            global_error = (global_error - parent.error_estimate).max(0.0)
                + left.error_estimate
                + right.error_estimate;
            regions.swap_remove(index);
            regions.push(left);
            regions.push(right);
        }
    }
}
//}}}

//{{{ fun: evaluate_region
fn evaluate_region<F>(
    domain: Interval,
    depth: usize,
    low_rule: &FixedQuadrature,
    high_rule: &FixedQuadrature,
    f: &mut F,
) -> Result<Region, IntegrationError>
where
    F: FnMut(f64) -> f64,
{
    let low_estimate = low_rule.integrate_over(domain, &mut *f)?;
    let high_estimate = high_rule.integrate_over(domain, &mut *f)?;
    Ok(Region {
        domain,
        depth,
        high_estimate,
        error_estimate: (high_estimate - low_estimate).abs(),
    })
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
/// Builds a one-dimensional adaptive quadrature and integrates `f` once.
/// See the [one-shot example](https://topohedrallabs.github.io/topohedral-integrate/latest/user-guide/adaptive-quadrature/#one-shot-helpers).
///
/// Use [`AdaptiveQuadrature::builder`] to configure the consuming `builder`. Construct an
/// [`AdaptiveQuadrature`] directly when the same configuration will be reused.
///
/// # Errors
///
/// Returns [`OptionsError`] if the rule pair is invalid or adaptive integration fails.
///
/// # Panics
///
/// Panics raised by `f` are not caught and propagate to the caller.
pub fn adaptive_quad<F>(
    f: F,
    builder: Builder,
) -> Result<AdaptiveResult, OptionsError>
where
    F: FnMut(f64) -> f64,
{
    Ok(builder.build()?.integrate(f)?)
}
//}}}
