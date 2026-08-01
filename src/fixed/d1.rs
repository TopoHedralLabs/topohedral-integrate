//! Fixed quadrature for one-dimensional real-valued functions.

use crate::common::OptionsError;
use crate::config::{
    validate_subdivisions, ConfigError, ConfigIssue, Interval, PolynomialDegree, RuleAxis,
};
use crate::gauss::{legendre_rules, lobatto_rules, GaussFamily, GaussRule, MAX_DEGREE};
use crate::integration::{EvaluationPoint, IntegrationError};

/// Legacy configuration for one-dimensional fixed quadrature.
///
/// Every value is validated by [`FixedQuad::new`] rather than on assignment, so the fields are
/// private and reached through accessors. New code should prefer
/// [`crate::FixedQuadrature1D::builder`].
#[derive(Clone, Debug, PartialEq)]
pub struct FixedQuadOpts {
    gauss_type: GaussFamily,
    order: usize,
    bounds: (f64, f64),
    subdiv: Vec<f64>,
}

impl FixedQuadOpts {
    /// Describes a rule of the requested family and minimum polynomial exactness on `bounds`.
    pub const fn new(
        gauss_type: GaussFamily,
        order: usize,
        bounds: (f64, f64),
    ) -> Self {
        Self {
            gauss_type,
            order,
            bounds,
            subdiv: Vec::new(),
        }
    }

    /// Replaces the interior subdivision points. Empty input means no subdivision.
    #[must_use]
    pub fn with_subdivisions<I>(
        mut self,
        points: I,
    ) -> Self
    where
        I: IntoIterator<Item = f64>,
    {
        self.subdiv = points.into_iter().collect();
        self
    }

    /// Returns the Gauss quadrature family used on every subinterval.
    pub const fn gauss_type(&self) -> GaussFamily {
        self.gauss_type
    }

    /// Returns the minimum polynomial exactness requested for the rule.
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Returns the integration interval as `(lower, upper)`.
    pub const fn bounds(&self) -> (f64, f64) {
        self.bounds
    }

    /// Returns the requested interior subdivision points.
    pub fn subdivisions(&self) -> &[f64] {
        &self.subdiv
    }

    fn validate(&self) -> Result<(Interval, Vec<f64>), ConfigError> {
        let mut issues = Vec::new();
        if self.order > MAX_DEGREE {
            issues.push(ConfigIssue::UnsupportedRuleDegree {
                axis: RuleAxis::OneDimensional,
                family: self.gauss_type,
                degree: self.order,
                maximum: MAX_DEGREE,
            });
        }

        let domain = match Interval::new(self.bounds.0, self.bounds.1) {
            Ok(domain) => Some(domain),
            Err(error) => {
                issues.extend_from_slice(error.issues());
                None
            }
        };

        let mut subdivisions = Vec::new();
        if let Some(domain) = domain {
            match validate_subdivisions(domain, self.subdiv.iter().copied()) {
                Ok(points) => subdivisions = points,
                Err(error) => issues.extend_from_slice(error.issues()),
            }
        }

        match domain {
            Some(domain) if issues.is_empty() => Ok((domain, subdivisions)),
            _ => Err(ConfigError::from_issues(issues)),
        }
    }
}

/// A mapped point and weight for one-dimensional fixed quadrature.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    point: f64,
    weight: f64,
}

impl Node {
    /// Returns the evaluation point.
    #[inline]
    pub const fn point(&self) -> f64 {
        self.point
    }

    /// Returns the quadrature weight.
    #[inline]
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

/// Consuming builder for a one-dimensional fixed quadrature.
#[derive(Clone, Debug, PartialEq)]
pub struct Builder {
    domain: Interval,
    rule: GaussRule,
    subdivisions: Vec<f64>,
}

impl Builder {
    /// Replaces the interior subdivision points.
    ///
    /// Empty input means no subdivisions.
    ///
    /// # Errors
    ///
    /// Returns a configuration error unless every point is finite, strictly increasing, unique,
    /// and strictly interior to the integration domain.
    pub fn subdivisions<I>(
        mut self,
        points: I,
    ) -> Result<Self, crate::ConfigError>
    where
        I: IntoIterator<Item = f64>,
    {
        self.subdivisions = validate_subdivisions(self.domain, points)?;
        Ok(self)
    }

    /// Maps the reference rule onto the configured domain and subdivisions.
    pub fn build(self) -> FixedQuad {
        let nodes = mapped_nodes(self.domain, &self.rule, &self.subdivisions);
        FixedQuad {
            domain: self.domain,
            rule: self.rule,
            subdivisions: self.subdivisions,
            nodes,
        }
    }
}

/// A reusable one-dimensional fixed quadrature.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedQuad {
    domain: Interval,
    rule: GaussRule,
    subdivisions: Vec<f64>,
    nodes: Box<[Node]>,
}

impl FixedQuad {
    /// Starts a consuming builder for `domain` and `rule`.
    pub fn builder(
        domain: Interval,
        rule: GaussRule,
    ) -> Builder {
        Builder {
            domain,
            rule,
            subdivisions: Vec::new(),
        }
    }

    /// Builds a reusable fixed quadrature from the legacy options structure.
    pub fn new(opts: FixedQuadOpts) -> Result<Self, OptionsError> {
        let (domain, subdivisions) = opts.validate()?;
        let degree = PolynomialDegree::new_unchecked(opts.order);
        let rule = match opts.gauss_type {
            GaussFamily::Legendre => legendre_rules()?.get_for_degree(degree)?.clone(),
            GaussFamily::Lobatto => lobatto_rules()?.get_for_degree(degree)?.clone(),
        };

        Ok(Self::builder(domain, rule)
            .subdivisions(subdivisions)?
            .build())
    }

    /// Returns the configured integration domain.
    pub const fn domain(&self) -> Interval {
        self.domain
    }

    /// Returns the reference Gaussian rule.
    pub const fn rule(&self) -> &GaussRule {
        &self.rule
    }

    /// Returns the validated interior subdivision points.
    pub fn subdivision_points(&self) -> &[f64] {
        &self.subdivisions
    }

    /// Returns all mapped nodes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Iterates over all mapped nodes.
    pub fn iter(&self) -> std::slice::Iter<'_, Node> {
        self.nodes.iter()
    }

    /// Returns the total number of mapped points.
    pub fn point_count(&self) -> usize {
        self.nodes.len()
    }

    /// Integrates over the configured domain.
    ///
    /// # Errors
    ///
    /// Returns [`IntegrationError::NonFiniteIntegrand`] if `f` returns NaN or infinity.
    pub fn integrate<F>(
        &self,
        mut f: F,
    ) -> Result<f64, IntegrationError>
    where
        F: FnMut(f64) -> f64,
    {
        integrate_nodes(&self.nodes, &mut f)
    }

    /// Integrates over another interval by proportionally remapping every node and subdivision.
    ///
    /// # Errors
    ///
    /// Returns [`IntegrationError::NonFiniteIntegrand`] if `f` returns NaN or infinity.
    pub fn integrate_over<F>(
        &self,
        domain: Interval,
        mut f: F,
    ) -> Result<f64, IntegrationError>
    where
        F: FnMut(f64) -> f64,
    {
        let scale = domain.length() / self.domain.length();
        let mut integral = 0.0;
        for node in &self.nodes {
            let point = domain.lower() + scale * (node.point - self.domain.lower());
            let value = f(point);
            if !value.is_finite() {
                return Err(IntegrationError::NonFiniteIntegrand {
                    point: EvaluationPoint::OneDimensional(point),
                    value,
                });
            }
            integral += scale * node.weight * value;
        }
        Ok(integral)
    }
}

impl<'a> IntoIterator for &'a FixedQuad {
    type Item = &'a Node;
    type IntoIter = std::slice::Iter<'a, Node>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

fn integrate_nodes<F>(
    nodes: &[Node],
    f: &mut F,
) -> Result<f64, IntegrationError>
where
    F: FnMut(f64) -> f64,
{
    let mut integral = 0.0;
    for node in nodes {
        let value = f(node.point);
        if !value.is_finite() {
            return Err(IntegrationError::NonFiniteIntegrand {
                point: EvaluationPoint::OneDimensional(node.point),
                value,
            });
        }
        integral += node.weight * value;
    }
    Ok(integral)
}

pub(super) fn mapped_nodes(
    domain: Interval,
    rule: &GaussRule,
    subdivisions: &[f64],
) -> Box<[Node]> {
    let point_count = rule.point_count().value();
    let mut nodes = Vec::with_capacity(point_count * (subdivisions.len() + 1));
    let (reference_lower, reference_upper) = rule.family().range();
    let reference_length = reference_upper - reference_lower;
    let mut lower = domain.lower();

    for upper in subdivisions
        .iter()
        .copied()
        .chain(std::iter::once(domain.upper()))
    {
        let scale = (upper - lower) / reference_length;
        for (&point, &weight) in rule.points().iter().zip(rule.weights()) {
            nodes.push(Node {
                point: lower + scale * (point - reference_lower),
                weight: scale * weight,
            });
        }
        lower = upper;
    }

    nodes.into_boxed_slice()
}

/// Integrates `f` once using the legacy options structure.
pub fn fixed_quad<F>(
    mut f: F,
    opts: FixedQuadOpts,
) -> Result<f64, OptionsError>
where
    F: FnMut(f64) -> f64,
{
    Ok(FixedQuad::new(opts)?.integrate(&mut f)?)
}

impl From<GaussRule> for FixedQuad {
    fn from(rule: GaussRule) -> Self {
        let (lower, upper) = rule.family().range();
        Self::builder(Interval::new_unchecked(lower, upper), rule).build()
    }
}
