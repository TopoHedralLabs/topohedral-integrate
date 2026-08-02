//! Fixed quadrature for one-dimensional real-valued functions.

use crate::config::{validate_subdivisions, Interval};
use crate::gauss::GaussRule;
use crate::integration::{EvaluationPoint, IntegrationError};

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
    pub fn build(self) -> Quadrature {
        let nodes = mapped_nodes(self.domain, &self.rule, &self.subdivisions);
        Quadrature {
            domain: self.domain,
            rule: self.rule,
            subdivisions: self.subdivisions,
            nodes,
        }
    }
}

/// A reusable one-dimensional fixed quadrature.
#[derive(Clone, Debug, PartialEq)]
pub struct Quadrature {
    domain: Interval,
    rule: GaussRule,
    subdivisions: Vec<f64>,
    nodes: Box<[Node]>,
}

impl Quadrature {
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

impl<'a> IntoIterator for &'a Quadrature {
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

/// Builds a one-dimensional fixed quadrature and integrates `f` once.
///
/// # Errors
///
/// Returns [`IntegrationError::NonFiniteIntegrand`] if `f` returns NaN or infinity.
pub fn fixed_quad<F>(
    f: F,
    builder: Builder,
) -> Result<f64, IntegrationError>
where
    F: FnMut(f64) -> f64,
{
    builder.build().integrate(f)
}

impl From<GaussRule> for Quadrature {
    fn from(rule: GaussRule) -> Self {
        let (lower, upper) = rule.family().range();
        Self::builder(Interval::new_unchecked(lower, upper), rule).build()
    }
}
