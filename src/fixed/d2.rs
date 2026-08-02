//! Fixed tensor-product quadrature for two-dimensional real-valued functions.

use super::d1;
use crate::config::{validate_subdivisions, ConfigError, Rectangle};
use crate::gauss::GaussRule;
use crate::integration::{EvaluationPoint, IntegrationError};

/// A tensor product of independent Gaussian rules on the `u` and `v` axes.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRule {
    u: GaussRule,
    v: GaussRule,
}

impl TensorRule {
    /// Constructs a tensor-product rule.
    pub const fn new(
        u: GaussRule,
        v: GaussRule,
    ) -> Self {
        Self { u, v }
    }

    /// Returns the `u`-axis Gaussian rule.
    pub const fn u(&self) -> &GaussRule {
        &self.u
    }

    /// Returns the `v`-axis Gaussian rule.
    pub const fn v(&self) -> &GaussRule {
        &self.v
    }
}

/// A mapped point and weight for two-dimensional fixed quadrature.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    u: f64,
    v: f64,
    weight: f64,
}

impl Node {
    /// Returns the `u` coordinate.
    #[inline]
    pub const fn u(&self) -> f64 {
        self.u
    }

    /// Returns the `v` coordinate.
    #[inline]
    pub const fn v(&self) -> f64 {
        self.v
    }

    /// Returns the tensor-product weight.
    #[inline]
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

/// Consuming builder for a two-dimensional fixed quadrature.
#[derive(Clone, Debug, PartialEq)]
pub struct Builder {
    domain: Rectangle,
    rule: TensorRule,
    u_subdivisions: Vec<f64>,
    v_subdivisions: Vec<f64>,
}

impl Builder {
    /// Replaces the interior subdivision coordinates on both axes.
    ///
    /// Empty input on either axis means no subdivisions on that axis.
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

    /// Maps the tensor rule onto the configured rectangle and subdivisions.
    pub fn build(self) -> Quadrature {
        let u_nodes = d1::mapped_nodes(self.domain.u(), self.rule.u(), &self.u_subdivisions);
        let v_nodes = d1::mapped_nodes(self.domain.v(), self.rule.v(), &self.v_subdivisions);
        let mut nodes = Vec::with_capacity(u_nodes.len() * v_nodes.len());

        for u in &u_nodes {
            for v in &v_nodes {
                nodes.push(Node {
                    u: u.point(),
                    v: v.point(),
                    weight: u.weight() * v.weight(),
                });
            }
        }

        Quadrature {
            domain: self.domain,
            rule: self.rule,
            u_subdivisions: self.u_subdivisions,
            v_subdivisions: self.v_subdivisions,
            nodes: nodes.into_boxed_slice(),
        }
    }
}

/// A reusable two-dimensional fixed tensor-product quadrature.
#[derive(Clone, Debug, PartialEq)]
pub struct Quadrature {
    domain: Rectangle,
    rule: TensorRule,
    u_subdivisions: Vec<f64>,
    v_subdivisions: Vec<f64>,
    nodes: Box<[Node]>,
}

impl Quadrature {
    /// Starts a consuming builder for `domain` and `rule`.
    pub fn builder(
        domain: Rectangle,
        rule: TensorRule,
    ) -> Builder {
        Builder {
            domain,
            rule,
            u_subdivisions: Vec::new(),
            v_subdivisions: Vec::new(),
        }
    }

    /// Returns the configured rectangular domain.
    pub const fn domain(&self) -> Rectangle {
        self.domain
    }

    /// Returns the tensor-product reference rule.
    pub const fn rule(&self) -> &TensorRule {
        &self.rule
    }

    /// Returns the validated `u`-axis subdivision coordinates.
    pub fn u_subdivision_points(&self) -> &[f64] {
        &self.u_subdivisions
    }

    /// Returns the validated `v`-axis subdivision coordinates.
    pub fn v_subdivision_points(&self) -> &[f64] {
        &self.v_subdivisions
    }

    /// Returns all mapped tensor-product nodes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Iterates over all mapped tensor-product nodes.
    pub fn iter(&self) -> std::slice::Iter<'_, Node> {
        self.nodes.iter()
    }

    /// Returns the total number of mapped tensor-product points.
    pub fn point_count(&self) -> usize {
        self.nodes.len()
    }

    /// Integrates over the configured rectangle.
    ///
    /// # Errors
    ///
    /// Returns [`IntegrationError::NonFiniteIntegrand`] if `f` returns NaN or infinity.
    pub fn integrate<F>(
        &self,
        mut f: F,
    ) -> Result<f64, IntegrationError>
    where
        F: FnMut(f64, f64) -> f64,
    {
        let mut integral = 0.0;
        for node in &self.nodes {
            let value = f(node.u, node.v);
            if !value.is_finite() {
                return Err(IntegrationError::NonFiniteIntegrand {
                    point: EvaluationPoint::TwoDimensional {
                        u: node.u,
                        v: node.v,
                    },
                    value,
                });
            }
            integral += node.weight * value;
        }
        Ok(integral)
    }

    /// Integrates over another rectangle by proportionally remapping every node and subdivision.
    ///
    /// # Errors
    ///
    /// Returns [`IntegrationError::NonFiniteIntegrand`] if `f` returns NaN or infinity.
    pub fn integrate_over<F>(
        &self,
        domain: Rectangle,
        mut f: F,
    ) -> Result<f64, IntegrationError>
    where
        F: FnMut(f64, f64) -> f64,
    {
        let u_scale = domain.u().length() / self.domain.u().length();
        let v_scale = domain.v().length() / self.domain.v().length();
        let mut integral = 0.0;

        for node in &self.nodes {
            let u = domain.u().lower() + u_scale * (node.u - self.domain.u().lower());
            let v = domain.v().lower() + v_scale * (node.v - self.domain.v().lower());
            let value = f(u, v);
            if !value.is_finite() {
                return Err(IntegrationError::NonFiniteIntegrand {
                    point: EvaluationPoint::TwoDimensional { u, v },
                    value,
                });
            }
            integral += u_scale * v_scale * node.weight * value;
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

/// Builds a two-dimensional fixed quadrature and integrates `f` once.
///
/// # Errors
///
/// Returns [`IntegrationError::NonFiniteIntegrand`] if `f` returns NaN or infinity.
pub fn fixed_quad<F>(
    f: F,
    builder: Builder,
) -> Result<f64, IntegrationError>
where
    F: FnMut(f64, f64) -> f64,
{
    builder.build().integrate(f)
}
