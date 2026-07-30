//! Gauss quadrature rules for one-dimensional real-valued integrands.
//!
//! Gauss quadrature approximates an integral with a weighted sum of function values at points
//! associated with an orthogonal-polynomial family.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
use std::sync::OnceLock;
//}}}
//{{{ dep imports
// use nalgebra as na;
use thiserror::Error;
use topohedral_linalg::{DMatrix, SubViewable};
//}}}
//--------------------------------------------------------------------------------------------------
//{{{ collection: quadrature
//{{{ static: MAX_DEGREE
/// Largest requested polynomial exactness degree included in each cached rule set.
pub(crate) const MAX_DEGREE: usize = 100;
//}}}
//{{{ static: LEGENDRE_RULES
static LEGENDRE_RULES: OnceLock<Result<GaussRuleSet, RuleError>> = OnceLock::new();
//}}}
//{{{ static: LOBATTO_RULES
static LOBATTO_RULES: OnceLock<Result<GaussRuleSet, RuleError>> = OnceLock::new();
//}}}
//{{{ fun: legendre_rules
/// Returns the cached Gauss-Legendre rules through degree 100.
///
/// Repeated calls borrow the same process-wide rule set.
///
/// # Errors
///
/// Returns the deterministic rule-generation error recorded while initializing the cache.
pub fn legendre_rules() -> Result<&'static GaussRuleSet, RuleError> {
    LEGENDRE_RULES
        .get_or_init(|| GaussRuleSet::through_degree(GaussFamily::Legendre, MAX_DEGREE))
        .as_ref()
        .map_err(Clone::clone)
}
//}}}
//{{{ fun: lobatto_rules
/// Returns the cached Gauss-Lobatto rules through degree 100.
///
/// Repeated calls borrow the same process-wide rule set.
///
/// # Errors
///
/// Returns the deterministic rule-generation error recorded while initializing the cache.
pub fn lobatto_rules() -> Result<&'static GaussRuleSet, RuleError> {
    LOBATTO_RULES
        .get_or_init(|| GaussRuleSet::through_degree(GaussFamily::Lobatto, MAX_DEGREE))
        .as_ref()
        .map_err(Clone::clone)
}
//}}}
//}}}
//{{{ enum: GaussFamily
/// A supported Gauss quadrature family.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GaussFamily {
    /// Gauss-Legendre quadrature on `[-1, 1]` with unit weight.
    Legendre,
    /// Gauss-Lobatto quadrature family on `[-1, 1]`.
    ///
    /// Rules retrieved from a [`GaussRuleSet`] include both endpoints.
    Lobatto,
}

impl GaussFamily {
    /// Returns the reference-interval integral of the weight used by this family's node-generating
    /// recurrence.
    ///
    /// For Lobatto rules, this is the integral of the Jacobi `(1, 1)` weight used to generate the
    /// interior nodes, not the sum of the final Lobatto weights.
    pub fn weight_integral(&self) -> f64 {
        match self {
            Self::Legendre => 2.0,
            Self::Lobatto => 4.0 / 3.0,
        }
    }

    /// Returns this family's reference interval.
    pub fn range(&self) -> (f64, f64) {
        match self {
            Self::Legendre => (-1.0, 1.0),
            Self::Lobatto => (-1.0, 1.0),
        }
    }

    /// Returns the number of points needed for a rule with at least `degree` polynomial exactness.
    ///
    /// An `n`-point Legendre rule has exactness `2 * n - 1`, while an `n`-point Lobatto rule has
    /// exactness `2 * n - 3`. Consequently, an even requested degree selects a rule whose actual
    /// exactness is one degree higher.
    fn point_count_for_degree(
        self,
        degree: usize,
    ) -> usize {
        match self {
            Self::Legendre => degree / 2 + 1,
            Self::Lobatto => degree / 2 + 2,
        }
    }

    fn exactness_for_point_count(
        self,
        point_count: usize,
    ) -> Option<usize> {
        match self {
            Self::Legendre if point_count >= 1 => point_count.checked_mul(2)?.checked_sub(1),
            Self::Lobatto if point_count >= 2 => point_count.checked_mul(2)?.checked_sub(3),
            _ => None,
        }
    }

    /// Returns the minimum supported point count for this quadrature family.
    fn minimum_point_count(self) -> usize {
        match self {
            Self::Legendre => 1,
            Self::Lobatto => 2,
        }
    }

    fn maximum_point_count(self) -> usize {
        self.point_count_for_degree(MAX_DEGREE)
    }
}
//}}}
//{{{ enum: RuleError
/// An error encountered while constructing or retrieving a Gaussian rule.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RuleError {
    /// The requested polynomial degree exceeds the supported maximum.
    #[error("polynomial degree {degree} exceeds the supported maximum of {maximum}")]
    DegreeOutOfRange {
        /// Requested polynomial degree.
        degree: usize,
        /// Largest degree available from the constructor or rule set.
        maximum: usize,
    },
    /// The requested point count is outside the supported range.
    #[error(
        "{family:?} point count {point_count} is outside the supported range {minimum}..={maximum}"
    )]
    PointCountOutOfRange {
        /// Quadrature family for which a rule was requested.
        family: GaussFamily,
        /// Requested number of points.
        point_count: usize,
        /// Smallest supported number of points.
        minimum: usize,
        /// Largest supported number of points.
        maximum: usize,
    },
    /// The eigendecomposition backend could not generate a rule.
    #[error("failed to generate a {family:?} rule with {point_count} points: {message}")]
    GenerationFailed {
        /// Quadrature family being generated.
        family: GaussFamily,
        /// Number of points in the requested rule.
        point_count: usize,
        /// Deterministic backend diagnostic retained by the cache.
        message: String,
    },
}
//}}}
//{{{ collection: GaussRuleSet
//{{{ struct: GaussRuleSet
/// A collection of Gauss quadrature rules through a maximum requested exactness degree.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussRuleSet {
    family: GaussFamily,
    maximum_degree: usize,
    rules: Vec<GaussRule>,
}
//}}}
//{{{ impl: GaussRuleSet
impl GaussRuleSet {
    /// Builds every rule for `family` through the requested polynomial exactness `degree`.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::DegreeOutOfRange`] when `degree` exceeds 100, or
    /// [`RuleError::GenerationFailed`] if the eigendecomposition backend fails.
    pub fn through_degree(
        family: GaussFamily,
        degree: usize,
    ) -> Result<Self, RuleError> {
        validate_degree(degree, MAX_DEGREE)?;

        let minimum = family.minimum_point_count();
        let maximum = family.point_count_for_degree(degree);
        let mut rules = Vec::with_capacity(maximum - minimum + 1);
        for point_count in minimum..=maximum {
            rules.push(build_gauss_rule(family, point_count)?);
        }

        Ok(Self {
            family,
            maximum_degree: degree,
            rules,
        })
    }

    /// Returns the smallest stored rule whose exactness is at least `degree`.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::DegreeOutOfRange`] when `degree` exceeds the maximum requested while
    /// constructing this set.
    pub fn get_for_degree(
        &self,
        degree: usize,
    ) -> Result<&GaussRule, RuleError> {
        validate_degree(degree, self.maximum_degree)?;
        self.get_by_point_count(self.family.point_count_for_degree(degree))
    }

    /// Returns the stored rule containing exactly `point_count` points.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::PointCountOutOfRange`] when `point_count` is not represented by this
    /// set.
    pub fn get_by_point_count(
        &self,
        point_count: usize,
    ) -> Result<&GaussRule, RuleError> {
        let minimum = self.family.minimum_point_count();
        let maximum = self.maximum_point_count();
        if !(minimum..=maximum).contains(&point_count) {
            return Err(RuleError::PointCountOutOfRange {
                family: self.family,
                point_count,
                minimum,
                maximum,
            });
        }
        Ok(&self.rules[point_count - minimum])
    }

    /// Returns the quadrature family shared by every rule in this set.
    pub fn family(&self) -> GaussFamily {
        self.family
    }

    /// Returns the largest requested polynomial degree represented by this set.
    pub fn maximum_degree(&self) -> usize {
        self.maximum_degree
    }

    /// Returns the smallest point count represented by this set.
    pub fn minimum_point_count(&self) -> usize {
        self.family.minimum_point_count()
    }

    /// Returns the largest point count represented by this set.
    pub fn maximum_point_count(&self) -> usize {
        self.family.point_count_for_degree(self.maximum_degree)
    }

    /// Returns the number of rules in this set.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Returns `true` when this set contains no rules.
    ///
    /// Generated rule sets are never empty; this method accompanies [`Self::len`] for collection
    /// API consistency.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Iterates over the stored rules in ascending point-count order.
    pub fn iter(&self) -> std::slice::Iter<'_, GaussRule> {
        self.rules.iter()
    }
}
//}}}
//}}}
//{{{ collection: GaussRule
//{{{ struct: GaussRule
/// A specific Gauss quadrature rule, represented by points and associated weights.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussRule {
    family: GaussFamily,
    exactness: usize,
    points: Vec<f64>,
    weights: Vec<f64>,
}
//}}}
//{{{ impl: GaussRule
impl GaussRule {
    fn from_points_weights(
        family: GaussFamily,
        points: Vec<f64>,
        weights: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(points.len(), weights.len());
        let exactness = family
            .exactness_for_point_count(points.len())
            .expect("validated point counts have representable exactness");
        Self {
            family,
            exactness,
            points,
            weights,
        }
    }

    /// Constructs the smallest quadrature rule with at least `degree` polynomial exactness.
    ///
    /// Legendre rules support one point and Lobatto rules support two points at minimum. An even
    /// requested degree selects a rule whose actual exactness is one degree higher.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::DegreeOutOfRange`] when `degree` exceeds 100, or
    /// [`RuleError::GenerationFailed`] if the eigendecomposition backend fails.
    pub fn for_degree(
        family: GaussFamily,
        degree: usize,
    ) -> Result<Self, RuleError> {
        validate_degree(degree, MAX_DEGREE)?;
        build_gauss_rule(family, family.point_count_for_degree(degree))
    }

    /// Constructs a quadrature rule containing exactly `point_count` points.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::PointCountOutOfRange`] when the count is outside the supported range,
    /// or [`RuleError::GenerationFailed`] if the eigendecomposition backend fails.
    pub fn with_point_count(
        family: GaussFamily,
        point_count: usize,
    ) -> Result<Self, RuleError> {
        validate_point_count(family, point_count)?;
        build_gauss_rule(family, point_count)
    }

    /// Returns the quadrature family used to construct this rule.
    pub fn family(&self) -> GaussFamily {
        self.family
    }

    /// Returns the rule's actual polynomial exactness.
    pub fn exactness(&self) -> usize {
        self.exactness
    }

    /// Returns the number of quadrature points and weights.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Returns the quadrature points in ascending order.
    pub fn points(&self) -> &[f64] {
        &self.points
    }

    /// Returns the weights corresponding to [`Self::points`].
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}
//}}}
//}}}
//{{{ fun: validate_degree
fn validate_degree(
    degree: usize,
    maximum: usize,
) -> Result<(), RuleError> {
    if degree > maximum {
        Err(RuleError::DegreeOutOfRange { degree, maximum })
    } else {
        Ok(())
    }
}
//}}}
//{{{ fun: validate_point_count
fn validate_point_count(
    family: GaussFamily,
    point_count: usize,
) -> Result<(), RuleError> {
    let minimum = family.minimum_point_count();
    let maximum = family.maximum_point_count();
    if !(minimum..=maximum).contains(&point_count) {
        Err(RuleError::PointCountOutOfRange {
            family,
            point_count,
            minimum,
            maximum,
        })
    } else {
        Ok(())
    }
}
//}}}
//{{{ fun: build_gauss_rule
/// Builds a quadrature rule with exactly `point_count` points.
fn build_gauss_rule(
    family: GaussFamily,
    point_count: usize,
) -> Result<GaussRule, RuleError> {
    validate_point_count(family, point_count)?;

    let (points, weights) = match family {
        GaussFamily::Legendre => golub_welsch(
            family,
            point_count,
            point_count,
            family.weight_integral(),
            legendre_recursion_coeffs,
        )?,
        GaussFamily::Lobatto => {
            let endpoint_weight = 2.0 / ((point_count as f64) * ((point_count - 1) as f64));
            let mut points = Vec::with_capacity(point_count);
            let mut weights = Vec::with_capacity(point_count);

            points.push(-1.0);
            weights.push(endpoint_weight);

            if point_count > 2 {
                let interior_count = point_count - 2;
                let (interior_points, _) = golub_welsch(
                    family,
                    point_count,
                    interior_count,
                    family.weight_integral(),
                    lobatto_recursion_coeffs,
                )?;

                for point in interior_points {
                    let polynomial = legendre(point_count - 1, point);
                    points.push(point);
                    weights.push(endpoint_weight / polynomial.powi(2));
                }
            }

            points.push(1.0);
            weights.push(endpoint_weight);
            (points, weights)
        }
    };

    Ok(GaussRule::from_points_weights(family, points, weights))
}
//}}}
//{{{ fun: golub_welsch
/// Computes the Golub-Welsch algorithm to generate Gauss quadrature points and weights for
/// numerical integration.
///
/// # Arguments
///
/// * `family` - type of quadrature rule
/// * `rule_point_count` - number of points in the final quadrature rule
/// * `matrix_size` - number of points generated by this eigendecomposition
/// * `weight_integral` - integral of the recurrence's weight function
/// * `recurrence_fcn` - recurrence function which provided the recurrence coefficients for the
///                      orthogonal polynomial family
///
/// # Returns
/// A tuple of two vectors, the first being the quadrature points and the second being the quadrature
/// weights.
///
/// # Theory
///
/// Starting from the 3-term orthogonal polynomial recurrence relation:
/// \\[
///     p_{i+1}(x) = (a_{i}x + b_{i})p_{i}(x) - c_{i}p_{i-1}
/// \\]
/// And rearranging to give:
/// \\[
///     xp_{i}(x) =
///         -\frac{c_{i}}{a_{i}}p_{i-1}(x) + \frac{b_{i}}{a_{i}}p_{i}(x) + \frac{1}{a_{i}}p_{i+1}(x)
/// \\]
/// Which may in turn be represented with the following system of equations
/// \\[
///     x
///     \begin{bmatrix}
///         p_{0}(x) \\\\ p_{1}(x) \\\\ \vdots \\\\ p_{n-2} \\\\ p_{n-1}(x)
///     \end{bmatrix}
///     =
///     \begin{bmatrix}
///         -b_{1}/a_{1} & 1 / a_{1} & 0 & ... & 0 \\\\
///         c_{2}/a_{2}  & -b_{2} / a_{2} & 1 / a_{2}   & ... & 0 \\\\
///         \vdots    &    \ddots      &    \ddots & \ddots  & \vdots \\\\
///         0  & ... & c_{n-1}/a_{n-1}  & -b_{n-1} / a_{n-1} & 1 / a_{n-1} \\\\
///         0    &   ... &   0     & c_{n}/a_{n} & -b_{n} \ a_{n}
///     \end{bmatrix}
///     \begin{bmatrix}
///         p_{0}(x) \\\\ p_{1}(x) \\\\ \vdots \\\\ p_{n-2} \\\\ p_{n-1}(x)
///     \end{bmatrix}
///     +
///     \begin{bmatrix}
///         0 \\\\ 0 \\\\ \vdots \\\\ 0 \\\\ p_{n}(x) / a_{n}
///     \end{bmatrix}
/// \\]
/// Which may be written as:
/// \\[
///     x\mathbf{p}(x) = \mathbf{T}\mathbf{p}(x) + \frac{p_{n}(x)}{a_{n}}\mathbf{e_{n}}
/// \\]
/// For each of the roots of orthogonal polynomials, which are the quadrature
/// points of an n-point rule, the following eigenvalue problem is
/// created:
/// \\[
///   t_{j}\mathbf{p}(t_{j}) = \mathbf{T}\mathbf{p}(t_{j})
/// \\]
/// The above system is then converted to the following symmetric
/// system by diagonal similarity transform which preserves the eigenvalues
/// \\[
///   t_{j}\mathbf{q}(t_{j}) = \mathbf{J} \mathbf{q}(t_{j})
/// \\]
///
/// Where:
///
/// \\[
///     \mathbf{J} =
///         \begin{bmatrix}
///             \alpha_{1} & \beta_{1} & 0 & ... & 0 \\\\
///              \beta_{1} & \alpha_{2} & \beta_{2} & ... & 0 \\\\
///              \vdots & \ddots & \ddots  & \ddots & \vdots \\\\
///              0 & ... & \beta_{n-2} & \alpha_{n-1} & \beta_{n-1} \\\\
///              0 & 0 & ... & \beta_{n-1} & \alpha_{n}
///         \end{bmatrix}
/// \\]
///
/// where:
///
/// \\[
///   \alpha_{i} = -\frac{b_{i}}{a_{i}}, \quad
///   \beta_{i} = \left( \frac{c_{i+1}}{a_{i}a_{i+1}}\right)^{1/2}
/// \\]
#[allow(clippy::doc_overindented_list_items)]
fn golub_welsch<F: Fn(usize) -> (f64, f64, f64)>(
    family: GaussFamily,
    rule_point_count: usize,
    matrix_size: usize,
    weight_integral: f64,
    recurrence_fcn: F,
) -> Result<(Vec<f64>, Vec<f64>), RuleError> {
    debug_assert!(matrix_size > 0);

    let mut tmat = DMatrix::<f64>::zeros(matrix_size, matrix_size);

    for i in 0..matrix_size {
        let (ai, bi, _) = recurrence_fcn(i);
        tmat[(i, i)] = -(bi / ai);

        if i + 1 < matrix_size {
            let (aj, _, cj) = recurrence_fcn(i + 1);
            let beta = (cj / (ai * aj)).sqrt();
            tmat[(i, i + 1)] = beta;
            tmat[(i + 1, i)] = beta;
        }
    }

    //{{{ com: eigendecompose
    let eigen_decomp = tmat.symeig().map_err(|error| RuleError::GenerationFailed {
        family,
        point_count: rule_point_count,
        message: error.to_string(),
    })?;
    //}}}
    //{{{ com: compute quadrature points and weights from eigenvalues and eigenvectors
    let qpoints: &Vec<f64> = &eigen_decomp.eigvals;
    let qweights: Vec<f64> = eigen_decomp
        .eigvecs
        .row(0)
        .iter()
        .map(|x| x.powi(2) * weight_integral)
        .collect();
    let mut combined: Vec<(f64, f64)> = qpoints
        .iter()
        .cloned()
        .zip(qweights.iter().cloned())
        .collect();
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (qpoints_final, qweights_final): (Vec<f64>, Vec<f64>) = combined.iter().cloned().unzip();
    //}}}
    //{{{ ret
    Ok((qpoints_final, qweights_final))
    //}}}
}
//..................................................................................................
//}}}
//{{{ fun: legendre_recursion_coeffs
/// Computes the recurrence coefficients for the nth Legendre polynomial using the recurrence relation.
///
/// Returns a tuple containing the coefficients (a, b, c) for the i'th polynomial.
fn legendre_recursion_coeffs(i: usize) -> (f64, f64, f64) {
    let i_f64 = i as f64;
    let ai = (2.0 * i_f64 + 1.0) / (i_f64 + 1.0);
    let bi = 0.0;
    let ci = i_f64 / (i_f64 + 1.0);
    (ai, bi, ci)
}
//}}}
//{{{ fun: lobatto_recursion_coeffs
/// Computes the recurrence coefficients `ai`, `bi`, and `ci` for
/// the Lobatto quadrature rule at index `i`.
///
/// Returns a tuple containing the coefficients (a, b, c) for the i'th polynomial.
fn lobatto_recursion_coeffs(i: usize) -> (f64, f64, f64) {
    let i_f64 = i as f64;
    let ai = ((2.0 * i_f64 + 3.0) * (i_f64 + 2.0)) / ((i_f64 + 1.0) * (i_f64 + 3.0));
    let bi = 0.0;
    let ci = ((i_f64 + 1.0) * (i_f64 + 2.0)) / ((i_f64 + 3.0) * (i_f64 + 1.0));
    (ai, bi, ci)
}
//}}}
//{{{ fun: legendre
fn legendre(
    n: usize,
    x: f64,
) -> f64 {
    debug_assert!((-1.0..=1.0).contains(&x));

    let (mut leg_n, mut leg_1, mut leg_2) = (1.0f64, 1.0f64, 0.0f64);
    for i in 0..n {
        let ii = i as f64;
        let ai = (2.0 * ii + 1.0) / (ii + 1.0);
        let ci = ii / (ii + 1.0);
        leg_n = ai * x * leg_1 - ci * leg_2;
        leg_2 = leg_1;
        leg_1 = leg_n;
    }
    leg_n
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {}
//}}}
