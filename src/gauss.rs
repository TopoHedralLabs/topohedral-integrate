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
use topohedral_linalg::{DMatrix, SubViewable};
//}}}
//--------------------------------------------------------------------------------------------------
//{{{ collection: quadrature
//{{{ static: MAX_ORDER
/// Largest requested polynomial exactness degree included in each cached rule set.
pub static MAX_ORDER: usize = 100;
//}}}
//{{{ static: LEGENDRE_POINTS
static LEGENDRE_POINTS: OnceLock<GuassQuadSet> = OnceLock::new();
//}}}
//{{{ static: LOBATTO_POINTS
static LOBATTO_POINTS: OnceLock<GuassQuadSet> = OnceLock::new();
//}}}
//{{{ fun: get_legendre_points
/// Returns the lazily initialized Gauss-Legendre rules through the configured maximum exactness.
pub fn get_legendre_points() -> &'static GuassQuadSet {
    LEGENDRE_POINTS.get_or_init(|| GuassQuadSet::new(GaussQuadType::Legendre, MAX_ORDER))
}
//}}}
//{{{ fun: get_lobatto_points
/// Returns the lazily initialized Gauss-Lobatto rules through the configured maximum exactness.
pub fn get_lobatto_points() -> &'static GuassQuadSet {
    LOBATTO_POINTS.get_or_init(|| GuassQuadSet::new(GaussQuadType::Lobatto, MAX_ORDER))
}
//}}}
//}}}
//{{{ enum: GaussQuadType
/// A supported Gauss quadrature family.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GaussQuadType {
    /// Gauss-Legendre quadrature on `[-1, 1]` with unit weight.
    Legendre,
    /// Gauss-Lobatto quadrature family on `[-1, 1]`.
    ///
    /// Rules retrieved from a [`GuassQuadSet`] include both endpoints.
    Lobatto,
}

impl GaussQuadType {
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

    /// Returns the number of points needed for a rule with at least `order` polynomial exactness.
    ///
    /// An `n`-point Legendre rule has exactness `2 * n - 1`, while an `n`-point Lobatto rule has
    /// exactness `2 * n - 3`. Consequently, an even requested order selects a rule whose actual
    /// exactness is one degree higher.
    pub fn nqp_from_order(
        &self,
        order: usize,
    ) -> usize {
        match self {
            Self::Legendre => order / 2 + 1,
            Self::Lobatto => order / 2 + 2,
        }
    }

    /// Returns the polynomial exactness of an `nqp`-point rule.
    ///
    /// # Panics
    ///
    /// Panics if `nqp` is less than one for Legendre or less than two for Lobatto, or if the
    /// resulting exactness cannot be represented by `usize`.
    pub fn order_from_nqp(
        &self,
        nqp: usize,
    ) -> usize {
        match self {
            Self::Legendre => {
                assert!(nqp >= 1, "Legendre rules require at least one point");
                nqp.checked_mul(2)
                    .and_then(|value| value.checked_sub(1))
                    .expect("Legendre exactness must fit in usize")
            }
            Self::Lobatto => {
                assert!(nqp >= 2, "Lobatto rules require at least two points");
                nqp.checked_mul(2)
                    .and_then(|value| value.checked_sub(3))
                    .expect("Lobatto exactness must fit in usize")
            }
        }
    }

    /// Returns the minimum supported point count for this quadrature family.
    fn min_nqp(&self) -> usize {
        match self {
            Self::Legendre => 1,
            Self::Lobatto => 2,
        }
    }
}
//}}}
//{{{ collection: GuassQuadSet
//{{{ struct: GuassQuadSet
/// A collection of Gauss quadrature rules through a maximum requested exactness degree.
///
/// The misspelling in this type's name is part of the established public API.
pub struct GuassQuadSet {
    /// Orthogonal-polynomial family used by every rule in the set.
    pub gauss_type: GaussQuadType,
    /// Largest requested polynomial exactness represented by the set.
    pub max_order: usize,
    /// Smallest number of points in a stored rule.
    pub min_nqp: usize,
    /// Largest number of points in a stored rule.
    pub max_nqp: usize,
    /// Quadrature points, indexed by `nqp - min_nqp`.
    pub points: Vec<Vec<f64>>,
    /// Quadrature weights, indexed by `nqp - min_nqp`.
    pub weights: Vec<Vec<f64>>,
}
//}}}
//{{{ impl: GuassQuadSet
impl GuassQuadSet {
    /// Builds all rules for `gauss_type` through the requested polynomial exactness `order`.
    pub fn new(
        gauss_type: GaussQuadType,
        order: usize,
    ) -> Self {
        let min_nqp = gauss_type.min_nqp();
        let max_nqp = gauss_type.nqp_from_order(order);
        let num_rules = max_nqp - min_nqp + 1;
        let mut points = Vec::with_capacity(num_rules);
        let mut weights = Vec::with_capacity(num_rules);

        for nqp in min_nqp..=max_nqp {
            let rule = build_gauss_quad(gauss_type, nqp);
            points.push(rule.points);
            weights.push(rule.weights);
        }

        Self {
            gauss_type,
            max_order: order,
            min_nqp,
            max_nqp,
            points,
            weights,
        }
    }

    /// Returns a clone of the rule containing `nqp` points.
    ///
    /// # Panics
    ///
    /// Panics if `nqp` is outside `min_nqp..=max_nqp`.
    pub fn gauss_quad_from_nqp(
        &self,
        nqp: usize,
    ) -> GaussQuad {
        assert!(
            nqp >= self.min_nqp && nqp <= self.max_nqp,
            "point count must be within the stored rule range"
        );
        let points_nqp = self.points[nqp - self.min_nqp].clone();
        let weights_nqp = self.weights[nqp - self.min_nqp].clone();
        GaussQuad::from_points_weights(self.gauss_type, points_nqp, weights_nqp)
    }

    /// Returns the smallest stored rule whose exactness is at least `order`.
    ///
    /// # Panics
    ///
    /// Panics if `order` cannot be represented by a rule in this set.
    pub fn gauss_quad_from_order(
        &self,
        order: usize,
    ) -> GaussQuad {
        assert!(
            order <= self.max_order,
            "requested exactness must not exceed the stored maximum"
        );
        let nqp = self.gauss_type.nqp_from_order(order);
        self.gauss_quad_from_nqp(nqp)
    }
}
//}}}
//}}}
//{{{ collection: GaussQuad
//{{{ struct: GaussQuad
/// A specific Gauss quadrature rule, represented by points and associated weights.
#[derive(Debug, Clone)]
pub struct GaussQuad {
    /// Orthogonal-polynomial family used to construct the rule.
    pub gauss_type: GaussQuadType,
    /// Number of quadrature points and weights.
    pub nqp: usize,
    /// Quadrature points in ascending order.
    pub points: Vec<f64>,
    /// Weight associated with each entry in [`Self::points`].
    pub weights: Vec<f64>,
}
//}}}
//{{{ impl: GaussQuad
impl GaussQuad {
    fn from_points_weights(
        gauss_type: GaussQuadType,
        points: Vec<f64>,
        weights: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(points.len(), weights.len());
        let nqp = points.len();
        Self {
            gauss_type,
            nqp,
            points,
            weights,
        }
    }

    /// Constructs the smallest quadrature rule with at least `order` polynomial exactness.
    ///
    /// Legendre rules support one point and Lobatto rules support two points at minimum. An even
    /// requested order selects a rule whose actual exactness is one degree higher.
    pub fn new(
        gauss_type: GaussQuadType,
        order: usize,
    ) -> Self {
        let nqp = gauss_type.nqp_from_order(order);
        build_gauss_quad(gauss_type, nqp)
    }
}
//}}}
//}}}
//{{{ fun: build_gauss_quad
/// Builds a quadrature rule with exactly `nqp` points.
fn build_gauss_quad(
    gauss_type: GaussQuadType,
    nqp: usize,
) -> GaussQuad {
    assert!(
        nqp >= gauss_type.min_nqp(),
        "point count is below the minimum for the quadrature family"
    );

    let (points, weights) = match gauss_type {
        GaussQuadType::Legendre => {
            golub_welsch(nqp, gauss_type.weight_integral(), legendre_recursion_coeffs)
        }
        GaussQuadType::Lobatto => {
            let endpoint_weight = 2.0 / ((nqp as f64) * ((nqp - 1) as f64));
            let mut points = Vec::with_capacity(nqp);
            let mut weights = Vec::with_capacity(nqp);

            points.push(-1.0);
            weights.push(endpoint_weight);

            if nqp > 2 {
                let interior_count = nqp - 2;
                let (interior_points, _) = golub_welsch(
                    interior_count,
                    gauss_type.weight_integral(),
                    lobatto_recursion_coeffs,
                );

                for point in interior_points {
                    let polynomial = legendre(nqp - 1, point);
                    points.push(point);
                    weights.push(endpoint_weight / polynomial.powi(2));
                }
            }

            points.push(1.0);
            weights.push(endpoint_weight);
            (points, weights)
        }
    };

    GaussQuad::from_points_weights(gauss_type, points, weights)
}
//}}}
//{{{ fun: golub_welsch
/// Computes the Golub-Welsch algorithm to generate Gauss quadrature points and weights for
/// numerical integration.
///
/// # Arguments
///
/// * `nqp` - number of quadrature points
/// * `gauss_type` - type of quadrature rule
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
    nqp: usize,
    weight_integral: f64,
    recurrence_fcn: F,
) -> (Vec<f64>, Vec<f64>) {
    assert!(nqp > 0, "Golub-Welsch requires at least one point");

    let mut tmat = DMatrix::<f64>::zeros(nqp, nqp);

    for i in 0..nqp {
        let (ai, bi, _) = recurrence_fcn(i);
        tmat[(i, i)] = -(bi / ai);

        if i + 1 < nqp {
            let (aj, _, cj) = recurrence_fcn(i + 1);
            let beta = (cj / (ai * aj)).sqrt();
            tmat[(i, i + 1)] = beta;
            tmat[(i + 1, i)] = beta;
        }
    }

    //{{{ com: eigendecompose
    let eigen_decomp = tmat.symeig().unwrap();
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
    (qpoints_final, qweights_final)
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
