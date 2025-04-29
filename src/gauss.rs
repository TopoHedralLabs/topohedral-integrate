//! This module contains the implementation of Gauss quadrature rules.
//!
//! It is limited to 1D and to the standard real-numbers
//!
//! Gauss quadrature is a numerical integration method that approximates the integral of a function
//! using a weighted sum of function values at specific points. These points are the roots of a
//! family of orthogonal polynomials.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
use std::sync::OnceLock;
//}}}
//{{{ dep imports
use nalgebra as na;
//}}}
//--------------------------------------------------------------------------------------------------
//{{{ collection: quadrature
//{{{ static: MAX_ORDER
pub static MAX_ORDER: usize = 100;
//}}}
//{{{ static: LEGENDRE_POINTS
static LEGENDRE_POINTS: OnceLock<GuassQuadSet> = OnceLock::new();
//}}}
//{{{ static: LOBATTO_POINTS
static LOBATTO_POINTS: OnceLock<GuassQuadSet> = OnceLock::new();
//}}}
//{{{ fun: get_legendre_points
pub fn get_legendre_points() -> &'static GuassQuadSet {
    LEGENDRE_POINTS.get_or_init(|| GuassQuadSet::new(GaussQuadType::Legendre, MAX_ORDER))
}
//}}}
//{{{ fun: get_lobatto_points
pub fn get_lobatto_points() -> &'static GuassQuadSet {
    LOBATTO_POINTS.get_or_init(|| GuassQuadSet::new(GaussQuadType::Lobatto, MAX_ORDER))
}
//}}}
//}}}
//{{{ enum: GaussQuadType
/// An enumeration of supported Gauss quadrature rules. Each member corresponds to the orthogonal
/// polynomial family used for the rule.
///
/// `Legendre` specifies Gauss-Legendre quadrature.
/// `Lobatto` specifies Gauss-Lobatto quadrature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GaussQuadType {
    Legendre,
    Lobatto,
}

impl GaussQuadType {
    /// Returns the integral over the reference domain of the weight function assocaited with the
    /// orthogonal polynomail family.
    pub fn weight_integral(&self) -> f64 {
        match self {
            Self::Legendre => 2.0,
            Self::Lobatto => 1.33333333333333,
        }
    }

    /// Returns the range over which the polynomial family is defined.
    pub fn range(&self) -> (f64, f64) {
        match self {
            Self::Legendre => (-1.0, 1.0),
            Self::Lobatto => (-1.0, 1.0),
        }
    }

    pub fn nqp_from_order(&self, order: usize) -> usize {
        match self {
            Self::Legendre => order.div_ceil(2),
            Self::Lobatto => (order + 3) / 2,
        }
    }

    pub fn order_from_nqp(&self, nqp: usize) -> usize {
        match self {
            Self::Legendre => 2 * nqp - 1,
            Self::Lobatto => 2 * nqp - 3,
        }
    }
}
//}}}
//{{{ collection: GuassQuadSet
//{{{ struct: GuassQuadSet
/// Struct to represent a collection of Gauss quadrature rules up to a given order.
pub struct GuassQuadSet {
    /// underlying orthogonal polynomial family
    pub gauss_type: GaussQuadType,
    /// maximum order of the quadrature rule
    pub max_order: usize,
    /// minimum number of quadrature points
    pub min_nqp: usize,
    /// maximum number of quadrature points
    pub max_nqp: usize,
    /// quadrature points, the nqp-point rule is in `points[nqp - nqp_min]`
    pub points: Vec<Vec<f64>>,
    /// quadrature weights, the nqp-point rule is in `wieghts[nqp - nqp_min]`
    pub weights: Vec<Vec<f64>>,
}
//}}}
//{{{ impl: GuassQuadSet
impl GuassQuadSet {
    pub fn new(gauss_type: GaussQuadType, order: usize) -> Self {
        // set the min and max nqp's available for the given quadrature rule
        let min_nqp = 2;
        let max_nqp = match gauss_type {
            GaussQuadType::Legendre => order.div_ceil(2),
            GaussQuadType::Lobatto => (order + 3) / 2,
        };
        // preallocate the points and weights
        let mut points = vec![vec![0.0; max_nqp]; max_nqp - min_nqp + 1];
        let mut weights = vec![vec![0.0; max_nqp]; max_nqp - min_nqp + 1];

        match gauss_type {
            GaussQuadType::Lobatto => {
                points[0] = vec![-1.0f64, 1.0f64];
                weights[0] = vec![1.0f64, 1.0f64];

                points[1] = vec![-1.0f64, 0.0f64, 1.0f64];
                const W1: f64 = 1.0f64 / 3.0f64;
                const W2: f64 = 4.0f64 / 3.0f64;
                weights[1] = vec![W1, W2, W1];

                for i in (min_nqp + 2)..max_nqp {
                    let nqp = i - min_nqp;
                    let (points_i, _weights_i) =
                        golub_welsch(nqp, gauss_type, lobatto_recursion_coeffs);
                    points[i - min_nqp][0] = -1.0f64;
                    points[i - min_nqp][1..nqp + 1].copy_from_slice(&points_i);
                    points[i - min_nqp][nqp + 1] = 1.0f64;

                    let ii = i as f64;
                    let wi = 2.0f64 / (ii * (ii - 1.0f64));
                    weights[i - min_nqp][0] = wi;
                    weights[i - min_nqp][1..nqp + 1]
                        .iter_mut()
                        .enumerate()
                        .for_each(|(j, wj)| {
                            let xj = points_i[j];
                            let leg_j = legendre(i - 1, xj);
                            *wj = wi / leg_j.powi(2);
                        });
                    weights[i - min_nqp][nqp + 1] = wi;
                }
            }
            GaussQuadType::Legendre => {
                for i in min_nqp..max_nqp {
                    let (points_i, weights_i) =
                        golub_welsch(i, gauss_type, legendre_recursion_coeffs);
                    points[i - min_nqp] = points_i;
                    weights[i - min_nqp] = weights_i;
                }
            }
        }

        for i in min_nqp..max_nqp + 1 {
            points[i - min_nqp].resize(i, 0.0);
            weights[i - min_nqp].resize(i, 0.0);
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

    pub fn gauss_quad_from_nqp(&self, nqp: usize) -> GaussQuad {
        debug_assert!(nqp >= self.min_nqp && nqp <= self.max_nqp);
        let points_nqp = self.points[nqp - self.min_nqp].clone();
        let weights_nqp = self.weights[nqp - self.min_nqp].clone();
        GaussQuad::from_points_weights(self.gauss_type, points_nqp, weights_nqp)
    }

    pub fn gauss_quad_from_order(&self, order: usize) -> GaussQuad {
        debug_assert!(order <= self.max_order);
        let nqp = match self.gauss_type {
            GaussQuadType::Legendre => order.div_ceil(2),
            GaussQuadType::Lobatto => (order + 3) / 2,
        };
        self.gauss_quad_from_nqp(nqp)
    }
}
//}}}
//}}}
//{{{ collection: GaussQuad
//{{{ struct: GaussQuad
/// This struct represents a specific quadrature rule, meaning a set of quadrature points and
/// a set of assocciated weights.
#[derive(Debug, Clone)]
pub struct GaussQuad {
    /// underlying orthogonal polynomial family
    pub gauss_type: GaussQuadType,
    /// number of quadrature points
    pub nqp: usize,
    /// quadrature points
    pub points: Vec<f64>,
    /// quadrature weights
    pub weights: Vec<f64>,
}
//}}}
//{{{ impl: GaussQuad
impl GaussQuad {
    fn from_points_weights(gauss_type: GaussQuadType, points: Vec<f64>, weights: Vec<f64>) -> Self {
        debug_assert_eq!(points.len(), weights.len());
        let nqp = points.len();
        Self {
            gauss_type,
            nqp,
            points,
            weights,
        }
    }

    pub fn new(gauss_type: GaussQuadType, order: usize) -> Self {
        let nqp = match gauss_type {
            GaussQuadType::Legendre => order.div_ceil(2),
            GaussQuadType::Lobatto => (order + 3) / 2,
        };

        let (points, weights) = match gauss_type {
            GaussQuadType::Legendre => {
                let (points, weights) = golub_welsch(nqp, gauss_type, legendre_recursion_coeffs);
                (points, weights)
            }
            GaussQuadType::Lobatto => {
                let (points, weights) = golub_welsch(nqp, gauss_type, lobatto_recursion_coeffs);
                (points, weights)
            }
        };

        Self::from_points_weights(gauss_type, points, weights)
    }
}
//}}}
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
    gauss_type: GaussQuadType,
    recurrence_fcn: F,
) -> (Vec<f64>, Vec<f64>) {
    //{{{ init
    let mut tmat = na::DMatrix::<f64>::zeros(nqp, nqp);

    let (mut ai, mut _bi, mut _ci): (f64, f64, f64);
    let (mut aj, mut _bj, mut cj): (f64, f64, f64);
    let (mut alpha_i, alpha_j): (f64, f64);
    let (mut beta_i, mut beta_h): (f64, f64);
    //}}}
    //{{{ com: deal with row 0
    {
        (ai, _bi, _ci) = recurrence_fcn(0);
        (aj, _bj, cj) = recurrence_fcn(1);
        alpha_i = -(_bi / ai);
        beta_i = (cj / (ai * aj)).sqrt();
        tmat[(0, 0)] = alpha_i;
        tmat[(0, 1)] = beta_i;
        beta_h = beta_i;
    }
    //}}}
    //{{{ com: deal with rows 1 to nqp-2
    for i in 1..nqp - 1 {
        (ai, _bi, _ci) = recurrence_fcn(i);
        (aj, _bj, cj) = recurrence_fcn(i + 1);
        alpha_i = -(_bi / ai);
        beta_i = (cj / (ai * aj)).sqrt();

        tmat[(i, i - 1)] = beta_h;
        tmat[(i, i)] = alpha_i;
        tmat[(i, i + 1)] = beta_i;
        beta_h = beta_i;
    }
    //}}}
    //{{{ com: deal with row nqp-1
    {
        (_, _bi, _ci) = recurrence_fcn(nqp - 1);
        (aj, _bj, _) = recurrence_fcn(nqp);
        alpha_j = -(_bj / aj);
        tmat[(nqp - 1, nqp - 2)] = beta_h;
        tmat[(nqp - 1, nqp - 1)] = alpha_j;
    }
    //}}}
    //{{{ com: eigendecompose
    let eigen_decomp = tmat.symmetric_eigen();
    //}}}
    //{{{ com: compute quadrature points and weights from eigenvalues and eigenvectors
    let qpoints: Vec<f64> = eigen_decomp.eigenvalues.iter().copied().collect();
    let mu0 = gauss_type.weight_integral();
    let qweights: Vec<f64> = eigen_decomp
        .eigenvectors
        .row(0)
        .iter()
        .map(|x| x.powi(2) * mu0)
        .collect();
    let mut combined: Vec<(f64, f64)> = qpoints
        .iter()
        .cloned()
        .zip(qweights.iter().cloned())
        .collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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
fn legendre(n: usize, x: f64) -> f64 {
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
