//! This module contains the implementation of Gauss quadrature rules.
//!
//! It is limited to 1D and to the standard real-number
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
    gauss_type: GaussQuadType,
    /// maximum order of the quadrature rule
    max_order: usize,
    /// minimum number of quadrature points
    min_nqp: usize,
    /// maximum number of quadrature points
    max_nqp: usize,
    /// quadrature points, the nqp-point rule is in `points[nqp - nqp_min]`
    points: Vec<Vec<f64>>,
    /// quadrature weights, the nqp-point rule is in `wieghts[nqp - nqp_min]`
    weights: Vec<Vec<f64>>,
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
/// 	        \beta_{1} & \alpha_{2} & \beta_{2} & ... & 0 \\\\
/// 	        \vdots & \ddots & \ddots  & \ddots & \vdots \\\\
/// 	        0 & ... & \beta_{n-2} & \alpha_{n-1} & \beta_{n-1} \\\\
/// 	        0 & 0 & ... & \beta_{n-1} & \alpha_{n}
///         \end{bmatrix}
/// \\]
///
/// where:
///
/// \\[
///   \alpha_{i} = -\frac{b_{i}}{a_{i}}, \quad
///   \beta_{i} = \left( \frac{c_{i+1}}{a_{i}a_{i+1}}\right)^{1/2}
/// \\]
fn golub_welsch<F: Fn(usize) -> (f64, f64, f64)>(
    nqp: usize,
    gauss_type: GaussQuadType,
    recurrence_fcn: F,
) -> (Vec<f64>, Vec<f64>) {
    //{{{ init
    let mut T = na::DMatrix::<f64>::zeros(nqp, nqp);
    let (mut ai, mut bi, mut ci) = (0.0f64, 0.0f64, 0.0f64);
    let (mut aj, mut bj, mut cj) = (0.0f64, 0.0f64, 0.0f64);
    let (mut alpha_i, mut alpha_j) = (0.0f64, 0.0f64);
    let (mut beta_i, beta_j, mut beta_h) = (0.0f64, 0.0f64, 0.0f64);
    //}}}
    //{{{ com: deal with row 0
    {
        (ai, bi, ci) = recurrence_fcn(0);
        (aj, bj, cj) = recurrence_fcn(1);
        alpha_i = -(bi / ai);
        beta_i = (cj / (ai * aj)).sqrt();
        T[(0, 0)] = alpha_i;
        T[(0, 1)] = beta_i;
        beta_h = beta_i;
    }
    //}}}
    //{{{ com: deal with rows 1 to nqp-2
    for i in 1..nqp - 1 {
        (ai, bi, ci) = recurrence_fcn(i);
        (aj, bj, cj) = recurrence_fcn(i + 1);
        alpha_i = -(bi / ai);
        beta_i = (cj / (ai * aj)).sqrt();

        T[(i, i - 1)] = beta_h;
        T[(i, i)] = alpha_i;
        T[(i, i + 1)] = beta_i;
        beta_h = beta_i;
    }
    //}}}
    //{{{ com: deal with row nqp-1
    {
        (ai, bi, ci) = recurrence_fcn(nqp - 1);
        (aj, bj, cj) = recurrence_fcn(nqp);
        alpha_j = -(bj / aj);
        beta_i = (cj / (ai * aj)).sqrt();
        T[(nqp - 1, nqp - 2)] = beta_h;
        T[(nqp - 1, nqp - 1)] = alpha_j;
    }
    //}}}
    //{{{ com: eigendecompose
    let eigen_decomp = T.symmetric_eigen();
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
mod tests {

    use super::*;
    use approx::assert_relative_eq;
    use approx::ulps_eq;
    use serde::Deserialize;
    use std::fs;

    const MAX_REL: f64 = 1e-10;

    #[test]
    fn test_get_legendre_points() {
        let points = get_legendre_points();
        let points_5 = points.gauss_quad_from_nqp(5);
        let points_ok = [
            -0.9061798459386642,
            -0.5384693101056831,
            2.1044260617163113e-16,
            0.5384693101056824,
            0.9061798459386633,
        ];
        let weights_k = [
            0.23692688505618958,
            0.4786286704993664,
            0.5688888888888887,
            0.47862867049936586,
            0.2369268850561888,
        ];

        for i in 0..5 {
            assert!(ulps_eq!(points_ok[i], points_5.points[i], max_ulps = 4));
            assert!(ulps_eq!(weights_k[i], points_5.weights[i], max_ulps = 4));
        }
    }

    #[test]
    fn test_get_lobatto_points() {
        let points = get_lobatto_points();
        let points_5 = points.gauss_quad_from_nqp(5);
        let points_ok = [
            -1.0,
            -0.6546536707079771,
            5.307881287095001e-17,
            0.6546536707079771,
            1.0,
        ];
        let weights_ok = [
            0.1,
            0.5444444444444444,
            0.7111111111111111,
            0.5444444444444444,
            0.1,
        ];

        for i in 0..5 {
            assert!(ulps_eq!(points_ok[i], points_5.points[i], max_ulps = 4));
            assert!(ulps_eq!(weights_ok[i], points_5.weights[i], max_ulps = 4));
        }
    }

    #[derive(Deserialize)]
    struct GaussQuadTest4 {
        points: Vec<f64>,
        weights: Vec<f64>,
    }

    #[derive(Deserialize)]
    struct GaussQuadTest3 {
        n2: GaussQuadTest4,
        n3: GaussQuadTest4,
        n4: GaussQuadTest4,
        n5: GaussQuadTest4,
        n6: GaussQuadTest4,
        n11: GaussQuadTest4,
        n26: GaussQuadTest4,
        n37: GaussQuadTest4,
    }

    #[derive(Deserialize)]
    struct GaussQuadTest2 {
        description: String,
        values: GaussQuadTest3,
    }

    #[derive(Deserialize)]
    struct GaussQuadTest1 {
        legendre: GaussQuadTest2,
        lobatto: GaussQuadTest2,
    }

    impl GaussQuadTest1 {
        fn new() -> Self {
            let json_file =
                fs::read_to_string("assets/gauss-quad.json").expect("Unable to read file");
            serde_json::from_str(&json_file).expect("Could not deserialize")
        }
    }

    macro_rules! legendre_test {
        ($test_name: ident, $dataset: ident, $idx: expr) => {
            #[test]
            fn $test_name() {
                let test_data = GaussQuadTest1::new();
                let leg = GuassQuadSet::new(GaussQuadType::Legendre, 90);

                let points1 = test_data.legendre.values.$dataset.points;
                let weights1 = test_data.legendre.values.$dataset.weights;
                let points2 = leg.points[$idx].clone();
                let weights2 = leg.weights[$idx].clone();
                assert_eq!(points1.len(), points2.len());
                for i in 0..points1.len() {
                    assert_relative_eq!(points1[i], points2[i], epsilon = MAX_REL);
                    assert_relative_eq!(weights1[i], weights2[i], epsilon = MAX_REL);
                }
            }
        };
    }
    legendre_test!(legendre_test1, n2, 0);
    legendre_test!(legendre_test2, n3, 1);
    legendre_test!(legendre_test3, n4, 2);
    legendre_test!(legendre_test4, n5, 3);
    legendre_test!(legendre_test5, n6, 4);
    legendre_test!(legendre_test6, n11, 9);
    legendre_test!(legendre_test7, n26, 24);
    legendre_test!(legendre_test8, n37, 35);
    //..............................................................................................

    macro_rules! lobatto_test {
        ($test_name: ident, $dataset: ident, $idx: expr) => {
            #[test]
            fn $test_name() {
                let test_data = GaussQuadTest1::new();
                let leg = GuassQuadSet::new(GaussQuadType::Lobatto, 90);

                let points1 = test_data.lobatto.values.$dataset.points;
                let weights1 = test_data.lobatto.values.$dataset.weights;
                let points2 = leg.points[$idx].clone();
                let weights2 = leg.weights[$idx].clone();

                assert_eq!(points1.len(), points2.len());
                for i in 0..points1.len() {
                    assert_relative_eq!(points1[i], points2[i], epsilon = MAX_REL);
                    assert_relative_eq!(weights1[i], weights2[i], epsilon = MAX_REL);
                }
            }
        };
    }
    lobatto_test!(lobatto_test1, n2, 0);
    lobatto_test!(lobatto_test2, n3, 1);
    lobatto_test!(lobatto_test3, n4, 2);
    lobatto_test!(lobatto_test4, n5, 3);
    lobatto_test!(lobatto_test5, n6, 4);
    lobatto_test!(lobatto_test6, n11, 9);
    lobatto_test!(lobatto_test7, n26, 24);
    lobatto_test!(lobatto_test8, n37, 35);
    //..............................................................................................
}
//}}}
