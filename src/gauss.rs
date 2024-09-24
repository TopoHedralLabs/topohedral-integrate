//! This module contains the implementation of Gauss quadrature rules.
//!
//! Gauss quadrature is a numerical integration method that approximates the integral of a function
//! using a weighted sum of function values at specific points. These points are the roots of a 
//! family of orthogonal polynomials.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
use nalgebra as na;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum:   GaussQuadType
/// An enumeration of supported Gauss quadrature rules. Each member corresponds to the orthogonal 
/// polynomial family used for the rule.
///
/// `Legendre` specifies Gauss-Legendre quadrature.
/// `Lobatto` specifies Gauss-Lobatto quadrature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GaussQuadType
{
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
            Self::Lobatto =>  (-1.0, 1.0),
        }
    }
}
//}}}
//{{{ collection: GuassQuadSet
//{{{ struct: GuassQuadSet
/// Struct to represent a collection of Gauss quadrature rules up to a given order.
pub struct GuassQuadSet
{
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
impl GuassQuadSet
{
    pub fn new(
        gauss_type: GaussQuadType,
        order: usize,
    ) -> Self
    {
        // set the min and max nqp's available for the given quadrature rule
        let min_nqp = 2;
        let max_nqp = match gauss_type
        {
            GaussQuadType::Legendre => (order + 1) / 2,
            GaussQuadType::Lobatto => (order + 3) / 2,
        };
        // preallocate the points and weights
        let mut points = vec![vec![0.0; max_nqp]; max_nqp - min_nqp + 1];
        let mut weights = vec![vec![0.0; max_nqp]; max_nqp - min_nqp + 1];

        match gauss_type
        {
            GaussQuadType::Lobatto =>
            {
                points[0] = vec![-1.0f64, 1.0f64];
                weights[0] = vec![1.0f64, 1.0f64];

                points[1] = vec![-1.0f64, 0.0f64, 1.0f64];
                const W1: f64 = 1.0f64 / 3.0f64;
                const W2: f64 = 4.0f64 / 3.0f64;
                weights[1] = vec![W1, W2, W1];

                for i in (min_nqp + 2)..max_nqp
                {
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
            GaussQuadType::Legendre =>
            {
                for i in min_nqp..max_nqp
                {
                    let (points_i, weights_i) =
                        golub_welsch(i, gauss_type, legendre_recursion_coeffs);
                    points[i - min_nqp] = points_i;
                    weights[i - min_nqp] = weights_i;
                }
            }
        }

        for i in min_nqp..max_nqp+1
        {
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

    pub fn gauss_quad_from_nqp(&self, nqp: usize) -> GaussQuad 
    {
        debug_assert!(nqp >= self.min_nqp && nqp <= self.max_nqp);
        let points_nqp = self.points[nqp - self.min_nqp].clone();
        let weights_nqp = self.weights[nqp - self.min_nqp].clone();
        GaussQuad::new(self.gauss_type, points_nqp, weights_nqp)
    }
}
//}}}
//}}}
//{{{ collection: GaussQuad
//{{{ struct: GaussQuad
/// This struct represents a specific quadrature rule, meaning a set of quadrature points and 
/// a set of assocciated weights.
#[derive(Debug, Clone)]
pub struct GaussQuad 
{
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
impl GaussQuad
{

    fn new(gauss_type: GaussQuadType, points: Vec<f64>, weights: Vec<f64>) -> Self
    {
        debug_assert_eq!(points.len(), weights.len());
        let nqp = points.len();
        Self {
            gauss_type,
            nqp,
            points,
            weights,
        }
    }

    pub fn integrate<F: Fn(f64) -> f64>(&self, f: F, range: (f64, f64)) -> f64
    {
        debug_assert!(range.0 < range.1);

        let (a, b) = self.gauss_type.range();
        let (c, d) = range;
        let mut sum = 0.0;
        let jac = ((d - c) / (b - a));
        for i in 0..self.nqp
        {
            let xi = c + ((d - c) / (b - a)) * (self.points[i] - a);
            let wi = self.weights[i];
            sum +=  jac * wi * f(xi);
        }
        sum
    }
}
//}}}
//}}}
//{{{ fun:    golub_welsch
/// Computes the Golub-Welsch algorithm to generate Gauss quadrature points and weights for numerical integration.
///
/// Takes the number of quadrature points, quadrature type, and a recurrence function. The recurrence function takes
/// an index and returns the recurrence coefficients for that index.
///
/// The Golub-Welsch algorithm uses the recurrence coefficients to construct a tridiagonal matrix which is
/// then eigendecomposed to obtain the quadrature points and weights.
fn golub_welsch<F: Fn(usize) -> (f64, f64, f64)>(
    nqp: usize,
    gauss_type: GaussQuadType,
    recurrence_fcn: F,
) -> (Vec<f64>, Vec<f64>)
{
    // ............................................................ compute the T matrix
    let mut T = na::DMatrix::<f64>::zeros(nqp, nqp);
    let (mut ai, mut bi, mut ci) = (0.0f64, 0.0f64, 0.0f64);
    let (mut aj, mut bj, mut cj) = (0.0f64, 0.0f64, 0.0f64);
    let (mut alpha_i, mut alpha_j) = (0.0f64, 0.0f64);
    let (mut beta_i, mut beta_j, mut beta_h) = (0.0f64, 0.0f64, 0.0f64);

    // deal with row 0
    {
        (ai, bi, ci) = recurrence_fcn(0);
        (aj, bj, cj) = recurrence_fcn(1);
        alpha_i = -(bi / ai);
        beta_i = (cj / (ai * aj)).sqrt();
        T[(0, 0)] = alpha_i;
        T[(0, 1)] = beta_i;
        beta_h = beta_i;
    }
    // deal with rows 1 to nqp-2
    for i in 1..nqp - 1
    {
        (ai, bi, ci) = recurrence_fcn(i);
        (aj, bj, cj) = recurrence_fcn(i + 1);
        alpha_i = -(bi / ai);
        beta_i = (cj / (ai * aj)).sqrt();

        T[(i, i - 1)] = beta_h;
        T[(i, i)] = alpha_i;
        T[(i, i + 1)] = beta_i;
        beta_h = beta_i;
    }
    // deal with row nqp-1
    {
        (ai, bi, ci) = recurrence_fcn(nqp - 1);
        (aj, bj, cj) = recurrence_fcn(nqp);
        alpha_j = -(bj / aj);
        beta_i = (cj / (ai * aj)).sqrt();
        T[(nqp - 1, nqp - 2)] = beta_h;
        T[(nqp - 1, nqp - 1)] = alpha_j;
    }

    // . .......................................................... compute the eigen decomp

    let eigen_decomp = T.symmetric_eigen();

    let qpoints: Vec<f64> = eigen_decomp.eigenvalues.iter().copied().collect();
    let mu0 = gauss_type.weight_integral();
    let qweights: Vec<f64> = eigen_decomp.eigenvectors.row(0).iter().map(|x| x.powi(2) * mu0).collect();    
    let mut combined: Vec<(f64, f64)> = qpoints.iter().cloned().zip(qweights.iter().cloned()).collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (qpoints_final, qweights_final): (Vec<f64>, Vec<f64>) = combined.iter().cloned().unzip();
    (qpoints_final, qweights_final)
}
//..................................................................................................
//}}}
//{{{ fun:    legendre_recursion_coeffs
/// Computes the recurrence coefficients for the nth Legendre polynomial using the recurrence relation.
///
/// Returns a tuple containing the coefficients (a, b, c) for the i'th polynomial.
fn legendre_recursion_coeffs(i: usize) -> (f64, f64, f64)
{
    let i_f64 = i as f64;
    let ai = (2.0 * i_f64 + 1.0) / (i_f64 + 1.0);
    let bi = 0.0;
    let ci = i_f64 / (i_f64 + 1.0);
    (ai, bi, ci)
}
//}}}
//{{{ fun:    lobatto_recursion_coeffs
/// Computes the recurrence coefficients `ai`, `bi`, and `ci` for
/// the Lobatto quadrature rule at index `i`.
///
/// Returns a tuple containing the coefficients (a, b, c) for the i'th polynomial.
fn lobatto_recursion_coeffs(i: usize) -> (f64, f64, f64)
{
    let i_f64 = i as f64;
    let ai = ((2.0 * i_f64 + 3.0) * (i_f64 + 2.0)) / ((i_f64 + 1.0) * (i_f64 + 3.0));
    let bi = 0.0;
    let ci = ((i_f64 + 1.0) * (i_f64 + 2.0)) / ((i_f64 + 3.0) * (i_f64 + 1.0));
    (ai, bi, ci)
}
//}}}
//{{{ fun:    legendre
fn legendre(
    n: usize,
    x: f64,
) -> f64
{
    debug_assert!(x >= -1.0 && x <= 1.0);

    let (mut leg_n, mut leg_1, mut leg_2) = (1.0f64, 1.0f64, 0.0f64);
    for i in 0..n
    {
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
mod tests
{
  
    use super::*;
    use std::fs;
    use approx::{assert_relative_eq};

    use serde::Deserialize;

    #[derive(Deserialize)]
    struct GaussQuadTest4 
    {
        points: Vec<f64>, 
        weights: Vec<f64>,
    }

    #[derive(Deserialize)]
    struct GaussQuadTest3
    {
        N2: GaussQuadTest4, 
        N3: GaussQuadTest4,
        N4: GaussQuadTest4,
        N5: GaussQuadTest4,
        N6: GaussQuadTest4,
        N11: GaussQuadTest4,
        N26: GaussQuadTest4,
        N37: GaussQuadTest4,
    }

    #[derive(Deserialize)]
    struct GaussQuadTest2
    {
        description: String,
        values: GaussQuadTest3,
    }

    #[derive(Deserialize)]
    struct GaussQuadTest1
    {
        legendre: GaussQuadTest2,
        lobatto: GaussQuadTest2,
    }

    impl GaussQuadTest1
    {
        fn new() -> Self
        {
            let json_file = fs::read_to_string("assets/gauss-quad.json").expect("Unable to read file");
            serde_json::from_str(&json_file).expect("Could not deserialize")
        }
    }


    macro_rules! legendre_test {
        ($test_name: ident, $dataset: ident, $idx: expr) => {
            #[test]
            fn $test_name()
            {
                let test_data = GaussQuadTest1::new();
                let leg = GuassQuadSet::new(GaussQuadType::Legendre, 90);

                let points1 = test_data.legendre.values.$dataset.points;
                let weights1 = test_data.legendre.values.$dataset.weights;
                let points2 = leg.points[$idx].clone();
                let weights2 = leg.weights[$idx].clone();
                assert_eq!(points1.len(), points2.len());
                for i in 0..points1.len()
                {
                    assert_relative_eq!(points1[i], points2[i], epsilon=1e-10); 
                    assert_relative_eq!(weights1[i], weights2[i], epsilon=1e-10); 
                }
            }
        };
    }
    legendre_test!(legendre_test1, N2, 0);
    legendre_test!(legendre_test2, N3, 1);
    legendre_test!(legendre_test3, N4, 2);
    legendre_test!(legendre_test4, N5, 3);
    legendre_test!(legendre_test5, N6, 4);
    legendre_test!(legendre_test6, N11, 9);
    legendre_test!(legendre_test7, N26, 24);
    legendre_test!(legendre_test8, N37, 35);
    //..............................................................................................

    macro_rules! lobatto_test{
        ($test_name: ident, $dataset: ident, $idx: expr) => {
            #[test]
            fn $test_name()
            {
                let test_data = GaussQuadTest1::new();
                let leg = GuassQuadSet::new(GaussQuadType::Lobatto, 90);

                let points1 = test_data.lobatto.values.$dataset.points;
                let weights1 = test_data.lobatto.values.$dataset.weights;
                let points2 = leg.points[$idx].clone();
                let weights2 = leg.weights[$idx].clone();

                assert_eq!(points1.len(), points2.len());
                for i in 0..points1.len()
                {
                    assert_relative_eq!(points1[i], points2[i], epsilon=1e-10); 
                    assert_relative_eq!(weights1[i], weights2[i], epsilon=1e-10); 
                }
            }
        };
    }
    lobatto_test!(lobatto_test1, N2, 0);
    lobatto_test!(lobatto_test2, N3, 1);
    lobatto_test!(lobatto_test3, N4, 2);
    lobatto_test!(lobatto_test4, N5, 3);
    lobatto_test!(lobatto_test5, N6, 4);
    lobatto_test!(lobatto_test6, N11, 9);
    lobatto_test!(lobatto_test7, N26, 24);
    lobatto_test!(lobatto_test8, N37, 35);
    //..............................................................................................

    #[derive(Deserialize)]
    struct PolyIntegralTestData3
    {
        coeffs: Vec<f64>, 
        integral: f64
    }

    #[derive(Deserialize)]
    struct PolyIntegralTestData2
    {
        range: (f64, f64),
        P0: PolyIntegralTestData3,
        P1: PolyIntegralTestData3,
        P2: PolyIntegralTestData3,
        P3: PolyIntegralTestData3,
        P4: PolyIntegralTestData3,
        P5: PolyIntegralTestData3,
        P6: PolyIntegralTestData3,
        P7: PolyIntegralTestData3,
        P8: PolyIntegralTestData3,
        P9: PolyIntegralTestData3,
        P10: PolyIntegralTestData3,
        P11: PolyIntegralTestData3,
        P12: PolyIntegralTestData3, 
        P13: PolyIntegralTestData3,
        P14: PolyIntegralTestData3,
        P15: PolyIntegralTestData3,

    }

    // #[derive(Deserialize)]
    // struct Pol

    #[derive(Deserialize)]
    struct PolyIntegralTestData1
    {
        description: String, 
        values: PolyIntegralTestData2,
    }


    impl PolyIntegralTestData1 {

        fn new() -> Self {
            let json_file = fs::read_to_string("assets/poly-integrals.json").expect("Unable to read file");
            serde_json::from_str(&json_file).expect("Could not deserialize")
        }
    }


    macro_rules! poly_integral_legendre_test {
        ($test_name: ident, $dataset: ident, $nqp: expr) => {
            #[test]            
            fn $test_name()
            {
                let test_data = PolyIntegralTestData1::new();
                let coeffs = test_data.values.$dataset.coeffs;
                let integral1 = test_data.values.$dataset.integral;
                let range = test_data.values.range;

                let pol = |x: f64| {
                    let mut sum = 0.0;
                    for (i, c) in coeffs.iter().enumerate() {
                        sum += c * x.powi(i as i32);
                    }
                    sum
                };

                let leg = GuassQuadSet::new(GaussQuadType::Legendre, 90);
                let leg6 = leg.gauss_quad_from_nqp($nqp);
                let integral2 = leg6.integrate(pol, range);

                assert_relative_eq!(integral1, integral2, epsilon = 1e-5);
            }
        };
    }

    // 2-point integrals
    poly_integral_legendre_test!(poly_integral_legendre_test1, P0, 2);
    poly_integral_legendre_test!(poly_integral_legendre_test2, P1, 2);
    poly_integral_legendre_test!(poly_integral_legendre_test3, P2, 2);
    poly_integral_legendre_test!(poly_integral_legendre_test4, P3, 2);
    // 3-point-integrals
    poly_integral_legendre_test!(poly_integral_legendre_test5, P0, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test6, P1, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test7, P2, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test8, P3, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test9, P4, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test10, P5, 3);
    // 4-point-integrals
    poly_integral_legendre_test!(poly_integral_legendre_test11, P0, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test12, P1, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test13, P2, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test14, P3, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test15, P4, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test16, P5, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test17, P6, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test18, P7, 4);
    // 5-point-integrals
    poly_integral_legendre_test!(poly_integral_legendre_test19, P0, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test20, P1, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test21, P2, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test22, P3, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test23, P4, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test24, P5, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test25, P6, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test26, P7, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test27, P8, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test28, P9, 5);


    macro_rules! poly_integral_lobatto_test {
        ($test_name: ident, $dataset: ident, $nqp: expr) => {
            #[test]            
            fn $test_name()
            {
                let test_data = PolyIntegralTestData1::new();
                let coeffs = test_data.values.$dataset.coeffs;
                let integral1 = test_data.values.$dataset.integral;
                let range = test_data.values.range;

                let pol = |x: f64| {
                    let mut sum = 0.0;
                    for (i, c) in coeffs.iter().enumerate() {
                        sum += c * x.powi(i as i32);
                    }
                    sum
                };

                let lob = GuassQuadSet::new(GaussQuadType::Lobatto, 90);
                let lob_nqp = lob.gauss_quad_from_nqp($nqp);
                let integral2 = lob_nqp.integrate(pol, range);

                assert_relative_eq!(integral1, integral2, epsilon = 1e-5);
            }
        };
    }

    // 2-point integrals
    poly_integral_lobatto_test!(poly_integral_lobatto_test1, P0, 2);
    poly_integral_lobatto_test!(poly_integral_lobatto_test2, P1, 2);
    // 3-point-integrals
    poly_integral_lobatto_test!(poly_integral_lobatto_test5, P0, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test6, P1, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test7, P2, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test8, P3, 3);
    // 4-point-integrals
    poly_integral_lobatto_test!(poly_integral_lobatto_test11, P0, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test12, P1, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test13, P2, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test14, P3, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test15, P4, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test16, P5, 4);
    // 5-point-integrals
    poly_integral_lobatto_test!(poly_integral_lobatto_test19, P0, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test20, P1, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test21, P2, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test22, P3, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test23, P4, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test24, P5, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test25, P6, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test26, P7, 5);
}
//}}}
