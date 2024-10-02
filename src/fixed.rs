//! This module provides methods for performing fixed quadrature rules for one-dimensional and 
//! two-dimensional real-valued functions.
//!
//! There are two entry points for the 1D fixed quadrature algorithm: 
//! 
//! - The function `fixed_quad`, which can be used when one merely wants to compute the
//!   integral of a function over a given interval once and therefore does not wish to store the 
//!   quadrature rule itself.
//! - The struct `FixedQuad`, which will store the quadrature rule and can be re-used for 
//!   different functions.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
use crate::gauss::{GaussQuad, GaussQuadType, get_legendre_points, get_lobatto_points};
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod d1
pub mod d1 {
    //! This module contains the implementation of fixed quadrature rules for one-dimensional
    //! real-valued functions.

    use super::*;

    //{{{ struct: FixedQuadOpts
    /// Represents options for a fixed quadrature integration method.
    ///
    /// This struct holds the configuration options for performing a fixed quadrature
    /// integration, such as the Gaussian quadrature type, order, integration bounds,
    /// and optional subdivision points.
    pub struct FixedQuadOpts {
        /// The Gaussian quadrature type to use for each dimension.
        pub gauss_type: GaussQuadType,
        /// The order of the Gaussian quadrature to use for each dimension.
        pub order: usize,
        /// The bounds of the integration region.
        pub bounds: (f64, f64),
        /// Optional subdivision points to use within the integration region.
        pub subdiv: Option<Vec<f64>>,
    }
    //}}}
    //{{{ fun: fixed_quad
    pub fn fixed_quad<F: Fn(f64) -> f64>(f: &F, opts: &FixedQuadOpts) -> f64 
    {
        let quad_rule = FixedQuad::new(opts);
        quad_rule.integrate(f)
    }
    //}}}
    //{{{ struct: FixedQuad
    pub struct FixedQuad {
        pub points_weights: Vec<f64>,
    }
    //}}}
    //{{{ impl: FixedQuad
    impl FixedQuad {
        //{{{ fun: new
        pub fn new(opts: &FixedQuadOpts) -> Self {
            //{{{ loc
            let gauss_rule = match opts.gauss_type {
                GaussQuadType::Legendre => get_legendre_points().gauss_quad_from_order(opts.order),
                GaussQuadType::Lobatto => get_lobatto_points().gauss_quad_from_order(opts.order),
            };

            let num_divs = match &opts.subdiv {
                Some(subdiv) => subdiv.len() + 1,
                None => 1,
            };
            let nqp = gauss_rule.nqp * num_divs;

            let mut points_weights = Vec::new();
            points_weights.reserve(nqp);

            let (a, b) = gauss_rule.gauss_type.range();
            //}}}
            //{{{ com: compute points and weights
            match &opts.subdiv {
                //{{{ case: subdivided 
                Some(subdiv) => {
                    //{{{ com: start div
                    {
                        let (c, d) = (opts.bounds.0, *subdiv.first().unwrap());
                        let jac = (d - c) / (b - a);
                        for i in 0..gauss_rule.nqp {
                            let zi = gauss_rule.points[i];
                            let xi = c + jac * (zi - a);
                            let wi = jac * gauss_rule.weights[i];
                            points_weights.push(xi);
                            points_weights.push(wi);
                        }
                    }
                    //}}}
                    //{{{ com: mid divs
                    for i in 0..subdiv.len() - 1 {
                        let (c, d) = (subdiv[i], subdiv[i+1]);
                        let jac = (d - c) / (b - a);
                        for i in 0..gauss_rule.nqp {
                            let zi = gauss_rule.points[i];
                            let xi = c + jac * (zi - a);
                            let wi = jac * gauss_rule.weights[i];
                            points_weights.push(xi);
                            points_weights.push(wi);
                        }
                    }
                    //}}}
                    //{{{ com: end div
                    {
                        let (c, d) = (*subdiv.last().unwrap(), opts.bounds.1);
                        let jac = (d - c) / (b - a);
                        for i in 0..gauss_rule.nqp {
                            let zi = gauss_rule.points[i];
                            let xi = c + jac * (zi - a);
                            let wi = jac * gauss_rule.weights[i];
                            points_weights.push(xi);
                            points_weights.push(wi);
                        }
                    }
                    //}}}
                }, 
                //}}}
                //{{{ case: not subdivided
                None => {
                    let (c, d) = opts.bounds;
                    let jac = (d - c) / (b - a);
                    for i in 0..gauss_rule.nqp {
                        let zi = gauss_rule.points[i];
                        let xi = c + jac * (zi - a);
                        let wi = jac * gauss_rule.weights[i];
                        points_weights.push(xi);
                        points_weights.push(wi);
                    }
                },
                //}}}
            }
            //}}}
            //{{{ ret
            Self {
                points_weights,
            }
            //}}}
        }
        //}}}
        //{{{ fun: integrate
        pub fn integrate<F: Fn(f64) -> f64>(&self, f: &F) -> f64 {
            let mut integral = 0.0;
            for i in 0..self.points_weights.len() / 2 {
                let xi = self.points_weights[2 * i];
                let wi = self.points_weights[2 * i + 1];
                integral += f(xi) * wi;
            }
            integral
        }
        //}}}
    }
    //}}}
}
//}}}
//{{{ mod d2
pub mod d2 {

    use super::*;

    #[derive(Debug)]
    pub struct FixedQuadOpts {
        pub gauss_type: (GaussQuadType, GaussQuadType),
        pub order: (usize, usize),
        pub bounds: (f64, f64, f64, f64),
        pub subdiv: Option<(Vec<f64>, Vec<f64>)>,
    }


    pub fn fixed_quad<F: Fn(f64, f64) -> f64>(f: &F, opts: &FixedQuadOpts) -> f64
    {
        let guass_rule_u = GaussQuad::new(opts.gauss_type.0, opts.order.0);
        let gauss_rule_v = GaussQuad::new(opts.gauss_type.1, opts.order.1); 

        let mut integral = 0.0;

        match &opts.subdiv {
            Some((subdiv_u, subdiv_v)) => {

            }, 
            None => {

            }
        }


        todo!()
    }
}
//}}}






//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{

    use super::*;
    use approx::assert_relative_eq;
    use std::fs;
    use approx::ulps_eq;
    use serde::Deserialize;



    #[derive(Deserialize)]
    struct PolyIntegralTestData3 {
        coeffs: Vec<f64>,
        integral: f64,
    }

    #[derive(Deserialize)]
    struct PolyIntegralTestData2 {
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
    struct PolyIntegralTestData1 {
        description: String,
        values: PolyIntegralTestData2,
    }

    impl PolyIntegralTestData1 {
        fn new() -> Self {
            let json_file =
                fs::read_to_string("assets/poly-integrals.json").expect("Unable to read file");
            serde_json::from_str(&json_file).expect("Could not deserialize")
        }
    }

    macro_rules! poly_integral_legendre_test {
        ($test_name: ident, $dataset: ident, $nqp: expr) => {
            #[test]
            fn $test_name() {
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

                {
                    let opts = d1::FixedQuadOpts {
                        gauss_type: GaussQuadType::Legendre,
                        order: 2 * $nqp - 1,
                        bounds: range,
                        subdiv: None,
                    };
                    let integral2 = d1::fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, epsilon = 1e-5);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;

                    let opts = d1::FixedQuadOpts {
                        gauss_type: GaussQuadType::Legendre,
                        order: 2 * $nqp - 1,
                        bounds: range,
                        subdiv: vec![a, b].into(),
                    };
                    let integral2 = d1::fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, epsilon = 1e-5);
                }
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
            fn $test_name() {
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

                {
                    let opts = d1::FixedQuadOpts {
                        gauss_type: GaussQuadType::Lobatto,
                        order: 2 * $nqp - 3,
                        bounds: range,
                        subdiv: None,
                    };
                    let integral2 = d1::fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, epsilon = 1e-5);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;

                    let opts = d1::FixedQuadOpts {
                        gauss_type: GaussQuadType::Lobatto,
                        order: 2 * $nqp - 3,
                        bounds: range,
                        subdiv: vec![a, b].into(),
                    };
                    let integral2 = d1::fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, epsilon = 1e-5);
                }
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