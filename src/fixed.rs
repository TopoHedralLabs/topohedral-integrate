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
            //{{{ com: perform the quadrature
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
  
}
//}}}