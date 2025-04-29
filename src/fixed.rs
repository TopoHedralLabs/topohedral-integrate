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
use crate::common::{append_reason, OptionsError, OptionsStruct};
use crate::gauss::{get_legendre_points, get_lobatto_points, GaussQuad, GaussQuadType};
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
        /// The Gaussian quadrature type
        pub gauss_type: GaussQuadType,
        /// The order of the Gaussian quadrature to use for each dimension.
        pub order: usize,
        /// The bounds of the integration region.
        pub bounds: (f64, f64),
        /// Optional subdivision points to use within the integration region.
        pub subdiv: Option<Vec<f64>>,
    }
    //}}}
    //{{{ impl OptionsStruct for FixedQuadOpts
    impl OptionsStruct for FixedQuadOpts {
        fn is_ok(&self, full: bool) -> Result<(), OptionsError> {
            let mut ok = true;
            let mut err = if full {
                OptionsError::InvalidOptionsFull(String::new())
            } else {
                OptionsError::InvalidOptionsShort
            };

            if self.bounds.0 > self.bounds.1 {
                ok = false;
                append_reason(
                    &mut err,
                    "Bounds invalid, low bound greater than high bound",
                );
            }

            if let Some(ref v) = self.subdiv {
                if v.is_empty() {
                    append_reason(&mut err, "Initial subdivisions invalid, must be non-empty");
                    ok = false
                }
                for vi in v {
                    if *vi <= self.bounds.0 || *vi >= self.bounds.1 {
                        append_reason(
                            &mut err,
                            "Initial subdivisions invalid, must be inside bounds",
                        );
                        ok = false;
                        break;
                    }
                }
            }

            if ok {
                Ok(())
            } else {
                Err(err)
            }
        }
    }
    //}}}
    //{{{ struct: FixedQuad
    #[derive(Debug)]
    pub struct FixedQuad {
        /// The set of points and weights for the fixed quadrature rule. Point `i` and weight `i`
        /// are stored in `points_weights[2 * i]` and `points_weights[2 * i + 1]`, respectively.
        pub points_weights: Vec<f64>,
        /// The underlying Gaussian quadrature type
        pub gauss_type: GaussQuadType,
        /// The order of the Gaussian quadrature to use for each dimension.
        pub order: usize,
        /// The bounds of the integration region.
        pub bounds: (f64, f64),
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

            let mut points_weights = Vec::with_capacity(nqp);

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
                        let (c, d) = (subdiv[i], subdiv[i + 1]);
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
                }
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
                } //}}}
            }
            //}}}
            //{{{ ret
            Self {
                points_weights,
                gauss_type: gauss_rule.gauss_type,
                order: opts.order,
                bounds: opts.bounds,
            }
            //}}}
        }
        //}}}
        //{{{ fun: integrate
        pub fn integrate<F: Fn(f64) -> f64>(&self, f: &F, bounds: Option<(f64, f64)>) -> f64 {
            let mut integral = 0.0;

            match bounds {
                Some(bounds) => {
                    let (a, b) = self.bounds;
                    let (c, d) = bounds;
                    let jac = (d - c) / (b - a);

                    for i in 0..self.points_weights.len() / 2 {
                        let xi = c + jac * (self.points_weights[2 * i] - a);
                        let wi = self.points_weights[2 * i + 1];
                        integral += f(xi) * wi;
                    }
                    integral *= jac;
                }
                None => {
                    for i in 0..self.points_weights.len() / 2 {
                        let xi = self.points_weights[2 * i];
                        let wi = self.points_weights[2 * i + 1];
                        integral += f(xi) * wi;
                    }
                }
            }
            integral
        }
        //}}}
        //{{{ fun: nqp
        pub fn nqp(&self) -> usize {
            self.points_weights.len() / 2
        }
        //}}}
    }
    //}}}
    //{{{ fun: fixed_quad
    pub fn fixed_quad<F: Fn(f64) -> f64>(f: &F, opts: &FixedQuadOpts) -> f64 {
        let quad_rule = FixedQuad::new(opts);
        quad_rule.integrate(f, None)
    }
    //}}}
    //{{{ impl: From<GaussQuad> for FixedQuad
    impl From<GaussQuad> for FixedQuad {
        fn from(value: GaussQuad) -> Self {
            let nqp = value.nqp;
            let mut points_weights = Vec::with_capacity(nqp * 2);

            for i in 0..nqp {
                let xi = value.points[i];
                let wi = value.weights[i];
                points_weights.push(xi);
                points_weights.push(wi);
            }

            Self {
                points_weights,
                gauss_type: value.gauss_type,
                order: value.gauss_type.order_from_nqp(nqp),
                bounds: value.gauss_type.range(),
            }
        }
    }
    //}}}

    //----------------------------------------------------------------------------------------------
    //{{{ mod: tests
    #[cfg(test)]
    mod tests {}
    //}}}
}
//}}}
//{{{ mod d2
pub mod d2 {
    //! This module contains the implementation of fixed quadrature rules for two-dimensional
    //! real-valued functions.

    use super::*;

    //{{{ struct: FixedQuadOpts
    #[derive(Debug)]
    pub struct FixedQuadOpts {
        /// Gauss quadrature rule to use in u and v directions
        pub gauss_type: (GaussQuadType, GaussQuadType),
        /// Order of the Gauss quadrature rule to use in u and v directions
        pub order: (usize, usize),
        /// Bounds of the integration region in order ``(umin, max, vmin, vmax)``
        pub bounds: (f64, f64, f64, f64),
        /// Optional Subdivision of the integration region in order ``(u, v)``
        pub subdiv: Option<(Vec<f64>, Vec<f64>)>,
    }
    //}}}
    //{{{ impl: OptionsStruct for FixedQuadOpts
    impl OptionsStruct for FixedQuadOpts {
        fn is_ok(&self, full: bool) -> Result<(), OptionsError> {
            let mut ok = true;
            let mut err = if full {
                OptionsError::InvalidOptionsFull(String::new())
            } else {
                OptionsError::InvalidOptionsShort
            };

            if self.bounds.0 > self.bounds.1 || self.bounds.2 > self.bounds.3 {
                ok = false;
                append_reason(
                    &mut err,
                    "Bounds invalid, low bound greater than high bound",
                );
            }

            if let Some(subdiv) = &self.subdiv {
                if subdiv.0.is_empty() && subdiv.1.is_empty() {
                    ok = false;
                    append_reason(
                        &mut err,
                        "Initial subdivision invalid, at least 1 must be non-empty",
                    );
                }

                for u in &subdiv.0 {
                    if *u < self.bounds.0 || *u > self.bounds.1 {
                        ok = false;
                        append_reason(
                            &mut err,
                            "Initial subdivision invalid, must be within bounds",
                        );
                    }
                }
                for v in &subdiv.1 {
                    if *v < self.bounds.2 || *v > self.bounds.3 {
                        ok = false;
                        append_reason(
                            &mut err,
                            "Initial subdivision invalid, must be within bounds",
                        );
                    }
                }
            }

            if ok {
                Ok(())
            } else {
                Err(err)
            }
        }
    }
    //}}}
    //{{{ struct: FixedQuad
    #[derive(Debug)]
    pub struct FixedQuad {
        /// The set of points and weights for the fixed quadrature rule. Point `i` and weight `i`
        /// are stored in `points_weights[3*i..3*i+1]`, and `points_weights[3*i+2]`, respectively.
        pub points_weights: Vec<f64>,
        /// The underlying Gaussian quadrature type in each dimension
        pub gauss_type: (GaussQuadType, GaussQuadType),
        /// The order of the Gaussian quadrature to use for each dimension.
        pub order: (usize, usize),
        /// Bounds of the integration region in order ``(umin, max, vmin, vmax)``
        pub bounds: (f64, f64, f64, f64),
    }
    //}}}
    //{{{ impl: FixedQuad
    impl FixedQuad {
        //{{{ fun: new
        pub fn new(opts: &FixedQuadOpts) -> Self {
            let (u_gauss_type, v_gauss_type) = opts.gauss_type;
            let (u_order, v_order) = opts.order;
            let (umin, umax, vmin, vmax) = opts.bounds;
            let mut u_subdiv: Option<Vec<f64>> = None;
            let mut v_subdiv: Option<Vec<f64>> = None;

            if let Some(subdiv) = &opts.subdiv {
                if !subdiv.0.is_empty() {
                    u_subdiv = Some(subdiv.0.clone());
                }
                if !subdiv.1.is_empty() {
                    v_subdiv = Some(subdiv.1.clone());
                }
            }

            let fixed_rule_u = d1::FixedQuad::new(&d1::FixedQuadOpts {
                gauss_type: u_gauss_type,
                order: u_order,
                bounds: (umin, umax),
                subdiv: u_subdiv,
            });

            let fixed_rule_v = d1::FixedQuad::new(&d1::FixedQuadOpts {
                gauss_type: v_gauss_type,
                order: v_order,
                bounds: (vmin, vmax),
                subdiv: v_subdiv,
            });

            let nqp = fixed_rule_u.nqp() * fixed_rule_v.nqp();
            let mut points_weights = Vec::<f64>::with_capacity(3 * nqp);

            for i in 0..fixed_rule_u.nqp() {
                let xi = fixed_rule_u.points_weights[2 * i];
                let wi = fixed_rule_u.points_weights[2 * i + 1];

                for j in 0..fixed_rule_v.nqp() {
                    let xj = fixed_rule_v.points_weights[2 * j];
                    let wj = fixed_rule_v.points_weights[2 * j + 1];

                    points_weights.push(xi);
                    points_weights.push(xj);
                    points_weights.push(wi * wj);
                }
            }

            Self {
                points_weights,
                gauss_type: opts.gauss_type,
                order: opts.order,
                bounds: opts.bounds,
            }
        }
        //}}}
        //{{{ fun: integrate
        pub fn integrate<F: Fn(f64, f64) -> f64>(
            &self,
            f: &F,
            bounds: Option<(f64, f64, f64, f64)>,
        ) -> f64 {
            let mut integral = 0.0;

            match bounds {
                Some(bounds) => {
                    let (a, b, c, d) = bounds;
                    let (umin, umax, vmin, vmax) = self.bounds;
                    let jac_u = (b - a) / (umax - umin);
                    let jac_v = (d - c) / (vmax - vmin);

                    for i in 0..self.points_weights.len() / 3 {
                        let xi1 = a + jac_u * (self.points_weights[3 * i] - umin);
                        let xi2 = c + jac_v * (self.points_weights[3 * i + 1] - vmin);
                        let wi = self.points_weights[3 * i + 2];
                        integral += f(xi1, xi2) * wi;
                    }
                    integral *= jac_u * jac_v;
                }
                None => {
                    for i in 0..self.points_weights.len() / 3 {
                        let xi1 = self.points_weights[3 * i];
                        let xi2 = self.points_weights[3 * i + 1];
                        let wi = self.points_weights[3 * i + 2];
                        integral += f(xi1, xi2) * wi;
                    }
                }
            }
            integral
        }
        //}}}
        //{{{ fun: nqp
        pub fn nqp(&self) -> usize {
            self.points_weights.len() / 3
        }
        //}}}
    }
    //}}}
    //{{{ fun: fixed_quad
    pub fn fixed_quad<F: Fn(f64, f64) -> f64>(f: &F, opts: &FixedQuadOpts) -> f64 {
        let quad_rule = FixedQuad::new(opts);
        quad_rule.integrate(f, None)
    }
    //}}}

    //-------------------------------------------------------------------------------------------------
    //{{{ mod: tests
    #[cfg(test)]
    mod tests {}
    //}}}
}
//}}}
