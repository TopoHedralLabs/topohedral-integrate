//! This module contains the implementation of fixed quadrature rules for one-dimensional
//! real-valued functions.

//{{{ crate imports
use crate::common::{append_reason, OptionsError, OptionsStruct};
use crate::gauss::{get_legendre_points, get_lobatto_points, GaussQuad, GaussQuadType};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

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
