//! This module contains the implementation of fixed quadrature rules for two-dimensional
//! real-valued functions.

//{{{ crate imports
use crate::common::{append_reason, OptionsError, OptionsStruct};
use crate::gauss::GaussQuadType;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------
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
