//! This module contains the implementation of fixed quadrature rules for two-dimensional
//! real-valued functions.

//{{{ crate imports
use crate::common::{append_reason, OptionsError, OptionsVerify};
use crate::gauss::GaussQuadType;
use crate::gauss::MAX_ORDER;
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
impl OptionsVerify for FixedQuadOpts {
    fn is_ok(
        &self,
        full: bool,
    ) -> Result<(), OptionsError> {
        let mut ok = true;
        let mut err = if full {
            OptionsError::InvalidOptionsFull(String::new())
        } else {
            OptionsError::InvalidOptionsShort
        };

        let valid_u_order =
            self.order.0 <= MAX_ORDER && self.gauss_type.0.nqp_from_order(self.order.0) >= 2;
        let valid_v_order =
            self.order.1 <= MAX_ORDER && self.gauss_type.1.nqp_from_order(self.order.1) >= 2;
        if !valid_u_order || !valid_v_order {
            ok = false;
            append_reason(&mut err, "Quadrature order is not supported");
        }

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
    /// The options used to construct the rule.
    pub opts: FixedQuadOpts,
}
//}}}
//{{{ impl: FixedQuad
impl FixedQuad {
    //{{{ fun: new
    pub fn new(opts: FixedQuadOpts) -> Result<Self, OptionsError> {
        opts.is_ok(true)?;

        let (u_gauss_type, v_gauss_type) = opts.gauss_type;
        let (u_order, v_order) = opts.order;
        let (umin, umax, vmin, vmax) = opts.bounds;
        let u_subdiv = opts
            .subdiv
            .as_ref()
            .and_then(|subdiv| (!subdiv.0.is_empty()).then_some(subdiv.0.as_slice()));
        let v_subdiv = opts
            .subdiv
            .as_ref()
            .and_then(|subdiv| (!subdiv.1.is_empty()).then_some(subdiv.1.as_slice()));

        let fixed_rule_u = d1::build_points_weights(u_gauss_type, u_order, (umin, umax), u_subdiv);

        let fixed_rule_v = d1::build_points_weights(v_gauss_type, v_order, (vmin, vmax), v_subdiv);

        let nqp_u = fixed_rule_u.len() / 2;
        let nqp_v = fixed_rule_v.len() / 2;
        let nqp = nqp_u * nqp_v;
        let mut points_weights = Vec::<f64>::with_capacity(3 * nqp);

        for i in 0..nqp_u {
            let xi = fixed_rule_u[2 * i];
            let wi = fixed_rule_u[2 * i + 1];

            for j in 0..nqp_v {
                let xj = fixed_rule_v[2 * j];
                let wj = fixed_rule_v[2 * j + 1];

                points_weights.push(xi);
                points_weights.push(xj);
                points_weights.push(wi * wj);
            }
        }

        Ok(Self {
            points_weights,
            opts,
        })
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
                let (umin, umax, vmin, vmax) = self.opts.bounds;
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
pub fn fixed_quad<F: Fn(f64, f64) -> f64>(
    f: &F,
    opts: FixedQuadOpts,
) -> Result<f64, OptionsError> {
    let quad_rule = FixedQuad::new(opts)?;
    Ok(quad_rule.integrate(f, None))
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {}
//}}}
