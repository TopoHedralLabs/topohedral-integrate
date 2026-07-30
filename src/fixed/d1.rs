//! This module contains the implementation of fixed quadrature rules for one-dimensional
//! real-valued functions.

//{{{ crate imports
use crate::common::{append_reason, OptionsError, OptionsVerify};
use crate::gauss::{legendre_rules, lobatto_rules, GaussFamily, GaussRule, RuleError, MAX_DEGREE};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: FixedQuadOpts
/// Configuration for one-dimensional fixed quadrature.
#[derive(Debug)]
pub struct FixedQuadOpts {
    /// Gauss quadrature family used on every subinterval.
    pub gauss_type: GaussFamily,
    /// Minimum polynomial exactness requested for the rule.
    pub order: usize,
    /// Integration interval `(lower, upper)`.
    pub bounds: (f64, f64),
    /// Optional interior subdivision points.
    ///
    /// Supply points in strictly increasing order to partition the interval into non-overlapping
    /// subintervals. The constructor validates that every point lies strictly inside `bounds`.
    pub subdiv: Option<Vec<f64>>,
}
//}}}
//{{{ impl OptionsStruct for FixedQuadOpts
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

        if self.order > MAX_DEGREE {
            ok = false;
            append_reason(&mut err, "Quadrature order is not supported");
        }

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
/// A reusable one-dimensional fixed quadrature rule.
#[derive(Debug)]
pub struct FixedQuad {
    /// The set of points and weights for the fixed quadrature rule. Point `i` and weight `i`
    /// are stored in `points_weights[2 * i]` and `points_weights[2 * i + 1]`, respectively.
    pub points_weights: Vec<f64>,
    /// The options used to construct the rule.
    pub opts: FixedQuadOpts,
}
//}}}
//{{{ impl: FixedQuad
impl FixedQuad {
    //{{{ fun: new
    /// Builds a reusable fixed quadrature rule from `opts`.
    ///
    /// Returns [`OptionsError`] when the options are invalid or the cached Gaussian rule could not
    /// be initialized.
    pub fn new(opts: FixedQuadOpts) -> Result<Self, OptionsError> {
        opts.is_ok(true)?;
        let points_weights = build_points_weights(
            opts.gauss_type,
            opts.order,
            opts.bounds,
            opts.subdiv.as_deref(),
        )?;

        Ok(Self {
            points_weights,
            opts,
        })
    }
    //}}}
    //{{{ fun: integrate
    /// Integrates `f` using this rule.
    ///
    /// When `bounds` is `Some((lower, upper))`, the stored rule is linearly remapped from its
    /// configured bounds to that interval. When it is `None`, the configured bounds are used.
    pub fn integrate<F: Fn(f64) -> f64>(
        &self,
        f: &F,
        bounds: Option<(f64, f64)>,
    ) -> f64 {
        let mut integral = 0.0;

        match bounds {
            Some(bounds) => {
                let (a, b) = self.opts.bounds;
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
    /// Returns the total number of quadrature points, including all subintervals.
    pub fn nqp(&self) -> usize {
        self.points_weights.len() / 2
    }
    //}}}
}
//}}}
//{{{ fun: build_points_weights
pub(super) fn build_points_weights(
    gauss_type: GaussFamily,
    order: usize,
    bounds: (f64, f64),
    subdiv: Option<&[f64]>,
) -> Result<Vec<f64>, RuleError> {
    let gauss_rule = match gauss_type {
        GaussFamily::Legendre => legendre_rules()?.get_for_degree(order)?,
        GaussFamily::Lobatto => lobatto_rules()?.get_for_degree(order)?,
    };

    let num_divs = subdiv.map_or(1, |subdiv| subdiv.len() + 1);
    let mut points_weights = Vec::with_capacity(2 * gauss_rule.point_count() * num_divs);
    let (a, b) = gauss_rule.family().range();

    let mut append_interval = |c: f64, d: f64| {
        let jac = (d - c) / (b - a);
        for i in 0..gauss_rule.point_count() {
            let zi = gauss_rule.points()[i];
            let xi = c + jac * (zi - a);
            let wi = jac * gauss_rule.weights()[i];
            points_weights.push(xi);
            points_weights.push(wi);
        }
    };

    match subdiv {
        Some(subdiv) => {
            append_interval(bounds.0, subdiv[0]);
            for interval in subdiv.windows(2) {
                append_interval(interval[0], interval[1]);
            }
            append_interval(subdiv[subdiv.len() - 1], bounds.1);
        }
        None => append_interval(bounds.0, bounds.1),
    }

    Ok(points_weights)
}
//}}}
//{{{ fun: fixed_quad
/// Integrates `f` over `opts.bounds` using a newly constructed fixed rule.
///
/// Returns [`OptionsError`] when `opts` is invalid or the cached Gaussian rule could not be
/// initialized.
pub fn fixed_quad<F: Fn(f64) -> f64>(
    f: &F,
    opts: FixedQuadOpts,
) -> Result<f64, OptionsError> {
    let quad_rule = FixedQuad::new(opts)?;
    Ok(quad_rule.integrate(f, None))
}
//}}}
//{{{ impl: From<GaussRule> for FixedQuad
/// Converts a Gauss rule on its reference interval into a reusable fixed rule.
impl From<GaussRule> for FixedQuad {
    fn from(value: GaussRule) -> Self {
        let point_count = value.point_count();
        let mut points_weights = Vec::with_capacity(point_count * 2);

        for i in 0..point_count {
            let xi = value.points()[i];
            let wi = value.weights()[i];
            points_weights.push(xi);
            points_weights.push(wi);
        }

        Self {
            points_weights,
            opts: FixedQuadOpts {
                gauss_type: value.family(),
                order: value.exactness(),
                bounds: value.family().range(),
                subdiv: None,
            },
        }
    }
}
//}}}

//----------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {}
//}}}
