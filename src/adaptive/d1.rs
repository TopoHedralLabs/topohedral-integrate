//! This module contains the implementation of adaptive quadrature rules for one-dimensional
//! real-valued functions.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{append_reason, OptionsError, OptionsVerify};
use crate::fixed as fi;
// use crate::gauss::GaussQuadType;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: AdaptiveQuadOpts
/// Configuration for one-dimensional adaptive quadrature.
///
/// The algorithm estimates each subinterval's error from the difference between the low- and
/// high-order rules, then bisects subintervals whose estimate exceeds [`Self::tol`].
#[derive(Debug)]
pub struct AdaptiveQuadOpts {
    /// Integration interval `(lower, upper)`.
    pub bounds: (f64, f64),
    /// Options for the low-order Gauss quadrature rule.
    pub fixed_rule_low: fi::d1::FixedQuadOpts,
    /// Options for the high-order Gauss quadrature rule.
    pub fixed_rule_high: fi::d1::FixedQuadOpts,
    /// Positive error tolerance applied independently to each subinterval.
    pub tol: f64,
    /// Reserved maximum refinement depth.
    ///
    /// This value must be positive, but the current implementation does not use it to limit
    /// refinement.
    pub max_depth: usize,
    /// Optional initial interior subdivision points.
    ///
    /// Provide points in strictly increasing order and do not include either bound. The
    /// implementation validates that every point is strictly inside [`Self::bounds`].
    pub init_subdiv: Option<Vec<f64>>,
}
//}}}
//{{{ impl: OptionsStruct for AdaptiveQuadOpts
impl OptionsVerify for AdaptiveQuadOpts {
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

        if self.bounds.0 >= self.bounds.1 {
            append_reason(
                &mut err,
                "Bounds invalid, low bound greater than high bound",
            );
            ok = false;
        }

        if self.fixed_rule_low.order >= self.fixed_rule_high.order {
            append_reason(
                &mut err,
                "Gauss rule order mismatch, low order greater than high order",
            );
            ok = false;
        }

        if self.tol <= 0.0 {
            append_reason(&mut err, "Tolerance invalid, must be positive");
            ok = false;
        }

        if self.max_depth == 0 {
            append_reason(
                &mut err,
                "Maximum number of subdivisions invalid, must be positive",
            );
            ok = false;
        }

        if let Some(ref v) = self.init_subdiv {
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
//{{{ struct: AdaptiveQuadResult
/// Value and diagnostics returned by one-dimensional adaptive quadrature.
#[derive(Debug)]
pub struct AdaptiveQuadResult {
    /// Approximate integral, computed by summing the low-order-rule estimates.
    pub integral: f64,
    /// Sum of the absolute differences between low- and high-order estimates on terminal
    /// subintervals.
    pub error_estimate: f64,
    /// Number of terminal subintervals.
    pub num_subdiv: usize,
    /// Number of function calls made by both rules.
    pub num_fn_eval: usize,
}
//}}}
//{{{ fun: error_estimate
/// Computes the low-order integral estimate and the difference between the two rules.
fn error_estimate<F: Fn(f64) -> f64>(
    f: &F,
    fixed_rule_low: &fi::d1::FixedQuad,
    fixed_rule_high: &fi::d1::FixedQuad,
    bounds: (f64, f64),
) -> (f64, f64) {
    let integral_low = fixed_rule_low.integrate(f, Some(bounds));
    let integral_high = fixed_rule_high.integrate(f, Some(bounds));
    let err = (integral_high - integral_low).abs();
    (integral_low, err)
}
//}}}
//{{{ fun: adaptive_quad
/// Adaptively integrates `f` over [`AdaptiveQuadOpts::bounds`].
///
/// Given a real-valued function of one variable, this function returns an approximation of its
/// integral over the interval
/// \\[
///     I \approx  \int_{a}^{b} f(x) dx
/// \\]
///
/// The tolerance is checked per subinterval, so the returned aggregate error estimate may exceed
/// `opts.tol`. `max_depth` is validated but is not currently enforced.
///
/// # Errors
///
/// Returns [`OptionsError`] if `opts` is invalid.
///
/// # Returns
///
/// An [`AdaptiveQuadResult`] containing the approximation and diagnostics.
///
/// # Examples
///
/// ## Example 1
/// In this example, we integrate a polynomial over `[-3, 10]` using Gauss-Legendre rules of
/// order 10 and 30.
/// ```
///
/// use topohedral_integrate::{
///     adaptive_quad_1d, AdaptiveQuadOpts1D, FixedQuadOpts1D, GaussQuadType,
/// };
///
/// let f = |x: f64| 7.0 * x.powi(4) + 2.0 * x.powi(3) - 11.0 * x.powi(2) + 15.0 * x + 1.0;
/// let opts = AdaptiveQuadOpts1D {
///     bounds: (-3.0, 10.0),
///     fixed_rule_low: FixedQuadOpts1D {
///         gauss_type: GaussQuadType::Legendre,
///         order: 10,
///         bounds: (-1.0, 1.0),
///         subdiv: None,
///     },
///     fixed_rule_high: FixedQuadOpts1D {
///         gauss_type: GaussQuadType::Legendre,
///         order: 30,
///         bounds: (-1.0, 1.0),
///         subdiv:None,
///     },
///     tol: 1e-5,
///     max_depth: 1000,
///     init_subdiv: None,
///  };
/// let res = adaptive_quad_1d(&f, opts)?;
/// # Ok::<(), topohedral_integrate::OptionsError>(())
/// ```
#[allow(clippy::doc_overindented_list_items)]
pub fn adaptive_quad<F: Fn(f64) -> f64>(
    f: &F,
    opts: AdaptiveQuadOpts,
) -> Result<AdaptiveQuadResult, OptionsError> {
    opts.is_ok(true)?;

    //{{{ trace
    info!("opts: {:?}", opts);
    //}}}
    //{{{ init
    let AdaptiveQuadOpts {
        bounds,
        fixed_rule_low,
        fixed_rule_high,
        tol,
        max_depth: _,
        init_subdiv,
    } = opts;
    let fixed_rule_low = fi::d1::FixedQuad::new(fixed_rule_low)?;
    let fixed_rule_high = fi::d1::FixedQuad::new(fixed_rule_high)?;

    let non_val = -1.0f64;
    let mut intervals = Vec::<[f64; 4]>::new();
    match &init_subdiv {
        Some(subdiv) => {
            intervals.push([bounds.0, *subdiv.first().unwrap(), non_val, non_val]);
            for i in 0..subdiv.len() - 1 {
                intervals.push([subdiv[i], subdiv[i + 1], non_val, non_val]);
            }
            intervals.push([*subdiv.last().unwrap(), bounds.1, non_val, non_val]);
        }
        None => {
            intervals.push([bounds.0, bounds.1, non_val, non_val]);
        }
    }
    let mut has_converged = false;
    let mut marked = Vec::<usize>::with_capacity(100);
    let mut num_fn_eval = 0;
    let nqp = fixed_rule_low.nqp() + fixed_rule_high.nqp();
    //}}}
    //{{{ com: perform adaptive quadrature
    let mut iter = 0;
    while !has_converged {
        //{{{ trace
        debug!(
            "================================================== iter = {}",
            iter
        );
        //}}}
        //{{{ com: compute error estimates, mark intervals for splitting
        //{{{ trace
        debug!("Computing error estimates, marking intervals for splitting");
        //}}}
        for (i, interval) in intervals.iter_mut().enumerate() {
            //{{{ trace
            debug!("......................................");
            debug!("i = {} interval = {:?}", i, interval);
            //}}}
            let bounds = (interval[0], interval[1]);
            if interval[2] == non_val {
                let (integral, err_est) =
                    error_estimate(f, &fixed_rule_low, &fixed_rule_high, bounds);
                //{{{ trace
                debug!("integral = {}, err_est = {}", integral, err_est);
                //}}}
                num_fn_eval += nqp;
                interval[2] = integral;
                interval[3] = err_est;

                if err_est > tol {
                    //{{{ trace
                    debug!("pushing i = {} to marked", i);
                    //}}}
                    marked.push(i);
                }
            }
        }
        //}}}
        //{{{ com: split marked intervals
        //{{{ trace
        debug!("Splitting marked intervals");
        //}}}
        if marked.is_empty() {
            //{{{ trace
            debug!("marked is empty, convergence has been reached");
            //}}}
            has_converged = true;
        } else {
            for j in &marked {
                //{{{ trace
                debug!("........................");
                debug!("j = {}", j);
                //}}}
                //{{{ com: find mid point
                let interval = intervals[*j];
                let old_low = interval[0];
                let old_high = interval[1];
                let mid = (old_high + old_low) / 2.0;
                //{{{ trace
                debug!("interval = {:?} mid = {}", interval, mid);
                //}}}
                //}}}
                //{{{ com: set new bounds on left half
                intervals[*j][1] = mid;
                intervals[*j][2] = non_val;
                intervals[*j][3] = non_val;
                //}}}
                //{{{ com: set new bounds on right half
                let new_interval = [mid, old_high, non_val, non_val];
                //{{{ trace
                debug!("new_interval = {:?}", new_interval);
                //}}}
                intervals.push(new_interval);
                //}}}
            }
            marked.clear();
        }
        //}}}
        iter += 1;
    }
    //}}}
    //{{{ com: sum up integrals and errors
    let mut integral = 0.0;
    let mut err_est = 0.0;
    for interval in &intervals {
        integral += interval[2];
        err_est += interval[3];
    }
    //}}}
    //{{{ ret
    Ok(AdaptiveQuadResult {
        integral,
        error_estimate: err_est,
        num_subdiv: intervals.len(),
        num_fn_eval,
    })
    //}}}
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
//{{{ note
// The true integrals used in these tests were computed using the sympy package in the file
// assets/adaptive-integrals-1d.py.
//}}}

#[cfg(test)]
mod tests {}
//}}}
