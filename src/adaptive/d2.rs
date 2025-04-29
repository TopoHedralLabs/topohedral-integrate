//! This module contains the implementation of adaptive quadrature rules for two-dimensional
//! real-valued functions.

//{{{ crate imports
use crate::common::{append_reason, OptionsError, OptionsStruct};
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
#[derive(Debug)]
pub struct AdaptiveQuadOpts {
    pub bounds: (f64, f64, f64, f64),
    pub fixed_rule_low: fi::d2::FixedQuad,
    pub fixed_rule_high: fi::d2::FixedQuad,
    pub tol: f64,
    pub max_depth: (usize, usize),
    pub init_subdiv: Option<(Vec<f64>, Vec<f64>)>,
}
//}}}
//{{{ impl: OptionsStruct for AdaptiveQuadOpts
impl OptionsStruct for AdaptiveQuadOpts {
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

        if self.fixed_rule_low.order >= self.fixed_rule_high.order {
            ok = false;
            append_reason(
                &mut err,
                "Gauss rule order mismatch, low order greater than high order",
            );
        }

        if self.tol < 0.0 {
            ok = false;
            append_reason(&mut err, "Tolerance invalid, must be positive");
        }

        if self.max_depth == (0, 0) {
            ok = false;
            append_reason(
                &mut err,
                "Maximum number of subdivisions invalid, must be positive",
            );
        }

        if let Some(subdiv) = &self.init_subdiv {
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
//{{{ struct: AdaptiveQuadResult
#[derive(Debug)]
pub struct AdaptiveQuadResult {
    pub integral: f64,
    pub error_estimate: f64,
    pub num_subdiv: usize,
    pub num_fn_eval: usize,
}
//}}}
//{{{ fun: error_estimate
/// Computes the error estimate for the integral of a function $f$
fn error_estimate<F: Fn(f64, f64) -> f64>(
    f: &F,
    fixed_rule_low: &fi::d2::FixedQuad,
    fixed_rule_high: &fi::d2::FixedQuad,
    bounds: (f64, f64, f64, f64),
) -> (f64, f64) {
    let integral_low = fixed_rule_low.integrate(f, Some(bounds));
    let integral_high = fixed_rule_high.integrate(f, Some(bounds));
    let err = (integral_high - integral_low).abs();
    (integral_low, err)
}
//}}}
//{{{ fun: adaptive_quad
pub fn adaptive_quad<F: Fn(f64, f64) -> f64>(f: &F, opts: &AdaptiveQuadOpts) -> AdaptiveQuadResult {
    //{{{ trace
    info!("opts: {:?}", opts);
    //}}}
    //{{{ init
    let non_val = -1.0f64;
    let mut intervals = Vec::<[f64; 6]>::new();
    let mut intervals_u = Vec::<f64>::new();
    let mut intervals_v = Vec::<f64>::new();
    let mut has_converged = false;
    let mut marked = Vec::<usize>::with_capacity(100);
    let mut num_fn_eval = 0;
    let nqp = opts.fixed_rule_low.nqp() + opts.fixed_rule_high.nqp();
    //}}}
    //{{{ com: find the initial intervals in u and v
    match &opts.init_subdiv {
        Some(subdiv) => {
            intervals_u.push(opts.bounds.0);
            intervals_u.extend_from_slice(subdiv.0.as_slice());
            intervals_u.push(opts.bounds.1);

            intervals_v.push(opts.bounds.2);
            intervals_v.extend_from_slice(subdiv.1.as_slice());
            intervals_v.push(opts.bounds.3);
        }
        None => {
            intervals_u.push(opts.bounds.0);
            intervals_u.push(opts.bounds.1);

            intervals_v.push(opts.bounds.2);
            intervals_v.push(opts.bounds.3);
        }
    }
    //}}}
    //{{{ com: find the initial intevals as [ulow uhigh vlow vhigh] tuples
    for i in 0..intervals_u.len() - 1 {
        let ui = intervals_u[i];
        let ui_1 = intervals_u[i + 1];

        for j in 0..intervals_v.len() - 1 {
            let vi = intervals_v[j];
            let vi_1 = intervals_v[j + 1];

            intervals.push([ui, ui_1, vi, vi_1, non_val, non_val]);
        }
    }
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
        for (i, interval) in intervals.iter_mut().enumerate() {
            let bounds = (interval[0], interval[1], interval[2], interval[3]);
            if interval[4] == non_val {
                let (integral, err_est) =
                    error_estimate(f, &opts.fixed_rule_low, &opts.fixed_rule_high, bounds);
                //{{{ trace
                debug!("integral = {}, err_est = {}", integral, err_est);
                //}}}
                num_fn_eval += nqp;
                interval[4] = integral;
                interval[5] = err_est;

                if err_est > opts.tol {
                    marked.push(i);
                }
            }
        }
        //}}}
        //{{{ com: split marked intervals
        if marked.is_empty() {
            has_converged = true;
        } else {
            for j in &marked {
                let interval = intervals[*j];
                let ul = interval[0];
                let uh = interval[1];
                let vl = interval[2];
                let vh = interval[3];
                let mid_u = (uh + ul) / 2.0;
                let mid_v = (vh + vl) / 2.0;
                intervals[*j][1] = mid_u;
                intervals[*j][3] = mid_v;
                intervals[*j][4] = non_val;
                intervals[*j][5] = non_val;

                let new_interval_1 = [mid_u, uh, vl, mid_v, non_val, non_val];
                let new_interval_2 = [ul, mid_u, mid_v, vh, non_val, non_val];
                let new_interval_3 = [mid_u, uh, mid_v, vh, non_val, non_val];
                intervals.push(new_interval_1);
                intervals.push(new_interval_2);
                intervals.push(new_interval_3);
            }
            marked.clear();
        }
        //}}}
        iter += 1;
    }
    //}}}
    //{{{ com: sum up the integrals and errors
    let mut integral = 0.0;
    let mut err_est = 0.0;

    for interval in &intervals {
        integral += interval[4];
        err_est += interval[5];
    }
    //}}}
    //{{{ ret
    AdaptiveQuadResult {
        integral,
        error_estimate: err_est,
        num_subdiv: intervals.len(),
        num_fn_eval,
    }
    //}}}
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {}
//}}}
