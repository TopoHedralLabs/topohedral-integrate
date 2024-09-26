//! This module contains the implementation of adaptive quadrature rules.
//!
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::gauss::{GaussQuad, GaussQuadType};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: d1
mod d1 {
    //!  This module contains the implementation of adaptive quadrature rules for one-dimensional
    //! real-valued functions.


    use core::num;

    use super::*;

    //{{{ struct: AdaptiveQuadOpts
    /// Configures the options for the adaptive quadrature algorithm.
    ///
    /// The `AdaptiveQuadOpts` struct holds the necessary parameters to control the
    /// behavior of the adaptive quadrature algorithm, such as the integration bounds,
    /// the Gauss quadrature rules to use, the error tolerance, and the maximum number
    /// of subdivisions.
    ///
    /// The `bounds` field specifies the interval over which the integration is performed.
    /// The `gauss_rule_low` and `gauss_rule_high` fields specify the low-order and
    /// high-order Gauss quadrature rules to use, respectively. The `tol` field sets the
    /// error tolerance for the integration, and the `max_depth` field sets the maximum
    /// number of subdivisions allowed.
    #[derive(Debug)]
    pub struct AdaptiveQuadOpts {
        /// bounds of the integral
        pub bounds: (f64, f64),
        /// Low-order Gauss quadrature rule
        pub gauss_rule_low: GaussQuad,
        /// High-order Gauss quadrature rule
        pub gauss_rule_high: GaussQuad,
        /// exit-tolerance for the integral
        pub tol: f64,
        /// Maximum number of subdivisions
        pub max_subdiv: usize,
    }
    //}}}
    //{{{ struct: AdaptiveQuadResult
    /// The result of the adaptive quadrature algorithm.
    #[derive(Debug)]
    pub struct AdaptiveQuadResult {
        /// integral value
        pub integral: f64,
        /// error estimate
        pub error_estimate: f64,
        /// number of subdivisions
        pub num_subdiv: usize,
        /// number of function evaluations
        pub num_fn_eval: usize,
    }
    //}}}
    //{{{ fun: error_estimate
    fn error_estimate<F: Fn(f64) -> f64>(
        f: &F,
        gauss_rule_low: &GaussQuad,
        gauss_rule_high: &GaussQuad,
        bounds: (f64, f64),
    ) -> (f64, f64) {
        let integral_low = gauss_rule_low.integrate(f, bounds);
        let integral_high = gauss_rule_high.integrate(f, bounds);
        let err = (integral_high - integral_low).abs();
        (integral_low, err)
    }
    //}}}
    //{{{ fun: adaptive_quad
    pub fn adaptive_quad<F: Fn(f64) -> f64>(f: &F, opts: &AdaptiveQuadOpts) -> AdaptiveQuadResult {
        //{{{ trace
        info!("opts: {:?}", opts);
        //}}}
        //{{{ init 
        let non_val = -1.0f64;
        let mut intervals = Vec::<[f64; 4]>::new();
        intervals.push([opts.bounds.0, opts.bounds.1, non_val, non_val]);
        let mut has_converged = false;
        let mut marked = Vec::<usize>::new();
        marked.reserve(100);
        let mut num_fn_eval = 0;
        let nqp = opts.gauss_rule_low.nqp + opts.gauss_rule_high.nqp;
        //}}}
        //{{{ com: perform adaptive quadrature
        let mut iter = 0;
        while !has_converged {
            //{{{ trace
            debug!("================================================== iter = {}", iter);
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
                        error_estimate(f, &opts.gauss_rule_low, &opts.gauss_rule_high, bounds);
                    //{{{ trace
                    debug!("integral = {}, err_est = {}", integral, err_est);
                    //}}}
                    num_fn_eval += nqp;
                    interval[2] = integral;
                    interval[3] = err_est;

                    if err_est > opts.tol {
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
            if marked.is_empty() 
            {
                //{{{ trace
                debug!("marked is empty, convergence has been reached");
                //}}}
                has_converged = true;
            } 
            else 
            {
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
        let out = AdaptiveQuadResult {
            integral: integral,
            error_estimate: err_est,
            num_subdiv: intervals.len(),
            num_fn_eval: num_fn_eval,
        };
        //{{{ trace
        info!("out = {:?}", out);
        //}}}
        out
        //}}}
    }
    //}}}
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {

    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq, ulps_eq, AbsDiff};


    #[test]
    fn test_adaptive_quad_1d_1() {

        let f =  |x: f64| 7.0 * x.powi(4) + 2.0 * x.powi(3) - 11.0 * x.powi(2) + 15.0 * x + 1.0;
        let opts = d1::AdaptiveQuadOpts {
            bounds: (-3.0, 10.0), 
            gauss_rule_low: GaussQuad::new(GaussQuadType::Legendre, 10),
            gauss_rule_high: GaussQuad::new(GaussQuadType::Legendre, 30), 
            tol: 1e-5,
            max_subdiv: 1000,
        };

        let res = d1::adaptive_quad(&f, &opts);

        let true_integral = 2133443.0 / 15.0;
        let err_ub = (res.num_subdiv as f64) * opts.tol;
        assert_relative_eq!(res.integral, true_integral, epsilon = 1e-9); 
        assert!(res.error_estimate < err_ub);
        assert_eq!(res.num_subdiv, 1);
        assert_eq!(res.num_fn_eval, 20);


    }

    #[test]
    fn test_adaptive_quad_1d_2() {
        let f = |x: f64| x.sin();
        let opts = d1::AdaptiveQuadOpts {
            bounds: (0.0, 30.0),
            gauss_rule_low: GaussQuad::new(GaussQuadType::Legendre, 10),
            gauss_rule_high: GaussQuad::new(GaussQuadType::Legendre, 30),
            tol: 1e-5,
            max_subdiv: 1000,
        };
        let res = d1::adaptive_quad(&f, &opts);

        let true_integral = 1.0 - (30.0f64).cos(); 
        let err_ub = (res.num_subdiv as f64) * opts.tol;
        assert_abs_diff_eq!(res.integral, true_integral, epsilon = err_ub); 
        assert!(res.error_estimate < err_ub);
        assert_eq!(res.num_subdiv, 8);
        assert_eq!(res.num_fn_eval, 300);
    }

    
}
//}}}
