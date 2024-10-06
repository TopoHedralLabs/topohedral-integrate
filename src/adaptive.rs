//! This module contains the implementation of adaptive quadrature rules for both 1D and 2D. 
//!
//! The entry point for the 1D adaptive quadrature algorithm is the `adaptive_quad` function, which 
//! resides in the `d1` module along with its options and results structs.
//! 
//! The entry point
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::gauss::GaussQuadType;
use crate::fixed as fi;
use crate::common::{OptionsStruct, OptionsError, append_reason};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: d1
pub mod d1 {
    //! This module contains the implementation of adaptive quadrature rules for one-dimensional
    //! real-valued functions.



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
    /// 
    /// Additionally the `init_subdiv` field can be used to provide an initial set of subdivisions, 
    /// this can be useful if the user possesses some prior knowledge about the function, such 
    /// as the location of any singularities and discontinuities. Generally, if subdivisions 
    /// contain only smooth regions, the algorithm will converge quickly. 
    #[derive(Debug)]
    pub struct AdaptiveQuadOpts {
        /// bounds of the integral
        pub bounds: (f64, f64),
        /// Low-order Gauss quadrature rule
        pub fixed_rule_low: fi::d1::FixedQuad,
        /// High-order Gauss quadrature rule
        pub fixed_rule_high: fi::d1::FixedQuad,
        /// exit-tolerance for the integral
        pub tol: f64,
        /// Maximum number of subdivisions
        pub max_subdiv: usize,
        /// Optional initial subdivisions, provided as a set of stricty increasing values inside the 
        /// range provided by `bounds`. Do not include the bounds themselves.
        pub init_subdiv: Option<Vec<f64>>,
    }
    //}}}
    //{{{ impl: OptionsStruct for AdaptiveQuadOpts  
    impl OptionsStruct for AdaptiveQuadOpts {
        fn is_ok(&self, full: bool) -> Result<(), OptionsError> {

            let mut ok = true;

            let mut err = if full {
                OptionsError::InvalidOptionsFull(String::new())
            } 
            else {
                OptionsError::InvalidOptionsShort
            };

            if self.bounds.0 >= self.bounds.1 {
                append_reason(&mut err, "Bounds invalid, low bound greater than high bound");    
                ok = false;
            }

            if self.fixed_rule_low.gauss_type != GaussQuadType::Legendre{
                append_reason(&mut err, "Gauss rule type invalid, must be Gauss-Legendre");
                ok = false;
            }

            if self.fixed_rule_low.nqp() >= self.fixed_rule_high.nqp() {
                append_reason(&mut err, "Gauss rule order mismatch, low order greater than high order");
                ok = false;
            }

            if self.tol <= 0.0 {
                append_reason(&mut err, "Tolerance invalid, must be positive");
                ok = false;
            }

            if self.max_subdiv == 0 {
                append_reason(&mut err, "Maximum number of subdivisions invalid, must be positive");
                ok = false;
            }

            match self.init_subdiv {
                Some(ref v) => {
                    if v.len() == 0 {
                        append_reason(&mut err, "Initial subdivisions invalid, must be non-empty");
                        ok = false
                    }
                    for i in 0..v.len() {
                        if v[i] <= self.bounds.0 || v[i] >= self.bounds.1 {
                            append_reason(&mut err, "Initial subdivisions invalid, must be inside bounds");
                            ok = false;
                            break;
                        }
                    }
                },
                None => {}
            }


            let out = if ok {
                Ok(())
            } else {
                Err(err)
            };  
            out
        }
    }
    //}}}
    //{{{ struct: AdaptiveQuadResult
    /// The result of the adaptive quadrature algorithm, including diagnostic information such as 
    /// the number of subdivisions, the error estimate and the number of function evaluations.
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
    /// Computes the error estimate for the integral of a function $f$
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
    /// Performs adaptive quadrature integration on the given function `f` using the options 
    /// specified in `opts`. 
    /// 
    /// Given a real-valued function of single variable $f(x): \mathbb{R} \rightarrow \mathbb{R}$ this 
    /// function will return an approximation of the integral of the function over the interval
    /// \\[
    ///     I \approx  \int_{a}^{b} f(x) dx
    /// \\]
    /// 
    /// # Parameters
    /// - `f`: Real-valued function of single variable $f(x)$. Note that the function is of type 
    ///   `Fn(f64) -> f64` therefore it must not alter its internal state when called.
    /// - `AdaptiveQuadOpts`: this struct contains the necessary configuration for the adaptive 
    ///    quadrature algorithm, including the integration bounds, the Gauss quadrature rules to 
    ///    use, the error tolerance, and the maximum number of subdivisions.
    /// 
    /// # Returns 
    ///
    /// The `AdaptiveQuadResult` struct, which contains the result of the integration, including the 
    /// integral value, the error estimate, the number of subdivisions, and the number of function 
    /// evaluations.
    /// 
    /// # Examples
    /// 
    /// ## Example 1
    /// In this example we integrate the function $f(x) = 7x^4 - 2x^3 - 11x^2 + 15x + 1$ over the 
    /// inteval $[-3, 10]$. We use Gauss-Legendre qadrature rules of order 10 and 30, respectively.
    /// ```
    /// use topohedral_integrate::fixed as fi;
    /// use topohedral_integrate::adaptive::d1; 
    /// use topohedral_integrate::gauss::{GaussQuad, GaussQuadType};
    /// 
    /// let f =  |x: f64| 7.0 * x.powi(4) + 2.0 * x.powi(3) - 11.0 * x.powi(2) + 15.0 * x + 1.0;
    /// let opts = d1::AdaptiveQuadOpts {
    ///     bounds: (-3.0, 10.0), 
    ///         fixed_rule_low: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
    ///         gauss_type: GaussQuadType::Legendre,
    ///         order: 10, 
    ///         bounds: (-1.0, 1.0), 
    ///         subdiv: None,    
    ///     }),
    ///     fixed_rule_high: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
    ///         gauss_type: GaussQuadType::Legendre,
    ///         order: 30, 
    ///         bounds: (-1.0, 1.0), 
    ///         subdiv:None,    
    ///     }),
    ///     tol: 1e-5,
    ///     max_subdiv: 1000,
    ///     init_subdiv: None,
    ///  };
    /// let res = d1::adaptive_quad(&f, &opts);
    /// ```
    pub fn adaptive_quad<F: Fn(f64) -> f64>(f: &F, opts: &AdaptiveQuadOpts) -> AdaptiveQuadResult {
        //{{{ trace
        info!("opts: {:?}", opts);
        //}}}
        //{{{ init 
        let non_val = -1.0f64;
        let mut intervals = Vec::<[f64; 4]>::new();
        match &opts.init_subdiv {
            Some(subdiv) => {
                intervals.push([opts.bounds.0, *subdiv.first().unwrap(), non_val, non_val]);
                for i in 0..subdiv.len() - 1 {
                    intervals.push([subdiv[i], subdiv[i + 1], non_val, non_val]);
                }
                intervals.push([*subdiv.last().unwrap(), opts.bounds.1, non_val, non_val]);
            }
            None => {
                intervals.push([opts.bounds.0, opts.bounds.1, non_val, non_val]);
            }
        }
        let mut has_converged = false;
        let mut marked = Vec::<usize>::new();
        marked.reserve(100);
        let mut num_fn_eval = 0;
        let nqp = opts.fixed_rule_low.nqp() + opts.fixed_rule_high.nqp();
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
                        error_estimate(f, &opts.fixed_rule_low, &opts.fixed_rule_high, bounds);
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
    //! Some simple guidelines for these unit tests:
    //! 
    //! * `assert_abs_diff_eq!` is used for comparing floats.
    //! * We assert the number of divisions and number of function evaluations not because these 
    //!   are correct but because we wish to detect changes in behaviour. If the behaviour changes 
    //!   in a way that you can justify then change the expected values.

    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};


    /// Test to check that the options struct finds errors
    #[test]
    fn test_adaptive_quad_opts_1d() {
        let opts = d1::AdaptiveQuadOpts {
            bounds: (1.0, 0.0),
            fixed_rule_low: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 30, 
                bounds: (-1.0, 1.0), 
                subdiv: None,    
            }),
            fixed_rule_high: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 10, 
                bounds: (-1.0, 1.0), 
                subdiv:None,    
            }),
            tol: -1e-5,
            max_subdiv: 0,
            init_subdiv: None,
        };

        let is_ok = opts.is_ok(true);
        assert!(is_ok.is_err());
        match is_ok {
            Ok(_) => panic!("Expected error"),
            Err(err) => {
                assert_eq!(err.to_string(), 
                "The options are invalid with reasons:\
	             \n\tBounds invalid, low bound greater than high bound\
	             \n\tGauss rule order mismatch, low order greater than high order\
	             \n\tTolerance invalid, must be positive\
	             \n\tMaximum number of subdivisions invalid, must be positive");
            }
        }



    }

    /// Simple smooth polynomial function, should be integrated exactly to machine precision with 
    /// a single interval.
    #[test]
    fn test_adaptive_quad_1d_1() {

        let f =  |x: f64| 7.0 * x.powi(4) + 2.0 * x.powi(3) - 11.0 * x.powi(2) + 15.0 * x + 1.0;
        let opts = d1::AdaptiveQuadOpts {
            bounds: (-3.0, 10.0), 
            fixed_rule_low: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 10, 
                bounds: (-1.0, 1.0), 
                subdiv: None,    
            }),
            fixed_rule_high: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 30, 
                bounds: (-1.0, 1.0), 
                subdiv:None,    
            }),
            tol: 1e-5,
            max_subdiv: 1000,
            init_subdiv: None,
        };

        let res = d1::adaptive_quad(&f, &opts);

        let true_integral = 2133443.0 / 15.0;
        let err_ub = (res.num_subdiv as f64) * opts.tol;
        assert_relative_eq!(res.integral, true_integral, epsilon = 1e-9); 
        assert!(res.error_estimate < err_ub);
        assert_eq!(res.num_subdiv, 1);
        assert_eq!(res.num_fn_eval, 20);


    }

    /// Smooth but highly oscillatory function, should be integrated to tolerance with a small 
    /// nunber of intervals.
    #[test]
    fn test_adaptive_quad_1d_2() {
        let f = |x: f64| x.sin();
        let opts = d1::AdaptiveQuadOpts {
            bounds: (0.0, 30.0),
            fixed_rule_low: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 10, 
                bounds: (-1.0, 1.0), 
                subdiv: None,    
            }),
            fixed_rule_high: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 30, 
                bounds: (-1.0, 1.0), 
                subdiv:None,    
            }),
            tol: 1e-5,
            max_subdiv: 1000,
            init_subdiv: None,
        };
        let res = d1::adaptive_quad(&f, &opts);

        let true_integral = 1.0 - (30.0f64).cos(); 
        let err_ub = (res.num_subdiv as f64) * opts.tol;
        assert_abs_diff_eq!(res.integral, true_integral, epsilon = err_ub); 
        assert!(res.error_estimate < err_ub);
        assert_eq!(res.num_subdiv, 8);
        assert_eq!(res.num_fn_eval, 300);
    }

    /// Peicewise linear function with a discontinuity at x = -1.0.
    #[test]
    fn test_adaptive_quad_1d_3() {

        let f = |x: f64| (x + 1.0).abs();
        let mut opts = d1::AdaptiveQuadOpts {
            bounds: (-3.0, 4.0),
            fixed_rule_low: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 10, 
                bounds: (-1.0, 1.0), 
                subdiv: None,    
            }),
            fixed_rule_high: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 30, 
                bounds: (-1.0, 1.0), 
                subdiv:None,    
            }),
            tol: 1e-5,
            max_subdiv: 1000,
            init_subdiv: None,
        };

        let res1 = d1::adaptive_quad(&f, &opts);

        let true_integral = 29.0 / 2.0;

        // no init_subdiv
        {
            let err_ub = res1.num_subdiv as f64  * opts.tol;
            assert_abs_diff_eq!(res1.integral, true_integral, epsilon = err_ub);
            assert!(res1.error_estimate < err_ub);  
            assert_eq!(res1.num_subdiv,  8);
            assert_eq!(res1.num_fn_eval, 300);
        }

        opts.init_subdiv = Some(vec![-1.0]);
        let res2 = d1::adaptive_quad(&f, &opts);

        // with init_subdiv
        {
            let err_ub = res2.num_subdiv as f64  * opts.tol;
            assert_abs_diff_eq!(res1.integral, true_integral, epsilon = err_ub);
            assert!(res2.error_estimate < err_ub);  
            assert_eq!(res2.num_subdiv,  2);
            assert_eq!(res2.num_fn_eval, 40);
        }
    }

    #[test]
    fn test_adaptive_quad_1d_4() {

        let f = |x: f64| (-x.powi(2)).exp();
        let opts = d1::AdaptiveQuadOpts {
            bounds: (-3.0, 3.0),
            fixed_rule_low: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 10, 
                bounds: (-1.0, 1.0), 
                subdiv: None,    
            }),
            fixed_rule_high: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 30, 
                bounds: (-1.0, 1.0), 
                subdiv:None,    
            }),
            tol: 1e-5,
            max_subdiv: 1000,
            init_subdiv: None,
        };

        let res = d1::adaptive_quad(&f, &opts); 

        // sqrt(pi) * erf(3)
        let true_integral = 1.77241469651904;
        let err_ub = (res.num_subdiv as f64) * opts.tol;
        assert_abs_diff_eq!(res.integral, true_integral, epsilon = err_ub); 
        assert!(res.error_estimate < err_ub);
        assert_eq!(res.num_subdiv, 4);
        assert_eq!(res.num_fn_eval, 140);

    }

    /// Logarithmic function, which is smooth on R+ but with a singular point at x = 0.0.
    #[test]
    fn test_adaptive_quad_1d_5() {

        let f = |x: f64| x.ln();
        let opts = d1::AdaptiveQuadOpts {
            bounds: (0.0, 10.0),
            fixed_rule_low: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 10, 
                bounds: (-1.0, 1.0), 
                subdiv: None,    
            }),
            fixed_rule_high: fi::d1::FixedQuad::new(&fi::d1::FixedQuadOpts {
                gauss_type: GaussQuadType::Legendre,
                order: 30, 
                bounds: (-1.0, 1.0), 
                subdiv:None,    
            }),
            tol: 1e-5,
            max_subdiv: 1000,
            init_subdiv: None,
        };

        let res = d1::adaptive_quad(&f, &opts); 

        let true_integral = -10.0 + 10.0 * 10.0f64.ln();
        let err_ub = (res.num_subdiv as f64) * opts.tol;
        assert_abs_diff_eq!(res.integral, true_integral, epsilon = err_ub); 
        assert!(res.error_estimate < err_ub);
        assert_eq!(res.num_subdiv, 16);
        assert_eq!(res.num_fn_eval, 620);
    }

    
}
//}}}
