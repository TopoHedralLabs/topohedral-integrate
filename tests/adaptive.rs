#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]
mod d1_tests {

    //! Some simple guidelines for these unit tests:
    //!
    //! * `assert_abs_diff_eq!` is used for comparing floats.
    //! * We assert the number of divisions and number of function evaluations not because these
    //!   are correct but because we wish to detect changes in behaviour. If the behaviour changes
    //!   in a way that you can justify then change the expected values.

    use approx::assert_abs_diff_eq;
    use topohedral_integrate::adaptive::d1;
    use topohedral_integrate::fixed::d1::{
        FixedQuad as FixedQuadD1, FixedQuadOpts as FixedQuadOptsD1,
    };
    use topohedral_integrate::gauss::GaussQuadType;
    use topohedral_integrate::OptionsStruct;

    /// Test to check that the options struct finds errors
    #[test]
    fn test_adaptive_quad_opts() {
        let opts = d1::AdaptiveQuadOpts {
            bounds: (1.0, 0.0),
            fixed_rule_low: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 30,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 10,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            tol: -1e-5,
            max_depth: 0,
            init_subdiv: None,
        };

        let is_ok = opts.is_ok(true);
        assert!(is_ok.is_err());
        match is_ok {
            Ok(_) => panic!("Expected error"),
            Err(err) => {
                assert_eq!(
                    err.to_string(),
                    "The options are invalid with reasons:\
                    \n\tBounds invalid, low bound greater than high bound\
                    \n\tGauss rule order mismatch, low order greater than high order\
                    \n\tTolerance invalid, must be positive\
                    \n\tMaximum number of subdivisions invalid, must be positive"
                );
            }
        }
    }

    /// Simple smooth polynomial function, should be integrated exactly to machine precision with
    /// a single interval.
    #[test]
    fn test_adaptive_quad_1() {
        let f = |x: f64| 7.0 * x.powi(4) + 2.0 * x.powi(3) - 11.0 * x.powi(2) + 15.0 * x + 1.0;
        let opts = d1::AdaptiveQuadOpts {
            bounds: (-3.0, 10.0),
            fixed_rule_low: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 10,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 30,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            tol: 1e-5,
            max_depth: 1000,
            init_subdiv: None,
        };

        let res = d1::adaptive_quad(&f, &opts);

        let true_integral = 2133443.0 / 15.0;
        let err_ub = (res.num_subdiv as f64) * opts.tol;
        assert_abs_diff_eq!(res.integral, true_integral, epsilon = err_ub);
        assert!(res.error_estimate < err_ub);
        assert_eq!(res.num_subdiv, 1);
        assert_eq!(res.num_fn_eval, 20);
    }

    /// Smooth but highly oscillatory function, should be integrated to tolerance with a small
    /// nunber of intervals.
    #[test]
    fn test_adaptive_quad_2() {
        let f = |x: f64| x.sin();
        let opts = d1::AdaptiveQuadOpts {
            bounds: (0.0, 30.0),
            fixed_rule_low: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 10,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 30,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            tol: 1e-5,
            max_depth: 1000,
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
    fn test_adaptive_quad_3() {
        let f = |x: f64| (x + 1.0).abs();
        let mut opts = d1::AdaptiveQuadOpts {
            bounds: (-3.0, 4.0),
            fixed_rule_low: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 10,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 30,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            tol: 1e-5,
            max_depth: 1000,
            init_subdiv: None,
        };

        let res1 = d1::adaptive_quad(&f, &opts);

        let true_integral = 29.0 / 2.0;

        // no init_subdiv
        {
            let err_ub = res1.num_subdiv as f64 * opts.tol;
            assert_abs_diff_eq!(res1.integral, true_integral, epsilon = err_ub);
            assert!(res1.error_estimate < err_ub);
            assert_eq!(res1.num_subdiv, 8);
            assert_eq!(res1.num_fn_eval, 300);
        }

        opts.init_subdiv = Some(vec![-1.0]);
        let res2 = d1::adaptive_quad(&f, &opts);

        // with init_subdiv
        {
            let err_ub = res2.num_subdiv as f64 * opts.tol;
            assert_abs_diff_eq!(res1.integral, true_integral, epsilon = err_ub);
            assert!(res2.error_estimate < err_ub);
            assert_eq!(res2.num_subdiv, 2);
            assert_eq!(res2.num_fn_eval, 40);
        }
    }

    #[test]
    fn test_adaptive_quad_4() {
        let f = |x: f64| (-x.powi(2)).exp();
        let opts = d1::AdaptiveQuadOpts {
            bounds: (-3.0, 3.0),
            fixed_rule_low: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 10,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 30,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            tol: 1e-5,
            max_depth: 1000,
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
    fn test_adaptive_quad_5() {
        let f = |x: f64| x.ln();
        let opts = d1::AdaptiveQuadOpts {
            bounds: (0.0, 10.0),
            fixed_rule_low: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 10,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD1::new(&FixedQuadOptsD1 {
                gauss_type: GaussQuadType::Legendre,
                order: 30,
                bounds: (-1.0, 1.0),
                subdiv: None,
            }),
            tol: 1e-5,
            max_depth: 1000,
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

mod d2_tests {

    //! Some notes on the tests:
    //! - We use absolute difference using the error upper bound as that is the guarantee
    //!   that this function provides. We do this rather than use a relative error.
    //!
    //!

    use approx::assert_abs_diff_eq;
    use topohedral_integrate::adaptive::d2;
    use topohedral_integrate::fixed::d2::{
        FixedQuad as FixedQuadD2, FixedQuadOpts as FixedQuadOptsD2,
    };
    use topohedral_integrate::gauss::GaussQuadType;
    use topohedral_integrate::OptionsStruct;

    #[test]
    fn test_adaptive_quad_opts() {
        let opts = d2::AdaptiveQuadOpts {
            bounds: (1.0, 0.0, 1.0, 0.0),
            fixed_rule_low: FixedQuadD2::new(&FixedQuadOptsD2 {
                gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                order: (30, 30),
                bounds: (-1.0, 1.0, -1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD2::new(&FixedQuadOptsD2 {
                gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                order: (10, 10),
                bounds: (-1.0, 1.0, -1.0, 1.0),
                subdiv: None,
            }),
            tol: -1e-5,
            max_depth: (0, 0),
            init_subdiv: None,
        };

        let is_ok = opts.is_ok(true);
        assert!(is_ok.is_err());
        match is_ok {
            Ok(_) => panic!("Expected error"),
            Err(err) => {
                assert_eq!(
                    err.to_string(),
                    "The options are invalid with reasons:\
                    \n\tBounds invalid, low bound greater than high bound\
                    \n\tGauss rule order mismatch, low order greater than high order\
                    \n\tTolerance invalid, must be positive\
                    \n\tMaximum number of subdivisions invalid, must be positive"
                );
            }
        }
    }

    #[test]
    fn test_adaptive_quad_1() {
        let f = |x: f64, y: f64| {
            0.3 * x.powi(4) * y.powi(4) + 2.0 * x.powi(3) * y.powi(3) - 0.1 * x.powi(2) * y.powi(2)
                + 100.0 * x * y
                + 200.0
        };

        let opts = d2::AdaptiveQuadOpts {
            bounds: (-0.3, 5.0, -3.0, 2.0),
            fixed_rule_low: FixedQuadD2::new(&FixedQuadOptsD2 {
                gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                order: (10, 10),
                bounds: (-1.0, 1.0, -1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD2::new(&FixedQuadOptsD2 {
                gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                order: (30, 30),
                bounds: (-1.0, 1.0, -1.0, 1.0),
                subdiv: None,
            }),
            tol: 1e-5,
            max_depth: (10, 10),
            init_subdiv: None,
        };

        let res = d2::adaptive_quad(&f, &opts);
        let true_integral = 7372.07722038889f64;
        let err_ub = (res.num_subdiv as f64) * opts.tol;
        assert_abs_diff_eq!(res.integral, true_integral, epsilon = err_ub);
        assert!(res.error_estimate < err_ub);
        assert_eq!(res.num_subdiv, 1);
        assert_eq!(res.num_fn_eval, 250);
    }

    #[test]
    fn test_adaptive_quad_2() {
        let f = |x: f64, y: f64| x.sin() * y.sin();

        let opts = d2::AdaptiveQuadOpts {
            bounds: (0.0, 30.0, 0.0, 30.0),
            fixed_rule_low: FixedQuadD2::new(&FixedQuadOptsD2 {
                gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                order: (10, 10),
                bounds: (-1.0, 1.0, -1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD2::new(&FixedQuadOptsD2 {
                gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                order: (30, 30),
                bounds: (-1.0, 1.0, -1.0, 1.0),
                subdiv: None,
            }),
            tol: 1e-5,
            max_depth: (10, 10),
            init_subdiv: None,
        };

        let res = d2::adaptive_quad(&f, &opts);
        let true_integral = -2.0 * 30.0f64.cos() + 30.0f64.cos().powi(2) + 1.0;

        let err_ub = (res.num_subdiv as f64) * opts.tol;
        assert_abs_diff_eq!(res.integral, true_integral, epsilon = err_ub);
        assert!(res.error_estimate < err_ub);
        assert_eq!(res.num_subdiv, 64);
        assert_eq!(res.num_fn_eval, 21250);
    }

    #[test]
    fn test_adaptive_quad_3() {
        let f = |x: f64, y: f64| (x + 1.0).abs() * (y - 2.0).abs();

        let mut opts = d2::AdaptiveQuadOpts {
            bounds: (-3.0, 4.0, 0.0, 5.0),
            fixed_rule_low: FixedQuadD2::new(&FixedQuadOptsD2 {
                gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                order: (10, 10),
                bounds: (-1.0, 1.0, -1.0, 1.0),
                subdiv: None,
            }),
            fixed_rule_high: FixedQuadD2::new(&FixedQuadOptsD2 {
                gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                order: (30, 30),
                bounds: (-1.0, 1.0, -1.0, 1.0),
                subdiv: None,
            }),
            tol: 1e-5,
            max_depth: (10, 10),
            init_subdiv: None,
        };

        let true_integral = 377.0 / 4.0;

        // no init_subdiv
        {
            let res = d2::adaptive_quad(&f, &opts);
            let err_ub = (res.num_subdiv as f64) * opts.tol;
            assert_abs_diff_eq!(res.integral, true_integral, epsilon = err_ub);
            assert!(res.error_estimate < err_ub);
            assert_eq!(res.num_subdiv, 403);
            assert_eq!(res.num_fn_eval, 134250);
        }
        // with init_subdiv
        {
            opts.init_subdiv = Some((vec![-1.0], vec![2.0]));
            let res = d2::adaptive_quad(&f, &opts);
            let err_ub = (res.num_subdiv as f64) * opts.tol;
            assert_abs_diff_eq!(res.integral, true_integral, epsilon = err_ub);
            assert!(res.error_estimate < err_ub);
            assert_eq!(res.num_subdiv, 4);
            assert_eq!(res.num_fn_eval, 1000);
        }
    }
}
