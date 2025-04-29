//{{{ mod: d1_tests
mod d1_tests {
    //{{{ collection: imports

    use approx::assert_relative_eq;
    use serde::Deserialize;
    use std::fs;
    use topohedral_integrate::fixed::d1::*;
    use topohedral_integrate::gauss::GaussQuadType;
    use topohedral_integrate::OptionsStruct;

    const MAX_REL: f64 = 1e-14;
    //}}}
    //{{{ collection: test data
    #[derive(Deserialize)]
    struct PolyIntegralTestData3 {
        coeffs: Vec<f64>,
        integral: f64,
    }

    #[derive(Deserialize)]
    struct PolyIntegralTestData2 {
        range: (f64, f64),
        p0: PolyIntegralTestData3,
        p1: PolyIntegralTestData3,
        p2: PolyIntegralTestData3,
        p3: PolyIntegralTestData3,
        p4: PolyIntegralTestData3,
        p5: PolyIntegralTestData3,
        p6: PolyIntegralTestData3,
        p7: PolyIntegralTestData3,
        p8: PolyIntegralTestData3,
        p9: PolyIntegralTestData3,
    }

    #[derive(Deserialize)]
    struct PolyIntegralTestData1 {
        values: PolyIntegralTestData2,
    }

    impl PolyIntegralTestData1 {
        fn new() -> Self {
            let json_file =
                fs::read_to_string("assets/poly-integrals-1d.json").expect("Unable to read file");
            serde_json::from_str(&json_file).expect("Could not deserialize")
        }
    }
    //}}}
    //{{{ collection: misc tests
    #[test]
    fn test_fixed_quad_opts() {
        let opts = FixedQuadOpts {
            gauss_type: GaussQuadType::Legendre,
            order: 1,
            bounds: (1.0, 0.0),
            subdiv: Some(Vec::<f64>::new()),
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
                \n\tInitial subdivisions invalid, must be non-empty"
                );
            }
        }
    }
    //}}}
    //{{{ collection: legendre tests
    macro_rules! poly_integral_legendre_test {
        ($test_name: ident, $dataset: ident, $nqp: expr) => {
            #[test]
            fn $test_name() {
                let test_data = PolyIntegralTestData1::new();
                let coeffs = test_data.values.$dataset.coeffs;
                let integral1 = test_data.values.$dataset.integral;
                let range = test_data.values.range;

                let pol = |x: f64| {
                    let mut sum = 0.0;
                    for (i, c) in coeffs.iter().enumerate() {
                        sum += c * x.powi(i as i32);
                    }
                    sum
                };

                {
                    let opts = FixedQuadOpts {
                        gauss_type: GaussQuadType::Legendre,
                        order: 2 * $nqp - 1,
                        bounds: range,
                        subdiv: None,
                    };
                    let integral2 = fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;

                    let opts = FixedQuadOpts {
                        gauss_type: GaussQuadType::Legendre,
                        order: 2 * $nqp - 1,
                        bounds: range,
                        subdiv: vec![a, b].into(),
                    };
                    let integral2 = fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
            }
        };
    }

    // 2-point integrals
    poly_integral_legendre_test!(poly_integral_legendre_test1, p0, 2);
    poly_integral_legendre_test!(poly_integral_legendre_test2, p1, 2);
    poly_integral_legendre_test!(poly_integral_legendre_test3, p2, 2);
    poly_integral_legendre_test!(poly_integral_legendre_test4, p3, 2);
    // 3-point-integrals
    poly_integral_legendre_test!(poly_integral_legendre_test5, p0, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test6, p1, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test7, p2, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test8, p3, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test9, p4, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test10, p5, 3);
    // 4-point-integrals
    poly_integral_legendre_test!(poly_integral_legendre_test11, p0, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test12, p1, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test13, p2, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test14, p3, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test15, p4, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test16, p5, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test17, p6, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test18, p7, 4);
    // 5-point-integrals
    poly_integral_legendre_test!(poly_integral_legendre_test19, p0, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test20, p1, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test21, p2, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test22, p3, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test23, p4, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test24, p5, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test25, p6, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test26, p7, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test27, p8, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test28, p9, 5);
    // 6-point-integrals

    //}}}
    //{{{ collection: lobatto tests
    macro_rules! poly_integral_lobatto_test {
        ($test_name: ident, $dataset: ident, $nqp: expr) => {
            #[test]
            fn $test_name() {
                let test_data = PolyIntegralTestData1::new();
                let coeffs = test_data.values.$dataset.coeffs;
                let integral1 = test_data.values.$dataset.integral;
                let range = test_data.values.range;

                let pol = |x: f64| {
                    let mut sum = 0.0;
                    for (i, c) in coeffs.iter().enumerate() {
                        sum += c * x.powi(i as i32);
                    }
                    sum
                };

                {
                    let opts = FixedQuadOpts {
                        gauss_type: GaussQuadType::Lobatto,
                        order: 2 * $nqp - 3,
                        bounds: range,
                        subdiv: None,
                    };
                    let integral2 = fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;

                    let opts = FixedQuadOpts {
                        gauss_type: GaussQuadType::Lobatto,
                        order: 2 * $nqp - 3,
                        bounds: range,
                        subdiv: vec![a, b].into(),
                    };
                    let integral2 = fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
            }
        };
    }

    // 2-point integrals
    poly_integral_lobatto_test!(poly_integral_lobatto_test1, p0, 2);
    poly_integral_lobatto_test!(poly_integral_lobatto_test2, p1, 2);
    // 3-point-integrals
    poly_integral_lobatto_test!(poly_integral_lobatto_test5, p0, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test6, p1, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test7, p2, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test8, p3, 3);
    // 4-point-integrals
    poly_integral_lobatto_test!(poly_integral_lobatto_test11, p0, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test12, p1, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test13, p2, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test14, p3, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test15, p4, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test16, p5, 4);
    // 5-point-integrals
    poly_integral_lobatto_test!(poly_integral_lobatto_test19, p0, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test20, p1, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test21, p2, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test22, p3, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test23, p4, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test24, p5, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test25, p6, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test26, p7, 5);
    //}}}
}
//}}}
//{{{ mod: d2_tests
mod d2_tests {

    //{{{ collection: imports
    use approx::assert_relative_eq;
    use topohedral_integrate::fixed::d2;
    use topohedral_integrate::gauss::GaussQuadType;
    use topohedral_integrate::OptionsStruct;

    use serde::Deserialize;
    use std::fs;

    const MAX_REL: f64 = 1e-14;
    //}}}
    //{{{ collection: test data
    #[derive(Deserialize)]
    struct PolyIntegralTestData3 {
        coeffs_u: Vec<f64>,
        coeffs_v: Vec<f64>,
        integral: f64,
    }

    #[derive(Deserialize)]
    struct PolyIntegralTestData2 {
        range: (f64, f64, f64, f64),
        p0: PolyIntegralTestData3,
        p1: PolyIntegralTestData3,
        p2: PolyIntegralTestData3,
        p3: PolyIntegralTestData3,
        p4: PolyIntegralTestData3,
        p5: PolyIntegralTestData3,
        p6: PolyIntegralTestData3,
        p7: PolyIntegralTestData3,
        p8: PolyIntegralTestData3,
        p9: PolyIntegralTestData3,
    }

    #[derive(Deserialize)]
    struct PolyIntegralTestData1 {
        values: PolyIntegralTestData2,
    }

    impl PolyIntegralTestData1 {
        fn new() -> Self {
            let json_file =
                fs::read_to_string("assets/poly-integrals-2d.json").expect("Unable to read file");
            serde_json::from_str(&json_file).expect("Could not deserialize")
        }
    }
    //}}}
    //{{{ collection: misc tests
    #[test]
    fn test_fixed_quad_opts1() {
        let opts = d2::FixedQuadOpts {
            gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
            order: (2, 2),
            bounds: (2.0, 0.0, 2.0, 0.0),
            subdiv: Some((Vec::<f64>::new(), Vec::new())),
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
                    \n\tInitial subdivision invalid, at least 1 must be non-empty"
                );
            }
        }
    }

    #[test]
    fn test_fixed_quad_opts2() {
        let opts = d2::FixedQuadOpts {
            gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
            order: (2, 2),
            bounds: (1.0, 2.0, 1.0, 2.0),
            subdiv: Some((vec![0.0], vec![0.0])),
        };

        let is_ok = opts.is_ok(true);
        assert!(is_ok.is_err());
        match is_ok {
            Ok(_) => panic!("Expected error"),
            Err(err) => {
                assert_eq!(
                    err.to_string(),
                    "The options are invalid with reasons:\
                    \n\tInitial subdivision invalid, must be within bounds\
                    \n\tInitial subdivision invalid, must be within bounds"
                );
            }
        }
    }
    //}}}
    //{{{ collection: legendre tests
    macro_rules! poly_integral_legendre_test {
        ($test_name: ident, $dataset: ident, $nqp: expr) => {
            #[test]
            fn $test_name() {
                let test_data = PolyIntegralTestData1::new();
                let coeffs_u = test_data.values.$dataset.coeffs_u;
                let coeffs_v = test_data.values.$dataset.coeffs_v;
                let integral1 = test_data.values.$dataset.integral;
                let range = test_data.values.range;

                let pol = |x: f64, y: f64| {
                    let mut sum_u = 0.0;
                    for (i, c) in coeffs_u.iter().enumerate() {
                        sum_u += c * x.powi(i as i32);
                    }
                    let mut sum_v = 0.0;
                    for (i, c) in coeffs_v.iter().enumerate() {
                        sum_v += c * y.powi(i as i32);
                    }
                    sum_u * sum_v
                };

                {
                    let opts = d2::FixedQuadOpts {
                        gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                        order: (2 * $nqp - 1, 2 * $nqp - 1),
                        bounds: range,
                        subdiv: None,
                    };
                    let integral2 = d2::fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;
                    let dy = (range.3 - range.2) / 3.0;
                    let c = range.2 + dy;
                    let d = range.2 + 2.0 * dy;

                    let opts = d2::FixedQuadOpts {
                        gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                        order: (2 * $nqp - 1, 2 * $nqp - 1),
                        bounds: range,
                        subdiv: Some((vec![a, b], vec![c, d])),
                    };
                    let integral2 = d2::fixed_quad(&pol, &opts);

                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
            }
        };
    }

    // 2-point integrals
    poly_integral_legendre_test!(poly_integral_legendre_test1, p0, 2);
    poly_integral_legendre_test!(poly_integral_legendre_test2, p1, 2);
    poly_integral_legendre_test!(poly_integral_legendre_test3, p2, 2);
    poly_integral_legendre_test!(poly_integral_legendre_test4, p3, 2);
    // 3-point-integrals
    poly_integral_legendre_test!(poly_integral_legendre_test5, p0, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test6, p1, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test7, p2, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test8, p3, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test9, p4, 3);
    poly_integral_legendre_test!(poly_integral_legendre_test10, p5, 3);
    // 4-point-integrals
    poly_integral_legendre_test!(poly_integral_legendre_test11, p0, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test12, p1, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test13, p2, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test14, p3, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test15, p4, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test16, p5, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test17, p6, 4);
    poly_integral_legendre_test!(poly_integral_legendre_test18, p7, 4);
    // 5-point-integrals
    poly_integral_legendre_test!(poly_integral_legendre_test19, p0, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test20, p1, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test21, p2, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test22, p3, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test23, p4, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test24, p5, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test25, p6, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test26, p7, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test27, p8, 5);
    poly_integral_legendre_test!(poly_integral_legendre_test28, p9, 5);
    //}}}
    //{{{ collection: lobatto tests
    macro_rules! poly_integral_lobatto_test {
        ($test_name: ident, $dataset: ident, $nqp: expr) => {
            #[test]
            fn $test_name() {
                let test_data = PolyIntegralTestData1::new();
                let coeffs_u = test_data.values.$dataset.coeffs_u;
                let coeffs_v = test_data.values.$dataset.coeffs_v;
                let integral1 = test_data.values.$dataset.integral;
                let range = test_data.values.range;

                let pol = |x: f64, y: f64| {
                    let mut sum_u = 0.0;
                    for (i, c) in coeffs_u.iter().enumerate() {
                        sum_u += c * x.powi(i as i32);
                    }
                    let mut sum_v = 0.0;
                    for (i, c) in coeffs_v.iter().enumerate() {
                        sum_v += c * y.powi(i as i32);
                    }
                    sum_u * sum_v
                };

                {
                    let opts = d2::FixedQuadOpts {
                        gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
                        order: (2 * $nqp - 1, 2 * $nqp - 1),
                        bounds: range,
                        subdiv: None,
                    };
                    let integral2 = d2::fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;
                    let dy = (range.3 - range.2) / 3.0;
                    let c = range.2 + dy;
                    let d = range.2 + 2.0 * dy;

                    let opts = d2::FixedQuadOpts {
                        gauss_type: (GaussQuadType::Lobatto, GaussQuadType::Lobatto),
                        order: (2 * $nqp - 1, 2 * $nqp - 1),
                        bounds: range,
                        subdiv: Some((vec![a, b], vec![c, d])),
                    };
                    let integral2 = d2::fixed_quad(&pol, &opts);
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
            }
        };
    }

    poly_integral_lobatto_test!(poly_integral_lobatto_test1, p0, 2);
    poly_integral_lobatto_test!(poly_integral_lobatto_test2, p1, 2);
    poly_integral_lobatto_test!(poly_integral_lobatto_test3, p2, 2);
    poly_integral_lobatto_test!(poly_integral_lobatto_test4, p3, 2);
    poly_integral_lobatto_test!(poly_integral_lobatto_test5, p0, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test6, p1, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test7, p2, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test8, p3, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test9, p4, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test10, p5, 3);
    poly_integral_lobatto_test!(poly_integral_lobatto_test11, p0, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test12, p1, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test13, p2, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test14, p3, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test15, p4, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test16, p5, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test17, p6, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test18, p7, 4);
    poly_integral_lobatto_test!(poly_integral_lobatto_test19, p0, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test20, p1, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test21, p2, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test22, p3, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test23, p4, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test24, p5, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test25, p6, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test26, p7, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test27, p8, 5);
    poly_integral_lobatto_test!(poly_integral_lobatto_test28, p9, 5);
    //}}}
}
//}}}
