//{{{ mod: d1_tests
mod d1_tests {
    //{{{ collection: imports

    use approx::assert_relative_eq;
    use serde::Deserialize;
    use std::fs;
    use topohedral_integrate::{
        fixed_quad_1d as fixed_quad, ConfigIssue, FixedQuadOpts1D as FixedQuadOpts,
        FixedQuadrature1D as FixedQuad, GaussFamily, OptionsError, RuleAxis,
    };

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
        let opts = FixedQuadOpts::new(GaussFamily::Legendre, 101, (1.0, 0.0))
            .with_subdivisions(Vec::<f64>::new());

        let OptionsError::Config(error) = FixedQuad::new(opts).unwrap_err() else {
            panic!("expected structured configuration error");
        };
        assert!(matches!(
            error.issues(),
            [
                ConfigIssue::UnsupportedRuleDegree {
                    axis: RuleAxis::OneDimensional,
                    family: GaussFamily::Legendre,
                    degree: 101,
                    maximum: 100,
                },
                ConfigIssue::InvalidIntervalOrder {
                    lower: 1.0,
                    upper: 0.0,
                }
            ]
        ));
        assert_eq!(
            error.to_string(),
            "invalid configuration; one-dimensional legendre rule degree 101 exceeds supported maximum 100; interval bounds must satisfy lower < upper, received (1, 0)"
        );
    }

    #[test]
    fn test_fixed_quad_stores_opts() {
        let rule = FixedQuad::new(
            FixedQuadOpts::new(GaussFamily::Legendre, 3, (-1.0, 1.0)).with_subdivisions([0.0]),
        )
        .unwrap();

        assert_eq!(rule.rule().exactness().value(), 3);
        assert_eq!(rule.domain().bounds(), (-1.0, 1.0));
        assert_eq!(rule.subdivision_points(), &[0.0]);
    }

    #[test]
    fn one_point_legendre_rule_integrates_linear_function() {
        let integral = fixed_quad(
            &|x: f64| x,
            FixedQuadOpts::new(GaussFamily::Legendre, 0, (2.0, 5.0)),
        )
        .unwrap();

        assert_relative_eq!(integral, 10.5, epsilon = MAX_REL);
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
                    let opts = FixedQuadOpts::new(GaussFamily::Legendre, 2 * $nqp - 1, range);
                    let integral2 = fixed_quad(&pol, opts).unwrap();
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;

                    let opts = FixedQuadOpts::new(GaussFamily::Legendre, 2 * $nqp - 1, range)
                        .with_subdivisions([a, b]);
                    let integral2 = fixed_quad(&pol, opts).unwrap();
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
                    let opts = FixedQuadOpts::new(GaussFamily::Lobatto, 2 * $nqp - 3, range);
                    let integral2 = fixed_quad(&pol, opts).unwrap();
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;

                    let opts = FixedQuadOpts::new(GaussFamily::Lobatto, 2 * $nqp - 3, range)
                        .with_subdivisions([a, b]);
                    let integral2 = fixed_quad(&pol, opts).unwrap();
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
    use topohedral_integrate::{ConfigIssue, GaussFamily, OptionsError, RuleAxis};

    mod d2 {
        pub use topohedral_integrate::{
            fixed_quad_2d as fixed_quad, FixedQuadOpts2D as FixedQuadOpts,
            FixedQuadrature2D as FixedQuad,
        };
    }

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
        let opts = d2::FixedQuadOpts::new(
            (GaussFamily::Legendre, GaussFamily::Legendre),
            (101, 102),
            (2.0, 0.0, 2.0, 0.0),
        )
        .with_subdivisions(Vec::<f64>::new(), Vec::<f64>::new());

        let OptionsError::Config(error) = d2::FixedQuad::new(opts).unwrap_err() else {
            panic!("expected structured configuration error");
        };
        assert!(matches!(
            error.issues(),
            [
                ConfigIssue::UnsupportedRuleDegree {
                    axis: RuleAxis::U,
                    family: GaussFamily::Legendre,
                    degree: 101,
                    maximum: 100,
                },
                ConfigIssue::UnsupportedRuleDegree {
                    axis: RuleAxis::V,
                    family: GaussFamily::Legendre,
                    degree: 102,
                    maximum: 100,
                },
                ConfigIssue::InvalidIntervalOrder {
                    lower: 2.0,
                    upper: 0.0,
                },
                ConfigIssue::InvalidIntervalOrder {
                    lower: 2.0,
                    upper: 0.0,
                }
            ]
        ));
    }

    #[test]
    fn test_fixed_quad_opts2() {
        let opts = d2::FixedQuadOpts::new(
            (GaussFamily::Legendre, GaussFamily::Legendre),
            (3, 3),
            (1.0, 2.0, 1.0, 2.0),
        )
        .with_subdivisions([0.0], [0.0]);

        let OptionsError::Config(error) = d2::FixedQuad::new(opts).unwrap_err() else {
            panic!("expected structured configuration error");
        };
        assert!(matches!(
            error.issues(),
            [
                ConfigIssue::SubdivisionOutsideInterval {
                    index: 0,
                    value: 0.0,
                    lower: 1.0,
                    upper: 2.0,
                },
                ConfigIssue::SubdivisionOutsideInterval {
                    index: 0,
                    value: 0.0,
                    lower: 1.0,
                    upper: 2.0,
                }
            ]
        ));
    }

    #[test]
    fn test_fixed_quad_stores_opts() {
        let rule = d2::FixedQuad::new(
            d2::FixedQuadOpts::new(
                (GaussFamily::Legendre, GaussFamily::Lobatto),
                (3, 3),
                (-1.0, 1.0, -2.0, 2.0),
            )
            .with_subdivisions([0.0], [1.0]),
        )
        .unwrap();

        assert_eq!(rule.rule().u().exactness().value(), 3);
        assert_eq!(rule.rule().v().exactness().value(), 3);
        assert_eq!(rule.domain().u().bounds(), (-1.0, 1.0));
        assert_eq!(rule.domain().v().bounds(), (-2.0, 2.0));
        assert_eq!(rule.u_subdivision_points(), &[0.0]);
        assert_eq!(rule.v_subdivision_points(), &[1.0]);
    }

    #[test]
    fn one_point_legendre_rule_integrates_bilinear_function() {
        let integral = d2::fixed_quad(
            &|x: f64, y: f64| x + y,
            d2::FixedQuadOpts::new(
                (GaussFamily::Legendre, GaussFamily::Legendre),
                (0, 0),
                (1.0, 3.0, -2.0, 2.0),
            ),
        )
        .unwrap();

        assert_relative_eq!(integral, 16.0, epsilon = MAX_REL);
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
                    let opts = d2::FixedQuadOpts::new(
                        (GaussFamily::Legendre, GaussFamily::Legendre),
                        (2 * $nqp - 1, 2 * $nqp - 1),
                        range,
                    );
                    let integral2 = d2::fixed_quad(&pol, opts).unwrap();
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;
                    let dy = (range.3 - range.2) / 3.0;
                    let c = range.2 + dy;
                    let d = range.2 + 2.0 * dy;

                    let opts = d2::FixedQuadOpts::new(
                        (GaussFamily::Legendre, GaussFamily::Legendre),
                        (2 * $nqp - 1, 2 * $nqp - 1),
                        range,
                    )
                    .with_subdivisions([a, b], [c, d]);
                    let integral2 = d2::fixed_quad(&pol, opts).unwrap();

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
                    let opts = d2::FixedQuadOpts::new(
                        (GaussFamily::Legendre, GaussFamily::Legendre),
                        (2 * $nqp - 1, 2 * $nqp - 1),
                        range,
                    );
                    let integral2 = d2::fixed_quad(&pol, opts).unwrap();
                    assert_relative_eq!(integral1, integral2, max_relative = MAX_REL);
                }
                {
                    let dx = (range.1 - range.0) / 3.0;
                    let a = range.0 + dx;
                    let b = range.0 + 2.0 * dx;
                    let dy = (range.3 - range.2) / 3.0;
                    let c = range.2 + dy;
                    let d = range.2 + 2.0 * dy;

                    let opts = d2::FixedQuadOpts::new(
                        (GaussFamily::Lobatto, GaussFamily::Lobatto),
                        (2 * $nqp - 1, 2 * $nqp - 1),
                        range,
                    )
                    .with_subdivisions([a, b], [c, d]);
                    let integral2 = d2::fixed_quad(&pol, opts).unwrap();
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
