use topohedral_integrate::{
    AxisDepths, ConfigIssue, FixedQuadrature1d, GaussFamily, GaussRule, Interval, PointCount,
    PolynomialDegree, Rectangle, RefinementDepth, Tolerance,
};

#[test]
fn degree_and_point_count_validate_boundaries() {
    assert_eq!(PolynomialDegree::new(0).unwrap().value(), 0);
    assert_eq!(
        PolynomialDegree::try_from(PolynomialDegree::MAXIMUM)
            .unwrap()
            .value(),
        101
    );
    assert!(matches!(
        PolynomialDegree::new(102).unwrap_err().issues(),
        [ConfigIssue::PolynomialDegreeOutOfRange {
            degree: 102,
            maximum: 101,
        }]
    ));

    assert_eq!(PointCount::new(PointCount::MINIMUM).unwrap().value(), 1);
    assert_eq!(
        PointCount::try_from(PointCount::MAXIMUM).unwrap().value(),
        52
    );
    for invalid in [0, 53, usize::MAX] {
        assert!(matches!(
            PointCount::new(invalid).unwrap_err().issues(),
            [ConfigIssue::PointCountOutOfRange { point_count, .. }]
                if *point_count == invalid
        ));
    }
}

#[test]
fn interval_requires_finite_strictly_increasing_bounds() {
    let interval = Interval::new(-2.0, 3.0).unwrap();
    assert_eq!(interval.lower(), -2.0);
    assert_eq!(interval.upper(), 3.0);
    assert_eq!(interval.bounds(), (-2.0, 3.0));
    assert_eq!(interval.length(), 5.0);
    assert_eq!(interval.to_string(), "[-2, 3]");

    for (lower, upper) in [
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
    ] {
        assert!(matches!(
            Interval::new(lower, upper).unwrap_err().issues(),
            [ConfigIssue::NonFiniteInterval { .. }]
        ));
    }

    for (lower, upper) in [(1.0, 1.0), (2.0, -2.0)] {
        assert!(matches!(
            Interval::new(lower, upper).unwrap_err().issues(),
            [ConfigIssue::InvalidIntervalOrder { .. }]
        ));
    }
}

#[test]
fn configuration_issues_retain_offending_values_and_use_concise_messages() {
    let nonfinite = Interval::new(f64::NAN, 2.0).unwrap_err();
    assert!(matches!(
        nonfinite.issues(),
        [ConfigIssue::NonFiniteInterval { lower, upper: 2.0 }] if lower.is_nan()
    ));
    assert_eq!(
        nonfinite.to_string(),
        "invalid configuration; interval bounds must be finite, received (NaN, 2)"
    );

    let tolerance = Tolerance::new(-1.0, f64::INFINITY).unwrap_err();
    assert!(matches!(
        tolerance.issues(),
        [ConfigIssue::InvalidTolerance {
            absolute: -1.0,
            relative,
        }] if relative.is_infinite()
    ));
    assert!(!tolerance.to_string().ends_with('.'));
}

#[test]
fn rectangle_contains_validated_axis_intervals() {
    let u = Interval::new(-1.0, 1.0).unwrap();
    let v = Interval::new(2.0, 5.0).unwrap();
    let rectangle = Rectangle::new(u, v);
    assert_eq!(rectangle.u(), u);
    assert_eq!(rectangle.v(), v);
    assert_eq!(rectangle.to_string(), "[-1, 1] × [2, 5]");

    let error = Rectangle::from_bounds(1.0, 1.0, f64::NAN, 2.0).unwrap_err();
    assert_eq!(error.issues().len(), 2);
    assert!(matches!(
        error.issues()[0],
        ConfigIssue::InvalidIntervalOrder { .. }
    ));
    assert!(matches!(
        error.issues()[1],
        ConfigIssue::NonFiniteInterval { .. }
    ));
}

#[test]
fn tolerance_requires_finite_nonnegative_nonzero_components() {
    let mixed = Tolerance::new(1e-8, 1e-6).unwrap();
    assert_eq!(mixed.absolute_value(), 1e-8);
    assert_eq!(mixed.relative_value(), 1e-6);

    let absolute = Tolerance::absolute(1e-8).unwrap();
    assert_eq!(absolute.absolute_value(), 1e-8);
    assert_eq!(absolute.relative_value(), 0.0);

    let relative = Tolerance::relative(1e-6).unwrap();
    assert_eq!(relative.absolute_value(), 0.0);
    assert_eq!(relative.relative_value(), 1e-6);

    for (absolute, relative) in [
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (f64::NAN, 1.0),
        (1.0, f64::INFINITY),
    ] {
        assert!(matches!(
            Tolerance::new(absolute, relative).unwrap_err().issues(),
            [ConfigIssue::InvalidTolerance { .. }]
        ));
    }
}

#[test]
fn refinement_depth_zero_is_valid_and_defaults_to_32() {
    assert_eq!(RefinementDepth::new(0).value(), 0);
    assert_eq!(RefinementDepth::default().value(), 32);

    let axes = AxisDepths::from_values(0, 7);
    assert_eq!(axes.u().value(), 0);
    assert_eq!(axes.v().value(), 7);

    let default_axes = AxisDepths::default();
    assert_eq!(default_axes.u().value(), 32);
    assert_eq!(default_axes.v().value(), 32);
}

#[test]
fn fixed_builder_uses_strict_subdivision_validation() {
    let domain = Interval::new(-1.0, 1.0).unwrap();
    let rule =
        GaussRule::for_degree(GaussFamily::Legendre, PolynomialDegree::new(3).unwrap()).unwrap();
    let make =
        |subdiv: Vec<f64>| FixedQuadrature1d::builder(domain, rule.clone()).subdivisions(subdiv);

    let without_subdivisions = FixedQuadrature1d::builder(domain, rule.clone()).build();
    let empty_subdivisions = make(Vec::new()).unwrap().build();
    assert_eq!(without_subdivisions.nodes(), empty_subdivisions.nodes());

    for invalid in [
        vec![f64::NAN],
        vec![0.0, 0.0],
        vec![0.5, -0.5],
        vec![-1.0],
        vec![1.0],
    ] {
        assert!(make(invalid).is_err());
    }
}
