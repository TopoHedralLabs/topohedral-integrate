use approx::assert_abs_diff_eq;
use topohedral_integrate::{
    adaptive_quad_1d, adaptive_quad_2d, AdaptiveQuadrature1d, AdaptiveQuadrature2d, AxisDepths,
    ConfigIssue, EvaluationPoint, GaussFamily, GaussRule, IntegrationError, Interval, OptionsError,
    PolynomialDegree, Rectangle, RefinementDepth, RuleAxis, TensorRule2d, Tolerance,
};

fn rule(degree: usize) -> GaussRule {
    GaussRule::for_degree(
        GaussFamily::Legendre,
        PolynomialDegree::new(degree).unwrap(),
    )
    .unwrap()
}

fn tensor_rule(degree: usize) -> TensorRule2d {
    let rule = rule(degree);
    TensorRule2d::new(rule.clone(), rule)
}

#[test]
fn one_dimensional_builder_validates_actual_rule_exactness() {
    let error = AdaptiveQuadrature1d::builder(
        Interval::new(-1.0, 1.0).unwrap(),
        rule(2),
        rule(3),
        Tolerance::absolute(1e-8).unwrap(),
    )
    .build()
    .unwrap_err();

    assert!(matches!(
        error.issues(),
        [ConfigIssue::NonIncreasingRuleExactness {
            axis: RuleAxis::OneDimensional,
            low: 3,
            high: 3,
        }]
    ));
}

#[test]
fn one_dimensional_free_function_builds_and_integrates_once() {
    let builder = AdaptiveQuadrature1d::builder(
        Interval::new(-1.0, 1.0).unwrap(),
        rule(3),
        rule(5),
        Tolerance::absolute(1e-12).unwrap(),
    );

    let result = adaptive_quad_1d(|x| x * x, builder).unwrap();
    assert_abs_diff_eq!(result.integral(), 2.0 / 3.0, epsilon = 1e-14);
    assert_eq!(result.evaluation_count(), 5);
}

#[test]
fn one_dimensional_free_function_retains_structured_configuration_error() {
    let builder = AdaptiveQuadrature1d::builder(
        Interval::new(-1.0, 1.0).unwrap(),
        rule(2),
        rule(3),
        Tolerance::absolute(1e-12).unwrap(),
    );

    let OptionsError::Config(error) = adaptive_quad_1d(|_| 0.0, builder).unwrap_err() else {
        panic!("expected structured configuration error");
    };
    assert!(matches!(
        error.issues(),
        [ConfigIssue::NonIncreasingRuleExactness {
            axis: RuleAxis::OneDimensional,
            low: 3,
            high: 3,
        }]
    ));
}

#[test]
fn one_dimensional_result_uses_high_rule_and_global_tolerance() {
    let tolerance = Tolerance::new(1e-10, 1e-10).unwrap();
    let quadrature = AdaptiveQuadrature1d::builder(
        Interval::new(0.0, 30.0).unwrap(),
        rule(5),
        rule(11),
        tolerance,
    )
    .max_depth(RefinementDepth::new(20))
    .build()
    .unwrap();

    let result = quadrature.integrate(f64::sin).unwrap();
    let expected = 1.0 - 30.0f64.cos();
    let limit = tolerance.absolute_value() + tolerance.relative_value() * result.integral().abs();
    assert_abs_diff_eq!(result.integral(), expected, epsilon = 1e-10);
    assert!(result.error_estimate() <= limit);
    assert!(result.terminal_region_count() > 1);
}

#[test]
fn one_dimensional_relative_only_tolerance_converges_globally() {
    let tolerance = Tolerance::relative(1e-10).unwrap();
    let quadrature = AdaptiveQuadrature1d::builder(
        Interval::new(-1.0, 1.0).unwrap(),
        rule(3),
        rule(7),
        tolerance,
    )
    .max_depth(RefinementDepth::new(20))
    .build()
    .unwrap();

    let result = quadrature.integrate(f64::exp).unwrap();
    let expected = std::f64::consts::E - 1.0 / std::f64::consts::E;
    assert_abs_diff_eq!(result.integral(), expected, epsilon = 1e-10);
    assert!(result.error_estimate() <= tolerance.relative_value() * result.integral().abs());
}

#[test]
fn one_dimensional_integrand_may_mutate_state() {
    let quadrature = AdaptiveQuadrature1d::builder(
        Interval::new(-1.0, 1.0).unwrap(),
        rule(3),
        rule(5),
        Tolerance::absolute(1e-12).unwrap(),
    )
    .build()
    .unwrap();
    let mut calls = 0;

    let result = quadrature
        .integrate(|x| {
            calls += 1;
            x * x
        })
        .unwrap();

    assert_abs_diff_eq!(result.integral(), 2.0 / 3.0, epsilon = 1e-14);
    assert_eq!(calls, result.evaluation_count());
    assert_eq!(result.terminal_region_count(), 1);
    assert_eq!(result.evaluation_count(), 5);
}

#[test]
fn one_dimensional_initial_subdivisions_are_terminal_regions() {
    let quadrature = AdaptiveQuadrature1d::builder(
        Interval::new(-2.0, 2.0).unwrap(),
        rule(3),
        rule(5),
        Tolerance::absolute(1e-12).unwrap(),
    )
    .subdivisions([-1.0, 0.0, 1.0])
    .unwrap()
    .build()
    .unwrap();

    let result = quadrature.integrate(|x| x * x).unwrap();
    assert_eq!(quadrature.subdivision_points(), &[-1.0, 0.0, 1.0]);
    assert_eq!(result.terminal_region_count(), 4);
    assert_eq!(result.evaluation_count(), 20);
}

#[test]
fn one_dimensional_depth_zero_returns_high_order_partial_result() {
    let domain = Interval::new(-1.0, 1.0).unwrap();
    let low = rule(1);
    let high = rule(3);
    let expected_high = topohedral_integrate::FixedQuadrature1d::builder(domain, high.clone())
        .build()
        .integrate(|x| x.powi(4))
        .unwrap();
    let quadrature =
        AdaptiveQuadrature1d::builder(domain, low, high, Tolerance::absolute(1e-15).unwrap())
            .max_depth(RefinementDepth::new(0))
            .build()
            .unwrap();

    let IntegrationError::MaxDepthReached { partial_result } =
        quadrature.integrate(|x| x.powi(4)).unwrap_err()
    else {
        panic!("expected depth exhaustion");
    };
    assert_eq!(partial_result.integral(), expected_high);
    assert!(partial_result.error_estimate() > 0.0);
    assert_eq!(partial_result.terminal_region_count(), 1);
    assert_eq!(partial_result.evaluation_count(), 3);
}

#[test]
fn one_dimensional_nonprogressing_midpoint_is_reported() {
    let lower = 1.0f64;
    let upper = f64::from_bits(lower.to_bits() + 1);
    let domain = Interval::new(lower, upper).unwrap();
    let quadrature = AdaptiveQuadrature1d::builder(
        domain,
        rule(1),
        rule(3),
        Tolerance::absolute(f64::MIN_POSITIVE).unwrap(),
    )
    .max_depth(RefinementDepth::new(1))
    .build()
    .unwrap();

    let error = quadrature
        .integrate(|x| if x == upper { 1e300 } else { 0.0 })
        .unwrap_err();
    assert!(matches!(
        error,
        IntegrationError::NonProgressingInterval { interval, .. } if interval == domain
    ));
}

#[test]
fn one_dimensional_nonfinite_integrand_is_propagated() {
    let quadrature = AdaptiveQuadrature1d::builder(
        Interval::new(-1.0, 1.0).unwrap(),
        rule(1),
        rule(3),
        Tolerance::absolute(1e-8).unwrap(),
    )
    .build()
    .unwrap();

    assert!(matches!(
        quadrature.integrate(|_| f64::NAN).unwrap_err(),
        IntegrationError::NonFiniteIntegrand {
            point: EvaluationPoint::OneDimensional(_),
            value,
        } if value.is_nan()
    ));
}

#[test]
fn two_dimensional_builder_validates_each_axis_exactness() {
    let error = AdaptiveQuadrature2d::builder(
        Rectangle::from_bounds(-1.0, 1.0, -1.0, 1.0).unwrap(),
        TensorRule2d::new(rule(2), rule(5)),
        TensorRule2d::new(rule(3), rule(4)),
        Tolerance::absolute(1e-8).unwrap(),
    )
    .build()
    .unwrap_err();

    assert!(matches!(
        error.issues(),
        [
            ConfigIssue::NonIncreasingRuleExactness {
                axis: RuleAxis::U,
                low: 3,
                high: 3,
            },
            ConfigIssue::NonIncreasingRuleExactness {
                axis: RuleAxis::V,
                low: 5,
                high: 5,
            }
        ]
    ));
}

#[test]
fn two_dimensional_free_function_builds_and_integrates_once() {
    let builder = AdaptiveQuadrature2d::builder(
        Rectangle::from_bounds(0.0, 1.0, 0.0, 1.0).unwrap(),
        tensor_rule(3),
        tensor_rule(5),
        Tolerance::absolute(1e-12).unwrap(),
    );

    let result = adaptive_quad_2d(|u, v| u * u + v * v, builder).unwrap();
    assert_abs_diff_eq!(result.integral(), 2.0 / 3.0, epsilon = 1e-14);
    assert_eq!(result.evaluation_count(), 13);
}

#[test]
fn two_dimensional_polynomial_uses_one_typed_region() {
    let quadrature = AdaptiveQuadrature2d::builder(
        Rectangle::from_bounds(-1.0, 1.0, -1.0, 1.0).unwrap(),
        tensor_rule(3),
        tensor_rule(5),
        Tolerance::absolute(1e-12).unwrap(),
    )
    .build()
    .unwrap();
    let mut calls = 0;

    let result = quadrature
        .integrate(|u, v| {
            calls += 1;
            u * u + v * v
        })
        .unwrap();

    assert_abs_diff_eq!(result.integral(), 8.0 / 3.0, epsilon = 1e-14);
    assert_eq!(result.terminal_region_count(), 1);
    assert_eq!(result.evaluation_count(), 13);
    assert_eq!(calls, result.evaluation_count());
}

#[test]
fn two_dimensional_splits_both_available_axes() {
    let quadrature = AdaptiveQuadrature2d::builder(
        Rectangle::from_bounds(-1.0, 1.0, -1.0, 1.0).unwrap(),
        tensor_rule(1),
        tensor_rule(3),
        Tolerance::absolute(1e-12).unwrap(),
    )
    .max_depth(AxisDepths::from_values(1, 1))
    .build()
    .unwrap();

    let result = quadrature
        .integrate(|u, v| f64::from(u > 0.0 && v > 0.0))
        .unwrap();
    assert_abs_diff_eq!(result.integral(), 1.0, epsilon = 1e-14);
    assert_eq!(result.terminal_region_count(), 4);
    assert_eq!(result.evaluation_count(), 25);
}

#[test]
fn two_dimensional_splits_only_axis_with_remaining_depth() {
    let quadrature = AdaptiveQuadrature2d::builder(
        Rectangle::from_bounds(-1.0, 1.0, -1.0, 1.0).unwrap(),
        tensor_rule(1),
        tensor_rule(3),
        Tolerance::absolute(1e-12).unwrap(),
    )
    .max_depth(AxisDepths::from_values(0, 1))
    .build()
    .unwrap();

    let result = quadrature.integrate(|_, v| f64::from(v > 0.0)).unwrap();
    assert_abs_diff_eq!(result.integral(), 2.0, epsilon = 1e-14);
    assert_eq!(result.terminal_region_count(), 2);
    assert_eq!(result.evaluation_count(), 15);
}

#[test]
fn two_dimensional_depth_exhaustion_contains_partial_result() {
    let quadrature = AdaptiveQuadrature2d::builder(
        Rectangle::from_bounds(-1.0, 1.0, -1.0, 1.0).unwrap(),
        tensor_rule(1),
        tensor_rule(3),
        Tolerance::absolute(1e-12).unwrap(),
    )
    .max_depth(AxisDepths::from_values(0, 0))
    .build()
    .unwrap();

    let IntegrationError::MaxDepthReached { partial_result } = quadrature
        .integrate(|u, v| f64::from(u > 0.0 && v > 0.0))
        .unwrap_err()
    else {
        panic!("expected depth exhaustion");
    };
    assert_abs_diff_eq!(partial_result.integral(), 1.0, epsilon = 1e-14);
    assert_eq!(partial_result.terminal_region_count(), 1);
    assert_eq!(partial_result.evaluation_count(), 5);
}

#[test]
fn two_dimensional_nonprogressing_midpoint_is_reported() {
    let lower = 1.0f64;
    let upper = f64::from_bits(lower.to_bits() + 1);
    let rectangle = Rectangle::new(
        Interval::new(lower, upper).unwrap(),
        Interval::new(-1.0, 1.0).unwrap(),
    );
    let quadrature = AdaptiveQuadrature2d::builder(
        rectangle,
        tensor_rule(1),
        tensor_rule(3),
        Tolerance::absolute(f64::MIN_POSITIVE).unwrap(),
    )
    .max_depth(AxisDepths::from_values(1, 0))
    .build()
    .unwrap();

    let error = quadrature
        .integrate(|u, _| if u == upper { 1e300 } else { 0.0 })
        .unwrap_err();
    assert!(matches!(
        error,
        IntegrationError::NonProgressingRectangle { rectangle: failed, .. }
            if failed == rectangle
    ));
}

#[test]
fn two_dimensional_initial_subdivisions_form_a_grid() {
    let quadrature = AdaptiveQuadrature2d::builder(
        Rectangle::from_bounds(-1.0, 1.0, -1.0, 1.0).unwrap(),
        tensor_rule(3),
        tensor_rule(5),
        Tolerance::absolute(1e-12).unwrap(),
    )
    .subdivisions([0.0], [0.0])
    .unwrap()
    .build()
    .unwrap();

    let result = quadrature.integrate(|u, v| u * u + v * v).unwrap();
    assert_eq!(quadrature.u_subdivision_points(), &[0.0]);
    assert_eq!(quadrature.v_subdivision_points(), &[0.0]);
    assert_eq!(result.terminal_region_count(), 4);
    assert_eq!(result.evaluation_count(), 52);
}
