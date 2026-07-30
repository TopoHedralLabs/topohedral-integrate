use approx::{assert_abs_diff_eq, assert_relative_eq};
use serde::Deserialize;
use std::{fs, ptr};
use topohedral_integrate::{
    legendre_rules, lobatto_rules, ConfigIssue, GaussFamily, GaussRule, GaussRuleSet, PointCount,
    PolynomialDegree, RuleError,
};

const MAX_REL: f64 = 1e-10;
const MAX_ABS: f64 = 1e-10;

fn degree(value: usize) -> PolynomialDegree {
    PolynomialDegree::new(value).unwrap()
}

fn count(value: usize) -> PointCount {
    PointCount::new(value).unwrap()
}

#[derive(Deserialize)]
struct ExpectedRule {
    points: Vec<f64>,
    weights: Vec<f64>,
}

#[derive(Deserialize)]
struct ExpectedRules {
    n2: ExpectedRule,
    n3: ExpectedRule,
    n4: ExpectedRule,
    n5: ExpectedRule,
    n6: ExpectedRule,
    n11: ExpectedRule,
    n26: ExpectedRule,
    n37: ExpectedRule,
}

#[derive(Deserialize)]
struct ExpectedFamily {
    values: ExpectedRules,
}

#[derive(Deserialize)]
struct ExpectedData {
    legendre: ExpectedFamily,
    lobatto: ExpectedFamily,
}

impl ExpectedData {
    fn load() -> Self {
        let json = fs::read_to_string("assets/gauss-quad.json").expect("unable to read test data");
        serde_json::from_str(&json).expect("unable to deserialize test data")
    }
}

fn assert_matches_expected(
    actual: &GaussRule,
    expected: &ExpectedRule,
) {
    assert_eq!(actual.point_count().value(), expected.points.len());
    for ((actual_point, actual_weight), (expected_point, expected_weight)) in actual
        .points()
        .iter()
        .zip(actual.weights())
        .zip(expected.points.iter().zip(&expected.weights))
    {
        assert_relative_eq!(actual_point, expected_point, epsilon = MAX_REL);
        assert_relative_eq!(actual_weight, expected_weight, epsilon = MAX_REL);
    }
}

macro_rules! legendre_test {
    ($test_name:ident, $dataset:ident, $point_count:expr) => {
        #[test]
        fn $test_name() {
            let expected = ExpectedData::load();
            let rules = GaussRuleSet::through_degree(GaussFamily::Legendre, degree(90)).unwrap();
            assert_matches_expected(
                rules.get_by_point_count(count($point_count)).unwrap(),
                &expected.legendre.values.$dataset,
            );
        }
    };
}

macro_rules! lobatto_test {
    ($test_name:ident, $dataset:ident, $point_count:expr) => {
        #[test]
        fn $test_name() {
            let expected = ExpectedData::load();
            let rules = GaussRuleSet::through_degree(GaussFamily::Lobatto, degree(90)).unwrap();
            assert_matches_expected(
                rules.get_by_point_count(count($point_count)).unwrap(),
                &expected.lobatto.values.$dataset,
            );
        }
    };
}

legendre_test!(legendre_n2, n2, 2);
legendre_test!(legendre_n3, n3, 3);
legendre_test!(legendre_n4, n4, 4);
legendre_test!(legendre_n5, n5, 5);
legendre_test!(legendre_n6, n6, 6);
legendre_test!(legendre_n11, n11, 11);
legendre_test!(legendre_n26, n26, 26);
legendre_test!(legendre_n37, n37, 37);

lobatto_test!(lobatto_n2, n2, 2);
lobatto_test!(lobatto_n3, n3, 3);
lobatto_test!(lobatto_n4, n4, 4);
lobatto_test!(lobatto_n5, n5, 5);
lobatto_test!(lobatto_n6, n6, 6);
lobatto_test!(lobatto_n11, n11, 11);
lobatto_test!(lobatto_n26, n26, 26);
lobatto_test!(lobatto_n37, n37, 37);

fn integrate_monomial(
    rule: &GaussRule,
    degree: usize,
) -> f64 {
    rule.points()
        .iter()
        .zip(rule.weights())
        .map(|(point, weight)| weight * point.powi(degree as i32))
        .sum()
}

fn exact_monomial_integral(degree: usize) -> f64 {
    if degree.is_multiple_of(2) {
        2.0 / (degree + 1) as f64
    } else {
        0.0
    }
}

fn assert_rule_structure(rule: &GaussRule) {
    assert_eq!(rule.points().len(), rule.point_count().value());
    assert_eq!(rule.weights().len(), rule.point_count().value());
    assert!(rule.points().iter().all(|point| point.is_finite()));
    assert!(rule
        .weights()
        .iter()
        .all(|weight| weight.is_finite() && *weight > 0.0));
    assert!(rule.points().windows(2).all(|points| points[0] < points[1]));
    assert_abs_diff_eq!(rule.weights().iter().sum::<f64>(), 2.0, epsilon = MAX_ABS);

    for index in 0..rule.point_count().value() {
        let mirror = rule.point_count().value() - index - 1;
        assert_abs_diff_eq!(
            rule.points()[index],
            -rule.points()[mirror],
            epsilon = MAX_ABS
        );
        assert_abs_diff_eq!(
            rule.weights()[index],
            rule.weights()[mirror],
            epsilon = MAX_ABS
        );
    }

    if rule.family() == GaussFamily::Lobatto {
        assert_eq!(rule.points().first(), Some(&-1.0));
        assert_eq!(rule.points().last(), Some(&1.0));
    }
}

#[test]
fn minimum_rules_are_supported() {
    let legendre = GaussRule::for_degree(GaussFamily::Legendre, degree(0)).unwrap();
    assert_eq!(legendre.point_count(), count(1));
    assert_eq!(legendre.exactness(), degree(1));
    assert_eq!(legendre.points(), &[0.0]);
    assert_eq!(legendre.weights(), &[2.0]);

    let lobatto_two = GaussRule::with_point_count(GaussFamily::Lobatto, count(2)).unwrap();
    assert_eq!(lobatto_two.exactness(), degree(1));
    assert_eq!(lobatto_two.points(), &[-1.0, 1.0]);
    assert_eq!(lobatto_two.weights(), &[1.0, 1.0]);

    let lobatto_three = GaussRule::with_point_count(GaussFamily::Lobatto, count(3)).unwrap();
    assert_eq!(lobatto_three.exactness(), degree(3));
    assert_eq!(lobatto_three.points(), &[-1.0, 0.0, 1.0]);
    assert_abs_diff_eq!(lobatto_three.weights()[0], 1.0 / 3.0, epsilon = MAX_ABS);
    assert_abs_diff_eq!(lobatto_three.weights()[1], 4.0 / 3.0, epsilon = MAX_ABS);
    assert_abs_diff_eq!(lobatto_three.weights()[2], 1.0 / 3.0, epsilon = MAX_ABS);
}

#[test]
fn requested_degree_selects_the_minimum_rule() {
    for family in [GaussFamily::Legendre, GaussFamily::Lobatto] {
        for degree in 0..=100 {
            let rule = GaussRule::for_degree(family, self::degree(degree)).unwrap();
            assert!(rule.exactness().value() >= degree);

            if rule.point_count().value()
                > match family {
                    GaussFamily::Legendre => 1,
                    GaussFamily::Lobatto => 2,
                }
            {
                let previous =
                    GaussRule::with_point_count(family, count(rule.point_count().value() - 1))
                        .unwrap();
                assert!(previous.exactness().value() < degree);
            }
        }
    }
}

#[test]
fn rules_integrate_monomials_through_their_exactness() {
    for family in [GaussFamily::Legendre, GaussFamily::Lobatto] {
        let minimum = match family {
            GaussFamily::Legendre => 1,
            GaussFamily::Lobatto => 2,
        };
        for point_count in minimum..=10 {
            let rule = GaussRule::with_point_count(family, count(point_count)).unwrap();
            assert_rule_structure(&rule);

            for degree in 0..=rule.exactness().value() {
                assert_abs_diff_eq!(
                    integrate_monomial(&rule, degree),
                    exact_monomial_integral(degree),
                    epsilon = MAX_ABS
                );
            }
        }
    }
}

#[test]
fn rule_set_lookup_and_iteration_borrow_stored_rules() {
    let rules = GaussRuleSet::through_degree(GaussFamily::Legendre, degree(10)).unwrap();
    assert_eq!(rules.family(), GaussFamily::Legendre);
    assert_eq!(rules.maximum_degree(), degree(10));
    assert_eq!(rules.minimum_point_count(), count(1));
    assert_eq!(rules.maximum_point_count(), count(6));
    assert_eq!(rules.len(), 6);
    assert!(!rules.is_empty());
    assert_eq!(
        rules
            .iter()
            .map(|rule| rule.point_count().value())
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6]
    );

    let by_degree = rules.get_for_degree(degree(10)).unwrap();
    let by_count = rules.get_by_point_count(count(6)).unwrap();
    assert!(ptr::eq(by_degree, by_count));
}

#[test]
fn cached_rule_sets_are_borrowed_and_complete() {
    let legendre = legendre_rules().unwrap();
    assert!(ptr::eq(legendre, legendre_rules().unwrap()));
    let legendre_max = legendre.get_for_degree(degree(100)).unwrap();
    assert_eq!(legendre_max.point_count(), count(51));
    assert_rule_structure(legendre_max);
    assert_abs_diff_eq!(
        integrate_monomial(legendre_max, 100),
        exact_monomial_integral(100),
        epsilon = MAX_ABS
    );

    let lobatto = lobatto_rules().unwrap();
    assert!(ptr::eq(lobatto, lobatto_rules().unwrap()));
    let lobatto_max = lobatto.get_for_degree(degree(100)).unwrap();
    assert_eq!(lobatto_max.point_count(), count(52));
    assert_rule_structure(lobatto_max);
    assert_abs_diff_eq!(
        integrate_monomial(lobatto_max, 100),
        exact_monomial_integral(100),
        epsilon = MAX_ABS
    );
}

#[test]
fn unsupported_degrees_and_point_counts_return_errors() {
    assert_eq!(
        GaussRule::for_degree(GaussFamily::Legendre, degree(101)),
        Err(RuleError::DegreeOutOfRange {
            degree: 101,
            maximum: 100,
        })
    );
    assert!(matches!(
        PointCount::new(0).unwrap_err().issues(),
        [ConfigIssue::PointCountOutOfRange {
            point_count: 0,
            minimum: 1,
            maximum: 52,
        }]
    ));
    assert!(matches!(
        PointCount::new(usize::MAX).unwrap_err().issues(),
        [ConfigIssue::PointCountOutOfRange {
            point_count: usize::MAX,
            minimum: 1,
            maximum: 52,
        }]
    ));
    assert!(matches!(
        GaussRule::with_point_count(GaussFamily::Lobatto, count(1)),
        Err(RuleError::PointCountOutOfRange {
            point_count: 1,
            minimum: 2,
            maximum: 52,
            ..
        })
    ));
    assert!(matches!(
        GaussRule::with_point_count(GaussFamily::Legendre, count(52)),
        Err(RuleError::PointCountOutOfRange {
            point_count: 52,
            minimum: 1,
            maximum: 51,
            ..
        })
    ));

    let rules = GaussRuleSet::through_degree(GaussFamily::Lobatto, degree(8)).unwrap();
    assert!(matches!(
        rules.get_for_degree(degree(9)),
        Err(RuleError::DegreeOutOfRange {
            degree: 9,
            maximum: 8,
        })
    ));
    assert!(matches!(
        rules.get_by_point_count(count(7)),
        Err(RuleError::PointCountOutOfRange {
            point_count: 7,
            minimum: 2,
            maximum: 6,
            ..
        })
    ));
}
