use approx::{assert_abs_diff_eq, assert_relative_eq};
use serde::Deserialize;
use std::fs;
use topohedral_integrate::{
    get_legendre_points, get_lobatto_points, GaussQuad, GaussQuadType, GuassQuadSet,
};

const MAX_REL: f64 = 1e-10;
const MAX_ABS: f64 = 1e-10;

#[derive(Deserialize)]
struct GaussQuadTest4 {
    points: Vec<f64>,
    weights: Vec<f64>,
}

#[derive(Deserialize)]
struct GaussQuadTest3 {
    n2: GaussQuadTest4,
    n3: GaussQuadTest4,
    n4: GaussQuadTest4,
    n5: GaussQuadTest4,
    n6: GaussQuadTest4,
    n11: GaussQuadTest4,
    n26: GaussQuadTest4,
    n37: GaussQuadTest4,
}

#[derive(Deserialize)]
struct GaussQuadTest2 {
    values: GaussQuadTest3,
}

#[derive(Deserialize)]
struct GaussQuadTest1 {
    legendre: GaussQuadTest2,
    lobatto: GaussQuadTest2,
}

impl GaussQuadTest1 {
    fn new() -> Self {
        let json_file = fs::read_to_string("assets/gauss-quad.json").expect("Unable to read file");
        serde_json::from_str(&json_file).expect("Could not deserialize")
    }
}

macro_rules! legendre_test {
    ($test_name: ident, $dataset: ident, $nqp: expr) => {
        #[test]
        fn $test_name() {
            let test_data = GaussQuadTest1::new();
            let leg = GuassQuadSet::new(GaussQuadType::Legendre, 90);

            let points1 = test_data.legendre.values.$dataset.points;
            let weights1 = test_data.legendre.values.$dataset.weights;
            let rule = leg.gauss_quad_from_nqp($nqp);
            let points2 = rule.points;
            let weights2 = rule.weights;
            assert_eq!(points1.len(), points2.len());
            for i in 0..points1.len() {
                assert_relative_eq!(points1[i], points2[i], epsilon = MAX_REL);
                assert_relative_eq!(weights1[i], weights2[i], epsilon = MAX_REL);
            }
        }
    };
}
legendre_test!(legendre_test1, n2, 2);
legendre_test!(legendre_test2, n3, 3);
legendre_test!(legendre_test3, n4, 4);
legendre_test!(legendre_test4, n5, 5);
legendre_test!(legendre_test5, n6, 6);
legendre_test!(legendre_test6, n11, 11);
legendre_test!(legendre_test7, n26, 26);
legendre_test!(legendre_test8, n37, 37);
//..............................................................................................

macro_rules! lobatto_test {
    ($test_name: ident, $dataset: ident, $nqp: expr) => {
        #[test]
        fn $test_name() {
            let test_data = GaussQuadTest1::new();
            let leg = GuassQuadSet::new(GaussQuadType::Lobatto, 90);

            let points1 = test_data.lobatto.values.$dataset.points;
            let weights1 = test_data.lobatto.values.$dataset.weights;
            let rule = leg.gauss_quad_from_nqp($nqp);
            let points2 = rule.points;
            let weights2 = rule.weights;

            assert_eq!(points1.len(), points2.len());
            for i in 0..points1.len() {
                assert_relative_eq!(points1[i], points2[i], epsilon = MAX_REL);
                assert_relative_eq!(weights1[i], weights2[i], epsilon = MAX_REL);
            }
        }
    };
}
lobatto_test!(lobatto_test1, n2, 2);
lobatto_test!(lobatto_test2, n3, 3);
lobatto_test!(lobatto_test3, n4, 4);
lobatto_test!(lobatto_test4, n5, 5);
lobatto_test!(lobatto_test5, n6, 6);
lobatto_test!(lobatto_test6, n11, 11);
lobatto_test!(lobatto_test7, n26, 26);
lobatto_test!(lobatto_test8, n37, 37);
//..............................................................................................

fn integrate_monomial(
    rule: &GaussQuad,
    degree: usize,
) -> f64 {
    rule.points
        .iter()
        .zip(&rule.weights)
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

fn assert_rule_structure(rule: &GaussQuad) {
    assert_eq!(rule.points.len(), rule.nqp);
    assert_eq!(rule.weights.len(), rule.nqp);
    assert!(rule.points.iter().all(|point| point.is_finite()));
    assert!(rule
        .weights
        .iter()
        .all(|weight| weight.is_finite() && *weight > 0.0));
    assert!(rule.points.windows(2).all(|points| points[0] < points[1]));
    assert_abs_diff_eq!(rule.weights.iter().sum::<f64>(), 2.0, epsilon = MAX_ABS);

    for i in 0..rule.nqp {
        let j = rule.nqp - i - 1;
        assert_abs_diff_eq!(rule.points[i], -rule.points[j], epsilon = MAX_ABS);
        assert_abs_diff_eq!(rule.weights[i], rule.weights[j], epsilon = MAX_ABS);
    }

    if rule.gauss_type == GaussQuadType::Lobatto {
        assert_eq!(rule.points.first(), Some(&-1.0));
        assert_eq!(rule.points.last(), Some(&1.0));
    }
}

#[test]
fn minimum_rules_are_supported() {
    let legendre = GaussQuad::new(GaussQuadType::Legendre, 0);
    assert_eq!(legendre.nqp, 1);
    assert_eq!(legendre.points, vec![0.0]);
    assert_eq!(legendre.weights, vec![2.0]);

    let lobatto_2 = GaussQuad::new(GaussQuadType::Lobatto, 0);
    assert_eq!(lobatto_2.nqp, 2);
    assert_eq!(lobatto_2.points, vec![-1.0, 1.0]);
    assert_eq!(lobatto_2.weights, vec![1.0, 1.0]);

    let lobatto_3 = GaussQuad::new(GaussQuadType::Lobatto, 2);
    assert_eq!(lobatto_3.nqp, 3);
    assert_eq!(lobatto_3.points, vec![-1.0, 0.0, 1.0]);
    assert_abs_diff_eq!(lobatto_3.weights[0], 1.0 / 3.0, epsilon = MAX_ABS);
    assert_abs_diff_eq!(lobatto_3.weights[1], 4.0 / 3.0, epsilon = MAX_ABS);
    assert_abs_diff_eq!(lobatto_3.weights[2], 1.0 / 3.0, epsilon = MAX_ABS);
}

#[test]
fn requested_degree_selects_the_minimum_rule() {
    for gauss_type in [GaussQuadType::Legendre, GaussQuadType::Lobatto] {
        for degree in 0..=100 {
            let nqp = gauss_type.nqp_from_order(degree);
            assert!(gauss_type.order_from_nqp(nqp) >= degree);

            let min_nqp = match gauss_type {
                GaussQuadType::Legendre => 1,
                GaussQuadType::Lobatto => 2,
            };
            if nqp > min_nqp {
                assert!(gauss_type.order_from_nqp(nqp - 1) < degree);
            }
        }
    }
}

#[test]
fn rules_integrate_monomials_through_their_exactness() {
    for nqp in 1..=10 {
        let gauss_type = GaussQuadType::Legendre;
        let rule = GaussQuad::new(gauss_type, gauss_type.order_from_nqp(nqp));
        assert_eq!(rule.nqp, nqp);
        assert_rule_structure(&rule);

        for degree in 0..=gauss_type.order_from_nqp(nqp) {
            assert_abs_diff_eq!(
                integrate_monomial(&rule, degree),
                exact_monomial_integral(degree),
                epsilon = MAX_ABS
            );
        }
    }

    for nqp in 2..=10 {
        let gauss_type = GaussQuadType::Lobatto;
        let rule = GaussQuad::new(gauss_type, gauss_type.order_from_nqp(nqp));
        assert_eq!(rule.nqp, nqp);
        assert_rule_structure(&rule);

        for degree in 0..=gauss_type.order_from_nqp(nqp) {
            assert_abs_diff_eq!(
                integrate_monomial(&rule, degree),
                exact_monomial_integral(degree),
                epsilon = MAX_ABS
            );
        }
    }
}

#[test]
fn cached_maximum_degree_rules_are_populated() {
    let legendre = get_legendre_points().gauss_quad_from_order(100);
    assert_eq!(legendre.nqp, 51);
    assert_rule_structure(&legendre);
    assert_abs_diff_eq!(
        integrate_monomial(&legendre, 100),
        exact_monomial_integral(100),
        epsilon = MAX_ABS
    );

    let lobatto = get_lobatto_points().gauss_quad_from_order(100);
    assert_eq!(lobatto.nqp, 52);
    assert_rule_structure(&lobatto);
    assert_abs_diff_eq!(
        integrate_monomial(&lobatto, 100),
        exact_monomial_integral(100),
        epsilon = MAX_ABS
    );
}

#[test]
#[should_panic(expected = "Legendre rules require at least one point")]
fn legendre_exactness_rejects_zero_points() {
    GaussQuadType::Legendre.order_from_nqp(0);
}

#[test]
#[should_panic(expected = "Lobatto rules require at least two points")]
fn lobatto_exactness_rejects_one_point() {
    GaussQuadType::Lobatto.order_from_nqp(1);
}
