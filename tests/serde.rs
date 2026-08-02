#![cfg(feature = "serde")]

use topohedral_integrate::{
    AdaptiveQuadrature1d, FixedNode1d, FixedQuadrature1d, GaussFamily, GaussRule, GaussRuleSet,
    Interval, PointCount, PolynomialDegree, Tolerance,
};

fn degree(value: usize) -> PolynomialDegree {
    PolynomialDegree::new(value).unwrap()
}

#[test]
fn validated_values_reject_invalid_deserialized_representations() {
    assert!(serde_json::from_str::<PolynomialDegree>("102").is_err());
    assert!(serde_json::from_str::<PointCount>("0").is_err());
    assert!(serde_json::from_str::<Interval>(r#"{"lower":1.0,"upper":-1.0}"#).is_err());
    assert!(serde_json::from_str::<Tolerance>(r#"{"absolute":0.0,"relative":0.0}"#).is_err());
}

#[test]
fn rules_round_trip_by_regenerating_canonical_data() {
    let rule = GaussRule::for_degree(GaussFamily::Lobatto, degree(12)).unwrap();
    let json = serde_json::to_string(&rule).unwrap();
    let decoded: GaussRule = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, rule);

    let rules = GaussRuleSet::through_degree(GaussFamily::Legendre, degree(20)).unwrap();
    let json = serde_json::to_string(&rules).unwrap();
    let decoded: GaussRuleSet = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, rules);
}

#[test]
fn nodes_and_adaptive_results_round_trip() {
    let domain = Interval::new(-1.0, 1.0).unwrap();
    let low = GaussRule::for_degree(GaussFamily::Legendre, degree(3)).unwrap();
    let high = GaussRule::for_degree(GaussFamily::Legendre, degree(5)).unwrap();

    let fixed = FixedQuadrature1d::builder(domain, high.clone()).build();
    let node_json = serde_json::to_string(&fixed.nodes()[0]).unwrap();
    let node: FixedNode1d = serde_json::from_str(&node_json).unwrap();
    assert_eq!(node, fixed.nodes()[0]);
    assert!(serde_json::from_str::<FixedNode1d>(r#"{"point":0.0,"weight":-1.0}"#).is_err());

    let result =
        AdaptiveQuadrature1d::builder(domain, low, high, Tolerance::absolute(1e-12).unwrap())
            .build()
            .unwrap()
            .integrate(|x| x * x)
            .unwrap();
    let result_json = serde_json::to_string(&result).unwrap();
    let decoded: topohedral_integrate::AdaptiveResult = serde_json::from_str(&result_json).unwrap();
    approx::assert_relative_eq!(
        result.integral(),
        decoded.integral(),
        epsilon = f64::EPSILON
    );
    approx::assert_relative_eq!(
        result.error_estimate(),
        decoded.error_estimate(),
        epsilon = f64::EPSILON
    );
    assert_eq!(
        result.terminal_region_count(),
        decoded.terminal_region_count()
    );
    assert_eq!(result.evaluation_count(), decoded.evaluation_count());
}
