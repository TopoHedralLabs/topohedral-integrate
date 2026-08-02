//! Compile-time assertions for the crate's visibility and trait policy.
//!
//! The bodies of these tests are type-checked whether or not they run, so a public type that
//! loses a required trait fails the build rather than a runtime assertion.

use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use topohedral_integrate::{
    AdaptiveQuadrature1d, AdaptiveQuadrature2d, AdaptiveQuadratureBuilder1d,
    AdaptiveQuadratureBuilder2d, AdaptiveResult, AxisDepths, ConfigError, ConfigIssue,
    EvaluationPoint, FixedNode1d, FixedNode2d, FixedQuadrature1d, FixedQuadrature2d,
    FixedQuadratureBuilder1d, FixedQuadratureBuilder2d, GaussFamily, GaussRule, GaussRuleSet,
    IntegrationError, Interval, OptionsError, PointCount, PolynomialDegree, Rectangle,
    RefinementDepth, RuleAxis, RuleError, TensorRule2d, Tolerance,
};

fn assert_send_sync_debug<T: Send + Sync + Debug + 'static>() {}

fn assert_clone_partial_eq<T: Clone + PartialEq>() {}

fn assert_copy_eq_hash<T: Copy + Eq + Hash>() {}

fn assert_display<T: Display>() {}

fn assert_error<T: Error>() {}

fn assert_default<T: Default>() {}

/// Every public owned type is thread-safe and printable.
#[test]
fn public_types_are_send_sync_and_debug() {
    // Validated values and configuration.
    assert_send_sync_debug::<PolynomialDegree>();
    assert_send_sync_debug::<PointCount>();
    assert_send_sync_debug::<Interval>();
    assert_send_sync_debug::<Rectangle>();
    assert_send_sync_debug::<Tolerance>();
    assert_send_sync_debug::<RefinementDepth>();
    assert_send_sync_debug::<AxisDepths>();
    assert_send_sync_debug::<RuleAxis>();
    assert_send_sync_debug::<ConfigIssue>();
    assert_send_sync_debug::<ConfigError>();

    // Gaussian rules.
    assert_send_sync_debug::<GaussFamily>();
    assert_send_sync_debug::<GaussRule>();
    assert_send_sync_debug::<GaussRuleSet>();
    assert_send_sync_debug::<RuleError>();

    // Integration results and errors.
    assert_send_sync_debug::<EvaluationPoint>();
    assert_send_sync_debug::<AdaptiveResult>();
    assert_send_sync_debug::<IntegrationError>();
    assert_send_sync_debug::<OptionsError>();

    // Fixed quadrature.
    assert_send_sync_debug::<FixedNode1d>();
    assert_send_sync_debug::<FixedNode2d>();
    assert_send_sync_debug::<TensorRule2d>();
    assert_send_sync_debug::<FixedQuadratureBuilder1d>();
    assert_send_sync_debug::<FixedQuadratureBuilder2d>();
    assert_send_sync_debug::<FixedQuadrature1d>();
    assert_send_sync_debug::<FixedQuadrature2d>();

    // Adaptive quadrature.
    assert_send_sync_debug::<AdaptiveQuadratureBuilder1d>();
    assert_send_sync_debug::<AdaptiveQuadratureBuilder2d>();
    assert_send_sync_debug::<AdaptiveQuadrature1d>();
    assert_send_sync_debug::<AdaptiveQuadrature2d>();
}

/// Discrete enums and newtypes carry the full value-type trait set.
#[test]
fn discrete_types_are_copy_eq_and_hash() {
    assert_copy_eq_hash::<PolynomialDegree>();
    assert_copy_eq_hash::<PointCount>();
    assert_copy_eq_hash::<RefinementDepth>();
    assert_copy_eq_hash::<AxisDepths>();
    assert_copy_eq_hash::<RuleAxis>();
    assert_copy_eq_hash::<GaussFamily>();

    // `RuleError` is float-free but owns a diagnostic string, so it is hashable but not `Copy`.
    fn assert_eq_hash<T: Eq + Hash>() {}
    assert_eq_hash::<RuleError>();
}

/// Float-bearing values, rules, configurations, and results are cloneable and comparable.
#[test]
fn float_bearing_types_are_clone_and_partial_eq() {
    assert_clone_partial_eq::<Interval>();
    assert_clone_partial_eq::<Rectangle>();
    assert_clone_partial_eq::<Tolerance>();
    assert_clone_partial_eq::<ConfigIssue>();
    assert_clone_partial_eq::<ConfigError>();
    assert_clone_partial_eq::<GaussRule>();
    assert_clone_partial_eq::<GaussRuleSet>();
    assert_clone_partial_eq::<EvaluationPoint>();
    assert_clone_partial_eq::<AdaptiveResult>();
    assert_clone_partial_eq::<IntegrationError>();
    assert_clone_partial_eq::<OptionsError>();
    assert_clone_partial_eq::<FixedNode1d>();
    assert_clone_partial_eq::<FixedNode2d>();
    assert_clone_partial_eq::<TensorRule2d>();
    assert_clone_partial_eq::<FixedQuadratureBuilder1d>();
    assert_clone_partial_eq::<FixedQuadratureBuilder2d>();
    assert_clone_partial_eq::<FixedQuadrature1d>();
    assert_clone_partial_eq::<FixedQuadrature2d>();
    assert_clone_partial_eq::<AdaptiveQuadratureBuilder1d>();
    assert_clone_partial_eq::<AdaptiveQuadratureBuilder2d>();
    assert_clone_partial_eq::<AdaptiveQuadrature1d>();
    assert_clone_partial_eq::<AdaptiveQuadrature2d>();

    // Small, allocation-free value types are additionally `Copy`.
    fn assert_copy<T: Copy>() {}
    assert_copy::<Interval>();
    assert_copy::<Rectangle>();
    assert_copy::<Tolerance>();
    assert_copy::<EvaluationPoint>();
    assert_copy::<FixedNode1d>();
    assert_copy::<FixedNode2d>();
}

/// Errors and domain values format themselves for end users.
#[test]
fn errors_and_domain_values_implement_display() {
    assert_error::<ConfigError>();
    assert_error::<RuleError>();
    assert_error::<IntegrationError>();
    assert_error::<OptionsError>();

    assert_display::<ConfigIssue>();
    assert_display::<PolynomialDegree>();
    assert_display::<PointCount>();
    assert_display::<Interval>();
    assert_display::<Rectangle>();
    assert_display::<Tolerance>();
    assert_display::<RefinementDepth>();
    assert_display::<AxisDepths>();
    assert_display::<RuleAxis>();
    assert_display::<GaussFamily>();
    assert_display::<EvaluationPoint>();
}

/// `Default` is implemented only where a meaningful default exists.
#[test]
fn only_refinement_depths_implement_default() {
    assert_default::<RefinementDepth>();
    assert_default::<AxisDepths>();

    assert_eq!(RefinementDepth::default().value(), RefinementDepth::DEFAULT);
    assert_eq!(
        AxisDepths::default(),
        AxisDepths::uniform(RefinementDepth::default())
    );
}

#[test]
fn display_output_is_concise_and_lowercase() {
    assert_eq!(
        Tolerance::new(1e-8, 1e-6).unwrap().to_string(),
        "absolute 0.00000001, relative 0.000001"
    );
    assert_eq!(AxisDepths::from_values(4, 8).to_string(), "u 4, v 8");
    assert_eq!(EvaluationPoint::OneDimensional(0.5).to_string(), "x = 0.5");
    assert_eq!(
        EvaluationPoint::TwoDimensional { u: 0.5, v: -1.5 }.to_string(),
        "(u, v) = (0.5, -1.5)"
    );
    assert_eq!(
        IntegrationError::NonFiniteIntegrand {
            point: EvaluationPoint::OneDimensional(0.5),
            value: f64::INFINITY,
        }
        .to_string(),
        "integrand returned non-finite value inf at x = 0.5"
    );
}
