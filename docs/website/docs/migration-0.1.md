# Migrating to 0.1.0

Version 0.1.0 is a deliberate breaking release. It replaces public,
freely-mutable configuration with validated values and builders, repairs the
Gaussian mathematics, and makes fixed and adaptive integration report
structured failures. Deprecated compatibility aliases are intentionally not
provided.

## Removed symbols

| 0.0.x symbol | 0.1.0 replacement |
| --- | --- |
| `GaussQuadType` | `GaussFamily` |
| `GaussQuad` | `GaussRule` |
| `GuassQuadSet` | `GaussRuleSet` |
| `get_legendre_points` | `legendre_rules` |
| `get_lobatto_points` | `lobatto_rules` |
| `FixedQuadOpts1D` | `FixedQuadrature1d::builder` / `FixedQuadratureBuilder1d` |
| `FixedQuadOpts2D` | `FixedQuadrature2d::builder` / `FixedQuadratureBuilder2d` |
| `FixedQuad1D` | `FixedQuadrature1d` |
| `FixedQuad2D` | `FixedQuadrature2d` |
| `AdaptiveQuadOpts1D` | `AdaptiveQuadrature1d::builder` / `AdaptiveQuadratureBuilder1d` |
| `AdaptiveQuadOpts2D` | `AdaptiveQuadrature2d::builder` / `AdaptiveQuadratureBuilder2d` |
| `AdaptiveQuadResult1D` | `AdaptiveResult` |
| `AdaptiveQuadResult2D` | `AdaptiveResult` |
| `OptionsError::InvalidOptionsShort` | `OptionsError::Config(ConfigError)` |
| `OptionsError::InvalidOptionsFull` | `OptionsError::Config(ConfigError)` |

The private `OptionsVerify` machinery and the non-root `append_reason` helper
were removed. `ConfigError::issues()` now retains every structured
`ConfigIssue`, including offending values.

## Gaussian rules

Rule construction now distinguishes requested polynomial exactness from point
count:

```rust
use topohedral_integrate::{GaussFamily, GaussRule, PointCount, PolynomialDegree};

let by_degree = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(9)?,
)?;
let by_count = GaussRule::with_point_count(
    GaussFamily::Lobatto,
    PointCount::new(6)?,
)?;
# assert_eq!(by_degree.point_count().value(), 5);
# assert_eq!(by_count.point_count().value(), 6);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`GaussRule` and `GaussRuleSet` fields are private. Use accessors such as
`family`, `point_count`, `points`, `weights`, `rule_for_degree`, and
`rule_by_point_count`.

## Fixed integration

Replace an options literal and constructor with validated values and a
builder. Integration now accepts `FnMut` by value and returns
`Result<f64, IntegrationError>`:

```rust
use topohedral_integrate::{
    FixedQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
};

let domain = Interval::new(-1.0, 1.0)?;
let rule = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(9)?,
)?;
let quadrature = FixedQuadrature1d::builder(domain, rule)
    .subdivisions([0.0])?
    .build();
let value = quadrature.integrate(|x| x * x)?;
# assert!((value - 2.0 / 3.0).abs() < 1e-12);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The free functions remain flat root-level entry points. Their second argument
is now a builder:

- `fixed_quad_1d(f, builder)`;
- `fixed_quad_2d(f, builder)`.

Packed public `points_weights` vectors were replaced by iterators and typed
`FixedNode1d` / `FixedNode2d` values. Use `point` and `weight` in 1d, or `u`,
`v`, and `weight` in 2d. `point_count` replaces the old `nqp` method.

## Adaptive integration

Adaptive integration requires independently constructed low- and high-order
rules. The high rule must be more exact in every applicable axis. Tolerance
and refinement limits are validated values:

```rust
use topohedral_integrate::{
    AdaptiveQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
    RefinementDepth, Tolerance,
};

let low = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(3)?,
)?;
let high = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(9)?,
)?;
let quadrature = AdaptiveQuadrature1d::builder(
    Interval::new(0.0, 1.0)?,
    low,
    high,
    Tolerance::new(1e-10, 1e-8)?,
)
.max_depth(RefinementDepth::new(24))
.build()?;
let result = quadrature.integrate(|x| x.exp())?;
# assert!(result.integral().is_finite());
# Ok::<(), Box<dyn std::error::Error>>(())
```

`AdaptiveResult` exposes the high-order integral, global error estimate,
terminal-region count, and exact evaluation count through accessors. If depth
is exhausted before the global tolerance is met,
`IntegrationError::MaxDepthReached` contains this same result as
`partial_result`.

The one-shot signatures are now:

- `adaptive_quad_1d(f, builder)`;
- `adaptive_quad_2d(f, builder)`.

Both return `Result<AdaptiveResult, OptionsError>` so builder validation and
integration failures remain distinguishable.

## Feature changes

`enable_trace` was removed. Use the positively named `trace` feature. It
enables instrumentation in this crate and its linear-algebra dependency but
does not enable ANSI colour. Serde support is optional behind `serde`; all
features are disabled by default.
