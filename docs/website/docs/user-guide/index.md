# User Guide

The public API is flat: Gaussian rules, validated configuration values, fixed
quadratures, and adaptive entry points are all available at the crate root.
Dimensional types use explicit `1D` and `2D` suffixes.

The API is divided into four areas:

## Validated values

[Validated values](validated-values.md) represent polynomial degrees, point
counts, intervals, rectangles, tolerances, and refinement depths. Constructors
reject non-finite or otherwise invalid input before numerical work begins.

## Gaussian rules

[Gaussian rules](gaussian-rules.md) produce points and weights on the standard
interval \([-1, 1]\). Use `GaussRule` for one rule, or `GaussRuleSet` when
several orders are needed. Both Gauss-Legendre and Gauss-Lobatto families are
available through `GaussFamily`.

## Fixed quadrature

[Fixed quadrature](fixed-quadrature.md) maps a Gaussian rule onto a 1D interval
or 2D rectangle. The corresponding `Quadrature` types precompute typed nodes
and can be reused for multiple functions.

## Adaptive quadrature

[Adaptive quadrature](adaptive-quadrature.md) repeatedly subdivides intervals
or rectangles, refining the largest-error region until the global
absolute-plus-relative tolerance is met. `AdaptiveQuadrature1D` and
`AdaptiveQuadrature2D` return a shared `AdaptiveResult` with the high-order
integral and diagnostic information. The `adaptive_quad_1d` and
`adaptive_quad_2d` helpers provide the corresponding one-shot interface.

## Structured validation

Fixed quadrature starts from validated domain and Gaussian-rule values. Its
builder validates subdivisions before constructing the reusable quadrature:

```rust
use topohedral_integrate::{
    FixedQuadrature1D, GaussFamily, GaussRule, Interval, PolynomialDegree,
};

let domain = Interval::new(-1.0, 1.0).unwrap();
let degree = PolynomialDegree::new(9).unwrap();
let rule = GaussRule::for_degree(GaussFamily::Legendre, degree).unwrap();
let quadrature = FixedQuadrature1D::builder(domain, rule)
    .subdivisions([0.0])
    .unwrap()
    .build();
assert_eq!(quadrature.subdivision_points(), &[0.0]);
```
