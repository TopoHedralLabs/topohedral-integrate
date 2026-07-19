# Gaussian Rules

Gaussian quadrature approximates an integral on \([-1, 1]\) with a weighted
sum

\[
    \int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{n} w_i f(x_i).
\]

The crate supports two rule families:

- `GaussQuadType::Legendre` uses interior points and an \(n\)-point rule of
  order \(2n-1\);
- `GaussQuadType::Lobatto` includes both interval endpoints and an \(n\)-point
  rule of order \(2n-3\).

The `GaussQuadType::nqp_from_order` and `order_from_nqp` methods convert between
polynomial order and number of quadrature points.

## `GaussQuad`: one rule

`GaussQuad::new(gauss_type, order)` constructs one rule. Its public fields are:

- `gauss_type`: the selected family;
- `nqp`: the number of quadrature points;
- `points`: the \(x_i\) values;
- `weights`: the corresponding \(w_i\) values.

This example constructs the same kind of Legendre rule exercised by the test
suite and uses its points and weights directly:

```rust
use topohedral_integrate::{GaussQuad, GaussQuadType};

let rule = GaussQuad::new(GaussQuadType::Legendre, 9);
assert_eq!(rule.nqp, 5);

let integral: f64 = rule
    .points
    .iter()
    .zip(&rule.weights)
    .map(|(&x, &w)| w * x.powi(8))
    .sum();

assert!((integral - 2.0 / 9.0).abs() < 1e-12);
```

`GaussQuad` is a reference-interval rule. To map points and weights to arbitrary
bounds, use a fixed quadrature rule rather than performing the transformation
by hand.

## `GuassQuadSet`: a family of rules

`GuassQuadSet::new(gauss_type, max_order)` precomputes rules up to a requested
order. The public type name is currently spelled `GuassQuadSet`.

Use `gauss_quad_from_nqp` to select by point count or
`gauss_quad_from_order` to select by polynomial order:

```rust
use topohedral_integrate::{GaussQuadType, GuassQuadSet};

let rules = GuassQuadSet::new(GaussQuadType::Legendre, 90);
let rule = rules.gauss_quad_from_nqp(37);

assert_eq!(rule.gauss_type, GaussQuadType::Legendre);
assert_eq!(rule.nqp, 37);
assert_eq!(rule.points.len(), rule.weights.len());
```

This construction mirrors the Gaussian-rule tests, which compare selected
Legendre and Lobatto point and weight arrays against reference data.

## Shared cached sets

`get_legendre_points()` and `get_lobatto_points()` return process-wide,
lazily initialized rule sets. They are useful when callers need several rules
without constructing a new `GuassQuadSet` each time:

```rust
use topohedral_integrate::get_lobatto_points;

let rule = get_lobatto_points().gauss_quad_from_nqp(5);
assert_eq!(rule.points.first(), Some(&-1.0));
assert_eq!(rule.points.last(), Some(&1.0));
```

The fixed quadrature types use these cached sets internally.
