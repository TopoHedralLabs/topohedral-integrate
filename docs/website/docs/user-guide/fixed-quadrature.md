# Fixed Quadrature

A fixed quadrature maps Gaussian points and weights from \([-1, 1]\) onto the
requested integration interval or rectangle. Constructing a
`FixedQuadrature1d` or `FixedQuadrature2d` does this mapping once, so it can be
reused to integrate several functions over the same domain.

## One-dimensional entry point

Build a one-dimensional rule from validated domain and Gaussian-rule values:

```rust
use topohedral_integrate::{
    FixedQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
};

let domain = Interval::new(-2.0, 3.0).expect("valid interval");
let degree = PolynomialDegree::new(9).expect("supported degree");
let rule = GaussRule::for_degree(GaussFamily::Legendre, degree)
    .expect("rule generation succeeds");
let quadrature = FixedQuadrature1d::builder(domain, rule)
    .subdivisions([0.0])
    .expect("valid subdivisions")
    .build();
let integral = quadrature
    .integrate(|x: f64| x.powi(4))
    .expect("finite integrand");

assert!((integral - 55.0).abs() < 1e-12);
```

For a single integration, pass the builder directly to the matching free
function instead of retaining the mapped quadrature:

```rust
use topohedral_integrate::{
    fixed_quad_1d, FixedQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
};

let domain = Interval::new(-1.0, 1.0).unwrap();
let rule = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(5).unwrap(),
)
.unwrap();
let integral = fixed_quad_1d(|x| x * x, FixedQuadrature1d::builder(domain, rule)).unwrap();
assert!((integral - 2.0 / 3.0).abs() < 1e-12);
```

Subdivisions split the configured range before applying the rule. They are
useful when a function is only piecewise smooth. Do not include the outer
bounds. The builder accepts any `IntoIterator<Item = f64>` and validates that
the coordinates are finite, strictly increasing, and inside the domain.

`integrate_over` reuses the quadrature on another validated interval. Existing
subdivisions move proportionally with the endpoints:

```rust
use topohedral_integrate::{
    FixedQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
};

let original = Interval::new(-1.0, 1.0).unwrap();
let target = Interval::new(0.0, 2.0).unwrap();
let degree = PolynomialDegree::new(9).unwrap();
let rule = GaussRule::for_degree(GaussFamily::Legendre, degree).unwrap();
let quadrature = FixedQuadrature1d::builder(original, rule).build();
let integral = quadrature
    .integrate_over(target, |x: f64| x.powi(2))
    .expect("finite integrand");
assert!((integral - 8.0 / 3.0).abs() < 1e-12);
```

## Two-dimensional entry point

`FixedQuadrature2d` takes a `Rectangle` and a `TensorRule2d`, so each coordinate
can use an independent Gaussian rule. The function supplied to `integrate` has
type `FnMut(f64, f64) -> f64`:

```rust
use topohedral_integrate::{
    FixedQuadrature2d, GaussFamily, GaussRule, PolynomialDegree, Rectangle, TensorRule2d,
};

let domain = Rectangle::from_bounds(-1.0, 1.0, -1.0, 1.0).unwrap();
let degree = PolynomialDegree::new(5).unwrap();
let u_rule = GaussRule::for_degree(GaussFamily::Legendre, degree).unwrap();
let v_rule = GaussRule::for_degree(GaussFamily::Legendre, degree).unwrap();
let quadrature =
    FixedQuadrature2d::builder(domain, TensorRule2d::new(u_rule, v_rule)).build();
let integral = quadrature
    .integrate(|x: f64, y: f64| x.powi(2) * y.powi(2))
    .expect("finite integrand");

assert!((integral - 4.0 / 9.0).abs() < 1e-12);
```

Call `.subdivisions(u_points, v_points)` before `.build()` to subdivide either
axis. An empty iterator leaves that axis unsubdivided.

## Nodes and integration errors

Mapped nodes are exposed as typed values rather than a packed `Vec<f64>`.
Their accessor methods make the representation unambiguous:

```rust
use topohedral_integrate::{
    FixedQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
};

let rule = GaussRule::for_degree(
    GaussFamily::Lobatto,
    PolynomialDegree::new(7).unwrap(),
)
.unwrap();
let quadrature =
    FixedQuadrature1d::builder(Interval::new(-1.0, 1.0).unwrap(), rule).build();

for node in &quadrature {
    assert!(node.point().is_finite());
    assert!(node.weight() > 0.0);
}
```

The node structs have the same contiguous floating-point layout and byte
footprint as the old packed pairs and triples. `integrate` accepts `FnMut`, so
closures may retain mutable state. It returns
`IntegrationError::NonFiniteIntegrand` as soon as the integrand produces NaN
or infinity, including the offending coordinate and value.
