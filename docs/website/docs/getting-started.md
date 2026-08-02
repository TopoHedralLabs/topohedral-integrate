# Getting Started

## Installation

Add the crate and its registry dependency to `Cargo.toml`:

```toml
[dependencies]
topohedral-integrate = "0.0.2"
```

The version above reflects the current development package; use the released
version when the crate is published.

## A first integral

The following example integrates \(x^2\) over \([-1, 1]\) with a reusable
five-point Gauss-Legendre rule:

```rust
use topohedral_integrate::{
    FixedQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
};

let domain = Interval::new(-1.0, 1.0).expect("valid interval");
let degree = PolynomialDegree::new(9).expect("supported degree");
let rule = GaussRule::for_degree(GaussFamily::Legendre, degree)
    .expect("rule generation succeeds");
let quadrature = FixedQuadrature1d::builder(domain, rule).build();
let integral = quadrature
    .integrate(|x: f64| x.powi(2))
    .expect("finite integrand");

assert!((integral - 2.0 / 3.0).abs() < 1e-12);
```

`order` is the maximum polynomial degree for which the underlying Gaussian
rule is designed to be exact. Here, Legendre order 9 selects five quadrature
points. See the [user guide](user-guide/index.md) for rule generation,
two-dimensional integration, subdivisions, and adaptive integration.
