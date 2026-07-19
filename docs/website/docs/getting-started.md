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
use topohedral_integrate::{FixedQuad1D, FixedQuadOpts1D, GaussQuadType};

let opts = FixedQuadOpts1D {
    gauss_type: GaussQuadType::Legendre,
    order: 9,
    bounds: (-1.0, 1.0),
    subdiv: None,
};
let rule = FixedQuad1D::new(opts).expect("valid quadrature options");
let integral = rule.integrate(&|x: f64| x.powi(2), None);

assert!((integral - 2.0 / 3.0).abs() < 1e-12);
```

`order` is the maximum polynomial degree for which the underlying Gaussian
rule is designed to be exact. Here, Legendre order 9 selects five quadrature
points. See the [user guide](user-guide/index.md) for rule generation,
two-dimensional integration, subdivisions, and adaptive integration.
