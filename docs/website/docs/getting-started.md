# Getting Started

## Installation

Packages are published through the TopoHedral Labs Cloudsmith registry. Add
the registry to `.cargo/config.toml`:

```toml
[registries.cloudsmith]
index = "sparse+https://cargo.cloudsmith.io/topohedrallabs/topohedral/"
credential-provider = "cargo:token"
```

Provide `CARGO_REGISTRIES_CLOUDSMITH_TOKEN`, then add the crate to
`Cargo.toml`:

```toml
[dependencies]
topohedral-integrate = { version = "0.1", registry = "cloudsmith" }
```

The crate has no default features. Enable `serde` for validated serialization
and deserialization, and `trace` for instrumentation without terminal colour:

```toml
[dependencies]
topohedral-integrate = {
    version = "0.1",
    registry = "cloudsmith",
    features = ["serde", "trace"],
}
```

## A first integral

The following example integrates \(x^2\) over \([-1, 1]\) with a reusable
five-point Gauss-Legendre rule:

```rust
use topohedral_integrate::{
    FixedQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let domain = Interval::new(-1.0, 1.0)?;
let degree = PolynomialDegree::new(9)?;
let rule = GaussRule::for_degree(GaussFamily::Legendre, degree)?;
let quadrature = FixedQuadrature1d::builder(domain, rule).build();
let integral = quadrature.integrate(|x: f64| x.powi(2))?;

assert!((integral - 2.0 / 3.0).abs() < 1e-12);
# Ok(())
# }
```

`order` is the maximum polynomial degree for which the underlying Gaussian
rule is designed to be exact. Here, Legendre order 9 selects five quadrature
points. See the [user guide](user-guide/index.md) for rule generation,
two-dimensional integration, subdivisions, and adaptive integration.

Version 0.1.0 deliberately breaks the 0.0.x API. The
[migration guide](migration-0.1.md) maps every removed root symbol to its
replacement.
