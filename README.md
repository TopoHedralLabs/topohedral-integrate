# topohedral-integrate

Validated Gaussian fixed and adaptive quadrature for one- and two-dimensional
real-valued functions.

The crate provides:

- Gauss–Legendre and Gauss–Lobatto rule generation;
- reusable fixed quadrature over validated intervals and rectangles;
- globally error-controlled adaptive quadrature;
- structured configuration, rule-generation, and integration errors; and
- optional Serde support that preserves validation during deserialization.

## Installation

Packages are published through the TopoHedral Labs Cloudsmith registry. Add the
registry to `.cargo/config.toml`:

```toml
[registries.cloudsmith]
index = "sparse+https://cargo.cloudsmith.io/topohedrallabs/topohedral/"
credential-provider = "cargo:token"
```

Provide `CARGO_REGISTRIES_CLOUDSMITH_TOKEN`, then add the dependency:

```toml
[dependencies]
topohedral-integrate = { version = "0.1", registry = "cloudsmith" }
```

Optional features are disabled by default:

```toml
topohedral-integrate = {
    version = "0.1",
    registry = "cloudsmith",
    features = ["serde", "trace"],
}
```

`serde` supports validated serialization and deserialization. `trace` enables
instrumentation in the integration and linear-algebra stack without enabling
terminal colours.

## Example

```rust
use topohedral_integrate::{
    FixedQuadrature1d, GaussFamily, GaussRule, Interval, PolynomialDegree,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Interval::new(-1.0, 1.0)?;
    let rule = GaussRule::for_degree(
        GaussFamily::Legendre,
        PolynomialDegree::new(5)?,
    )?;
    let quadrature = FixedQuadrature1d::builder(domain, rule).build();
    let integral = quadrature.integrate(|x| x * x)?;

    assert!((integral - 2.0 / 3.0).abs() < 1e-12);
    Ok(())
}
```

See the [user guide](https://topohedrallabs.github.io/topohedral-integrate/)
and [API documentation](https://topohedrallabs.github.io/topohedral-integrate/latest/api/topohedral_integrate/).

Version `0.1.0` is a deliberate breaking redesign of the `0.0.x` API. See
[CHANGELOG.md](CHANGELOG.md) for the migration table.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
