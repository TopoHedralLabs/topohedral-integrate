# User Guide

The crate exposes its complete public API at the crate root. Import symbols
such as `GaussQuad`, `FixedQuad1D`, and `adaptive_quad_2d` directly from
`topohedral_integrate`; the internal `gauss`, `fixed`, and `adaptive` modules
are not part of the public interface.

The API is divided into three areas:

## Gaussian rules

[Gaussian rules](gaussian-rules.md) produce points and weights on the standard
interval \([-1, 1]\). Use `GaussQuad` for one rule, or `GuassQuadSet` when
several orders are needed. Both Gauss-Legendre and Gauss-Lobatto families are
available through `GaussQuadType`.

## Fixed quadrature

[Fixed quadrature](fixed-quadrature.md) maps a Gaussian rule onto a 1D interval
or 2D rectangle. `FixedQuad1D` and `FixedQuad2D` precompute their points and
weights and can be reused for multiple functions.

## Adaptive quadrature

[Adaptive quadrature](adaptive-quadrature.md) repeatedly subdivides intervals
or rectangles whose low- and high-order estimates differ by more than the
requested tolerance. The `adaptive_quad_1d` and `adaptive_quad_2d` functions
return both the integral and diagnostic information.

## Options validation

Options are validated by the public operation that consumes or uses them.
`FixedQuad1D::new`, `FixedQuad2D::new`, the one-shot fixed helpers, and the
adaptive integration functions return `Result` and report invalid options as
an `OptionsError`:

```rust
use topohedral_integrate::{FixedQuad1D, FixedQuadOpts1D, GaussQuadType};

let result = FixedQuad1D::new(FixedQuadOpts1D {
    gauss_type: GaussQuadType::Legendre,
    order: 9,
    bounds: (-1.0, 1.0),
    subdiv: None,
});
assert!(result.is_ok());
```
