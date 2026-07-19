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

The fixed and adaptive options types implement `OptionsVerify`. Validation is
explicit; call `opts.is_ok(true)` before constructing a rule or starting an
integration when options may come from user input:

```rust
use topohedral_integrate::OptionsVerify;

fn validate<T: OptionsVerify>(opts: &T) {
    opts.is_ok(true).expect("invalid quadrature options");
}
```

Passing `true` asks for a detailed `OptionsError`; passing `false` performs the
short check without building diagnostic text.
