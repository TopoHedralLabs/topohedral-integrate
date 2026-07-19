# Fixed Quadrature

A fixed quadrature rule maps Gaussian points and weights from \([-1, 1]\) onto
the requested integration interval or rectangle. Constructing a `FixedQuad1D`
or `FixedQuad2D` does this mapping once, so the rule can be reused to integrate
several functions over the same domain.

## One-dimensional entry point

Configure `FixedQuad1D` with `FixedQuadOpts1D`:

- `gauss_type`: `GaussQuadType::Legendre` or `GaussQuadType::Lobatto`;
- `order`: polynomial order of the Gaussian rule;
- `bounds`: `(lower, upper)`;
- `subdiv`: optional strictly interior subdivision points.

`FixedQuad1D::new` precomputes the mapped points and weights.
`FixedQuad1D::integrate(&f, None)` then integrates over the configured bounds,
and `nqp()` reports the total point count, including all subdivisions.

The polynomial tests use the same pattern:

```rust
use topohedral_integrate::{
    FixedQuad1D, FixedQuadOpts1D, GaussQuadType, OptionsVerify,
};

let opts = FixedQuadOpts1D {
    gauss_type: GaussQuadType::Legendre,
    order: 9,
    bounds: (-2.0, 3.0),
    subdiv: Some(vec![0.0]),
};
opts.is_ok(true).expect("valid fixed-quadrature options");

let rule = FixedQuad1D::new(&opts);
let integral = rule.integrate(&|x: f64| x.powi(4), None);

assert!((integral - 55.0).abs() < 1e-12);
```

Subdivisions split the configured range before applying the rule. They are
useful when a function is only piecewise smooth. Do not include the outer
bounds in `subdiv`.

The optional bounds passed to `integrate` reuse the same rule on a different
interval:

```rust
use topohedral_integrate::{FixedQuad1D, FixedQuadOpts1D, GaussQuadType};

let rule = FixedQuad1D::new(&FixedQuadOpts1D {
    gauss_type: GaussQuadType::Legendre,
    order: 9,
    bounds: (-1.0, 1.0),
    subdiv: None,
});
let integral = rule.integrate(&|x: f64| x.powi(2), Some((0.0, 2.0)));
assert!((integral - 8.0 / 3.0).abs() < 1e-12);
```

## Two-dimensional entry point

`FixedQuad2D` is the tensor-product counterpart. Its `FixedQuadOpts2D` fields
use one value per coordinate direction:

- `gauss_type`: `(u_rule, v_rule)`;
- `order`: `(u_order, v_order)`;
- `bounds`: `(u_min, u_max, v_min, v_max)`;
- `subdiv`: optional `(u_subdivisions, v_subdivisions)` vectors.

The function supplied to `integrate` has type `Fn(f64, f64) -> f64`:

```rust
use topohedral_integrate::{
    FixedQuad2D, FixedQuadOpts2D, GaussQuadType, OptionsVerify,
};

let opts = FixedQuadOpts2D {
    gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
    order: (5, 5),
    bounds: (-1.0, 1.0, -1.0, 1.0),
    subdiv: None,
};
opts.is_ok(true).expect("valid fixed-quadrature options");

let rule = FixedQuad2D::new(&opts);
let integral = rule.integrate(&|x: f64, y: f64| x.powi(2) * y.powi(2), None);

assert!((integral - 4.0 / 9.0).abs() < 1e-12);
```

For a subdivided rectangle, either subdivision vector may be empty, but they
cannot both be empty. For example, `Some((vec![0.0], vec![]))` splits only the
\(u\) direction.

## One-shot helpers

When a rule will be used only once, `fixed_quad_1d` and `fixed_quad_2d`
construct the corresponding fixed rule and immediately integrate the function:

```rust
use topohedral_integrate::{fixed_quad_1d, FixedQuadOpts1D, GaussQuadType};

let opts = FixedQuadOpts1D {
    gauss_type: GaussQuadType::Lobatto,
    order: 7,
    bounds: (-1.0, 1.0),
    subdiv: None,
};

let integral = fixed_quad_1d(&|x: f64| x.powi(4), &opts);
assert!((integral - 2.0 / 5.0).abs() < 1e-12);
```

Prefer the structs when point generation should be amortized across multiple
integrals.
