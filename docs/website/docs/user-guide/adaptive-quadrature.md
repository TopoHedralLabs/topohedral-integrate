# Adaptive Quadrature

Adaptive quadrature compares low- and high-order fixed rules on each current
subdomain. A subdomain is split when the two estimates differ by more than
`tol`. This concentrates evaluations around oscillation, singular behavior, or
loss of smoothness.

The entry points are the functions `adaptive_quad_1d` and
`adaptive_quad_2d`. Each takes a function by reference and a dimension-specific
options struct.

## One-dimensional integration

`AdaptiveQuadOpts1D` contains:

- `bounds`: `(lower, upper)` for the complete integral;
- `fixed_rule_low` and `fixed_rule_high`: reusable `FixedQuad1D` estimators,
  with the low rule having a lower order than the high rule;
- `tol`: the accepted low/high difference on each final subinterval;
- `max_depth`: the subdivision-depth configuration;
- `init_subdiv`: optional known breakpoints inside `bounds`.

The following is based on the piecewise-linear integration test. Supplying the
known corner at \(x=-1\) allows the algorithm to start with smooth pieces:

```rust
use topohedral_integrate::{
    adaptive_quad_1d, AdaptiveQuadOpts1D, FixedQuad1D, FixedQuadOpts1D,
    GaussQuadType, OptionsVerify,
};

fn legendre_rule(order: usize) -> FixedQuad1D {
    FixedQuad1D::new(&FixedQuadOpts1D {
        gauss_type: GaussQuadType::Legendre,
        order,
        bounds: (-1.0, 1.0),
        subdiv: None,
    })
}

let opts = AdaptiveQuadOpts1D {
    bounds: (-3.0, 4.0),
    fixed_rule_low: legendre_rule(10),
    fixed_rule_high: legendre_rule(30),
    tol: 1e-5,
    max_depth: 1000,
    init_subdiv: Some(vec![-1.0]),
};
opts.is_ok(true).expect("valid adaptive options");

let result = adaptive_quad_1d(&|x: f64| (x + 1.0).abs(), &opts);

assert!((result.integral - 29.0 / 2.0).abs() < 1e-10);
assert!(result.error_estimate < result.num_subdiv as f64 * opts.tol);
```

The low and high fixed rules are conventionally built on \([-1, 1]\); the
adaptive routine remaps them to each subinterval.

## Two-dimensional integration

`AdaptiveQuadOpts2D` uses the same design with tensor-product fixed rules:

- `bounds`: `(u_min, u_max, v_min, v_max)`;
- `fixed_rule_low` and `fixed_rule_high`: `FixedQuad2D` values;
- `tol`: the accepted low/high difference on each final rectangle;
- `max_depth`: `(u_depth, v_depth)`;
- `init_subdiv`: optional `(u_breakpoints, v_breakpoints)`.

```rust
use topohedral_integrate::{
    adaptive_quad_2d, AdaptiveQuadOpts2D, FixedQuad2D, FixedQuadOpts2D,
    GaussQuadType, OptionsVerify,
};

fn legendre_rule(order: usize) -> FixedQuad2D {
    FixedQuad2D::new(&FixedQuadOpts2D {
        gauss_type: (GaussQuadType::Legendre, GaussQuadType::Legendre),
        order: (order, order),
        bounds: (-1.0, 1.0, -1.0, 1.0),
        subdiv: None,
    })
}

let opts = AdaptiveQuadOpts2D {
    bounds: (0.0, 1.0, 0.0, 1.0),
    fixed_rule_low: legendre_rule(3),
    fixed_rule_high: legendre_rule(7),
    tol: 1e-8,
    max_depth: (10, 10),
    init_subdiv: None,
};
opts.is_ok(true).expect("valid adaptive options");

let result = adaptive_quad_2d(&|x: f64, y: f64| x.powi(2) + y.powi(2), &opts);
assert!((result.integral - 2.0 / 3.0).abs() < 1e-12);
```

When a discontinuity or corner location is known in advance, provide it through
`init_subdiv`. For example, the test function
`(x + 1.0).abs() * (y - 2.0).abs()` uses
`Some((vec![-1.0], vec![2.0]))`.

## Results and error estimates

`adaptive_quad_1d` returns `AdaptiveQuadResult1D`, and `adaptive_quad_2d`
returns `AdaptiveQuadResult2D`. Both expose:

- `integral`: the sum of the accepted low-order estimates;
- `error_estimate`: the sum of the low/high differences;
- `num_subdiv`: the number of final subdomains;
- `num_fn_eval`: the number of function evaluations.

The tolerance is applied to each final subdomain rather than directly to the
sum. Consequently, a useful conservative bound in the test suite is
`num_subdiv as f64 * tol`.
