# Adaptive Quadrature

Adaptive quadrature compares low- and high-order Gaussian rules on each current
region. It returns the high-order estimate and uses the absolute low/high
difference as that region's error estimate. The algorithm repeatedly refines
the splittable region with the largest error until the global condition

$$
E \leq \epsilon_{abs} + \epsilon_{rel}|I|
$$

is satisfied.

## One-dimensional integration

Construct an `AdaptiveQuadrature1D` from a validated interval, low and high
Gaussian rules, and a `Tolerance`:

```rust
use topohedral_integrate::{
    AdaptiveQuadrature1D, GaussFamily, GaussRule, Interval, PolynomialDegree,
    RefinementDepth, Tolerance,
};

let domain = Interval::new(-3.0, 4.0).unwrap();
let low = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(5).unwrap(),
)
.unwrap();
let high = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(11).unwrap(),
)
.unwrap();
let tolerance = Tolerance::new(1e-10, 1e-10).unwrap();
let quadrature = AdaptiveQuadrature1D::builder(domain, low, high, tolerance)
    .max_depth(RefinementDepth::new(20))
    .subdivisions([-1.0])
    .unwrap()
    .build()
    .unwrap();

let result = quadrature.integrate(|x| (x + 1.0).abs()).unwrap();
assert!((result.integral() - 29.0 / 2.0).abs() < 1e-10);
assert!(
    result.error_estimate()
        <= tolerance.absolute_value()
            + tolerance.relative_value() * result.integral().abs()
);
```

Initial subdivisions describe known breakpoints and start at depth zero. The
configured maximum depth is counted independently from each initial region. A
depth of zero evaluates those regions once without permitting refinement.

## Two-dimensional integration

`AdaptiveQuadrature2D` accepts tensor-product rules and independent maximum
depths for the two axes:

```rust
use topohedral_integrate::{
    AdaptiveQuadrature2D, AxisDepths, GaussFamily, GaussRule, PolynomialDegree,
    Rectangle, TensorRule2D, Tolerance,
};

let domain = Rectangle::from_bounds(0.0, 1.0, 0.0, 1.0).unwrap();
let low = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(3).unwrap(),
)
.unwrap();
let high = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(7).unwrap(),
)
.unwrap();
let quadrature = AdaptiveQuadrature2D::builder(
    domain,
    TensorRule2D::new(low.clone(), low),
    TensorRule2D::new(high.clone(), high),
    Tolerance::absolute(1e-10).unwrap(),
)
.max_depth(AxisDepths::from_values(10, 10))
.build()
.unwrap();

let result = quadrature.integrate(|u, v| u * u + v * v).unwrap();
assert!((result.integral() - 2.0 / 3.0).abs() < 1e-12);
```

When both axes of the selected region remain below their depth limits, the
region is split into four. After one axis reaches its limit, refinement
continues by splitting only the other axis.

## One-shot helpers

`adaptive_quad_1d` and `adaptive_quad_2d` are thin wrappers for configurations
that will be used once. Pass the configured builder without calling `build`:

```rust
use topohedral_integrate::{
    adaptive_quad_1d, AdaptiveQuadrature1D, GaussFamily, GaussRule, Interval,
    PolynomialDegree, Tolerance,
};

let low = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(3).unwrap(),
)
.unwrap();
let high = GaussRule::for_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(7).unwrap(),
)
.unwrap();
let builder = AdaptiveQuadrature1D::builder(
    Interval::new(-1.0, 1.0).unwrap(),
    low,
    high,
    Tolerance::absolute(1e-10).unwrap(),
);
let result = adaptive_quad_1d(|x| x * x, builder).unwrap();
assert!((result.integral() - 2.0 / 3.0).abs() < 1e-12);
```

Building an `AdaptiveQuadrature1D` or `AdaptiveQuadrature2D` remains preferable
when its validated rules and configuration will be reused.

## Results and convergence failures

Both dimensions return `AdaptiveResult`, whose accessors provide:

- `integral()`: the sum of terminal high-order estimates;
- `error_estimate()`: the sum of terminal low/high differences;
- `terminal_region_count()`: the number of final subdomains;
- `evaluation_count()`: the exact number of integrand evaluations.

If the global tolerance cannot be met because every applicable depth is
exhausted, integration returns
`IntegrationError::MaxDepthReached { partial_result }`. The partial result has
the same four diagnostics and is the best high-order estimate available.

If an interval is so narrow that its floating-point midpoint equals an
endpoint, integration returns `NonProgressingInterval` or
`NonProgressingRectangle`. NaN and infinity returned by the integrand remain
`NonFiniteIntegrand` errors.
