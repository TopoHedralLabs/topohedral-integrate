# Validated Values

The public value types reject invalid numerical configuration before a rule or
integrator is built. Their fields are private, so a successfully constructed
value continues to satisfy its invariants.

## Domains

`Interval::new(lower, upper)` requires finite, strictly increasing bounds.
`Rectangle` contains one validated interval for each axis:

```rust
use topohedral_integrate::{Interval, Rectangle};

let u = Interval::new(-1.0, 1.0)?;
let v = Interval::new(2.0, 5.0)?;
let rectangle = Rectangle::new(u, v);

assert_eq!(rectangle.u().bounds(), (-1.0, 1.0));
assert_eq!(rectangle.v().bounds(), (2.0, 5.0));
# Ok::<(), topohedral_integrate::ConfigError>(())
```

Subdivision validation accepts any iterator of coordinates. Empty input means
no subdivisions. Nonempty coordinates must be finite, unique, strictly
increasing, and strictly inside the corresponding interval.

## Tolerances and refinement depths

`Tolerance` contains absolute and relative components. Both must be finite and
nonnegative, and at least one must be positive:

```rust
use topohedral_integrate::{AxisDepths, RefinementDepth, Tolerance};

let tolerance = Tolerance::new(1e-10, 1e-8)?;
assert_eq!(tolerance.absolute_value(), 1e-10);
assert_eq!(tolerance.relative_value(), 1e-8);

let depths = AxisDepths::from_values(0, 12);
assert_eq!(depths.u(), RefinementDepth::new(0));
assert_eq!(depths.v(), RefinementDepth::new(12));
# Ok::<(), topohedral_integrate::ConfigError>(())
```

A refinement depth of zero means that adaptive integration evaluates its
initial regions without refining them. `RefinementDepth::default()` and
`AxisDepths::default()` use depth 32, as do the adaptive builders.

## Rule values

`PolynomialDegree` and `PointCount` distinguish exactness requests from rule
sizes. Gaussian constructors accept these types instead of unvalidated
integers. A `PointCount` covers the combined range of both Gaussian families;
the rule constructor performs the remaining family-specific check.

Validation failures return `ConfigError`. Its `issues()` method exposes every
structured `ConfigIssue` found during validation, including the rejected
values. Multi-axis and compatibility-option validation retains every issue it
can evaluate rather than collapsing them into one string. Error messages are
concise lowercase summaries without trailing punctuation; callers should
match `ConfigIssue` variants when they need programmatic diagnostics.

One-shot compatibility operations can also generate a Gaussian rule or
evaluate an integrand. Their `OptionsError` transparently wraps `ConfigError`,
`RuleError`, or `IntegrationError`; validation failures are always carried by
the `Config` variant without loss of structured issues.
