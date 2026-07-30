# Gaussian Rules

Gaussian quadrature approximates an integral on \([-1, 1]\) with a weighted
sum

\[
    \int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{n} w_i f(x_i).
\]

The crate supports two rule families:

- `GaussFamily::Legendre` uses interior points, supports \(n \ge 1\), and an
  \(n\)-point rule has polynomial exactness \(2n-1\);
- `GaussFamily::Lobatto` includes both interval endpoints, supports \(n \ge 2\),
  and an \(n\)-point rule has polynomial exactness \(2n-3\).

Construction uses the smallest point count whose exactness is at least the
requested degree:

\[
    n_\mathrm{Legendre} = \left\lfloor\frac{p}{2}\right\rfloor + 1,
    \qquad
    n_\mathrm{Lobatto} = \left\lfloor\frac{p}{2}\right\rfloor + 2.
\]

For an even requested degree \(p\), the selected rule has actual exactness
\(p+1\). `GaussRule::exactness` reports that actual value.

## `GaussRule`: one rule

Use `GaussRule::for_degree` to request polynomial exactness, or
`GaussRule::with_point_count` to request an exact number of points. Both
constructors return `Result` and support the same range as the degree-100
cache.

```rust
use topohedral_integrate::{GaussFamily, GaussRule, PolynomialDegree};

let degree = PolynomialDegree::new(9)?;
let rule = GaussRule::for_degree(GaussFamily::Legendre, degree)?;
assert_eq!(rule.point_count().value(), 5);
assert_eq!(rule.exactness().value(), 9);

let integral: f64 = rule
    .points()
    .iter()
    .zip(rule.weights())
    .map(|(&x, &w)| w * x.powi(8))
    .sum();

assert!((integral - 2.0 / 9.0).abs() < 1e-12);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`GaussRule` is a reference-interval rule. Use a fixed quadrature rule to map it
to arbitrary bounds.

## `GaussRuleSet`: a generated family

`GaussRuleSet::through_degree` generates every supported point count through
the rule needed for the requested maximum exactness. Lookups borrow the stored
rules, so selecting a rule does not clone its point and weight vectors.

```rust
use topohedral_integrate::{GaussFamily, GaussRuleSet, PointCount, PolynomialDegree};

let rules = GaussRuleSet::through_degree(
    GaussFamily::Legendre,
    PolynomialDegree::new(90)?,
)?;
let rule = rules.get_by_point_count(PointCount::new(37)?)?;
let same_rule = rules.get_for_degree(PolynomialDegree::new(72)?)?;

assert_eq!(rule.family(), GaussFamily::Legendre);
assert_eq!(rule.point_count().value(), 37);
assert!(std::ptr::eq(rule, same_rule));
assert_eq!(rules.iter().count(), rules.len());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Rule sets accept lookup by requested degree or point count and iterate in
ascending point-count order. They do not support arbitrary insertion because
that would break their single-family, generated-rule invariant.

## Shared cached sets

`legendre_rules()` and `lobatto_rules()` return borrowed, process-wide,
lazily initialized rule sets through requested degree 100. Initialization
errors are retained in the cache and returned deterministically on every call.

```rust
use topohedral_integrate::{lobatto_rules, PointCount};

let rules = lobatto_rules()?;
let rule = rules.get_by_point_count(PointCount::new(5)?)?;
assert_eq!(rule.points().first(), Some(&-1.0));
assert_eq!(rule.points().last(), Some(&1.0));
# Ok::<(), Box<dyn std::error::Error>>(())
```

The fixed quadrature types use these cached sets internally.
