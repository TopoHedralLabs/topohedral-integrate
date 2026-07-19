# TopoHedral-Integrate

`topohedral-integrate` provides Gaussian quadrature rules and fixed and
adaptive numerical integration in one and two dimensions.

The public API is flat: all user-facing types and functions are imported
directly from `topohedral_integrate`. Dimension-specific names end in `1D` or
`2D`, for example `FixedQuad1D`, `FixedQuad2D`, `adaptive_quad_1d`, and
`adaptive_quad_2d`.

Use the [getting-started guide](getting-started.md) for a first integral, or go
directly to one of the user-guide areas:

- [Gaussian rules](user-guide/gaussian-rules.md) explains how to generate and
  inspect points and weights on the reference interval;
- [fixed quadrature](user-guide/fixed-quadrature.md) covers reusable 1D and 2D
  integration rules;
- [adaptive quadrature](user-guide/adaptive-quadrature.md) covers automatic
  subdivision and error estimates.
