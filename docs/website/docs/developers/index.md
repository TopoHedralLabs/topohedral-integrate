# Developer Notes

The crate is organized around quadrature construction and integration
algorithms:

- `gauss` generates Legendre and Lobatto points and weights;
- `fixed::d1` and `fixed::d2` map Gaussian rules onto intervals and rectangles;
- `adaptive::d1` and `adaptive::d2` refine fixed-rule estimates;
- `common` defines options validation and its error type.

Public APIs are re-exported from `src/lib.rs`, so new user-facing types should be
documented there and tested through the corresponding top-level entry point.
