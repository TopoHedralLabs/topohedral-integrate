# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-02

Version 0.1.0 is a deliberate breaking redesign of the 0.0.x API. Deprecated
aliases are not retained. See the
[complete migration table](docs/website/docs/migration-0.1.md).

### Added

- Validated degrees, point counts, domains, tolerances, and refinement depths.
- Structured configuration, rule-generation, and integration errors.
- Reusable fixed and adaptive quadratures with flat `1d` and `2d` names.
- Optional validated Serde support behind `serde`.
- Global adaptive error control, partial results, terminal-region counts, and
  exact evaluation counts.

### Changed

- Repaired Legendre and Lobatto rule generation, exactness conversion,
  endpoints, cache boundaries, and eigendecomposition error handling.
- Adaptive integration returns the high-order estimate and refines the
  greatest local-error region against a global absolute-plus-relative
  tolerance.
- Public fields are private and exposed through invariant-preserving accessors.
- Typed fixed nodes replace packed point/weight vectors without changing their
  contiguous floating-point layout.
- `trace` replaces `enable_trace`; all crate features are disabled by default.
- The dependency graph uses opt-in tracing and excludes terminal colour.
- The minimum supported Rust version is 1.85.

### Removed

- `GaussQuadType`, `GaussQuad`, `GuassQuadSet`, `get_legendre_points`, and
  `get_lobatto_points`.
- `FixedQuadOpts1D`, `FixedQuadOpts2D`, `FixedQuad1D`, and `FixedQuad2D`.
- `AdaptiveQuadOpts1D`, `AdaptiveQuadOpts2D`, `AdaptiveQuadResult1D`, and
  `AdaptiveQuadResult2D`.
- `OptionsError::InvalidOptionsShort`, `OptionsError::InvalidOptionsFull`, the
  private `OptionsVerify` trait, and the non-root `append_reason` helper.

## [0.0.2]

- Last release of the original freely mutable options and packed-node API.

[0.1.0]: https://github.com/TopoHedralLabs/topohedral-integrate/compare/v0.0.2...v0.1.0
