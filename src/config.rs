//! Validated scalar, domain, and configuration values.

use std::error::Error;
use std::fmt;

/// Largest polynomial degree representable by [`PolynomialDegree`].
///
/// Requested Gaussian rules are currently limited to degree 100, but their actual exactness can
/// be degree 101.
const MAX_POLYNOMIAL_DEGREE: usize = 101;

/// Largest point count representable by [`PointCount`].
const MAX_POINT_COUNT: usize = 52;

/// Coordinate whose Gaussian rule failed adaptive rule-pair validation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RuleAxis {
    /// The single coordinate of a one-dimensional rule.
    OneDimensional,
    /// The `u` coordinate of a tensor-product rule.
    U,
    /// The `v` coordinate of a tensor-product rule.
    V,
}

impl fmt::Display for RuleAxis {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::OneDimensional => formatter.write_str("one-dimensional"),
            Self::U => formatter.write_str("u-axis"),
            Self::V => formatter.write_str("v-axis"),
        }
    }
}

/// A single invalid configuration value.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ConfigIssue {
    /// A polynomial degree lies outside the representable range.
    PolynomialDegreeOutOfRange {
        /// Rejected value.
        degree: usize,
        /// Largest representable value.
        maximum: usize,
    },
    /// A point count lies outside the representable range.
    PointCountOutOfRange {
        /// Rejected value.
        point_count: usize,
        /// Smallest representable value.
        minimum: usize,
        /// Largest representable value.
        maximum: usize,
    },
    /// One or both interval bounds are non-finite.
    NonFiniteInterval {
        /// Rejected lower bound.
        lower: f64,
        /// Rejected upper bound.
        upper: f64,
    },
    /// Interval bounds are not strictly increasing.
    InvalidIntervalOrder {
        /// Rejected lower bound.
        lower: f64,
        /// Rejected upper bound.
        upper: f64,
    },
    /// Absolute and relative tolerances do not form a valid tolerance.
    InvalidTolerance {
        /// Rejected absolute tolerance.
        absolute: f64,
        /// Rejected relative tolerance.
        relative: f64,
    },
    /// An adaptive high-order rule is not more exact than its low-order rule.
    NonIncreasingRuleExactness {
        /// Coordinate to which the rule pair applies.
        axis: RuleAxis,
        /// Actual exactness of the low-order rule.
        low: usize,
        /// Actual exactness of the high-order rule.
        high: usize,
    },
    /// A subdivision coordinate is non-finite.
    NonFiniteSubdivision {
        /// Position in the supplied iterator.
        index: usize,
        /// Rejected coordinate.
        value: f64,
    },
    /// A subdivision coordinate is not strictly inside its interval.
    SubdivisionOutsideInterval {
        /// Position in the supplied iterator.
        index: usize,
        /// Rejected coordinate.
        value: f64,
        /// Interval lower bound.
        lower: f64,
        /// Interval upper bound.
        upper: f64,
    },
    /// Consecutive subdivision coordinates are not strictly increasing.
    SubdivisionsNotStrictlyIncreasing {
        /// Position of the earlier coordinate.
        previous_index: usize,
        /// Earlier coordinate.
        previous: f64,
        /// Position of the later coordinate.
        index: usize,
        /// Later coordinate.
        value: f64,
    },
}

impl fmt::Display for ConfigIssue {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::PolynomialDegreeOutOfRange { degree, maximum } => write!(
                formatter,
                "polynomial degree {degree} exceeds the representable maximum of {maximum}"
            ),
            Self::PointCountOutOfRange {
                point_count,
                minimum,
                maximum,
            } => write!(
                formatter,
                "point count {point_count} is outside the representable range {minimum}..={maximum}"
            ),
            Self::NonFiniteInterval { lower, upper } => write!(
                formatter,
                "interval bounds must be finite, received ({lower}, {upper})"
            ),
            Self::InvalidIntervalOrder { lower, upper } => write!(
                formatter,
                "interval bounds must satisfy lower < upper, received ({lower}, {upper})"
            ),
            Self::InvalidTolerance { absolute, relative } => write!(
                formatter,
                "tolerances must be finite and nonnegative with at least one positive component, received ({absolute}, {relative})"
            ),
            Self::NonIncreasingRuleExactness { axis, low, high } => write!(
                formatter,
                "{axis} high-rule exactness {high} must exceed low-rule exactness {low}"
            ),
            Self::NonFiniteSubdivision { index, value } => write!(
                formatter,
                "subdivision coordinate {index} must be finite, received {value}"
            ),
            Self::SubdivisionOutsideInterval {
                index,
                value,
                lower,
                upper,
            } => write!(
                formatter,
                "subdivision coordinate {index} with value {value} must be strictly inside ({lower}, {upper})"
            ),
            Self::SubdivisionsNotStrictlyIncreasing {
                previous_index,
                previous,
                index,
                value,
            } => write!(
                formatter,
                "subdivision coordinates must be strictly increasing, but coordinate {previous_index} is {previous} and coordinate {index} is {value}"
            ),
        }
    }
}

/// One or more invalid configuration values.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfigError {
    issues: Vec<ConfigIssue>,
}

impl ConfigError {
    fn one(issue: ConfigIssue) -> Self {
        Self {
            issues: vec![issue],
        }
    }

    pub(crate) fn from_issues(issues: Vec<ConfigIssue>) -> Self {
        debug_assert!(!issues.is_empty());
        Self { issues }
    }

    /// Returns every issue discovered during validation.
    pub fn issues(&self) -> &[ConfigIssue] {
        &self.issues
    }
}

impl fmt::Display for ConfigError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "invalid configuration")?;
        for issue in &self.issues {
            write!(formatter, "; {issue}")?;
        }
        Ok(())
    }
}

impl Error for ConfigError {}

/// A validated polynomial degree.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PolynomialDegree(usize);

impl PolynomialDegree {
    /// Largest representable degree.
    pub const MAXIMUM: usize = MAX_POLYNOMIAL_DEGREE;

    /// Validates and constructs a polynomial degree.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when `degree` exceeds [`Self::MAXIMUM`].
    pub fn new(degree: usize) -> Result<Self, ConfigError> {
        if degree <= Self::MAXIMUM {
            Ok(Self(degree))
        } else {
            Err(ConfigError::one(ConfigIssue::PolynomialDegreeOutOfRange {
                degree,
                maximum: Self::MAXIMUM,
            }))
        }
    }

    pub(crate) const fn new_unchecked(degree: usize) -> Self {
        Self(degree)
    }

    /// Returns the underlying degree.
    pub const fn value(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for PolynomialDegree {
    type Error = ConfigError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<PolynomialDegree> for usize {
    fn from(value: PolynomialDegree) -> Self {
        value.value()
    }
}

impl fmt::Display for PolynomialDegree {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A validated Gaussian point count.
///
/// The generic range covers both supported families. A constructor taking a [`crate::GaussFamily`]
/// performs the remaining family-specific validation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PointCount(usize);

impl PointCount {
    /// Smallest representable point count.
    pub const MINIMUM: usize = 1;
    /// Largest representable point count.
    pub const MAXIMUM: usize = MAX_POINT_COUNT;

    /// Validates and constructs a point count.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when `point_count` is outside
    /// [`Self::MINIMUM`]`..=`[`Self::MAXIMUM`].
    pub fn new(point_count: usize) -> Result<Self, ConfigError> {
        if (Self::MINIMUM..=Self::MAXIMUM).contains(&point_count) {
            Ok(Self(point_count))
        } else {
            Err(ConfigError::one(ConfigIssue::PointCountOutOfRange {
                point_count,
                minimum: Self::MINIMUM,
                maximum: Self::MAXIMUM,
            }))
        }
    }

    pub(crate) const fn new_unchecked(point_count: usize) -> Self {
        Self(point_count)
    }

    /// Returns the underlying point count.
    pub const fn value(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for PointCount {
    type Error = ConfigError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<PointCount> for usize {
    fn from(value: PointCount) -> Self {
        value.value()
    }
}

impl fmt::Display for PointCount {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A finite, nonempty interval with strictly increasing bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    /// Validates and constructs an interval.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] unless both bounds are finite and `lower < upper`.
    pub fn new(
        lower: f64,
        upper: f64,
    ) -> Result<Self, ConfigError> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(ConfigError::one(ConfigIssue::NonFiniteInterval {
                lower,
                upper,
            }));
        }
        if lower >= upper {
            return Err(ConfigError::one(ConfigIssue::InvalidIntervalOrder {
                lower,
                upper,
            }));
        }
        Ok(Self { lower, upper })
    }

    pub(crate) const fn new_unchecked(
        lower: f64,
        upper: f64,
    ) -> Self {
        Self { lower, upper }
    }

    /// Returns the lower bound.
    pub const fn lower(self) -> f64 {
        self.lower
    }

    /// Returns the upper bound.
    pub const fn upper(self) -> f64 {
        self.upper
    }

    /// Returns the interval length.
    pub fn length(self) -> f64 {
        self.upper - self.lower
    }

    /// Returns the interval bounds as a tuple.
    pub const fn bounds(self) -> (f64, f64) {
        (self.lower, self.upper)
    }
}

impl fmt::Display for Interval {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "[{}, {}]", self.lower, self.upper)
    }
}

/// A rectangle composed of validated `u` and `v` intervals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rectangle {
    u: Interval,
    v: Interval,
}

impl Rectangle {
    /// Constructs a rectangle from validated axis intervals.
    pub const fn new(
        u: Interval,
        v: Interval,
    ) -> Self {
        Self { u, v }
    }

    /// Validates four bounds and constructs a rectangle.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] containing the invalid `u` and/or `v` interval issues.
    pub fn from_bounds(
        u_lower: f64,
        u_upper: f64,
        v_lower: f64,
        v_upper: f64,
    ) -> Result<Self, ConfigError> {
        let u = Interval::new(u_lower, u_upper);
        let v = Interval::new(v_lower, v_upper);

        match (u, v) {
            (Ok(u), Ok(v)) => Ok(Self::new(u, v)),
            (Err(u_error), Err(v_error)) => {
                let mut issues = u_error.issues;
                issues.extend(v_error.issues);
                Err(ConfigError::from_issues(issues))
            }
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    /// Returns the `u` interval.
    pub const fn u(self) -> Interval {
        self.u
    }

    /// Returns the `v` interval.
    pub const fn v(self) -> Interval {
        self.v
    }
}

impl fmt::Display for Rectangle {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "{} × {}", self.u, self.v)
    }
}

/// Absolute and relative error tolerances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    absolute: f64,
    relative: f64,
}

impl Tolerance {
    /// Validates and constructs an absolute-plus-relative tolerance.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] unless both components are finite and nonnegative and at least one
    /// component is positive.
    pub fn new(
        absolute: f64,
        relative: f64,
    ) -> Result<Self, ConfigError> {
        if !absolute.is_finite()
            || !relative.is_finite()
            || absolute < 0.0
            || relative < 0.0
            || (absolute == 0.0 && relative == 0.0)
        {
            return Err(ConfigError::one(ConfigIssue::InvalidTolerance {
                absolute,
                relative,
            }));
        }
        Ok(Self { absolute, relative })
    }

    /// Constructs a purely absolute tolerance.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] unless `absolute` is finite and positive.
    pub fn absolute(absolute: f64) -> Result<Self, ConfigError> {
        Self::new(absolute, 0.0)
    }

    /// Constructs a purely relative tolerance.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] unless `relative` is finite and positive.
    pub fn relative(relative: f64) -> Result<Self, ConfigError> {
        Self::new(0.0, relative)
    }

    /// Returns the absolute component.
    pub const fn absolute_value(self) -> f64 {
        self.absolute
    }

    /// Returns the relative component.
    pub const fn relative_value(self) -> f64 {
        self.relative
    }
}

/// Maximum adaptive-refinement depth.
///
/// A depth of zero evaluates the initial regions without permitting refinement.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RefinementDepth(usize);

impl RefinementDepth {
    /// Default maximum refinement depth used by adaptive builders.
    pub const DEFAULT: usize = 32;

    /// Constructs a refinement depth. Zero is valid.
    pub const fn new(depth: usize) -> Self {
        Self(depth)
    }

    /// Returns the underlying depth.
    pub const fn value(self) -> usize {
        self.0
    }
}

impl Default for RefinementDepth {
    fn default() -> Self {
        Self::new(Self::DEFAULT)
    }
}

impl From<usize> for RefinementDepth {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl From<RefinementDepth> for usize {
    fn from(value: RefinementDepth) -> Self {
        value.value()
    }
}

impl fmt::Display for RefinementDepth {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Maximum adaptive-refinement depths for two axes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AxisDepths {
    u: RefinementDepth,
    v: RefinementDepth,
}

impl AxisDepths {
    /// Constructs independent `u` and `v` refinement depths.
    pub const fn new(
        u: RefinementDepth,
        v: RefinementDepth,
    ) -> Self {
        Self { u, v }
    }

    /// Constructs independent axis depths from their underlying integer values.
    pub const fn from_values(
        u: usize,
        v: usize,
    ) -> Self {
        Self::new(RefinementDepth::new(u), RefinementDepth::new(v))
    }

    /// Constructs equal depths for both axes.
    pub const fn uniform(depth: RefinementDepth) -> Self {
        Self::new(depth, depth)
    }

    /// Returns the `u`-axis depth.
    pub const fn u(self) -> RefinementDepth {
        self.u
    }

    /// Returns the `v`-axis depth.
    pub const fn v(self) -> RefinementDepth {
        self.v
    }
}

impl Default for AxisDepths {
    fn default() -> Self {
        Self::uniform(RefinementDepth::default())
    }
}

impl From<(RefinementDepth, RefinementDepth)> for AxisDepths {
    fn from((u, v): (RefinementDepth, RefinementDepth)) -> Self {
        Self::new(u, v)
    }
}

pub(crate) fn validate_subdivisions<I>(
    interval: Interval,
    points: I,
) -> Result<Vec<f64>, ConfigError>
where
    I: IntoIterator<Item = f64>,
{
    let points: Vec<f64> = points.into_iter().collect();
    if points.is_empty() {
        return Ok(points);
    }

    let mut issues = Vec::new();
    for (index, value) in points.iter().copied().enumerate() {
        if !value.is_finite() {
            issues.push(ConfigIssue::NonFiniteSubdivision { index, value });
        } else if value <= interval.lower || value >= interval.upper {
            issues.push(ConfigIssue::SubdivisionOutsideInterval {
                index,
                value,
                lower: interval.lower,
                upper: interval.upper,
            });
        }

        if index > 0 {
            let previous = points[index - 1];
            if previous.is_finite() && value.is_finite() && value <= previous {
                issues.push(ConfigIssue::SubdivisionsNotStrictlyIncreasing {
                    previous_index: index - 1,
                    previous,
                    index,
                    value,
                });
            }
        }
    }

    if issues.is_empty() {
        Ok(points)
    } else {
        Err(ConfigError::from_issues(issues))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subdivision_validation_accepts_any_iterator_and_normalizes_empty_input() {
        let interval = Interval::new(-1.0, 1.0).unwrap();
        assert_eq!(
            validate_subdivisions(interval, [-0.5, 0.0, 0.5]),
            Ok(vec![-0.5, 0.0, 0.5])
        );
        assert_eq!(
            validate_subdivisions(interval, std::iter::empty()),
            Ok(Vec::new())
        );
    }

    #[test]
    fn subdivision_validation_collects_all_issues() {
        let interval = Interval::new(-1.0, 1.0).unwrap();
        let error = validate_subdivisions(interval, [0.5, 0.5, f64::NAN, 2.0]).unwrap_err();
        assert_eq!(error.issues().len(), 3);
        assert!(matches!(
            error.issues()[0],
            ConfigIssue::SubdivisionsNotStrictlyIncreasing { .. }
        ));
        assert!(matches!(
            error.issues()[1],
            ConfigIssue::NonFiniteSubdivision { .. }
        ));
        assert!(matches!(
            error.issues()[2],
            ConfigIssue::SubdivisionOutsideInterval { .. }
        ));
    }
}
