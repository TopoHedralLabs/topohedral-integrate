//{{{ collection: imports
use approx::assert_relative_eq;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use topohedral_integrate::{
    fixed_quad_1d, fixed_quad_2d, FixedQuadrature1d, FixedQuadrature2d, GaussFamily, GaussRule,
    Interval, PolynomialDegree, Rectangle, TensorRule2d,
};

const MAX_RELATIVE_ERROR: f64 = 1e-14;
//}}}

//{{{ collection: test data
#[derive(Deserialize)]
struct Polynomial1d {
    coeffs: Vec<f64>,
    integral: f64,
}

#[derive(Deserialize)]
struct Values1d {
    range: (f64, f64),
    #[serde(flatten)]
    polynomials: BTreeMap<String, Polynomial1d>,
}

#[derive(Deserialize)]
struct Data1d {
    values: Values1d,
}

#[derive(Deserialize)]
struct Polynomial2d {
    coeffs_u: Vec<f64>,
    coeffs_v: Vec<f64>,
    integral: f64,
}

#[derive(Deserialize)]
struct Values2d {
    range: (f64, f64, f64, f64),
    #[serde(flatten)]
    polynomials: BTreeMap<String, Polynomial2d>,
}

#[derive(Deserialize)]
struct Data2d {
    values: Values2d,
}

fn data_1d() -> Data1d {
    let json = fs::read_to_string("assets/poly-integrals-1d.json").expect("read 1d test data");
    serde_json::from_str(&json).expect("deserialize 1d test data")
}

fn data_2d() -> Data2d {
    let json = fs::read_to_string("assets/poly-integrals-2d.json").expect("read 2d test data");
    serde_json::from_str(&json).expect("deserialize 2d test data")
}
//}}}

//{{{ collection: helpers
fn degree(value: usize) -> PolynomialDegree {
    PolynomialDegree::new(value).unwrap()
}

fn builder_1d(
    family: GaussFamily,
    exactness: usize,
    bounds: (f64, f64),
) -> topohedral_integrate::FixedQuadratureBuilder1d {
    let rule = GaussRule::for_degree(family, degree(exactness)).unwrap();
    FixedQuadrature1d::builder(Interval::new(bounds.0, bounds.1).unwrap(), rule)
}

fn builder_2d(
    family: GaussFamily,
    u_exactness: usize,
    v_exactness: usize,
    bounds: (f64, f64, f64, f64),
) -> topohedral_integrate::FixedQuadratureBuilder2d {
    let u_rule = GaussRule::for_degree(family, degree(u_exactness)).unwrap();
    let v_rule = GaussRule::for_degree(family, degree(v_exactness)).unwrap();
    let domain = Rectangle::from_bounds(bounds.0, bounds.1, bounds.2, bounds.3).unwrap();
    FixedQuadrature2d::builder(domain, TensorRule2d::new(u_rule, v_rule))
}
//}}}

//{{{ collection: one-dimensional tests
#[test]
fn one_point_legendre_rule_integrates_linear_function() {
    let integral = fixed_quad_1d(|x| x, builder_1d(GaussFamily::Legendre, 0, (2.0, 5.0))).unwrap();

    assert_relative_eq!(integral, 10.5, epsilon = MAX_RELATIVE_ERROR);
}

#[test]
fn fixed_1d_integrates_polynomial_dataset() {
    let data = data_1d();
    let range = data.values.range;

    for family in [GaussFamily::Legendre, GaussFamily::Lobatto] {
        for polynomial in data.values.polynomials.values() {
            let exactness = polynomial.coeffs.len() - 1;
            let evaluate = |x: f64| {
                polynomial
                    .coeffs
                    .iter()
                    .enumerate()
                    .map(|(power, coefficient)| coefficient * x.powi(power as i32))
                    .sum()
            };

            let integral = fixed_quad_1d(&evaluate, builder_1d(family, exactness, range)).unwrap();
            assert_relative_eq!(
                polynomial.integral,
                integral,
                max_relative = MAX_RELATIVE_ERROR
            );

            let width = (range.1 - range.0) / 3.0;
            let subdivided = builder_1d(family, exactness, range)
                .subdivisions([range.0 + width, range.0 + 2.0 * width])
                .unwrap();
            let integral = fixed_quad_1d(&evaluate, subdivided).unwrap();
            assert_relative_eq!(
                polynomial.integral,
                integral,
                max_relative = MAX_RELATIVE_ERROR
            );
        }
    }
}
//}}}

//{{{ collection: two-dimensional tests
#[test]
fn fixed_2d_integrates_polynomial_dataset() {
    let data = data_2d();
    let range = data.values.range;

    for family in [GaussFamily::Legendre, GaussFamily::Lobatto] {
        for polynomial in data.values.polynomials.values() {
            let u_exactness = polynomial.coeffs_u.len() - 1;
            let v_exactness = polynomial.coeffs_v.len() - 1;
            let evaluate = |u: f64, v: f64| {
                let u_value: f64 = polynomial
                    .coeffs_u
                    .iter()
                    .enumerate()
                    .map(|(power, coefficient)| coefficient * u.powi(power as i32))
                    .sum();
                let v_value: f64 = polynomial
                    .coeffs_v
                    .iter()
                    .enumerate()
                    .map(|(power, coefficient)| coefficient * v.powi(power as i32))
                    .sum();
                u_value * v_value
            };

            let builder = builder_2d(family, u_exactness, v_exactness, range);
            let integral = fixed_quad_2d(&evaluate, builder).unwrap();
            assert_relative_eq!(
                polynomial.integral,
                integral,
                max_relative = MAX_RELATIVE_ERROR
            );

            let u_width = (range.1 - range.0) / 3.0;
            let v_width = (range.3 - range.2) / 3.0;
            let subdivided = builder_2d(family, u_exactness, v_exactness, range)
                .subdivisions(
                    [range.0 + u_width, range.0 + 2.0 * u_width],
                    [range.2 + v_width, range.2 + 2.0 * v_width],
                )
                .unwrap();
            let integral = fixed_quad_2d(&evaluate, subdivided).unwrap();
            assert_relative_eq!(
                polynomial.integral,
                integral,
                max_relative = MAX_RELATIVE_ERROR
            );
        }
    }
}
//}}}
