use approx::assert_abs_diff_eq;
use std::mem::{align_of, size_of};
use topohedral_integrate::{
    EvaluationPoint, FixedNode1D, FixedNode2D, FixedQuadrature1D, FixedQuadrature2D, GaussFamily,
    GaussRule, IntegrationError, Interval, PointCount, PolynomialDegree, Rectangle, TensorRule2D,
};

fn degree(value: usize) -> PolynomialDegree {
    PolynomialDegree::new(value).unwrap()
}

fn count(value: usize) -> PointCount {
    PointCount::new(value).unwrap()
}

#[test]
fn typed_nodes_have_the_same_compact_layout_as_packed_f64_values() {
    assert_eq!(size_of::<FixedNode1D>(), 2 * size_of::<f64>());
    assert_eq!(align_of::<FixedNode1D>(), align_of::<f64>());
    assert_eq!(size_of::<FixedNode2D>(), 3 * size_of::<f64>());
    assert_eq!(align_of::<FixedNode2D>(), align_of::<f64>());
}

#[test]
fn one_dimensional_builder_maps_and_iterates_typed_nodes() {
    let domain = Interval::new(2.0, 6.0).unwrap();
    let rule = GaussRule::for_degree(GaussFamily::Legendre, degree(1)).unwrap();
    let quadrature = FixedQuadrature1D::builder(domain, rule)
        .subdivisions([3.0, 4.5])
        .unwrap()
        .build();

    assert_eq!(quadrature.domain(), domain);
    assert_eq!(quadrature.subdivision_points(), &[3.0, 4.5]);
    assert_eq!(quadrature.point_count(), 3);
    assert_eq!(quadrature.iter().count(), quadrature.nodes().len());
    assert_eq!((&quadrature).into_iter().count(), quadrature.point_count());

    let points: Vec<_> = quadrature.iter().map(|node| node.point()).collect();
    let weights: Vec<_> = quadrature.iter().map(|node| node.weight()).collect();
    assert_eq!(points, vec![2.5, 3.75, 5.25]);
    assert_eq!(weights, vec![1.0, 1.5, 1.5]);
}

#[test]
fn one_dimensional_integration_accepts_fn_mut_and_reports_nonfinite_values() {
    let domain = Interval::new(-1.0, 1.0).unwrap();
    let rule = GaussRule::for_degree(GaussFamily::Legendre, degree(9)).unwrap();
    let quadrature = FixedQuadrature1D::builder(domain, rule).build();
    let mut evaluations = 0;

    let integral = quadrature
        .integrate(|x| {
            evaluations += 1;
            x.powi(4)
        })
        .unwrap();
    assert_eq!(evaluations, quadrature.point_count());
    assert_abs_diff_eq!(integral, 2.0 / 5.0, epsilon = 1e-14);

    let failing_point = quadrature.nodes()[2].point();
    let error = quadrature
        .integrate(|x| if x == failing_point { f64::NAN } else { x })
        .unwrap_err();
    assert!(matches!(
        error,
        IntegrationError::NonFiniteIntegrand {
            point: EvaluationPoint::OneDimensional(point),
            value,
        } if point == failing_point && value.is_nan()
    ));
}

#[test]
fn integrate_over_remaps_subdivisions_proportionally() {
    let rule = GaussRule::with_point_count(GaussFamily::Lobatto, count(2)).unwrap();
    let quadrature = FixedQuadrature1D::builder(Interval::new(0.0, 10.0).unwrap(), rule)
        .subdivisions([2.0, 7.0])
        .unwrap()
        .build();
    let mut evaluated_points = Vec::new();

    let integral = quadrature
        .integrate_over(Interval::new(100.0, 200.0).unwrap(), |x| {
            evaluated_points.push(x);
            1.0
        })
        .unwrap();

    assert_abs_diff_eq!(integral, 100.0, epsilon = 1e-13);
    assert_eq!(
        evaluated_points,
        vec![100.0, 120.0, 120.0, 170.0, 170.0, 200.0]
    );
}

#[test]
fn two_dimensional_builder_maps_typed_tensor_nodes() {
    let rectangle = Rectangle::new(
        Interval::new(0.0, 2.0).unwrap(),
        Interval::new(-1.0, 1.0).unwrap(),
    );
    let tensor_rule = TensorRule2D::new(
        GaussRule::for_degree(GaussFamily::Legendre, degree(3)).unwrap(),
        GaussRule::for_degree(GaussFamily::Legendre, degree(3)).unwrap(),
    );
    let quadrature = FixedQuadrature2D::builder(rectangle, tensor_rule)
        .subdivisions([1.0], std::iter::empty())
        .unwrap()
        .build();
    let mut evaluations = 0;

    let integral = quadrature
        .integrate(|u, v| {
            evaluations += 1;
            u * u + v * v
        })
        .unwrap();

    assert_eq!(quadrature.domain(), rectangle);
    assert_eq!(quadrature.u_subdivision_points(), &[1.0]);
    assert!(quadrature.v_subdivision_points().is_empty());
    assert_eq!(evaluations, quadrature.point_count());
    assert_eq!(quadrature.iter().count(), quadrature.nodes().len());
    assert_abs_diff_eq!(integral, 20.0 / 3.0, epsilon = 1e-13);
}

#[test]
fn two_dimensional_integration_reports_the_evaluation_coordinates() {
    let rectangle = Rectangle::new(
        Interval::new(-1.0, 1.0).unwrap(),
        Interval::new(-1.0, 1.0).unwrap(),
    );
    let tensor_rule = TensorRule2D::new(
        GaussRule::for_degree(GaussFamily::Legendre, degree(1)).unwrap(),
        GaussRule::for_degree(GaussFamily::Legendre, degree(1)).unwrap(),
    );
    let quadrature = FixedQuadrature2D::builder(rectangle, tensor_rule).build();

    let error = quadrature.integrate(|_, _| f64::INFINITY).unwrap_err();
    assert!(matches!(
        error,
        IntegrationError::NonFiniteIntegrand {
            point: EvaluationPoint::TwoDimensional { u: 0.0, v: 0.0 },
            value,
        } if value.is_infinite()
    ));
}
