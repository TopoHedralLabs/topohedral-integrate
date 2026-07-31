use criterion::{criterion_group, criterion_main, Criterion};
use std::{hint::black_box, time::Duration};
use topohedral_integrate::{
    legendre_rules, FixedQuadrature1D, FixedQuadrature2D, Interval, PolynomialDegree, Rectangle,
    TensorRule2D,
};

fn subdivisions() -> Vec<f64> {
    (1..16).map(|index| -1.0 + index as f64 / 8.0).collect()
}

fn fixed_1d(subdivided: bool) -> FixedQuadrature1D {
    let degree = PolynomialDegree::new(100).unwrap();
    let rule = legendre_rules()
        .unwrap()
        .get_for_degree(degree)
        .unwrap()
        .clone();
    let builder = FixedQuadrature1D::builder(Interval::new(-1.0, 1.0).unwrap(), rule);
    if subdivided {
        builder.subdivisions(subdivisions()).unwrap().build()
    } else {
        builder.build()
    }
}

fn fixed_2d() -> FixedQuadrature2D {
    let degree = PolynomialDegree::new(50).unwrap();
    let rule = legendre_rules().unwrap().get_for_degree(degree).unwrap();
    let tensor_rule = TensorRule2D::new(rule.clone(), rule.clone());
    FixedQuadrature2D::builder(
        Rectangle::new(
            Interval::new(-1.0, 1.0).unwrap(),
            Interval::new(-1.0, 1.0).unwrap(),
        ),
        tensor_rule,
    )
    .build()
}

fn benchmarks(criterion: &mut Criterion) {
    let rule_1d = fixed_1d(false);
    let subdivided_rule_1d = fixed_1d(true);
    let rule_2d = fixed_2d();
    let remapped_domain = Interval::new(2.0, 5.0).unwrap();

    criterion.bench_function("fixed/integrate_1d/51_points/cheap", |bencher| {
        bencher.iter(|| {
            black_box(
                rule_1d
                    .integrate(|x| x.mul_add(x, 0.5f64.mul_add(x, 1.0)))
                    .unwrap(),
            )
        });
    });

    criterion.bench_function("fixed/integrate_1d/816_points/cheap", |bencher| {
        bencher.iter(|| {
            black_box(
                subdivided_rule_1d
                    .integrate(|x| x.mul_add(x, 0.5f64.mul_add(x, 1.0)))
                    .unwrap(),
            )
        });
    });

    criterion.bench_function("fixed/integrate_1d/51_points/transcendental", |bencher| {
        bencher.iter(|| black_box(rule_1d.integrate(|x| x.sin() * (-x * x).exp()).unwrap()));
    });

    criterion.bench_function("fixed/integrate_1d/816_points/remapped", |bencher| {
        bencher.iter(|| {
            black_box(
                subdivided_rule_1d
                    .integrate_over(remapped_domain, |x| x.mul_add(x, 0.5f64.mul_add(x, 1.0)))
                    .unwrap(),
            )
        });
    });

    criterion.bench_function("fixed/integrate_2d/676_points/cheap", |bencher| {
        bencher.iter(|| {
            black_box(
                rule_2d
                    .integrate(|u, v| u.mul_add(u, v.mul_add(v, u * v)))
                    .unwrap(),
            )
        });
    });

    criterion.bench_function("fixed/construct_1d/816_points", |bencher| {
        bencher.iter(|| black_box(fixed_1d(true)));
    });

    criterion.bench_function("fixed/construct_2d/676_points", |bencher| {
        bencher.iter(|| black_box(fixed_2d()));
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(100);
    targets = benchmarks
}
criterion_main!(benches);
