#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]

use approx::assert_relative_eq;
use serde::Deserialize;
use std::fs;
use topohedral_integrate::gauss::*;

const MAX_REL: f64 = 1e-10;

#[derive(Deserialize)]
struct GaussQuadTest4 {
    points: Vec<f64>,
    weights: Vec<f64>,
}

#[derive(Deserialize)]
struct GaussQuadTest3 {
    n2: GaussQuadTest4,
    n3: GaussQuadTest4,
    n4: GaussQuadTest4,
    n5: GaussQuadTest4,
    n6: GaussQuadTest4,
    n11: GaussQuadTest4,
    n26: GaussQuadTest4,
    n37: GaussQuadTest4,
}

#[derive(Deserialize)]
struct GaussQuadTest2 {
    values: GaussQuadTest3,
}

#[derive(Deserialize)]
struct GaussQuadTest1 {
    legendre: GaussQuadTest2,
    lobatto: GaussQuadTest2,
}

impl GaussQuadTest1 {
    fn new() -> Self {
        let json_file = fs::read_to_string("assets/gauss-quad.json").expect("Unable to read file");
        serde_json::from_str(&json_file).expect("Could not deserialize")
    }
}

macro_rules! legendre_test {
    ($test_name: ident, $dataset: ident, $idx: expr) => {
        #[test]
        fn $test_name() {
            let test_data = GaussQuadTest1::new();
            let leg = GuassQuadSet::new(GaussQuadType::Legendre, 90);

            let points1 = test_data.legendre.values.$dataset.points;
            let weights1 = test_data.legendre.values.$dataset.weights;
            let points2 = leg.points[$idx].clone();
            let weights2 = leg.weights[$idx].clone();
            assert_eq!(points1.len(), points2.len());
            for i in 0..points1.len() {
                assert_relative_eq!(points1[i], points2[i], epsilon = MAX_REL);
                assert_relative_eq!(weights1[i], weights2[i], epsilon = MAX_REL);
            }
        }
    };
}
legendre_test!(legendre_test1, n2, 0);
legendre_test!(legendre_test2, n3, 1);
legendre_test!(legendre_test3, n4, 2);
legendre_test!(legendre_test4, n5, 3);
legendre_test!(legendre_test5, n6, 4);
legendre_test!(legendre_test6, n11, 9);
legendre_test!(legendre_test7, n26, 24);
legendre_test!(legendre_test8, n37, 35);
//..............................................................................................

macro_rules! lobatto_test {
    ($test_name: ident, $dataset: ident, $idx: expr) => {
        #[test]
        fn $test_name() {
            let test_data = GaussQuadTest1::new();
            let leg = GuassQuadSet::new(GaussQuadType::Lobatto, 90);

            let points1 = test_data.lobatto.values.$dataset.points;
            let weights1 = test_data.lobatto.values.$dataset.weights;
            let points2 = leg.points[$idx].clone();
            let weights2 = leg.weights[$idx].clone();

            assert_eq!(points1.len(), points2.len());
            for i in 0..points1.len() {
                assert_relative_eq!(points1[i], points2[i], epsilon = MAX_REL);
                assert_relative_eq!(weights1[i], weights2[i], epsilon = MAX_REL);
            }
        }
    };
}
lobatto_test!(lobatto_test1, n2, 0);
lobatto_test!(lobatto_test2, n3, 1);
lobatto_test!(lobatto_test3, n4, 2);
lobatto_test!(lobatto_test4, n5, 3);
lobatto_test!(lobatto_test5, n6, 4);
lobatto_test!(lobatto_test6, n11, 9);
lobatto_test!(lobatto_test7, n26, 24);
lobatto_test!(lobatto_test8, n37, 35);
//..............................................................................................
