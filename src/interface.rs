#![allow(clippy::new_ret_no_self)]
use statrs::distribution::{
    LogNormal as LogNormalDist, Normal as NormalDist, Poisson as PoissonDist,
    Triangular as TriangularDist, Uniform as UniformDist,
};

use crate::distributions::{Constant as ConstantDist, ContinuousDist, DiscreteDist, DistNode};

pub struct Normal {}
impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> DistNode {
        // TODO: Handle nan cases for mean and std_dev
        DistNode::Continuous(ContinuousDist::Normal {
            dist: NormalDist::new(mean, std_dev.abs()).unwrap(),
        })
    }
}

pub struct LogNormal {}
impl LogNormal {
    pub fn new(mean: f64, std_dev: f64) -> DistNode {
        DistNode::Continuous(ContinuousDist::LogNormal {
            dist: LogNormalDist::new(mean, std_dev).unwrap(),
        })
    }
}

pub struct Triangular {}
impl Triangular {
    pub fn new(min: f64, mode: f64, max: f64) -> DistNode {
        DistNode::Continuous(ContinuousDist::Triangular {
            dist: TriangularDist::new(min, max, mode).unwrap(),
        })
    }
}

pub struct Uniform {}
impl Uniform {
    pub fn new(min: f64, max: f64) -> DistNode {
        DistNode::Continuous(ContinuousDist::Uniform {
            dist: UniformDist::new(min, max).unwrap(),
        })
    }
}

pub struct Poisson {}
impl Poisson {
    pub fn new(lambda: f64) -> DistNode {
        DistNode::Discrete(DiscreteDist::Poisson {
            dist: PoissonDist::new(lambda).unwrap(),
        })
    }
}

pub struct Constant {}
impl Constant {
    pub fn new(number: f64) -> DistNode {
        DistNode::Discrete(DiscreteDist::Constant {
            dist: ConstantDist::new(number),
        })
    }
}
