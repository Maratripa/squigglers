#![allow(clippy::new_ret_no_self)]
use statrs::distribution::{
    LogNormal as LogNormalDist, Normal as NormalDist, Poisson as PoissonDist,
    Triangular as TriangularDist, Uniform as UniformDist,
};

use crate::distributions::{Constant as ConstantDist, ContinuousDist, DiscreteDist, DistNode};
use crate::utils::{ppf, square};

pub struct Normal {}
impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> DistNode {
        // TODO: Handle nan cases for mean and std_dev
        DistNode::Continuous(ContinuousDist::Normal {
            dist: NormalDist::new(mean, std_dev.abs()).unwrap(),
        })
    }

    pub fn from_mean(mean: f64, std_dev: f64) -> DistNode {
        DistNode::Continuous(ContinuousDist::Normal {
            dist: NormalDist::new(mean, std_dev.abs()).unwrap(),
        })
    }

    pub fn from_range(x: f64, y: f64, credibility: f64) -> DistNode {
        let mean = (x + y) / 2.;
        let cdf_value = 0.5 * (1. + (credibility / 100.));
        let normed_sigma = ppf(cdf_value);
        let std_dev = (y - mean) / normed_sigma;

        Self::from_mean(mean, std_dev)
    }
}

pub struct LogNormal {}
impl LogNormal {
    pub fn new(mean: f64, std_dev: f64) -> DistNode {
        DistNode::Continuous(ContinuousDist::LogNormal {
            dist: LogNormalDist::new(mean, std_dev).unwrap(),
        })
    }

    pub fn from_loc(location: f64, scale: f64) -> DistNode {
        DistNode::Continuous(ContinuousDist::LogNormal {
            dist: LogNormalDist::new(location, scale).unwrap(),
        })
    }

    pub fn from_mean(mean: f64, std_dev: f64) -> DistNode {
        let location = (square(mean) / (square(mean) + square(std_dev)).sqrt()).ln();

        let scale = (1. + square(std_dev) / square(mean)).ln() / 2.;

        Self::from_loc(location, scale)
    }

    pub fn from_range(x: f64, y: f64, credibility: f64) -> DistNode {
        let mean = (x.ln() + y.ln()) / 2.;
        let cdf_value = 0.5 * (1. + (credibility / 100.));
        let normed_sigma = ppf(cdf_value);
        let std_dev = (y.ln() - mean) / normed_sigma;

        Self::from_mean(mean, std_dev)
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
