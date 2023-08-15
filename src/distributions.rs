use dist_derive::DistributionDerive;
use ndarray::{Array, Dim};
use rand::thread_rng;
use rand_distr::Distribution as Dst;
use statrs::{
    distribution::{LogNormal, Normal, Poisson, Triangular, Uniform},
    statistics::Distribution,
};
use std::ops::{Add, Mul, Sub};

#[derive(Debug, PartialEq)]
pub struct Constant {
    number: f64,
}

impl Constant {
    pub fn new(number: f64) -> Self {
        Self { number }
    }

    fn number(&self) -> f64 {
        self.number
    }
}

impl Dst<f64> for Constant {
    fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
        self.number
    }
}

#[derive(DistributionDerive, PartialEq, Debug)]
pub enum ContinuousDist {
    LogNormal { dist: LogNormal },
    Normal { dist: Normal },
    Triangular { dist: Triangular },
    Uniform { dist: Uniform },
}

#[derive(DistributionDerive, PartialEq, Debug)]
pub enum DiscreteDist {
    Constant { dist: Constant },
    Poisson { dist: Poisson },
}

#[derive(PartialEq, Debug)]
pub enum Ops {
    Add(Box<DistNode>, Box<DistNode>),
    Mul(Box<DistNode>, Box<DistNode>),
    Sub(Box<DistNode>, Box<DistNode>),
}

#[derive(PartialEq, Debug)]
pub enum DistNode {
    Operation(Ops),
    Continuous(ContinuousDist),
    Discrete(DiscreteDist),
}

impl DistNode {
    pub fn sample(&self) -> f64 {
        match self {
            Self::Operation(op) => match op {
                Ops::Add(d1, d2) => d1.sample() + d2.sample(),
                Ops::Mul(d1, d2) => d1.sample() * d2.sample(),
                Ops::Sub(d1, d2) => d1.sample() - d2.sample(),
            },
            Self::Continuous(dist) => dist.sample(&mut thread_rng()),
            Self::Discrete(dist) => dist.sample(&mut thread_rng()),
        }
    }

    pub fn nsample(&self, n: usize) -> Array<f64, Dim<[usize; 1]>> {
        match self {
            Self::Operation(op) => match op {
                Ops::Add(d1, d2) => d1.nsample(n) + d2.nsample(n),
                Ops::Mul(d1, d2) => d1.nsample(n) * d2.nsample(n),
                Ops::Sub(d1, d2) => d1.nsample(n) - d2.nsample(n),
            },
            Self::Continuous(dist) => dist.sample_iter(&mut thread_rng()).take(n).collect(),
            Self::Discrete(dist) => dist.sample_iter(&mut thread_rng()).take(n).collect(),
        }
    }
}

impl Add for DistNode {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (&self, &rhs) {
            (DistNode::Continuous(d1), DistNode::Continuous(d2)) => match (d1, d2) {
                (ContinuousDist::Normal { dist: d3 }, ContinuousDist::Normal { dist: d4 }) => {
                    DistNode::Continuous(ContinuousDist::Normal {
                        dist: Normal::new(
                            d3.mean().unwrap() + d4.mean().unwrap(),
                            f64::powi(d3.std_dev().unwrap(), 2)
                                + f64::powi(d4.std_dev().unwrap(), 2),
                        )
                        .expect("invalid normal distribution parameters"),
                    })
                }
                (_, _) => DistNode::Operation(Ops::Add(Box::new(self), Box::new(rhs))),
            },
            (DistNode::Discrete(d1), DistNode::Discrete(d2)) => match (d1, d2) {
                (DiscreteDist::Constant { dist: d3 }, DiscreteDist::Constant { dist: d4 }) => {
                    DistNode::Discrete(DiscreteDist::Constant {
                        dist: Constant::new(d3.number() + d4.number()),
                    })
                }
                (_, _) => DistNode::Operation(Ops::Add(Box::new(self), Box::new(rhs))),
            },
            (_, _) => DistNode::Operation(Ops::Add(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Mul for DistNode {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::Operation(Ops::Mul(Box::new(self), Box::new(rhs)))
    }
}

impl Sub for DistNode {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Operation(Ops::Sub(Box::new(self), Box::new(rhs)))
    }
}
