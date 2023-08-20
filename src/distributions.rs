use std::fmt::Display;

use dist_derive::DistributionDerive;
use ndarray::{Array, Dim};
use rand::{thread_rng, Rng};
use rand_distr::Distribution as Dst;
use statrs::distribution::{LogNormal, Normal, Poisson, Triangular, Uniform};

#[derive(Debug, PartialEq)]
pub struct Constant {
    number: f64,
}

impl Constant {
    pub fn new(number: f64) -> Self {
        Self { number }
    }

    pub fn number(&self) -> f64 {
        self.number
    }
}

impl Dst<f64> for Constant {
    fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
        self.number
    }
}

#[derive(Debug, PartialEq)]
pub struct Discrete {
    values: Box<[f64]>,
    weights: Box<[f64]>,
}

impl Discrete {
    pub fn new(values: Vec<f64>, weights: Vec<f64>) -> Self {
        Self {
            values: values.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
        }
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

impl Dst<f64> for Discrete {
    fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
        let cumulative_wights: Vec<f64> = self
            .weights
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        let r = thread_rng().gen_range(0.0..1.0);

        for (i, v) in self.values.iter().enumerate() {
            if r < cumulative_wights[i] {
                return v.to_owned();
            }
        }

        return self.values.last().unwrap().to_owned();
    }
}

#[derive(DistributionDerive, PartialEq, Debug)]
pub enum ContinuousDist {
    LogNormal { dist: LogNormal },
    Normal { dist: Normal },
    Triangular { dist: Triangular },
    Uniform { dist: Uniform },
}

impl Display for ContinuousDist {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::LogNormal { dist: _ } => write!(f, "Log-Normal"),
            Self::Normal { dist: _ } => write!(f, "Normal"),
            Self::Triangular { dist: _ } => write!(f, "Triangular"),
            Self::Uniform { dist: _ } => write!(f, "Uniform"),
        }
    }
}

#[derive(DistributionDerive, PartialEq, Debug)]
pub enum DiscreteDist {
    Constant { dist: Constant },
    Poisson { dist: Poisson },
    Discrete { dist: Discrete },
}

impl Display for DiscreteDist {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Constant { dist: _ } => write!(f, "Constant"),
            Self::Poisson { dist: _ } => write!(f, "Poisson"),
            Self::Discrete { dist: _ } => write!(f, "Discrete"),
        }
    }
}

#[derive(PartialEq, Debug)]
pub enum Ops {
    Add(Box<DistNode>, Box<DistNode>),
    Mul(Box<DistNode>, Box<DistNode>),
    Sub(Box<DistNode>, Box<DistNode>),
    Div(Box<DistNode>, Box<DistNode>),
}

impl Display for Ops {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Add(d1, d2) => write!(f, "({} + {})", d1, d2),
            Self::Mul(d1, d2) => write!(f, "({} * {})", d1, d2),
            Self::Sub(d1, d2) => write!(f, "({} - {})", d1, d2),
            Self::Div(d1, d2) => write!(f, "({} / {})", d1, d2),
        }
    }
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
                Ops::Div(d1, d2) => d1.sample() / d2.sample(),
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
                Ops::Div(d1, d2) => d1.nsample(n) / d2.nsample(n),
            },
            Self::Continuous(dist) => dist.sample_iter(&mut thread_rng()).take(n).collect(),
            Self::Discrete(dist) => dist.sample_iter(&mut thread_rng()).take(n).collect(),
        }
    }
}

impl Display for DistNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Operation(op) => write!(f, "{op}"),
            Self::Continuous(dist) => write!(f, "{dist}"),
            Self::Discrete(dist) => write!(f, "{dist}"),
        }
    }
}
