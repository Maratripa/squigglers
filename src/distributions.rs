use dist_derive::DistributionDerive;
use ndarray::{Array, Dim};
use rand::thread_rng;
use rand_distr::Distribution as Dst;
use statrs::{
    distribution::{LogNormal, Normal, Poisson, Triangular, Uniform},
    statistics::{Distribution, Median},
};
use std::ops::{Add, Div, Mul, Sub};

use crate::utils::square;

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
    Div(Box<DistNode>, Box<DistNode>),
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

impl Add for DistNode {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (&self, &rhs) {
            (DistNode::Continuous(d1), DistNode::Continuous(d2)) => match (d1, d2) {
                (ContinuousDist::Normal { dist: d3 }, ContinuousDist::Normal { dist: d4 }) => {
                    DistNode::Continuous(ContinuousDist::Normal {
                        dist: Normal::new(
                            d3.mean().unwrap() + d4.mean().unwrap(),
                            (square(d3.std_dev().unwrap()) + square(d4.std_dev().unwrap())).sqrt(),
                        )
                        .unwrap(),
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

impl Add<f64> for DistNode {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        match &self {
            Self::Continuous(ContinuousDist::Normal { dist }) => {
                Self::Continuous(ContinuousDist::Normal {
                    dist: Normal::new(dist.mean().unwrap() + rhs, dist.std_dev().unwrap()).unwrap(),
                })
            }
            Self::Discrete(DiscreteDist::Constant { dist }) => {
                Self::Discrete(DiscreteDist::Constant {
                    dist: Constant::new(dist.number() + rhs),
                })
            }
            _ => Self::Operation(Ops::Add(
                Box::new(self),
                Box::new(DistNode::Discrete(DiscreteDist::Constant {
                    dist: Constant::new(rhs),
                })),
            )),
        }
    }
}

impl Mul for DistNode {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match (&self, &rhs) {
            (Self::Continuous(d1), Self::Continuous(d2)) => match (d1, d2) {
                (
                    ContinuousDist::LogNormal { dist: d3 },
                    ContinuousDist::LogNormal { dist: d4 },
                ) => {
                    let loc3 = d3.median().ln();
                    let loc4 = d4.median().ln();

                    let var3 = 2. * d3.mean().unwrap().ln() - loc3;
                    let var4 = 2. * d4.mean().unwrap().ln() - loc4;

                    let location = loc3 + loc4;
                    let scale = (var3 + var4).sqrt();

                    Self::Continuous(ContinuousDist::LogNormal {
                        dist: LogNormal::new(location, scale).unwrap(),
                    })
                }
                (_, _) => Self::Operation(Ops::Mul(Box::new(self), Box::new(rhs))),
            },
            (Self::Discrete(d1), Self::Discrete(d2)) => match (d1, d2) {
                (DiscreteDist::Constant { dist: d3 }, DiscreteDist::Constant { dist: d4 }) => {
                    Self::Discrete(DiscreteDist::Constant {
                        dist: Constant::new(d3.number() * d4.number()),
                    })
                }
                (_, _) => Self::Operation(Ops::Mul(Box::new(self), Box::new(rhs))),
            },
            (_, _) => Self::Operation(Ops::Mul(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Mul<f64> for DistNode {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        if rhs == 0.0 {
            return Self::Discrete(DiscreteDist::Constant {
                dist: Constant::new(0.0),
            });
        }
        match &self {
            Self::Continuous(ContinuousDist::Normal { dist }) => {
                Self::Continuous(ContinuousDist::Normal {
                    dist: Normal::new(
                        dist.mean().unwrap() * rhs,
                        dist.std_dev().unwrap() * rhs.abs(),
                    )
                    .unwrap(),
                })
            }
            Self::Continuous(ContinuousDist::LogNormal { dist }) => {
                let loc = dist.median().ln();
                let scale = 2. * dist.mean().unwrap().ln() - loc;
                Self::Continuous(ContinuousDist::LogNormal {
                    dist: LogNormal::new(loc + rhs.abs().ln(), scale).unwrap(),
                })
            }
            Self::Discrete(DiscreteDist::Constant { dist }) => {
                Self::Discrete(DiscreteDist::Constant {
                    dist: Constant::new(dist.number() * rhs),
                })
            }
            _ => Self::Operation(Ops::Mul(
                Box::new(self),
                Box::new(DistNode::Discrete(DiscreteDist::Constant {
                    dist: Constant { number: rhs },
                })),
            )),
        }
    }
}

impl Sub for DistNode {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (DistNode::Continuous(d1), DistNode::Continuous(d2)) => match (d1, d2) {
                (ContinuousDist::Normal { dist: d3 }, ContinuousDist::Normal { dist: d4 }) => {
                    DistNode::Continuous(ContinuousDist::Normal {
                        dist: Normal::new(
                            d3.mean().unwrap() - d4.mean().unwrap(),
                            (square(d3.std_dev().unwrap()) + square(d4.std_dev().unwrap())).sqrt(),
                        )
                        .unwrap(),
                    })
                }
                (_, _) => DistNode::Operation(Ops::Sub(Box::new(self), Box::new(rhs))),
            },
            (DistNode::Discrete(d1), DistNode::Discrete(d2)) => match (d1, d2) {
                (DiscreteDist::Constant { dist: d3 }, DiscreteDist::Constant { dist: d4 }) => {
                    DistNode::Discrete(DiscreteDist::Constant {
                        dist: Constant::new(d3.number() - d4.number()),
                    })
                }
                (_, _) => DistNode::Operation(Ops::Sub(Box::new(self), Box::new(rhs))),
            },
            (_, _) => DistNode::Operation(Ops::Sub(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Sub<f64> for DistNode {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        match &self {
            Self::Continuous(ContinuousDist::Normal { dist }) => {
                Self::Continuous(ContinuousDist::Normal {
                    dist: Normal::new(dist.mean().unwrap() - rhs, dist.std_dev().unwrap()).unwrap(),
                })
            }
            Self::Discrete(DiscreteDist::Constant { dist }) => {
                Self::Discrete(DiscreteDist::Constant {
                    dist: Constant::new(dist.number() - rhs),
                })
            }
            _ => Self::Operation(Ops::Sub(
                Box::new(self),
                Box::new(DistNode::Discrete(DiscreteDist::Constant {
                    dist: Constant::new(rhs),
                })),
            )),
        }
    }
}

impl Sub<DistNode> for f64 {
    type Output = DistNode;
    fn sub(self, rhs: DistNode) -> DistNode {
        match &rhs {
            DistNode::Continuous(ContinuousDist::Normal { dist }) => {
                DistNode::Continuous(ContinuousDist::Normal {
                    dist: Normal::new(self - dist.mean().unwrap(), dist.std_dev().unwrap())
                        .unwrap(),
                })
            }
            DistNode::Discrete(DiscreteDist::Constant { dist }) => {
                DistNode::Discrete(DiscreteDist::Constant {
                    dist: Constant::new(self - dist.number()),
                })
            }
            _ => DistNode::Operation(Ops::Sub(
                Box::new(DistNode::Discrete(DiscreteDist::Constant {
                    dist: Constant::new(self),
                })),
                Box::new(rhs),
            )),
        }
    }
}

impl Div for DistNode {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Self::Continuous(d1), Self::Continuous(d2)) => match (d1, d2) {
                (
                    ContinuousDist::LogNormal { dist: d3 },
                    ContinuousDist::LogNormal { dist: d4 },
                ) => {
                    let loc3 = d3.median().ln();
                    let loc4 = d4.median().ln();

                    let var3 = 2. * d3.mean().unwrap().ln() - loc3;
                    let var4 = 2. * d4.mean().unwrap().ln() - loc4;

                    let location = loc3 - loc4;
                    let scale = (var3 + var4).sqrt();

                    Self::Continuous(ContinuousDist::LogNormal {
                        dist: LogNormal::new(location, scale).unwrap(),
                    })
                }
                (_, _) => Self::Operation(Ops::Div(Box::new(self), Box::new(rhs))),
            },
            (Self::Discrete(d1), Self::Discrete(d2)) => match (d1, d2) {
                (DiscreteDist::Constant { dist: d3 }, DiscreteDist::Constant { dist: d4 }) => {
                    Self::Discrete(DiscreteDist::Constant {
                        dist: Constant::new(d3.number() / d4.number()),
                    })
                }
                (_, _) => Self::Operation(Ops::Div(Box::new(self), Box::new(rhs))),
            },
            (_, _) => Self::Operation(Ops::Div(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Div<f64> for DistNode {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        if rhs == 0.0 {
            return Self::Discrete(DiscreteDist::Constant {
                dist: Constant::new(0.0),
            });
        }
        match &self {
            Self::Continuous(ContinuousDist::Normal { dist }) => {
                Self::Continuous(ContinuousDist::Normal {
                    dist: Normal::new(
                        dist.mean().unwrap() / rhs,
                        dist.std_dev().unwrap() / rhs.abs(),
                    )
                    .unwrap(),
                })
            }
            Self::Continuous(ContinuousDist::LogNormal { dist }) => {
                let loc = dist.median().ln();
                let scale = 2. * dist.mean().unwrap().ln() - loc;
                Self::Continuous(ContinuousDist::LogNormal {
                    dist: LogNormal::new(loc - rhs.abs().ln(), scale).unwrap(),
                })
            }
            Self::Discrete(DiscreteDist::Constant { dist }) => {
                Self::Discrete(DiscreteDist::Constant {
                    dist: Constant::new(dist.number() / rhs),
                })
            }
            _ => Self::Operation(Ops::Div(
                Box::new(self),
                Box::new(DistNode::Discrete(DiscreteDist::Constant {
                    dist: Constant { number: rhs },
                })),
            )),
        }
    }
}

impl Div<DistNode> for f64 {
    type Output = DistNode;
    fn div(self, rhs: DistNode) -> DistNode {
        if self == 0.0 {
            return DistNode::Discrete(DiscreteDist::Constant {
                dist: Constant::new(0.0),
            });
        }
        match &rhs {
            DistNode::Continuous(ContinuousDist::LogNormal { dist }) => {
                let loc = dist.median().ln();
                let scale = 2. * dist.mean().unwrap().ln() - loc;
                DistNode::Continuous(ContinuousDist::LogNormal {
                    dist: LogNormal::new(self.abs().ln() - loc, scale).unwrap(),
                })
            }
            DistNode::Discrete(DiscreteDist::Constant { dist }) => {
                let const_num: f64 = if dist.number() == 0.0 {
                    0.0
                } else {
                    self / dist.number()
                };

                DistNode::Discrete(DiscreteDist::Constant {
                    dist: Constant::new(const_num),
                })
            }
            _ => DistNode::Operation(Ops::Div(
                Box::new(DistNode::Discrete(DiscreteDist::Constant {
                    dist: Constant { number: self },
                })),
                Box::new(rhs),
            )),
        }
    }
}
