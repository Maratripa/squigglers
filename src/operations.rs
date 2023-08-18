use std::ops::{Add, Div, Mul, Sub};

use statrs::statistics::{Distribution, Median};

use crate::{
    distributions::{ContinuousDist, DiscreteDist, DistNode, Ops},
    interface,
    utils::square,
};

impl Add for DistNode {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (&self, &rhs) {
            (DistNode::Continuous(d1), DistNode::Continuous(d2)) => match (d1, d2) {
                (ContinuousDist::Normal { dist: d3 }, ContinuousDist::Normal { dist: d4 }) => {
                    interface::Normal::from_mean(
                        d3.mean().unwrap() + d4.mean().unwrap(),
                        square(d3.std_dev().unwrap()) + square(d3.std_dev().unwrap()),
                    )
                }
                (_, _) => DistNode::Operation(Ops::Add(Box::new(self), Box::new(rhs))),
            },
            (DistNode::Discrete(d1), DistNode::Discrete(d2)) => match (d1, d2) {
                (DiscreteDist::Constant { dist: d3 }, DiscreteDist::Constant { dist: d4 }) => {
                    interface::Constant::new(d3.number() + d4.number())
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
                interface::Normal::from_mean(dist.mean().unwrap() + rhs, dist.std_dev().unwrap())
            }
            Self::Discrete(DiscreteDist::Constant { dist }) => {
                interface::Constant::new(dist.number() + rhs)
            }
            _ => Self::Operation(Ops::Add(
                Box::new(self),
                Box::new(interface::Constant::new(rhs)),
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

                    interface::LogNormal::from_loc(location, scale)
                }
                (_, _) => Self::Operation(Ops::Mul(Box::new(self), Box::new(rhs))),
            },
            (Self::Discrete(d1), Self::Discrete(d2)) => match (d1, d2) {
                (DiscreteDist::Constant { dist: d3 }, DiscreteDist::Constant { dist: d4 }) => {
                    interface::Constant::new(d3.number() * d4.number())
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
            return interface::Constant::new(0.0);
        }
        match &self {
            Self::Continuous(ContinuousDist::Normal { dist }) => interface::Normal::from_mean(
                dist.mean().unwrap() * rhs,
                dist.std_dev().unwrap() * rhs.abs(),
            ),
            Self::Continuous(ContinuousDist::LogNormal { dist }) => {
                let loc = dist.median().ln();
                let scale = 2. * dist.mean().unwrap().ln() - loc;

                interface::LogNormal::from_loc(loc + rhs.abs().ln(), scale)
            }
            Self::Discrete(DiscreteDist::Constant { dist }) => {
                interface::Constant::new(dist.number() * rhs)
            }
            _ => Self::Operation(Ops::Mul(
                Box::new(self),
                Box::new(interface::Constant::new(rhs)),
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
                    interface::Normal::from_mean(
                        d3.mean().unwrap() - d4.mean().unwrap(),
                        (square(d3.std_dev().unwrap()) + square(d4.std_dev().unwrap())).sqrt(),
                    )
                }
                (_, _) => DistNode::Operation(Ops::Sub(Box::new(self), Box::new(rhs))),
            },
            (DistNode::Discrete(d1), DistNode::Discrete(d2)) => match (d1, d2) {
                (DiscreteDist::Constant { dist: d3 }, DiscreteDist::Constant { dist: d4 }) => {
                    interface::Constant::new(d3.number() - d4.number())
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
                interface::Normal::from_mean(dist.mean().unwrap() - rhs, dist.std_dev().unwrap())
            }
            Self::Discrete(DiscreteDist::Constant { dist }) => {
                interface::Constant::new(dist.number() - rhs)
            }
            _ => Self::Operation(Ops::Sub(
                Box::new(self),
                Box::new(interface::Constant::new(rhs)),
            )),
        }
    }
}

impl Sub<DistNode> for f64 {
    type Output = DistNode;
    fn sub(self, rhs: DistNode) -> DistNode {
        match &rhs {
            DistNode::Continuous(ContinuousDist::Normal { dist }) => {
                interface::Normal::from_mean(self - dist.mean().unwrap(), dist.std_dev().unwrap())
            }
            DistNode::Discrete(DiscreteDist::Constant { dist }) => {
                interface::Constant::new(self - dist.number())
            }
            _ => DistNode::Operation(Ops::Sub(
                Box::new(interface::Constant::new(self)),
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

                    interface::LogNormal::from_loc(location, scale)
                }
                (_, _) => Self::Operation(Ops::Div(Box::new(self), Box::new(rhs))),
            },
            (Self::Discrete(d1), Self::Discrete(d2)) => match (d1, d2) {
                (DiscreteDist::Constant { dist: d3 }, DiscreteDist::Constant { dist: d4 }) => {
                    interface::Constant::new(d3.number() / d4.number())
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
            return interface::Constant::new(0.0);
        }
        match &self {
            Self::Continuous(ContinuousDist::Normal { dist }) => interface::Normal::from_mean(
                dist.mean().unwrap() / rhs,
                dist.std_dev().unwrap() / rhs.abs(),
            ),
            Self::Continuous(ContinuousDist::LogNormal { dist }) => {
                let loc = dist.median().ln();
                let scale = 2. * dist.mean().unwrap().ln() - loc;
                interface::LogNormal::from_loc(loc - rhs.abs().ln(), scale)
            }
            Self::Discrete(DiscreteDist::Constant { dist }) => {
                interface::Constant::new(dist.number() / rhs)
            }
            _ => Self::Operation(Ops::Div(
                Box::new(self),
                Box::new(interface::Constant::new(rhs)),
            )),
        }
    }
}

impl Div<DistNode> for f64 {
    type Output = DistNode;
    fn div(self, rhs: DistNode) -> DistNode {
        if self == 0.0 {
            return interface::Constant::new(0.0);
        }
        match &rhs {
            DistNode::Continuous(ContinuousDist::LogNormal { dist }) => {
                let loc = dist.median().ln();
                let scale = 2. * dist.mean().unwrap().ln() - loc;
                interface::LogNormal::from_loc(self.abs().ln() - loc, scale)
            }
            DistNode::Discrete(DiscreteDist::Constant { dist }) => {
                let const_num: f64 = if dist.number() == 0.0 {
                    0.0
                } else {
                    self / dist.number()
                };

                interface::Constant::new(const_num)
            }
            _ => DistNode::Operation(Ops::Div(
                Box::new(interface::Constant::new(self)),
                Box::new(rhs),
            )),
        }
    }
}
