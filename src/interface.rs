#![allow(clippy::new_ret_no_self)]
use statrs::distribution::{LogNormal as LogNormalDist, Normal as NormalDist};

use crate::distributions::{ContinuousDist, DiscreteDist, DistNode};

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
