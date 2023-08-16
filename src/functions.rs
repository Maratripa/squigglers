use crate::{
    distributions::DistNode,
    interface::{LogNormal, Normal},
};

/// Initialize a distribution from `x` to `y`.
///
/// The distribution will be lognormal by default, unless `x` is less than or equal to 0,
/// in which case it will become a normal distribution.
pub fn to(x: f64, y: f64, credibility: f64) -> DistNode {
    if x > 0.0 {
        LogNormal::from_range(x, y, credibility)
    } else {
        Normal::from_range(x, y, credibility)
    }
}
