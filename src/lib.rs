mod distributions;
pub mod interface;

#[cfg(test)]
mod tests {

    use super::distributions::{ContinuousDist, DiscreteDist, DistNode};
    use statrs::distribution::Normal;

    #[test]
    fn normal_sum_optimization() {
        let d1 = DistNode::Continuous(ContinuousDist::Normal {
            dist: Normal::new(0., 1.).unwrap(),
        });
        let d2 = DistNode::Continuous(ContinuousDist::Normal {
            dist: Normal::new(2., 1.5).unwrap(),
        });

        let d3 = DistNode::Continuous(ContinuousDist::Normal {
            dist: Normal::new(2., 3.25).unwrap(),
        });

        assert_eq!(d1 + d2, d3);
    }
}
