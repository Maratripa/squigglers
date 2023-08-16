use statrs::distribution::{ContinuousCDF, Normal};

pub fn square(num: f64) -> f64 {
    f64::powi(num, 2)
}

pub fn ppf(q: f64) -> f64 {
    Normal::new(0., 1.).unwrap().inverse_cdf(q)
}
