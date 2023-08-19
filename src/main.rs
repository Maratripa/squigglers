use squigglers::functions::to;
use squigglers::interface::{Constant, LogNormal, Normal};

fn main() {
    let d1 = Normal::from_mean(0., 1.);
    let d2 = LogNormal::from_mean(3., 0.5);
    let d3 = Constant::new(10.);
    let d4 = Normal::from_mean(-2., 0.1);

    let d = d1 + (d2 + d3) + d4;

    println!("{d}");
}
