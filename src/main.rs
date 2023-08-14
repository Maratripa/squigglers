use squigglers::interface::Normal;

fn main() {
    let d1 = Normal::new(3., 1.5);

    let d2 = Normal::new(20., 0.1);

    println!("{:?}", (d1 + d2).nsample(5));
}
