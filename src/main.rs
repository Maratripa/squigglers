use squigglers::functions::to;

fn main() {
    let pop_of_ny_2022 = to(8.1 * 1_000_000.0, 8.4 * 1_000_000.0, 90.0);
    let pct_of_pop_w_pianos = to(0.2, 1.0, 90.0) * 0.01;
    let pianos_per_piano_tuner = to(2_000.0, 50_000.0, 90.0);
    let piano_tuners_per_piano = 1.0 / pianos_per_piano_tuner;
    let total_tuners_in_2022 = pop_of_ny_2022 * pct_of_pop_w_pianos * piano_tuners_per_piano;

    let samples = total_tuners_in_2022.nsample(1000);

    println!(
        "Mean: {}, SD: {}",
        samples.mean().unwrap(),
        samples.std(1.0)
    );
}
