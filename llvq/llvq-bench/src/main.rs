//! Gaussian-source rate-distortion report (paper §4 protocol).
//!
//! Usage: `cargo run --release -p llvq-bench [-- train_n eval_n seed]`

use llvq_bench::*;
use llvq_core::SplitMix64;
use llvq_search::Searcher;

fn main() {
    let args: Vec<u64> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric args: train_n eval_n seed"))
        .collect();
    let train_n = *args.first().unwrap_or(&4_000) as usize;
    let eval_n = *args.get(1).unwrap_or(&20_000) as usize;
    let seed = *args.get(2).unwrap_or(&0x64A4_2026);

    let s = Searcher::new();
    let mut rng = SplitMix64::new(seed);
    let train: Vec<_> = (0..train_n).map(|_| gauss_block(&mut rng)).collect();
    let eval: Vec<_> = (0..eval_n).map(|_| gauss_block(&mut rng)).collect();

    eprintln!("searching {train_n} train + {eval_n} eval blocks (union m ≤ 3)…");
    let t0 = std::time::Instant::now();
    let train_dots = precompute(&s, &train);
    let eval_dots = precompute(&s, &eval);
    eprintln!(
        "search pass: {:.1}s ({:.0} blocks/s total)",
        t0.elapsed().as_secs_f64(),
        (train_n + eval_n) as f64 / t0.elapsed().as_secs_f64()
    );

    // Spherical shaping: fit β on train, evaluate on eval.
    let beta = optimize_beta(&train_dots, 0.4, 1.1, 140);
    let mse_sph = spherical_mse(&eval_dots, beta);
    let r_sph = rate_spherical();

    // Shape–gain: Lloyd–Max gain codebooks fitted on train projections.
    let train_t: Vec<f64> = train_dots.iter().map(BlockDots::t).collect();
    let rows_sg: Vec<(u32, f64, f64)> = [0u32, 2]
        .into_iter()
        .map(|k| {
            let centroids = lloyd_max(&train_t, k, 60);
            (k, rate_shape_gain(k), shape_gain_mse(&eval_dots, &centroids))
        })
        .collect();

    println!("\nGaussian source N(0,1), {eval_n} eval blocks of 24 (seed {seed:#x})");
    println!("codebook: Shell(2) ∪ Shell(3), spherical β* = {beta:.3}\n");
    println!(
        "{:<38} {:>9} {:>9} {:>12} {:>9}",
        "method", "bits/dim", "MSE", "SQNR(bits)", "Ret(%)"
    );
    let row = |name: &str, r: f64, mse: f64| {
        println!(
            "{name:<38} {r:>9.4} {mse:>9.4} {:>12.4} {:>9.2}",
            sqnr_bits(mse),
            retention_pct(mse, r)
        );
    };
    row(
        "Lloyd–Max scalar 1-bit (analytic)",
        1.0,
        lloyd_max_1bit_scalar_mse(),
    );
    row("LLVQ spherical shaping (m ≤ 3)", r_sph, mse_sph);
    for (k, r, mse) in &rows_sg {
        row(&format!("LLVQ shape–gain, {k}-bit gain (m ≤ 3)"), *r, *mse);
    }
    println!(
        "{:<38} {:>9.4} {:>9.4} {:>12.4} {:>9.2}",
        "Shannon limit @ r_sph",
        r_sph,
        2f64.powf(-2.0 * r_sph),
        r_sph,
        100.0
    );
}
