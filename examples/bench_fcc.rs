//! Benchmarks `compute_neighbor_list` on the same FCC Cu supercells used in
//! `scripts/bench_competitors.py` (Phase 0), so the numbers are directly
//! comparable to the recorded ASE/Vesin/torch_nl baseline in
//! `docs/decision_log.md`.
//!
//! Run with `cargo run --release --example bench_fcc`.

use ferrosim::{compute_neighbor_list, Cell, System};
use std::hint::black_box;
use std::time::Instant;

const LATTICE_A: f64 = 3.6;
const BASIS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
];

fn fcc_cu_supercell(target_atoms: usize) -> System {
    let reps = ((target_atoms as f64 / 4.0).cbrt().round() as usize).max(1);
    let side = LATTICE_A * reps as f64;
    let mut positions = Vec::with_capacity(4 * reps * reps * reps * 3);
    for ix in 0..reps {
        for iy in 0..reps {
            for iz in 0..reps {
                for b in BASIS {
                    positions.push((ix as f64 + b[0]) * LATTICE_A);
                    positions.push((iy as f64 + b[1]) * LATTICE_A);
                    positions.push((iz as f64 + b[2]) * LATTICE_A);
                }
            }
        }
    }
    let cell = Cell::new([[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]]);
    System::new(positions, cell, [true, true, true])
}

fn time_it(mut f: impl FnMut(), n_repeats: u32) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..n_repeats {
        let t0 = Instant::now();
        f();
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

fn main() {
    let cutoff = 6.0;
    println!("{:>8} | {:>12}", "n_atoms", "ferrosim_s");
    println!("{}", "-".repeat(25));
    for target in [50usize, 500, 2000] {
        let system = fcc_cu_supercell(target);
        let n = system.n_atoms();
        let t = time_it(|| { black_box(compute_neighbor_list(black_box(&system), cutoff)); }, 7);
        println!("{n:>8} | {t:>12.5}");
    }
}
