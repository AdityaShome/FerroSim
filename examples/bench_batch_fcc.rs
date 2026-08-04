//! Benchmarks `compute_neighbor_lists_batched` on the same batched-workload
//! shape as `scripts/bench_batch_competitors.py` (Phase 2): many independent
//! small FCC Cu cells processed together, matching the actual MLIP
//! training/inference workload identified in Phase 0.
//!
//! Run with `cargo run --release --example bench_batch_fcc`.

use ferrosim::{compute_neighbor_lists_batched, Cell, System};
use std::hint::black_box;
use std::time::Instant;

const LATTICE_A: f64 = 3.6;
const BASIS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
];

fn fcc_cu_system(target_atoms: usize) -> System {
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
    let n_atoms_per_system = 32;
    println!("batch benchmark: n_atoms_per_system={n_atoms_per_system}, cutoff={cutoff}");
    println!("{:>10} | {:>14}", "batch_size", "ferrosim_s");
    println!("{}", "-".repeat(29));
    for batch_size in [8usize, 32, 128] {
        let systems: Vec<System> = (0..batch_size).map(|_| fcc_cu_system(n_atoms_per_system)).collect();
        let t = time_it(
            || {
                black_box(compute_neighbor_lists_batched(black_box(&systems), cutoff));
            },
            7,
        );
        println!("{batch_size:>10} | {t:>14.5}");
    }
}
