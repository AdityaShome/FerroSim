//! Compares the GPU (`wgpu`) batched path against the CPU (`rayon`) batched
//! path on the same batched FCC Cu workload as `bench_batch_fcc.rs`, so GPU
//! benchmark numbers are directly comparable to both the CPU-batched and the
//! Vesin/torch_nl/ASE numbers already recorded in `docs/decision_log.md`.
//!
//! Run with `cargo run --release --example bench_gpu_batch_fcc`.

use ferrosim::{compute_neighbor_lists_batched, compute_neighbor_lists_batched_gpu, Cell, System};
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

    // One untimed warm-up call: pays the one-time wgpu adapter/device/shader
    // init cost outside the timed loop, mirroring how a long-running process
    // (e.g. an MD trajectory or training loop) would actually use this.
    let warmup = vec![fcc_cu_system(n_atoms_per_system)];
    let _ = compute_neighbor_lists_batched_gpu(&warmup, cutoff);

    println!("batch benchmark: n_atoms_per_system={n_atoms_per_system}, cutoff={cutoff}");
    println!("{:>10} | {:>12} | {:>12}", "batch_size", "cpu_s", "gpu_s");
    println!("{}", "-".repeat(40));
    for batch_size in [8usize, 32, 128, 512, 2048, 8192] {
        let systems: Vec<System> = (0..batch_size).map(|_| fcc_cu_system(n_atoms_per_system)).collect();
        let t_cpu = time_it(
            || {
                black_box(compute_neighbor_lists_batched(black_box(&systems), cutoff));
            },
            7,
        );
        let t_gpu = time_it(
            || {
                black_box(compute_neighbor_lists_batched_gpu(black_box(&systems), cutoff));
            },
            7,
        );
        println!("{batch_size:>10} | {t_cpu:>12.5} | {t_gpu:>12.5}");
    }
}
