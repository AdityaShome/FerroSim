use ferrosim::{compute_neighbor_lists_batched, compute_neighbor_lists_batched_gpu, Cell, System};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// The GPU path uses f32 internally (CPU uses f64), so these tests use random
// continuous configurations rather than deliberately-exact cutoff-boundary
// cases: the probability a random pair's distance lands within f32-vs-f64
// rounding distance of the cutoff is ~0, so this stays non-flaky while still
// being a real correctness check, not a "doesn't crash" check.

fn random_positions(rng: &mut ChaCha8Rng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..3 * n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn cubic_cell(side: f64) -> Cell {
    Cell::new([[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]])
}

fn triclinic_cell() -> Cell {
    Cell::new([[5.0, 0.0, 0.0], [1.5, 4.0, 0.0], [0.7, 0.3, 3.5]])
}

/// Phase 3 gate: GPU output must match the (already validated against
/// brute-force) CPU batched implementation exactly, across orthogonal,
/// triclinic, and mixed-PBC systems, before any GPU benchmark number is
/// trusted.
#[test]
fn gpu_matches_cpu_batched() {
    let mut rng = ChaCha8Rng::seed_from_u64(555);
    let cutoff = 3.5;

    let mut systems = Vec::new();
    for _ in 0..10 {
        let n = rng.gen_range(1..40);
        let side = rng.gen_range(6.0..15.0);
        let positions = random_positions(&mut rng, n, 0.0, side);
        systems.push(System::new(positions, cubic_cell(side), [true, true, true]));
    }
    for _ in 0..10 {
        let n = rng.gen_range(1..30);
        let positions = random_positions(&mut rng, n, -2.0, 6.0);
        systems.push(System::new(positions, triclinic_cell(), [true, true, true]));
    }
    for _ in 0..5 {
        let n = rng.gen_range(1..20);
        let positions = random_positions(&mut rng, n, 0.0, 10.0);
        let pbc = [rng.gen_bool(0.5), rng.gen_bool(0.5), rng.gen_bool(0.5)];
        systems.push(System::new(positions, cubic_cell(10.0), pbc));
    }

    let cpu = compute_neighbor_lists_batched(&systems, cutoff);
    let gpu = compute_neighbor_lists_batched_gpu(&systems, cutoff);

    assert_eq!(cpu.len(), gpu.len());
    for (idx, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(
            c.sorted(),
            g.sorted(),
            "system {idx} (n_atoms={}) disagreed between CPU and GPU batched paths",
            systems[idx].n_atoms()
        );
    }
}

/// Real-world-shaped case: the same FCC Cu supercell batch used in the
/// benchmarks, to catch anything the purely-random property test might miss
/// (e.g. lattice-aligned coincidental distances).
#[test]
fn gpu_matches_cpu_on_fcc_batch() {
    const LATTICE_A: f64 = 3.6;
    const BASIS: [[f64; 3]; 4] =
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]];

    fn fcc_cu_system(reps: usize) -> System {
        let side = LATTICE_A * reps as f64;
        let mut positions = Vec::new();
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

    let systems: Vec<System> = (1..=4).map(fcc_cu_system).collect();
    let cutoff = 6.0;

    let cpu = compute_neighbor_lists_batched(&systems, cutoff);
    let gpu = compute_neighbor_lists_batched_gpu(&systems, cutoff);

    for (idx, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(c.sorted(), g.sorted(), "FCC system {idx} disagreed between CPU and GPU");
    }
}

#[test]
fn gpu_empty_batch_returns_empty_vec() {
    let systems: Vec<System> = vec![];
    let out = compute_neighbor_lists_batched_gpu(&systems, 2.0);
    assert!(out.is_empty());
}

/// Forces the auto-retry-on-overflow path: a small `max_output_pairs` guess
/// would be exceeded by a dense system, so this exercises the retry loop
/// itself (via a genuinely dense system relative to a large cutoff).
#[test]
fn gpu_handles_dense_system_needing_retry() {
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let side = 6.0;
    let n = 150;
    let positions = random_positions(&mut rng, n, 0.0, side);
    let systems = vec![System::new(positions, cubic_cell(side), [true, true, true])];
    let cutoff = 5.5; // large relative to the cell -> dense pair count

    let cpu = compute_neighbor_lists_batched(&systems, cutoff);
    let gpu = compute_neighbor_lists_batched_gpu(&systems, cutoff);
    assert_eq!(cpu[0].sorted(), gpu[0].sorted());
}
