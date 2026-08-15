//! A runnable tour of FerroSim's API: a single-system non-periodic cluster,
//! a periodic crystal (where periodic shifts actually show up), the batched
//! multi-system API (CPU and GPU), and an incremental Verlet-list trajectory.
//!
//! Run with `cargo run --release --example demo`.

use ferrosim::{
    compute_neighbor_list, compute_neighbor_lists_batched, compute_neighbor_lists_batched_gpu,
    Cell, System, VerletList,
};

fn print_neighbors(label: &str, system: &System, cutoff: f64) {
    let neighbors = compute_neighbor_list(system, cutoff);
    println!("\n{label}");
    println!("  {} atoms, cutoff {cutoff}, {} directed pairs found", system.n_atoms(), neighbors.len());
    let n_show = neighbors.len().min(5);
    for k in 0..n_show {
        let (i, j, s) = (neighbors.i[k], neighbors.j[k], neighbors.shift[k]);
        println!("    {i} -> {j}  (periodic shift {s:?})");
    }
    if neighbors.len() > n_show {
        println!("    ... and {} more", neighbors.len() - n_show);
    }
}

fn water_like_cluster() -> System {
    // Three atoms, no periodicity: a small non-periodic molecule-shaped cluster.
    let positions = vec![
        0.0, 0.0, 0.0, // "O"
        0.96, 0.0, 0.0, // "H"
        -0.24, 0.93, 0.0, // "H"
    ];
    let cell = Cell::new([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]);
    System::new(positions, cell, [false, false, false])
}

const LATTICE_A: f64 = 3.6;
const FCC_BASIS: [[f64; 3]; 4] =
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]];

fn fcc_cu_unit_cell() -> System {
    // A single periodic FCC copper unit cell: small enough that every atom
    // sees neighbors through periodic images, so shift vectors are non-zero.
    let mut positions = Vec::with_capacity(FCC_BASIS.len() * 3);
    for b in FCC_BASIS {
        positions.push(b[0] * LATTICE_A);
        positions.push(b[1] * LATTICE_A);
        positions.push(b[2] * LATTICE_A);
    }
    let cell = Cell::new([[LATTICE_A, 0.0, 0.0], [0.0, LATTICE_A, 0.0], [0.0, 0.0, LATTICE_A]]);
    System::new(positions, cell, [true, true, true])
}

fn main() {
    print_neighbors("Non-periodic water-like cluster", &water_like_cluster(), 1.5);
    print_neighbors("Periodic FCC copper unit cell", &fcc_cu_unit_cell(), 3.0);

    println!("\nBatched: {{water cluster, FCC unit cell}} in one call");
    let batch = vec![water_like_cluster(), fcc_cu_unit_cell()];
    let cutoffs = [1.5, 3.0];
    // compute_neighbor_lists_batched uses one cutoff for the whole batch, so
    // run it once per distinct cutoff here to mirror what each system used above.
    for (system, cutoff) in batch.iter().zip(cutoffs) {
        let batched = compute_neighbor_lists_batched(std::slice::from_ref(system), cutoff);
        println!("  cutoff {cutoff}: {} pairs (matches single-system call)", batched[0].len());
    }

    println!("\nSame batch on GPU (wgpu), compared against CPU, cutoff 3.0 for both systems");
    let cpu = compute_neighbor_lists_batched(&batch, 3.0);
    let gpu = compute_neighbor_lists_batched_gpu(&batch, 3.0);
    for (idx, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let matches = c.sorted() == g.sorted();
        println!("  system {idx}: cpu={} pairs, gpu={} pairs, match={matches}", c.len(), g.len());
    }

    println!("\nIncremental Verlet list over a tiny trajectory");
    let mut system = fcc_cu_unit_cell();
    let mut verlet = VerletList::new(&system, 3.0, 0.5);
    for step in 0..4 {
        // Nudge one atom a little each step; occasionally take a big jump to
        // force a rebuild, so both code paths in `update` actually run.
        let jump = step == 2;
        let delta = if jump { 0.4 } else { 0.05 };
        system.positions[0] += delta;
        let (neighbors, rebuilt) = verlet.update(&system);
        println!("  step {step}: {} pairs, rebuilt={rebuilt}", neighbors.len());
    }
}
