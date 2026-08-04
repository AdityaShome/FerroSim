use rayon::prelude::*;

use crate::celllist::{build_bin_grid, neighbors_for_atom, BinGrid};
use crate::neighbor_list::NeighborList;
use crate::system::System;

/// Computes neighbor lists for a batch of independent atomic systems (the
/// actual MLIP training/inference workload shape: many small, unrelated
/// systems processed together, as opposed to one large single-system MD run).
///
/// Rather than looping `compute_neighbor_list` per system (which would create
/// one small `rayon` task pool per system, starving core utilization when
/// individual systems are small), this flattens every system's per-atom
/// search into a single `rayon` parallel iteration spanning the whole batch —
/// so the thread pool always has enough independent work to schedule
/// regardless of how small any one system is.
///
/// All systems share the same `cutoff`, matching the common case of a single
/// MLIP model applied across a batch (per-system cutoffs are not supported).
pub fn compute_neighbor_lists_batched(systems: &[System], cutoff: f64) -> Vec<NeighborList> {
    assert!(cutoff > 0.0, "cutoff must be positive");

    let grids: Vec<BinGrid> = systems.par_iter().map(|s| build_bin_grid(s, cutoff)).collect();

    let work: Vec<(usize, usize)> = systems
        .iter()
        .enumerate()
        .flat_map(|(si, s)| (0..s.n_atoms()).map(move |ai| (si, ai)))
        .collect();

    let cutoff_sq = cutoff * cutoff;
    let results: Vec<(usize, u32, u32, [i32; 3])> = work
        .into_par_iter()
        .flat_map_iter(|(si, ai)| {
            neighbors_for_atom(&systems[si], cutoff_sq, &grids[si], ai)
                .into_iter()
                .map(move |(i, j, s)| (si, i, j, s))
        })
        .collect();

    let mut out: Vec<NeighborList> = (0..systems.len()).map(|_| NeighborList::default()).collect();
    for (si, i, j, s) in results {
        out[si].push(i, j, s);
    }
    out
}
