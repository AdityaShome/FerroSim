// GPU port of the CPU cell-list search (src/celllist.rs::neighbors_for_atom).
// One thread per atom, across the *whole batch* flattened together (mirrors
// src/batch.rs's rayon flat_map strategy): each system's bins occupy a
// disjoint range of "global bin ids" (via per-system `bin_offset`), so a
// thread for an atom in system s only ever visits bins belonging to s.
//
// Binning itself (assigning atoms to bins, building the CSR bin_start/
// bin_atoms arrays) is done on the CPU host side, not here — it's O(n) and
// cheap relative to the O(n * shell^3) search this shader parallelizes; see
// docs/decision_log.md for the reasoning.

struct Params {
    cutoff_sq: f32,
    n_atoms_total: u32,
    max_output_pairs: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<f32>;      // flat xyz, len 3*n_atoms_total
@group(0) @binding(1) var<storage, read> floor_coord: array<i32>;    // flat xyz, len 3*n_atoms_total
@group(0) @binding(2) var<storage, read> atom_bin: array<i32>;       // flat xyz local bin coords, len 3*n_atoms_total
@group(0) @binding(3) var<storage, read> atom_system: array<u32>;    // len n_atoms_total
@group(0) @binding(4) var<storage, read> sys_ints: array<i32>;       // flat 10 per system: nb0,nb1,nb2,sh0,sh1,sh2,pbc0,pbc1,pbc2,bin_offset
@group(0) @binding(5) var<storage, read> sys_cell: array<f32>;       // flat 9 per system (row-major 3x3 lattice matrix)
@group(0) @binding(6) var<storage, read> bin_start: array<u32>;      // CSR offsets, len total_bins + 1
@group(0) @binding(7) var<storage, read> bin_atoms: array<u32>;      // CSR global atom indices, len n_atoms_total
@group(0) @binding(8) var<uniform> params: Params;
@group(0) @binding(9) var<storage, read_write> out_count: atomic<u32>;
@group(0) @binding(10) var<storage, read_write> out_pairs: array<i32>; // flat 5 per pair: i, j, s0, s1, s2

// Mirrors celllist::wrap_axis exactly: returns (valid, wrapped, shift).
// WGSL's `%` is truncating (sign of dividend), so a manual correction turns
// it into Rust's `rem_euclid`.
fn wrap_axis(idx: i32, n_bins: i32, periodic: i32) -> vec3<i32> {
    if (periodic != 0) {
        var wrapped = idx % n_bins;
        if (wrapped < 0) {
            wrapped = wrapped + n_bins;
        }
        let shift = (idx - wrapped) / n_bins;
        return vec3<i32>(1, wrapped, shift);
    }
    if (idx < 0 || idx >= n_bins) {
        return vec3<i32>(0, 0, 0);
    }
    return vec3<i32>(1, idx, 0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms_total) {
        return;
    }

    let sys = atom_system[i];
    let ibase = sys * 10u;
    let nb0 = sys_ints[ibase + 0u];
    let nb1 = sys_ints[ibase + 1u];
    let nb2 = sys_ints[ibase + 2u];
    let sh0 = sys_ints[ibase + 3u];
    let sh1 = sys_ints[ibase + 4u];
    let sh2 = sys_ints[ibase + 5u];
    let pbc0 = sys_ints[ibase + 6u];
    let pbc1 = sys_ints[ibase + 7u];
    let pbc2 = sys_ints[ibase + 8u];
    let bin_offset = u32(sys_ints[ibase + 9u]);

    let bi0 = atom_bin[i * 3u + 0u];
    let bi1 = atom_bin[i * 3u + 1u];
    let bi2 = atom_bin[i * 3u + 2u];
    let fi0 = floor_coord[i * 3u + 0u];
    let fi1 = floor_coord[i * 3u + 1u];
    let fi2 = floor_coord[i * 3u + 2u];
    let pi = vec3<f32>(positions[i * 3u + 0u], positions[i * 3u + 1u], positions[i * 3u + 2u]);

    let cbase = sys * 9u;

    for (var d0 = -sh0; d0 <= sh0; d0 = d0 + 1) {
        let w0 = wrap_axis(bi0 + d0, nb0, pbc0);
        if (w0.x == 0) {
            continue;
        }
        for (var d1 = -sh1; d1 <= sh1; d1 = d1 + 1) {
            let w1 = wrap_axis(bi1 + d1, nb1, pbc1);
            if (w1.x == 0) {
                continue;
            }
            for (var d2 = -sh2; d2 <= sh2; d2 = d2 + 1) {
                let w2 = wrap_axis(bi2 + d2, nb2, pbc2);
                if (w2.x == 0) {
                    continue;
                }

                let bin_id_local = w0.y + nb0 * (w1.y + nb1 * w2.y);
                let global_bin = bin_offset + u32(bin_id_local);
                let start = bin_start[global_bin];
                let end = bin_start[global_bin + 1u];

                for (var k = start; k < end; k = k + 1u) {
                    let j = bin_atoms[k];
                    let fj0 = floor_coord[j * 3u + 0u];
                    let fj1 = floor_coord[j * 3u + 1u];
                    let fj2 = floor_coord[j * 3u + 2u];

                    let s0 = w0.z - fj0 + fi0;
                    let s1 = w1.z - fj1 + fi1;
                    let s2 = w2.z - fj2 + fi2;

                    if (i == j && s0 == 0 && s1 == 0 && s2 == 0) {
                        continue;
                    }

                    let sf0 = f32(s0);
                    let sf1 = f32(s1);
                    let sf2 = f32(s2);
                    let shift_cart = vec3<f32>(
                        sf0 * sys_cell[cbase + 0u] + sf1 * sys_cell[cbase + 3u] + sf2 * sys_cell[cbase + 6u],
                        sf0 * sys_cell[cbase + 1u] + sf1 * sys_cell[cbase + 4u] + sf2 * sys_cell[cbase + 7u],
                        sf0 * sys_cell[cbase + 2u] + sf1 * sys_cell[cbase + 5u] + sf2 * sys_cell[cbase + 8u],
                    );

                    let pj = vec3<f32>(positions[j * 3u + 0u], positions[j * 3u + 1u], positions[j * 3u + 2u]);
                    let d = pj + shift_cart - pi;
                    let dist_sq = dot(d, d);

                    if (dist_sq <= params.cutoff_sq) {
                        let slot = atomicAdd(&out_count, 1u);
                        if (slot < params.max_output_pairs) {
                            let o = slot * 5u;
                            out_pairs[o + 0u] = i32(i);
                            out_pairs[o + 1u] = i32(j);
                            out_pairs[o + 2u] = s0;
                            out_pairs[o + 3u] = s1;
                            out_pairs[o + 4u] = s2;
                        }
                    }
                }
            }
        }
    }
}
