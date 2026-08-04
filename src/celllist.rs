use std::collections::HashMap;

use crate::neighbor_list::NeighborList;
use crate::system::System;

/// For a candidate bin index `idx` along one axis with `n_bins` bins:
/// - if periodic: wrap into `[0, n_bins)` and return the integer number of
///   whole periodic cells crossed (so the caller can reconstruct the true
///   periodic shift); always valid ("ok" = true).
/// - if not periodic: valid only when `idx` already falls inside
///   `[0, n_bins)` — there is no bin to search outside the data's extent.
fn wrap_axis(idx: i32, n_bins: i32, periodic: bool) -> Option<(i32, i32)> {
    if periodic {
        let wrapped = idx.rem_euclid(n_bins);
        let shift = (idx - wrapped) / n_bins;
        Some((wrapped, shift))
    } else if idx < 0 || idx >= n_bins {
        None
    } else {
        Some((idx, 0))
    }
}

/// Fast cell-list (spatial hashing) neighbor-list construction: partitions
/// the simulation cell into a grid of bins sized to the cutoff radius (via
/// the cell's perpendicular widths, so triclinic cells are handled
/// correctly), bins atoms in O(n), then only checks neighbor candidates in
/// adjacent bins rather than all pairs.
///
/// Produces a full list (both `(i, j, S)` and `(j, i, -S)`), matching
/// `bruteforce::compute_neighbor_list_bruteforce`'s convention.
pub fn compute_neighbor_list(system: &System, cutoff: f64) -> NeighborList {
    assert!(cutoff > 0.0, "cutoff must be positive");
    let n = system.n_atoms();
    let mut nl = NeighborList::default();
    if n == 0 {
        return nl;
    }

    let frac: Vec<[f64; 3]> = (0..n).map(|a| system.cell.fractional(system.position(a))).collect();
    let perp = system.cell.perpendicular_widths();

    // Per-axis: number of bins, the "wrapped" coordinate atoms are binned on,
    // and (for periodic axes) the integer floor removed by wrapping — needed
    // later to reconstruct the true periodic shift for a found pair.
    let mut num_bins = [1usize; 3];
    let mut bin_span = [0.0f64; 3];
    let mut wrapped = vec![[0.0f64; 3]; n];
    let mut floor_coord = vec![[0i32; 3]; n];

    for k in 0..3 {
        if system.pbc[k] {
            bin_span[k] = perp[k];
            num_bins[k] = ((perp[k] / cutoff).floor() as usize).max(1);
            for a in 0..n {
                let f = frac[a][k].floor();
                floor_coord[a][k] = f as i32;
                wrapped[a][k] = frac[a][k] - f;
            }
        } else {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for a in 0..n {
                lo = lo.min(frac[a][k]);
                hi = hi.max(frac[a][k]);
            }
            let extent = (hi - lo).max(1e-12);
            bin_span[k] = extent;
            num_bins[k] = ((extent / cutoff).floor() as usize).max(1);
            for a in 0..n {
                floor_coord[a][k] = 0;
                wrapped[a][k] = frac[a][k] - lo;
            }
        }
    }

    // How many neighbor-bin shells to search per axis: normally 1 (bins are
    // sized >= cutoff), but more when the cell itself is smaller than the
    // cutoff (num_bins forced to 1 above, so a single bin can be narrower
    // than cutoff and multiple periodic images must be checked).
    let mut shell = [1i32; 3];
    for k in 0..3 {
        let bin_width = bin_span[k] / num_bins[k] as f64;
        shell[k] = (cutoff / bin_width).ceil().max(1.0) as i32;
        if !system.pbc[k] {
            shell[k] = shell[k].min(num_bins[k] as i32);
        }
    }

    let mut bins: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    let mut atom_bin = vec![[0i32; 3]; n];
    for a in 0..n {
        let idx = [
            ((wrapped[a][0] * num_bins[0] as f64).floor() as i32).clamp(0, num_bins[0] as i32 - 1),
            ((wrapped[a][1] * num_bins[1] as f64).floor() as i32).clamp(0, num_bins[1] as i32 - 1),
            ((wrapped[a][2] * num_bins[2] as f64).floor() as i32).clamp(0, num_bins[2] as i32 - 1),
        ];
        atom_bin[a] = idx;
        bins.entry((idx[0], idx[1], idx[2])).or_default().push(a);
    }

    let cutoff_sq = cutoff * cutoff;

    for i in 0..n {
        let bi = atom_bin[i];
        let pi = system.position(i);
        for d0 in -shell[0]..=shell[0] {
            let Some((nb0, bshift0)) = wrap_axis(bi[0] + d0, num_bins[0] as i32, system.pbc[0])
            else {
                continue;
            };
            for d1 in -shell[1]..=shell[1] {
                let Some((nb1, bshift1)) =
                    wrap_axis(bi[1] + d1, num_bins[1] as i32, system.pbc[1])
                else {
                    continue;
                };
                for d2 in -shell[2]..=shell[2] {
                    let Some((nb2, bshift2)) =
                        wrap_axis(bi[2] + d2, num_bins[2] as i32, system.pbc[2])
                    else {
                        continue;
                    };

                    let Some(atoms) = bins.get(&(nb0, nb1, nb2)) else {
                        continue;
                    };
                    for &j in atoms {
                        // `bshift` is relative to atom i's *wrapped* (canonical,
                        // floor-removed) position, not its true fractional position —
                        // so it must be corrected by both atoms' removed floors to give
                        // the true periodic shift between the actual atom positions.
                        let s = [
                            bshift0 - floor_coord[j][0] + floor_coord[i][0],
                            bshift1 - floor_coord[j][1] + floor_coord[i][1],
                            bshift2 - floor_coord[j][2] + floor_coord[i][2],
                        ];
                        if i == j && s == [0, 0, 0] {
                            continue;
                        }
                        let pj = system.position(j);
                        let shift_cart =
                            system.cell.cartesian([s[0] as f64, s[1] as f64, s[2] as f64]);
                        let dx = pj[0] + shift_cart[0] - pi[0];
                        let dy = pj[1] + shift_cart[1] - pi[1];
                        let dz = pj[2] + shift_cart[2] - pi[2];
                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        if dist_sq <= cutoff_sq {
                            nl.push(i as u32, j as u32, s);
                        }
                    }
                }
            }
        }
    }

    nl
}
