use crate::neighbor_list::NeighborList;
use crate::system::System;

/// Naive O(n^2) neighbor-list construction, kept deliberately simple and slow
/// so it is trustworthy as a correctness reference for the cell-list
/// implementation. Works directly in Cartesian coordinates (position + shift
/// converted to Cartesian via `cell.cartesian`) rather than reusing the
/// cell-list's fractional-coordinate binning machinery, so the two
/// implementations are genuinely independent checks on each other.
pub fn compute_neighbor_list_bruteforce(system: &System, cutoff: f64) -> NeighborList {
    assert!(cutoff > 0.0, "cutoff must be positive");
    let n = system.n_atoms();
    let mut nl = NeighborList::default();
    if n == 0 {
        return nl;
    }

    // Upper bound on how many periodic images along each axis could possibly
    // contain a neighbor within `cutoff`. `cutoff / perp[k]` alone is only a
    // valid bound for atoms already wrapped into the `[0, 1)` fractional
    // range; positions here are arbitrary raw Cartesian input, so atoms can
    // start out more than one cell apart before any shift is applied. Pad by
    // the actual fractional-coordinate spread of the input along each axis
    // to cover that too.
    let perp = system.cell.perpendicular_widths();
    let mut n_shell = [0i32; 3];
    for k in 0..3 {
        if system.pbc[k] {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for a in 0..n {
                let f = system.cell.fractional(system.position(a))[k];
                lo = lo.min(f);
                hi = hi.max(f);
            }
            let span = (hi - lo).max(0.0).ceil();
            n_shell[k] = (cutoff / perp[k]).ceil().max(0.0) as i32 + span as i32 + 1;
        }
    }

    let cutoff_sq = cutoff * cutoff;

    for i in 0..n {
        let pi = system.position(i);
        for j in i..n {
            let pj = system.position(j);
            let s0_range = if system.pbc[0] { -n_shell[0]..=n_shell[0] } else { 0..=0 };
            for s0 in s0_range.clone() {
                let s1_range = if system.pbc[1] { -n_shell[1]..=n_shell[1] } else { 0..=0 };
                for s1 in s1_range.clone() {
                    let s2_range = if system.pbc[2] { -n_shell[2]..=n_shell[2] } else { 0..=0 };
                    for s2 in s2_range.clone() {
                        if i == j && s0 == 0 && s1 == 0 && s2 == 0 {
                            continue;
                        }
                        let shift = [s0, s1, s2];
                        let shift_cart =
                            system.cell.cartesian([s0 as f64, s1 as f64, s2 as f64]);
                        let dx = pj[0] + shift_cart[0] - pi[0];
                        let dy = pj[1] + shift_cart[1] - pi[1];
                        let dz = pj[2] + shift_cart[2] - pi[2];
                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        if dist_sq <= cutoff_sq {
                            // When i == j, the shift loop already visits `shift` and
                            // `-shift` as separate iterations, so only push once here —
                            // otherwise self-image pairs get double-counted.
                            if i == j {
                                nl.push(i as u32, j as u32, shift);
                            } else {
                                nl.push(i as u32, j as u32, shift);
                                nl.push(j as u32, i as u32, [-s0, -s1, -s2]);
                            }
                        }
                    }
                }
            }
        }
    }
    nl
}
