use crate::celllist::compute_neighbor_list;
use crate::neighbor_list::NeighborList;
use crate::system::System;

/// Incrementally-updated neighbor list for MD trajectories: builds a
/// candidate pair list at an extended cutoff (`cutoff + skin`) once, then on
/// each [`VerletList::update`] call only recomputes distances for those
/// already-known candidate pairs (cheap) instead of rebuilding the full
/// cell-list from scratch (expensive) — rebuilding only when some atom has
/// moved more than `skin / 2` since the last rebuild.
///
/// Correctness invariant: for any two atoms, the change in their separation
/// distance between rebuilds is bounded by the sum of their individual
/// displacements (triangle inequality). If every atom has moved at most
/// `skin / 2`, that sum is at most `skin`, so a pair excluded from the
/// `cutoff + skin` candidate set (i.e. more than `skin` away from `cutoff`)
/// cannot have entered `cutoff`, and every pair that could be within `cutoff`
/// was already in the candidate set. This is why the trigger is `skin / 2`
/// per atom, not `skin`.
pub struct VerletList {
    cutoff: f64,
    skin: f64,
    reference_positions: Vec<f64>,
    candidates: Vec<(u32, u32, [i32; 3])>,
}

impl VerletList {
    /// Builds a new Verlet list for `system` with the given `cutoff` and
    /// skin buffer. `skin` must be non-negative; `skin = 0.0` degrades to a
    /// full rebuild on every [`VerletList::update`] call.
    pub fn new(system: &System, cutoff: f64, skin: f64) -> Self {
        assert!(cutoff > 0.0, "cutoff must be positive");
        assert!(skin >= 0.0, "skin must be non-negative");
        let mut vl = Self { cutoff, skin, reference_positions: Vec::new(), candidates: Vec::new() };
        vl.rebuild(system);
        vl
    }

    fn rebuild(&mut self, system: &System) {
        let extended = compute_neighbor_list(system, self.cutoff + self.skin);
        self.candidates = extended
            .i
            .iter()
            .zip(extended.j.iter())
            .zip(extended.shift.iter())
            .map(|((&i, &j), &s)| (i, j, s))
            .collect();
        self.reference_positions = system.positions.clone();
    }

    fn needs_rebuild(&self, system: &System) -> bool {
        let half_skin_sq = (self.skin / 2.0).powi(2);
        for a in 0..system.n_atoms() {
            let dx = system.positions[3 * a] - self.reference_positions[3 * a];
            let dy = system.positions[3 * a + 1] - self.reference_positions[3 * a + 1];
            let dz = system.positions[3 * a + 2] - self.reference_positions[3 * a + 2];
            if dx * dx + dy * dy + dz * dz > half_skin_sq {
                return true;
            }
        }
        false
    }

    /// Recomputes the neighbor list for `system`'s current positions,
    /// rebuilding the underlying candidate set first if needed. `system`
    /// must have the same atom count, cell, and PBC flags as when this
    /// `VerletList` was constructed — only positions may change between
    /// calls. Returns `(neighbor_list, did_rebuild)`.
    pub fn update(&mut self, system: &System) -> (NeighborList, bool) {
        assert_eq!(
            system.positions.len(),
            self.reference_positions.len(),
            "VerletList::update called with a different atom count than the system it was built for"
        );
        let did_rebuild = self.needs_rebuild(system);
        if did_rebuild {
            self.rebuild(system);
        }

        let cutoff_sq = self.cutoff * self.cutoff;
        let mut nl = NeighborList::default();
        for &(i, j, s) in &self.candidates {
            let pi = system.position(i as usize);
            let pj = system.position(j as usize);
            let shift_cart = system.cell.cartesian([s[0] as f64, s[1] as f64, s[2] as f64]);
            let dx = pj[0] + shift_cart[0] - pi[0];
            let dy = pj[1] + shift_cart[1] - pi[1];
            let dz = pj[2] + shift_cart[2] - pi[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq <= cutoff_sq {
                nl.push(i, j, s);
            }
        }
        (nl, did_rebuild)
    }
}
