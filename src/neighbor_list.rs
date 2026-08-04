/// Output of a neighbor-list construction: parallel arrays of source/destination
/// atom indices plus the periodic shift (in lattice-vector units) needed to
/// translate atom `j` to the specific periodic image that is within cutoff of
/// atom `i`. Matches the `i, j, S` convention used by ASE / torch_nl, so this
/// is a drop-in-compatible output format.
///
/// This is always a *full* list: both `(i, j, S)` and `(j, i, -S)` are present
/// for every neighboring pair, since MLIP force/energy calculations need each
/// atom's complete neighbor set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NeighborList {
    pub i: Vec<u32>,
    pub j: Vec<u32>,
    pub shift: Vec<[i32; 3]>,
}

impl NeighborList {
    pub fn len(&self) -> usize {
        self.i.len()
    }

    pub fn is_empty(&self) -> bool {
        self.i.is_empty()
    }

    pub fn push(&mut self, i: u32, j: u32, shift: [i32; 3]) {
        self.i.push(i);
        self.j.push(j);
        self.shift.push(shift);
    }

    /// Canonical (sorted) form of the pair list, so two `NeighborList`s
    /// covering the same physical pairs compare equal regardless of the
    /// order pairs were discovered in. Used to compare the cell-list
    /// implementation against the brute-force reference in tests.
    pub fn sorted(&self) -> Vec<(u32, u32, [i32; 3])> {
        let mut v: Vec<(u32, u32, [i32; 3])> = self
            .i
            .iter()
            .zip(self.j.iter())
            .zip(self.shift.iter())
            .map(|((&i, &j), &s)| (i, j, s))
            .collect();
        v.sort();
        v
    }
}
