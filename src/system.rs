use crate::cell::Cell;

/// An atomic system: positions (flat, not nested — matters for performance),
/// a simulation cell, and which axes are periodic.
#[derive(Debug, Clone)]
pub struct System {
    /// Flat Cartesian positions, length `3 * n_atoms` (x0,y0,z0,x1,y1,z1,...).
    pub positions: Vec<f64>,
    pub cell: Cell,
    pub pbc: [bool; 3],
}

impl System {
    pub fn new(positions: Vec<f64>, cell: Cell, pbc: [bool; 3]) -> Self {
        assert!(
            positions.len() % 3 == 0,
            "positions must be a flat array of length 3 * n_atoms"
        );
        Self { positions, cell, pbc }
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len() / 3
    }

    pub fn position(&self, i: usize) -> [f64; 3] {
        [self.positions[3 * i], self.positions[3 * i + 1], self.positions[3 * i + 2]]
    }
}
