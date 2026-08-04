/// A simulation cell defined by three lattice vectors (rows of `matrix`).
///
/// Supports orthogonal and triclinic (non-orthogonal) cells uniformly — both
/// are just a 3x3 matrix, the only difference is whether off-diagonal terms
/// are zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    pub matrix: [[f64; 3]; 3],
}

fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

impl Cell {
    pub fn new(matrix: [[f64; 3]; 3]) -> Self {
        Self { matrix }
    }

    /// Volume (signed) of the cell, i.e. det(matrix).
    pub fn signed_volume(&self) -> f64 {
        let m = &self.matrix;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    pub fn volume(&self) -> f64 {
        self.signed_volume().abs()
    }

    /// Standard closed-form inverse of a 3x3 matrix.
    fn inverse(&self) -> [[f64; 3]; 3] {
        let m = &self.matrix;
        let det = self.signed_volume();
        assert!(det.abs() > 1e-12, "cell matrix is singular (zero volume)");
        let inv_det = 1.0 / det;
        let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
        let (d, e, f) = (m[1][0], m[1][1], m[1][2]);
        let (g, h, i) = (m[2][0], m[2][1], m[2][2]);
        [
            [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
            [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
            [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
        ]
    }

    /// Cartesian position from fractional coordinates: `cart = frac . matrix`
    /// (row-vector convention, matrix rows are the lattice vectors).
    pub fn cartesian(&self, frac: [f64; 3]) -> [f64; 3] {
        let m = &self.matrix;
        [
            frac[0] * m[0][0] + frac[1] * m[1][0] + frac[2] * m[2][0],
            frac[0] * m[0][1] + frac[1] * m[1][1] + frac[2] * m[2][1],
            frac[0] * m[0][2] + frac[1] * m[1][2] + frac[2] * m[2][2],
        ]
    }

    /// Fractional coordinates from a Cartesian position: `frac = cart . matrix^-1`.
    pub fn fractional(&self, cart: [f64; 3]) -> [f64; 3] {
        let inv = self.inverse();
        [
            cart[0] * inv[0][0] + cart[1] * inv[1][0] + cart[2] * inv[2][0],
            cart[0] * inv[0][1] + cart[1] * inv[1][1] + cart[2] * inv[2][1],
            cart[0] * inv[0][2] + cart[1] * inv[1][2] + cart[2] * inv[2][2],
        ]
    }

    /// Distance between opposite faces along each lattice direction
    /// (`volume / area of the opposite face`). This is the standard quantity
    /// used to size cell-list bins for triclinic cells: a bin of this width
    /// along axis `k` is guaranteed to only need to check adjacent bins for
    /// any cutoff <= this width.
    pub fn perpendicular_widths(&self) -> [f64; 3] {
        let m = &self.matrix;
        let (a, b, c) = (m[0], m[1], m[2]);
        let v = self.volume();
        [
            v / norm(cross(b, c)),
            v / norm(cross(c, a)),
            v / norm(cross(a, b)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_cartesian_roundtrip_orthogonal() {
        let cell = Cell::new([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
        let cart = [1.2, -3.4, 2.7];
        let frac = cell.fractional(cart);
        let back = cell.cartesian(frac);
        for k in 0..3 {
            assert!((back[k] - cart[k]).abs() < 1e-9);
        }
    }

    #[test]
    fn fractional_cartesian_roundtrip_triclinic() {
        // A skewed (monoclinic-like) cell.
        let cell = Cell::new([[5.0, 0.0, 0.0], [1.5, 4.0, 0.0], [0.7, 0.3, 3.5]]);
        let cart = [2.0, 1.0, 0.5];
        let frac = cell.fractional(cart);
        let back = cell.cartesian(frac);
        for k in 0..3 {
            assert!((back[k] - cart[k]).abs() < 1e-9);
        }
    }

    #[test]
    fn perpendicular_width_matches_orthogonal_case() {
        let cell = Cell::new([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
        let w = cell.perpendicular_widths();
        assert!((w[0] - 5.0).abs() < 1e-9);
        assert!((w[1] - 3.0).abs() < 1e-9);
        assert!((w[2] - 4.0).abs() < 1e-9);
    }
}
