use ferrosim::{compute_neighbor_list, compute_neighbor_list_bruteforce, Cell, System};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn assert_matches_bruteforce(system: &System, cutoff: f64) {
    let fast = compute_neighbor_list(system, cutoff);
    let reference = compute_neighbor_list_bruteforce(system, cutoff);
    assert_eq!(
        fast.sorted(),
        reference.sorted(),
        "cell-list disagreed with brute-force reference (n_atoms={}, cutoff={}, pbc={:?})",
        system.n_atoms(),
        cutoff,
        system.pbc
    );
}

fn random_positions(rng: &mut ChaCha8Rng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..3 * n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn cubic_cell(side: f64) -> Cell {
    Cell::new([[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]])
}

fn triclinic_cell() -> Cell {
    Cell::new([[5.0, 0.0, 0.0], [1.5, 4.0, 0.0], [0.7, 0.3, 3.5]])
}

#[test]
fn single_atom_no_self_neighbor() {
    let system = System::new(vec![0.0, 0.0, 0.0], cubic_cell(10.0), [true, true, true]);
    let nl = compute_neighbor_list(&system, 3.0);
    assert!(nl.is_empty());
    assert_matches_bruteforce(&system, 3.0);
}

#[test]
fn two_atoms_exact_cutoff_boundary() {
    // Distance exactly equal to cutoff must be included (uses <=).
    let system = System::new(
        vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        cubic_cell(20.0),
        [true, true, true],
    );
    assert_matches_bruteforce(&system, 3.0);
    let nl = compute_neighbor_list(&system, 3.0);
    assert_eq!(nl.len(), 2); // (0,1) and (1,0)
}

#[test]
fn two_atoms_just_past_cutoff_excluded() {
    let system = System::new(
        vec![0.0, 0.0, 0.0, 3.0001, 0.0, 0.0],
        cubic_cell(20.0),
        [true, true, true],
    );
    assert_matches_bruteforce(&system, 3.0);
    let nl = compute_neighbor_list(&system, 3.0);
    assert!(nl.is_empty());
}

#[test]
fn small_cell_large_cutoff_orthogonal() {
    // Cell smaller than the cutoff forces multiple periodic image shells.
    let system = System::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], cubic_cell(2.0), [true, true, true]);
    assert_matches_bruteforce(&system, 8.0);
}

#[test]
fn small_cell_large_cutoff_triclinic() {
    let system = System::new(
        vec![0.2, 0.3, 0.1, 1.0, 0.5, 2.0, 2.5, 1.1, 0.4],
        triclinic_cell(),
        [true, true, true],
    );
    assert_matches_bruteforce(&system, 12.0);
}

#[test]
fn non_periodic_cluster() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let positions = random_positions(&mut rng, 20, 0.0, 15.0);
    let system = System::new(positions, cubic_cell(15.0), [false, false, false]);
    assert_matches_bruteforce(&system, 4.0);
}

#[test]
fn mixed_pbc_slab() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let positions = random_positions(&mut rng, 30, 0.0, 12.0);
    let system = System::new(positions, cubic_cell(12.0), [true, true, false]);
    assert_matches_bruteforce(&system, 3.5);
}

#[test]
fn sparse_system_small_cutoff() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let positions = random_positions(&mut rng, 50, 0.0, 100.0);
    let system = System::new(positions, cubic_cell(100.0), [true, true, true]);
    assert_matches_bruteforce(&system, 2.0);
}

#[test]
fn atoms_outside_fundamental_domain() {
    // Positions with fractional coordinates outside [0, 1) (i.e. raw,
    // unwrapped input) previously broke the periodic-shift reconstruction in
    // both the cell-list and the brute-force reference.
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    for trial in 0..20 {
        let n = rng.gen_range(2..40);
        let cutoff = rng.gen_range(1.0..6.0);
        let positions = random_positions(&mut rng, n, -8.0, 8.0);
        let system = System::new(positions, cubic_cell(4.0), [true, true, true]);
        let fast = compute_neighbor_list(&system, cutoff);
        let reference = compute_neighbor_list_bruteforce(&system, cutoff);
        assert_eq!(
            fast.sorted(),
            reference.sorted(),
            "trial {trial} failed: n={n}, cutoff={cutoff}"
        );
    }
}

#[test]
fn random_orthogonal_configs() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for trial in 0..50 {
        let n = rng.gen_range(2..80);
        let side = rng.gen_range(5.0..25.0);
        let cutoff = rng.gen_range(1.0..8.0);
        let positions = random_positions(&mut rng, n, 0.0, side);
        let system = System::new(positions, cubic_cell(side), [true, true, true]);
        let fast = compute_neighbor_list(&system, cutoff);
        let reference = compute_neighbor_list_bruteforce(&system, cutoff);
        assert_eq!(
            fast.sorted(),
            reference.sorted(),
            "trial {trial} failed: n={n}, side={side}, cutoff={cutoff}"
        );
    }
}

#[test]
fn random_triclinic_configs() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let cell = triclinic_cell();
    for trial in 0..50 {
        let n = rng.gen_range(2..60);
        let cutoff = rng.gen_range(1.0..6.0);
        let positions = random_positions(&mut rng, n, -1.0, 5.0);
        let system = System::new(positions, cell.clone(), [true, true, true]);
        let fast = compute_neighbor_list(&system, cutoff);
        let reference = compute_neighbor_list_bruteforce(&system, cutoff);
        assert_eq!(
            fast.sorted(),
            reference.sorted(),
            "trial {trial} failed: n={n}, cutoff={cutoff}"
        );
    }
}

#[test]
fn random_mixed_pbc_configs() {
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    for trial in 0..50 {
        let n = rng.gen_range(2..40);
        let side = rng.gen_range(4.0..20.0);
        let cutoff = rng.gen_range(1.0..7.0);
        let pbc = [rng.gen_bool(0.5), rng.gen_bool(0.5), rng.gen_bool(0.5)];
        let positions = random_positions(&mut rng, n, 0.0, side);
        let system = System::new(positions, cubic_cell(side), pbc);
        let fast = compute_neighbor_list(&system, cutoff);
        let reference = compute_neighbor_list_bruteforce(&system, cutoff);
        assert_eq!(
            fast.sorted(),
            reference.sorted(),
            "trial {trial} failed: n={n}, side={side}, cutoff={cutoff}, pbc={pbc:?}"
        );
    }
}
