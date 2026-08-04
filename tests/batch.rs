use ferrosim::{compute_neighbor_list, compute_neighbor_lists_batched, Cell, System};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn random_positions(rng: &mut ChaCha8Rng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..3 * n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn cubic_cell(side: f64) -> Cell {
    Cell::new([[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]])
}

fn triclinic_cell() -> Cell {
    Cell::new([[5.0, 0.0, 0.0], [1.5, 4.0, 0.0], [0.7, 0.3, 3.5]])
}

/// The batched API must produce, for every system in the batch, exactly the
/// same result as calling the single-system API on that system alone —
/// batching is purely a scheduling/throughput optimization, not a different
/// algorithm.
#[test]
fn batched_matches_looped_single_system() {
    let mut rng = ChaCha8Rng::seed_from_u64(2024);
    let cutoff = 3.5;

    let mut systems = Vec::new();
    for _ in 0..12 {
        let n = rng.gen_range(1..30);
        let side = rng.gen_range(6.0..15.0);
        let positions = random_positions(&mut rng, n, 0.0, side);
        systems.push(System::new(positions, cubic_cell(side), [true, true, true]));
    }
    // Mix in triclinic systems and one non-periodic cluster.
    let n = rng.gen_range(1..20);
    systems.push(System::new(
        random_positions(&mut rng, n, -2.0, 6.0),
        triclinic_cell(),
        [true, true, true],
    ));
    let n = rng.gen_range(1..20);
    systems.push(System::new(random_positions(&mut rng, n, 0.0, 10.0), cubic_cell(10.0), [false, false, false]));

    let batched = compute_neighbor_lists_batched(&systems, cutoff);
    assert_eq!(batched.len(), systems.len());

    for (idx, system) in systems.iter().enumerate() {
        let single = compute_neighbor_list(system, cutoff);
        assert_eq!(
            batched[idx].sorted(),
            single.sorted(),
            "system {idx} (n_atoms={}) disagreed between batched and single-system paths",
            system.n_atoms()
        );
    }
}

#[test]
fn batch_handles_empty_and_single_atom_systems() {
    let systems = vec![
        System::new(vec![], cubic_cell(10.0), [true, true, true]),
        System::new(vec![0.0, 0.0, 0.0], cubic_cell(10.0), [true, true, true]),
        System::new(vec![], cubic_cell(5.0), [false, false, false]),
    ];
    let batched = compute_neighbor_lists_batched(&systems, 2.0);
    assert!(batched[0].is_empty());
    assert!(batched[1].is_empty());
    assert!(batched[2].is_empty());
}

#[test]
fn empty_batch_returns_empty_vec() {
    let systems: Vec<System> = vec![];
    let batched = compute_neighbor_lists_batched(&systems, 2.0);
    assert!(batched.is_empty());
}
