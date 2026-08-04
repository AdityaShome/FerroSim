use ferrosim::{compute_neighbor_list, Cell, System, VerletList};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn cubic_cell(side: f64) -> Cell {
    Cell::new([[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]])
}

/// Hand-worked case: verifies the rebuild trigger fires exactly when the
/// documented invariant says it must (total displacement since the last
/// rebuild exceeds `skin / 2`), not before and not after, and that the
/// returned list is always correct regardless of whether a rebuild happened.
#[test]
fn verlet_rebuild_trigger_exact() {
    let cutoff = 3.0;
    let skin = 1.0; // rebuild trigger: displacement > 0.5
    let cell = cubic_cell(20.0);
    let mut system = System::new(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0], cell, [false, false, false]);
    let mut vl = VerletList::new(&system, cutoff, skin);

    // Move atom 1 from x=2.0 to x=2.4: displacement 0.4 < 0.5, no rebuild.
    system.positions[3] = 2.4;
    let (nl, rebuilt) = vl.update(&system);
    assert!(!rebuilt, "displacement 0.4 should stay under the skin/2 trigger");
    assert_eq!(nl.len(), 2); // (0,1) and (1,0), distance 2.4 <= cutoff 3.0

    // Move atom 1 further to x=2.9: cumulative displacement from the original
    // rebuild reference (x=2.0) is 0.9 > 0.5, rebuild must trigger.
    system.positions[3] = 2.9;
    let (nl, rebuilt) = vl.update(&system);
    assert!(rebuilt, "cumulative displacement 0.9 should exceed the skin/2 trigger");
    assert_eq!(nl.len(), 2); // distance 2.9 <= cutoff 3.0, still a neighbor

    // Move atom 1 to x=4.0: now outside cutoff entirely. Displacement from
    // the last rebuild reference (x=2.9) is 1.1 > 0.5, rebuild triggers, and
    // the pair correctly drops out.
    system.positions[3] = 4.0;
    let (nl, rebuilt) = vl.update(&system);
    assert!(rebuilt);
    assert!(nl.is_empty(), "distance 4.0 exceeds cutoff 3.0");

    assert_eq!(compute_neighbor_list(&system, cutoff).sorted(), nl.sorted());
}

/// Property test: across a random trajectory with a mix of small steps (no
/// rebuild expected) and occasional large jumps (rebuild expected), the
/// incrementally-updated list must always exactly equal a fresh full
/// recompute — this is the correctness invariant a Verlet list exists to
/// preserve. Also checks both the rebuild and no-rebuild code paths actually
/// get exercised, so the test isn't accidentally only covering one branch.
#[test]
fn verlet_matches_full_rebuild_across_trajectory() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let cutoff = 3.0;
    let skin = 1.2;
    let side = 12.0;
    let n = 25;

    let positions: Vec<f64> = (0..3 * n).map(|_| rng.gen_range(0.0..side)).collect();
    let mut system = System::new(positions, cubic_cell(side), [true, true, true]);
    let mut vl = VerletList::new(&system, cutoff, skin);

    let mut saw_rebuild = false;
    let mut saw_no_rebuild = false;

    for step in 0..40 {
        // Every 7th step, take a large jump (bigger than the skin) on one
        // random atom to deliberately force a rebuild; otherwise take small
        // steps for every atom.
        if step % 7 == 0 {
            let a = rng.gen_range(0..n);
            for k in 0..3 {
                system.positions[3 * a + k] += rng.gen_range(-2.0..2.0);
            }
        } else {
            for p in system.positions.iter_mut() {
                *p += rng.gen_range(-0.15..0.15);
            }
        }

        let (incremental, rebuilt) = vl.update(&system);
        let fresh = compute_neighbor_list(&system, cutoff);
        assert_eq!(
            incremental.sorted(),
            fresh.sorted(),
            "step {step} disagreed between incremental Verlet update and full rebuild"
        );
        if rebuilt {
            saw_rebuild = true;
        } else {
            saw_no_rebuild = true;
        }
    }

    assert!(saw_rebuild, "test never exercised the rebuild path");
    assert!(saw_no_rebuild, "test never exercised the no-rebuild path");
}

#[test]
fn verlet_zero_skin_always_rebuilds() {
    let cell = cubic_cell(20.0);
    let mut system = System::new(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0], cell, [false, false, false]);
    let mut vl = VerletList::new(&system, 3.0, 0.0);
    system.positions[3] = 2.001;
    let (_, rebuilt) = vl.update(&system);
    assert!(rebuilt, "zero skin means any movement at all should trigger a rebuild");
}
