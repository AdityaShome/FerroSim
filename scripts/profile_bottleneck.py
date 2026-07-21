"""
Phase 0, Task 3: profile the neighbor-list-construction vs. neural-network-forward-pass
split in a real MACE pipeline, at a few system sizes, to get our own numbers rather
than relying solely on the papers cited in project.md.

Usage:
    python scripts/profile_bottleneck.py
"""
import time

import numpy as np
from ase.build import bulk
from ase.neighborlist import primitive_neighbor_list
from mace.calculators import mace_mp


def make_system(n_target_atoms: int):
    """Build an FCC Cu supercell with at least n_target_atoms atoms."""
    unit = bulk("Cu", "fcc", a=3.6, cubic=True)  # 4 atoms/cell
    reps = max(1, round((n_target_atoms / len(unit)) ** (1 / 3)))
    atoms = unit.repeat((reps, reps, reps))
    return atoms


def time_neighbor_list(atoms, cutoff, n_repeats=5):
    # ASE's own reference neighbor-list builder (the "default backend").
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        primitive_neighbor_list(
            "ijS",
            pbc=atoms.pbc,
            cell=atoms.cell.array,
            positions=atoms.positions,
            cutoff=cutoff,
        )
        times.append(time.perf_counter() - t0)
    return min(times)


def time_forward_pass(atoms, calc, n_repeats=5):
    atoms = atoms.copy()
    atoms.calc = calc
    # warm up (first call includes lazy graph construction / JIT-ish overhead)
    atoms.get_potential_energy()
    times = []
    for _ in range(n_repeats):
        atoms.calc.results = {}  # force recompute, bypass ASE's own caching
        t0 = time.perf_counter()
        atoms.get_potential_energy()
        atoms.get_forces()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    cutoff = 6.0  # Angstrom, typical MACE-MP receptive-field-per-layer cutoff
    print("Loading MACE-MP (small) on CPU...")
    calc = mace_mp(model="small", device="cpu", default_dtype="float32")

    print(f"{'n_atoms':>8} | {'nlist_s':>10} | {'forward_s':>10} | {'nlist_%':>8}")
    print("-" * 46)
    for target in (50, 500, 5000):
        atoms = make_system(target)
        n = len(atoms)
        t_nlist = time_neighbor_list(atoms, cutoff)
        t_fwd = time_forward_pass(atoms, calc)
        pct = 100 * t_nlist / (t_nlist + t_fwd)
        print(f"{n:>8} | {t_nlist:>10.5f} | {t_fwd:>10.5f} | {pct:>7.1f}%")


if __name__ == "__main__":
    main()
