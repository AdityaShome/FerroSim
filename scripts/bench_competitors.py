"""
Phase 0, Task 4: benchmark existing neighbor-list tools (Vesin, torch_nl, ASE's own
`primitive_neighbor_list`) on the same test systems used in profile_bottleneck.py.
This is the competitive baseline FerroSim's Phase 1 CPU implementation will be
compared against.

Usage:
    python scripts/bench_competitors.py
"""
import time

import numpy as np
import torch
from ase.build import bulk
from ase.neighborlist import primitive_neighbor_list
from vesin import NeighborList as VesinNeighborList
from torch_nl import compute_neighborlist


def make_system(n_target_atoms: int):
    unit = bulk("Cu", "fcc", a=3.6, cubic=True)
    reps = max(1, round((n_target_atoms / len(unit)) ** (1 / 3)))
    return unit.repeat((reps, reps, reps))


def time_it(fn, n_repeats=7):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def bench_ase(atoms, cutoff):
    return time_it(
        lambda: primitive_neighbor_list(
            "ijS", pbc=atoms.pbc, cell=atoms.cell.array,
            positions=atoms.positions, cutoff=cutoff,
        )
    )


def bench_vesin(atoms, cutoff):
    nl = VesinNeighborList(cutoff=cutoff, full_list=True)
    return time_it(
        lambda: nl.compute(
            points=atoms.positions,
            box=atoms.cell.array,
            periodic=bool(atoms.pbc.any()),
            quantities="ij",
        )
    )


def bench_torch_nl(atoms, cutoff):
    pos = torch.tensor(atoms.positions, dtype=torch.float64)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float64).unsqueeze(0)
    pbc = torch.tensor(atoms.pbc).unsqueeze(0)
    batch = torch.zeros(len(atoms), dtype=torch.long)
    return time_it(
        lambda: compute_neighborlist(
            cutoff, pos, cell, pbc, batch, self_interaction=False
        )
    )


def main():
    cutoff = 6.0
    print(f"{'n_atoms':>8} | {'ase_s':>10} | {'vesin_s':>10} | {'torch_nl_s':>10}")
    print("-" * 48)
    for target in (50, 500, 2000):
        atoms = make_system(target)
        n = len(atoms)
        t_ase = bench_ase(atoms, cutoff)
        t_vesin = bench_vesin(atoms, cutoff)
        t_tnl = bench_torch_nl(atoms, cutoff)
        print(f"{n:>8} | {t_ase:>10.5f} | {t_vesin:>10.5f} | {t_tnl:>10.5f}")


if __name__ == "__main__":
    main()
