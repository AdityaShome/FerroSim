"""
Phase 2, Task 5: benchmark existing tools on a *batched* multi-system
workload - many independent, small FCC Cu cells processed together - which is
the actual MLIP training/inference workload shape (per Phase 0's competitive
research), as opposed to Phase 0/Phase 1's single large-system benchmarks.

ASE and Vesin have no native batching API, so they're benchmarked as a Python
loop over the batch (the natural way to use them for this workload). torch_nl
has a native `batch` tensor argument that processes the whole batch in one
call - its actual intended fast path - so it's benchmarked that way.

Usage:
    python scripts/bench_batch_competitors.py
"""
import time

import numpy as np
import torch
from ase.build import bulk
from ase.neighborlist import primitive_neighbor_list
from vesin import NeighborList as VesinNeighborList
from torch_nl import compute_neighborlist


def make_small_system(n_target_atoms: int):
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


def bench_ase_looped(systems, cutoff):
    def run():
        for atoms in systems:
            primitive_neighbor_list(
                "ijS", pbc=atoms.pbc, cell=atoms.cell.array,
                positions=atoms.positions, cutoff=cutoff,
            )
    return time_it(run)


def bench_vesin_looped(systems, cutoff):
    def run():
        nl = VesinNeighborList(cutoff=cutoff, full_list=True)
        for atoms in systems:
            nl.compute(
                points=atoms.positions, box=atoms.cell.array,
                periodic=bool(atoms.pbc.any()), quantities="ij",
            )
    return time_it(run)


def bench_torch_nl_batched(systems, cutoff):
    pos = torch.cat([torch.tensor(s.positions, dtype=torch.float64) for s in systems])
    cell = torch.stack([torch.tensor(s.cell.array, dtype=torch.float64) for s in systems])
    pbc = torch.stack([torch.tensor(s.pbc) for s in systems])
    batch = torch.cat([
        torch.full((len(s),), i, dtype=torch.long) for i, s in enumerate(systems)
    ])
    return time_it(
        lambda: compute_neighborlist(cutoff, pos, cell, pbc, batch, self_interaction=False)
    )


def main():
    cutoff = 6.0
    n_atoms_per_system = 32
    print(f"batch benchmark: n_atoms_per_system={n_atoms_per_system}, cutoff={cutoff}")
    print(f"{'batch_size':>10} | {'ase_s':>10} | {'vesin_s':>10} | {'torch_nl_s':>10}")
    print("-" * 52)
    for batch_size in (8, 32, 128):
        systems = [make_small_system(n_atoms_per_system) for _ in range(batch_size)]
        t_ase = bench_ase_looped(systems, cutoff)
        t_vesin = bench_vesin_looped(systems, cutoff)
        t_tnl = bench_torch_nl_batched(systems, cutoff)
        print(f"{batch_size:>10} | {t_ase:>10.5f} | {t_vesin:>10.5f} | {t_tnl:>10.5f}")


if __name__ == "__main__":
    main()
