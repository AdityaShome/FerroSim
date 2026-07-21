# Phase 0 Gate Summary

Per project.md, Phase 0's gate requires, before starting Phase 1:
(a) own profiled numbers, (b) a working competitor baseline, (c) a documented
differentiation-angle decision. Full detail and reasoning for each is in
`docs/decision_log.md`; this is the short version for the gate check itself.

## (a) Own profiled numbers ✅
CPU-only (no CUDA on this machine), MACE-MP-small, FCC Cu supercells, 6 Å cutoff:
neighbor-list construction is 3-10% of wall-clock time; NN forward pass dominates.
This reframes (doesn't contradict) the papers' GPU-centric bottleneck claims — see
decision log for the full interpretation.

## (b) Competitor baseline ✅
Benchmarked ASE's reference `primitive_neighbor_list`, Vesin, and torch_nl on
identical systems (CPU). Vesin is ~40-100x faster than ASE and ~8-30x faster than
torch_nl at the tested sizes — the real bar Phase 1's CPU implementation must clear.

## (c) Differentiation-angle decision ✅
Searched arXiv/GitHub for recent work (last 1-3 months). Found **NVIDIA ALCHEMI
Toolkit-Ops**, a CUDA+PyTorch-locked batched-GPU neighbor-list kernel set that
partially closes the "batched GPU neighbor lists for MLIPs" gap — but only within
NVIDIA/PyTorch, not as a portable or framework-independent library.

**Decision:** proceed to Phase 1 as scoped in project.md. Reframe the Phase 3 GPU
differentiation from "batched GPU" (partially claimed by NVIDIA now) to **"portable,
non-CUDA-locked, non-PyTorch-locked"** GPU batching via `wgpu`, plus the CPU-only
SIMD angle (now more concretely motivated by our own Task 3 data, not just cited
literature). Track ALCHEMI Toolkit-Ops as the benchmark to beat/differentiate against
once Phase 3 begins.

## Repo state at gate
- `.venv/` — Python env with ase, vesin, mace-torch, torch-nl, py-spy (gitignored).
- `scripts/profile_bottleneck.py` — Task 3 profiling script (re-runnable).
- `scripts/bench_competitors.py` — Task 4 competitor benchmark (re-runnable).
- `docs/phase0_domain_notes.md` — Task 1 domain primer.
- `docs/decision_log.md` — full running log with all data and reasoning.
- Rust crate scaffolded (`cargo init --lib`, name `ferrosim`) but empty — real Phase 1
  implementation work has not started yet.

**Status: gate passed. Ready to begin Phase 1** (core cell-list algorithm, single-
system, CPU, orthogonal + triclinic, in Rust).
