# FerroSim Decision Log

Running record of what was tried, what worked, what didn't, and why.
Per project.md's standing instructions, this is maintained throughout all phases
and feeds the Phase 4 writeup and interview prep.

## Phase 0 — Domain ramp-up and feasibility check

### 2026-07-21 — Environment setup
- Toolchain confirmed available: Python 3.10.11, Rust (cargo/rustc 1.92.0).
- Created `.venv` for the Python side (ASE, MACE, vesin, torch_nl, py-spy) rather than
  polluting global site-packages — keeps Phase 0 profiling reproducible.
- Installed `ase` + `vesin` first (small, fast) before the heavier `mace-torch` (pulls
  in PyTorch) so early scaffolding work wasn't blocked on a large download.

### 2026-07-21 — Task 2: end-to-end MACE pipeline confirmed
- `mace_mp(model='small', device='cpu')` on an FCC Cu unit cell (4 atoms) runs
  end-to-end: energy -16.36 eV, forces shape (4,3). Model checkpoint (31 MB,
  MACE-MP-0 `small`) downloaded once and cached to `~/.cache/mace/`.
- Confirms the neighbor-list construction happens inside `MACECalculator.calculate()`,
  ahead of the model's forward pass, consistent with project.md's description of where
  in the pipeline it sits.

### 2026-07-21 — Task 3: own profiling numbers (`scripts/profile_bottleneck.py`)
Measured ASE's reference `primitive_neighbor_list` vs. MACE-MP-small forward pass
(CPU-only PyTorch, no CUDA available on this machine), FCC Cu supercells, 6 Å cutoff:

| n_atoms | nlist (s) | forward (s) | nlist % of total |
|--------:|----------:|------------:|------------------:|
|      32 |   0.01402 |     0.12726 |               9.9% |
|     500 |   0.06506 |     2.28948 |               2.8% |
|    5324 |   0.95008 |    32.91876 |               2.8% |

**Interpretation:** on CPU, neighbor-list construction is a small fraction (3-10%) of
wall-clock time — the NN forward pass dominates overwhelmingly. This does *not*
contradict the papers cited in project.md (TorchSim, TorchMD, etc.) — it clarifies
their claims are implicitly GPU-centric: GPUs accelerate the NN forward pass far more
than a CPU-bound neighbor-list construction step, which is exactly what shifts the
bottleneck toward graph/neighbor-list construction on GPU-accelerated pipelines. On
CPU-only hardware (this machine), the forward pass itself is the bottleneck, which
directly reinforces project.md's Fallback Angle 3 (CPU-only inference, SIMD-optimized,
per Jacobs et al. 2025) as a credible differentiation angle independent of whether the
GPU-batched angle is available — worth weighing seriously in the Phase 0 gate decision
below, alongside the primary GPU-batched angle assuming later GPU access.
- Caveat: only tested one architecture family (MACE-MP foundation model) and one
  crystal structure (FCC Cu supercells); should sanity-check with a molecular (non-
  periodic) system too before treating this as conclusive.

### 2026-07-21 — Task 4: competitor baseline (`scripts/bench_competitors.py`)
FCC Cu supercells, 6 Å cutoff, CPU, best-of-7 timing:

| n_atoms | ASE `primitive_neighbor_list` (s) | Vesin (s) | torch_nl (s) |
|--------:|-----------------------------------:|----------:|-------------:|
|      32 |                             0.01687 |   0.00042 |      0.00340 |
|     500 |                             0.06666 |   0.00080 |      0.02995 |
|    2048 |                             0.44023 |   0.00453 |      0.13943 |

**Interpretation:** Vesin (Rust core + Python bindings already) is dramatically faster
than both ASE's pure-Python reference and torch_nl at every tested size — roughly
40-100x faster than ASE, ~8-30x faster than torch_nl, and its lead widens with system
size. This is the real bar FerroSim's Phase 1 CPU implementation must clear or beat to
be worth building at all: Vesin already occupies "fast, Rust-backed, single-system"
territory. This sharpens why project.md's differentiation angle is specifically
**batched multi-system execution** and **portable GPU** — Vesin's docs/API should be
checked in Phase 1 for whether it already supports batching before finalizing that as
the differentiator (need to confirm in Task 5 / Phase 1 kickoff).
- API note: Vesin's `NeighborList.compute()` takes points/box/periodic directly (no
  ASE `Atoms` wrapping needed) and returns full `ij` pair quantities on request — a
  reasonable API shape to mirror for FerroSim's own Python bindings in Phase 3.
- torch_nl requires an explicit `batch` tensor mapping atoms to systems already —
  worth checking in Phase 1 whether it actually batches efficiently (parallel kernel)
  or just loops internally; if the latter, that's a second concrete gap to exploit
  beyond raw single-system speed.

### 2026-07-21 — Task 5: recent-work search (arXiv/GitHub, last ~1-3 months)
Searched for work that might already close the "batched, portable, Rust-based
neighbor lists for MLIPs" gap project.md identifies as the primary opportunity.

**Most significant finding — NVIDIA ALCHEMI Toolkit-Ops** (`NVIDIA/nvalchemi-toolkit-ops`,
v0.3.0, actively documented): a collection of GPU-first, **batched** PyTorch kernels for
computational chemistry/materials workflows, explicitly including neighbor-list
construction (both O(N) cell-list and O(N^2) variants), and benchmarked as
outperforming MACE/TensorNet's built-in graph construction on H100 GPUs. This is close
to a direct hit on the primary gap — batched GPU neighbor lists for MLIPs, from NVIDIA
itself, released since project.md was scoped.
- **However**, it is CUDA-only (NVIDIA GPUs specifically) and tightly coupled to
  PyTorch — no evidence of AMD/Metal/cross-vendor portability, and it's a Python/
  PyTorch op library, not a standalone Rust library usable outside a PyTorch process.
  This leaves FerroSim's `wgpu`-portability angle (Phase 3) and Rust-native,
  framework-independent embeddability genuinely undifferentiated by this release.

**Also found — TorchSim's neighbor-list module** now does automatic backend selection
across Alchemiops (NVIDIA), Vesin, torch_nl, and a pure-PyTorch fallback. This confirms
batching *orchestration* at the Python layer already exists; it does not mean the
underlying single-system kernels are natively batched (still worth checking directly
whether Vesin/torch_nl batch efficiently or are looped under TorchSim's dispatch,
before assuming batching itself remains fully open).

**Also found — "GPU-Native Compressed Neighbor Lists with a Space-Filling-Curve Data
Layout"** (Thaler & Keller, CSCS, arXiv 2602.19873, IPDPS 2026): GPU neighbor lists
(NVIDIA + AMD) with a compressed SFC layout, benchmarked against GROMACS. This targets
classical large-single-system HPC MD (astrophysics/cosmology mentioned explicitly as a
target use case), not the batched-many-small-systems MLIP training/inference workload
FerroSim targets — relevant prior art for GPU neighbor-list technique, but doesn't
close FerroSim's specific gap. Notably it does span NVIDIA+AMD, i.e. proof this class
of technique isn't inherently CUDA-locked, which supports feasibility of the wgpu
portability angle rather than undermining it.

**Net read on the Phase 0 gate:** the primary gap is *narrowed but not closed*. NVIDIA
has closed the "batched GPU neighbor lists exist at all for MLIPs" gap, but only within
a CUDA+PyTorch-locked toolkit. FerroSim's remaining differentiated angles, in order of
strength given all findings above:
1. **Portable GPU (wgpu: cross-vendor, non-PyTorch-locked)** — still open; ALCHEMI is
   CUDA-only, and the CSCS paper shows cross-vendor GPU neighbor lists are technically
   viable, just not yet done for this batched-MLIP use case.
2. **CPU-only, SIMD-optimized path** — still open, and our own Task 3 profiling data
   makes this more concretely motivated than project.md's citation of Jacobs et al.
   2025 alone: on CPU, forward-pass time dominates so heavily that a faster CPU
   neighbor list mostly matters for deployment contexts without GPU access at all,
   which is exactly the underserved case Jacobs et al. flag.
3. **Triclinic cell handling** — status unconfirmed for ALCHEMI Toolkit-Ops; needs a
   direct check in Phase 1 rather than assumed open or closed.
4. Rust-native, framework-independent embeddability (usable outside a PyTorch process
   entirely) — still a genuine gap; every competitor found (Vesin aside) is Python/
   PyTorch-first.

**Recommendation surfaced to user (2026-07-21):** proceed to Phase 1 as scoped in
project.md, but track ALCHEMI Toolkit-Ops explicitly as the closest competitor to
benchmark against once Phase 3 (GPU path) begins, and treat "portable, non-PyTorch-
locked" as the headline framing of FerroSim's GPU differentiation rather than "batched
GPU" alone (since NVIDIA has now demonstrated batched GPU neighbor lists are
achievable, if CUDA-locked).

## Phase 1 — Core algorithm: correct, single-system, CPU neighbor lists in Rust

### 2026-08-05 — Core data structures and design
- `cargo init --lib`; crate structure: `cell.rs` (lattice matrix + fractional/Cartesian
  conversion), `system.rs` (flat position array + cell + PBC flags), `neighbor_list.rs`
  (output type), `bruteforce.rs` (O(n²) reference), `celllist.rs` (the real algorithm).
- Positions are a flat `Vec<f64>` (length `3*n_atoms`), not `Vec<[f64;3]>`, per
  project.md's explicit performance guidance.
- `NeighborList` output matches ASE/torch_nl's `i, j, S` convention (parallel index
  arrays + integer periodic-shift vectors) for drop-in compatibility, and is always a
  *full* list (`(i,j,S)` and `(j,i,-S)` both present) since MLIP force/energy
  calculations need each atom's complete neighbor set.
- Triclinic cells are sized using **perpendicular width** (`V / face_area` per axis, not
  axis-aligned bounding box) — the standard technique for correctly sizing cell-list
  bins in skewed cells, since a naive bounding box either wastes bins or misses
  neighbors depending on skew direction.
- Brute-force reference implemented independently of the cell-list's fractional/binning
  machinery (works directly in raw Cartesian coordinates) specifically so the two
  algorithms are a genuine cross-check on each other, not two paths through the same bug.

### 2026-08-05 — Two real correctness bugs found and fixed via the brute-force cross-check
Writing `tests/correctness.rs` (property tests: random orthogonal/triclinic/mixed-PBC
configs, edge cases) immediately surfaced two bugs — exactly the kind of thing the
brute-force oracle exists to catch:

1. **Brute-force self-pair double-count.** For `i == j`, the reference's shift loop
   visits both `+shift` and `-shift` as separate iterations, but the code also manually
   pushed the mirror `(j,i,-shift)` on every match — double-counting every periodic
   self-image pair. Fixed by only pushing once when `i == j`.
2. **Cell-list shift-reconstruction bug (the real algorithmic bug).** The formula used
   to reconstruct the periodic shift `S` for a found pair was `S = bin_shift -
   floor(frac_j)`, missing a `+ floor(frac_i)` term. This is invisible whenever all
   input positions already lie inside `[0,1)` fractional coordinates (floor = 0 for
   everyone, e.g. any test that generates positions in `[0, cell_side)` for an
   axis-aligned cell) — which is why initial orthogonal/mixed-PBC tests passed while
   masking the bug. It only surfaces for atoms whose fractional coordinate falls outside
   `[0,1)` before wrapping (raw input spanning more than one periodic image), which the
   triclinic random test happened to generate. Root cause: `bin_shift` is computed
   relative to atom `i`'s *wrapped* (canonical, floor-removed) bin position, not its true
   fractional position, so the reconstructed shift needs correcting by both atoms'
   removed floors, not just atom `j`'s.
3. **Follow-on brute-force bug, same root cause.** After fixing (2), one triclinic
   random-trial case still disagreed: the brute-force reference's `n_shell` bound
   (`ceil(cutoff/perp)`) is only valid for positions pre-wrapped into `[0,1)` — it
   doesn't account for raw input atoms already starting more than one cell apart. Fixed
   by padding `n_shell` with the actual fractional-coordinate span of the input data per
   axis, keeping the reference dead-simple (just widen the search window) rather than
   adopting the cell-list's wrap-and-correct machinery.

**Lesson for the rest of the project:** test position generators must include cases
where fractional coordinates fall outside `[0,1)` (unwrapped/raw input) — this is not an
edge case in practice (MD trajectories routinely drift atoms outside the nominal cell
between wraps) and both implementations independently got it wrong until an explicit
`atoms_outside_fundamental_domain` regression test was added.

**Result:** all 12 correctness tests pass (`cargo test`), including 50-trial randomized
sweeps over orthogonal, triclinic, and mixed-PBC configurations, plus explicit edge
cases (single atom, exact-cutoff-boundary, small-cell-large-cutoff, sparse systems,
non-periodic clusters, unwrapped input). Phase 1's correctness gate is met.

### 2026-08-05 — Task 5: initial single-system benchmark vs Vesin/torch_nl/ASE
`examples/bench_fcc.rs` (`cargo run --release --example bench_fcc`) reconstructs the
identical FCC Cu supercells used in Phase 0's `scripts/bench_competitors.py` (cubic
`bulk("Cu","fcc",a=3.6)` repeated to hit the same atom counts, 6 Å cutoff, best-of-7
timing), so the numbers are directly comparable to the Phase 0 table above:

| n_atoms | ASE (s) | Vesin (s) | torch_nl (s) | **FerroSim (s)** |
|--------:|--------:|----------:|--------------:|------------------:|
|      32 | 0.01687 |   0.00042 |       0.00340 |        **0.00020** |
|     500 | 0.06666 |   0.00080 |       0.02995 |        **0.00262** |
|    2048 | 0.44023 |   0.00453 |       0.13943 |        **0.01364** |

**Interpretation:** FerroSim already beats ASE (~85-320x) and torch_nl (~10-17x) at
every size, unsurprising since neither is a tuned, allocation-lean, compiled cell-list
core (ASE's is pure Python; torch_nl carries tensor/batch overhead designed for
autodiff, not raw speed). Against Vesin specifically — the actual bar per Phase 0 —
FerroSim is *faster* at the smallest size (32 atoms: 2.1x) but *3.0-3.3x slower* at
500 and 2048 atoms. Plausible causes, to investigate in Phase 2 rather than now: (a)
`HashMap<(i32,i32,i32), Vec<usize>>` binning has hashing + allocation overhead per bin
that a flat/sorted array binning scheme (which Vesin's Rust core reportedly uses) would
avoid; (b) no parallelism yet (Phase 2's explicit `rayon` task); (c) recomputing
`cell.cartesian(shift)` per candidate pair inside the innermost loop instead of
precomputing the small set of possible shift vectors once. This is judged a "close, not
dramatic" gap per the Phase 1 gate language (large gaps = algorithmic issue; this is a
missing-optimization-scale gap, ~3x, not ASE's 40-100x or even torch_nl's double-digit
gap) — proceeding to Phase 2 as scoped, with the above three items as concrete first
optimization targets rather than a full redesign.

## Phase 2 — Performance optimization + batched multi-system execution

### 2026-08-05 — Task 1: `rayon`-parallelized single-system search
Refactored `celllist.rs` into `build_bin_grid` (sequential — O(n), binning atoms into
a `HashMap<(i32,i32,i32), Vec<usize>>`) and `neighbors_for_atom` (a pure function of
`(system, cutoff, grid, atom index)`, no shared mutable state), then parallelized the
per-atom search itself via `(0..n).into_par_iter().flat_map_iter(...)`. Binning stays
sequential — it's O(n) and cheap relative to the O(n·shell³) search, and a shared
mutable `HashMap` isn't worth parallelizing carefully for the gain.

Scaling on the 12-logical-core dev machine (`examples/bench_fcc.rs`, `RAYON_NUM_THREADS`
forced via env var, FCC Cu, 6 Å cutoff, release build, best-of-7):

| n_atoms | 1 thread (s) | 2 (s) | 4 (s) | 8 (s) | 12 (s) | speedup (1→12) |
|--------:|-------------:|------:|------:|------:|-------:|----------------:|
|      32 |      0.00042 | 0.00027 | 0.00023 | 0.00018 | 0.00016 |            2.6x |
|     500 |      0.00383 | 0.00293 | 0.00212 | 0.00175 | 0.00179 |            2.1x |
|    2048 |      0.01826 | 0.01122 | 0.00717 | 0.00617 | 0.00598 |            3.1x |

**Interpretation:** scaling is real but far from linear (~3x on 12 cores, not 12x) at
every tested size — this is expected and *not* treated as a bug per se, but is a
concrete area to revisit: likely causes are (a) `flat_map_iter` allocating a `Vec` per
atom then concatenating, rather than a lock-free/pre-sized shared output buffer, (b) the
shared `HashMap` bin lookups causing cache-line contention across threads even though
there's no write contention, (c) work granularity — 2048 atoms split across 12 rayon
tasks may not be enough work per task to amortize scheduling overhead. Not chasing this
further right now since Task 2 (batching) is the actual differentiator; batching should
also improve core utilization by giving rayon many independent systems' worth of work
to schedule instead of subdividing one system's atom loop.

Re-checked against the Phase 1 Vesin baseline at 12 threads: 32 atoms FerroSim is now
2.6x *faster* than Vesin (was 2.1x single-threaded), 500 atoms 2.2x slower (was 3.3x),
2048 atoms only 1.3x slower (was 3.0x) — parallelism alone closed most of the Phase 1
gap at the largest tested size.

### 2026-08-05 — Task 2: batched multi-system API (`src/batch.rs`)
`compute_neighbor_lists_batched(&[System], cutoff)` builds each system's bin grid
(parallelized across systems via `rayon`, cheap relative to search), then flattens
*every system's per-atom search* into one `rayon` parallel iteration spanning the whole
batch, rather than looping the single-system function per system. This matters
specifically because the target workload is many *small* systems (Phase 0's core
differentiation angle) — subdividing one 32-atom system's own atom loop across 12
threads (as the single-system path does) starves the thread pool with too little work
per task, but flattening 128 systems × 32 atoms = 4096 independent search tasks into one
pool gives rayon plenty of granular, independent work regardless of individual system
size. Verified against the single-system path (`tests/batch.rs`): batched output must
exactly equal per-system looped `compute_neighbor_list` output, including empty-batch
and zero-atom-system edge cases.

### 2026-08-05 — Task 3: incremental Verlet-list updates (`src/verlet.rs`)
`VerletList::new(system, cutoff, skin)` builds a candidate pair list once at an extended
cutoff (`cutoff + skin`); `VerletList::update(system)` recomputes distances only for
those existing candidates (cheap) and only rebuilds the candidate set (full cell-list
recompute at `cutoff + skin`) when some atom has moved more than `skin / 2` since the
last rebuild. Correctness rests on a standard triangle-inequality argument (documented
in the module): if every atom moves at most `skin / 2`, any pair's separation can change
by at most `skin`, so a pair excluded from the `cutoff + skin` candidate set cannot have
entered `cutoff`, and nothing within `cutoff` can be missing from candidates — this is
why the trigger is `skin / 2` per atom, not `skin`.

Tested (`tests/verlet.rs`) via: (a) a hand-worked exact case checking the rebuild flag
fires precisely when cumulative displacement crosses `skin/2` and not before; (b) a
40-step random-trajectory property test mixing small per-atom steps (expected: no
rebuild) with occasional large single-atom jumps (expected: rebuild), asserting the
incremental result exactly equals a fresh full `compute_neighbor_list` call at *every*
step regardless of which path was taken — this is the actual correctness invariant a
Verlet list exists to preserve, not just "doesn't crash"; (c) `skin = 0.0` degrading to
rebuild-every-call, confirming the trigger's boundary behavior.

### 2026-08-05 — Task 5: batched-workload benchmark vs Vesin/torch_nl
`scripts/bench_batch_competitors.py` / `examples/bench_batch_fcc.rs`: batches of 8/32/128
independent 32-atom FCC Cu cells, 6 Å cutoff. ASE and Vesin have no native batching API
so are benchmarked as a Python loop (their natural usage for this workload); torch_nl has
a native `batch` tensor argument processing the whole batch in one call, benchmarked that
way (its actual intended fast path):

| batch_size | ASE looped (s) | Vesin looped (s) | torch_nl batched (s) | **FerroSim batched (s)** |
|-----------:|----------------:|-------------------:|-----------------------:|----------------------------:|
|          8 |         0.14339 |             0.00219 |                 0.02034 |                  **0.00104** |
|         32 |         0.54510 |             0.00795 |                 0.07656 |                  **0.00317** |
|        128 |         1.80146 |             0.03024 |                 0.34305 |                  **0.01408** |

**Interpretation:** this is the headline result Phase 0 predicted. On the single-system
benchmark (Phase 1/Task 1 above), FerroSim trailed Vesin by up to 3x; on the *batched*
workload — the actual MLIP training/inference shape — FerroSim is now **2.1-2.5x faster
than Vesin at every batch size tested**, because Vesin's per-call Python-loop overhead
and lack of cross-system parallelism don't amortize the way FerroSim's single flattened
`rayon` dispatch across the whole batch does. Also notable: torch_nl's "batched" call
(its actual native batching path, not a loop) scales almost exactly linearly with batch
size (8→32→128 is a ~4x/~4.5x time increase for a 4x systems increase both times) with
no evidence of sub-linear scaling from batching — this confirms the Phase 0 suspicion
("worth checking whether torch_nl actually batches efficiently... or just loops
internally") that torch_nl's batch API does not meaningfully parallelize across systems
on CPU. **Phase 2's core differentiation claim is empirically validated**: FerroSim's
batched execution shows a clear efficiency advantage over both a naive per-system loop
(Vesin) and a competitor's own native batching path (torch_nl) — the Phase 2 gate
("batched execution must show a clear efficiency advantage over naively looping the
single-system implementation") is met.

<!-- Append further entries below as Phase 0 tasks proceed. -->
