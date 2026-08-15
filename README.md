# FerroSim

A fast, portable neighbor-list construction library for molecular dynamics (MD) and machine-learning interatomic potential (MLIP) simulation, written in Rust.

Neighbor-list construction figuring out which atoms are within a cutoff radius of each other sits directly in the hot path of every MD/MLIP simulation step. Several 2025 papers (TorchSim, TorchMD, InstaDeep's `mlip`) explicitly flag it as a bottleneck in current tooling, especially for the workload MLIP training/inference actually looks like: **many independent, small atomic systems processed in a batch**, not one giant single-system MD run. Existing tools (ASE, Vesin, torch_nl) are either pure Python, don't batch efficiently across systems, or are locked to a specific GPU vendor.

FerroSim isn't a new algorithm cell lists and Verlet lists are decades-old, textbook techniques. The value is in the engineering: a correct, batched, portable (CPU + cross-vendor GPU) Rust implementation that beats existing tools on the batched workload MLIP pipelines actually run.

## Status

Actively developed, following a phased build plan. Phases 0-3 are complete:

- **Phase 0** profiled the real bottleneck, benchmarked competitors, picked a differentiation angle. See [`docs/decision_log.md`](docs/decision_log.md).
- **Phase 1** correct single-system CPU neighbor lists (cell list + brute-force reference), orthogonal and triclinic cells, validated against the brute-force reference.
- **Phase 2** `rayon` parallelism, a batched multi-system API, and incremental Verlet lists for MD trajectories.
- **Phase 3** a `wgpu` GPU-accelerated batched path (cross-vendor: Vulkan/Metal/DX12, not CUDA-locked), validated against the CPU path.

Python bindings, ASE integration, and the real MACE speedup case study (Phase 3's remaining tasks) are in progress. Every phase's numbers, dead ends, and honest performance findings are recorded in [`docs/decision_log.md`](docs/decision_log.md) as they happen.

## Why

- **Batched execution.** MLIP training/inference processes thousands of small independent systems, not one big one. FerroSim flattens a whole batch's per-atom search into a single parallel dispatch (CPU: one `rayon` pool across every atom in the batch; GPU: one compute-shader dispatch across every atom in the batch) instead of looping per system this is where the actual speedup over competitors comes from, not raw single-system speed.
- **Portable GPU.** The GPU path uses [`wgpu`](https://wgpu.rs/), which targets Vulkan/Metal/DX12/WebGPU, instead of CUDA it runs on AMD/Intel/Apple GPUs, not just NVIDIA.
- **Triclinic cells handled correctly**, including atoms whose fractional coordinates fall outside `[0,1)` (unwrapped trajectory data) a case that silently breaks naive implementations (see the Phase 1 bugs in the decision log).
- **Rust-native**, usable without a Python/PyTorch process Python bindings (planned) will be a thin wrapper over this core, not the other way around.

## Quickstart

```rust
use ferrosim::{compute_neighbor_list, Cell, System};

// A cubic cell, 10 Å per side, periodic in all three directions.
let cell = Cell::new([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
let positions = vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2, 0.0]; // 3 atoms
let system = System::new(positions, cell, [true, true, true]);

let neighbors = compute_neighbor_list(&system, /* cutoff = */ 2.0);
for ((i, j), shift) in neighbors.i.iter().zip(&neighbors.j).zip(&neighbors.shift) {
    println!("{i} -> {j}, periodic shift {shift:?}");
}
```

Output is `(i, j, shift)` parallel arrays the same convention ASE and `torch_nl` use and is always a *full* list (`(i,j,S)` and `(j,i,-S)` both present), since force/energy calculations need each atom's complete neighbor set.

### Batched, many systems at once

```rust
use ferrosim::{compute_neighbor_lists_batched, System};

let systems: Vec<System> = /* ... build many small systems ... */ vec![];
let all_neighbors = compute_neighbor_lists_batched(&systems, 6.0);
// all_neighbors[k] corresponds to systems[k]
```

The same batch also runs on GPU with an identical signature:

```rust
use ferrosim::compute_neighbor_lists_batched_gpu;

let all_neighbors = compute_neighbor_lists_batched_gpu(&systems, 6.0);
```

### Trajectories: incremental Verlet lists

For simulating the *same* system over many timesteps, rebuilding the full neighbor list every step is wasteful. `VerletList` builds a candidate set with a "skin" buffer once, then cheaply re-checks distances each step, automatically rebuilding only when some atom has moved more than `skin / 2` since the last rebuild:

```rust
use ferrosim::VerletList;

let mut verlet = VerletList::new(&system, /* cutoff = */ 6.0, /* skin = */ 1.0);
for step_system in trajectory {
    let (neighbors, did_rebuild) = verlet.update(&step_system);
    // neighbors is always exactly equal to a fresh full rebuild would give
}
```

## Building and testing

```bash
cargo build --release
cargo test --release
```

The GPU tests (`tests/gpu.rs`) require a working Vulkan/Metal/DX12 adapter; run `cargo run --example gpu_probe` to check what `wgpu` detects on your machine.

## Benchmarks

```bash
cargo run --release --example bench_fcc              # single-system vs ASE/Vesin/torch_nl
cargo run --release --example bench_batch_fcc         # batched CPU vs Vesin/torch_nl
cargo run --release --example bench_gpu_batch_fcc      # batched CPU (rayon) vs GPU (wgpu)
```

Headline result so far (batched FCC copper supercells, 6 Å cutoff full methodology and every number in [`docs/decision_log.md`](docs/decision_log.md)): on the batched workload that matches real MLIP training/inference shape, FerroSim's CPU path is **2.1-2.5x faster than Vesin** and beats `torch_nl`'s own native batching path by a wide margin, because neither competitor parallelizes efficiently across systems the way a single flattened `rayon`/`wgpu` dispatch over the whole batch does.

## Project layout

```
src/
  cell.rs            simulation cell (lattice matrix, fractional/Cartesian conversion)
  system.rs           flat atom positions + cell + periodic flags
  neighbor_list.rs     output type (i, j, shift), ASE/torch_nl-compatible
  bruteforce.rs        O(n^2) reference implementation, used only for correctness checks
  celllist.rs          the real algorithm: cell-list binning + parallel search
  batch.rs             batched multi-system API
  verlet.rs            incremental Verlet-list updates for MD trajectories
  gpu.rs                wgpu-backed GPU batched path
  shaders/celllist.wgsl the GPU compute shader
tests/                 correctness tests (brute-force cross-checks, batch/GPU parity, Verlet invariants)
examples/               benchmarks against ASE/Vesin/torch_nl, and a GPU adapter probe
docs/decision_log.md    running log of what was tried, what worked, and why
```

## License

Not yet published; a license (MIT or Apache-2.0) will be added before the public release in Phase 4.
