# Phase 0 — Domain Primer (neighbor lists for MD / MLIP)

Minimum-viable domain knowledge per project.md Phase 0, Task 1. Goal is fluency, not
expertise.

## Periodic boundary conditions (PBC)
- A simulation cell is defined by 3 lattice vectors (a 3x3 matrix). Atoms "wrap around"
  the cell edges — an atom leaving one face re-enters from the opposite face.
- For neighbor search, this means atom `i` near a cell boundary can be within cutoff of
  atom `j`'s *periodic image*, not just its literal stored position. The **minimum-image
  convention** picks, for each pair, the periodic image of `j` closest to `i`.
- Orthogonal cells (lattice vectors along x/y/z, diagonal matrix) make minimum-image
  arithmetic trivial (just wrap each coordinate independently). **Triclinic** cells
  (non-orthogonal lattice vectors — e.g. hexagonal, monoclinic crystals) require
  converting to fractional coordinates, wrapping, and converting back, and cell-list
  binning must account for the skewed shape when deciding cell size / which neighbor
  cells to check. This is the documented harder case some existing tools skimp on.

## Cell lists vs. Verlet lists
- **Cell list (spatial hashing / binning):** partition the box into a grid of cells with
  edge length >= cutoff radius. Bin every atom into its cell in O(N). For each atom,
  only atoms in the same or the 26 (3D) adjacent cells can possibly be within cutoff —
  so the pair search drops from O(N^2) to ~O(N). Must be rebuilt whenever atoms move
  enough to change cell membership; cheap to rebuild since it's just re-binning.
- **Verlet list (neighbor list with a skin buffer):** build the neighbor list using
  cutoff + skin distance (skin ~ a few tenths of an Angstrom) instead of the bare
  cutoff, so it stays valid for several timesteps even as atoms move — you only need to
  *rebuild* it once any atom has moved more than skin/2 since the last rebuild
  (a stricter bound would be needed if tracking pairwise displacement isn't exact, but
  half-skin per atom is the standard conservative bookkeeping). In between rebuilds you
  just reuse the same pair list and let force calculations filter by the true cutoff.
  This trades a slightly larger candidate list for far fewer expensive rebuilds — big win
  over many MD timesteps.
- The two techniques compose: cell lists are commonly used as the mechanism to *build*
  a Verlet list efficiently, rather than being alternatives.
- For FerroSim: cell lists are the Phase 1 primitive; the Verlet skin-buffer logic is a
  Phase 2 addition on top, specifically valuable for trajectory (many-timestep,
  same-system) workloads rather than one-shot batched training-data inference.

## ASE (Atomic Simulation Environment) representation
- `ase.Atoms` is the core object: holds `positions` (Nx3 array, Angstrom), `numbers`
  (atomic numbers), `cell` (3x3 lattice vectors, zero if non-periodic), and `pbc`
  (3-element bool, which axes are periodic).
- `atoms.get_positions()`, `atoms.get_cell()`, `atoms.pbc` are the fields FerroSim's
  bindings need to accept. ASE's own `ase.neighborlist.NeighborList` /
  `primitive_neighbor_list` is the "default backend" FerroSim aims to be a faster,
  drop-in replacement for (per project.md Phase 3, task 3-4).
- A `Calculator` in ASE is the interface that, given an `Atoms` object, returns
  energy/forces — this is where MLIPs (MACE, CHGNet) plug in, and where neighbor-list
  construction sits in the hot path (calculator.calculate() -> neighbor list -> model
  forward pass -> forces/energy).

## Where this leaves FerroSim
The algorithms (cell lists, Verlet lists) are textbook and not the hard part. The gap
this project targets is engineering: doing this in Rust, batched across many
independent small systems at once (not one large system), portable across CPU/GPU, with
bindings clean enough to be a drop-in for ASE/torch_nl/Vesin users.
