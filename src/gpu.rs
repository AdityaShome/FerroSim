//! GPU-accelerated batched neighbor-list construction via `wgpu` (Vulkan /
//! DX12 / Metal — portable across vendors, unlike CUDA-locked competitors).
//!
//! Binning (assigning atoms to cell-list bins) stays on the CPU/rayon path
//! (`celllist::build_bin_grid`) — it's O(n) and cheap. This module ports the
//! O(n * shell^3) per-atom search itself to a GPU compute shader
//! (`shaders/celllist.wgsl`), one thread per atom, across the *whole batch*
//! flattened together so even many small systems keep the GPU saturated.
//!
//! Numeric note: GPU buffers use `f32`, not the CPU path's `f64` — a
//! deliberate precision/throughput tradeoff typical of GPU compute; see
//! `docs/decision_log.md` for the correctness-tolerance implications.

use std::sync::OnceLock;

use wgpu::util::DeviceExt;

use crate::celllist::build_bin_grid;
use crate::neighbor_list::NeighborList;
use crate::system::System;

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

static GPU_STATE: OnceLock<GpuState> = OnceLock::new();

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn init_gpu_state() -> GpuState {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("FerroSim GPU path: no compute-capable wgpu adapter found");
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ferrosim_device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .expect("FerroSim GPU path: failed to create device/queue");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ferrosim_celllist"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/celllist.wgsl").into()),
        });

        let entries = [
            storage_entry(0, true),  // positions
            storage_entry(1, true),  // floor_coord
            storage_entry(2, true),  // atom_bin
            storage_entry(3, true),  // atom_system
            storage_entry(4, true),  // sys_ints
            storage_entry(5, true),  // sys_cell
            storage_entry(6, true),  // bin_start
            storage_entry(7, true),  // bin_atoms
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(9, false),  // out_count
            storage_entry(10, false), // out_pairs
        ];
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ferrosim_celllist_bgl"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ferrosim_celllist_pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ferrosim_celllist_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        GpuState { device, queue, pipeline, bind_group_layout }
    })
}

fn gpu_state() -> &'static GpuState {
    GPU_STATE.get_or_init(init_gpu_state)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    cutoff_sq: f32,
    n_atoms_total: u32,
    max_output_pairs: u32,
    _pad: u32,
}

/// Host-side flattened batch data, built once per call and reused across
/// overflow-retry attempts (only `max_output_pairs` / the output buffers
/// change between retries).
struct BatchBuffers {
    positions: Vec<f32>,
    floor_coord: Vec<i32>,
    atom_bin: Vec<i32>,
    atom_system: Vec<u32>,
    sys_ints: Vec<i32>,
    sys_cell: Vec<f32>,
    bin_start: Vec<u32>,
    bin_atoms: Vec<u32>,
    system_atom_offset: Vec<usize>,
    n_atoms_total: usize,
    cutoff: f64,
}

fn build_batch_buffers(systems: &[System], cutoff: f64) -> BatchBuffers {
    let grids: Vec<_> = systems.iter().map(|s| build_bin_grid(s, cutoff)).collect();

    let mut system_atom_offset = Vec::with_capacity(systems.len());
    let mut n_atoms_total = 0usize;
    for s in systems {
        system_atom_offset.push(n_atoms_total);
        n_atoms_total += s.n_atoms();
    }

    // Each system's bin_start segment holds `volume` start-offsets plus one
    // trailing sentinel (end of its last bin), so per-system `bin_offset`
    // into the *global* bin_start array is the cumulative sum of
    // `volume + 1` over all prior systems, not just `volume`.
    let volumes: Vec<usize> = grids.iter().map(|g| g.num_bins[0] * g.num_bins[1] * g.num_bins[2]).collect();
    let mut bin_offsets = Vec::with_capacity(systems.len());
    let mut cursor = 0usize;
    for &v in &volumes {
        bin_offsets.push(cursor);
        cursor += v + 1;
    }
    let bin_start_total_len = cursor;

    let mut positions = vec![0.0f32; 3 * n_atoms_total];
    let mut floor_coord = vec![0i32; 3 * n_atoms_total];
    let mut atom_bin = vec![0i32; 3 * n_atoms_total];
    let mut atom_system = vec![0u32; n_atoms_total];
    let mut sys_ints = vec![0i32; 10 * systems.len()];
    let mut sys_cell = vec![0f32; 9 * systems.len()];
    let mut bin_start = vec![0u32; bin_start_total_len];
    let mut bin_atoms = vec![0u32; n_atoms_total];

    for (si, (system, grid)) in systems.iter().zip(grids.iter()).enumerate() {
        let offset = system_atom_offset[si];
        for a in 0..system.n_atoms() {
            let g = offset + a;
            let p = system.position(a);
            positions[3 * g] = p[0] as f32;
            positions[3 * g + 1] = p[1] as f32;
            positions[3 * g + 2] = p[2] as f32;
            floor_coord[3 * g] = grid.floor_coord[a][0];
            floor_coord[3 * g + 1] = grid.floor_coord[a][1];
            floor_coord[3 * g + 2] = grid.floor_coord[a][2];
            atom_bin[3 * g] = grid.atom_bin[a][0];
            atom_bin[3 * g + 1] = grid.atom_bin[a][1];
            atom_bin[3 * g + 2] = grid.atom_bin[a][2];
            atom_system[g] = si as u32;
        }

        sys_ints[10 * si] = grid.num_bins[0] as i32;
        sys_ints[10 * si + 1] = grid.num_bins[1] as i32;
        sys_ints[10 * si + 2] = grid.num_bins[2] as i32;
        sys_ints[10 * si + 3] = grid.shell[0];
        sys_ints[10 * si + 4] = grid.shell[1];
        sys_ints[10 * si + 5] = grid.shell[2];
        sys_ints[10 * si + 6] = system.pbc[0] as i32;
        sys_ints[10 * si + 7] = system.pbc[1] as i32;
        sys_ints[10 * si + 8] = system.pbc[2] as i32;
        sys_ints[10 * si + 9] = bin_offsets[si] as i32;

        for r in 0..3 {
            for c in 0..3 {
                sys_cell[9 * si + 3 * r + c] = system.cell.matrix[r][c] as f32;
            }
        }

        // Emit this system's bins in the same row-major linearization the
        // shader uses (id = x + nx*(y + ny*z)), converting local atom
        // indices to global ones as they're written into bin_atoms, followed
        // by a trailing sentinel so the shader's `bin_start[global_bin + 1]`
        // lookup stays in range even at this system's very last bin.
        let mut atom_cursor = offset;
        let mut bin_cursor = bin_offsets[si];
        for iz in 0..grid.num_bins[2] as i32 {
            for iy in 0..grid.num_bins[1] as i32 {
                for ix in 0..grid.num_bins[0] as i32 {
                    bin_start[bin_cursor] = atom_cursor as u32;
                    bin_cursor += 1;
                    if let Some(local_atoms) = grid.bins.get(&(ix, iy, iz)) {
                        for &local_a in local_atoms {
                            bin_atoms[atom_cursor] = (offset + local_a) as u32;
                            atom_cursor += 1;
                        }
                    }
                }
            }
        }
        bin_start[bin_cursor] = atom_cursor as u32;
        debug_assert_eq!(atom_cursor, offset + system.n_atoms());
    }

    BatchBuffers {
        positions,
        floor_coord,
        atom_bin,
        atom_system,
        sys_ints,
        sys_cell,
        bin_start,
        bin_atoms,
        system_atom_offset,
        n_atoms_total,
        cutoff,
    }
}

fn dispatch(state: &GpuState, buffers: &BatchBuffers, max_output_pairs: u32) -> (u32, Vec<i32>) {
    let device = &state.device;
    let queue = &state.queue;

    let mk = |label: &str, data: &[u8], usage: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage })
    };

    let positions_buf = mk("positions", bytemuck::cast_slice(&buffers.positions), wgpu::BufferUsages::STORAGE);
    let floor_buf = mk("floor_coord", bytemuck::cast_slice(&buffers.floor_coord), wgpu::BufferUsages::STORAGE);
    let atom_bin_buf = mk("atom_bin", bytemuck::cast_slice(&buffers.atom_bin), wgpu::BufferUsages::STORAGE);
    let atom_sys_buf = mk("atom_system", bytemuck::cast_slice(&buffers.atom_system), wgpu::BufferUsages::STORAGE);
    let sys_ints_buf = mk("sys_ints", bytemuck::cast_slice(&buffers.sys_ints), wgpu::BufferUsages::STORAGE);
    let sys_cell_buf = mk("sys_cell", bytemuck::cast_slice(&buffers.sys_cell), wgpu::BufferUsages::STORAGE);
    let bin_start_buf = mk("bin_start", bytemuck::cast_slice(&buffers.bin_start), wgpu::BufferUsages::STORAGE);
    let bin_atoms_buf = mk("bin_atoms", bytemuck::cast_slice(&buffers.bin_atoms), wgpu::BufferUsages::STORAGE);

    let params = Params {
        cutoff_sq: (buffers.cutoff * buffers.cutoff) as f32,
        n_atoms_total: buffers.n_atoms_total as u32,
        max_output_pairs,
        _pad: 0,
    };
    let params_buf = mk("params", bytemuck::bytes_of(&params), wgpu::BufferUsages::UNIFORM);

    let out_count_buf = mk(
        "out_count",
        bytemuck::cast_slice(&[0u32]),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let out_pairs_len = (max_output_pairs as usize * 5).max(5);
    let out_pairs_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_pairs"),
        size: (out_pairs_len * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ferrosim_celllist_bg"),
        layout: &state.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: positions_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: floor_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: atom_bin_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: atom_sys_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: sys_ints_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: sys_cell_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bin_start_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bin_atoms_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: out_count_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: out_pairs_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&state.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (buffers.n_atoms_total as u32).div_ceil(64).max(1);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    // Two-phase readback: `out_pairs` is sized generously (to avoid frequent
    // overflow retries), but the actual valid prefix is usually far smaller
    // than `max_output_pairs`. Copying the *entire* buffer back regardless of
    // how many pairs were actually found was the dominant cost in this
    // function (a several-hundred-MB GPU->CPU transfer for buffers sized for
    // worst-case pair counts) — so read `out_count` first, then issue a
    // second, right-sized copy for only the valid `out_pairs` prefix.
    let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_count_staging"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&out_count_buf, 0, &count_staging, 0, 4);
    queue.submit(Some(encoder.finish()));

    let count: u32 = {
        let count_slice = count_staging.slice(..);
        count_slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("device poll failed");
        let data = count_slice.get_mapped_range().expect("failed to map out_count buffer");
        let v = bytemuck::cast_slice::<u8, u32>(&data)[0];
        drop(data);
        count_staging.unmap();
        v
    };

    let valid_pairs = (count.min(max_output_pairs) as usize) * 5;
    let valid_bytes = (valid_pairs * std::mem::size_of::<i32>()) as u64;
    let pairs: Vec<i32> = if valid_bytes == 0 {
        Vec::new()
    } else {
        let pairs_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_pairs_staging"),
            size: valid_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder2.copy_buffer_to_buffer(&out_pairs_buf, 0, &pairs_staging, 0, valid_bytes);
        queue.submit(Some(encoder2.finish()));

        let pairs_slice = pairs_staging.slice(..);
        pairs_slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("device poll failed");
        let data = pairs_slice.get_mapped_range().expect("failed to map out_pairs buffer");
        let v = bytemuck::cast_slice::<u8, i32>(&data).to_vec();
        drop(data);
        pairs_staging.unmap();
        v
    };

    (count, pairs)
}

/// GPU-accelerated equivalent of [`crate::compute_neighbor_lists_batched`]:
/// same batched-workload shape and the same output convention, computed via
/// a `wgpu` compute shader instead of `rayon`. Automatically retries with a
/// larger output buffer if the batch produces more pairs than the initial
/// (generously-sized) guess allows.
pub fn compute_neighbor_lists_batched_gpu(systems: &[System], cutoff: f64) -> Vec<NeighborList> {
    assert!(cutoff > 0.0, "cutoff must be positive");
    if systems.is_empty() {
        return Vec::new();
    }

    let buffers = build_batch_buffers(systems, cutoff);
    let state = gpu_state();

    let mut max_output_pairs = (buffers.n_atoms_total as u32 * 150).max(1024);
    let (count, pairs) = loop {
        let (count, pairs) = dispatch(state, &buffers, max_output_pairs);
        if count <= max_output_pairs {
            break (count, pairs);
        }
        assert!(
            max_output_pairs < 1 << 28,
            "FerroSim GPU path: output pair count ({count}) exceeds a sane retry ceiling; \
             something is likely wrong with the input rather than the buffer being merely undersized"
        );
        max_output_pairs *= 2;
    };

    let mut out: Vec<NeighborList> = (0..systems.len()).map(|_| NeighborList::default()).collect();
    for k in 0..count as usize {
        let base = k * 5;
        let i_global = pairs[base] as usize;
        let j_global = pairs[base + 1] as usize;
        let s = [pairs[base + 2], pairs[base + 3], pairs[base + 4]];
        let si = buffers.atom_system[i_global] as usize;
        debug_assert_eq!(si, buffers.atom_system[j_global] as usize);
        let i_local = i_global - buffers.system_atom_offset[si];
        let j_local = j_global - buffers.system_atom_offset[si];
        out[si].push(i_local as u32, j_local as u32, s);
    }
    out
}
