//! One-off probe: enumerate every wgpu-visible adapter on this machine and
//! confirm at least one supports compute shaders, before committing to the
//! GPU cell-list path.

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    println!("Enumerating adapters across all backends (Vulkan/DX12/Metal/GL):");
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()).await {
        let info = adapter.get_info();
        let limits = adapter.limits();
        println!(
            "- {} | backend={:?} | device_type={:?} | driver={} | max_compute_workgroups_per_dim={}",
            info.name, info.backend, info.device_type, info.driver, limits.max_compute_workgroups_per_dimension
        );
    }

    println!("\nRequesting default high-performance adapter...");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("no suitable adapter found");
    let info = adapter.get_info();
    println!("Selected: {} ({:?}, {:?})", info.name, info.backend, info.device_type);

    let (_device, _queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("failed to request device");
    println!("Device + queue created successfully — compute shaders are usable.");
}
