use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

const MAX_PARTICLES: usize = 128;

/// Particle colors matching the engine's particle_jibber palette.
/// Engine uses 0x80 alpha (semi-transparent).  We use brighter variants
/// so the effect is clearly visible.
const PARTICLE_COLORS: [[f32; 4]; 4] = [
    [1.0, 0.9, 0.2, 0.9],   // bright yellow
    [1.0, 0.4, 0.1, 0.9],   // orange
    [1.0, 1.0, 1.0, 0.8],   // white
    [0.6, 0.8, 1.0, 0.8],   // light blue
];

/// UV rect of the buildit jibber glow sprite in the STUFF/PARTICLE_PC.DDS
/// 256x256 atlas: the cross-flare glow at pixels (69,5)-(122,58) inclusive.
/// Stored as (u0, v0, u1, v1) in normalized [0,1] coordinates (v flipped, so
/// top row is v=0 in the DDS/GPU layout).
const GLOW_UV: (f32, f32, f32, f32) = (69.0 / 256.0, 5.0 / 256.0, 123.0 / 256.0, 59.0 / 256.0);

/// Particle growth lifecycle, matching the engine's GROW particle mode.
///
/// From the binary (FUN_0064eb00 / g_particle_template):
///   rate  _DAT_008102e4 = 0.5        (seconds to reach full size)
///   size  template +0x14 = 0.1       (full/target size, world units)
///   seed  template +0x8c = 0.0167    (initial size clock, ~1/60)
/// The sprite spawns at `seed` and ramps linearly to `size` over `rate`
/// seconds, then holds. Lifetime is ~7s, so particles outlive their grow.
const GROW_TIME: f32 = 0.5;
const BASE_SIZE: f32 = 0.1;
const SEED_SIZE: f32 = 0.1 / 30.0;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleVertex {
    pos: [f32; 3],
    quad: [f32; 2],
    color: [f32; 4],
    uv: [f32; 2],
}

#[derive(Clone)]
struct Particle {
    pos: Vec3,
    vel: Vec3,
    age: f32,
    lifetime: f32,
    color: [f32; 4],
    /// Animated size: grows from a small seed to `base_size` over `grow_time`
    /// (matches the engine's GROW particle mode), then holds until death.
    size: f32,
    /// Final/full size the particle ramps toward.
    base_size: f32,
    /// Seconds over which the sprite grows from seed to `base_size`.
    grow_time: f32,
    phase: f32,
    osc_x_freq: f32,
    osc_x_amp: f32,
    osc_y_freq: f32,
    osc_y_amp: f32,
}

pub struct BuilditParticles {
    particles: Vec<Particle>,
    next: usize,
    gravity: f32,
    spawn_rate: f32,
    spawn_accum: f32,
    vertex_buffer: wgpu::Buffer,
    max_vertices: u32,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleUniform {
    view_proj: [[f32; 4]; 4],
}

const PARTICLE_WGSL: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u_cam: CameraUniform;
@group(0) @binding(1) var u_particle_tex: texture_2d<f32>;
@group(0) @binding(2) var u_particle_sampler: sampler;

struct VertexInput {
    @location(0) a_pos: vec3<f32>,
    @location(1) a_quad: vec2<f32>,
    @location(2) a_color: vec4<f32>,
    @location(3) a_uv: vec2<f32>,
};
struct VertexOutput {
    @builtin(position) v_pos: vec4<f32>,
    @location(0) v_color: vec4<f32>,
    @location(1) v_uv: vec2<f32>,
};

@vertex
fn vs_particle(in: VertexInput) -> VertexOutput {
    let center = u_cam.view_proj * vec4<f32>(in.a_pos, 1.0);
    var out: VertexOutput;
    out.v_color = in.a_color;
    out.v_uv = in.a_uv;
    // Billboard in clip space. `center.w` is negative or ~0 (particle at/behind
    // the camera plane), scaling the quad by it sends the clip-space corner to
    // infinity and a single spawned particle flashes a full-screen polygon.
    // Collapse such particles to an off-screen point so they are clipped away.
    if center.w <= 0.001 {
        out.v_pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        return out;
    }
    // Constant world-size billboard: offset NDC by `size` after the divide,
    // which in clip space is `size * center.w`.
    let size = 0.12;
    out.v_pos = center + vec4<f32>(in.a_quad * size * center.w, 0.0, 0.0);
    return out;
}

@fragment
fn fs_particle(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the soft glow sprite from the particle atlas, tinted by the
    // particle color/alpha. The sprite is white->transparent, so sampling it
    // modulates the vertex color to give a soft-edged billboard.
    let sprite = textureSample(u_particle_tex, u_particle_sampler, in.v_uv);
    return vec4<f32>(in.v_color.rgb, in.v_color.a * sprite.a);
}
"#;

impl BuilditParticles {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_WGSL.into()),
        });

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle uniform"),
            contents: bytemuck::cast_slice(&[ParticleUniform {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Load the soft-particle sprite atlas (STUFF/PARTICLE_PC.DDS, 256x256
        // DXT5) and sample the buildit jibber glow cell from it.
        let (tex_view, sampler) = load_particle_texture(device, queue);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let max_vertices = (MAX_PARTICLES * 6) as u32;
        let vertex_size = std::mem::size_of::<ParticleVertex>();
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle vertex buffer"),
            size: (max_vertices as usize * vertex_size) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_particle"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: vertex_size as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 20,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 36,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_particle"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });

        Self {
            particles: Vec::new(),
            next: 0,
            gravity: -2.0,
            spawn_rate: 60.0,
            spawn_accum: 0.0,
            vertex_buffer,
            max_vertices,
            pipeline,
            bind_group,
            uniform_buf,
        }
    }

    pub fn spawn_at(
        &mut self,
        pos: Vec3,
        count: usize,
        phase: f32,
        osc_x_freq: f32,
        osc_x_amp: f32,
        osc_y_freq: f32,
        osc_y_amp: f32,
    ) {
        for _ in 0..count {
            if self.particles.len() >= MAX_PARTICLES {
                let p = &mut self.particles[self.next % MAX_PARTICLES];
                p.pos = pos;
                p.vel = Vec3::new(
                    (rand_f32() - 0.5) * 0.5,
                    rand_f32() * 1.5,
                    (rand_f32() - 0.5) * 0.5,
                );
                p.age = 0.0;
                p.lifetime = GROW_TIME + 0.4 + rand_f32() * 0.5;
                p.color = PARTICLE_COLORS[self.next % PARTICLE_COLORS.len()];
                p.size = SEED_SIZE;
                p.base_size = BASE_SIZE;
                p.grow_time = GROW_TIME;
                p.phase = phase + rand_f32() * std::f32::consts::TAU;
                p.osc_x_freq = osc_x_freq;
                p.osc_x_amp = osc_x_amp;
                p.osc_y_freq = osc_y_freq;
                p.osc_y_amp = osc_y_amp;
                self.next += 1;
            } else {
                let p = Particle {
                    pos,
                    vel: Vec3::new(
                        (rand_f32() - 0.5) * 0.5,
                        rand_f32() * 1.5,
                        (rand_f32() - 0.5) * 0.5,
                    ),
                    age: 0.0,
                    lifetime: GROW_TIME + 0.4 + rand_f32() * 0.5,
                    color: PARTICLE_COLORS[self.particles.len() % PARTICLE_COLORS.len()],
                    size: SEED_SIZE,
                    base_size: BASE_SIZE,
                    grow_time: GROW_TIME,
                    phase: phase + rand_f32() * std::f32::consts::TAU,
                    osc_x_freq,
                    osc_x_amp,
                    osc_y_freq,
                    osc_y_amp,
                };
                self.particles.push(p);
                self.next += 1;
            }
        }
    }

    pub fn update(&mut self, dt: f32, game_time: f32) {
        self.particles.retain_mut(|p| {
            p.age += dt;
            if p.age >= p.lifetime {
                return false;
            }
            // GROW mode: ramp size from a seed to the base over grow_time,
            // then hold at base (matches the engine's particle GROW mode).
            let t = (p.age / p.grow_time).clamp(0.0, 1.0);
            p.size = SEED_SIZE + (p.base_size - SEED_SIZE) * t;
            p.vel.y += self.gravity * dt;
            let t = game_time + p.phase;
            p.vel.x += (t * p.osc_x_freq).sin() * p.osc_x_amp * dt;
            p.vel.y += (t * p.osc_y_freq).sin() * p.osc_y_amp * dt;
            p.pos += p.vel * dt;
            true
        });
    }

    pub fn spawn_buildit_particles(
        &mut self,
        sub_obj_pos: Vec3,
        dt: f32,
        game_time: f32,
    ) {
        self.spawn_accum += self.spawn_rate * dt;
        let mut to_spawn = self.spawn_accum as usize;
        self.spawn_accum -= to_spawn as f32;
        if to_spawn > 8 {
            to_spawn = 8;
        }
        for _ in 0..to_spawn {
            let offset = Vec3::new(
                (rand_f32() - 0.5) * 0.2,
                (rand_f32() - 0.5) * 0.15,
                (rand_f32() - 0.5) * 0.2,
            );
            self.spawn_at(
                sub_obj_pos + offset,
                1,
                game_time * 0.5,
                2.0 + rand_f32(),
                0.3,
                3.0 + rand_f32(),
                0.2,
            );
        }
    }

    pub fn reset_spawner(&mut self) {
        self.spawn_accum = 0.0;
    }

    pub fn render(
        &mut self,
        rpass: &mut wgpu::RenderPass<'_>,
        queue: &wgpu::Queue,
        view_proj: &Mat4,
    ) {
        if self.particles.is_empty() {
            return;
        }
        let uniform = ParticleUniform {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[uniform]));

        let mut verts: Vec<ParticleVertex> = Vec::with_capacity(self.particles.len() * 6);
        let (u0, v0, u1, v1) = GLOW_UV;
        for p in &self.particles {
            // Alpha holds full while the sprite grows/plateaus, then fades out
            // over the last ~25% of its lifetime before disappearing.
            let life_frac = (p.age / p.lifetime).clamp(0.0, 1.0);
            let fade = if life_frac > 0.75 {
                1.0 - (life_frac - 0.75) / 0.25
            } else {
                1.0
            };
            let alpha = p.color[3] * fade;
            let c = [p.color[0], p.color[1], p.color[2], alpha];
            let s = p.size;
            let px = p.pos.x;
            let py = p.pos.y;
            let pz = p.pos.z;
            verts.push(ParticleVertex { pos: [px, py, pz], quad: [-s, -s], color: c, uv: [u0, v0] });
            verts.push(ParticleVertex { pos: [px, py, pz], quad: [s, -s], color: c, uv: [u1, v0] });
            verts.push(ParticleVertex { pos: [px, py, pz], quad: [s, s], color: c, uv: [u1, v1] });
            verts.push(ParticleVertex { pos: [px, py, pz], quad: [-s, -s], color: c, uv: [u0, v0] });
            verts.push(ParticleVertex { pos: [px, py, pz], quad: [s, s], color: c, uv: [u1, v1] });
            verts.push(ParticleVertex { pos: [px, py, pz], quad: [-s, s], color: c, uv: [u0, v1] });
        }

        let vertex_size = std::mem::size_of::<ParticleVertex>();
        let byte_count = verts.len() * vertex_size;
        if byte_count as u32 > self.max_vertices * vertex_size as u32 {
            return;
        }
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&verts));

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.draw(0..verts.len() as u32, 0..1);
    }

    pub fn is_spawning(&self) -> bool {
        self.spawn_accum > 0.0 || !self.particles.is_empty()
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
}

fn rand_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(0xDEAD_BEEF_CAFE_1234) };
    }
    SEED.with(|s| {
        let v = s.get().wrapping_mul(6364136223846793005).wrapping_add(1);
        s.set(v);
        // Take the top 16 bits of the 64-bit LCG and normalize to [0,1].
        ((v >> 48) as u16) as f32 / u16::MAX as f32
    })
}

/// Load STUFF/PARTICLE_PC.DDS (256x256 DXT5) and return (texture_view,
/// sampler). The buildit jibber glow sprite is a cell in this atlas. Falls
/// back to a 1x1 white texture + nearest sampler if the file is missing so
/// particle rendering still works (sprites become tinted quads).
fn load_particle_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("particle sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    // STUFF/PARTICLE_PC.DDS sits beside the map/level data under backup/.
    let path = std::env::current_dir()
        .map(|c| c.join("backup").join("STUFF").join("PARTICLE_PC.DDS"))
        .unwrap_or_default();
    let fallback_tex = || {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("particle fallback texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    };

    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("particle texture not found at {}; using fallback", path.display());
        return (fallback_tex(), sampler);
    };
    // DDS header is 128 bytes; DXT5 payload is w*h bytes (256x256 -> 65536).
    if bytes.len() < 128 + 4 {
        eprintln!("particle texture {} malformed", path.display());
        return (fallback_tex(), sampler);
    }
    let w = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let fourcc = u32::from_le_bytes(bytes[84..88].try_into().unwrap());
    let fmt = if fourcc == 0x35545844 {
        rustt::ghg::TextureFmt::Dxt5
    } else if fourcc == 0x31545844 {
        rustt::ghg::TextureFmt::Dxt1
    } else {
        eprintln!("particle texture {} unsupported fourcc", path.display());
        return (fallback_tex(), sampler);
    };
    let Ok(rgba) = rustt::dxt::decode_rgba(w, h, fmt, &bytes[128..]) else {
        eprintln!("particle texture {} failed to decode", path.display());
        return (fallback_tex(), sampler);
    };

    let size = wgpu::Extent3d {
        width: w as u32,
        height: h as u32,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particle texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((w * 4) as u32),
            rows_per_image: Some(h as u32),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (view, sampler)
}
