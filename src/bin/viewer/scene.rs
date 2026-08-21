use std::sync::Arc;

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use bytemuck::Zeroable;
use rustt::dxt;
use rustt::ghg::{Parsed, TextureFmt};
use rustt::glb::MeshData;
use rustt::map::Map;
use rustt::mapmesh;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    nrm: [f32; 3],
    uv: [f32; 2],
    /// 4 skin weights (u8, 255 = full influence), matching the raw part skin
    /// block. Skinning runs on the GPU in the vertex shader.
    weights: [u8; 4],
    /// 4 skin bone indices. Already remapped from the part's local skin-bone
    /// list to GLOBAL bone ids so every mesh can share one bone-matrix buffer.
    bones: [u8; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub base_color: [f32; 4],
    pub has_tex: u32,
    _pad: [u32; 3],
}

pub const MAX_MORPH_SLOTS: usize = 64;

/// Per-mesh morph uniform, mirroring `MorphUniform` in shaders.wgsl. The
/// `weights` array holds the current BSA blend-shape weights (one entry per
/// slot); the rest is per-part metadata written once at load.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MorphUniform {
    pub weights: [f32; MAX_MORPH_SLOTS],
    pub num_v: u32,
    pub slot_count: u32,
    pub delta_base: u32,
    pub enabled: u32,
    pub _pad: [u32; 4],
}

pub struct Bounds {
    pub center: Vec3,
    pub radius: f32,
}

pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    /// World-space center of this mesh's geometry. Used for radius-based
    /// culling so the game can skip meshes far from the player (the hub's
    /// other chambers) instead of drawing the whole map.
    pub bounds: Bounds,

    pub material: usize,
    pub transparent: bool,
    /// Index of the source ghg part, for looking up shape-key (dynamic
    /// buffer) data. Render item i maps to part i but empty parts can be
    /// skipped during mesh building, so this is tracked explicitly.
    pub part: usize,
}

pub struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub tex_id: i16,
    pub diffuse: [f32; 4],
}

pub struct TexInfo {
    pub w: u32,
    pub h: u32,
    pub fmt: &'static str,
    pub texture: Arc<wgpu::Texture>,
    pub view: Arc<wgpu::TextureView>,
}

pub struct GpuScene {
    pub meshes: Vec<GpuMesh>,
    pub materials: Vec<GpuMaterial>,
    pub textures: Vec<TexInfo>,
    pub preview_ids: Vec<Option<imgui::TextureId>>,
    pub bounds: Bounds,
    pub apply_bones: bool,
    pub material_layout: wgpu::BindGroupLayout,
    /// Rest (bind) world matrices per bone, used to compute per-frame skin
    /// matrices `animated_world[i] * rest_world[i]^-1`.
    rest_worlds: Vec<Mat4>,
    /// Shared storage buffer of skin matrices (one mat4 per global bone),
    /// written once per frame by `set_skin_mats`.
    skin_bone_buffer: wgpu::Buffer,
    skin_layout: wgpu::BindGroupLayout,
    skin_bind_group: wgpu::BindGroup,
    /// Shape-key (blend-shape) morphing, applied on the GPU per frame. A
    /// shared storage buffer holds every morphing part's slot deltas; each
    /// morphing mesh additionally gets its own uniform buffer (current BSA
    /// weights + part metadata) and bind group. Non-morphing meshes bind the
    /// shared default group (enabled == 0, no-op).
    morph_layout: wgpu::BindGroupLayout,
    default_morph_bind_group: wgpu::BindGroup,
    /// Parallel to `meshes`: per-mesh morph bind group + uniform buffer.
    morph_groups: Vec<Option<wgpu::BindGroup>>,
    morph_uniforms: Vec<Option<(wgpu::Buffer, MorphUniform)>>,
}

impl GpuScene {
    pub fn create_material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        parsed: &Parsed,
        is_srgb: bool,
        allowed_layers: &[u32],
    ) -> Self {
        let material_layout = Self::create_material_layout(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("model sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let white_view = make_white_texture(device, is_srgb);

        let tex_srcs: Vec<TexSrc> = parsed
            .textures
            .iter()
            .map(|t| TexSrc {
                w: t.w,
                h: t.h,
                fmt: t.fmt,
                payload: &t.payload,
            })
            .collect();
        let textures = build_textures(device, queue, &tex_srcs, is_srgb);
        let raw = rustt::glb::build_meshes(parsed);
        let (meshes, bounds) = build_static_meshes(device, parsed, &raw, allowed_layers);
        let mat_srcs: Vec<(i16, [f32; 4])> = parsed
            .materials
            .iter()
            .map(|m| (m.tex_id, [m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0]))
            .collect();
        let materials = build_materials(
            device,
            &material_layout,
            &white_view,
            &sampler,
            &textures,
            &mat_srcs,
        );

        let rest_worlds: Vec<Mat4> = parsed.bones.iter().map(|b| b.world).collect();
        let bone_count = rest_worlds.len().max(1);
        let skin_layout = Self::create_skin_layout(device);
        let skin_bone_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin bone matrices"),
            size: (bone_count as u64) * std::mem::size_of::<Mat4>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skin_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin bind group"),
            layout: &skin_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: skin_bone_buffer.as_entire_binding(),
            }],
        });
        // Start with identity skin matrices (bind pose).
        let mats: Vec<[f32; 16]> = (0..bone_count).map(|_| Mat4::IDENTITY.to_cols_array()).collect();
        queue.write_buffer(&skin_bone_buffer, 0, bytemuck::cast_slice(&mats));

        // ---- shape-key (blend-shape) morph data ----
        let (morph_layout, default_morph_bind_group) = build_default_morph(device, queue);
        // One shared storage buffer: for each part with dynamic buffers, its
        // slots laid out slot-major (`delta_base` in vec4 elements). The shader
        // reads this as `array<vec3<f32>>`, whose storage stride is 16 bytes
        // (vec3 aligns to 16), so every delta is padded to vec4 here. Empty
        // slots become zeros so the shader can index by `slot * num_v + vid`.
        let mut delta_data: Vec<[f32; 4]> = Vec::new();
        let mut part_delta_base: Vec<usize> = Vec::with_capacity(parsed.parts.len());
        for part in &parsed.parts {
            if part.dynamic_buffers.is_empty() {
                part_delta_base.push(0);
                continue;
            }
            part_delta_base.push(delta_data.len());
            let nv = part.num_v;
            for slot in &part.dynamic_buffers {
                match slot {
                    Some(buf) => {
                        for d in buf {
                            delta_data.push([d[0], d[1], d[2], 0.0]);
                        }
                    }
                    None => delta_data.extend(std::iter::repeat([0.0; 4]).take(nv)),
                }
            }
        }
        let morph_delta_storage = if delta_data.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("morph delta storage"),
                // One 16-byte-aligned vec3 element; enough for the layout's
                // minimum binding size so non-morphing models bind cleanly.
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("morph delta storage"),
                contents: bytemuck::cast_slice(&delta_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        // Per-mesh morph groups for the parts that actually have shape keys.
        let mut morph_groups = vec![None; meshes.len()];
        let mut morph_uniforms: Vec<Option<(wgpu::Buffer, MorphUniform)>> =
            vec![None; meshes.len()];
        for (i, mesh) in meshes.iter().enumerate() {
            let Some(part) = parsed.parts.get(mesh.part) else {
                continue;
            };
            if part.dynamic_buffers.is_empty() {
                continue;
            }
            let meta = MorphUniform {
                weights: [0.0; MAX_MORPH_SLOTS],
                num_v: part.num_v as u32,
                slot_count: part.dynamic_buffers.len() as u32,
                delta_base: part_delta_base[mesh.part] as u32,
                enabled: 1,
                _pad: [0; 4],
            };
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("morph uniform"),
                contents: bytemuck::bytes_of(&meta),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("morph bind group"),
                layout: &morph_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: morph_delta_storage.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: mesh.index_buffer.as_entire_binding(),
                    },
                ],
            });
            morph_groups[i] = Some(bind_group);
            morph_uniforms[i] = Some((uniform_buffer, meta));
        }

        Self {
            meshes,
            materials,
            textures,
            preview_ids: Vec::new(),
            bounds,
            apply_bones: true,
            material_layout,
            rest_worlds,
            skin_bone_buffer,
            skin_layout,
            skin_bind_group,
            morph_layout,
            default_morph_bind_group,
            morph_groups,
            morph_uniforms,
        }
    }

    /// Build a scene from a parsed map file (`.GSC`). No bones and no shape
    /// keys: every mesh gets the shared default morph group and an identity
    /// skin matrix, and meshes are static geometry built from the map's
    /// triangle strips.
    pub fn from_map(device: &wgpu::Device, queue: &wgpu::Queue, map: &Map, is_srgb: bool) -> Self {
        let material_layout = Self::create_material_layout(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("map sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let white_view = make_white_texture(device, is_srgb);

        // Map textures are single base-level DXT1/DXT5 payloads; the decoder
        // reads the base level and ignores any extra mip levels after it.
        let mut tex_slot_to_index = vec![usize::MAX; map.textures.len()];
        let mut tex_srcs: Vec<TexSrc> = Vec::new();
        for (slot, t) in map.textures.iter().enumerate() {
            tex_slot_to_index[slot] = tex_srcs.len();
            tex_srcs.push(TexSrc {
                w: t.w,
                h: t.h,
                fmt: t.fmt,
                payload: &t.payload,
            });
        }
        let textures = build_textures(device, queue, &tex_srcs, is_srgb);

        // One material per map material; the raw texture id is remapped to
        // the index of the decoded texture (or -1 for white fallback).
        let mat_srcs: Vec<(i16, [f32; 4])> = map
            .materials
            .iter()
            .map(|m| {
                let tex_id = map
                    .tex_slot(m.tex_id)
                    .filter(|&s| s < tex_slot_to_index.len() && tex_slot_to_index[s] != usize::MAX)
                    .map(|s| tex_slot_to_index[s] as i16)
                    .unwrap_or(-1);
                (tex_id, [m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0])
            })
            .collect();
        let materials = build_materials(
            device,
            &material_layout,
            &white_view,
            &sampler,
            &textures,
            &mat_srcs,
        );

        let (meshes, bounds) = build_map_meshes(device, map);
        println!(
            "map scene: {} meshes, {} triangles",
            meshes.len(),
            meshes.iter().map(|m| m.index_count as usize / 3).sum::<usize>()
        );

        // No bones: one identity rest world so the shared skin buffer stays
        // well-formed. apply_bones stays off, so set_skin_mats writes
        // identity.
        let rest_worlds = vec![Mat4::IDENTITY];
        let bone_count = 1;
        let skin_layout = Self::create_skin_layout(device);
        let skin_bone_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin bone matrices"),
            size: (bone_count as u64) * std::mem::size_of::<Mat4>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skin_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin bind group"),
            layout: &skin_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: skin_bone_buffer.as_entire_binding(),
            }],
        });
        let mats: Vec<[f32; 16]> = vec![Mat4::IDENTITY.to_cols_array()];
        queue.write_buffer(&skin_bone_buffer, 0, bytemuck::cast_slice(&mats));

        let (morph_layout, default_morph_bind_group) = build_default_morph(device, queue);
        let morph_groups = vec![None; meshes.len()];
        let morph_uniforms: Vec<Option<(wgpu::Buffer, MorphUniform)>> = vec![None; meshes.len()];

        Self {
            meshes,
            materials,
            textures,
            preview_ids: Vec::new(),
            bounds,
            apply_bones: false,
            material_layout,
            rest_worlds,
            skin_bone_buffer,
            skin_layout,
            skin_bind_group,
            morph_layout,
            default_morph_bind_group,
            morph_groups,
            morph_uniforms,
        }
    }

    pub fn create_skin_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub fn create_morph_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("morph layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn skin_bind_group(&self) -> &wgpu::BindGroup {
        &self.skin_bind_group
    }

    pub fn skin_layout(&self) -> &wgpu::BindGroupLayout {
        &self.skin_layout
    }

    pub fn morph_layout(&self) -> &wgpu::BindGroupLayout {
        &self.morph_layout
    }

    pub fn morph_bind_group(&self, i: usize) -> &wgpu::BindGroup {
        self.morph_groups
            .get(i)
            .and_then(|o| o.as_ref())
            .unwrap_or(&self.default_morph_bind_group)
    }

    /// Upload the current BSA blend-shape weights to every morphing mesh's
    /// uniform. Slot `s` takes `weights[s]` (0 when out of range), matching
    /// the game's channel-index-to-slot mapping. A ~1 KB total write per
    /// frame, far cheaper than rebuilding vertex buffers.
    pub fn set_morph_weights(&mut self, queue: &wgpu::Queue, weights: &[f32]) {
        for (buffer, meta) in self.morph_uniforms.iter().filter_map(|e| e.as_ref()) {
            let mut m = *meta;
            for s in 0..MAX_MORPH_SLOTS {
                m.weights[s] = weights.get(s).copied().unwrap_or(0.0);
            }
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&m));
        }
    }

    /// Toggle animation posing. The bind pose is the identity skin matrix, so
    /// this only flips a flag: the per-frame `set_skin_mats` write picks it up.
    pub fn set_apply_bones(&mut self, _device: &wgpu::Device, _parsed: &Parsed, v: bool) {
        self.apply_bones = v;
    }

    /// Upload per-frame skin matrices for all bones: `worlds[i] * rest_worlds[i]^-1`
    /// (identity when `apply_bones` is off). This is the only per-frame GPU work
    /// for animation — a ~2 KB buffer write, far cheaper than CPU skinning.
    pub fn set_skin_mats(&mut self, queue: &wgpu::Queue, worlds: &[Mat4]) {
        let mats: Vec<[f32; 16]> = self
            .rest_worlds
            .iter()
            .enumerate()
            .map(|(i, rest)| {
                if self.apply_bones {
                    if let Some(w) = worlds.get(i) {
                        return (*w * rest.inverse()).to_cols_array();
                    }
                }
                Mat4::IDENTITY.to_cols_array()
            })
            .collect();
        queue.write_buffer(&self.skin_bone_buffer, 0, bytemuck::cast_slice(&mats));
    }

    /// Write a new base color into a material's uniform buffer.
    pub fn set_material_color(&mut self, queue: &wgpu::Queue, i: usize, color: [f32; 4]) {
        let Some(mat) = self.materials.get_mut(i) else {
            return;
        };
        mat.diffuse = color;
        let uniform = MaterialUniform {
            base_color: color,
            has_tex: (mat.tex_id >= 0) as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&mat.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Register the decoded ghg textures with the imgui renderer for in-window previews.
    pub fn register_preview_textures(
        &mut self,
        device: &wgpu::Device,
        renderer: &mut imgui_wgpu::Renderer,
    ) {
        self.preview_ids.clear();
        for t in &self.textures {
            let size = wgpu::Extent3d {
                width: t.w,
                height: t.h,
                depth_or_array_layers: 1,
            };
            let imgui_tex = imgui_wgpu::Texture::from_raw_parts(
                device,
                renderer,
                t.texture.clone(),
                t.view.clone(),
                None,
                Some(&imgui_wgpu::RawTextureConfig {
                    label: Some("ghg preview"),
                    sampler_desc: wgpu::SamplerDescriptor {
                        mag_filter: wgpu::FilterMode::Linear,
                        min_filter: wgpu::FilterMode::Linear,
                        ..Default::default()
                    },
                }),
                size,
            );
            self.preview_ids
                .push(Some(renderer.textures.insert(imgui_tex)));
        }
    }
}

fn make_white_texture(device: &wgpu::Device, is_srgb: bool) -> Arc<wgpu::TextureView> {
    let format = if is_srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("white texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    Arc::new(view)
}

/// Texture source data shared by the ghg and map loaders. The payload is
/// still compressed; decoding happens inside `build_textures`.
struct TexSrc<'a> {
    w: usize,
    h: usize,
    fmt: TextureFmt,
    payload: &'a [u8],
}

/// Shared no-op morph bind group + layout for meshes without shape keys.
/// The storage buffers are minimal; the dummy uniform has `enabled == 0` so
/// the shader never reads them.
fn build_default_morph(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let morph_layout = GpuScene::create_morph_layout(device);
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("morph default storage"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dummy_uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("morph default uniform"),
        size: std::mem::size_of::<MorphUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &dummy_uniform,
        0,
        bytemuck::bytes_of(&MorphUniform::zeroed()),
    );
    let dummy_small = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("morph default storage"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("morph default bind group"),
        layout: &morph_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dummy_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dummy_small.as_entire_binding(),
            },
        ],
    });
    (morph_layout, group)
}

fn build_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    srcs: &[TexSrc],
    is_srgb: bool,
) -> Vec<TexInfo> {
    let format = if is_srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let mut out = Vec::with_capacity(srcs.len());
    for (i, t) in srcs.iter().enumerate() {
        let Ok(rgba) = dxt::decode_rgba(t.w, t.h, t.fmt, t.payload) else {
            eprintln!("  warning: texture {i} failed to decode; skipping");
            continue;
        };
        let w = t.w as u32;
        let h = t.h as u32;
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            size,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        out.push(TexInfo {
            w,
            h,
            fmt: match t.fmt {
                TextureFmt::Dxt1 => "DXT1",
                TextureFmt::Dxt5 => "DXT5",
            },
            texture: Arc::new(tex),
            view: Arc::new(view),
        });
    }
    out
}

fn build_static_meshes(
    device: &wgpu::Device,
    parsed: &Parsed,
    raw: &[MeshData],
    allowed_layers: &[u32],
) -> (Vec<GpuMesh>, Bounds) {
    let mut out = Vec::with_capacity(parsed.render.len());
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any = false;
    for (i, item) in parsed.render.iter().enumerate() {
        // Layers are LOD/quality variants (selected per-game via the sibling
        // `.TXT`). An empty `allowed_layers` means "no filtering" (render all).
        let layer = parsed.render_layer.get(i).copied().unwrap_or(u32::MAX);
        if !allowed_layers.is_empty() && !allowed_layers.contains(&layer) {
            continue;
        }
        let Some(md) = raw.get(i) else {
            continue;
        };
        if md.pos.is_empty() || md.idx.is_empty() {
            continue;
        }
        let mut md = md.clone();

        let material = mesh_material(parsed, i);
        // Meshes whose material samples a DXT5 texture carry a real alpha
        // channel (face decals, capes, cloth) and must be blended over the
        // opaque geometry, so they are drawn in a separate later pass.
        let transparent = parsed
            .materials
            .get(material)
            .and_then(|m| {
                if m.tex_id >= 0 {
                    parsed.textures.get(m.tex_id as usize)
                } else {
                    None
                }
            })
            .map(|t| t.fmt == TextureFmt::Dxt5)
            .unwrap_or(false);

        // Face decals sit right on the head surface and z-fight with it.
        // A depth-buffer bias is not enough: in float32 depth its effect
        // scales with camera distance, so it can't win the fight at every
        // zoom without visibly hovering when zoomed out. Instead, lift the
        // transparent mesh a hair off the surface along the direction from
        // its own bounding center (view-independent, zoom-independent).
        // Computed once in bind space; the vertex shader's skin matrix then
        // transforms the lifted position along with the rest of the mesh.
        if transparent && !md.pos.is_empty() {
            let mut cmin = [f32::INFINITY; 3];
            let mut cmax = [f32::NEG_INFINITY; 3];
            for p in &md.pos {
                for k in 0..3 {
                    cmin[k] = cmin[k].min(p[k]);
                    cmax[k] = cmax[k].max(p[k]);
                }
            }
            let center = Vec3::new(
                (cmin[0] + cmax[0]) * 0.5,
                (cmin[1] + cmax[1]) * 0.5,
                (cmin[2] + cmax[2]) * 0.5,
            );
            for p in md.pos.iter_mut() {
                let d = Vec3::from(*p) - center;
                let len = d.length();
                if len > 1e-6 {
                    *p = (Vec3::from(*p) + (d / len) * 0.002).into();
                }
            }
        }

        // Per-vertex skin data: 4 weights (u8) + 4 bone indices (u8). The part
        // stores LOCAL skin-bone indices, remapped here to GLOBAL bone ids so
        // the shared bone-matrix buffer can be indexed directly. Rigid parts
        // get a single full-weight influence; unskinned parts use bone 0.
        let skinned =
            !md.skin.is_empty() && !md.skin_bones.is_empty() && md.skin.len() >= md.pos.len() * 8;
        let rigid_bone = if skinned {
            None
        } else {
            (item.bone >= 0).then_some(item.bone as u8)
        };
        // Rigid parts store their vertices in the bone's LOCAL rest frame, not
        // model space, so the animated pose is just `anim_world[bone]`. The
        // shader's skin matrix is `anim_world * rest^-1`, so baking the bone's
        // rest world into these vertices here makes the shader evaluate to
        // `anim_world` again. Skinned parts already sit in model space and are
        // left untouched (the shared `world * rest^-1` matrix fits them).
        let rigid_rest = if skinned {
            None
        } else {
            (item.bone >= 0)
                .then(|| item.bone as usize)
                .and_then(|b| parsed.bones.get(b))
                .map(|b| b.world)
        };

        let mut vdata: Vec<Vertex> = Vec::with_capacity(md.pos.len());
        for v in 0..md.pos.len() {
            let mut p = md.pos[v];
            let mut n = md.nrm[v];
            if let Some(rest) = rigid_rest {
                p = rest.transform_point3(Vec3::from(p)).into();
                n = rest.transform_vector3(Vec3::from(n)).into();
            }
            let (weights, bones) = if skinned {
                let sw = &md.skin[v * 8..v * 8 + 8];
                let mut w = [0u8; 4];
                let mut b = [0u8; 4];
                let mut any = false;
                for k in 0..4 {
                    let li = sw[4 + k] as usize;
                    match md.skin_bones.get(li) {
                        // Local index out of range or global id that doesn't
                        // fit a u8 contributes nothing (mirrors the old CPU
                        // skinning path, which skipped such influences).
                        Some(&g) if g < 256 => {
                            w[k] = sw[k];
                            b[k] = g as u8;
                            if sw[k] != 0 {
                                any = true;
                            }
                        }
                        _ => {
                            w[k] = 0;
                            b[k] = 0;
                        }
                    }
                }
                // A vertex with no valid influences keeps its bind position
                // (full weight on the root bone instead of a zero matrix,
                // which would collapse it to the origin in the shader).
                if !any {
                    w = [255, 0, 0, 0];
                    b = [0, 0, 0, 0];
                }
                (w, b)
            } else {
                match rigid_bone {
                    Some(bone) => ([255, 0, 0, 0], [bone, 0, 0, 0]),
                    None => ([255, 0, 0, 0], [0, 0, 0, 0]),
                }
            };
            vdata.push(Vertex {
                pos: p,
                nrm: n,
                uv: md.uv[v],
                weights,
                bones,
            });
            for k in 0..3 {
                if p[k] < min[k] {
                    min[k] = p[k];
                }
                if p[k] > max[k] {
                    max[k] = p[k];
                }
            }
            any = true;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model vertex buffer"),
            contents: bytemuck::cast_slice(&vdata),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // Index buffers are u32 so the morph bind group can bind them as
        // `array<u32>` storage (u16 arrays aren't allowed in the storage
        // address space) and map vertex_index -> raw vertex id in the shader.
        let idx_u32: Vec<u32> = md.idx.iter().map(|&i| i as u32).collect();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model index buffer"),
            contents: bytemuck::cast_slice(&idx_u32),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
        });

        out.push(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: md.idx.len() as u32,
            bounds: Bounds {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            material,
            transparent,
            part: item.part,
        });
    }

    let bounds = if any {
        let center = Vec3::new(
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        );
        let mut radius = 0.0f32;
        // approximate by re-scanning half-extents (good enough for framing)
        for k in 0..3 {
            radius = radius.max(((max[k] - min[k]) * 0.5).abs());
        }
        Bounds {
            center,
            radius: radius.max(0.001),
        }
    } else {
        Bounds {
            center: Vec3::ZERO,
            radius: 1.0,
        }
    };
    (out, bounds)
}

fn mesh_material(parsed: &Parsed, i: usize) -> usize {
    match parsed.render.get(i) {
        Some(item) if item.mat >= 0 && (item.mat as usize) < parsed.materials.len() => {
            item.mat as usize
        }
        _ => 0,
    }
}

fn build_materials(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    white_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    textures: &[TexInfo],
    srcs: &[(i16, [f32; 4])],
) -> Vec<GpuMaterial> {
    let mut out = Vec::with_capacity(srcs.len());
    for (_i, (tex_id, diffuse)) in srcs.iter().enumerate() {
        let tex_view: &wgpu::TextureView =
            if *tex_id >= 0 && (*tex_id as usize) < textures.len() {
                textures[*tex_id as usize].view.as_ref()
            } else {
                white_view
            };
        let has_tex = if *tex_id >= 0 && (*tex_id as usize) < textures.len() {
            1
        } else {
            0
        };
        // The float diffuse alpha in ghg data is a leftover lighting value
        // (often 0.5 or 0.0 even for fully opaque parts; the byte alpha in
        // `rgba` is 255). Real translucency comes from DXT5 texture texels
        // (handled by the `transparent` pass), so opaque parts stay opaque.
        let base_color = [diffuse[0], diffuse[1], diffuse[2], 1.0];
        let uniform = MaterialUniform {
            base_color,
            has_tex,
            _pad: [0; 3],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        out.push(GpuMaterial {
            bind_group,
            uniform_buffer,
            tex_id: *tex_id,
            diffuse: base_color,
        });
    }
    out
}

fn build_map_meshes(device: &wgpu::Device, map: &Map) -> (Vec<GpuMesh>, Bounds) {
    let mut out = Vec::new();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any = false;
    for (i, part) in map.render_parts.iter().enumerate() {
        let Some(mesh) = map.meshes.get(part.mesh) else {
            continue;
        };
        let md = mapmesh::expand_mesh(map, mesh);
        let Some(md) = md else {
            continue;
        };
        if md.pos.is_empty() || md.idx.is_empty() {
            continue;
        }
        // Materials sampling a DXT5 texture carry alpha and go in the later
        // transparent pass.
        let transparent = map
            .materials
            .get(part.material)
            .and_then(|m| map.tex_slot(m.tex_id))
            .and_then(|s| map.textures.get(s))
            .map(|t| t.fmt == TextureFmt::Dxt5)
            .unwrap_or(false);
        let mut vdata: Vec<Vertex> = Vec::with_capacity(md.pos.len());
        let mut mmin = [f32::INFINITY; 3];
        let mut mmax = [f32::NEG_INFINITY; 3];
        for v in 0..md.pos.len() {
            let p = md.pos[v];
            let n = md.nrm[v];
            vdata.push(Vertex {
                pos: p,
                nrm: n,
                uv: md.uv[v],
                // Unskinned: full weight on the root bone keeps the bind
                // matrix (identity for maps) from moving anything.
                weights: [255, 0, 0, 0],
                bones: [0, 0, 0, 0],
            });
            for k in 0..3 {
                mmin[k] = mmin[k].min(p[k]);
                mmax[k] = mmax[k].max(p[k]);
                if p[k] < min[k] {
                    min[k] = p[k];
                }
                if p[k] > max[k] {
                    max[k] = p[k];
                }
            }
            any = true;
        }
        let mcenter = Vec3::new(
            (mmin[0] + mmax[0]) * 0.5,
            (mmin[1] + mmax[1]) * 0.5,
            (mmin[2] + mmax[2]) * 0.5,
        );
        let mut mradius = 0.0f32;
        for k in 0..3 {
            mradius = mradius.max(((mmax[k] - mmin[k]) * 0.5).abs());
        }
        let mbounds = Bounds {
            center: mcenter,
            radius: mradius.max(0.001),
        };
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("map vertex buffer"),
            contents: bytemuck::cast_slice(&vdata),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // u32 indices so the index buffer can double as the morph storage
        // binding (u16 arrays aren't allowed in storage address space).
        let idx_u32: Vec<u32> = md.idx.iter().map(|&i| i as u32).collect();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("map index buffer"),
            contents: bytemuck::cast_slice(&idx_u32),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
        });

        out.push(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count: md.idx.len() as u32,
            bounds: mbounds,
            material: part.material,
            transparent,
            part: i,
        });
    }

    let bounds = if any {
        let center = Vec3::new(
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        );
        let mut radius = 0.0f32;
        for k in 0..3 {
            radius = radius.max(((max[k] - min[k]) * 0.5).abs());
        }
        Bounds {
            center,
            radius: radius.max(0.001),
        }
    } else {
        Bounds {
            center: Vec3::ZERO,
            radius: 1.0,
        }
    };
    (out, bounds)
}
