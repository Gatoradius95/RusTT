use std::sync::Arc;

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use bytemuck::Zeroable;
use rustt::dxt;
use rustt::ghg::{Parsed, TextureFmt};
use rustt::glb::MeshData;
use rustt::map::Map;
use rustt::mapmesh;
use rustt::rtl::{self, LightSet, RtlLight};

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
    /// Baked per-vertex light (RGBA8) from the map vertex buffer (pos+12).
    /// Characters upload opaque white and are unaffected.
    color: [u8; 4],
    /// Lightmap UVs (raw file values, u in [0..1], v in [-1..0]; u <= 0
    /// selects the vertex-lit fallback). Zero for characters and for map
    /// materials without a lightmap stage.
    lm_uv: [f32; 2],
    /// Per-vertex tangent (XYZW). Byte4 from the file unpacked to f32:
    /// `(value/255)*2-1`. W is handedness for bitangent reconstruction.
    tangent: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub base_color: [f32; 4],
    pub has_tex: u32,
    /// Lighting stage derived from the MS00 `shaderDefines` bits (record
    /// +0x26C): 0 = DISABLE, 1 = LAMBERT, 6 = PHONG, ...
    pub lighting_stage: u32,
    /// 1 when the material's defines select PRELIGHT_FX — the baked
    /// per-vertex light multiply (`defines & 0x1000 && defines & 0x80000000`,
    /// BrickBench `generateDefines`). DISABLE (PRELIGHT_FX=0) stays albedo
    /// only.
    pub prelit: u32,
/// 1 when the model's highlight/glint texture should modulate the Phong
    /// specular (bound at material-bind-group binding 3).
    pub has_highlight: u32,
    /// Uber Shader 2.0 per-material params (MS00 record +0x12C/0x130/0x144/0x148):
    /// x = kCosPower, y = kSpecular, z = kFresnel, w = kFresnelPower.
    pub specular_params: [f32; 4],
    pub ambient_color: [f32; 4],
    pub incandescent_glow: [f32; 4],
    /// Lightmap stage (0 = DISABLE, 1 = LIGHTMAP_SMOOTH, 2 =
    /// LIGHTMAP_DIRECTIONAL). The game's `LIGHTMAP_STAGE` define, from MS00
    /// record +0x26E (the 239-directional-lightmap hub lighting).
    /// NOTE: `lm_stage`/`has_lm` sit at the END, mirroring the WGSL struct —
    /// the vec4s stay 16-aligned; the trailing block reads as specular data
    /// otherwise.
    pub lm_stage: u32,
    /// 1 when a valid lightmap texture set (LM0..2) is bound; 0 falls back
    /// to the baked vertex light.
    pub has_lm: u32,
    /// Material blend mode (low nibble of alpha_type MS00 +0x40): 0 = NONE,
    /// 1 = TRANSPARENT, 2 = ADDITIVE.  Opaque (0) forces alpha=1.0 in the
    /// shader so the pipeline's ALPHA_BLENDING is a no-op.
    pub blend_mode: u32,
    /// Alpha-test threshold: when > 0.0, fragments with alpha <= this value
    /// are discarded (D3D9 ALPHATESTENABLE + ALPHAREF = 0x10 ≈ 0.0627).
    /// 0.0 disables the test.
    pub alpha_cutoff: f32,
    /// Lightmap offset/scale (float bits): the game's `lightmapOffset`
    /// uniform — `lightmapCoord = uv * lm_bits.zw + lm_bits.xy`. Per-part
    /// lightmap states write (x, y, 1, 1); type 3/4 states carry a full
    /// vector. The default material path uses identity (0, 0, 1, 1).
    /// WGSL vec4<u32> aligns to 16, so the vec4 sits at 96 here too (the
    /// `blend_mode`/`alpha_cutoff` covers 88..96).
    pub lm_bits: [u32; 4],
    /// 1 when a normal map texture is bound (binding 7). Enables tangent-
    /// space normal mapping via screen-space TBN from UV/position derivatives.
    pub has_normal: u32,
    /// 1 when a specular map texture is bound (binding 8). Modulates the
    /// Phong specular intensity per-texel.
    pub has_specular: u32,
    /// 1 when a cubemap texture is bound (binding 9, envmap == Cube).
    pub has_cubemap: u32,
    /// Cubemap reflection strength (material +0x12C).
    pub reflection_power: f32,
}

/// Uber Shader 2.0 per-material defaults, matching the original's NuMaterial
/// defaults for unlit/plain materials.
pub const UBER_AMBIENT_COLOR: [f32; 4] = [0.10, 0.10, 0.10, 1.0];
pub const UBER_INCANDESCENT_GLOW: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

pub const MAX_MORPH_SLOTS: usize = 64;

/// Per-mesh morph uniform, mirroring `MorphUniform` in shaders.wgsl. The
/// `weights` array holds the current BSA blend-shape weights (one entry per
/// slot); the rest is per-part metadata written once at load.
///
/// Metadata is packed as two `vec4<u32>` (not scalars) because WGSL uniform
/// arrays require 16-byte strides and vec4s align to 16: `meta0` =
/// [num_v, slot_count, delta_base, enabled], `meta1` = [channel_base, 0, 0, 0].
///
/// `channel_base` is the BSA channel index of this part's slot 0: shape-key
/// channels are numbered across all morphing parts in part order, so the
/// face part (first) owns channels 0.., the teeth part owns the next run,
/// etc. Without the offset every part would read the face's channels and the
/// teeth would animate to the mouth/jaw shapes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MorphUniform {
    pub weights: [f32; MAX_MORPH_SLOTS],
    pub meta0: [u32; 4],
    pub meta1: [u32; 4],
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
    /// Material alpha-state from the map's material records: `blend_mode`
    /// (0 none, 1 src-alpha/1-src-alpha, 2 src-alpha/one, 3 reverse, 10
    /// none-fixed-alpha) and `depth_mode` (bits 14-15 of `alpha_type`:
    /// 0 normal, 1 no depth write, 2 always pass, 3 ignore depth). Characters
    /// pass 0/0 (solid occluders). The renderer picks the blend pipeline from
    /// these per mesh.
    pub blend_mode: u8,
    pub depth_mode: u8,
    /// Index of the source ghg part, for looking up shape-key (dynamic
    /// buffer) data. Render item i maps to part i but empty parts can be
    /// skipped during mesh building, so this is tracked explicitly.
    pub part: usize,
    /// AI2 trigger index this SO belongs to (for room-based culling).
    /// `None` means room geometry or an SO outside all triggers → always
    /// drawn.
    pub room: Option<usize>,
    /// Per-part lightmap override (display-command LIGHTMAP state): bind
    /// group + uniform carrying the page texture and the offset/scale.
    /// None = the plain material binding (material-set atlas or vertex-lit).
    pub lm_bind: Option<GpuLmBind>,
    /// Buildit visibility toggle. Matches the engine's
    /// `giz_subobj_set_visible` — bit0 of the render flags at +0x44.
    /// When false the mesh is skipped during draw calls entirely.
    pub visible: bool,
}

/// Per-part lightmap state (LIGHTMAP display command): the page texture
/// binding and the `lightmapOffset` uniform override.
pub struct GpuLmBind {
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
}

pub struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub tex_id: i16,
    pub diffuse: [f32; 4],
    /// Uber Shader 2.0 params, kept on the CPU so `set_material_color` can
    /// rewrite the uniform without losing them.
    pub specular_params: [f32; 4],
    pub ambient_color: [f32; 4],
    pub incandescent_glow: [f32; 4],
    pub lighting_stage: u8,
    /// 1 when the material's defines select PRELIGHT_FX (`defines & 0x1000`);
    /// the baked per-vertex light multiply. Characters pass 0.
    /// (opaque white, bit-for-bit unaffected).
    pub prelit: u8,
    /// Model-level highlight/glint texture flag, kept on the CPU so
    /// `set_material_color` can rewrite the uniform without losing it.
    pub has_highlight: u8,
    /// Lightmap stage (0 = DISABLE, 2 = LIGHTMAP_DIRECTIONAL). Kept on the
    /// CPU so `set_material_color` can rewrite the uniform without losing it.
    pub lm_stage: u8,
    /// 1 when a lightmap texture set (LM0..2) is actually bound.
    pub has_lm: u8,
    /// Material blend mode (low nibble of alpha_type): 0 = NONE, 1 = TRANSPARENT.
    /// Upper 16 bits encode refraction_type (0 = none, 3 = REFRACTION_GLASS).
    pub blend_mode: u32,
    /// Alpha-test threshold: when > 0.0, fragments with alpha <= this value
    /// are discarded (D3D9 ALPHATESTENABLE). Needed by `set_material_color`
    /// to preserve the cutoff when rewriting the uniform.
    pub alpha_cutoff: f32,
    /// 1 when a normal map texture is bound (material binding 7).
    pub has_normal: u32,
    /// 1 when a specular map texture is bound (material binding 8).
    pub has_specular: u32,
    /// 1 when a cubemap texture is bound (material binding 9).
    pub has_cubemap: u32,
    /// Cubemap reflection power.
    pub reflection_power: f32,
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
    /// Per-part room assignment (AI2 trigger index). `None` for geometry
    /// that is always drawn (room geometry, SOs outside all triggers).
    part_rooms: Vec<Option<usize>>,
    /// Per-mesh transform buffer (group 4).  Identity for static geometry;
    /// buildit sub-objects get animated world transforms written each frame.
    mesh_xform_layout: wgpu::BindGroupLayout,
    mesh_xform_bind_group: wgpu::BindGroup,
    mesh_xform_buffer: wgpu::Buffer,
    /// Per-mesh SO/room type flag buffer (group 4, binding 1).
    /// 0 = room geometry, 1 = SO entity.  Used by the 'O' debug coloring mode.
    mesh_type_buffer: wgpu::Buffer,
    /// Pre-sorted draw order: indices into `meshes`, grouped to minimize
    /// bind-group / pipeline switches.  Opaque meshes sorted by
    /// (material, lightmap, depth_mode, morph_group); transparent meshes
    /// by (pipeline, material, lightmap, depth_mode).
    sorted_opaque: Vec<usize>,
    sorted_transparent: Vec<usize>,
    /// CPU-side staging copy of the mesh transform buffer.  Kept in sync
    /// with `mesh_xform_buffer` so `set_mesh_transforms` can do a single
    /// `queue.write_buffer` instead of N per-override calls.
    xform_staging: Vec<u8>,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
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
        // Nearest-neighbor sampler for the cubemap: avoids face-edge sparkle
        // caused by linear filtering sampling across cross-layout face
        // boundaries (matching BactaTank Classic's mip_off + filter=false).
        let cube_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cubemap nearest sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
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
        let mat_srcs: Vec<MatSrc> = parsed
            .materials
            .iter()
            .map(|m| {
                MatSrc {
                    tex_id: m.tex_id,
                    diffuse: [m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0],
                    lighting_stage: m.lighting_stage,
                    prelit: 0,
                    specular_params: m.specular_params,
                    lm_stage: 0,
                    lm0: -1,
                    lm1: -1,
                    lm2: -1,
                    blend_mode: 0,
                    tex_normal: m.tex_normal,
                    tex_specular: m.tex_specular,
                    tex_cubemap: m.tex_cubemap,
                    reflection_power: m.reflection_power,
                    shader_defines: m.shader_defines,
                }
            })
            .collect();
        let highlight_view: Option<&wgpu::TextureView> = parsed
            .highlight_tex
            .and_then(|i| textures.get(i))
            .map(|t| t.view.as_ref());
        let materials = build_materials(
            device,
            &material_layout,
            &white_view,
            &sampler,
            &cube_sampler,
            &textures,
            highlight_view,
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
        // Characters share one lights uniform (the classic rig); map meshes
        // get per-mesh lights from their `.RTL` (see `from_map`).
        let lights_buffer = create_lights_buffer(device, &LightSet::default());
        let (morph_layout, default_morph_bind_group) = build_default_morph(device, queue, &lights_buffer);
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
        // BSA channels number the shape-key slots of every morphing part in
        // part order: the face part (first) owns channels 0.., the teeth part
        // owns the next run, etc. Assign each part a channel base in part
        // order so meshes sharing a part agree on it.
        let mut part_channel_base = vec![0u32; parsed.parts.len()];
        {
            let mut base: u32 = 0;
            for (pi, part) in parsed.parts.iter().enumerate() {
                if part.dynamic_buffers.is_empty() {
                    continue;
                }
                part_channel_base[pi] = base;
                base += part.dynamic_buffers.len() as u32;
            }
        }
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
                meta0: [
                    part.num_v as u32,
                    part.dynamic_buffers.len() as u32,
                    part_delta_base[mesh.part] as u32,
                    1,
                ],
                meta1: [part_channel_base[mesh.part], 0, 0, 0],
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
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lights_buffer.as_entire_binding(),
                    },
                ],
            });
            morph_groups[i] = Some(bind_group);
            morph_uniforms[i] = Some((uniform_buffer, meta));
        }

        let mesh_xform_layout = Self::create_mesh_xform_layout(device);
        let xform_staging: Vec<u8> = {
            let ident = Mat4::IDENTITY.to_cols_array();
            let ident_bytes = bytemuck::cast_slice(&ident);
            (0..meshes.len())
                .flat_map(|_| ident_bytes.iter().copied())
                .collect()
        };
        let mesh_xform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh transforms"),
            size: xform_staging.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&mesh_xform_buffer, 0, &xform_staging);
        let mesh_type_data: Vec<u32> = vec![0u32; meshes.len()];
        let mesh_type_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh type debug"),
            contents: bytemuck::cast_slice(&mesh_type_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mesh_xform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh xform bind group"),
            layout: &mesh_xform_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh_xform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mesh_type_buffer.as_entire_binding(),
                },
            ],
        });

        let mut scene = Self {
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
            part_rooms: Vec::new(),
            mesh_xform_layout,
            mesh_xform_bind_group,
            mesh_xform_buffer,
            mesh_type_buffer,
            sorted_opaque: Vec::new(),
            sorted_transparent: Vec::new(),
            xform_staging,
        };
        scene.build_draw_order();
        scene
    }

    /// Build a scene from a parsed map file (`.GSC`). No bones and no shape
    /// keys: every mesh gets the shared default morph group and an identity
    /// skin matrix, and meshes are static geometry built from the map's
    /// triangle strips. The sibling `.RTL` light list lights each mesh from
    /// its own position (the original's per-part light baking); an empty
    /// list falls back to the classic character rig.
    pub fn from_map(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        map: &Map,
        lights: &[RtlLight],
        is_srgb: bool,
    ) -> Self {
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
        let cube_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cubemap nearest sampler (map)"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
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
        // PRELIGHT_FX = `shaderDefines & 0x1000 && shaderDefines & 0x80000000`
        // (the exe's FUN_00749620 emits the define only when BOTH bits are set;
        // bit 12 alone — positive value — falls to LIGHTING_STAGE=DISABLE).
        // Lighting is baked into the per-vertex light color and multiplied in
        // the shader (`fs_layer0_color`).
                let mut mat_srcs: Vec<MatSrc> = map
            .materials
            .iter()
            .map(|m| {
                let tex_id = map
                    .tex_slot(m.tex_id)
                    .filter(|&s| s < tex_slot_to_index.len() && tex_slot_to_index[s] != usize::MAX)
                    .map(|s| tex_slot_to_index[s] as i16)
                    .unwrap_or(-1);
                let prelit = (m.shader_defines & 0x1000 != 0 && m.shader_defines & 0x8000_0000 != 0) as u8;
                // Remap normal and specular texture indices through the same
                // slot→index translation as the diffuse texture.
                let remap_tex = |raw: i32| -> i16 {
                    if raw < 0 {
                        return -1;
                    }
                    map.tex_slot(raw as i16)
                        .filter(|&s| s < tex_slot_to_index.len() && tex_slot_to_index[s] != usize::MAX)
                        .map(|s| tex_slot_to_index[s] as i16)
                        .unwrap_or(-1)
                };
                // The lightmap set is the three textures immediately after
                // `lightmap_set_index` in real-texture space, remapped like
                // `tex_id`. The stage (0/2) is validated against LM0 in
                // build_materials (missing textures disable the set).
                // VIEWER_LM_PAGE_OFFSET (debug): offset from the set base in
                // real-texture space — e.g. +241 lands on the 1024x1024
                // top-view light pages (real 243..282 = slots 253..292).
                let lm_page = std::env::var("VIEWER_LM_PAGE_OFFSET")
                    .ok()
                    .and_then(|v| v.parse::<u16>().ok())
                    .unwrap_or(0);
                let lm_id = |i: u16| -> i16 {
                    let real = m.lightmap_set_index as u16 + i + lm_page;
                    if m.lightmap_stage() == 0 {
                        return -1;
                    }
                    map.tex_slot(real as i16)
                        .filter(|&s| s < tex_slot_to_index.len() && tex_slot_to_index[s] != usize::MAX)
                        .map(|s| tex_slot_to_index[s] as i16)
                        .unwrap_or(-1)
                };
                MatSrc {
                    tex_id,
                    diffuse: [m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0],
                    lighting_stage: m.lighting_stage,
                    prelit,
                    specular_params: m.specular_params,
                    lm_stage: m.lightmap_stage(),
                    lm0: lm_id(0),
                    lm1: lm_id(1),
                    lm2: lm_id(2),
                    blend_mode: m.blend_mode(),
                    tex_normal: remap_tex(m.tex_normal),
                    tex_specular: remap_tex(m.tex_specular),
                    tex_cubemap: remap_tex(m.tex_cubemap),
                    reflection_power: m.specular_params[1],
                    shader_defines: m.shader_defines,
                }
            })
            .collect();
        if std::env::var("GLASS_DIAG").is_ok() {
            for (mi, m) in map.materials.iter().enumerate() {
                let prelit = m.shader_defines & 0x1000 != 0 && m.shader_defines & 0x8000_0000 != 0;
                let tex_id_raw = map.tex_slot(m.tex_id);
                let has_tex = tex_id_raw.is_some();
                if m.id == 310 || (prelit && !has_tex) {
                    eprintln!(
                        "  GLASS_DIAG mi={} id={} bm={} prelit={} has_tex={} defs=0x{:08x} depth={} diff={:?} alpha_type=0x{:08x}",
                        mi, m.id, m.blend_mode(), prelit, has_tex, m.shader_defines, m.depth_mode(), m.diffuse, m.alpha_type,
                    );
                }
            }
        }
        // VIEWER_LM_OFF disables the lightmap set (debug A/B: the scene then
        // renders with the pure baked-vertex fallback). Value-aware: only "1",
        // "true", "yes" count, so "0" keeps lightmaps on.
        let lm_off = std::env::var("VIEWER_LM_OFF")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);
        if lm_off {
            for m in mat_srcs.iter_mut() {
                m.lm_stage = 0;
                m.lm0 = -1;
                m.lm1 = -1;
                m.lm2 = -1;
            }
        }
        let materials = build_materials(
            device,
            &material_layout,
            &white_view,
            &sampler,
            &cube_sampler,
            &textures,
            None,
            &mat_srcs,
        );
        {
            let staged = mat_srcs
                .iter()
                .filter(|s| s.lm_stage != 0 && s.lm0 >= 0)
                .count();
            let staged_prelit = mat_srcs
                .iter()
                .filter(|s| s.lm_stage != 0 && s.lm0 >= 0 && s.prelit != 0)
                .count();
            let full_lm = mat_srcs
                .iter()
                .filter(|s| s.lm_stage != 0 && s.lm0 >= 0 && s.lm1 >= 0 && s.lm2 >= 0)
                .count();
            let any_uv = map
                .meshes
                .iter()
                .filter_map(|m| rustt::mapmesh::expand_mesh(map, m))
                .map(|md| md.lm_uv.iter().filter(|u| u[0] > 0.0).count())
                .sum::<usize>();
            println!(
                "map scene: lightmaps: {}/{} staged ({} prelit, {} full set), {} lm-uv vertices u>0",
                staged,
                mat_srcs.len(),
                staged_prelit,
                full_lm,
                any_uv
            );
            for (i, s) in mat_srcs.iter().enumerate().take(mat_srcs.len()) {
                if s.lm_stage != 0 {
                    let d = |id: i16| -> String {
                        if id < 0 {
                            "none".to_string()
                        } else {
                            let t = &textures[id as usize];
                            format!("{}x{}", t.w, t.h)
                        }
                    };
                    println!(
                        "  mat {}: stage={} prelit={} ls={} lm0={} lm1={} lm2={}",
                        i,
                        s.lm_stage,
                        s.prelit,
                        s.lighting_stage,
                        d(s.lm0),
                        d(s.lm1),
                        d(s.lm2)
                    );
                    if i > 6 {
                        break;
                    }
                }
            }
        }

        let (meshes, bounds) = build_map_meshes(device, map, &material_layout, &sampler, &cube_sampler, &white_view, &textures, lm_off);
        println!(
            "map scene: {} meshes, {} triangles",
            meshes.len(),
            meshes.iter().map(|m| m.index_count as usize / 3).sum::<usize>()
        );
        if lights.is_empty() {
            println!("map scene: no lights (no RTL), falling back to the default rig");
        } else {
            println!("map scene: {} lights", lights.len());
        }

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

        // Per-mesh lights from the RTL list, computed at each mesh's bounds
        // center like the original's per-part light baking. Every map mesh
        // gets its own bind group (morph delta/uniform/index bind the shared
        // dummies; binding 3 is the per-mesh lights uniform).
        let lights_buffer = create_lights_buffer(device, &LightSet::default());
        let (morph_layout, default_morph_bind_group) = build_default_morph(device, queue, &lights_buffer);
        let dummies = morph_dummies(device);
        let mut morph_groups = Vec::with_capacity(meshes.len());
        for mesh in &meshes {
            let set = rtl::compute_light_set(lights, mesh.bounds.center.to_array());
            let per_mesh_lights = create_lights_buffer(device, &set);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("map lights bind group"),
                layout: &morph_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dummies.0.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dummies.1.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dummies.2.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: per_mesh_lights.as_entire_binding(),
                    },
                ],
            });
            morph_groups.push(Some(bind_group));
        }
        let morph_uniforms: Vec<Option<(wgpu::Buffer, MorphUniform)>> = vec![None; meshes.len()];

        // Per-mesh transform buffer (group 4): one Mat4 per mesh, identity
        // for static geometry, animated transforms for buildit sub-objects.
        let mesh_xform_layout = Self::create_mesh_xform_layout(device);
        let xform_staging: Vec<u8> = {
            let ident = Mat4::IDENTITY.to_cols_array();
            let ident_bytes = bytemuck::cast_slice(&ident);
            (0..meshes.len())
                .flat_map(|_| ident_bytes.iter().copied())
                .collect()
        };
        let mesh_xform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh transforms"),
            size: xform_staging.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&mesh_xform_buffer, 0, &xform_staging);
        let mesh_type_buffer = Self::build_mesh_type_buffer(device, queue, &meshes, map);
        let mesh_xform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh xform bind group"),
            layout: &mesh_xform_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh_xform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mesh_type_buffer.as_entire_binding(),
                },
            ],
        });

        let mut scene = Self {
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
            part_rooms: Vec::new(),
            mesh_xform_layout,
            mesh_xform_bind_group,
            mesh_xform_buffer,
            mesh_type_buffer,
            sorted_opaque: Vec::new(),
            sorted_transparent: Vec::new(),
            xform_staging,
        };
        scene.build_draw_order();
        scene
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

    pub fn create_mesh_xform_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh xform layout"),
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
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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

    /// Build a per-mesh type buffer: 0 = room geometry, 1 = SO entity.
    fn build_mesh_type_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[GpuMesh],
        map: &Map,
    ) -> wgpu::Buffer {
        let types: Vec<u32> = meshes
            .iter()
            .map(|m| {
                if let Some(rp) = map.render_parts.get(m.part) {
                    if rp.name.is_some() { 1u32 } else { 0u32 }
                } else {
                    0u32
                }
            })
            .collect();
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh type debug"),
            contents: bytemuck::cast_slice(&types),
            usage: wgpu::BufferUsages::STORAGE,
        });
        buf
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
                // Per-mesh lights: the uber shader's `u_lights` uniform
                // (scene ambient + 3 light colors + 3 light directions).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

    pub fn skin_bind_group(&self) -> &wgpu::BindGroup {
        &self.skin_bind_group
    }

    /// Set room assignments for render parts. Index `i` corresponds to
    /// render part `i`; `None` means always draw, `Some(idx)` means only
    /// draw when trigger `idx` is active.
    pub fn set_part_rooms(&mut self, part_rooms: Vec<Option<usize>>) {
        for mesh in &mut self.meshes {
            mesh.room = part_rooms.get(mesh.part).copied().flatten();
        }
        self.part_rooms = part_rooms;
    }

    pub fn skin_layout(&self) -> &wgpu::BindGroupLayout {
        &self.skin_layout
    }

    pub fn morph_layout(&self) -> &wgpu::BindGroupLayout {
        &self.morph_layout
    }

    pub fn mesh_xform_layout(&self) -> &wgpu::BindGroupLayout {
        &self.mesh_xform_layout
    }

    /// Upload per-mesh transform overrides.  Patches the CPU-side staging
    /// buffer and does a single `queue.write_buffer` for the entire buffer.
    /// Non-listed meshes keep their identity transform.
    pub fn set_mesh_transforms(&mut self, queue: &wgpu::Queue, overrides: &[(usize, Mat4)]) {
        let mat_size = std::mem::size_of::<Mat4>();
        // Reset staging to identity so any previous overrides are cleared.
        let ident = Mat4::IDENTITY.to_cols_array();
        let ident_bytes = bytemuck::cast_slice(&ident);
        for chunk in self.xform_staging.chunks_exact_mut(mat_size) {
            chunk.copy_from_slice(ident_bytes);
        }
        // Patch overrides.
        for &(idx, xform) in overrides {
            let offset = idx * mat_size;
            if offset + mat_size <= self.xform_staging.len() {
                self.xform_staging[offset..offset + mat_size]
                    .copy_from_slice(bytemuck::cast_slice(&xform.to_cols_array()));
            }
        }
        // Single GPU upload.
        queue.write_buffer(&self.mesh_xform_buffer, 0, &self.xform_staging);
    }

    pub fn morph_bind_group(&self, i: usize) -> &wgpu::BindGroup {
        self.morph_groups
            .get(i)
            .and_then(|o| o.as_ref())
            .unwrap_or(&self.default_morph_bind_group)
    }

    pub fn mesh_xform_bind_group(&self) -> &wgpu::BindGroup {
        &self.mesh_xform_bind_group
    }

    /// Pre-sort draw order for minimal bind-group / pipeline switches.
    /// Called once after construction.
    pub fn build_draw_order(&mut self) {
        let n = self.meshes.len();
        let mut opaque: Vec<usize> = (0..n).collect();
        let mut transparent: Vec<usize> = (0..n).collect();

        opaque.retain(|&i| {
            let m = &self.meshes[i];
            let depth_normal = m.depth_mode == 0 || m.depth_mode == 2;
            !m.transparent && depth_normal && (m.blend_mode == 0 || m.blend_mode == 1)
        });
        transparent.retain(|&i| {
            let m = &self.meshes[i];
            let depth_normal = m.depth_mode == 0 || m.depth_mode == 2;
            m.transparent || !depth_normal || (m.blend_mode != 0 && m.blend_mode != 1)
        });

        let opaque_sort_key = |meshes: &[GpuMesh], i: &usize| {
            let m = &meshes[*i];
            let lm = m.lm_bind.is_some() as u8;
            let morph_default = (self.morph_groups.get(*i).and_then(|o| o.as_ref()).is_none()) as u8;
            (m.material, lm, m.depth_mode, m.blend_mode, morph_default)
        };
        let transparent_sort_key = |meshes: &[GpuMesh], i: &usize| {
            let m = &meshes[*i];
            let pipeline = if m.blend_mode == 2 {
                if m.depth_mode == 0 || m.depth_mode == 2 { 0u8 } else { 1 }
            } else {
                2
            };
            let lm = m.lm_bind.is_some() as u8;
            let morph_default = (self.morph_groups.get(*i).and_then(|o| o.as_ref()).is_none()) as u8;
            (pipeline, m.material, lm, m.depth_mode, morph_default)
        };

        opaque.sort_by_key(|i| opaque_sort_key(&self.meshes, i));
        transparent.sort_by_key(|i| transparent_sort_key(&self.meshes, i));

        self.sorted_opaque = opaque;
        self.sorted_transparent = transparent;
    }

    pub fn sorted_opaque(&self) -> &[usize] {
        &self.sorted_opaque
    }

    pub fn sorted_transparent(&self) -> &[usize] {
        &self.sorted_transparent
    }

    /// Upload the current BSA blend-shape weights to every morphing mesh's
    /// uniform. `weights` is the flat channel array (channel `c` = `weights[c]`,
    /// 0 when out of range); each part's shader reads it starting at its own
    /// `channel_base`, so slot `s` of part `p` takes channel
    /// `p.channel_base + s`. A ~1 KB total write per frame, far cheaper than
    /// rebuilding vertex buffers.
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
            lighting_stage: mat.lighting_stage as u32,
            prelit: mat.prelit as u32,
            has_highlight: mat.has_highlight as u32,
            lm_stage: mat.lm_stage as u32,
            has_lm: mat.has_lm as u32,
            blend_mode: mat.blend_mode as u32,
            specular_params: mat.specular_params,
            ambient_color: mat.ambient_color,
            incandescent_glow: mat.incandescent_glow,
            alpha_cutoff: mat.alpha_cutoff,
            lm_bits: [0, 0, 1.0f32.to_bits(), 1.0f32.to_bits()],
            has_normal: mat.has_normal,
            has_specular: mat.has_specular,
            has_cubemap: mat.has_cubemap,
            reflection_power: mat.reflection_power,
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
    lights_buffer: &wgpu::Buffer,
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: lights_buffer.as_entire_binding(),
            },
        ],
    });
    (morph_layout, group)
}

fn create_lights_buffer(device: &wgpu::Device, lights: &LightSet) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lights uniform"),
        contents: bytemuck::bytes_of(lights),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Shared dummy bindings for static meshes (no morph data): a tiny storage
/// buffer, an empty morph uniform and a tiny index-storage buffer.
fn morph_dummies(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map morph dummy storage"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map morph dummy uniform"),
        size: std::mem::size_of::<MorphUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let small = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map morph dummy index storage"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (storage, uniform, small)
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
    let dump_dir = std::env::var_os("VIEWER_DUMP_LM");
    for (i, t) in srcs.iter().enumerate() {
        let Ok(rgba) = dxt::decode_rgba(t.w, t.h, t.fmt, t.payload) else {
            eprintln!("  warning: texture {i} failed to decode; skipping");
            continue;
        };
        if let Some(dir) = &dump_dir {
            let dir = dir.to_string_lossy();
            let p = format!("{dir}\\lmtex_{i}.png");
            if let Ok(fl) = std::fs::File::create(&p) {
                let mut enc = png::Encoder::new(fl, t.w as u32, t.h as u32);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                if let Ok(mut w) = enc.write_header() {
                    let _ = w.write_image_data(&rgba);
                }
            }
        }
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
        // transparent mesh a hair off the surface along its per-vertex
        // normals (view-independent, zoom-independent). Radially lifting from
        // the bounding center barely separates flat decals from coplanar
        // opaque detail (e.g. the mouth quads against the mouth sliver), so
        // we use the winding normal, flipped to point away from the mesh's
        // own center, with a radial fallback for missing normals. Computed
        // once in bind space; the vertex shader's skin matrix then transforms
        // the lifted position along with the rest of the mesh.
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
            // Near-flat decals (mouth, face details) must lift uniformly
            // along the mesh's average normal. The per-vertex direction plus
            // the away-from-center flip folds the sheet: zero-normal verts get
            // pushed +/-Y while their same-position twins go -Z, the outer lip
            // corners (z behind the center) clear by only ~2.5mm against the
            // ~4mm opaque slab underneath, and they splay sideways. A single
            // rigid translation keeps the sheet planar with even clearance.
            let mut avg = Vec3::ZERO;
            let mut n_ok = 0usize;
            for i in 0..md.nrm.len().min(md.pos.len()) {
                let l = Vec3::from(md.nrm[i]).length();
                if l > 1e-6 {
                    avg += Vec3::from(md.nrm[i]);
                    n_ok += 1;
                }
            }
            let flat = n_ok > 0 && (avg.length() / n_ok as f32) > 0.3;
            for (i, p) in md.pos.iter_mut().enumerate() {
                if flat {
                    // 0.004 clears the mouth slab: the face lower-face (3.7mm
                    // thick) and the teeth strip (3.3mm) interleave over ~4mm,
                    // so 0.002 left most of the decal behind the opaque teeth.
                    *p = (Vec3::from(*p) + (avg / avg.length()) * 0.004).into();
                    continue;
                }
                let out = Vec3::from(*p) - center;
                let ol = out.length();
                let dir = if i < md.nrm.len() {
                    let raw = Vec3::from(md.nrm[i]);
                    let l = raw.length();
                    if l > 1e-6 {
                        let n = if out.dot(raw) < 0.0 { -raw } else { raw };
                        n / l
                    } else if ol > 1e-6 {
                        out / ol
                    } else {
                        Vec3::Z
                    }
                } else if ol > 1e-6 {
                    out / ol
                } else {
                    Vec3::Z
                };
                // 0.004 clears the mouth slab: the face lower-face (3.7mm
                // thick) and the teeth strip (3.3mm) interleave over ~4mm,
                // so 0.002 left most of the decal behind the opaque teeth.
                *p = (Vec3::from(*p) + dir * 0.004).into();
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
                color: md.color.get(v).copied().unwrap_or([255, 255, 255, 255]),
                lm_uv: md.lm_uv.get(v).copied().unwrap_or([0.0, 0.0]),
                tangent: md.tangent.get(v).copied().unwrap_or([0.0, 0.0, 1.0, 1.0]),
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
            blend_mode: 0,
            depth_mode: 0,
            part: item.part,
            room: None,
            lm_bind: None,
            visible: true,
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

/// Source material data for `build_materials`. Consolidates the per-material
/// fields extracted from both GHG and MAP parsers.
struct MatSrc {
    tex_id: i16,
    diffuse: [f32; 4],
    lighting_stage: u8,
    prelit: u8,
    specular_params: [f32; 4],
    lm_stage: u8,
    lm0: i16,
    lm1: i16,
    lm2: i16,
    blend_mode: u8,
    tex_normal: i16,
    tex_specular: i16,
    tex_cubemap: i16,
    reflection_power: f32,
    shader_defines: u32,
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
    cube_sampler: &wgpu::Sampler,
    textures: &[TexInfo],
    highlight_view: Option<&wgpu::TextureView>,
    srcs: &[MatSrc],
) -> Vec<GpuMaterial> {
    let mut out = Vec::with_capacity(srcs.len());
    let view_at = |id: i16| -> &wgpu::TextureView {
        if id >= 0 && (id as usize) < textures.len() {
            textures[id as usize].view.as_ref()
        } else {
            white_view
        }
    };
    let diag_no_prelit = std::env::var("VIEWER_NOPRELIT").is_ok();
    for (_i, s) in srcs.iter().enumerate() {
        let MatSrc { tex_id, diffuse, lighting_stage, prelit, specular_params, lm_stage, lm0, lm1, lm2, blend_mode, tex_normal, tex_specular, tex_cubemap, reflection_power, shader_defines } = s;
        let tex_view = view_at(*tex_id);
        let has_tex = if *tex_id >= 0 && (*tex_id as usize) < textures.len() {
            1
        } else {
            0
        };
        // Alpha-test cutoff for DXT5 cutout materials (face decals, foliage).
        // The original D3D9 engine uses ALPHATESTENABLE with ALPHAREF=0x10
        // (GREATER func), discarding fragments with alpha <= 16/255 ≈ 0.0627.
        // Only DXT5 opaque (blend_mode 0) materials need the test — blend_mode
        // 1/2 materials use real alpha blending and should not discard.
        let alpha_cutoff = if *tex_id >= 0
            && (*tex_id as usize) < textures.len()
            && textures[*tex_id as usize].fmt == "DXT5"
            && *blend_mode == 0
        {
            16.0f32 / 255.0
        } else {
            0.0f32
        };
        // LM0 is the alpha/animation channel: when it is missing the whole
        // lightmap set is disabled regardless of the recorded stage.
        let has_lm = if *lm0 >= 0 && (*lm0 as usize) < textures.len() {
            *lm_stage as u32
        } else {
            0
        };
        let has_highlight = if highlight_view.is_some() { 1 } else { 0 };
        let has_normal = if *tex_normal >= 0 && (*tex_normal as usize) < textures.len() { 1 } else { 0 };
        let has_specular = if *tex_specular >= 0 && (*tex_specular as usize) < textures.len() { 1 } else { 0 };
        // EnvMap detection: shader_flags bits 5-6 == 1 (Cube) AND cubemap ID is valid.
        // BT_ENVMAP_BITS = 0x03, BT_ENVMAP_SHIFT = 0x05, BTEnvMapType::Cube = 1.
        let envmap_type = (*shader_defines >> 5) & 0x03;
        let has_cubemap = if envmap_type == 1 && *tex_cubemap >= 0 && (*tex_cubemap as usize) < textures.len() { 1 } else { 0 };
        // When the specular and cubemap slots point at the same texture, the
        // specular sample reads the sky image through mesh UVs — noisy garbage.
        // Suppress the specular sample in that case (the cubemap path handles
        // the reflection correctly).
        let has_specular = if has_cubemap == 1 && *tex_specular == *tex_cubemap { 0 } else { has_specular };
        // The float diffuse alpha in ghg data is a leftover lighting value
        // (often 0.5 or 0.0 even for fully opaque parts; the byte alpha in
        // `rgba` is 255). Real translucency comes from DXT5 texture texels
        // (handled by the `transparent` pass), so opaque parts stay opaque.
        let base_color = [diffuse[0], diffuse[1], diffuse[2], 1.0];
        // Detect glass refraction materials: prelit + no texture + transparent
        // blend.  Pack refraction_type into the upper 16 bits of blend_mode
        // so the shader can decode it without an extra uniform field.
        let is_glass = (*prelit != 0) && (has_tex == 0) && (*blend_mode == 1);
        let refraction_type: u32 = if is_glass { 3 } else { 0 }; // 3 = REFRACTION_GLASS
        let packed_blend = (*blend_mode as u32) | (refraction_type << 16);
        let uniform = MaterialUniform {
            base_color,
            has_tex,
            lighting_stage: *lighting_stage as u32,
            prelit: if diag_no_prelit { 0 } else { *prelit as u32 },
            has_highlight,
            lm_stage: has_lm,
            has_lm: if has_lm != 0 { 1 } else { 0 },
            blend_mode: packed_blend,
            specular_params: *specular_params,
            ambient_color: UBER_AMBIENT_COLOR,
            incandescent_glow: UBER_INCANDESCENT_GLOW,
            alpha_cutoff,
            lm_bits: [0, 0, 1.0f32.to_bits(), 1.0f32.to_bits()],
            has_normal,
            has_specular,
            has_cubemap,
            reflection_power: *reflection_power,
        };
        if _i < 5 || _i == 131 || _i == 310 {
            let bytes = bytemuck::bytes_of(&uniform);
            eprintln!(
                "  mat {_i}: blend_mode={} alpha_cutoff={alpha_cutoff:.4} (byte@88={:02x}) lm_bits@96={:02x?} total={} bytes",
                *blend_mode,
                bytes[88],
                &bytes[96..112],
                bytes.len(),
            );
        }
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        highlight_view.unwrap_or(white_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(view_at(*lm0)),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(view_at(*lm1)),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(view_at(*lm2)),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(view_at(*tex_normal)),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(view_at(*tex_specular)),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(view_at(*tex_cubemap)),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&cube_sampler),
                },
            ],
        });
        out.push(GpuMaterial {
            bind_group,
            uniform_buffer,
            tex_id: *tex_id,
            diffuse: base_color,
            specular_params: *specular_params,
            ambient_color: UBER_AMBIENT_COLOR,
            incandescent_glow: UBER_INCANDESCENT_GLOW,
            lighting_stage: *lighting_stage,
            prelit: *prelit,
            has_highlight: has_highlight as u8,
            lm_stage: has_lm as u8,
            has_lm: (has_lm != 0) as u8,
            blend_mode: packed_blend,
            alpha_cutoff,
            has_normal,
            has_specular,
            has_cubemap,
            reflection_power: *reflection_power,
        });
    }
    out
}fn build_map_meshes(
    device: &wgpu::Device,
    map: &Map,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    cube_sampler: &wgpu::Sampler,
    white_view: &wgpu::TextureView,
    textures: &[TexInfo],
    lm_off: bool,
) -> (Vec<GpuMesh>, Bounds) {
    // ONLY_PART (debug): isolate render parts in the scene — every other
    // part is skipped, so a screenshot shows one object alone against the
    // clear color. Accepts a single index or a comma list ("116" or
    // "116,117,118"). Used to visually confirm that parts picked from the
    // raycast picker window are the objects the user is looking at.
    let only_part: Option<std::collections::HashSet<usize>> = std::env::var("ONLY_PART")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect::<std::collections::HashSet<usize>>()
        })
        .filter(|s| !s.is_empty());
    if let Some(ref parts) = only_part {
        let list: Vec<String> = parts.iter().map(|p| p.to_string()).collect();
        println!("ONLY_PART: isolating render parts {}", list.join(","));
    }
    let view_at = |id: i32| -> &wgpu::TextureView {
        if id >= 0 && (id as usize) < textures.len() {
            textures[id as usize].view.as_ref()
        } else {
            white_view
        }
    };
    // --- Pass 1: expand all meshes into CPU-side MeshData ----------------
    struct CpuPart {
        md: rustt::glb::MeshData,
        part_idx: usize,
        material: usize,
        transparent: bool,
        blend_mode: u8,
        depth_mode: u8,
    }
    let mut cpu_parts: Vec<CpuPart> = Vec::new();
    for (i, part) in map.render_parts.iter().enumerate() {
        // ONLY_PART isolation: keep just the requested parts.
        if only_part.as_ref().is_some_and(|set| !set.contains(&i)) {
            continue;
        }
        let Some(mesh) = map.meshes.get(part.mesh) else {
            continue;
        };
        let Some(mut md) = mapmesh::expand_mesh(map, mesh) else {
            continue;
        };
        if md.pos.is_empty() || md.idx.is_empty() {
            continue;
        }
        // Apply the MTXLOAD model matrix so local-space meshes (chairs,
        // doors, …) end up at their world-space positions.
        mapmesh::apply_transform(&mut md, &part.transform);
        let transparent = map
            .materials
            .get(part.material)
            .and_then(|m| map.tex_slot(m.tex_id))
            .and_then(|s| map.textures.get(s))
            .map(|t| t.fmt == TextureFmt::Dxt5)
            .unwrap_or(false);
        let blend_mode = map
            .materials
            .get(part.material)
            .map(|m| m.blend_mode())
            .unwrap_or(0);
        let depth_mode = map
            .materials
            .get(part.material)
            .map(|m| m.depth_mode())
            .unwrap_or(0);
        cpu_parts.push(CpuPart { md, part_idx: i, material: part.material, transparent, blend_mode, depth_mode });
    }

    // --- Pass 2: create GPU buffers from the CPU-side data ----
    let mut out = Vec::new();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any = false;
    for cp in &cpu_parts {
        let md = &cp.md;
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
                color: md.color.get(v).copied().unwrap_or([255, 255, 255, 255]),
                lm_uv: md.lm_uv.get(v).copied().unwrap_or([0.0, 0.0]),
                tangent: md.tangent.get(v).copied().unwrap_or([0.0, 0.0, 1.0, 1.0]),
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
            material: cp.material,
            transparent: cp.transparent,
            blend_mode: cp.blend_mode,
            depth_mode: cp.depth_mode,
            part: cp.part_idx,
            room: None,
            lm_bind: None,
            visible: true,
        });
    }

    // Second pass over the parts that carry a LIGHTMAP display-command state:
    // bind the per-draw page texture(s) and the offset transform. The bind
    // group mirrors the material layout so the same pipeline works; the
    // uniform overrides the material's with `lm_bits` = lightmapOffset.
    for (mi, mesh) in out.iter_mut().enumerate() {
        if lm_off {
            break;
        }
        let Some(part) = map.render_parts.get(mesh.part) else { continue };
        if part.lightmap == 0 {
            continue;
        }
        let Some(st) = map.lightmaps.get(&part.lightmap) else {
            continue;
        };
        let m = &map.materials[part.material];
        let page0 = map.tex_slot(st.tex[0] as i16);
        let Some(pre) = page0 else { continue };
        let pre = pre as i32;
        // The part carries a lightmap display-command state (part.lightmap
        // != 0, checked above) with a valid page texture: the lightmap set is
        // live regardless of the material's LIGHTMAP_STAGE define parse (the
        // stage bits are consumed by the shader as a nonzero switch; stages 1
        // and 2 both sample LM0 as the diffuse light).
        let has_lm = if pre >= 0 && (pre as usize) < textures.len() {
            (st.ty.max(1)) as u32
        } else {
            0
        };
        let (x, y, z, w) = (st.off[0], st.off[1], st.off[2], st.off[3]);
        // Type 1/2 states use only x/y as the offset with unit scale (the
        // file's z/w are unused there; BrickBench sets (x, y, 1, 1)).
        let (sz, sw) = if st.ty <= 2 { (1.0f32, 1.0f32) } else { (z, w) };
        let tex_id = map
            .tex_slot(m.tex_id)
            .map(|s| s as i32)
            .unwrap_or(-1);
        let has_tex = if tex_id >= 0 && (tex_id as usize) < textures.len() {
            1
        } else {
            0
        };
        let prelit = (m.shader_defines & 0x1000 != 0 && m.shader_defines & 0x8000_0000 != 0) as u32;
        let bm = m.blend_mode();
        let is_glass = prelit != 0 && has_tex == 0 && bm == 1;
        let refraction_type: u32 = if is_glass { 3 } else { 0 };
        let packed_blend = (bm as u32) | (refraction_type << 16);
        // Remap normal/specular texture indices through the slot table.
        let norm_id = m.tex_normal;
        let norm_view = if norm_id >= 0 {
            map.tex_slot(norm_id as i16)
                .filter(|&s| s < textures.len())
                .map(|s| view_at(s as i32))
                .unwrap_or(white_view)
        } else {
            white_view
        };
        let spec_id = m.tex_specular;
        let spec_view = if spec_id >= 0 {
            map.tex_slot(spec_id as i16)
                .filter(|&s| s < textures.len())
                .map(|s| view_at(s as i32))
                .unwrap_or(white_view)
        } else {
            white_view
        };
        // Mirror build_materials: EnvMap detection from shader_defines bits
        // 5-6 == 1 (Cube) AND a valid cubemap ID. When the specular and
        // cubemap slots point at the same texture, the specular sample reads
        // the sky image through mesh UVs — suppress it (the cubemap path
        // handles the reflection).
        let cube_id = m.tex_cubemap;
        let cube_view = if cube_id >= 0 {
            map.tex_slot(cube_id as i16)
                .filter(|&s| s < textures.len())
                .map(|s| view_at(s as i32))
                .unwrap_or(white_view)
        } else {
            white_view
        };
        let envmap_type = (m.shader_defines >> 5) & 0x03;
        let has_cubemap = if envmap_type == 1 && cube_id >= 0 && map.tex_slot(cube_id as i16).is_some() { 1 } else { 0 };
        let has_specular = if has_cubemap == 1 && spec_id == cube_id { 0 } else { 1 };
        let uniform = MaterialUniform {
            base_color: [m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0],
            has_tex,
            lighting_stage: m.lighting_stage as u32,
            prelit,
            has_highlight: 0,
            lm_stage: has_lm,
            has_lm: if has_lm != 0 { 1 } else { 0 },
            blend_mode: packed_blend,
            specular_params: m.specular_params,
            ambient_color: UBER_AMBIENT_COLOR,
            incandescent_glow: UBER_INCANDESCENT_GLOW,
            alpha_cutoff: 0.0,
            lm_bits: [x.to_bits(), y.to_bits(), sz.to_bits(), sw.to_bits()],
            has_normal: if norm_id >= 0 && map.tex_slot(norm_id as i16).is_some() { 1 } else { 0 },
            has_specular,
            has_cubemap,
            reflection_power: m.specular_params[1],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("map lightmap uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let lm1s = map.tex_slot(st.tex[1] as i16).map(|s| s as i32).unwrap_or(-1);
        let lm2s = map.tex_slot(st.tex[2] as i16).map(|s| s as i32).unwrap_or(-1);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("map lightmap bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view_at(tex_id)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(white_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(view_at(pre)),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(view_at(lm1s)),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(view_at(lm2s)),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(norm_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(spec_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(cube_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&cube_sampler),
                },
            ],
        });
        mesh.lm_bind = Some(GpuLmBind {
            bind_group,
            uniform_buffer,
        });
        if mi % 400 == 0 {
            eprintln!(
                "  lm part {mi}: page tex[{}] pre={pre} off=({x:.4},{y:.4},{sz:.4},{sw:.4}) ty={}",
                st.tex[0], st.ty
            );
        }
        if matches!(mesh.part, 123 | 126 | 129 | 579 | 581) {
            eprintln!(
                "  lm part {} (mesh {mi}): key={} ty={} tex=[{},{},{},{}] off=({x:.4},{y:.4},{z:.4},{w:.4}) pre={pre} lm1s={lm1s} lm2s={lm2s} has_lm={has_lm} bytes={:02x?}",
                mesh.part,
                part.lightmap,
                st.ty,
                st.tex[0],
                st.tex[1],
                st.tex[2],
                st.tex[3],
                &bytemuck::bytes_of(&uniform)[76..104],
            );
        }
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

