use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use glam::Mat4;
use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::camera::OrbitCamera;
use crate::scene::GpuScene;
use rustt::ghg::Parsed;
use rustt::map::Map;
use rustt::rtl::RtlLight;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
    /// Per-draw world transform; written just before each scene draw.
    model: [f32; 16],
    /// World-space camera position plus fog state. Lighting is per-mesh now
    /// (`u_lights` in the scene's group 3), so it lives in the scene, not
    /// here.
    cam_pos: [f32; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

struct GridData {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

pub struct GpuRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub scene: GpuScene,
    pub show_grid: bool,
    pub show_wireframe: bool,
    /// Draw the grid without depth testing (read-through-walls overlay).
    pub show_grid_xray: bool,
    /// When true, the VIEWER_CULL sphere is active; toggled by 'C' key.
    pub cull_enabled: bool,
    /// When true, force alpha=1.0 on all fragments (debug: rule out blending seams).
    pub force_opaque: bool,
    /// When true, apply a post-process color correction curve that approximates
    /// the original D3D9 game's sRGB-space lighting look.
    pub color_correct_enabled: bool,
    /// When true, color all geometry by SO/room type: green = room, yellow = SO.
    pub so_coloring_enabled: bool,
    /// When false, skip cubemap sampling (debug toggle: key '0').
    pub cubemap_enabled: bool,
    /// When false, skip normal map sampling (debug toggle: key '1').
    pub normal_map_enabled: bool,
    /// Cached at init from `VIEWER_DRAWS` env var. When true, print per-mesh
    /// draw diagnostics in the opaque/transparent passes.
    pub diag_draws: bool,

    depth_view: wgpu::TextureView,
    /// Pipeline for truly opaque geometry (blend_mode 0): REPLACE blend,
    /// depth write ON. No alpha blending — eliminates part-boundary seam
    /// artifacts caused by vertex-color alpha < 1.0 bleeding through
    /// ALPHA_BLENDING.
    opaque_noblend_pipeline: wgpu::RenderPipeline,
    model_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    /// Additive lighting (material blend 2, src-alpha over one): the strobe
    /// and mood lights around the stage. `add_nzw_pipeline` is the no-depth-
    /// write variant for materials whose depth state says NO_WRITE.
    add_zw_pipeline: wgpu::RenderPipeline,
    add_nzw_pipeline: wgpu::RenderPipeline,
    model_wire_pipeline: Option<wgpu::RenderPipeline>,
    lines_pipeline: wgpu::RenderPipeline,
    /// Same lines, but depth-compare ALWAYS: lets the grid read through
    /// walls so the player can point at props hidden behind furniture.
    lines_xray_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    /// Bind group for the opaque pass: uses a 1×1 dummy texture instead of
    /// the backbuffer so wgpu doesn't see the backbuffer as both a texture
    /// resource and a render target simultaneously.
    camera_bind_group_opaque: wgpu::BindGroup,
    /// Bind group for the transparent pass: binds the actual backbuffer so
    /// glass refraction can sample the opaque scene.
    camera_bind_group_transparent: wgpu::BindGroup,
    camera_layout: wgpu::BindGroupLayout,
    /// View-projection matrix for the current frame, staged by `update_camera`
    /// and written into the camera buffer (combined with each scene's model
    /// matrix) right before its meshes are drawn.
    camera_view_proj: Mat4,
    /// World-space camera position, staged by `update_camera` for the uber
    /// shader's view vector and fog distance.
    camera_pos: [f32; 3],
    /// Offscreen backbuffer for refraction: opaque geometry renders to the
    /// swapchain, then is copied here so transparent glass can sample it.
    backbuffer_tex: wgpu::Texture,
    backbuffer_view: wgpu::TextureView,
    backbuffer_sampler: wgpu::Sampler,
    dummy_view: wgpu::TextureView,
    grid: GridData,
}

impl GpuRenderer {
    /// Create the wgpu instance, adapter, device/queue and surface for a
    /// window. Shared by the ghg and map entry points.
    fn init_surface(
        event_loop: &winit::event_loop::ActiveEventLoop,
        window: &Arc<winit::window::Window>,
    ) -> Result<(
        wgpu::Device,
        wgpu::Queue,
        wgpu::Surface<'static>,
        wgpu::SurfaceConfiguration,
    )> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_with_display_handle(Box::new(
                event_loop.owned_display_handle(),
            ))
        });

        let surface = instance
            .create_surface(window.clone())
            .context("creating surface")?;

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .context("requesting adapter")?;

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("viewer device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_bind_groups: 5,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        }))
        .context("requesting device")?;

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .context("surface has no default config")?;
        // COPY_SRC lets the debug screenshot path read the swapchain back.
        // COPY_DST lets the opaque pass blit into it for refraction.
        config.usage |= wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        // Force opaque alpha so the DWM and screen-capture tools (OBS Game
        // Capture) don't treat the window as potentially transparent — the
        // adapter default can be PreMultiplied/PostMultiplied on some configs.
        config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
        surface.configure(&device, &config);

        Ok((device, queue, surface, config))
    }

    /// Finish scene-independent setup (pipelines, depth, grid) and assemble
    /// the renderer around an already-built scene.
    fn finalize(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        scene: GpuScene,
    ) -> Result<Self> {
        // Camera uniform buffer (shared by all pipelines).
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // Both the vertex and fragment stages read the camera uniform
                    // (the fragment shader uses its lighting block for the uber
                    // shader).
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
        // Offscreen backbuffer for refraction (opaque scene copied here for
        // transparent glass to sample).
        let backbuffer_tex = create_backbuffer_tex(&device, config.width, config.height, config.format);
        let backbuffer_view = backbuffer_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let backbuffer_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("backbuffer sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // 1×1 dummy texture for the opaque pass (avoids using the backbuffer
        // as both a render target and texture resource simultaneously).
        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy 1x1"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let camera_bind_group_opaque = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group opaque"),
            layout: &camera_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&backbuffer_sampler),
                },
            ],
        });
        let camera_bind_group_transparent = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group transparent"),
            layout: &camera_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&backbuffer_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&backbuffer_sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders.wgsl"));

        let camera_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("camera pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let model_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("model pipeline layout"),
            bind_group_layouts: &[
                Some(&camera_layout),
                Some(&scene.material_layout),
                Some(scene.skin_layout()),
                Some(scene.morph_layout()),
                Some(scene.mesh_xform_layout()),
            ],
            immediate_size: 0,
        });

        let model_pipeline = make_model_pipeline(
            &device,
            &shader,
            &model_pipeline_layout,
            config.format,
            None,
            true,
            None,
            None,
            wgpu::BlendState::ALPHA_BLENDING,
            "model pipeline",
        );

        // Opaque no-blend pipeline: REPLACE blend for blend_mode 0 materials.
        // The original game sets D3DRS_ALPHABLENDENABLE=FALSE for opaque
        // materials. Without this, vertex-color alpha ~0.996 bleeds 0.4% of
        // the framebuffer background at part-boundary edges via ALPHA_BLENDING.
        let opaque_noblend_pipeline = make_model_pipeline(
            &device,
            &shader,
            &model_pipeline_layout,
            config.format,
            None,
            true,
            None,
            None,
            wgpu::BlendState::REPLACE,
            "opaque no-blend pipeline",
        );

        // Transparent meshes (DXT5-textured face decals, capes, cloth) are
        // drawn after the opaque geometry and must not write depth, otherwise
        // their invisible/translucent fragments would occlude the head behind
        // them (the "face cuts through the model" artifacts). Their vertices
        // are lifted off the surface in scene.rs; the small depth bias here
        // is D3D9-style polygon-offset insurance against interpolation gaps
        // at grazing angles where the lifted decal sits level with the
        // opaque surface (mouth quads vs the teeth strip).
        let transparent_pipeline = make_model_pipeline(
            &device,
            &shader,
            &model_pipeline_layout,
            config.format,
            None,
            false,
            Some(wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 1.0,
                clamp: 0.0,
            }),
            // LessEqual lets coplanar transparent fragments win over the
            // opaque depth that the lift leaves them exactly level with
            // (the mouth quads vs the teeth strip). A strict Less test
            // culls them whenever interpolation lands a hair behind.
            Some(wgpu::CompareFunction::LessEqual),
            wgpu::BlendState::ALPHA_BLENDING,
            "transparent pipeline",
        );

        // Additive map lighting (material blend 2 = TRANSPARENT_IGNORE_DEST,
        // src-alpha over one): the black strobe panels' backdrop contributes
        // nothing (black * whatever is behind) so the wall shows through,
        // and the colored dots add on top. The depth test always runs;
        // NO_WRITE materials (depth bits 14-15 = 1) just skip writing so
        // other additive panels behind them aren't occluded.
        let add_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let add_zw_pipeline = make_model_pipeline(
            &device,
            &shader,
            &model_pipeline_layout,
            config.format,
            None,
            true,
            None,
            None,
            add_blend,
            "additive pipeline",
        );
        let add_nzw_pipeline = make_model_pipeline(
            &device,
            &shader,
            &model_pipeline_layout,
            config.format,
            None,
            false,
            None,
            None,
            add_blend,
            "additive no-depth-write pipeline",
        );

        let wire_supported = device.features().contains(wgpu::Features::POLYGON_MODE_LINE);
        let model_wire_pipeline = if wire_supported {
            Some(make_model_pipeline(
                &device,
                &shader,
                &model_pipeline_layout,
                config.format,
                Some(wgpu::PolygonMode::Line),
                true,
                None,
                None,
                wgpu::BlendState::ALPHA_BLENDING,
                "wireframe pipeline",
            ))
        } else {
            None
        };

        let lines_pipeline = make_lines_pipeline(&device, &shader, &camera_pipeline_layout, config.format, wgpu::CompareFunction::LessEqual);
        let lines_xray_pipeline =
            make_lines_pipeline(&device, &shader, &camera_pipeline_layout, config.format, wgpu::CompareFunction::Always);

        let depth_view = create_depth_view(&device, config.width, config.height);
        let grid = make_grid(
            &device,
            [scene.bounds.center.x, scene.bounds.center.z],
            scene.bounds.radius.max(1.0),
        );

        Ok(Self {
            device,
            queue,
            surface,
            config,
            scene,
            show_grid: true,
            show_wireframe: false,
            show_grid_xray: false,
            cull_enabled: std::env::var("VIEWER_CULL").is_ok(),
            force_opaque: std::env::var("FORCE_OPAQUE").is_ok(),
            color_correct_enabled: std::env::var("COLOR_CORRECT").is_ok(),
            so_coloring_enabled: false,
            cubemap_enabled: true,
            normal_map_enabled: true,
            diag_draws: std::env::var("VIEWER_DRAWS").is_ok(),
            depth_view,
            opaque_noblend_pipeline,
            model_pipeline,
            transparent_pipeline,
            add_zw_pipeline,
            add_nzw_pipeline,
            model_wire_pipeline,
            lines_pipeline,
            lines_xray_pipeline,
            camera_buffer,
            camera_bind_group_opaque,
            camera_bind_group_transparent,
            camera_layout,
            camera_view_proj: Mat4::IDENTITY,
            camera_pos: [0.0, 0.0, 0.0],
            backbuffer_tex,
            backbuffer_view,
            backbuffer_sampler,
            dummy_view,
            grid,
        })
    }

    pub fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        window: &Arc<winit::window::Window>,
        parsed: &Parsed,
        file_name: &str,
        allowed_layers: &[u32],
    ) -> Result<Self> {
        let (device, queue, surface, config) = Self::init_surface(event_loop, window)?;
        let is_srgb = config.format.is_srgb();
        let scene = GpuScene::new(&device, &queue, parsed, is_srgb, allowed_layers);
        let _ = file_name;
        Self::finalize(device, queue, surface, config, scene)
    }

    /// Build a renderer for a parsed map file (`.GSC`). No layers, bones or
    /// shape keys; the scene is built from the map's render parts and lit
    /// per-mesh from the sibling `.RTL` light list.
    pub fn new_map(
        event_loop: &winit::event_loop::ActiveEventLoop,
        window: &Arc<winit::window::Window>,
        map: &Map,
        lights: &[RtlLight],
        file_name: &str,
    ) -> Result<Self> {
        let (device, queue, surface, config) = Self::init_surface(event_loop, window)?;
        let is_srgb = config.format.is_srgb();
        let scene = GpuScene::from_map(&device, &queue, map, lights, is_srgb);
        let _ = file_name;
        Self::finalize(device, queue, surface, config, scene)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, width, height);
        // Recreate backbuffer at new resolution and rebind the camera group.
        self.backbuffer_tex = create_backbuffer_tex(&self.device, width, height, self.config.format);
        self.backbuffer_view = self.backbuffer_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.camera_bind_group_opaque = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group opaque"),
            layout: &self.camera_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.backbuffer_sampler),
                },
            ],
        });
        self.camera_bind_group_transparent = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group transparent"),
            layout: &self.camera_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.backbuffer_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.backbuffer_sampler),
                },
            ],
        });
    }

    /// Rebuild the scene so only the given layer (LOD/quality) set is drawn.
    /// An empty `layers` renders every layer. Called when the user switches
    /// quality in the UI.
    pub fn set_layers(&mut self, parsed: &Parsed, layers: &[u32]) {
        self.scene = GpuScene::new(
            &self.device,
            &self.queue,
            parsed,
            self.config.format.is_srgb(),
            layers,
        );
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn backbuffer_view(&self) -> &wgpu::TextureView {
        &self.backbuffer_view
    }

    pub fn backbuffer_tex(&self) -> &wgpu::Texture {
        &self.backbuffer_tex
    }

    pub fn wireframe_supported(&self) -> bool {
        self.model_wire_pipeline.is_some()
    }

    /// Stage the camera's view-projection matrix for this frame. It is written
    /// to the camera uniform (with an identity model); callers that draw
    /// multiple scenes per frame call `write_camera` between submits to switch
    /// the model transform.
    pub fn update_camera(&mut self, camera: &OrbitCamera) {
        let aspect = self.config.width as f32 / (self.config.height as f32).max(1.0);
        self.camera_view_proj = camera.view_proj(aspect);
        let p = camera.position();
        self.camera_pos = [p.x, p.y, p.z];
        self.write_camera(Mat4::IDENTITY);
    }

    /// Write `camera_view_proj * model` into the camera uniform buffer.
    ///
    /// This is a `queue.write_buffer` (enqueued before any following submit),
    /// so callers must submit the draw commands that consume it before issuing
    /// another `write_camera` for a different model. Mixing models inside one
    /// submit makes every draw read the last-written matrix.
    pub fn write_camera(&self, model: Mat4) {
        // Bitfield packed into cam_pos.w: bit0 = force_opaque, bit1 = color_correct,
        // bit2 = so_coloring.
        // Shader decodes with: let mode = u32(u_cam.cam_pos.w + 0.5);
        let mut flags: f32 = 0.0;
        if self.force_opaque { flags += 1.0; }
        if self.color_correct_enabled { flags += 2.0; }
        if self.so_coloring_enabled { flags += 4.0; }
        if !self.cubemap_enabled { flags += 8.0; }
        if !self.normal_map_enabled { flags += 16.0; }
        let uniform = CameraUniform {
            view_proj: self.camera_view_proj.to_cols_array(),
            model: model.to_cols_array(),
            cam_pos: [
                self.camera_pos[0],
                self.camera_pos[1],
                self.camera_pos[2],
                flags,
            ],
            fog_color: [0.0, 0.0, 0.0, 1.0],
            fog_params: [0.0, 0.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Draw `self.scene` at the identity transform (viewer behaviour), then
    /// the coordinate grid on top so its lines read over the floor.
    pub fn draw_scene(&self, rpass: &mut wgpu::RenderPass<'_>) {
        let cull: Option<(glam::Vec3, f32)> = if self.cull_enabled {
            std::env::var("VIEWER_CULL").ok().and_then(|v| {
                let p: Vec<f32> = v.split(',').filter_map(|s| s.parse().ok()).collect();
                (p.len() >= 4).then(|| (glam::Vec3::new(p[0], p[1], p[2]), p[3]))
            })
        } else {
            None
        };
        if self.show_wireframe {
            if let Some(p) = &self.model_wire_pipeline {
                rpass.set_pipeline(p);
                self.draw_scene_meshes_culled(rpass, &self.scene, true, cull.as_ref(), &std::collections::HashSet::new());
            }
        } else {
            self.draw_scene_meshes_culled(rpass, &self.scene, false, cull.as_ref(), &std::collections::HashSet::new());
        }
        self.draw_grid(rpass);
    }

    /// Draw the coordinate grid (identity transform).
    ///
    /// A second, brighter grid renders when `show_grid_xray` is set (depth
    /// compare ALWAYS) so coordinates stay visible through walls, letting the
    /// player point at props hidden behind furniture.
    pub fn draw_grid(&self, rpass: &mut wgpu::RenderPass<'_>) {
        if !self.show_grid {
            return;
        }
        self.write_camera(Mat4::IDENTITY);
        if self.show_grid_xray {
            rpass.set_pipeline(&self.lines_xray_pipeline);
        } else {
            rpass.set_pipeline(&self.lines_pipeline);
        }
        rpass.set_bind_group(0, Some(&self.camera_bind_group_opaque), &[]);
        rpass.set_vertex_buffer(0, self.grid.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.grid.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rpass.draw_indexed(0..self.grid.index_count, 0, 0..1);
    }

    /// Draw one scene (opaque, then transparent meshes). The camera uniform
    /// must already hold the desired model matrix (see `write_camera`).
    ///
    /// When `cull` is set, only meshes whose bounds sphere intersects a sphere
    /// of `cull.radius` around `cull.center` are drawn. The game uses this to
    /// render just the room the player is in rather than the whole hub.
    pub fn draw_scene_meshes(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        scene: &GpuScene,
        wireframe: bool,
    ) {
        self.draw_scene_meshes_culled(
            rpass,
            scene,
            wireframe,
            None::<&(glam::Vec3, f32)>,
            &std::collections::HashSet::new(),
        );
    }

    /// `draw_scene_meshes` with an optional cull sphere (center, radius).
    pub fn draw_scene_meshes_culled(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        scene: &GpuScene,
        wireframe: bool,
        cull: Option<&(glam::Vec3, f32)>,
        active_rooms: &HashSet<usize>,
    ) {
        self.draw_scene_opaque_culled(rpass, scene, wireframe, cull, active_rooms);
        self.draw_scene_transparent_culled(rpass, scene, wireframe, cull, active_rooms);
    }

    /// Draw only the opaque pass of a scene (depth-write, REPLACE blend).
    pub fn draw_scene_opaque_culled(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        scene: &GpuScene,
        wireframe: bool,
        cull: Option<&(glam::Vec3, f32)>,
        active_rooms: &HashSet<usize>,
    ) {
        rpass.set_bind_group(0, Some(&self.camera_bind_group_opaque), &[]);
        rpass.set_bind_group(2, Some(scene.skin_bind_group()), &[]);
        rpass.set_bind_group(4, Some(scene.mesh_xform_bind_group()), &[]);
        if !wireframe {
            rpass.set_pipeline(&self.opaque_noblend_pipeline);
        }
        let diag_draws = self.diag_draws;
        if diag_draws {
            for &i in scene.sorted_opaque() {
                let mesh = &scene.meshes[i];
                if matches!(mesh.part, 123 | 126 | 129 | 579 | 581) {
                    eprintln!(
                        "mesh {i}: part={} mat={} idx={} transparent={} depth={} blend={} lm={}",
                        mesh.part, mesh.material, mesh.index_count, mesh.transparent,
                        mesh.depth_mode, mesh.blend_mode, mesh.lm_bind.is_some(),
                    );
                }
            }
        }
        let mut opaque = 0u32;
        let mut skip_mat = 0u32;
        let mut skip_cull = 0u32;
        // Cached bind-group pointers to skip redundant set_bind_group calls.
        // We compare raw pointer identity — two meshes sharing the same
        // material (and no lightmap) will have the same BindGroup, so we
        // can skip re-binding when adjacent in the sorted order.
        let mut last_bg1_ptr: usize = 0;
        let mut last_bg3_ptr: usize = 0;
        for &i in scene.sorted_opaque() {
            let mesh = &scene.meshes[i];
            if !cull_mesh(mesh, cull, active_rooms) {
                skip_cull += 1;
                continue;
            }
            let material = match scene.materials.get(mesh.material) {
                Some(m) => m,
                None => {
                    skip_mat += 1;
                    continue;
                }
            };
            // Group 1: material or lightmap bind group.
            let bg1: &wgpu::BindGroup = match &mesh.lm_bind {
                Some(lm) => &lm.bind_group,
                None => &material.bind_group,
            };
            let bg1_ptr = std::ptr::from_ref(bg1) as usize;
            if bg1_ptr != last_bg1_ptr {
                rpass.set_bind_group(1, Some(bg1), &[]);
                last_bg1_ptr = bg1_ptr;
            }
            // Group 3: morph bind group.
            let bg3 = scene.morph_bind_group(i);
            let bg3_ptr = std::ptr::from_ref(bg3) as usize;
            if bg3_ptr != last_bg3_ptr {
                rpass.set_bind_group(3, Some(bg3), &[]);
                last_bg3_ptr = bg3_ptr;
            }
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.index_count, 0, i as u32..(i as u32 + 1));
            opaque += 1;
        }
        if diag_draws {
            eprintln!(
                "draws: opaque={} skip_mat={} skip_cull={}/{} meshes",
                opaque,
                skip_mat,
                skip_cull,
                scene.meshes.len()
            );
        }
    }

    /// Draw only the transparent pass of a scene (alpha blend).
    pub fn draw_scene_transparent_culled(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        scene: &GpuScene,
        wireframe: bool,
        cull: Option<&(glam::Vec3, f32)>,
        active_rooms: &HashSet<usize>,
    ) {
        rpass.set_bind_group(0, Some(&self.camera_bind_group_transparent), &[]);
        rpass.set_bind_group(2, Some(scene.skin_bind_group()), &[]);
        rpass.set_bind_group(4, Some(scene.mesh_xform_bind_group()), &[]);
        let diag_draws = self.diag_draws;
        // Transparent meshes (face decals, capes, cloth) are drawn after the
        // opaque geometry so their translucent parts blend over it instead of
        // cutting through it. Map meshes whose material carries alpha state
        // (the strobe/mood lights, masked windows) join them, each getting
        // its own blend pipeline.
        let mut transparent = 0u32;
        let mut last_pipeline_ptr: usize = 0;
        let mut last_bg1_ptr: usize = 0;
        let mut last_bg3_ptr: usize = 0;
        for &i in scene.sorted_transparent() {
            let mesh = &scene.meshes[i];
            if !cull_mesh(mesh, cull, active_rooms) {
                continue;
            }
            if diag_draws && matches!(mesh.part, 123 | 126 | 129 | 579 | 581) {
                eprintln!(
                    "tmesh {i}: part={} mat={} idx={} transparent={} depth={} blend={} lm={}",
                    mesh.part, mesh.material, mesh.index_count, mesh.transparent,
                    mesh.depth_mode, mesh.blend_mode, mesh.lm_bind.is_some(),
                );
            }
            // Pipeline: additive (blend 2) vs standard transparent.
            if !wireframe {
                let pipeline: &wgpu::RenderPipeline = if mesh.blend_mode == 2 {
                    if mesh.depth_mode == 0 || mesh.depth_mode == 2 {
                        &self.add_zw_pipeline
                    } else {
                        &self.add_nzw_pipeline
                    }
                } else {
                    &self.transparent_pipeline
                };
                let p_ptr = std::ptr::from_ref(pipeline) as usize;
                if p_ptr != last_pipeline_ptr {
                    rpass.set_pipeline(pipeline);
                    last_pipeline_ptr = p_ptr;
                }
            }
            let material = match scene.materials.get(mesh.material) {
                Some(m) => m,
                None => continue,
            };
            // Group 1: material or lightmap bind group.
            let bg1: &wgpu::BindGroup = match &mesh.lm_bind {
                Some(lm) => &lm.bind_group,
                None => &material.bind_group,
            };
            let bg1_ptr = std::ptr::from_ref(bg1) as usize;
            if bg1_ptr != last_bg1_ptr {
                rpass.set_bind_group(1, Some(bg1), &[]);
                last_bg1_ptr = bg1_ptr;
            }
            // Group 3: morph bind group.
            let bg3 = scene.morph_bind_group(i);
            let bg3_ptr = std::ptr::from_ref(bg3) as usize;
            if bg3_ptr != last_bg3_ptr {
                rpass.set_bind_group(3, Some(bg3), &[]);
                last_bg3_ptr = bg3_ptr;
            }
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.index_count, 0, i as u32..(i as u32 + 1));
            transparent += 1;
        }
        if diag_draws {
            eprintln!("draws: transparent={transparent}");
        }
    }
}

/// True if the mesh should be drawn. Without a cull sphere every mesh draws;
/// with one, the mesh draws only when its bounds sphere overlaps the cull
/// sphere.  If the mesh has a `room` assignment, the trigger must also be
/// active (in `active_rooms`).  Room check is skipped when `cull` is None
/// (player pressed 'C' to disable culling).
fn cull_mesh(mesh: &crate::scene::GpuMesh, cull: Option<&(glam::Vec3, f32)>, active_rooms: &HashSet<usize>) -> bool {
    // Buildit visibility: matches the engine's giz_subobj_set_visible —
    // bit0 of the render flags at +0x44.
    if !mesh.visible {
        return false;
    }
    let Some(&(center, radius)) = cull else {
        return true;
    };
    // Room-based cull: SOs tagged with a trigger index are only drawn when
    // the player is in that room.
    if let Some(room) = mesh.room {
        if !active_rooms.contains(&room) {
            return false;
        }
    }
    let mb = &mesh.bounds;
    let rr = radius + mb.radius;
    let dx = center.x - mb.center.x;
    let dy = center.y - mb.center.y;
    let dz = center.z - mb.center.z;
    dx * dx + dy * dy + dz * dz <= rr * rr
}

fn make_model_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    polygon_mode: Option<wgpu::PolygonMode>,
    depth_write: bool,
    depth_bias: Option<wgpu::DepthBiasState>,
    depth_compare: Option<wgpu::CompareFunction>,
    blend: wgpu::BlendState,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_model"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 68,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![
                    0 => Float32x3,
                    1 => Float32x3,
                    2 => Float32x2,
                    3 => Unorm8x4,
                    4 => Uint8x4,
                    5 => Unorm8x4,
                    6 => Float32x2,
                    7 => Float32x4,
                ],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            // The ghg mesh data is wound counter-clockwise as seen from
            // outside (glTF convention). Winding direction varies between
            // parts, and thin meshes (capes, cloth) are effectively two-sided,
            // so we render both faces (no culling) to match the game.
            front_face: wgpu::FrontFace::Cw,
            cull_mode: None,
            polygon_mode: polygon_mode.unwrap_or(wgpu::PolygonMode::Fill),
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(depth_compare.unwrap_or(wgpu::CompareFunction::Less)),
            stencil: wgpu::StencilState::default(),
            bias: depth_bias.unwrap_or(wgpu::DepthBiasState::default()),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_model"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn make_lines_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("lines pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_lines"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 28,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Cw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_lines"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_backbuffer_tex(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("backbuffer texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn make_grid(device: &wgpu::Device, center: [f32; 2], radius: f32) -> GridData {
    let extent = (radius * 1.5).ceil() as i32;
    let mut verts: Vec<LineVertex> = Vec::new();
    let mut idx: Vec<u16> = Vec::new();

    // Minor 1 m lines, brighter 5 m majors, red/blue axes through the map
    // centre so cross-camera distances stay readable.
    let minor = [0.16, 0.16, 0.19, 1.0];
    let major = [0.26, 0.26, 0.31, 1.0];
    let axis_x = [0.90, 0.25, 0.25, 1.0];
    let axis_z = [0.25, 0.40, 0.90, 1.0];

    let push_line = |verts: &mut Vec<LineVertex>,
                         idx: &mut Vec<u16>,
                         a: [f32; 3],
                         b: [f32; 3],
                         c: [f32; 4]| {
        let base = verts.len() as u16;
        verts.push(LineVertex { pos: a, color: c });
        verts.push(LineVertex { pos: b, color: c });
        idx.push(base);
        idx.push(base + 1);
    };

    let e = extent as f32;
    for i in -extent..=extent {
        let f = i as f32;
        let col = if i == 0 { axis_x } else if i % 5 == 0 { major } else { minor };
        push_line(
            &mut verts,
            &mut idx,
            [center[0] + f, 0.0, center[1] - e],
            [center[0] + f, 0.0, center[1] + e],
            col,
        );
        let col = if i == 0 { axis_z } else if i % 5 == 0 { major } else { minor };
        push_line(
            &mut verts,
            &mut idx,
            [center[0] - e, 0.0, center[1] + f],
            [center[0] + e, 0.0, center[1] + f],
            col,
        );
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid vertex buffer"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid index buffer"),
        contents: bytemuck::cast_slice(&idx),
        usage: wgpu::BufferUsages::INDEX,
    });
    GridData {
        vertex_buffer,
        index_buffer,
        index_count: idx.len() as u32,
    }
}
