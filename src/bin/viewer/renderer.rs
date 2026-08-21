use std::sync::Arc;

use anyhow::{Context, Result};
use glam::Mat4;
use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::camera::OrbitCamera;
use crate::scene::GpuScene;
use rustt::ghg::Parsed;
use rustt::map::Map;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
    /// Per-draw world transform; written just before each scene draw.
    model: [f32; 16],
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

    depth_view: wgpu::TextureView,
    model_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    model_wire_pipeline: Option<wgpu::RenderPipeline>,
    lines_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    /// View-projection matrix for the current frame, staged by `update_camera`
    /// and written into the camera buffer (combined with each scene's model
    /// matrix) right before its meshes are drawn.
    camera_view_proj: Mat4,
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
            ..Default::default()
        }))
        .context("requesting device")?;

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .context("surface has no default config")?;
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
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
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
            "model pipeline",
        );

        // Transparent meshes (DXT5-textured face decals, capes, cloth) are
        // drawn after the opaque geometry and must not write depth, otherwise
        // their invisible/translucent fragments would occlude the head behind
        // them (the "face cuts through the model" artifacts). Their vertices
        // are lifted off the surface in scene.rs; the small depth bias here
        // is just insurance against interpolation gaps at grazing angles.
        let transparent_pipeline = make_model_pipeline(
            &device,
            &shader,
            &model_pipeline_layout,
            config.format,
            None,
            false,
            Some(wgpu::DepthBiasState {
                constant: -64,
                slope_scale: 2.0,
                clamp: 0.0,
            }),
            "transparent pipeline",
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
                "wireframe pipeline",
            ))
        } else {
            None
        };

        let lines_pipeline = make_lines_pipeline(&device, &shader, &camera_pipeline_layout, config.format);

        let depth_view = create_depth_view(&device, config.width, config.height);
        let grid = make_grid(&device, scene.bounds.radius.max(1.0));

        Ok(Self {
            device,
            queue,
            surface,
            config,
            scene,
            show_grid: true,
            show_wireframe: false,
            depth_view,
            model_pipeline,
            transparent_pipeline,
            model_wire_pipeline,
            lines_pipeline,
            camera_buffer,
            camera_bind_group,
            camera_view_proj: Mat4::IDENTITY,
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
    /// shape keys; the scene is built from the map's render parts.
    pub fn new_map(
        event_loop: &winit::event_loop::ActiveEventLoop,
        window: &Arc<winit::window::Window>,
        map: &Map,
        file_name: &str,
    ) -> Result<Self> {
        let (device, queue, surface, config) = Self::init_surface(event_loop, window)?;
        let is_srgb = config.format.is_srgb();
        let scene = GpuScene::from_map(&device, &queue, map, is_srgb);
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
        self.write_camera(Mat4::IDENTITY);
    }

    /// Write `camera_view_proj * model` into the camera uniform buffer.
    ///
    /// This is a `queue.write_buffer` (enqueued before any following submit),
    /// so callers must submit the draw commands that consume it before issuing
    /// another `write_camera` for a different model. Mixing models inside one
    /// submit makes every draw read the last-written matrix.
    pub fn write_camera(&self, model: Mat4) {
        let uniform = CameraUniform {
            view_proj: self.camera_view_proj.to_cols_array(),
            model: model.to_cols_array(),
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Draw the grid and `self.scene` at the identity transform (viewer
    /// behaviour).
    pub fn draw_scene(&self, rpass: &mut wgpu::RenderPass<'_>) {
        self.draw_grid(rpass);
        if self.show_wireframe {
            if let Some(p) = &self.model_wire_pipeline {
                rpass.set_pipeline(p);
                self.draw_scene_meshes(rpass, &self.scene, true);
            }
        } else {
            self.draw_scene_meshes(rpass, &self.scene, false);
        }
    }

    /// Draw the reference grid (identity transform).
    pub fn draw_grid(&self, rpass: &mut wgpu::RenderPass<'_>) {
        if !self.show_grid {
            return;
        }
        self.write_camera(Mat4::IDENTITY);
        rpass.set_pipeline(&self.lines_pipeline);
        rpass.set_bind_group(0, Some(&self.camera_bind_group), &[]);
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
        );
    }

    /// `draw_scene_meshes` with an optional cull sphere (center, radius).
    pub fn draw_scene_meshes_culled(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        scene: &GpuScene,
        wireframe: bool,
        cull: Option<&(glam::Vec3, f32)>,
    ) {
        rpass.set_bind_group(0, Some(&self.camera_bind_group), &[]);
        rpass.set_bind_group(2, Some(scene.skin_bind_group()), &[]);
        if !wireframe {
            rpass.set_pipeline(&self.model_pipeline);
        }
        for (i, mesh) in scene.meshes.iter().enumerate() {
            if mesh.transparent {
                continue;
            }
            if !cull_mesh(mesh, cull) {
                continue;
            }
            let material = match scene.materials.get(mesh.material) {
                Some(m) => m,
                None => continue,
            };
            rpass.set_bind_group(1, Some(&material.bind_group), &[]);
            rpass.set_bind_group(3, Some(scene.morph_bind_group(i)), &[]);
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
        // Transparent meshes (face decals, capes, cloth) are drawn after the
        // opaque geometry so their translucent parts blend over it instead of
        // cutting through it.
        if !wireframe {
            rpass.set_pipeline(&self.transparent_pipeline);
        }
        for (i, mesh) in scene.meshes.iter().enumerate() {
            if !mesh.transparent {
                continue;
            }
            if !cull_mesh(mesh, cull) {
                continue;
            }
            let material = match scene.materials.get(mesh.material) {
                Some(m) => m,
                None => continue,
            };
            rpass.set_bind_group(1, Some(&material.bind_group), &[]);
            rpass.set_bind_group(3, Some(scene.morph_bind_group(i)), &[]);
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }
}

/// True if the mesh should be drawn. Without a cull sphere every mesh draws;
/// with one, the mesh draws only when its bounds sphere overlaps the cull
/// sphere.
fn cull_mesh(mesh: &crate::scene::GpuMesh, cull: Option<&(glam::Vec3, f32)>) -> bool {
    let Some(&(center, radius)) = cull else {
        return true;
    };
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
                array_stride: 40,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![
                    0 => Float32x3,
                    1 => Float32x3,
                    2 => Float32x2,
                    3 => Unorm8x4,
                    4 => Uint8x4,
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
            depth_compare: Some(wgpu::CompareFunction::Less),
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
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

fn make_grid(device: &wgpu::Device, radius: f32) -> GridData {
    let extent = (radius * 1.5).ceil() as i32;
    let mut verts: Vec<LineVertex> = Vec::new();
    let mut idx: Vec<u16> = Vec::new();

    let grid_col = [0.30, 0.30, 0.32, 1.0];
    let axis_x = [0.90, 0.25, 0.25, 1.0];
    let axis_y = [0.25, 0.85, 0.30, 1.0];
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
        let col = if i == 0 { axis_x } else { grid_col };
        push_line(&mut verts, &mut idx, [f, 0.0, -e], [f, 0.0, e], col);
        let col = if i == 0 { axis_z } else { grid_col };
        push_line(&mut verts, &mut idx, [-e, 0.0, f], [e, 0.0, f], col);
    }
    push_line(&mut verts, &mut idx, [0.0, 0.0, 0.0], [0.0, e, 0.0], axis_y);

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
