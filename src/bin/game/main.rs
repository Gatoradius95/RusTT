//! Gameplay test: walk a minifig around the TCS hub (Mos Eisley cantina).
//!
//! Reuses the viewer's rendering stack via `#[path]` includes so shader /
//! renderer changes benefit both binaries. The player is `ANAKIN_PADAWAN_PC`,
//! driven with WASD on a third-person orbit camera and posed with the
//! character's own WALK / IDLE animations.

#![allow(dead_code)]

#[path = "../viewer/camera.rs"]
mod camera;
#[path = "../viewer/renderer.rs"]
mod renderer;
#[path = "../viewer/scene.rs"]
mod scene;

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::Window;

use rustt::an3::An3;
use rustt::ghg::Parsed;
use rustt::map::Map;

use camera::OrbitCamera;
use renderer::GpuRenderer;
use scene::GpuScene;

const UP: Vec3 = Vec3::Y;

/// Player spawn: open main-room floor near the cantina bar. (The MAINROOM
/// trigger center sits under a low bulkhead, which forced the chase camera
/// into a close-up; the AI locator BARMAN_1 is at (-25.17, -48.22).)
const SPAWN: Vec3 = Vec3::new(-26.0, 0.1, -49.5);

/// Walk speed in world units per second. Ground truth from ANAKIN_JEDI.TXT
/// (`walk_speed=0.6`; `run_speed=1.2` if a run is ever added).
const WALK_SPEED: f32 = 0.6;
/// AN3 frames advanced per real second while walking (`fpsec=30.0` on the
/// "walk" action in ANAKIN_JEDI.TXT).
const WALK_FPS: f32 = 30.0;
/// Camera height above the player's feet.
const CAM_EYE: f32 = 0.9;
/// Chase-cam distance from the player.
const CAM_DIST: f32 = 4.5;
/// Gap kept between the chase camera and any wall/ceiling it collides with.
const CAM_MARGIN: f32 = 0.25;
/// Closest the chase camera is allowed to get (avoids clipping into the player).
const CAM_MIN: f32 = 0.6;
/// Radius (world units) of the sphere around the player within which map
/// geometry is drawn. Meshes beyond this are skipped, so only the current
/// room renders instead of the entire hub (the large framerate drain).
const CULL_RADIUS: f32 = 20.0;
/// Uniform scale applied to the minifig model. ANAKIN_JEDI.TXT says
/// `scale=1.0` — the source model is already authored at the in-game size
/// (collision cylinder maxy=0.42 matches the model's native ~0.41 height).
const PLAYER_SCALE: f32 = 1.0;
/// Extra yaw (radians) applied to the model. The source model's face points
/// along its -Z axis (glTF convention), so this PI turns the face away from
/// the camera to match the direction of travel (which the walk logic runs
/// along +Z of the arc). Remove/empty the constant if the model's forward is
/// verified to be the opposite.
const PLAYER_FACE_OFFSET: f32 = std::f32::consts::PI;

struct Input {
    keys: HashSet<KeyCode>,
    left_down: bool,
    last_mouse: Option<(f64, f64)>,
    drag: (f32, f32),
    wheel: f32,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            keys: HashSet::new(),
            left_down: false,
            last_mouse: None,
            drag: (0.0, 0.0),
            wheel: 0.0,
        }
    }
}

struct Player {
    pos: Vec3,
    yaw: f32,
    /// Seconds since the last animation frame step.
    anim_clock: f32,
    moving: bool,
}

/// A map mesh's triangles plus a bounding sphere, for CPU raycasts (camera
/// collision). The sphere lets the per-frame raycast skip nearly all meshes.
struct ColMesh {
    center: Vec3,
    radius: f32,
    tris: Vec<[[f32; 3]; 3]>,
}

struct AppWindow {
    window: Arc<Window>,
    gpu: GpuRenderer,
    camera: OrbitCamera,
    input: Input,
    exit_requested: bool,
    player: Player,
    player_scene: GpuScene,
    anims: Vec<An3>,
    anim_parents: Vec<i32>,
    rest_locals: Vec<Mat4>,
    last_frame: Instant,
    /// Frame at which to save a screenshot, then quit (`--shot`).
    shot_frame: Option<u32>,
    shot_path: Option<String>,
    frames_done: u32,
    hide_player: bool,
    no_input: bool,
    player_only: bool,
    /// Hide the player once `frames_done` exceeds this (same-process A/B).
    hide_after: Option<u32>,
    shot2_frame: Option<u32>,
    shot2_path: Option<String>,
    spawn_y: f32,
    /// Collision triangles of the map (camera raycasts).
    col_meshes: Vec<ColMesh>,
    /// `--walk`: force the player to walk forward (screenshot facing test).
    walk_test: bool,
}

impl AppWindow {
    fn update(&mut self, dt: f32) {
        // With --no-input the camera stays fixed and the player never moves
        // (used for reproducible screenshot A/B comparisons).
        if !self.no_input {
            // Orbit + zoom the chase camera with the mouse.
            if self.input.left_down {
                self.camera.orbit(self.input.drag.0, self.input.drag.1);
            }
            if self.input.wheel != 0.0 {
                self.camera.zoom(self.input.wheel);
            }
            self.input.drag = (0.0, 0.0);
            self.input.wheel = 0.0;

            // Forward/strafe input is relative to the camera's facing, like a
            // standard third-person camera.
            let cam_pos = self.camera.position();
            let mut fwd = self.camera.target - cam_pos;
            fwd.y = 0.0;
            fwd = if fwd.length_squared() < 1e-8 {
                Vec3::NEG_Z
            } else {
                fwd.normalize()
            };
            let right = fwd.cross(UP);

            let (mut mx, mut mz) = (0.0f32, 0.0f32);
            if self.input.keys.contains(&KeyCode::KeyW) {
                mx += fwd.x;
                mz += fwd.z;
            }
            if self.input.keys.contains(&KeyCode::KeyS) {
                mx -= fwd.x;
                mz -= fwd.z;
            }
            if self.input.keys.contains(&KeyCode::KeyD) {
                mx += right.x;
                mz += right.z;
            }
            if self.input.keys.contains(&KeyCode::KeyA) {
                mx -= right.x;
                mz -= right.z;
            }
            let moving = mx != 0.0 || mz != 0.0;
            if moving {
                let dir = Vec3::new(mx, 0.0, mz).normalize();
                self.player.pos += dir * WALK_SPEED * dt;
                // `yaw` is the direction of travel; the model's backwards-face
                // convention is corrected by PLAYER_FACE_OFFSET at render time.
                self.player.yaw = dir.x.atan2(dir.z);
            }
            self.player.moving = moving;
        }

        // `--walk` screenshot aid: march straight away from the camera without
        // any input, so facing-while-walking can be verified headlessly.
        if self.walk_test {
            let cam_pos = self.camera.position();
            let mut fwd = self.camera.target - cam_pos;
            fwd.y = 0.0;
            if fwd.length_squared() > 1e-8 {
                let fwd = fwd.normalize();
                self.player.pos += fwd * WALK_SPEED * dt;
                self.player.yaw = fwd.x.atan2(fwd.z);
                self.player.moving = true;
            }
        }

        // Chase camera: orbit the player at eye height, pulled in front of the
        // first wall/ceiling along the view ray so the camera never leaves the
        // room (the hub rooms are small; an unclamped orbit ends up above the
        // roof and the player appears "out of bounds" under it).
        self.camera.target = self.player.pos + Vec3::new(0.0, CAM_EYE, 0.0);
        let (sp, cp) = self.camera.pitch.sin_cos();
        let (sy, cy) = self.camera.yaw.sin_cos();
        let back = Vec3::new(cp * cy, sp, cp * sy);
        let clear = ray_hit_dist(&self.col_meshes, self.camera.target, back, CAM_DIST)
            .unwrap_or(CAM_DIST);
        self.camera.distance = (clear - CAM_MARGIN).clamp(CAM_MIN, CAM_DIST);

        // Step the animation playhead.
        let idx = if self.player.moving { 0 } else { 1 }; // WALK, IDLE
        let an3 = &self.anims[idx.min(self.anims.len() - 1)];
        self.player.anim_clock += dt * WALK_FPS;
        let last = an3.num_frames.saturating_sub(1) as f32;
        if self.player.anim_clock > last {
            self.player.anim_clock = 0.0;
        }
        let world_frame = an3.remap_playhead(self.player.anim_clock);
        if let Ok(worlds) = an3.bone_worlds(&self.anim_parents, &self.rest_locals, world_frame) {
            self.player_scene.set_skin_mats(&self.gpu.queue, &worlds);
        }

        self.gpu.update_camera(&self.camera);
    }

    fn render(&mut self, view: &wgpu::TextureView) {
        let hide_now = self
            .hide_after
            .map(|n| self.frames_done > n)
            .unwrap_or(false);
        let draw_player = !self.hide_player && !hide_now;

        // The camera uniform is written via `queue.write_buffer`, which is
        // enqueued before the next submit runs. Each model therefore needs its
        // own submit so its draws read the matrix written for them, not the
        // last one written to the shared buffer.
        if !self.player_only {
            self.gpu.write_camera(Mat4::IDENTITY);
            let mut encoder = self
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("map pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.03,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.gpu.depth_view(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                self.gpu.draw_scene_meshes_culled(
                    &mut rpass,
                    &self.gpu.scene,
                    false,
                    Some(&(self.player.pos, CULL_RADIUS)),
                );
            }
            self.gpu.queue.submit(Some(encoder.finish()));
        }

        // Second pass: the animated minifig, blended over the map with the
        // depth buffer preserved so it occludes and is occluded correctly.
        if draw_player {
            self.gpu.write_camera(
                Mat4::from_translation(self.player.pos)
                    * Mat4::from_rotation_y(self.player.yaw + PLAYER_FACE_OFFSET)
                    * Mat4::from_scale(Vec3::splat(PLAYER_SCALE)),
            );
            let mut encoder = self
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("player pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if self.player_only {
                                wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 1.0,
                                    a: 1.0,
                                })
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.gpu.depth_view(),
                        depth_ops: Some(wgpu::Operations {
                            load: if self.player_only {
                                wgpu::LoadOp::Clear(1.0)
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                self.gpu
                    .draw_scene_meshes(&mut rpass, &self.player_scene, false);
            }
            self.gpu.queue.submit(Some(encoder.finish()));
        }
    }

    /// Copy the rendered swapchain image to a PNG (uses the surface texture
    /// before present; the surface must be configured with COPY_SRC).
    fn capture_surface(&self, frame: &wgpu::SurfaceTexture, path: &str) -> Result<()> {
        let w = self.gpu.config.width.max(1);
        let h = self.gpu.config.height.max(1);
        let bytes_per_row = w * 4;
        let buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot staging"),
            size: bytes_per_row as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll for screenshot");
        rx.recv().context("screenshot map_async")??;

        let data = slice.get_mapped_range();
        let mut rgba = vec![0u8; (bytes_per_row as usize) * h as usize];
        match self.gpu.config.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                rgba.copy_from_slice(&data);
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                for (dst, px) in rgba.chunks_exact_mut(4).zip(data.chunks_exact(4)) {
                    dst[0] = px[2];
                    dst[1] = px[1];
                    dst[2] = px[0];
                    dst[3] = px[3];
                }
            }
            f => anyhow::bail!("unsupported surface format {f:?} for screenshot"),
        }
        drop(data);

        let file = std::fs::File::create(path).with_context(|| format!("creating {path}"))?;
        let mut enc = png::Encoder::new(file, w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().context("png header")?;
        writer.write_image_data(&rgba).context("png write")?;
        Ok(())
    }
}

struct App {
    map: Option<Map<'static>>,
    player_parsed: Option<Parsed<'static>>,
    walk: Option<An3>,
    idle: Option<An3>,
    window: Option<AppWindow>,
    shot_frame: Option<u32>,
    shot_path: Option<String>,
    hide_player: bool,
    no_input: bool,
    player_only: bool,
    hide_after: Option<u32>,
    shot2_frame: Option<u32>,
    shot2_path: Option<String>,
    spawn_y: f32,
    col_meshes: Vec<ColMesh>,
    walk_test: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        match self.create_window(event_loop) {
            Ok(w) => self.window = Some(w),
            Err(e) => {
                eprintln!("failed to create game window: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_mut() else {
            return;
        };
        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => window.gpu.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if let Key::Named(NamedKey::Escape) = event.logical_key {
                    if event.state.is_pressed() {
                        event_loop.exit();
                    }
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            window.input.keys.insert(code);
                        }
                        ElementState::Released => {
                            window.input.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match (button, state) {
                (MouseButton::Left, ElementState::Pressed) => {
                    window.input.left_down = true;
                }
                (MouseButton::Left, ElementState::Released) => {
                    window.input.left_down = false;
                    window.input.last_mouse = None;
                }
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                if let Some((px, py)) = window.input.last_mouse {
                    let dx = (x - px) as f32;
                    let dy = (y - py) as f32;
                    if window.input.left_down {
                        window.input.drag.0 += dx;
                        window.input.drag.1 += dy;
                    }
                }
                window.input.last_mouse = Some((x, y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let v = match delta {
                    MouseScrollDelta::LineDelta(_, v) => *v,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y / 40.0) as f32,
                };
                window.input.wheel += v;
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - window.last_frame).as_secs_f32().min(0.1);
                window.last_frame = now;
                if window.exit_requested {
                    event_loop.exit();
                    return;
                }
                let frame = match window.gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        window
                            .gpu
                            .resize(window.gpu.config.width, window.gpu.config.height);
                        return;
                    }
                    other => {
                        eprintln!("get_current_texture error: {other:?}");
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                window.update(dt);
                window.render(&view);
                window.frames_done += 1;
                if window
                    .shot_frame
                    .map(|n| window.frames_done >= n)
                    .unwrap_or(false)
                {
                    let path = window.shot_path.clone().unwrap_or_default();
                    match window.capture_surface(&frame, &path) {
                        Ok(()) => eprintln!("screenshot saved to {path}"),
                        Err(e) => eprintln!("screenshot failed: {e:#}"),
                    }
                    window.shot_frame = None;
                    if window.shot2_path.is_none() {
                        frame.present();
                        event_loop.exit();
                        return;
                    }
                }
                if let Some(p2) = window
                    .shot2_frame
                    .and_then(|n| {
                        if window.frames_done >= n {
                            window.shot2_path.clone()
                        } else {
                            None
                        }
                    })
                {
                    match window.capture_surface(&frame, &p2) {
                        Ok(()) => eprintln!("screenshot saved to {p2}"),
                        Err(e) => eprintln!("screenshot failed: {e:#}"),
                    }
                    frame.present();
                    event_loop.exit();
                    return;
                }
                frame.present();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_mut() {
            window.window.request_redraw();
        }
    }
}

impl App {
    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> Result<AppWindow> {
        let size = LogicalSize::new(1280.0, 800.0);
        let attributes = Window::default_attributes()
            .with_title("rustt game - walk the cantina")
            .with_inner_size(size);
        let window = Arc::new(event_loop.create_window(attributes)?);

        let map = self.map.as_ref().expect("map loaded");
        let mut gpu = GpuRenderer::new_map(event_loop, &window, map, "hub")?;
        gpu.show_grid = false;
        // Allow the swapchain image to be read back for --shot screenshots.
        gpu.config.usage |= wgpu::TextureUsages::COPY_SRC;
        gpu.surface.configure(&gpu.device, &gpu.config);

        let is_srgb = gpu.config.format.is_srgb();
        let player_scene = GpuScene::new(
            &gpu.device,
            &gpu.queue,
            self.player_parsed.as_ref().expect("player model loaded"),
            is_srgb,
            // ANAKIN_PADAWAN's layers are inherited from ANAKIN_JEDI.TXT
            // (special set).
            &[0, 1, 5],
        );
        eprintln!(
            "map bounds: center {:?} radius {:.3} | player bounds: center {:?} radius {:.3}",
            gpu.scene.bounds.center,
            gpu.scene.bounds.radius,
            player_scene.bounds.center,
            player_scene.bounds.radius
        );

        let player = Player {
            pos: Vec3::new(SPAWN.x, self.spawn_y, SPAWN.z),
            yaw: 0.0,
            anim_clock: 0.0,
            moving: false,
        };

        let mut camera = OrbitCamera::default();
        camera.target = player.pos + Vec3::new(0.0, CAM_EYE, 0.0);
        camera.distance = CAM_DIST;
        // Face the open floor/bar side; the default yaw puts the camera under
        // the low booth overhang behind the spawn and the collision pull-in
        // collapses it to a close-up.
        camera.yaw = 0.7 + std::f32::consts::PI;
        // The cantina ceiling is only ~1.5 units up, so keep the chase camera
        // nearly level with the player; the default orbit pitch would put it
        // above the ceiling and the collision pull-in would collapse it to a
        // close-up of the head.
        camera.pitch = 0.12;

        let parsed = self.player_parsed.as_ref().expect("player model loaded");
        let walk = self.walk.take().expect("WALK anim loaded");
        let n = walk.num_bones;
        let anim_parents: Vec<i32> = (0..n)
            .map(|i| parsed.bones.get(i).map(|b| b.parent.min(n as i32 - 1)).unwrap_or(-1))
            .collect();
        let rest_locals: Vec<Mat4> = parsed.bones.iter().map(|b| b.local).collect();

        Ok(AppWindow {
            window,
            gpu,
            camera,
            input: Input::default(),
            exit_requested: false,
            player,
            player_scene,
            anims: vec![walk, self.idle.take().expect("IDLE anim loaded")],
            anim_parents,
            rest_locals,
            last_frame: Instant::now(),
            shot_frame: self.shot_frame,
            shot_path: self.shot_path.clone(),
            frames_done: 0,
            hide_player: self.hide_player,
            no_input: self.no_input,
            player_only: self.player_only,
            hide_after: self.hide_after,
            shot2_frame: self.shot2_frame,
            shot2_path: self.shot2_path.clone(),
            spawn_y: self.spawn_y,
            col_meshes: std::mem::take(&mut self.col_meshes),
            walk_test: self.walk_test,
        })
    }
}

fn load_an3(path: &Path) -> Result<An3> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let anim = An3::parse(&data).with_context(|| format!("parsing {}", path.display()))?;
    println!(
        "load {}: {} bones, {} frames, {} moving channels",
        path.display(),
        anim.num_bones,
        anim.num_frames,
        anim.num_moving
    );
    Ok(anim)
}

/// Lowest map surface hit by a ray cast straight down from `(x, high, z)`.
/// The main cantina floor is the lowest hit (y≈0 — confirmed by the AI
/// locators BARMAN_1 / JABBA_1 / MAINROOMIDLE_*, which all sit at y≈0).
/// The higher hit (y≈1.19) is the room's ceiling/roof; standing there put
/// the player on top of the building, out of bounds.
fn floor_height_at(map: &Map<'_>, x: f32, z: f32, high: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for md in rustt::mapmesh::expand_all(map) {
        for tri in md.idx.chunks_exact(3) {
            let a = md.pos[tri[0] as usize];
            let b = md.pos[tri[1] as usize];
            let c = md.pos[tri[2] as usize];
            let e0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                e0[1] * e1[2] - e0[2] * e1[1],
                e0[2] * e1[0] - e0[0] * e1[2],
                e0[0] * e1[1] - e0[1] * e1[0],
            ];
            if n[1].abs() < 1e-8 {
                continue;
            }
            // Plane intersection with the vertical ray x=x, z=z. `t` is the
            // downward distance from the ray origin (p = [x, high - t, z]).
            let t = (n[0] * (x - a[0]) + n[1] * (high - a[1]) + n[2] * (z - a[2])) / n[1];
            if !(t >= 0.0) || t > high {
                continue;
            }
            let p = [x, high - t, z];
            // Barycentric containment test.
            let v0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let v2 = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
            let d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
            let d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
            let d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
            let d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
            let d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];
            let denom = d00 * d11 - d01 * d01;
            if denom.abs() < 1e-12 {
                continue;
            }
            let vv = (d11 * d20 - d01 * d21) / denom;
            let ww = (d00 * d21 - d01 * d20) / denom;
            if vv >= 0.0 && ww >= 0.0 && vv + ww <= 1.0 {
                let h = p[1];
                if best.map_or(true, |b| h < b) {
                    best = Some(h);
                }
            }
        }
    }
    best
}

/// Nearest distance from `origin` along unit `dir` (up to `max_t`) to any
/// collision triangle, via Möller–Trumbore. Meshes whose bounding sphere the
/// ray segment cannot touch are skipped, so this stays cheap per frame.
fn ray_hit_dist(meshes: &[ColMesh], origin: Vec3, dir: Vec3, max_t: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for m in meshes {
        // Ray-segment vs bounding sphere reject.
        let oc = origin - m.center;
        let b = oc.dot(dir);
        let c = oc.dot(oc) - m.radius * m.radius;
        let disc = b * b - c;
        if disc < 0.0 {
            continue;
        }
        let sq = disc.sqrt();
        if -b - sq > max_t || -b + sq < 0.0 {
            continue;
        }
        for tri in &m.tris {
            let a = Vec3::from(tri[0]);
            let e0 = Vec3::from(tri[1]) - a;
            let e1 = Vec3::from(tri[2]) - a;
            let p = dir.cross(e1);
            let det = e0.dot(p);
            if det.abs() < 1e-8 {
                continue;
            }
            let inv = 1.0 / det;
            let s = origin - a;
            let u = s.dot(p) * inv;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(e0);
            let v = dir.dot(q) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = e1.dot(q) * inv;
            if t > 0.0 && t <= max_t && best.map_or(true, |bb| t < bb) {
                best = Some(t);
            }
        }
    }
    best
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dump_only = args.iter().any(|a| a == "--dump-only");
    let shot_path = args
        .iter()
        .position(|a| a == "--shot")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let shot_frame = args
        .iter()
        .position(|a| a == "--frame")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let hide_player = args.iter().any(|a| a == "--hide-player");
    let no_input = args.iter().any(|a| a == "--no-input");
    let player_only = args.iter().any(|a| a == "--player-only");
    let walk_test = args.iter().any(|a| a == "--walk");
    let hide_after = args
        .iter()
        .position(|a| a == "--hide-after")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let shot2_path = args
        .iter()
        .position(|a| a == "--shot2")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let shot2_frame = if shot2_path.is_some() {
        Some(
            args.iter()
                .position(|a| a == "--shot2-frame")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(30),
        )
    } else {
        None
    };
    let py_override = args
        .iter()
        .position(|a| a == "--py")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let positional: Vec<String> = {
        let mut out = Vec::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if a.starts_with("--") {
                // Skip the one-argument value of these flags.
                if matches!(
                    a.as_str(),
                    "--shot" | "--shot2" | "--frame" | "--shot2-frame" | "--hide-after" | "--py"
                ) {
                    let _ = it.next();
                }
            } else {
                out.push(a.clone());
            }
        }
        out
    };
    let map_path = positional
        .first()
        .cloned()
        .unwrap_or_else(|| "backup/LEVELS/MAP/MAP/MAP_PC.GSC".to_string());
    let char_path = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG".to_string());

    let map_bytes = std::fs::read(&map_path).with_context(|| format!("reading {map_path}"))?;
    let map_bytes: &'static [u8] = Box::leak(map_bytes.into_boxed_slice());
    let map = rustt::map::parse(map_bytes).with_context(|| format!("parsing {map_path}"))?;
    println!(
        "{}: {} render parts, {} meshes, {} materials, {} textures",
        map_path,
        map.render_parts.len(),
        map.meshes.len(),
        map.materials.len(),
        map.textures.len()
    );

    let ai_path = map_path.replace("MAP_PC.GSC", "AI/MAP.AI2");
    let ai = rustt::ai2::parse_file(&ai_path)?;
    println!(
        "{}: version {}, {} locators, {} creatures",
        ai_path,
        ai.version,
        ai.locators.len(),
        ai.creatures.len()
    );
    if let Some(main) = ai.triggers.iter().find(|t| t.name == "MAINROOM") {
        println!("spawn: MAINROOM at ({:.2}, {:.2}, {:.2})", main.pos.x, main.pos.y, main.pos.z);
    }
    let spawn_y = py_override.unwrap_or_else(|| {
        floor_height_at(&map, SPAWN.x, SPAWN.z, 1000.0)
            .map(|h| h + 0.02)
            .unwrap_or(SPAWN.y)
    });
    println!("spawn y = {spawn_y:.3} (from floor at SPAWN or --py override)");

    // Collision triangles for the camera raycast, kept per-mesh so each mesh
    // gets a bounding sphere the raycast can cheaply reject against.
    let mut col_meshes: Vec<ColMesh> = Vec::new();
    for md in rustt::mapmesh::expand_all(&map) {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for p in &md.pos {
            lo = lo.min(Vec3::from(*p));
            hi = hi.max(Vec3::from(*p));
        }
        let center = (lo + hi) * 0.5;
        let radius = (hi - lo).length() * 0.5;
        let tris = md
            .idx
            .chunks_exact(3)
            .map(|t| {
                [
                    md.pos[t[0] as usize],
                    md.pos[t[1] as usize],
                    md.pos[t[2] as usize],
                ]
            })
            .collect();
        col_meshes.push(ColMesh { center, radius, tris });
    }

    let char_bytes = std::fs::read(&char_path).with_context(|| format!("reading {char_path}"))?;
    let char_bytes: &'static [u8] = Box::leak(char_bytes.into_boxed_slice());
    let parsed = rustt::ghg::parse(char_bytes).with_context(|| format!("parsing {char_path}"))?;
    println!(
        "{}: {} render items, {} materials, {} textures, {} bones",
        char_path,
        parsed.render.len(),
        parsed.materials.len(),
        parsed.textures.len(),
        parsed.bones.len()
    );

    let char_dir = Path::new(&char_path).parent().expect("character folder");
    let walk = load_an3(&char_dir.join("WALK.AN3"))?;
    let idle = load_an3(&char_dir.join("IDLE.AN3"))?;

    if dump_only {
        println!("load path OK");
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        map: Some(map),
        player_parsed: Some(parsed),
        walk: Some(walk),
        idle: Some(idle),
        window: None,
        shot_frame: if shot_path.is_some() { Some(shot_frame) } else { None },
        shot_path,
        hide_player,
        no_input,
        player_only,
        hide_after,
        shot2_frame,
        shot2_path,
        spawn_y,
        col_meshes,
        walk_test,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
