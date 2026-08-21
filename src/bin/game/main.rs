//! Gameplay test: walk a minifig around the TCS hub (Mos Eisley cantina).
//!
//! Reuses the viewer's rendering stack via `#[path]` includes so shader /
//! renderer changes benefit both binaries. The player is `ANAKIN_PADAWAN_PC`,
//! driven with WASD on a third-person orbit camera and posed with the
//! character's own IDLE / WALK / RUN animations.
//!
//! Movement reproduces the exe's character model (see `research/ghidra docs.md`
//! §7b): a desired speed picked from input (keyboard = full deflection = run),
//! ramped toward the target with `acceleration`, a walk-before-run phase after
//! each direction change (`move_delay`), and a reversal slowdown
//! (`backwards_factor`) while the facing turns around. Because
//! run_speed × backwards_factor = walk_speed, a reversal settles into a walk
//! and then visibly breaks into a run again once the turn completes. The exe
//! flips the facing a full 180° on a new reversal (`FUN_005b3620` at
//! 0x004ad601) rather than sweeping it, and crossfades between animation clips
//! over `blend_in`/`blend_out` (0.2 s) via a two-buffer blend
//! (`NuAnimBuffBlendTwo`, `FUN_005ebb90`).

#![allow(dead_code)]

#[path = "../viewer/camera.rs"]
mod camera;
#[path = "../viewer/imgui_state.rs"]
mod imgui_state;
#[path = "../viewer/renderer.rs"]
mod renderer;
#[path = "../viewer/scene.rs"]
mod scene;
mod particles;

use std::collections::{HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::{Mat4, Vec3};
use imgui::Condition;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::Window;

use rustt::an3::{blended_bone_worlds, An3};
use rustt::bsa::Bsa;
use rustt::ghg::Parsed;
use rustt::giz::BuilditRuntime;
use rustt::map::Map;
use rustt::rtl::{self, RtlLight};

use camera::OrbitCamera;
use renderer::GpuRenderer;
use scene::GpuScene;

const UP: Vec3 = Vec3::Y;

/// Player spawn: open main-room floor near the cantina bar. (The MAINROOM
/// trigger center sits under a low bulkhead, which forced the chase camera
/// into a close-up; the AI locator BARMAN_1 is at (-25.17, -48.22).)
const SPAWN: Vec3 = Vec3::new(-26.0, 0.1, -49.5);

/// Walk speed in world units per second. Ground truth from ANAKIN_JEDI.TXT
/// (`walk_speed=0.6`).
const WALK_SPEED: f32 = 0.6;
/// Run speed in world units per second (`run_speed=1.2`). The desired speed
/// with a fully-deflected stick (keyboard input reads as full deflection).
const RUN_SPEED: f32 = 1.2;
/// Tiptoe speed (`tiptoe_speed=0.1095`) — the slowest movement class.
const TIPTOE_SPEED: f32 = 0.1095;
/// How fast the actual speed ramps toward the desired speed, in units/s²
/// (`acceleration=10`).
const ACCELERATION: f32 = 10.0;
/// Desired-speed multiplier while the facing opposes the movement direction
/// (`backwards_factor` keyword; not present in ANAKIN_JEDI.TXT so 0.5 is the
/// classic default). run_speed × backwards_factor = walk_speed, which is why
/// a reversal visibly settles into a walk.
const BACKWARDS_FACTOR: f32 = 0.5;
/// Seconds spent at walk speed before breaking into a run after the movement
/// direction changes (`move_delay=0.2` in ANAKIN_JEDI.TXT). The PC build has
/// no such keyword — this models the observed console behaviour.
const MOVE_DELAY: f32 = 0.2;
/// Facing rotation rate in radians per second while turning toward the
/// movement direction. Reversals do NOT sweep at this rate: the exe flips the
/// facing a full 180° instantly (`FUN_005b3620` at 0x004ad601).
const TURN_RATE: f32 = 4.0;
/// Crossfade duration between animation clips, in seconds (`blend_in`/
/// `blend_out` on the walk/run actions in ANAKIN_JEDI.TXT). The game evaluates
/// both clips into two buffers and ramps a blend weight over this time
/// (`NuAnimBuffBlendTwo`, `FUN_005ebb90`).
const BLEND_TIME: f32 = 0.2;
/// `dot(facing, move_dir)` below which the character counts as reversing and
/// applies `BACKWARDS_FACTOR` (~110°).
const REVERSE_DOT: f32 = -0.35;
/// Speed below which the IDLE clip plays.
const IDLE_EPS: f32 = 0.03;
/// Speed above which the RUN clip plays (below it the WALK clip plays).
const RUN_SWITCH: f32 = 0.8;
/// `dot(last_dir, dir)` below which a direction change restarts the
/// walk-before-run phase (~30°).
const DIR_CHANGE_DOT: f32 = 0.87;
/// AN3 frames advanced per real second while walking (`fpsec=30.0` on the
/// "walk" and "run" actions in ANAKIN_JEDI.TXT — run strides are longer in the
/// clip, so both play at 30 fps).
const WALK_FPS: f32 = 30.0;
/// Height (above the feet) the chase camera LOOKS AT — the minifig's torso.
/// The minifig is only 0.42 tall, so aiming at its head would make the camera
/// float above it.
const CAM_TARGET: f32 = 0.28;
/// Desired height of the chase camera above the floor. The real game's hub
/// camera uses `cam_height_above_terrain 0.8` (MAP.TXT).
const CAM_HEIGHT: f32 = 0.8;
/// Chase-cam distance from the player. The real game uses `cam_dist_to_target
/// 2` (MAP.TXT) — a closer, more intimate third-person view.
const CAM_DIST: f32 = 2.0;
/// Gap kept between the chase camera and any wall/ceiling it collides with.
const CAM_MARGIN: f32 = 0.25;
/// Max distance of the picker rays (char forward / camera view). The cantina
/// rooms are ~6-8 units across, so this reaches the far wall comfortably
/// while keeping the per-frame triangle tests to span hits.
const PICK_DIST: f32 = 12.0;
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
    just_pressed: HashSet<KeyCode>,
    left_down: bool,
    last_mouse: Option<(f64, f64)>,
    drag: (f32, f32),
    wheel: f32,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            keys: HashSet::new(),
            just_pressed: HashSet::new(),
            left_down: false,
            last_mouse: None,
            drag: (0.0, 0.0),
            wheel: 0.0,
        }
    }
}

/// In-flight crossfade between two animation clips. `from` is the clip fading
/// out, `to` the clip fading in; `clock` seconds have elapsed of `duration`.
struct Blend {
    from: usize,
    to: usize,
    clock: f32,
    duration: f32,
}

struct Player {
    pos: Vec3,
    /// Facing yaw (radians). Rotates smoothly toward the movement direction;
    /// a reversal snaps a full 180° (`FUN_005b3620`).
    yaw: f32,
    /// Actual speed in units/s, ramped toward the desired speed by
    /// `ACCELERATION` (drives both the walk/run pose and the position update).
    speed: f32,
    /// Seconds since the movement direction last changed (drives the
    /// walk-before-run phase).
    dir_clock: f32,
    /// Last movement direction (zero when not moving), for change detection.
    last_dir: Vec3,
    /// True while the facing opposes the movement direction (reversing).
    was_reversing: bool,
    /// Seconds since the last animation frame step.
    anim_clock: f32,
    /// Currently-playing clip index (0 idle / 1 walk / 2 run). During a blend
    /// this stays the fading-out clip until the crossfade completes.
    clip: usize,
    /// Active crossfade between clips, if any.
    blend: Option<Blend>,
}

/// A map mesh's triangles plus a bounding sphere, for CPU raycasts (camera
/// collision). The sphere lets the per-frame raycast skip nearly all meshes.
struct ColMesh {
    center: Vec3,
    radius: f32,
    tris: Vec<[[f32; 3]; 3]>,
}

/// Per-PART raycast data for the interactive picker window (the same
/// bounding-sphere + triangle-soup layout as `ColMesh`, but one entry per
/// render part instead of per mesh). A part is the drawable unit — it
/// references exactly one mesh + one material — so the picker can report
/// the identity of everything a ray passes through.
struct PartPick {
    part: usize,
    center: Vec3,
    radius: f32,
    tris: Vec<[[f32; 3]; 3]>,
}

/// Static identity of one render part, precomputed at load for the picker
/// window. `top_vcol` is the 3 most common vertex colours in RGBA8, on the
/// map's 0..127 baked-light scale — seeing whether a picked part's vertices
/// are dark (0, 49, 82 ...) or neutral 127 immediately explains its shading
/// (e.g. a non-prelit part with dark baked light renders much darker than a
/// neutral one).
struct PartInfo {
    part: usize,
    mesh: usize,
    material: usize,
    mat_id: i32,
    tex_id: i16,
    shader_defines: u32,
    lighting_stage: u8,
    lightmap_set_index: u8,
    top_vcol: Vec<([u8; 4], usize)>,
}

struct AppWindow {
    window: Arc<Window>,
    gpu: GpuRenderer,
    imgui: imgui_state::ImguiState,
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
    /// Per-part picker data (identities + raycast soups).
    pick_meshes: Vec<PartPick>,
    pick_info: Vec<PartInfo>,
    /// Per-frame picker data (index into pick_meshes, distance).
    /// `pick_char` is the ray from the character's chest along
    /// its facing; `pick_cam` the ray through the chase camera's view.
    pick_char: Vec<(usize, f32)>,
    pick_cam: Vec<(usize, f32)>,
    /// ONLY_PART isolation: the render parts to frame (None = normal play).
    only_part: Option<std::collections::HashSet<usize>>,
    /// Whether the player-radius cull sphere is active.
    cull_enabled: bool,
    /// `--walk`: force the player to walk forward (screenshot facing test).
    walk_test: bool,
    /// `--facecam`: override the chase camera to a tight face close-up.
    facecam: bool,
    /// `--morphs`: apply the sibling BSA facial blend shapes (viewer parity).
    morphs: bool,
    /// Loaded BSA for the current clip (facial blend shapes).
    bsa: Option<Bsa>,
    /// Per-channel BSA weights, evaluated each frame.
    bsa_weights: Vec<f32>,
    /// Rolling (x, y, z) position log shown in the HUD window.
    hud_history: VecDeque<(f32, f32, f32)>,
    /// 'F' key: show part info labels projected onto 3D models.
    show_part_labels: bool,
    /// All trigger volumes from the AI2 file.
    triggers: Vec<rustt::ai2::Trigger>,
    /// Name→index lookup for triggers (built from `triggers`).
    trigger_index: std::collections::HashMap<String, usize>,
    /// Names of triggers the player is currently inside.
    active_triggers: Vec<String>,
    /// Per-render-part room assignment (AI2 trigger index).
    part_rooms: Vec<Option<usize>>,
    /// Buildit runtime state machines (from GIZ data).
    buildit_runtimes: Vec<BuilditRuntime>,
    /// Current game time for jibber animation.
    game_time: f32,
    /// Maps buildit index → list of (sub_obj_index, Vec<GpuMesh index>).
    /// Built at load time so the update loop can write per-mesh transforms.
    buildit_mesh_map: Vec<Vec<(usize, Vec<usize>)>>,
    /// Cached GAME_CAM override parsed once at startup (target xyz, pitch, yaw, dist).
    game_cam: Option<[f32; 6]>,
    /// Reusable scratch set for room-based culling (avoids per-frame allocation).
    active_rooms: std::collections::HashSet<usize>,
    /// Index of the buildit the player is currently interacting with (None = idle).
    active_buildit: Option<usize>,
    /// Last diagnostic state_index (to detect changes for logging).
    last_diag_state: u8,
    /// Simple particle jibber overlay for buildit assembly animation.
    buildit_particles: particles::BuilditParticles,
    /// Scratch buffer for jibber per-mesh transform overrides.
    jibber_overrides: Vec<(usize, Mat4)>,
}

impl AppWindow {
    fn update(&mut self, dt: f32) {
        // 'C' toggles the player-radius cull sphere on/off.
        if self.input.just_pressed.contains(&KeyCode::KeyC) {
            self.cull_enabled = !self.cull_enabled;
            println!("cull: {}", if self.cull_enabled { "ON" } else { "OFF" });
        }
        // 'V' toggles FORCE_OPAQUE (alpha=1.0 on all fragments).
        if self.input.just_pressed.contains(&KeyCode::KeyV) {
            self.gpu.force_opaque = !self.gpu.force_opaque;
            println!("force_opaque: {}", if self.gpu.force_opaque { "ON" } else { "OFF" });
        }
        // 'P' toggles color correction (post-process curve approximating
        // the original D3D9 sRGB-space lighting look).
        if self.input.just_pressed.contains(&KeyCode::KeyP) {
            self.gpu.color_correct_enabled = !self.gpu.color_correct_enabled;
        }
        // 'O' toggles SO/room coloring: green = room geometry, yellow = SO entity.
        if self.input.just_pressed.contains(&KeyCode::KeyO) {
            self.gpu.so_coloring_enabled = !self.gpu.so_coloring_enabled;
            println!("so_coloring: {}", if self.gpu.so_coloring_enabled { "ON" } else { "OFF" });
        }
        if self.input.just_pressed.contains(&KeyCode::KeyF) {
            self.show_part_labels = !self.show_part_labels;
        }
        self.input.just_pressed.clear();

        // With --no-input the camera stays fixed and the player never moves
        // (used for reproducible screenshot A/B comparisons).
        let (mut mx, mut mz) = (0.0f32, 0.0f32);
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
            // Right is the camera's screen-right. The renderer uses a
            // left-handed view (glam `lh`), where screen-right = `up × fwd` —
            // the opposite of the right-handed `fwd × up`. Using the wrong
            // cross product mirrored strafing (A went right, D went left).
            let right = UP.cross(fwd);

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
        }

        // `--walk` screenshot aid: march straight away from the camera without
        // any input, so facing-while-walking can be verified headlessly.
        let mut dir = Vec3::new(mx, 0.0, mz);
        if self.walk_test {
            let cam_pos = self.camera.position();
            let mut fwd = self.camera.target - cam_pos;
            fwd.y = 0.0;
            if fwd.length_squared() > 1e-8 {
                dir = fwd.normalize();
            }
        }

        // --- Movement model (per exe §7b) ---
        // Desired speed from input magnitude (exe thresholds: <0.2 stop, <0.5
        // tiptoe, <0.8 walk, >=0.8 run). A key reads as full deflection (1.0),
        // so keyboard movement targets run speed.
        let input_on = dir.length_squared() > 1e-9;
        let mut target = 0.0f32;
        if input_on {
            let mag = dir.length();
            dir = dir.normalize();
            target = if mag < 0.2 {
                0.0
            } else if mag < 0.5 {
                TIPTOE_SPEED
            } else if mag < 0.8 {
                WALK_SPEED
            } else {
                RUN_SPEED
            };
            let facing = Vec3::new(self.player.yaw.sin(), 0.0, self.player.yaw.cos());
            // Reversing (input more than ~110° from the facing): apply the
            // backwards factor, which at the 0.5 default drops run speed down
            // to walk speed.
            let reversing = facing.dot(dir) < REVERSE_DOT;
            if reversing {
                target *= BACKWARDS_FACTOR;
                // The exe flips the facing a full 180° on a new reversal
                // (0x004ad601 → `FUN_005b3620`), it does not sweep the turn at
                // `TURN_RATE`. Snap it so an about-face reads instantly and
                // the character settles back into the walk phase.
                if !self.player.was_reversing {
                    self.player.yaw += std::f32::consts::PI;
                }
            }
            // A fresh direction (or completing a reversal) restarts the
            // walk-before-run phase.
            if self.player.last_dir.length_squared() < 1e-9
                || self.player.last_dir.dot(dir) < DIR_CHANGE_DOT
            {
                self.player.dir_clock = 0.0;
            }
            if !reversing && self.player.was_reversing {
                self.player.dir_clock = 0.0;
            }
            self.player.was_reversing = reversing;
            self.player.last_dir = dir;
        } else {
            self.player.dir_clock = 0.0;
            self.player.last_dir = Vec3::ZERO;
            self.player.was_reversing = false;
        }
        self.player.dir_clock += dt;
        if input_on && self.player.dir_clock < MOVE_DELAY && target > WALK_SPEED {
            target = WALK_SPEED;
        }

        // Ramp the actual speed toward the target with the character's
        // acceleration.
        let dv = ACCELERATION * dt;
        if self.player.speed < target {
            self.player.speed = (self.player.speed + dv).min(target);
        } else {
            self.player.speed = (self.player.speed - dv).max(target);
        }

        if input_on && self.player.speed > 1e-6 {
            self.player.pos += dir * self.player.speed * dt;
            // Track the movement direction at a bounded turn rate. A fresh
            // reversal already snapped the facing 180° above, so this only
            // closes the small residual angle (the snap put the facing within
            // ~70° of the input), which also clears the reversing state.
            let target_yaw = dir.x.atan2(dir.z);
            let mut da = target_yaw - self.player.yaw;
            while da > std::f32::consts::PI {
                da -= std::f32::consts::TAU;
            }
            while da < -std::f32::consts::PI {
                da += std::f32::consts::TAU;
            }
            let step = TURN_RATE * dt;
            self.player.yaw += da.clamp(-step, step);
        }

        // Chase camera: orbit the player at eye height, pulled in front of the
        // first wall/ceiling along the view ray so the camera never leaves the
        // room (the hub rooms are small; an unclamped orbit ends up above the
        // roof and the player appears "out of bounds" under it).
        self.camera.target = self.player.pos + Vec3::new(0.0, CAM_TARGET, 0.0);
        let (sp, cp) = self.camera.pitch.sin_cos();
        let (sy, cy) = self.camera.yaw.sin_cos();
        let back = Vec3::new(cp * cy, sp, cp * sy);
        let clear = ray_hit_dist(&self.col_meshes, self.camera.target, back, CAM_DIST)
            .unwrap_or(CAM_DIST);
        self.camera.distance = (clear - CAM_MARGIN).clamp(CAM_MIN, CAM_DIST);

        // `--facecam`: override the chase camera to a tight close-up of the
        // character's face (re-applied after the collision clamp above).
        if self.facecam {
            self.camera.target = self.player.pos + Vec3::new(0.0, 0.33, 0.0);
            self.camera.distance = 0.12;
            self.camera.yaw = -std::f32::consts::FRAC_PI_2;
            self.camera.pitch = 0.0;
        }

        // Trigger volume detection: test player AABB against AI2 triggers.
        self.active_triggers.clear();
        for t in &self.triggers {
            let diff = self.player.pos - t.pos;
            if diff.x.abs() < t.half_size.x
                && diff.y.abs() < t.half_size.y
                && diff.z.abs() < t.half_size.z
            {
                self.active_triggers.push(t.name.clone());
            }
        }

        // Buildit state machine: tracks assembly animation per the engine's
        // giz_buildit_trigger/giz_buildit_activate/giz_buildit_display_current.
        //
        // Lifecycle (from Ghidra decompilation):
        // 1. giz_buildit_activate: makes all sub-objects visible via
        //    giz_subobj_set_visible(mesh_data, 1), then hides
        //    [start_index..count) by clearing visibility bit (+0x50 bit0).
        //    Sets state_index = start_index, state = 0.
        // 2. Each frame in giz_buildit_trigger (player IS interacting):
        //    - Vtable callback advances state_index
        //    - giz_buildit_display_current renders ONE sub-object at
        //      state_index via giz_buildit_subobj_display, which stores
        //      the sub-object's world position (+0x24+0x30) into render_ctx
        //      for the particle jibber overlay
        //    - particle_jibber_init creates sinusoidal X/Y oscillation
        //      using mesh data params (+0xb0..+0xbc): X Freq, X Amp,
        //      Y Freq, Y Amp
        //    - Timer decrements; when 0 or flags&0x80 set → release player
        // 3. giz_buildit_deactivate: state=2, if HIDE_ON_DEACT hides all,
        //    clears ACTIVE flag
        //
        // Visual effect: the jibber is a particle overlay, not a mesh
        // transform. We approximate it by applying a sinusoidal Y-offset
        // to the sub-object at state_index during animation.
        self.game_time += dt;
        let interaction_dist: f32 = 3.0;
        let interaction_dist_sq = interaction_dist * interaction_dist;

        // If we have an active buildit, check if still valid.
        if let Some(ai) = self.active_buildit {
            let rt = &mut self.buildit_runtimes[ai];
            let dist_sq = self.player.pos.distance_squared(rt.pos);
            let in_range = dist_sq < interaction_dist_sq;
            let cycle_speed: f32 = 2.0; // sub-objects per second

            if !in_range {
                // Player left range: deactivate at current position.
                println!(
                    "DIAG: '{}' DEACTIVATE (out of range) at si={}/{}",
                    rt.name, rt.state_index, rt.sub_obj_count
                );
                rt.runtime_flags &= !rustt::giz::RT_ANIMATING;
                rt.state = 2;
                self.active_buildit = None;
            } else {
                // Advance state_index (jibber cycles through pieces).
                // Use fractional accumulator — u8 truncation loses sub-1.0 steps.
                rt.state_frac += cycle_speed * dt;
                while rt.state_frac >= 1.0 {
                    rt.state_frac -= 1.0;
                    rt.state_index = rt.state_index.wrapping_add(1);
                    if rt.state_index >= rt.sub_obj_count {
                        rt.state_index = 0;
                    }
                }
            }
        }

        // If no active buildit, try to activate the nearest one in range.
        // Only activate buildits in state 2 (idle/completed) — the engine's
        // giz_buildit_trigger uses anim_check_state + trigger_check_player_valid
        // to gate re-activation and prevent immediate re-triggering.
        if self.active_buildit.is_none() {
            let mut best: Option<(usize, f32)> = None;
            for (bi, rt) in self.buildit_runtimes.iter().enumerate() {
                // Only activate idle buildits (state=2 after completion,
                // or state=0 from init). Skip ones currently animating.
                if rt.runtime_flags & rustt::giz::RT_ANIMATING != 0 {
                    continue;
                }
                let dist_sq = self.player.pos.distance_squared(rt.pos);
                if dist_sq < interaction_dist_sq {
                    match best {
                        None => best = Some((bi, dist_sq)),
                        Some((_, best_d)) if dist_sq < best_d => best = Some((bi, dist_sq)),
                        _ => {}
                    }
                }
            }
            if let Some((bi, _)) = best {
                let rt = &mut self.buildit_runtimes[bi];
                rt.state = 0;
                rt.state_index = 0;
                rt.state_frac = 0.0;
                rt.runtime_flags |= rustt::giz::RT_ANIMATING;
                self.active_buildit = Some(bi);
                println!(
                    "buildit '{}' activated at ({:.1}, {:.1}, {:.1}), {} sub-objects",
                    rt.name, rt.pos.x, rt.pos.y, rt.pos.z, rt.sub_obj_count
                );
                self.last_diag_state = 0;
            }
        }

        // Per-frame diagnostic: always log active buildit state.
        if let Some(ai) = self.active_buildit {
            let rt = &self.buildit_runtimes[ai];
            let si = rt.state_index;
            let pc = self.buildit_particles.particle_count();
            if si != self.last_diag_state || self.game_time % 1.0 < dt {
                self.last_diag_state = si;
                println!(
                    "DIAG: active='{}' si={}/{} frac={:.2} pos=({:.1},{:.1},{:.1}) player=({:.1},{:.1},{:.1}) particles={}",
                    rt.name, si, rt.sub_obj_count, rt.state_frac,
                    rt.pos.x, rt.pos.y, rt.pos.z,
                    self.player.pos.x, self.player.pos.y, self.player.pos.z,
                    pc,
                );
            }
        }

        // Jibber visual: two effects combine to make pieces "bounce":
        //
        // 1. MESH BOUNCE — per-mesh Y-offset via set_mesh_transforms.
        //    The current indexed sub-object's meshes get a sinusoidal
        //    vertical offset, making the piece visibly jump up and down.
        //
        // 2. PARTICLE OVERLAY — billboard sprites spawned at the piece's
        //    bounding center (the engine's particle_jibber system).
        //
        // The engine's giz_buildit_display_current feeds the particle
        // system the sub-object's world position; giz_buildit_mesh_display
        // feeds render_submit_aabb for highlight rendering.  We replicate
        // both with the mesh offset + particle sprites.
        {
            self.jibber_overrides.clear();
            let animating = self.active_buildit
                .map(|ai| self.buildit_runtimes[ai].runtime_flags & rustt::giz::RT_ANIMATING != 0)
                .unwrap_or(false);

            if let Some(ai) = self.active_buildit {
                if animating {
                    let rt = &self.buildit_runtimes[ai];
                    let si = rt.state_index as usize;
                    if let Some(so_map) = self.buildit_mesh_map.get(ai) {
                        if let Some((_, mesh_ids)) = so_map.get(si) {
                            // Mesh bounce: sinusoidal Y offset for the current piece.
                            let t = self.game_time;
                            let bounce = (t * 6.0).sin().abs() * 0.08;
                            let offset = Mat4::from_translation(Vec3::new(0.0, bounce, 0.0));
                            for &m in mesh_ids {
                                self.jibber_overrides.push((m, offset));
                            }

                            // Particle overlay at the piece's bounding center.
                            if let Some(&m) = mesh_ids.first() {
                                if let Some(mesh) = self.gpu.scene.meshes.get(m) {
                                    let pos = mesh.bounds.center;
                                    self.buildit_particles.spawn_buildit_particles(
                                        pos,
                                        dt,
                                        self.game_time,
                                    );
                                }
                            }
                        }
                    }
                } else {
                    self.buildit_particles.reset_spawner();
                }
            } else {
                self.buildit_particles.reset_spawner();
            }

            self.gpu.scene.set_mesh_transforms(&self.gpu.queue, &self.jibber_overrides);
            self.buildit_particles.update(dt, self.game_time);
        }

        // ONLY_PART: frame just the isolated parts — the scene contains only
        // those parts (see scene.rs build_map_meshes), so this turns the next
        // --shot screenshot into a single-object verification image. The
        // camera looks at the union bounds of the listed parts; distance
        // scales with the bounding radius; the yaw/pitch give a
        // slightly-from-above three-quarter view.
        if let Some(ref parts) = self.only_part {
            let mut lo = Vec3::splat(f32::MAX);
            let mut hi = Vec3::splat(f32::MIN);
            let mut any = false;
            for pm in self.pick_meshes.iter().filter(|pm| parts.contains(&pm.part)) {
                lo = lo.min(pm.center - Vec3::splat(pm.radius));
                hi = hi.max(pm.center + Vec3::splat(pm.radius));
                any = true;
            }
            if any {
                let center = (lo + hi) * 0.5;
                let radius = (hi - lo).length() * 0.5;
                self.camera.target = center;
                self.camera.distance = (radius * 3.0).clamp(0.3, 2.0);
                self.camera.yaw = 0.7 + std::f32::consts::PI;
                self.camera.pitch = 0.3;
            }
        }

        // GAME_CAM (debug): re-apply cached override every frame so the chase
        // camera cannot clobber it.
        if let Some(v) = self.game_cam {
            self.camera.target = Vec3::new(v[0], v[1], v[2]);
            self.camera.pitch = v[3];
            self.camera.yaw = v[4];
            self.camera.distance = v[5];
        }

        // --- Picker rays: what is in front of the character / under the
        // camera ---
        // The character's visual forward is (sin yaw, 0, cos yaw): the draw
        // matrix rotates the model's local -Z face by (yaw +
        // PLAYER_FACE_OFFSET), which lands on exactly this vector — the same
        // `facing` the movement model uses. Cast from chest height so the ray
        // clears the floor and reads the furniture instead of the floorboards.
        let facing = Vec3::new(self.player.yaw.sin(), 0.0, self.player.yaw.cos());
        let origin = self.player.pos + Vec3::new(0.0, 0.35, 0.0);
        self.pick_char = ray_pick_all(&self.pick_meshes, origin, facing, PICK_DIST);
        let cam_pos = self.camera.position();
        let cam_dir = (self.camera.target - cam_pos).normalize();
        self.pick_cam = ray_pick_all(&self.pick_meshes, cam_pos, cam_dir, PICK_DIST);

        // Step the animation playhead. The clip is chosen by the actual speed,
        // so a freshly-moving player (or one completing a reversal) walks until
        // speed climbs past `RUN_SWITCH`, then breaks into the run clip. When
        // the desired clip changes, the game crossfades between two evaluated
        // animation buffers over `BLEND_TIME` instead of snapping the pose.
        let s = self.player.speed;
        let idx = if s < IDLE_EPS {
            0 // IDLE
        } else if s < RUN_SWITCH {
            1 // WALK
        } else {
            2 // RUN
        };
        let idx = idx.min(self.anims.len() - 1);

        // Start (or retarget) the crossfade whenever the desired clip changes.
        if let Some(b) = &mut self.player.blend {
            if b.to != idx {
                self.player.blend = Some(Blend {
                    from: b.to,
                    to: idx,
                    clock: 0.0,
                    duration: BLEND_TIME,
                });
            }
        } else if self.player.clip != idx {
            self.player.blend = Some(Blend {
                from: self.player.clip,
                to: idx,
                clock: 0.0,
                duration: BLEND_TIME,
            });
        }

        // A single shared playhead drives both clips; it loops at the length of
        // whichever clip is the active one (the fading-out clip during a blend).
        self.player.anim_clock += dt * WALK_FPS;
        let last = self.anims[self.player.clip]
            .num_frames
            .saturating_sub(1) as f32;
        if self.player.anim_clock > last {
            self.player.anim_clock = 0.0;
        }

        let an3 = |i: usize| &self.anims[i.min(self.anims.len() - 1)];
        let worlds = match self.player.blend {
            Some(ref mut b) => {
                b.clock += dt;
                let t = (b.clock / b.duration).clamp(0.0, 1.0);
                let fa = an3(b.from).remap_playhead(self.player.anim_clock);
                let fb = an3(b.to).remap_playhead(self.player.anim_clock);
                let blended = blended_bone_worlds(
                    an3(b.from),
                    an3(b.to),
                    &self.anim_parents,
                    &self.rest_locals,
                    fa,
                    fb,
                    t,
                );
                if b.clock >= b.duration {
                    self.player.clip = b.to;
                    self.player.blend = None;
                }
                blended
            }
            None => {
                let a = an3(self.player.clip);
                a.bone_worlds(
                    &self.anim_parents,
                    &self.rest_locals,
                    a.remap_playhead(self.player.anim_clock),
                )
            }
        };
        if let Ok(worlds) = worlds {
            self.player_scene.set_skin_mats(&self.gpu.queue, &worlds);
        }

        // `--morphs`: evaluate the sibling BSA (facial blend shapes) at the
        // current playhead and upload the weights, mirroring the viewer.
        if self.morphs {
            if let Some(bsa) = &self.bsa {
                let an3_frames = self.anims[self.player.clip].num_frames;
                let f = map_bsa_frame(self.player.anim_clock, an3_frames, bsa.length_in_frames);
                self.bsa_weights.clear();
                self.bsa_weights
                    .extend((0..bsa.total_channels()).map(|c| bsa.evaluate(c, f)));
                self.player_scene
                    .set_morph_weights(&self.gpu.queue, &self.bsa_weights);
            }
        }

        self.gpu.update_camera(&self.camera);
    }

    fn render(&mut self, view: &wgpu::TextureView, frame_tex: &wgpu::Texture) {
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
            let cull = if self.only_part.is_some() || !self.cull_enabled {
                None
            } else {
                Some(&(self.player.pos, CULL_RADIUS))
            };
            self.active_rooms.clear();
            self.active_rooms.extend(
                self.active_triggers
                    .iter()
                    .filter_map(|name| self.trigger_index.get(name).copied()),
            );
            // Pass 1: opaque geometry → backbuffer (offscreen).
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("map opaque pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: self.gpu.backbuffer_view(),
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
                self.gpu
                    .draw_scene_opaque_culled(&mut rpass, &self.gpu.scene, false, cull, &self.active_rooms);
            }
            // Copy opaque backbuffer → swapchain for refraction.
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: self.gpu.backbuffer_tex(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: frame_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.gpu.config.width.max(1),
                    height: self.gpu.config.height.max(1),
                    depth_or_array_layers: 1,
                },
            );
            // Pass 2: transparent geometry → swapchain (load opaque).
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("map transparent pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.gpu.depth_view(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                self.gpu
                    .draw_scene_transparent_culled(&mut rpass, &self.gpu.scene, false, cull, &self.active_rooms);
            }
            // Particle jibber overlay pass (billboard sprites on top of map).
            if self.buildit_particles.is_spawning() {
                let aspect = self.gpu.config.width as f32 / (self.gpu.config.height as f32).max(1.0);
                let view_proj = self.camera.view_proj(aspect);
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("particle pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.gpu.depth_view(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                self.buildit_particles.render(&mut rpass, &self.gpu.queue, &view_proj);
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

        // HUD overlay: an imgui window logging the character's position. It
        // renders in its own pass (Load, no clear) so the map+player image
        // survives and the window floats on top.
        {
            let ui = self.imgui.context.frame();
            let p = self.player.pos;
            self.hud_history.push_back((p.x, p.y, p.z));
            if self.hud_history.len() > 96 {
                self.hud_history.pop_front();
            }
            let yaw_deg = self.player.yaw.to_degrees();
            let clip = ["IDLE", "WALK", "RUN"][self.player.clip.min(2) as usize];
            if let Some(wt) = ui
                .window("cantina coords")
                .size([300.0, 300.0], Condition::FirstUseEver)
                .position([12.0, 12.0], Condition::FirstUseEver)
                .collapsible(true)
                .begin()
            {
                ui.text(format!("pos  x {:.3}  y {:.3}  z {:.3}", p.x, p.y, p.z));
                ui.text(format!(
                    "face {:.1} deg  speed {:.2}  clip {clip}",
                    yaw_deg,
                    self.player.speed
                ));
                if !self.active_triggers.is_empty() {
                    ui.text(format!("triggers: {}", self.active_triggers.join(", ")));
                }
                // Buildit state display.
                let active_count = self.buildit_runtimes.iter()
                    .filter(|rt| rt.is_active() || rt.runtime_flags & rustt::giz::RT_ANIMATING != 0)
                    .count();
                if active_count > 0 {
                    ui.text(format!("buildits: {}", active_count));
                    for rt in self.buildit_runtimes.iter()
                        .filter(|rt| rt.is_active() || rt.runtime_flags & rustt::giz::RT_ANIMATING != 0)
                    {
                        let status = if rt.runtime_flags & rustt::giz::RT_ANIMATING != 0 {
                            "animating"
                        } else if rt.is_active() {
                            "active"
                        } else {
                            "done"
                        };
                        ui.text(format!(
                            "  {} [{}] idx={}/{}",
                            rt.name, status, rt.state_index, rt.sub_obj_count
                        ));
                    }
                }
                ui.separator();
                ui.text(format!("history (last {} moves):", self.hud_history.len()));
                if let Some(tok) = ui
                    .child_window("hud-history")
                    .size([284.0, 150.0])
                    .always_vertical_scrollbar(true)
                    .begin()
                {
                    for (x, y, z) in &self.hud_history {
                        ui.text(format!("{x:.3}  {y:.3}  {z:.3}"));
                    }
                    tok.end();
                }
                wt.end();
            }

            // Picker window: every map part the two rays pass through,
            // nearest first. The per-part static identity (mesh/material
            // indices, shader defines, lightmap set) plus the top vertex
            // colours make it possible to reason about a specific object's
            // shading without dumping the whole map — e.g. point at the holo
            // table and read off whether its vertices carry dark baked light
            // and whether it is prelit.
            if let Some(wt) = ui
                .window("picker")
                .size([470.0, 400.0], Condition::FirstUseEver)
                .position([12.0, 320.0], Condition::FirstUseEver)
                .collapsible(true)
                .begin()
            {
                let section = |ui: &imgui::Ui, key: &str, title: &str, hits: &[(usize, f32)]| {
                    ui.text(format!("{title}: {} part(s)", hits.len()));
                    if hits.is_empty() {
                        ui.separator();
                        return;
                    }
                    if let Some(tok) = ui
                        .child_window(format!("picker-{key}"))
                        .size([450.0, 140.0])
                        .always_vertical_scrollbar(true)
                        .begin()
                    {
                        for &(pi, t) in hits {
                            let Some(info) = self.pick_info.get(pi) else {
                                continue;
                            };
                            let vcols = info
                                .top_vcol
                                .iter()
                                .map(|(c, n)| {
                                    format!("({},{},{},{})x{}", c[0], c[1], c[2], c[3], n)
                                })
                                .collect::<Vec<_>>()
                                .join(" ");
                            ui.text(format!(
                                "part={:4} mesh={:3} mat={:3} id={:4} tex={:3} defs=0x{:08x} ls={} lmset={} d={:5.2}  vcol {}",
                                info.part,
                                info.mesh,
                                info.material,
                                info.mat_id,
                                info.tex_id,
                                info.shader_defines,
                                info.lighting_stage,
                                info.lightmap_set_index,
                                t,
                                vcols
                            ));
                        }
                        tok.end();
                    }
                    ui.separator();
                };
                section(&ui, "char", "char fwd", &self.pick_char);
                section(&ui, "cam", "camera", &self.pick_cam);
                wt.end();
            }

            // Part info labels projected onto 3D models (F key toggle).
            if self.show_part_labels {
                let io = ui.io();
                let (w, h) = (io.display_size[0], io.display_size[1]);
                if w > 0.0 && h > 0.0 {
                    let aspect = w / h;
                    let vp = self.camera.view_proj(aspect);
                    let dl = ui.get_background_draw_list();
                    let white = imgui::ImColor32::from_rgb_f32s(1.0, 1.0, 1.0);
                    let yellow = imgui::ImColor32::from_rgb_f32s(1.0, 0.85, 0.2);
                    let black = imgui::ImColor32::from_rgb_f32s(0.0, 0.0, 0.0);
                    let cam_pos = self.camera.position();
                    let max_labels = 200usize;
                    let max_dist = 30.0f32;
                    let mut drawn = 0usize;
                    for mesh in &self.gpu.scene.meshes {
                        if drawn >= max_labels {
                            break;
                        }
                        let delta = mesh.bounds.center - cam_pos;
                        let dist = delta.length();
                        if dist > max_dist || dist < 0.1 {
                            continue;
                        }
                        if let Some((sx, sy)) = project_point(vp, mesh.bounds.center, w, h) {
                            let mat = self.gpu.scene.materials.get(mesh.material);
                            let bm = mat.map(|m| m.blend_mode).unwrap_or(0);
                            let ref_type = (bm >> 16) & 0xFFFF;
                            let bm_low = bm & 0xFFFF;
                            let is_glass = ref_type != 0;
                            let prelit = mat.map(|m| m.prelit).unwrap_or(0);
                            let tex_id = mat.map(|m| m.tex_id).unwrap_or(-1);
                            let label = if is_glass {
                                format!(
                                    "P{} M{}\nblend={} ref={} GLASS\nprelit={} tex={}",
                                    mesh.part, mesh.material, bm_low, ref_type, prelit, tex_id
                                )
                            } else {
                                format!(
                                    "P{} M{}\nblend={} prelit={} tex={}",
                                    mesh.part, mesh.material, bm_low, prelit, tex_id
                                )
                            };
                            let color = if is_glass { yellow } else { white };
                            // Shadow for readability.
                            dl.add_text([sx + 1.0, sy + 1.0], black, &label);
                            dl.add_text([sx, sy], color, &label);
                            drawn += 1;
                        }
                    }
                }
            }

            if self.imgui.last_cursor != ui.mouse_cursor() {
                self.imgui.last_cursor = ui.mouse_cursor();
                self.imgui.platform.prepare_render(ui, &self.window);
            }
        }
        let draw_data = self.imgui.context.render();
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hud pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.gpu.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.imgui
                .renderer
                .render(draw_data, &self.gpu.queue, &self.gpu.device, &mut rpass)
                .expect("imgui render failed");
        }
        self.gpu.queue.submit(Some(encoder.finish()));
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
    lights: Vec<RtlLight>,
    player_parsed: Option<Parsed<'static>>,
    idle: Option<An3>,
    walk: Option<An3>,
    run: Option<An3>,
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
    pick_meshes: Vec<PartPick>,
    pick_info: Vec<PartInfo>,
    walk_test: bool,
    facecam: bool,
    morphs: bool,
    bsa: Option<Bsa>,
    map_txt: Option<rustt::map_txt::MapTxt>,
    scp_scripts: std::collections::HashMap<String, rustt::scp::ScpScript>,
    triggers: Vec<rustt::ai2::Trigger>,
    buildit_runtimes: Vec<BuilditRuntime>,
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
                            window.input.just_pressed.insert(code);
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
                window.render(&view, &frame.texture);
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

        window.imgui.platform.handle_event::<()>(
            window.imgui.context.io_mut(),
            &window.window,
            &Event::WindowEvent {
                window_id: _window_id,
                event,
            },
        );
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
        let mut gpu = GpuRenderer::new_map(event_loop, &window, map, &self.lights, "hub")?;
        gpu.show_grid = false;

        // Build trigger name→index lookup.
        let trigger_index: std::collections::HashMap<String, usize> = self
            .triggers
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        // Assign each render part to a trigger (SO only). Room geometry
        // (identity transform) is always drawn.
        let mut part_rooms: Vec<Option<usize>> = Vec::with_capacity(map.render_parts.len());
        for rp in &map.render_parts {
            let tx = rp.transform[0][3];
            let ty = rp.transform[1][3];
            let tz = rp.transform[2][3];
            let pos = glam::Vec3::new(tx, ty, tz);
            let mut assigned = None;
            for (ti, t) in self.triggers.iter().enumerate() {
                let diff = pos - t.pos;
                if diff.x.abs() < t.half_size.x
                    && diff.y.abs() < t.half_size.y
                    && diff.z.abs() < t.half_size.z
                {
                    assigned = Some(ti);
                    break;
                }
            }
            part_rooms.push(assigned);
        }
        gpu.scene.set_part_rooms(part_rooms.clone());
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
            speed: 0.0,
            dir_clock: 0.0,
            last_dir: Vec3::ZERO,
            was_reversing: false,
            anim_clock: 0.0,
            clip: 0,
            blend: None,
        };

        let mut camera = OrbitCamera::default();
        camera.target = player.pos + Vec3::new(0.0, CAM_TARGET, 0.0);
        camera.distance = CAM_DIST;
        // Face the open floor/bar side; the default yaw puts the camera under
        // the low booth overhang behind the spawn and the collision pull-in
        // collapses it to a close-up.
        camera.yaw = 0.7 + std::f32::consts::PI;
        // Pitch the camera so it sits at CAM_HEIGHT above the floor while
        // looking at the torso (CAM_TARGET). The cantina ceiling is only ~1.5
        // units up, so keep it near level; a steeper default orbit pitch would
        // put the camera above the roof and the collision pull-in would
        // collapse it to a close-up of the head.
        camera.pitch = ((CAM_HEIGHT - CAM_TARGET) / CAM_DIST).asin();

        // GAME_CAM (debug): x,y,z,pitch,yaw,dist — absolute camera override
        // for automated junction shots with --shot --no-input.  Parsed once
        // here; the per-frame update() re-applies it from the cached field.
        let game_cam = std::env::var("GAME_CAM")
            .ok()
            .and_then(|ov| {
                let v: Vec<f32> = ov.split(',').filter_map(|s| s.parse().ok()).collect();
                if v.len() == 6 {
                    camera.target = Vec3::new(v[0], v[1], v[2]);
                    camera.pitch = v[3];
                    camera.yaw = v[4];
                    camera.distance = v[5];
                    println!("GAME_CAM override: target=({:.2},{:.2},{:.2}) pitch={:.2} yaw={:.2} dist={:.2}", v[0], v[1], v[2], v[3], v[4], v[5]);
                    Some([v[0], v[1], v[2], v[3], v[4], v[5]])
                } else {
                    None
                }
            });

        let parsed = self.player_parsed.as_ref().expect("player model loaded");
        let idle = self.idle.take().expect("IDLE anim loaded");
        let walk = self.walk.take().expect("WALK anim loaded");
        let run = self.run.take().expect("RUN anim loaded");
        let n = walk.num_bones;
        let anim_parents: Vec<i32> = (0..n)
            .map(|i| parsed.bones.get(i).map(|b| b.parent.min(n as i32 - 1)).unwrap_or(-1))
            .collect();
        let rest_locals: Vec<Mat4> = parsed.bones.iter().map(|b| b.local).collect();
        let imgui = imgui_state::ImguiState::new(&gpu.device, &gpu.queue, gpu.config.format, &window);

        // Build the buildit → GpuMesh mapping.  For each buildit, for each
        // sub-object name, find the SO entity render_part by name and map to
        // its GpuMesh.  NO shared-geometry expansion — multiple sub-objects
        // share the same (mesh, material) data, so any expansion pulls in
        // GpuMeshes from other sub-objects, causing cross-contamination.
        //
        // The room geometry copies (unnamed render_parts with the same mesh
        // data) remain visible but static.  The bounce transform lifts the
        // SO entity mesh above the room geometry base.
        let map_ref = self.map.as_ref().expect("map loaded");

        let mut buildit_mesh_map: Vec<Vec<(usize, Vec<usize>)>> = Vec::new();
        for bi in &self.buildit_runtimes {
            let mut so_map: Vec<(usize, Vec<usize>)> = Vec::new();
            let mut total_meshes = 0usize;
            for (si, so_name) in bi.sub_object_names.iter().enumerate() {
                let rp_idx = map_ref
                    .render_parts
                    .iter()
                    .position(|rp| rp.name.as_deref() == Some(so_name.as_str()));
                let mesh_ids: Vec<usize> = if let Some(rp_idx) = rp_idx {
                    gpu.scene
                        .meshes
                        .iter()
                        .enumerate()
                        .filter(|(_, m)| m.part == rp_idx)
                        .map(|(i, _)| i)
                        .collect()
                } else {
                    Vec::new()
                };
                total_meshes += mesh_ids.len();
                so_map.push((si, mesh_ids));
            }
            println!(
                "buildit '{}': {} sub-objects, {} total meshes mapped",
                bi.name, bi.sub_object_names.len(), total_meshes
            );
            for (si, so_name) in bi.sub_object_names.iter().enumerate() {
                let empty: Vec<usize> = Vec::new();
                let mesh_ids = so_map.get(si).map_or(&empty, |(_, m)| m);
                for &m in mesh_ids {
                    let part_idx = gpu.scene.meshes[m].part;
                    let rp = &map_ref.render_parts[part_idx];
                    let kind = if rp.name.is_some() { "SO" } else { "ROOM" };
                    println!(
                        "  [{}] '{}' → gpu_mesh[{}] part={} kind={} mesh_data={}",
                        si, so_name, m, part_idx, kind, rp.mesh
                    );
                }
                if mesh_ids.is_empty() {
                    println!("  [{}] '{}' → (no meshes)", si, so_name);
                }
            }
            buildit_mesh_map.push(so_map);
        }

        // Hide room geometry copies of buildit sub-objects.  SO entities and
        // room geometry share the same Map::meshes[] data, so both render the
        // same visual.  We hide the room geometry copies (name=None) so only
        // the SO entity GpuMeshes remain — those are the ones we transform.
        {
            use std::collections::HashSet;
            let so_mesh_indices: HashSet<usize> = buildit_mesh_map
                .iter()
                .flat_map(|so_map| so_map.iter().flat_map(|(_, ids)| ids))
                .copied()
                .collect();
            // Collect SO mesh data indices for comparison.
            let so_data_indices: HashSet<usize> = so_mesh_indices
                .iter()
                .map(|&gi| map_ref.render_parts[gpu.scene.meshes[gi].part].mesh)
                .collect();
            // Find room geometry GpuMeshes to hide (not SO meshes themselves,
            // not named meshes, but share mesh data with some buildit SO).
            let to_hide: Vec<usize> = gpu.scene.meshes
                .iter()
                .enumerate()
                .filter(|(gi, m)| {
                    !so_mesh_indices.contains(gi)
                        && map_ref.render_parts[m.part].name.is_none()
                        && so_data_indices.contains(&map_ref.render_parts[m.part].mesh)
                })
                .map(|(gi, _)| gi)
                .collect();
            for &gi in &to_hide {
                gpu.scene.meshes[gi].visible = false;
            }
            println!("hidden {} room geometry copies of buildit sub-objects", to_hide.len());
        }

        let buildit_particles = particles::BuilditParticles::new(&gpu.device, gpu.config.format);

        Ok(AppWindow {
            window,
            gpu,
            imgui,
            camera,
            input: Input::default(),
            exit_requested: false,
            player,
            player_scene,
            anims: vec![idle, walk, run],
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
            pick_meshes: std::mem::take(&mut self.pick_meshes),
            pick_info: std::mem::take(&mut self.pick_info),
            pick_char: Vec::new(),
            pick_cam: Vec::new(),
            only_part: std::env::var("ONLY_PART")
                .ok()
                .map(|v| {
                    v.split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .collect::<std::collections::HashSet<usize>>()
                })
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    std::env::var("FRAME_PARTS")
                        .ok()
                        .map(|v| {
                            v.split(',')
                                .filter_map(|s| s.trim().parse::<usize>().ok())
                                .collect::<std::collections::HashSet<usize>>()
                        })
                        .filter(|s| !s.is_empty())
                }),
            walk_test: self.walk_test,
            facecam: self.facecam,
            morphs: self.morphs,
            bsa: self.bsa.take(),
            bsa_weights: Vec::new(),
            hud_history: VecDeque::new(),
            cull_enabled: true,
            show_part_labels: false,
            triggers: std::mem::take(&mut self.triggers),
            trigger_index,
            active_triggers: Vec::new(),
            part_rooms,
            buildit_runtimes: std::mem::take(&mut self.buildit_runtimes),
            game_time: 0.0,
            buildit_mesh_map,
            game_cam,
            active_rooms: std::collections::HashSet::new(),
            active_buildit: None,
            last_diag_state: 0,
            buildit_particles,
            jibber_overrides: Vec::new(),
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

/// Scale the AN3 playhead (in frames) onto the BSA timeline (frame 0..bsa-1).
fn map_bsa_frame(playhead: f32, an3_frames: usize, bsa_frames: f32) -> f32 {
    let src = an3_frames.max(1) as f32;
    let dst = bsa_frames.max(1.0);
    if src <= 1.0 || dst <= 1.0 {
        return 0.0;
    }
    (playhead * (dst - 1.0) / (src - 1.0)).clamp(0.0, dst - 1.0)
}

fn load_bsa_sibling(anim_path: &Path) -> Option<Bsa> {
    let p = anim_path.with_extension("bsa");
    let data = std::fs::read(&p).ok()?;
    match Bsa::parse(&data) {
        Ok(bsa) => {
            println!(
                "load {}: {} frames, {} groups, {} channels",
                p.display(),
                bsa.length_in_frames,
                bsa.group_count,
                bsa.total_channels()
            );
            Some(bsa)
        }
        Err(e) => {
            println!("skip {}: {e}", p.display());
            None
        }
    }
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

/// Every part a ray from `origin` along unit `dir` passes through (up to
/// `max_t`), sorted nearest first. Same sphere-reject + Möller–Trumbore test
/// as `ray_hit_dist`, but the picker keeps ALL hits — the object of interest
/// (say a holo table behind the bar) still shows up even when a nearer wall
/// occludes part of the view. Only the first triangle hit per part is
/// recorded: the window lists parts, not triangle hits.
fn ray_pick_all(meshes: &[PartPick], origin: Vec3, dir: Vec3, max_t: f32) -> Vec<(usize, f32)> {
    let mut hits: Vec<(usize, f32)> = Vec::new();
    for (pi, m) in meshes.iter().enumerate() {
        // Ray-segment vs bounding sphere reject (as above).
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
            if t > 0.0 && t <= max_t {
                hits.push((pi, t));
                break;
            }
        }
    }
    hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

/// Project a world-space point to screen coordinates (pixels, imgui convention).
/// Returns `None` when the point is behind the camera or outside the clip volume.
fn project_point(vp: glam::Mat4, p: glam::Vec3, w: f32, h: f32) -> Option<(f32, f32)> {
    let clip = vp * p.extend(1.0);
    if clip.w <= 1e-5 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z >= 1.0 || ndc.z <= -1.0 {
        return None;
    }
    Some(((ndc.x * 0.5 + 0.5) * w, (ndc.y * -0.5 + 0.5) * h))
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
    let facecam = args.iter().any(|a| a == "--facecam");
    let morphs = args.iter().any(|a| a == "--morphs");
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
    let mut map = rustt::map::parse(map_bytes).with_context(|| format!("parsing {map_path}"))?;

    // Apply GIZ blowup positions if a sibling .GIZ file exists.
    let giz_path = map_path.replace("MAP_PC.GSC", "MAP.GIZ");
    let mut buildit_runtimes = Vec::new();
    if let Ok(giz_data) = std::fs::read(&giz_path) {
        if let Ok(giz) = rustt::giz::parse_giz(&giz_data) {
            let before = map.render_parts.len();
            let mut mesh_overrides = std::collections::HashMap::new();
            if let Some(rp) = map.render_parts.iter().position(|p| p.mesh == 982) {
                mesh_overrides.insert("chair_01".to_string(), rp);
            }
            map.apply_giz_blowups(&giz, &mesh_overrides);
            let after = map.render_parts.len();
            if after != before {
                println!("GIZ: applied {} blowup positions (+{} parts)", giz.blowups.len(), after - before);
            }
            map.apply_giz_buildits(&giz);
            buildit_runtimes = giz.buildits.iter().map(BuilditRuntime::new).collect();
            println!("GIZ: created {} buildit runtimes", buildit_runtimes.len());
        }
    }

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

    // --- MAP.TXT (text-level definition: doors, obstacles, cameras) ---
    let map_txt_path = map_path.replace("MAP_PC.GSC", "MAP.TXT");
    let map_txt = std::fs::read_to_string(&map_txt_path)
        .ok()
        .and_then(|d| rustt::map_txt::parse(&d).ok());
    if let Some(ref mt) = map_txt {
        println!(
            "{}: {} doors, {} obstacles, {} blowups, {} buildits, {} turrets, {} socks (cameras)",
            map_txt_path,
            mt.doors.len(),
            mt.obstacles.len(),
            mt.blowups.len(),
            mt.buildits.len(),
            mt.turrets.len(),
            mt.socks.len(),
        );
    }

    // --- SCP scripts (AI state machines) ---
    let ai_dir = map_path.replace("MAP_PC.GSC", "AI");
    let mut scp_scripts: std::collections::HashMap<String, rustt::scp::ScpScript> = std::collections::HashMap::new();
    if let Ok(entries) = std::fs::read_dir(&ai_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("scp")).unwrap_or(false) {
                let name = p.file_stem().unwrap().to_string_lossy().to_uppercase();
                if let Ok(data) = std::fs::read_to_string(&p) {
                    if let Ok(script) = rustt::scp::parse(&data) {
                        let states = script.states.len();
                        scp_scripts.insert(name.clone(), script);
                        println!("scp: {name} {states} states");
                    }
                }
            }
        }
    }

    // Sibling light list (MAP_PC.GSC -> MAP_PC.RTL, falling back to
    // MAP.RTL): per-mesh light baking for the map renderer; empty when no
    // candidate exists.
    let rtl_candidates = rtl::sibling_rtl_candidates(&map_path);
    let lights: Vec<RtlLight> = rtl_candidates
        .iter()
        .find_map(|rtl_path| std::fs::read(rtl_path).ok().map(|data| (rtl_path.clone(), data)))
        .map(|(rtl_path, data)| {
            let lights = rtl::parse(&data);
            println!("{}: {} lights", rtl_path.display(), lights.len());
            lights
        })
        .unwrap_or_else(|| {
            println!(
                "no RTL at {} (tried {:?}); map renders with the default rig",
                rtl_candidates[0].display(),
                rtl_candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>()
            );
            Vec::new()
        });
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

    // Per-part picker data, one entry per render part (a part = one mesh +
    // one material, so hits can be attributed to a named material). Triangles
    // + bounding sphere for the raycast, and the static identity shown in
    // the picker window: mesh/material indices, shader defines, lightmap
    // set, and the 3 most common vertex colours (RGBA8 on the 0..127 baked
    // light scale, e.g. (127,127,127,127) = neutral vs (49,49,49,127) =
    // darkened by bake).
    let mut pick_meshes: Vec<PartPick> = Vec::new();
    let mut pick_info: Vec<PartInfo> = Vec::new();
    for (pi, part) in map.render_parts.iter().enumerate() {
        let Some(m) = map.meshes.get(part.mesh) else { continue };
        let Some(mut md) = rustt::mapmesh::expand_mesh(&map, m) else { continue };
        if md.pos.is_empty() || md.idx.is_empty() {
            continue;
        }
        rustt::mapmesh::apply_transform(&mut md, &part.transform);
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
        let mut cs = std::collections::BTreeMap::<[u8; 4], usize>::new();
        for c in &md.color {
            *cs.entry(*c).or_insert(0) += 1;
        }
        let top_vcol: Vec<([u8; 4], usize)> = cs.into_iter().take(3).collect();
        let mat = map.materials.get(part.material);
        pick_meshes.push(PartPick { part: pi, center, radius, tris });
        pick_info.push(PartInfo {
            part: pi,
            mesh: part.mesh,
            material: part.material,
            mat_id: mat.map(|m| m.id).unwrap_or(-1),
            tex_id: mat.map(|m| m.tex_id).unwrap_or(-1),
            shader_defines: mat.map(|m| m.shader_defines).unwrap_or(0),
            lighting_stage: mat.map(|m| m.lighting_stage).unwrap_or(0),
            lightmap_set_index: mat.map(|m| m.lightmap_set_index).unwrap_or(0),
            top_vcol,
        });
    }
    println!(
        "picker: {} parts loaded ({} skipped)",
        pick_meshes.len(),
        map.render_parts.len() - pick_meshes.len()
    );

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
    let idle = load_an3(&char_dir.join("IDLE.AN3"))?;
    let walk = load_an3(&char_dir.join("WALK.AN3"))?;
    let run = load_an3(&char_dir.join("RUN.AN3"))?;
    let bsa = if morphs {
        load_bsa_sibling(&char_dir.join("IDLE.AN3"))
    } else {
        None
    };

    if dump_only {
        println!("load path OK");
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        map: Some(map),
        lights,
        player_parsed: Some(parsed),
        idle: Some(idle),
        walk: Some(walk),
        run: Some(run),
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
        pick_meshes,
        pick_info,
        walk_test,
        facecam,
        morphs,
        bsa,
        map_txt,
        scp_scripts,
        triggers: ai.triggers,
        buildit_runtimes,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
