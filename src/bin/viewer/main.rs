mod camera;
mod renderer;
mod scene;

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use imgui::{Condition, FontSource, Image, TreeNodeFlags};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use renderer::GpuRenderer;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, Event, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::Window,
};

use glam::Mat4;
use rustt::an3::An3;
use rustt::bsa::Bsa;
use rustt::ghg::Parsed;
use rustt::map::Map;

const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.12,
    g: 0.13,
    b: 0.16,
    a: 1.0,
};

struct InputState {
    last_mouse: Option<(f64, f64)>,
    drag: (f32, f32),
    pan: (f32, f32),
    wheel: f32,
    left_down: bool,
    right_down: bool,
    keys: HashSet<KeyCode>,
    want_capture_mouse: bool,
    want_capture_keyboard: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            last_mouse: None,
            drag: (0.0, 0.0),
            pan: (0.0, 0.0),
            wheel: 0.0,
            left_down: false,
            right_down: false,
            keys: HashSet::new(),
            want_capture_mouse: false,
            want_capture_keyboard: false,
        }
    }
}

struct ImguiState {
    context: imgui::Context,
    platform: WinitPlatform,
    renderer: imgui_wgpu::Renderer,
    last_frame: Instant,
    last_cursor: Option<imgui::MouseCursor>,
}

impl ImguiState {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        window: &Window,
    ) -> Self {
        let mut context = imgui::Context::create();
        let mut platform = WinitPlatform::new(&mut context);
        platform.attach_window(context.io_mut(), window, HiDpiMode::Default);
        context.set_ini_filename(None);

        let hidpi = window.scale_factor();
        context.io_mut().font_global_scale = (1.0 / hidpi) as f32;
        context.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                size_pixels: (13.0 * hidpi) as f32,
                ..Default::default()
            }),
        }]);

        let renderer_config = if format.is_srgb() {
            imgui_wgpu::RendererConfig {
                texture_format: format,
                depth_format: Some(wgpu::TextureFormat::Depth32Float),
                ..imgui_wgpu::RendererConfig::new()
            }
        } else {
            imgui_wgpu::RendererConfig {
                texture_format: format,
                depth_format: Some(wgpu::TextureFormat::Depth32Float),
                ..imgui_wgpu::RendererConfig::new_srgb()
            }
        };
        let renderer = imgui_wgpu::Renderer::new(&mut context, device, queue, renderer_config);

        Self {
            context,
            platform,
            renderer,
            last_frame: Instant::now(),
            last_cursor: None,
        }
    }

    fn rebuild_font(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scale_factor: f64) {
        self.context.fonts().clear();
        self.context.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: (13.0 * scale_factor) as f32,
                ..Default::default()
            }),
        }]);
        self.renderer
            .reload_font_texture(&mut self.context, device, queue);
    }
}

struct AppWindow {
    window: Arc<Window>,
    gpu: GpuRenderer,
    imgui: ImguiState,
    camera: camera::OrbitCamera,
    input: InputState,
    exit_requested: bool,
    show_scene: bool,
    show_materials: bool,
    show_bones: bool,
    show_textures: bool,
    show_anim: bool,
    file_name: String,
    /// The layer (LOD/quality) sets shown in the "Scene" combo.
    layer_sets: Option<LayerSets>,
    /// Currently selected quality index into `QUALITY_NAMES` / `quality_names`.
    quality: usize,
    quality_names: Vec<String>,
    anim_idx: usize,
    parsed_rest_locals: Vec<Mat4>,
    anim_frame: f32,
    anim_playing: bool,
    anim_speed: f32,
    anim_cam_framed: bool,
    /// Parent list for the currently selected animation (rebuilt on switch).
    anim_parents: Vec<i32>,
    /// Last animation index the parent list / skin matrices were built for.
    last_anim_idx: usize,
    /// Last frame the skin matrices were uploaded for (skips redundant writes
    /// when paused on an unchanged frame).
    last_skin_frame: f32,
    /// Per-channel BSA (facial blend-shape) weights for the selected
    /// animation's current frame. Rebuilt every redraw; empty when the selected
    /// animation has no sibling BSA.
    bsa_weights: Vec<f32>,
}

/// Loaded file data, split by file kind. Model files (`.GHG`) carry bones,
/// animations and blend shapes; map files (`.GSC`) are static geometry.
enum AppData {
    Model(AppModel),
    Map(AppMap),
}

struct AppModel {
    file_name: String,
    parsed: Parsed<'static>,
    /// Layer (LOD/quality) sets from the sibling `.TXT`; `None` when the model
    /// has no TXT (render all layers, the game's default for such models).
    layer_sets: Option<LayerSets>,
    anims: Vec<An3>,
    anim_paths: Vec<String>,
    /// Sibling blend-shape animation per `anims` entry (same name, same
    /// directory, `.BSA` extension); `None` when the AN3 has no BSA.
    bsas: Vec<Option<Bsa>>,
}

struct AppMap {
    file_name: String,
    map: Map<'static>,
}

struct App {
    data: AppData,
    window: Option<AppWindow>,
}

fn texture_preview(ui: &imgui::Ui, id: imgui::TextureId, w: u32, h: u32) {
    let scale = (128.0 / w.max(1) as f32)
        .min(1.0)
        .min(128.0 / h.max(1) as f32);
    let size = [w as f32 * scale, h as f32 * scale];
    Image::new(id, size).build(ui);
}

/// Load the same-named `.BSA` blend-shape (facial) animation next to an `.AN3`
/// path so the viewer can play face + body together. Returns `None` when there
/// is no sibling BSA or it fails to parse (the animation then plays AN3-only).
fn load_bsa_sibling(anim_path: &str) -> Option<Bsa> {
    let bsa_path = Path::new(anim_path).with_extension("BSA");
    if !bsa_path.exists() {
        return None;
    }
    let data = std::fs::read(&bsa_path).ok()?;
    match Bsa::parse(&data) {
        Ok(bsa) => {
            println!(
                "load BSA {}: {} frames, {} channels",
                bsa_path.display(),
                bsa.length_in_frames,
                bsa.total_channels()
            );
            Some(bsa)
        }
        Err(e) => {
            eprintln!("skip BSA {}: {e:#}", bsa_path.display());
            None
        }
    }
}

/// Quality names for the layer-selection combo, in display order; the index
/// selects the matching `layers_*` set via `quality_layers`.
const QUALITY_NAMES: [&str; 6] = ["special", "high", "medium", "low", "dead", "all"];

/// The layer (LOD/quality) sets a character declares in its sibling `.TXT`.
/// The game draws only ONE set per graphics quality; the viewer uses the same
/// sets so only the chosen variant renders instead of every layer overlapping.
#[derive(Default, Clone)]
struct LayerSets {
    special: Vec<u32>,
    high: Vec<u32>,
    medium: Vec<u32>,
    low: Vec<u32>,
    dead: Vec<u32>,
}

/// Read the sibling `.TXT` (base name without the platform suffix, e.g.
/// `BOBAFETT_PC.GHG` / `BOBAFETT_LR_PC.GHG` pair with `BOBAFETT.TXT`) and
/// collect its `layers_*` sets. Returns `None` when the model has no TXT.
fn load_layer_sets(model_path: &str) -> Option<LayerSets> {
    let p = Path::new(model_path);
    let dir = p.parent().unwrap_or_else(|| Path::new(""));
    let stem = p.file_stem()?.to_string_lossy().into_owned();
    // Probe progressively shorter stems, stripping known platform suffixes
    // (`_PC`, `_LR_PC`, `_PS2`, ...) until a `.TXT` matches.
    let mut txt_path: Option<std::path::PathBuf> = None;
    let mut probe = stem.clone();
    loop {
        let candidate = dir.join(format!("{probe}.TXT"));
        if candidate.exists() {
            txt_path = Some(candidate);
            break;
        }
        let Some((head, tail)) = probe.rsplit_once('_') else {
            break;
        };
        if !matches!(
            tail.to_ascii_uppercase().as_str(),
            "PC" | "LR" | "PS2" | "XB" | "XBOX" | "NGC" | "GC" | "D3D" | "DX"
        ) {
            break;
        }
        probe = head.to_string();
    }
    let mut txt_path = txt_path?;
    // A variant model's TXT may only contain `txt_file="base"` and inherit the
    // base model's `layers_*` lines (e.g. ANAKIN_PADAWAN -> anakin_jedi).
    for _ in 0..4 {
        let data = std::fs::read_to_string(&txt_path).ok()?;
        let sets = parse_layer_sets(&data);
        if !sets.is_empty() {
            return Some(sets);
        }
        let refer = data
            .lines()
            .find_map(|l| l.trim().strip_prefix("txt_file"))
            .and_then(|v| v.trim_start_matches('=').trim().split('"').nth(1))
            .map(|s| s.to_owned());
        let Some(refer) = refer else {
            return None;
        };
        let next = dir.join(format!("{refer}.TXT"));
        if next == txt_path || !next.exists() {
            return None;
        }
        txt_path = next;
    }
    None
}

fn parse_layer_sets(data: &str) -> LayerSets {
    let mut sets = LayerSets::default();
    for line in data.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim();
        if !key.starts_with("layers_") {
            continue;
        }
        // Lenient parse: some files contain typos like "1,4.6,7" (the game
        // tolerates them); both ',' and '.' act as separators.
        let nums: Vec<u32> = v
            .split([',', '.'])
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        match key {
            "layers_special" => sets.special = nums,
            "layers_high" => sets.high = nums,
            "layers_medium" => sets.medium = nums,
            "layers_low" => sets.low = nums,
            "layers_dead" => sets.dead = nums,
            _ => {}
        }
    }
    sets
}

impl LayerSets {
    fn is_empty(&self) -> bool {
        self.special.is_empty()
            && self.high.is_empty()
            && self.medium.is_empty()
            && self.low.is_empty()
            && self.dead.is_empty()
    }
}

/// Resolve a `QUALITY_NAMES` index to the layer set to draw. An empty result
/// means "no filtering" (render every layer), which is also the default when
/// the model has no TXT.
fn quality_layers(sets: &Option<LayerSets>, quality: usize) -> Vec<u32> {
    let Some(sets) = sets else {
        return Vec::new();
    };
    match quality {
        0 => sets.special.clone(),
        1 => sets.high.clone(),
        2 => sets.medium.clone(),
        3 => sets.low.clone(),
        4 => sets.dead.clone(),
        _ => Vec::new(),
    }
}

/// Map the AN3 playback playhead (in `[0, an3_frames)`) onto the BSA timeline
/// (`[0, bsa_frames)`). The game stores both with the same length, so this is
/// the identity in the common case; it scales gracefully if they ever diverge.
fn map_bsa_frame(playhead: f32, an3_frames: usize, bsa_frames: f32) -> f32 {
    let src = an3_frames.max(1) as f32;
    let dst = bsa_frames.max(1.0);
    if src <= 1.0 || dst <= 1.0 {
        return 0.0;
    }
    (playhead * (dst - 1.0) / (src - 1.0)).clamp(0.0, dst - 1.0)
}

#[allow(clippy::too_many_arguments)]
fn build_ui(
    ui: &imgui::Ui,
    parsed: &Parsed,
    file_name: &str,
    gpu: &mut GpuRenderer,
    camera: &mut camera::OrbitCamera,
    input: &mut InputState,
    show_scene: &mut bool,
    show_materials: &mut bool,
    show_bones: &mut bool,
    show_textures: &mut bool,
    show_anim: &mut bool,
    anims: &[An3],
    bsas: &[Option<Bsa>],
    bsa_weights: &[f32],
    anim_names: &[String],
    anim_idx: &mut usize,
    anim_frame: &mut f32,
    anim_playing: &mut bool,
    anim_speed: &mut f32,
    layer_sets: &Option<LayerSets>,
    quality: &mut usize,
    quality_names: &[String],
    exit_requested: &mut bool,
) {
    if let Some(tb) = ui.begin_main_menu_bar() {
        if let Some(menu) = ui.begin_menu("File") {
            if ui.menu_item("Quit") {
                *exit_requested = true;
            }
            menu.end();
        }
        if let Some(menu) = ui.begin_menu("View") {
            ui.checkbox("Grid", &mut gpu.show_grid);
            let wire_supported = gpu.wireframe_supported();
            {
                let _disabled = ui.begin_disabled(!wire_supported);
                ui.checkbox("Wireframe", &mut gpu.show_wireframe);
            }
            let mut apply = gpu.scene.apply_bones;
            if ui.checkbox("Apply bind pose", &mut apply) {
                gpu.scene.set_apply_bones(&gpu.device, parsed, apply);
                camera.frame(&gpu.scene.bounds);
            }
            ui.separator();
            ui.checkbox("Scene window", show_scene);
            ui.checkbox("Materials window", show_materials);
            ui.checkbox("Bones window", show_bones);
            ui.checkbox("Textures window", show_textures);
            ui.checkbox("Animation window", show_anim);
            menu.end();
        }
        if let Some(menu) = ui.begin_menu("Camera") {
            if ui.menu_item("Frame model") {
                camera.frame(&gpu.scene.bounds);
            }
            if ui.menu_item("Reset view") {
                camera.reset();
            }
            menu.end();
        }
        tb.end();
    }

    if *show_scene {
        if let Some(wt) = ui
            .window("Scene")
            .size([320.0, 240.0], Condition::FirstUseEver)
            .begin()
        {
            ui.text(file_name);
            ui.separator();
            ui.text(format!("render items: {}", parsed.render.len()));
            ui.text(format!("parts: {}", parsed.parts.len()));
            ui.text(format!("materials: {}", parsed.materials.len()));
            ui.text(format!("textures: {}", parsed.textures.len()));
            ui.text(format!("bones: {}", parsed.bones.len()));
            ui.text(format!(
                "triangles: {}",
                gpu.scene
                    .meshes
                    .iter()
                    .map(|m| m.index_count as usize / 3)
                    .sum::<usize>()
            ));
            ui.text(format!("vertex lists: {}", parsed.vertex_lists.len()));
            ui.text(format!("index lists: {}", parsed.index_lists.len()));
            if layer_sets.is_some() {
                let prev = *quality;
                if ui.combo_simple_string("Quality", quality, quality_names) && *quality != prev {
                    let allowed = quality_layers(layer_sets, *quality);
                    gpu.set_layers(parsed, &allowed);
                    camera.frame(&gpu.scene.bounds);
                }
            }
            ui.separator();
            let b = &gpu.scene.bounds;
            ui.text(format!(
                "center: ({:.2}, {:.2}, {:.2})",
                b.center.x, b.center.y, b.center.z
            ));
            ui.text(format!("radius: {:.2}", b.radius));
            ui.separator();
            ui.text(format!("fps: {:.1}", ui.io().framerate));
            ui.text(format!("cam dist: {:.2}", camera.distance));
            wt.end();
        }
    }

    if *show_materials {
        if let Some(wt) = ui
            .window("Materials")
            .size([360.0, 420.0], Condition::FirstUseEver)
            .begin()
        {
            for (i, m) in parsed.materials.iter().enumerate() {
                let open = ui.collapsing_header(
                    format!("Material {i} (id {})", m.id),
                    TreeNodeFlags::FRAMED | TreeNodeFlags::NO_TREE_PUSH_ON_OPEN,
                );
                if !open {
                    continue;
                }
                let mut col = gpu.scene.materials[i].diffuse;
                if ui.color_edit4(format!("##diffuse{i}"), &mut col) {
                    gpu.scene.set_material_color(&gpu.queue, i, col);
                }
                ui.text(format!("tex_id: {}", m.tex_id));
                ui.text(format!(
                    "rgba: {:02x}{:02x}{:02x}{:02x}",
                    m.rgba[0], m.rgba[1], m.rgba[2], m.rgba[3]
                ));
                if m.tex_id >= 0 {
                    if let Some(t) = gpu.scene.textures.get(m.tex_id as usize) {
                        if let Some(Some(pid)) = gpu.scene.preview_ids.get(m.tex_id as usize) {
                            texture_preview(ui, *pid, t.w, t.h);
                        }
                    }
                }
                ui.separator();
            }
            wt.end();
        }
    }

    if *show_bones {
        if let Some(wt) = ui
            .window("Bones")
            .size([360.0, 420.0], Condition::FirstUseEver)
            .begin()
        {
            for (i, b) in parsed.bones.iter().enumerate() {
                let open = ui.collapsing_header(
                    format!("{i}: {}", b.name),
                    TreeNodeFlags::FRAMED | TreeNodeFlags::NO_TREE_PUSH_ON_OPEN,
                );
                if !open {
                    continue;
                }
                ui.text(format!("parent: {}", b.parent));
                for r in 0..4 {
                    let row = b.world.row(r);
                    ui.text(format!(
                        "[ {:8.3} {:8.3} {:8.3} {:8.3} ]",
                        row.x, row.y, row.z, row.w
                    ));
                }
                ui.separator();
            }
            wt.end();
        }
    }

    if *show_anim {
        if let Some(wt) = ui
            .window("Animation")
            .size([360.0, 220.0], Condition::FirstUseEver)
            .begin()
        {
            if anims.is_empty() {
                ui.text("No animation loaded (pass an .AN3 as second arg).");
            } else {
                if anim_names.len() > 1 {
                    ui.combo_simple_string("Animation", anim_idx, anim_names);
                    ui.separator();
                }
                let an3 = &anims[(*anim_idx).min(anims.len() - 1)];
                ui.text(format!(
                    "{} bones, {} frames, {} moving ch",
                    an3.num_bones, an3.num_frames, an3.num_moving
                ));
                ui.separator();
                ui.checkbox("Play", anim_playing);
                let mut slider_time = *anim_frame;
                let last_frame = an3.num_frames.saturating_sub(1) as f32;
                if ui.slider("##time", 0.0, last_frame, &mut slider_time) {
                    *anim_frame = slider_time;
                    *anim_playing = false;
                }
                ui.slider("Speed", 0.1, 4.0, anim_speed);
                ui.text(format!("frame {:.1} / {}", anim_frame, last_frame));
                ui.separator();
                match bsas.get(*anim_idx).and_then(|o| o.as_ref()) {
                    Some(bsa) => {
                        ui.text(format!(
                            "BSA: {} frames, {} channels ({} groups)",
                            bsa.length_in_frames,
                            bsa.total_channels(),
                            bsa.group_count
                        ));
                        ui.text("shape-key weights:");
                        let mut line = String::new();
                        for (i, w) in bsa_weights.iter().enumerate() {
                            line.push_str(&format!("{i}:{w:.2} "));
                            if (i + 1) % 8 == 0 {
                                ui.text(line.trim_end());
                                line.clear();
                            }
                        }
                        if !line.is_empty() {
                            ui.text(line.trim_end());
                        }
                    }
                    None => {
                        ui.text("BSA: none (playing AN3 only)");
                    }
                }
            }
            wt.end();
        }
    }

    if *show_textures {
        if let Some(wt) = ui
            .window("Textures")
            .size([320.0, 400.0], Condition::FirstUseEver)
            .begin()
        {
            for (i, t) in gpu.scene.textures.iter().enumerate() {
                let open = ui.collapsing_header(
                    format!("Texture {i}  {}x{}  {}", t.w, t.h, t.fmt),
                    TreeNodeFlags::FRAMED | TreeNodeFlags::NO_TREE_PUSH_ON_OPEN,
                );
                if !open {
                    continue;
                }
                if let Some(Some(pid)) = gpu.scene.preview_ids.get(i) {
                    texture_preview(ui, *pid, t.w, t.h);
                }
                ui.separator();
            }
            wt.end();
        }
    }

    input.want_capture_mouse = ui.io().want_capture_mouse;
    input.want_capture_keyboard = ui.io().want_capture_keyboard;
}

#[allow(clippy::too_many_arguments)]
fn build_map_ui(
    ui: &imgui::Ui,
    map: &Map,
    file_name: &str,
    gpu: &mut GpuRenderer,
    camera: &mut camera::OrbitCamera,
    input: &mut InputState,
    show_scene: &mut bool,
    show_materials: &mut bool,
    show_textures: &mut bool,
    exit_requested: &mut bool,
) {
    if let Some(tb) = ui.begin_main_menu_bar() {
        if let Some(menu) = ui.begin_menu("File") {
            if ui.menu_item("Quit") {
                *exit_requested = true;
            }
            menu.end();
        }
        if let Some(menu) = ui.begin_menu("View") {
            ui.checkbox("Grid", &mut gpu.show_grid);
            let wire_supported = gpu.wireframe_supported();
            {
                let _disabled = ui.begin_disabled(!wire_supported);
                ui.checkbox("Wireframe", &mut gpu.show_wireframe);
            }
            ui.separator();
            ui.checkbox("Scene window", show_scene);
            ui.checkbox("Materials window", show_materials);
            ui.checkbox("Textures window", show_textures);
            menu.end();
        }
        if let Some(menu) = ui.begin_menu("Camera") {
            if ui.menu_item("Frame map") {
                camera.frame(&gpu.scene.bounds);
            }
            if ui.menu_item("Reset view") {
                camera.reset();
            }
            menu.end();
        }
        tb.end();
    }

    if *show_scene {
        if let Some(wt) = ui
            .window("Scene")
            .size([320.0, 240.0], Condition::FirstUseEver)
            .begin()
        {
            ui.text(file_name);
            ui.separator();
            ui.text(format!("render parts: {}", map.render_parts.len()));
            ui.text(format!("meshes: {}", map.meshes.len()));
            ui.text(format!("materials: {}", map.materials.len()));
            ui.text(format!("textures: {}", map.textures.len()));
            ui.text(format!(
                "triangles: {}",
                gpu.scene
                    .meshes
                    .iter()
                    .map(|m| m.index_count as usize / 3)
                    .sum::<usize>()
            ));
            ui.text(format!("vertex buffers: {}", map.vertex_buffers.len()));
            ui.text(format!("index buffers: {}", map.index_buffers.len()));
            ui.separator();
            let b = &gpu.scene.bounds;
            ui.text(format!(
                "center: ({:.2}, {:.2}, {:.2})",
                b.center.x, b.center.y, b.center.z
            ));
            ui.text(format!("radius: {:.2}", b.radius));
            ui.separator();
            ui.text(format!("fps: {:.1}", ui.io().framerate));
            ui.text(format!("cam dist: {:.2}", camera.distance));
            wt.end();
        }
    }

    if *show_materials {
        if let Some(wt) = ui
            .window("Materials")
            .size([360.0, 420.0], Condition::FirstUseEver)
            .begin()
        {
            for i in 0..gpu.scene.materials.len() {
                let id = map.materials.get(i).map(|m| m.id).unwrap_or(i as i32);
                let open = ui.collapsing_header(
                    format!("Material {i} (id {id})"),
                    TreeNodeFlags::FRAMED | TreeNodeFlags::NO_TREE_PUSH_ON_OPEN,
                );
                if !open {
                    continue;
                }
                let mut col = gpu.scene.materials[i].diffuse;
                if ui.color_edit4(format!("##diffuse{i}"), &mut col) {
                    gpu.scene.set_material_color(&gpu.queue, i, col);
                }
                ui.text(format!("tex_id: {}", gpu.scene.materials[i].tex_id));
                if gpu.scene.materials[i].tex_id >= 0 {
                    let tid = gpu.scene.materials[i].tex_id as usize;
                    if let Some(t) = gpu.scene.textures.get(tid) {
                        if let Some(Some(pid)) = gpu.scene.preview_ids.get(tid) {
                            texture_preview(ui, *pid, t.w, t.h);
                        }
                    }
                }
                ui.separator();
            }
            wt.end();
        }
    }

    if *show_textures {
        if let Some(wt) = ui
            .window("Textures")
            .size([320.0, 400.0], Condition::FirstUseEver)
            .begin()
        {
            for (i, t) in gpu.scene.textures.iter().enumerate() {
                let open = ui.collapsing_header(
                    format!("Texture {i}  {}x{}  {}", t.w, t.h, t.fmt),
                    TreeNodeFlags::FRAMED | TreeNodeFlags::NO_TREE_PUSH_ON_OPEN,
                );
                if !open {
                    continue;
                }
                if let Some(Some(pid)) = gpu.scene.preview_ids.get(i) {
                    texture_preview(ui, *pid, t.w, t.h);
                }
                ui.separator();
            }
            wt.end();
        }
    }

    input.want_capture_mouse = ui.io().want_capture_mouse;
    input.want_capture_keyboard = ui.io().want_capture_keyboard;
}

impl App {
    fn create_window(&self, event_loop: &ActiveEventLoop) -> Result<AppWindow> {
        let file_name = match &self.data {
            AppData::Model(m) => &m.file_name,
            AppData::Map(m) => &m.file_name,
        };
        let size = LogicalSize::new(1280.0, 800.0);
        let attributes = Window::default_attributes()
            .with_title(format!("rustt viewer - {}", file_name))
            .with_inner_size(size);
        let window = Arc::new(event_loop.create_window(attributes)?);

        // Default to the "special" quality set when the model has a TXT; render
        // every layer otherwise (models without a TXT have no LOD layers).
        let (layer_sets, quality, allowed) = match &self.data {
            AppData::Model(m) => {
                let q = if m.layer_sets.is_some() {
                    0
                } else {
                    QUALITY_NAMES.len() - 1
                };
                (m.layer_sets.clone(), q, quality_layers(&m.layer_sets, q))
            }
            AppData::Map(_) => (None, QUALITY_NAMES.len() - 1, Vec::new()),
        };
        let quality_names: Vec<String> = QUALITY_NAMES.iter().map(|s| s.to_string()).collect();

        let mut gpu = match &self.data {
            AppData::Model(m) => GpuRenderer::new(
                event_loop,
                &window,
                &m.parsed,
                &m.file_name,
                &allowed,
            )?,
            AppData::Map(m) => GpuRenderer::new_map(event_loop, &window, &m.map, &m.file_name)?,
        };
        let mut imgui = ImguiState::new(&gpu.device, &gpu.queue, gpu.config.format, &window);

        gpu.scene
            .register_preview_textures(&gpu.device, &mut imgui.renderer);

        let mut camera = camera::OrbitCamera::default();
        camera.frame(&gpu.scene.bounds);

        // Play the requested animation (index 0) by default; an auto-loaded
        // IDLE is available via the combo box. `parsed_rest_locals` feeds the
        // 4INA 0x20 logic: AN3 rotation channels are small deltas composed
        // against each bone's static parent-relative local rotation. The raw
        // mesh lives in bone-local space, so animated skinning is just
        // `anim_world[bone]` — no inverse-bind (passing `&[]` as the rest
        // leaves each part posed by its animated world directly).
        let anim_idx = 0usize;
        let parsed_rest_locals: Vec<Mat4> = match &self.data {
            AppData::Model(m) => m.parsed.bones.iter().map(|b| b.local).collect(),
            AppData::Map(_) => Vec::new(),
        };

        let show_anim = matches!(&self.data, AppData::Model(m) if !m.anims.is_empty());

        Ok(AppWindow {
            window,
            gpu,
            imgui,
            camera,
            input: InputState::default(),
            exit_requested: false,
            show_scene: true,
            show_materials: true,
            show_bones: matches!(&self.data, AppData::Model(_)),
            show_textures: true,
            show_anim,
            file_name: file_name.clone(),
            layer_sets,
            quality,
            quality_names,
            anim_idx,
            parsed_rest_locals,
            anim_frame: 0.0,
            anim_playing: false,
            anim_speed: 1.0,
            anim_cam_framed: false,
            anim_parents: Vec::new(),
            last_anim_idx: usize::MAX,
            last_skin_frame: f32::NEG_INFINITY,
            bsa_weights: Vec::new(),
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        match self.create_window(event_loop) {
            Ok(w) => self.window = Some(w),
            Err(e) => {
                eprintln!("failed to create viewer window: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_mut() else {
            return;
        };

        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                window.gpu.resize(size.width, size.height);
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                window
                    .imgui
                    .rebuild_font(&window.gpu.device, &window.gpu.queue, *scale_factor);
            }
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
                (winit::event::MouseButton::Left, ElementState::Pressed) => {
                    window.input.left_down = true;
                }
                (winit::event::MouseButton::Left, ElementState::Released) => {
                    window.input.left_down = false;
                    window.input.last_mouse = None;
                }
                (winit::event::MouseButton::Right, ElementState::Pressed) => {
                    window.input.right_down = true;
                }
                (winit::event::MouseButton::Right, ElementState::Released) => {
                    window.input.right_down = false;
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
                    if window.input.right_down {
                        window.input.pan.0 += dx;
                        window.input.pan.1 += dy;
                    }
                }
                window.input.last_mouse = Some((x, y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let v = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, v) => *v,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 40.0) as f32,
                };
                window.input.wheel += v;
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - window.imgui.last_frame).as_secs_f32().min(0.1);
                window
                    .imgui
                    .context
                    .io_mut()
                    .update_delta_time(now - window.imgui.last_frame);
                window.imgui.last_frame = now;

                if window.exit_requested {
                    event_loop.exit();
                    return;
                }

                let frame = match window.gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame) => frame,
                    wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
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

                // Split the window into disjoint field borrows so the imgui `Ui`
                // (borrowed from `imgui.context`) can be used alongside `gpu`,
                // `camera`, etc.
                let AppWindow {
                    window: win,
                    gpu,
                    imgui,
                    camera,
                    input,
                    exit_requested,
                    show_scene,
                    show_materials,
                    show_bones,
                    show_textures,
                    show_anim,
                    file_name,
                    layer_sets,
                    quality,
                    quality_names,
                    anim_idx,
                    anim_frame,
                    anim_playing,
                    anim_speed,
                    anim_parents,
                    last_anim_idx,
                    last_skin_frame,
                    parsed_rest_locals,
                    anim_cam_framed,
                    bsa_weights,
                } = window;

                if input.want_capture_mouse {
                    input.drag = (0.0, 0.0);
                    input.pan = (0.0, 0.0);
                    input.wheel = 0.0;
                } else {
                    camera.orbit(input.drag.0, input.drag.1);
                    camera.pan(input.pan.0, input.pan.1);
                    camera.zoom(input.wheel);
                    input.drag = (0.0, 0.0);
                    input.pan = (0.0, 0.0);
                    input.wheel = 0.0;
                }
                if !(input.want_capture_mouse || input.want_capture_keyboard) {
                    let fwd = (input.keys.contains(&KeyCode::KeyW) as i8
                        - input.keys.contains(&KeyCode::KeyS) as i8)
                        as f32;
                    let strafe = (input.keys.contains(&KeyCode::KeyD) as i8
                        - input.keys.contains(&KeyCode::KeyA) as i8)
                        as f32;
                    let up = (input.keys.contains(&KeyCode::KeyE) as i8
                        - input.keys.contains(&KeyCode::KeyQ) as i8)
                        as f32;
                    camera.move_target(fwd, strafe, up, dt);
                }

                if let AppData::Model(m) = &self.data {
                    if let Some(an3) = m.anims.get(*anim_idx) {
                        let last = an3.num_frames.saturating_sub(1) as f32;
                        if *anim_playing {
                            *anim_frame += dt * *anim_speed;
                            if *anim_frame > last {
                                *anim_frame = 0.0;
                            }
                        }
                        // Rebuild the parent list only when the animation switches.
                        let parents_changed = *anim_idx != *last_anim_idx;
                        if parents_changed {
                            let n = an3.num_bones;
                            *anim_parents = (0..n)
                                .map(|i| {
                                    m.parsed
                                        .bones
                                        .get(i)
                                        .map(|b| b.parent.min(n as i32 - 1))
                                        .unwrap_or(-1)
                                })
                                .collect();
                            *last_anim_idx = *anim_idx;
                        }
                        if gpu.scene.apply_bones {
                            // Upload skin matrices only when the pose actually
                            // changes (paused on an unchanged frame = no-op).
                            if parents_changed || *anim_frame != *last_skin_frame {
                                let world_frame = an3.remap_playhead(*anim_frame);
                                if let Ok(worlds) =
                                    an3.bone_worlds(anim_parents, &parsed_rest_locals, world_frame)
                                {
                                    gpu.scene.set_skin_mats(&gpu.queue, &worlds);
                                    *last_skin_frame = *anim_frame;
                                    if !*anim_cam_framed {
                                        camera.frame(&gpu.scene.bounds);
                                        *anim_cam_framed = true;
                                    }
                                }
                            }
                        }
                    }

                    // Sync the sibling BSA (facial blend shapes) to the AN3
                    // playhead. The BSA timeline matches the AN3 playback
                    // timeline, so the playhead maps 1:1 (scaled if lengths
                    // ever diverge). Evaluated every redraw; cheap.
                    *bsa_weights = m
                        .bsas
                        .get(*anim_idx)
                        .and_then(|o| o.as_ref())
                        .map(|bsa| {
                            let an3_frames = m.anims.get(*anim_idx).map_or(1, |a| a.num_frames);
                            let f = map_bsa_frame(*anim_frame, an3_frames, bsa.length_in_frames);
                            (0..bsa.total_channels())
                                .map(|c| bsa.evaluate(c, f))
                                .collect()
                        })
                        .unwrap_or_default();
                }

                // Push the evaluated BSA weights to the GPU morph uniforms so
                // blend shapes are applied in the vertex shader this frame.
                gpu.scene.set_morph_weights(&gpu.queue, bsa_weights.as_slice());

                imgui
                    .platform
                    .prepare_frame(imgui.context.io_mut(), win)
                    .expect("prepare_frame failed");

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: gpu.depth_view(),
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

                    gpu.update_camera(camera);
                    gpu.draw_scene(&mut rpass);

                    {
                        let ui = imgui.context.frame();
                        match &self.data {
                            AppData::Model(m) => build_ui(
                                ui,
                                &m.parsed,
                                file_name,
                                gpu,
                                camera,
                                input,
                                show_scene,
                                show_materials,
                                show_bones,
                                show_textures,
                                show_anim,
                                &m.anims,
                                &m.bsas,
                                bsa_weights,
                                &m.anim_paths
                                    .iter()
                                    .map(|p| {
                                        Path::new(p)
                                            .file_stem()
                                            .map(|s| s.to_string_lossy().into_owned())
                                            .unwrap_or_else(|| p.clone())
                                    })
                                    .collect::<Vec<_>>(),
                                anim_idx,
                                anim_frame,
                                anim_playing,
                                anim_speed,
                                layer_sets,
                                quality,
                                quality_names,
                                exit_requested,
                            ),
                            AppData::Map(m) => build_map_ui(
                                ui,
                                &m.map,
                                file_name,
                                gpu,
                                camera,
                                input,
                                show_scene,
                                show_materials,
                                show_textures,
                                exit_requested,
                            ),
                        }
                        if imgui.last_cursor != ui.mouse_cursor() {
                            imgui.last_cursor = ui.mouse_cursor();
                            imgui.platform.prepare_render(ui, win);
                        }
                    }

                    let draw_data = imgui.context.render();
                    imgui
                        .renderer
                        .render(draw_data, &gpu.queue, &gpu.device, &mut rpass)
                        .expect("imgui render failed");
                }

                gpu.queue.submit(Some(encoder.finish()));
                frame.present();
            }
            _ => {}
        }

        window.imgui.platform.handle_event::<()>(
            window.imgui.context.io_mut(),
            &window.window,
            &Event::WindowEvent { window_id, event },
        );
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(window) = self.window.as_mut() else {
            return;
        };
        window.window.request_redraw();
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .context("usage: viewer <model.ghg|map.gsc> [anim.an3]")?;
    let anim_path = args.next();

    let data = std::fs::read(&model_path).with_context(|| format!("reading {}", model_path))?;
    let data: &'static [u8] = Box::leak(data.into_boxed_slice());
    let file_name = Path::new(&model_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.clone());

    let app_data = if model_path.to_ascii_lowercase().ends_with(".gsc") {
        let map = rustt::map::parse(data).with_context(|| format!("parsing {}", model_path))?;
        println!(
            "{}: {} render parts, {} meshes, {} materials, {} textures",
            file_name,
            map.render_parts.len(),
            map.meshes.len(),
            map.materials.len(),
            map.textures.len()
        );
        AppData::Map(AppMap { file_name, map })
    } else {
        let parsed = rustt::ghg::parse(data).with_context(|| format!("parsing {}", model_path))?;
        println!(
            "{}: {} render items, {} materials, {} textures, {} bones",
            file_name,
            parsed.render.len(),
            parsed.materials.len(),
            parsed.textures.len(),
            parsed.bones.len()
        );

        // A TXT that declares no `layers_*` sets (e.g. SARLACC) is treated as
        // absent: the model has no LOD layers, so every layer renders.
        let layer_sets = load_layer_sets(&model_path).filter(|s| !s.is_empty());
        if let Some(sets) = &layer_sets {
            println!(
                "layer sets (sibling TXT): special={:?} high={:?} medium={:?} low={:?} dead={:?} (default: special)",
                sets.special, sets.high, sets.medium, sets.low, sets.dead
            );
        } else {
            println!(
                "no sibling TXT: rendering all {} layers",
                parsed
                    .render_layer
                    .last()
                    .map_or(1, |l| (l + 1) as usize)
            );
        }

        // Load the requested animation, plus the character's IDLE.AN3 (same
        // dir) which provides the model-rest skeleton if not already selected.
        // Each loaded AN3 also pulls in its sibling BSA (facial blend shapes).
        let mut anims: Vec<An3> = Vec::new();
        let mut anim_paths: Vec<String> = Vec::new();
        let mut bsas: Vec<Option<Bsa>> = Vec::new();
        if let Some(path) = anim_path {
            let data = std::fs::read(&path).with_context(|| format!("reading {}", path))?;
            let anim = rustt::an3::An3::parse(&data).with_context(|| format!("parsing {}", path))?;
            println!(
                "{}: {} bones, {} frames, {} moving channels",
                path, anim.num_bones, anim.num_frames, anim.num_moving
            );
            bsas.push(load_bsa_sibling(&path));
            anims.push(anim);
            anim_paths.push(path);
        }
        // If the passed anim isn't the IDLE, also try to load the character's
        // IDLE (same folder) for the rest skeleton.
        let base_upper = Path::new(anim_paths.first().map_or("", |s| s.as_str()))
            .file_stem()
            .map(|s| s.to_string_lossy().to_uppercase())
            .unwrap_or_default();
        if base_upper != "IDLE" {
            if let Some(anim_root) = anim_paths.first() {
                let idle = Path::new(anim_root).with_file_name("IDLE.AN3");
                if idle.exists()
                    && !anim_paths
                        .iter()
                        .any(|p| p.eq_ignore_ascii_case(&idle.to_string_lossy()))
                {
                    let data = std::fs::read(&idle)
                        .with_context(|| format!("reading {}", idle.display()))?;
                    match rustt::an3::An3::parse(&data) {
                        Ok(anim) => {
                            println!(
                                "load {}: {} bones, {} frames, {} moving channels (rest)",
                                idle.display(),
                                anim.num_bones,
                                anim.num_frames,
                                anim.num_moving
                            );
                            let idle_str = idle.to_string_lossy().into_owned();
                            bsas.push(load_bsa_sibling(&idle_str));
                            anims.push(anim);
                            anim_paths.push(idle_str);
                        }
                        Err(e) => eprintln!("skip IDLE.AN3: {e:#}"),
                    }
                }
            }
        }

        AppData::Model(AppModel {
            file_name,
            parsed,
            layer_sets,
            anims,
            anim_paths,
            bsas,
        })
    };

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        data: app_data,
        window: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_sets_found_for_pc_model() {
        // The real data ships platform-suffixed GHG names (BOBAFETT_PC.GHG)
        // whose TXT is BOBAFETT.TXT, and the viewer loads in that folder.
        let path = "backup/CHARS/BOBAFETT/BOBAFETT_PC.GHG";
        if !Path::new(path).exists() {
            eprintln!("skipping: {path} not present");
            return;
        }
        let sets = load_layer_sets(path).expect("BOBAFETT TXT should be discovered");
        assert_eq!(sets.special, vec![0, 1, 5]);
        assert_eq!(sets.high, vec![0, 2, 5]);
        assert_eq!(sets.medium, vec![0, 3, 5]);
        assert_eq!(sets.low, vec![0, 4, 5]);
        assert_eq!(sets.dead, vec![5, 6]);
    }

    #[test]
    fn layer_sets_none_when_no_txt() {
        let path = "backup/CHARS/SARLACC/SARLACC_PC.GHG";
        if !Path::new(path).exists() {
            eprintln!("skipping: {path} not present");
            return;
        }
        // SARLACC.TXT declares no layers_* sets and no txt_file, so the model
        // is single-layer: load_layer_sets yields nothing.
        assert!(load_layer_sets(path).is_none());
    }

    #[test]
    fn layer_sets_found_for_multi_underscore_stem() {
        let path = "backup/CHARS/ANAKIN/ANAKIN_JEDI_SCARRED_PC.GHG";
        if !Path::new(path).exists() {
            eprintln!("skipping: {path} not present");
            return;
        }
        // Strip only `_PC`, keep the middle underscore in the TXT stem. The
        // scarred variant carries its own (slightly different) layer sets.
        let sets = load_layer_sets(path).expect("ANAKIN_JEDI_SCARRED TXT should be discovered");
        assert_eq!(sets.special, vec![0, 1]);
        assert_eq!(sets.dead, vec![5, 6]);
    }

    #[test]
    fn layer_sets_inherited_via_txt_file() {
        let path = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
        if !Path::new(path).exists() {
            eprintln!("skipping: {path} not present");
            return;
        }
        // ANAKIN_PADAWAN.TXT only says txt_file="anakin_jedi"; the layers come
        // from ANAKIN_JEDI.TXT in the same folder.
        let sets = load_layer_sets(path).expect("ANAKIN_PADAWAN TXT should be discovered");
        assert_eq!(sets.special, vec![0, 1, 5]);
    }
}
