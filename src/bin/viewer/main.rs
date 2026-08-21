mod camera;
mod imgui_state;
mod renderer;
mod scene;

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use imgui::{Condition, Image, TreeNodeFlags};

use imgui_state::ImguiState;

/// Copy a rendered texture to a PNG (VIEWER_SHOT debug aid).
fn capture_texture(gpu: &GpuRenderer, tex: &wgpu::Texture, path: &str) -> Result<()> {
    let w = gpu.config.width.max(1);
    let h = gpu.config.height.max(1);
    let bytes_per_row = w as usize * 4;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot staging"),
        size: (bytes_per_row * h as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row as u32),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).expect("map_async receiver");
    });
    gpu.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv().context("map_async")??;

    let raw = slice.get_mapped_range();
    let mut rgba: Vec<u8> = raw.to_vec();
    drop(raw);
    buffer.unmap();
    // Swapchain/offscreen targets are BGRA on Windows; the png encoder writes
    // RGB, so flip the channels back.
    let bgra = matches!(
        tex.format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    if bgra {
        for px in rgba.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    let mut out = std::fs::File::create(path).context("creating shot file")?;
    let mut enc = png::Encoder::new(&mut out, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().context("png header")?;
    writer.write_image_data(&rgba).context("png data")?;

    // Quick luminance stats so automation can tell lit from fullbright.
    let mut sum = 0.0f64;
    let mut bright = 0usize;
    let mut dark = 0usize;
    let mut bands = [0usize; 4];
    let mut total = 0usize;
    for px in rgba.chunks_exact(4) {
        let l = 0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64;
        sum += l;
        if l > 0.9 * 255.0 {
            bright += 1;
        }
        if l < 0.03 * 255.0 {
            dark += 1;
        }
        let b = ((l / 64.0).floor() as usize).min(3);
        bands[b] += 1;
        total += 1;
    }
    let mean = sum / total.max(1) as f64;
    eprintln!(
        "shot stats: meanL={mean:.1}/255 ({:.0}% of max) bright>230={:.1}% dark<8={:.1}% bands<64:{:.0}% <128:{:.0}% <192:{:.0}% >=192:{:.0}%",
        mean / 255.0 * 100.0,
        bright as f64 / total.max(1) as f64 * 100.0,
        dark as f64 / total.max(1) as f64 * 100.0,
        bands[0] as f64 / total.max(1) as f64 * 100.0,
        (bands[0] + bands[1]) as f64 / total.max(1) as f64 * 100.0,
        (bands[0] + bands[1] + bands[2]) as f64 / total.max(1) as f64 * 100.0,
        bands[3] as f64 / total.max(1) as f64 * 100.0,
    );
    // VIEWER_SHOT_REF=<png>: pixel-diff against a reference so shader changes
    // can be judged even when the aggregate stats barely move.
    if let Ok(ref_path) = std::env::var("VIEWER_SHOT_REF") {
        let dec = png::Decoder::new(std::fs::File::open(&ref_path)?);
        let mut reader = dec.read_info().context("ref png header")?;
        let mut ref_rgba = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut ref_rgba).context("ref png frame")?;
        let ref_rgba = &ref_rgba[..info.buffer_size()];
        let rw = info.width as usize;
        let rh = info.height as usize;
        let mut changed = 0usize;
        let mut adiff = 0.0f64;
        let mut ncmp = 0usize;
        for y in 0..rh.min(h as usize) {
            for x in 0..rw.min(w as usize) {
                let a = (y * w as usize + x) * 4;
                let b = (y * rw + x) * 4;
                let la = 0.2126 * rgba[a] as f64 + 0.7152 * rgba[a + 1] as f64 + 0.0722 * rgba[a + 2] as f64;
                let lb = 0.2126 * ref_rgba[b] as f64 + 0.7152 * ref_rgba[b + 1] as f64 + 0.0722 * ref_rgba[b + 2] as f64;
                if (la - lb).abs() > 2.0 {
                    changed += 1;
                }
                adiff += (la - lb).abs();
                ncmp += 1;
            }
        }
        eprintln!(
            "shot diff vs {ref_path}: changed>2/255: {:.1}% mean|dl|={:.2}",
            changed as f64 / ncmp.max(1) as f64 * 100.0,
            adiff / ncmp.max(1) as f64
        );
    }
    Ok(())
}

/// Copy the just-presented swapchain image to a PNG (mirroring the game
/// client's capture_surface).
fn capture_surface(
    gpu: &GpuRenderer,
    frame: &wgpu::SurfaceTexture,
    path: &str,
) -> Result<()> {
    capture_texture(gpu, &frame.texture, path)
}

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
use rustt::rtl::RtlLight;

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
    just_pressed: HashSet<KeyCode>,
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
            just_pressed: HashSet::new(),
            want_capture_mouse: false,
            want_capture_keyboard: false,
        }
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
    /// Frames rendered so far (drives the env-var screenshot below).
    frames_done: u32,
    /// Save a PNG after this many frames (VIEWER_SHOT="<frame>:<path>").
    shot_frame: Option<u32>,
    shot_path: Option<String>,
    /// Offscreen render target for headless VIEWER_SHOT captures (rendering
    /// into the swapchain stalls forever while Windows reports the window
    /// occluded; the offscreen texture is independent of surface state).
    headless_view: Option<(wgpu::Texture, wgpu::TextureView)>,
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
    /// Sibling `.RTL` light list for per-mesh lighting; empty when absent.
    lights: Vec<RtlLight>,
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

/// Load the same-named `.RTL` light list next to a `.GSC` map so the viewer
/// can light each mesh from its own position like the original's per-part
/// baking. The exact sibling (`MAP_PC.RTL`) may be absent because the light
/// list is shipped as `MAP.RTL`, so the platform-tagged name is tried first
/// and the untagged name second. Returns an empty list when neither exists.
fn load_rtl_sibling(map_path: &str) -> Vec<RtlLight> {
    for rtl_path in rustt::rtl::sibling_rtl_candidates(map_path) {
        let Ok(data) = std::fs::read(&rtl_path) else {
            continue;
        };
        let lights = rustt::rtl::parse(&data);
        println!("load RTL {}: {} lights", rtl_path.display(), lights.len());
        return lights;
    }
    Vec::new()
}

/// Find a sibling `.GIZ` file next to a `.GSC` path.
fn find_sibling_giz(gsc_path: &str) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(gsc_path);
    // Same name but .GIZ extension (e.g. MAP_PC.GSC -> MAP.GIZ).
    let stem = p.file_stem()?.to_str()?;
    // Strip _PC suffix to find the base name.
    let base = stem.strip_suffix("_PC").unwrap_or(stem);
    let giz = p.with_file_name(format!("{base}.GIZ"));
    if giz.exists() {
        Some(giz)
    } else {
        None
    }
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
            ui.checkbox("Grid X-ray", &mut gpu.show_grid_xray);
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

    if gpu.show_grid {
        draw_coord_overlay(ui, camera, gpu);
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
            ui.checkbox("Grid X-ray", &mut gpu.show_grid_xray);
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

    if gpu.show_grid {
        draw_coord_overlay(ui, camera, gpu);
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
                let lighting_name = match gpu.scene.materials[i].lighting_stage {
                    0 => "unlit",
                    1 => "lambert",
                    2 => "gooch",
                    3 => "envmap",
                    4 => "aniso",
                    5 => "aniso_ward",
                    6 => "phong",
                    _ => "unknown",
                };
                ui.text(format!("lighting: {lighting_name}"));
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
            AppData::Map(m) => {
                GpuRenderer::new_map(event_loop, &window, &m.map, &m.lights, &m.file_name)?
            }
        };
        let mut imgui = ImguiState::new(&gpu.device, &gpu.queue, gpu.config.format, &window);

        gpu.scene
            .register_preview_textures(&gpu.device, &mut imgui.renderer);

        let mut camera = camera::OrbitCamera::default();
        camera.frame(&gpu.scene.bounds);

        // VIEWER_CAM="tx,ty,tz,yaw,pitch[,distance]": override the framed
        // camera so headless VIEWER_SHOT captures can target a prop.
        if let Ok(cv) = std::env::var("VIEWER_CAM") {
            let v: Vec<f32> = cv.split(',').filter_map(|s| s.parse().ok()).collect();
            if v.len() >= 5 {
                camera.target = glam::Vec3::new(v[0], v[1], v[2]);
                camera.yaw = v[3];
                camera.pitch = v[4];
                if v.len() >= 6 {
                    camera.distance = v[5];
                }
            }
        }

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

        let shot = std::env::var("VIEWER_SHOT").ok();
        let (shot_frame, shot_path) = match shot {
            Some(s) => {
                let (n, path) = s.split_once(':').unwrap_or(("30", &s));
                (n.parse::<u32>().ok(), Some(path.to_owned()))
            }
            None => (None, None),
        };
        let headless_view = if shot_path.is_some() {
            let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("headless capture texture"),
                size: wgpu::Extent3d {
                    width: gpu.config.width.max(1),
                    height: gpu.config.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: gpu.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            Some((tex, view))
        } else {
            None
        };

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
            frames_done: 0,
            shot_frame,
            shot_path,
            headless_view,
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
                            window.input.just_pressed.insert(code);
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
                if window.shot_path.is_some() {
                    eprintln!("redraw #{}: enter", window.frames_done);
                }
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

                let frame = if window.headless_view.is_some() {
                    None
                } else {
                    match window.gpu.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(frame) => Some(frame),
                        wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Some(frame),
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
                    frames_done,
                    shot_frame,
                    shot_path,
                    headless_view,
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

                // 'C' toggles the VIEWER_CULL sphere on/off.
                if !(input.want_capture_keyboard) && input.just_pressed.contains(&KeyCode::KeyC) {
                    gpu.cull_enabled = !gpu.cull_enabled;
                    println!("cull: {}", if gpu.cull_enabled { "ON" } else { "OFF" });
                }
                // 'P' toggles color correction (approximate D3D9 sRGB-space look).
                if !(input.want_capture_keyboard) && input.just_pressed.contains(&KeyCode::KeyP) {
                    gpu.color_correct_enabled = !gpu.color_correct_enabled;
                    println!("color_correct: {}", if gpu.color_correct_enabled { "ON" } else { "OFF" });
                }
                // 'O' toggles SO/room coloring: green = room geometry, yellow = SO entity.
                if !(input.want_capture_keyboard) && input.just_pressed.contains(&KeyCode::KeyO) {
                    gpu.so_coloring_enabled = !gpu.so_coloring_enabled;
                    println!("so_coloring: {}", if gpu.so_coloring_enabled { "ON" } else { "OFF" });
                }
                // '0' toggles cubemap reflections on/off for debugging specular noise.
                if !(input.want_capture_keyboard) && input.just_pressed.contains(&KeyCode::Digit0) {
                    gpu.cubemap_enabled = !gpu.cubemap_enabled;
                    println!("cubemap: {}", if gpu.cubemap_enabled { "ON" } else { "OFF" });
                }
                // '1' toggles normal map on/off for debugging specular noise.
                if !(input.want_capture_keyboard) && input.just_pressed.contains(&KeyCode::Digit1) {
                    gpu.normal_map_enabled = !gpu.normal_map_enabled;
                    println!("normal_map: {}", if gpu.normal_map_enabled { "ON" } else { "OFF" });
                }
                input.just_pressed.clear();

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

                let headless = shot_path.is_some();

                let view = match &headless_view {
                    Some((_, hv)) => hv.clone(),
                    None => frame
                        .as_ref()
                        .expect("surface frame")
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                };
                let mut encoder = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                gpu.update_camera(camera);

                let cull: Option<(glam::Vec3, f32)> = if gpu.cull_enabled {
                    std::env::var("VIEWER_CULL").ok().and_then(|v| {
                        let p: Vec<f32> = v.split(',').filter_map(|s| s.parse().ok()).collect();
                        (p.len() >= 4).then(|| (glam::Vec3::new(p[0], p[1], p[2]), p[3]))
                    })
                } else {
                    None
                };

                // Pass 1: opaque geometry → backbuffer (offscreen), grid too.
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("opaque pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: gpu.backbuffer_view(),
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

                    gpu.draw_scene_opaque_culled(&mut rpass, &gpu.scene, false, cull.as_ref(), &std::collections::HashSet::new());
                    gpu.draw_grid(&mut rpass);
                }

                // Copy opaque backbuffer → swapchain so transparent pass
                // can sample the opaque scene for refraction.
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: gpu.backbuffer_tex(),
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &frame.as_ref().expect("surface frame").texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: gpu.config.width.max(1),
                        height: gpu.config.height.max(1),
                        depth_or_array_layers: 1,
                    },
                );

                // Pass 2: transparent geometry → swapchain (load opaque + depth).
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("transparent pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: gpu.depth_view(),
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

                    gpu.draw_scene_transparent_culled(&mut rpass, &gpu.scene, false, cull.as_ref(), &std::collections::HashSet::new());

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
                    if !headless {
                        imgui
                            .renderer
                            .render(draw_data, &gpu.queue, &gpu.device, &mut rpass)
                            .expect("imgui render failed");
                    }
                }

                gpu.queue.submit(Some(encoder.finish()));

*frames_done += 1;
                if shot_path.is_some() {
                    eprintln!("redraw #{frames_done}: drawn");
                }
                if *shot_frame == Some(*frames_done) {
                    if let Some(path) = shot_path {
                        let res = match &headless_view {
                            Some((tex, _)) => capture_texture(gpu, tex, path),
                            None => capture_surface(gpu, frame.as_ref().expect("surface frame"), path),
                        };
                        match res {
                            Ok(()) => eprintln!("shot saved: {path}"),
                            Err(e) => eprintln!("shot failed: {e:#}"),
                        }
                    }
                    *exit_requested = true;
                }
                if let Some(f) = frame {
                    f.present();
                }
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
        let mut map = rustt::map::parse(data).with_context(|| format!("parsing {}", model_path))?;
        // Try to load the sibling .GIZ file for blowup positions.
        if let Some(giz_path) = find_sibling_giz(&model_path) {
            if let Ok(giz_data) = std::fs::read(&giz_path) {
                if let Ok(giz) = rustt::giz::parse_giz(&giz_data) {
                    let before = map.render_parts.len();
                    // Mesh overrides: templates whose SO has cmd_count=0 but
                    // whose mesh exists in room geometry. For "chair_01", the
                    // mesh is at render_part index 1421 (mesh 982, 216 tris).
                    let mut mesh_overrides = std::collections::HashMap::new();
                    // chair_01 SO has cmd_count=0 (no game-model), but its
                    // mesh is in room geometry.  Find it by mesh index.
                    // TODO: generalize this for all levels.
                    if let Some(rp) = map.render_parts.iter().position(|p| p.mesh == 982) {
                        mesh_overrides.insert("chair_01".to_string(), rp);
                    }
                    map.apply_giz_blowups(&giz, &mesh_overrides);
                    let after = map.render_parts.len();
                    if after != before {
                        println!("GIZ: applied {} blowup positions (+{} parts)", giz.blowups.len(), after - before);
                    }
                    map.apply_giz_buildits(&giz);
                    // NOTE: GIZ obstacle positions are NOT applied to render_parts.
                    // They are AI2 trigger/activation positions (where the player
                    // stands to interact), not rendering positions.  Door SOs
                    // already have correct transforms from GSC game-model commands.
                }
            }
        }
        println!(
            "{}: {} render parts, {} meshes, {} materials, {} textures",
            file_name,
            map.render_parts.len(),
            map.meshes.len(),
            map.materials.len(),
            map.textures.len()
        );
        let lights = load_rtl_sibling(&model_path);
        AppData::Map(AppMap {
            file_name,
            map,
            lights,
        })
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

/// World→screen projection matching the 3D camera's view-projection (Y-down
/// screen space, imgui convention). None when the point is behind the camera.
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

fn c32(r: u8, g: u8, b: u8) -> imgui::ImColor32 {
    imgui::ImColor32::from_rgb_f32s(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
}

/// Coordinate overlay for the grid: 2 m cross marks and 4 m labels projected
/// onto the floor around the camera, a marker at the camera's own position,
/// and a camera/bounds HUD. Only drawn when the grid is enabled.
fn draw_coord_overlay(ui: &imgui::Ui, camera: &camera::OrbitCamera, gpu: &GpuRenderer) {
    let io = ui.io();
    let (w, h) = (io.display_size[0], io.display_size[1]);
    if w <= 0.0 || h <= 0.0 {
        return;
    }
    let vp = camera.view_proj(w / h.max(1.0));
    let cam = camera.position();
    let dl = ui.get_foreground_draw_list();

    let yellow = c32(255, 220, 95);
    let amber = c32(255, 185, 55);
    let gray = c32(185, 185, 130);
    let red = c32(255, 85, 85);
    let cyan = c32(120, 235, 255);
    let black = c32(0, 0, 0);

    // Label window scales with camera distance: every 2 m cross marks, 4 m
    // labels close up, 10 m labels across the wide view.
    let half = (camera.distance * 0.8).clamp(20.0, 80.0);
    let step = 2.0f32;
    let mut i = ((cam.x - half) / step).ceil() * step;
    while i <= cam.x + half {
        let mut j = ((cam.z - half) / step).ceil() * step;
        while j <= cam.z + half {
            let p = glam::Vec3::new(i, 0.0, j);
            if let Some((sx, sy)) = project_point(vp, p, w, h) {
                let d2 = (i - cam.x) * (i - cam.x) + (j - cam.z) * (j - cam.z);
                let near = d2 <= 24.0 * 24.0;
                let on = (i.rem_euclid(10.0) == 0.0 && j.rem_euclid(10.0) == 0.0)
                    || (near && i.rem_euclid(4.0) == 0.0 && j.rem_euclid(4.0) == 0.0);
                if on {
                    let text = format!("{},{}", i as i32, j as i32);
                    let col = if i == 0.0 || j == 0.0 { amber } else { yellow };
                    dl.add_text([sx + 1.5, sy - 13.5], black, text.clone());
                    dl.add_text([sx + 1.0, sy - 14.0], col, text);
                } else {
                    dl.add_circle([sx, sy], 2.0, gray).build();
                }
            }
            j += step;
        }
        i += step;
    }

    if let Some((sx, sy)) = project_point(vp, glam::Vec3::new(cam.x, 0.0, cam.z), w, h) {
        dl.add_circle([sx, sy], 5.0, red).build();
        dl.add_circle([sx, sy], 9.0, red).build();
        let text = format!("cam {},{}", cam.x.round() as i32, cam.z.round() as i32);
        dl.add_text([sx + 12.0, sy + 6.0], black, text.clone());
        dl.add_text([sx + 11.0, sy + 5.0], red, text);
    }

    let b = &gpu.scene.bounds;
    let text = format!(
        "cam ({:.2}, {:.2}, {:.2})   bounds c=({:.1}, {:.1}, {:.1}) r={:.1}",
        cam.x,
        cam.y,
        cam.z,
        b.center.x,
        b.center.y,
        b.center.z,
        b.radius,
    );
    dl.add_text([11.0, 11.0], black, text.clone());
    dl.add_text([10.0, 10.0], cyan, text);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_wgsl_validates_under_naga() {
        let src = include_str!("shaders.wgsl");
        let module = naga::front::wgsl::parse_str(src)
            .expect("shaders.wgsl should parse as WGSL");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("shaders.wgsl should pass naga validation");
    }

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
