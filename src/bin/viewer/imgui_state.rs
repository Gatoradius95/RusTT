//! Shared imgui window state used by both the viewer and the game client
//! (the game includes this file via `#[path = "../viewer/imgui.rs"]`).

use std::time::Instant;

use imgui::FontSource;
use imgui_wgpu::RendererConfig;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::window::Window;

pub struct ImguiState {
    pub context: imgui::Context,
    pub platform: WinitPlatform,
    pub renderer: imgui_wgpu::Renderer,
    pub last_frame: Instant,
    pub last_cursor: Option<imgui::MouseCursor>,
}

impl ImguiState {
    pub fn new(
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
            RendererConfig {
                texture_format: format,
                depth_format: Some(wgpu::TextureFormat::Depth32Float),
                ..RendererConfig::new()
            }
        } else {
            RendererConfig {
                texture_format: format,
                depth_format: Some(wgpu::TextureFormat::Depth32Float),
                ..RendererConfig::new_srgb()
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

    pub fn rebuild_font(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scale_factor: f64,
    ) {
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