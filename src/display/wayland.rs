//! Linux backend: Wayland overlay via wlr-layer-shell.
//!
//! (Non-Linux platforms use the console backend, see `super::console`.)
//!
//! A small always-on-top panel (`Layer::Overlay`) anchored top-right,
//! translucent background, no keyboard grab, **empty input region** so
//! mouse clicks pass through to whatever is underneath.
//! The Wayland connection lives on a dedicated thread; the owner talks to
//! it through a calloop channel (`set_status` / `push_answer` / Drop=stop).
//!
//! Note: like any visible pixels, the overlay is part of the composited
//! output — a full-screen capture sees it. It does NOT appear in window
//! lists, docks, alt-tab, or window-scoped OBS sources. With a Hyprland
//! compositor configured with `no_screen_share` for this namespace, the
//! panel is skipped in share frames entirely; `--autohide-ms` shrinks the
//! window during which it is on screen at all.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use cosmic_text::{Attrs, Buffer, Color, FontSystem, Metrics, Shaping, SwashCache};
use log::{debug, info, warn};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, Region},
    delegate_registry,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        WaylandSurface,
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
    },
    shm::{Shm, ShmHandler, slot::SlotPool},
};
use tiny_skia::{Paint, PathBuilder, Pixmap, Stroke, Transform};
use wayland_client::{
    Connection, QueueHandle,
    protocol::{wl_shm, wl_surface},
};

use super::{MARGIN, OVERLAY_H, OVERLAY_W, OverlayGeometry};

/// Inner padding of the rendered panel.
const PAD: f32 = 12.0;
/// Vertical gap around the separator rules between stacked entries.
const BLOCK_GAP: f32 = 8.0;
/// Separator rule thickness between answer blocks.
const RULE: f32 = 1.0;
/// History bound — rendering drops what does not fit anyway.
const MAX_BLOCKS: usize = 20;

pub(crate) enum Cmd {
    /// Replace the transient status line on top of the stack.
    Status(String),
    /// A finished answer: newest block on top, older ones shift down.
    Answer(String),
    Stop,
}

/// Clonable control endpoint — many owners can drive one overlay
/// (pipeline pushes answers, main sets status, signal thread stops it).
#[derive(Clone)]
pub struct OverlayCtl {
    tx: calloop::channel::Sender<Cmd>,
}

impl OverlayCtl {
    pub fn set_status(&self, text: &str) {
        if self.tx.send(Cmd::Status(text.to_string())).is_err() {
            debug!("overlay already closed");
        }
    }

    pub fn push_answer(&self, text: &str) {
        if self.tx.send(Cmd::Answer(text.to_string())).is_err() {
            debug!("overlay already closed");
        }
    }
}

/// Owner-side handle. Dropping it closes the overlay and joins the thread.
pub struct OverlayHandle {
    tx: calloop::channel::Sender<Cmd>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// True after the compositor's first configure — proves the layer
    /// surface was accepted and mapped.
    pub configured: std::sync::Arc<AtomicBool>,
    ctl: OverlayCtl,
}

impl OverlayHandle {
    /// Spawn the overlay thread. Errors when there is no compositor.
    pub fn launch(geom: OverlayGeometry) -> Result<Self, String> {
        let configured = std::sync::Arc::new(AtomicBool::new(false));
        let flag = std::sync::Arc::clone(&configured);
        let (ready_tx, ready_rx) =
            mpsc::sync_channel::<Result<calloop::channel::Sender<Cmd>, String>>(1);

        let thread = std::thread::Builder::new()
            .name("liho-overlay".into())
            .spawn(move || {
                if let Err(e) = run_overlay(geom, ready_tx, flag) {
                    warn!("overlay thread failed: {e}");
                }
            })
            .map_err(|e| format!("spawn overlay thread: {e}"))?;

        let tx = ready_rx
            .recv()
            .map_err(|_| "overlay thread died before signalling".to_string())??;

        info!("Wayland overlay launched");
        Ok(Self {
            tx: tx.clone(),
            thread: Some(thread),
            configured,
            ctl: OverlayCtl { tx },
        })
    }

    /// Clonable control endpoint for secondary owners.
    pub fn ctl(&self) -> &OverlayCtl {
        &self.ctl
    }
}

impl Drop for OverlayHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Stop);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        info!("Wayland overlay stopped");
    }
}

struct AppState {
    registry_state: RegistryState,
    output_state: OutputState,
    shm: Shm,
    conn: Connection,
    layer: LayerSurface,
    pool: SlotPool,
    width: u32,
    height: u32,
    /// Transient line above the stack ("Думаю над ответом…").
    status: String,
    /// Answer blocks, newest first.
    blocks: Vec<String>,
    visible: bool,
    dirty: bool,
    hide_at: Option<std::time::Instant>,
    /// Attach is illegal until the first configure has been received+acked.
    configured_done: bool,
    exit: bool,
    font_system: FontSystem,
    swash: SwashCache,
    font_size: f32,
    autohide_ms: Option<u64>,
    configured_flag: std::sync::Arc<AtomicBool>,
}

fn run_overlay(
    geom: OverlayGeometry,
    ready_tx: mpsc::SyncSender<Result<calloop::channel::Sender<Cmd>, String>>,
    configured_flag: std::sync::Arc<AtomicBool>,
) -> Result<(), String> {
    let conn = Connection::connect_to_env().map_err(|e| format!("wayland connect: {e}"))?;
    let (globals, event_queue) = wayland_client::globals::registry_queue_init(&conn)
        .map_err(|e| format!("registry: {e}"))?;
    let qh = event_queue.handle();

    let compositor = CompositorState::bind(&globals, &qh).map_err(|_| "wl_compositor missing")?;
    let shm = Shm::bind(&globals, &qh).map_err(|_| "wl_shm missing")?;
    let layer_shell = LayerShell::bind(&globals, &qh)
        .map_err(|_| "zwlr_layer_shell_v1 missing — compositor does not support layer shell")?;

    let surface = compositor.create_surface(&qh);
    // Click-through: an empty input region applied before the initial commit.
    // A fresh sctk Region contains nothing and destroys itself on drop.
    let region = Region::new(&compositor).map_err(|e| format!("region: {e}"))?;
    surface.set_input_region(Some(region.wl_region()));
    drop(region);

    let layer =
        layer_shell.create_layer_surface(&qh, surface.clone(), Layer::Overlay, Some("LIhO"), None);
    layer.set_anchor(Anchor::TOP | Anchor::RIGHT);
    layer.set_margin(MARGIN, MARGIN, MARGIN, MARGIN);
    layer.set_keyboard_interactivity(KeyboardInteractivity::None);
    layer.set_size(OVERLAY_W, OVERLAY_H);
    layer.commit(); // initial commit → configure follows

    let pool = SlotPool::new((geom.width * geom.height * 4) as usize, &shm)
        .map_err(|e| format!("pool: {e}"))?;

    let mut state = AppState {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        shm,
        conn: conn.clone(),
        layer,
        pool,
        width: geom.width,
        height: geom.height,
        status: String::new(),
        blocks: Vec::new(),
        visible: false,
        dirty: false,
        hide_at: None,
        configured_done: false,
        exit: false,
        font_system: FontSystem::new(),
        swash: SwashCache::new(),
        font_size: geom.font_size,
        autohide_ms: geom.autohide_ms,
        configured_flag,
    };

    let mut event_loop: calloop::EventLoop<AppState> =
        calloop::EventLoop::try_new().map_err(|e| format!("calloop: {e}"))?;
    let loop_handle = event_loop.handle();

    calloop_wayland_source::WaylandSource::new(conn.clone(), event_queue)
        .insert(loop_handle.clone())
        .map_err(|e| format!("wayland source insert: {e}"))?;

    let (cmd_tx, cmd_rx) = calloop::channel::channel::<Cmd>();
    loop_handle
        .insert_source(cmd_rx, |ev, _, app| match ev {
            calloop::channel::Event::Msg(cmd) => apply_cmd(cmd, app),
            calloop::channel::Event::Closed => {}
        })
        .map_err(|e| format!("cmd source: {e}"))?;

    ready_tx.send(Ok(cmd_tx)).map_err(|_| "owner gone")?;

    while !state.exit {
        event_loop
            .dispatch(Some(Duration::from_millis(200)), &mut state)
            .map_err(|e| format!("dispatch: {e}"))?;
        // Protocol rule: ack_configure must reach the compositor BEFORE any
        // buffer attach for that configure. Handlers (which ack) ran during
        // dispatch; flush now so our frame's attach can't overtake it.
        if let Err(e) = state.conn.flush() {
            debug!("overlay pre-draw flush: {e}");
        }
        if state
            .hide_at
            .is_some_and(|deadline| std::time::Instant::now() >= deadline && state.visible)
        {
            state.visible = false;
            state.hide_at = None;
            // True unmap: an unmapped layer produces no redaction box in share
            // frames (Hyprland no_screen_share pass).
            unmap(&state.layer);
            state.configured_done = false;
            info!("overlay auto-hidden (unmapped)");
        }
        if !state.dirty || !state.configured_done {
            continue;
        }
        if state.visible {
            draw_frame(&mut state);
        } else {
            // Hidden: nothing to render; surface stays unmapped until the
            // next Status/Answer triggers a map-request commit.
            state.dirty = false;
            continue;
        }
        state.dirty = false;
        if let Err(e) = state.conn.flush() {
            debug!("overlay post-frame flush: {e}");
        }
    }
    Ok(())
}

fn apply_cmd(cmd: Cmd, app: &mut AppState) {
    match cmd {
        Cmd::Stop => app.exit = true,
        Cmd::Status(t) => {
            if app.status != t || !app.visible {
                app.status = t;
                app.visible = true;
                arm_autohide(app);
                app.dirty = true;
            }
        }
        Cmd::Answer(t) => {
            // New answer goes on top; the status line it was announced by
            // has served its purpose.
            app.status.clear();
            app.blocks.insert(0, t);
            app.blocks.truncate(MAX_BLOCKS);
            app.visible = true;
            arm_autohide(app);
            app.dirty = true;
        }
    }
}

/// Re-arm the auto-hide timer on every fresh content.
fn arm_autohide(app: &mut AppState) {
    app.hide_at = app
        .autohide_ms
        .map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
}

/// SlotPool grows by one buffer per redraw; redraws happen only on
/// content changes (a few per minute at most), so this stays bounded.
fn draw_frame(app: &mut AppState) {
    let (w, h) = (app.width.max(1), app.height.max(1));
    // Render first — it touches only text+fonts, not the pool borrow.
    let rgba = render_panel(
        w,
        h,
        app.font_size,
        &app.status,
        &app.blocks,
        &mut app.font_system,
        &mut app.swash,
    );
    let Ok((buffer, canvas)) =
        app.pool
            .create_buffer(w as i32, h as i32, (w * 4) as i32, wl_shm::Format::Argb8888)
    else {
        warn!("overlay: buffer alloc failed");
        return;
    };
    canvas.copy_from_slice(&rgba);

    let surface = app.layer.wl_surface();
    surface.damage_buffer(0, 0, w as i32, h as i32);
    if buffer.attach_to(surface).is_ok() {
        app.layer.commit();
        debug!(
            "overlay frame committed ({} block(s), {} status chars)",
            app.blocks.len(),
            app.status.chars().count()
        );
    }
}

/// Unmap the layer surface: attach a NULL buffer and commit. An unmapped
/// layer is skipped by Hyprland's no_screen_share pass entirely — no black
/// redaction box appears over its region in captures.
fn unmap(layer: &LayerSurface) {
    layer.wl_surface().attach(None, 0, 0);
    layer.commit();
}

/// Render panel (background + stacked text) into premultiplied ARGB8888.
/// Pure function of its inputs — unit-testable headless.
///
/// Layout: the transient status line on top, then answer blocks newest
/// first, separated by thin rules. Blocks that do not fit the panel are
/// dropped whole; the newest one is only clipped.
pub(crate) fn render_panel(
    w: u32,
    h: u32,
    font_size: f32,
    status: &str,
    blocks: &[String],
    font_system: &mut FontSystem,
    swash: &mut SwashCache,
) -> Vec<u8> {
    let mut pixmap = Pixmap::new(w, h).expect("pixmap alloc");

    // Translucent dark background + subtle border.
    let mut pb = PathBuilder::new();
    pb.push_rect(
        tiny_skia::Rect::from_xywh(0.5, 0.5, w as f32 - 1.0, h as f32 - 1.0).expect("rect"),
    );
    let path = pb.finish().expect("path");

    let bg = Paint {
        shader: tiny_skia::Shader::SolidColor(tiny_skia::Color::from_rgba8(10, 10, 16, 178)),
        anti_alias: true,
        ..Default::default()
    };
    pixmap.fill_path(
        &path,
        &bg,
        tiny_skia::FillRule::Winding,
        Transform::identity(),
        None,
    );

    let border_paint = Paint {
        shader: tiny_skia::Shader::SolidColor(tiny_skia::Color::from_rgba8(255, 255, 255, 38)),
        anti_alias: true,
        ..Default::default()
    };
    pixmap.stroke_path(
        &path,
        &border_paint,
        &Stroke {
            width: 1.0,
            ..Default::default()
        },
        Transform::identity(),
        None,
    );

    let max_w = w as f32 - PAD * 2.0;
    let mut y = PAD;

    if !status.trim().is_empty() {
        let bh = text_height(status, max_w, font_size, font_system);
        draw_text(&mut pixmap, font_system, swash, status, y, font_size);
        y += bh + BLOCK_GAP;
    }

    for (i, (text, bh)) in fitting_blocks(blocks, max_w, h as f32 - PAD - y, font_size, font_system)
        .iter()
        .enumerate()
    {
        if i > 0 {
            y += BLOCK_GAP;
            draw_rule(&mut pixmap, y);
            y += RULE + BLOCK_GAP;
        }
        draw_text(&mut pixmap, font_system, swash, text, y, font_size);
        y += bh;
    }

    pixmap.data().to_vec()
}

/// Height `text` occupies when wrapped at `width` (0 for empty text).
fn text_height(text: &str, width: f32, font_size: f32, font_system: &mut FontSystem) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let line_height = font_size * 1.3;
    let mut buffer = Buffer::new(font_system, Metrics::new(font_size, line_height));
    buffer.set_size(Some(width), None);
    buffer.set_text(text, &Attrs::new(), Shaping::Advanced, None);
    buffer.shape_until_scroll(font_system, true);
    buffer.layout_runs().count().max(1) as f32 * line_height
}

/// Which leading blocks fit in `avail` px of stack height, newest first.
/// Whole blocks only — the cut is at a block boundary — and the first block
/// is always kept (it is clipped instead of dropped).
fn fitting_blocks<'a>(
    blocks: &'a [String],
    width: f32,
    avail: f32,
    font_size: f32,
    font_system: &mut FontSystem,
) -> Vec<(&'a str, f32)> {
    let mut out: Vec<(&str, f32)> = Vec::new();
    let mut used = 0.0;
    for b in blocks {
        let bh = text_height(b, width, font_size, font_system);
        let extra = if out.is_empty() {
            0.0
        } else {
            BLOCK_GAP * 2.0 + RULE
        };
        if !out.is_empty() && used + extra + bh > avail {
            break;
        }
        used += extra + bh;
        out.push((b.as_str(), bh));
    }
    out
}

/// Draw one text block at the left padding, top edge `y`, clipped to the
/// panel's inner width and the space below `y`.
fn draw_text(
    pixmap: &mut Pixmap,
    font_system: &mut FontSystem,
    swash: &mut SwashCache,
    text: &str,
    y: f32,
    font_size: f32,
) {
    if text.is_empty() {
        return;
    }
    let line_height = font_size * 1.3;
    let mut buffer = Buffer::new(font_system, Metrics::new(font_size, line_height));
    buffer.set_size(
        Some(pixmap.width() as f32 - PAD * 2.0),
        Some(pixmap.height() as f32 - PAD - y),
    );
    buffer.set_text(text, &Attrs::new(), Shaping::Advanced, None);
    buffer.shape_until_scroll(font_system, true);

    let white = Color::rgb(235, 235, 240);
    let pw = pixmap.width() as i32;
    let ph = pixmap.height() as i32;
    let (ox, oy) = (PAD as i32, y as i32);
    buffer.draw(font_system, swash, white, |gx, gy, gw, gh, color| {
        for yy in 0..gh as i32 {
            for xx in 0..gw as i32 {
                let px = ox + gx + xx;
                let py = oy + gy + yy;
                if px < 0 || py < 0 || px >= pw || py >= ph {
                    continue;
                }
                blend_over(pixmap.data_mut(), pw, px, py, color);
            }
        }
    });
}

/// Thin separator rule between stacked answers.
fn draw_rule(pixmap: &mut Pixmap, y: f32) {
    let mut pb = PathBuilder::new();
    pb.push_rect(
        tiny_skia::Rect::from_xywh(PAD, y, pixmap.width() as f32 - PAD * 2.0, RULE).expect("rect"),
    );
    let path = pb.finish().expect("path");
    let paint = Paint {
        shader: tiny_skia::Shader::SolidColor(tiny_skia::Color::from_rgba8(255, 255, 255, 40)),
        anti_alias: true,
        ..Default::default()
    };
    pixmap.fill_path(
        &path,
        &paint,
        tiny_skia::FillRule::Winding,
        Transform::identity(),
        None,
    );
}

/// Premultiplied source-over: out = src_pm + dst_pm * (1 - src_a).
/// `data` is ARGB8888 little-endian premultiplied ([b, g, r, a] bytes).
fn blend_over(data: &mut [u8], stride_px: i32, px: i32, py: i32, color: Color) {
    let sa = color.a() as u32;
    if sa == 0 {
        return;
    }
    let idx = ((py * stride_px + px) * 4) as usize;
    let inv = 255 - sa;
    let mix = |d: u8, s: u8| -> u8 {
        let v = d as u32 * inv / 255 + s as u32 * sa / 255;
        v.min(255) as u8
    };
    data[idx] = mix(data[idx], (color.b() as u32 * sa / 255).min(255) as u8);
    data[idx + 1] = mix(data[idx + 1], (color.g() as u32 * sa / 255).min(255) as u8);
    data[idx + 2] = mix(data[idx + 2], (color.r() as u32 * sa / 255).min(255) as u8);
    data[idx + 3] = mix(data[idx + 3], 255);
}

impl CompositorHandler for AppState {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
    }
    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wayland_client::protocol::wl_output::Transform,
    ) {
    }
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {}
    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wayland_client::protocol::wl_output::WlOutput,
    ) {
    }
    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wayland_client::protocol::wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for AppState {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
        self.exit = true;
    }
    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _: u32,
    ) {
        self.width = configure.new_size.0.max(1);
        self.height = configure.new_size.1.max(1);
        self.configured_done = true;
        if !self.configured_flag.swap(true, Ordering::SeqCst) {
            self.status = "LIhO готов".to_string();
            self.visible = true;
            self.dirty = true;
        }
    }
}

impl ShmHandler for AppState {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl OutputHandler for AppState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wayland_client::protocol::wl_output::WlOutput,
    ) {
    }
    fn update_output(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wayland_client::protocol::wl_output::WlOutput,
    ) {
    }
    fn output_destroyed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wayland_client::protocol::wl_output::WlOutput,
    ) {
    }
}

delegate_registry!(AppState);

impl ProvidesRegistryState for AppState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

smithay_client_toolkit::delegate_dispatch2!(AppState);

#[cfg(test)]
mod tests {
    use super::*;

    fn render(text: &str) -> Vec<u8> {
        stack("", &[text.to_string()])
    }

    /// Panel with one transient status line and a stack of answer blocks.
    fn stack(status: &str, blocks: &[String]) -> Vec<u8> {
        let mut fs = FontSystem::new();
        let mut sw = SwashCache::new();
        render_panel(OVERLAY_W, OVERLAY_H, 16.0, status, blocks, &mut fs, &mut sw)
    }

    fn bright_pixel_count(rgba: &[u8]) -> usize {
        rgba.as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[2] > 170 && p[3] > 60)
            .count()
    }

    #[test]
    fn background_is_drawn_even_without_text() {
        let rgba = render("");
        assert_eq!(rgba.len(), (OVERLAY_W * OVERLAY_H * 4) as usize);
        // Background alpha ~178 everywhere except AA edges.
        let opaque = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[3] > 100)
            .count();
        assert!(opaque > (OVERLAY_W * OVERLAY_H) as usize * 9 / 10);
        assert_eq!(bright_pixel_count(&rgba), 0, "no text expected");
    }

    #[test]
    fn text_produces_bright_pixels() {
        let rgba = render("Привет, интервью!");
        assert!(bright_pixel_count(&rgba) > 150, "glyph pixels missing");
    }

    #[test]
    fn long_text_wraps_into_lower_half() {
        let text = "Расскажите о своём опыте работы с распределёнными системами и очередями сообщений. Какие гарантии доставки вы использовали?";
        let rgba = render(text);
        let w = OVERLAY_W as usize;
        let lower_bright = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .skip((OVERLAY_H as usize / 2) * w)
            .filter(|p| p[2] > 170 && p[3] > 60)
            .count();
        assert!(lower_bright > 50, "text did not wrap to lower half");
    }

    #[test]
    fn text_height_counts_wrapped_lines() {
        let mut fs = FontSystem::new();
        let one = text_height("одна строка", 400.0, 16.0, &mut fs);
        let two = text_height("две\nстроки", 400.0, 16.0, &mut fs);
        let wrapped = text_height(&"очень длинная строка ".repeat(8), 120.0, 16.0, &mut fs);
        assert!(one > 0.0);
        assert!((two - one * 2.0).abs() < 0.5, "{one} vs {two}");
        assert!(wrapped > two * 2.0, "narrow width must wrap: {wrapped}");
        assert_eq!(text_height("", 400.0, 16.0, &mut fs), 0.0);
    }

    #[test]
    fn stack_keeps_only_the_newest_blocks_that_fit() {
        let mut fs = FontSystem::new();
        let blocks: Vec<String> = ["новый", "средний", "старый"].map(String::from).to_vec();
        let line = 16.0 * 1.3;
        // Generous room for two one-line blocks, never three.
        let shown = fitting_blocks(&blocks, 400.0, line * 2.0 + 30.0, 16.0, &mut fs);
        let texts: Vec<&str> = shown.iter().map(|(t, _)| *t).collect();
        assert_eq!(texts, ["новый", "средний"], "the oldest drops off");
    }

    #[test]
    fn stack_stops_at_the_first_block_that_does_not_fit() {
        let mut fs = FontSystem::new();
        let blocks = vec![
            "короткий".to_string(),
            "слово ".repeat(60),
            "хвост".to_string(),
        ];
        let line = 16.0 * 1.3;
        let shown = fitting_blocks(&blocks, 200.0, line * 4.0, 16.0, &mut fs);
        assert_eq!(shown.len(), 1, "a block must not leave a hole in the stack");
    }

    #[test]
    fn first_block_is_kept_even_when_taller_than_the_panel() {
        let mut fs = FontSystem::new();
        let blocks = vec!["слово ".repeat(60)];
        let shown = fitting_blocks(&blocks, 200.0, 16.0 * 1.3, 16.0, &mut fs);
        assert_eq!(shown.len(), 1, "the newest answer is never dropped");
    }

    #[test]
    fn stacked_answers_render_newest_on_top() {
        let one = stack("", &["новый ответ".to_string()]);
        let two = stack("", &["новый ответ".to_string(), "старый ответ".to_string()]);
        let below = |rgba: &Vec<u8>| {
            rgba.as_chunks::<4>()
                .0
                .iter()
                .skip(45 * OVERLAY_W as usize)
                .filter(|p| p[2] > 170 && p[3] > 60)
                .count()
        };
        assert_eq!(below(&one), 0, "one block stays on top");
        assert!(below(&two) > 30, "the older answer must shift below");
    }

    #[test]
    fn status_sits_above_the_newest_answer() {
        let with = stack("Думаю над ответом…", &["ответ".to_string()]);
        let without = stack("", &["ответ".to_string()]);
        let above = |rgba: &Vec<u8>| {
            rgba.as_chunks::<4>()
                .0
                .iter()
                .take(45 * OVERLAY_W as usize)
                .filter(|p| p[2] > 170 && p[3] > 60)
                .count()
        };
        assert!(
            above(&with) > above(&without),
            "the status line must add text on top"
        );
    }

    #[test]
    fn rendering_is_deterministic() {
        assert_eq!(render("тест"), render("тест"));
    }

    /// Live check against the running compositor.
    /// Run manually: cargo test live_overlay -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "live overlay: needs a running Wayland session"]
    async fn live_overlay_maps_and_receives_text() {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(true)
            .try_init();
        let overlay = OverlayHandle::launch(OverlayGeometry::default()).expect("launch overlay");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline && !overlay.configured.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            overlay.configured.load(Ordering::SeqCst),
            "compositor never sent configure"
        );
        overlay
            .ctl()
            .push_answer("Проверка оверлея: вопрос интервью появится здесь");
        tokio::time::sleep(Duration::from_millis(600)).await;
        drop(overlay); // clean shutdown + join
    }
}
