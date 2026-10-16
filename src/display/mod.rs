//! Answer display with a per-platform backend.
//!
//! * **Linux** — [`wayland`]: a `wlr-layer-shell` overlay panel, always on top,
//!   click-through, top-right. Unmapped when hidden, so a Hyprland compositor
//!   with `no_screen_share` skips it in screen-share frames.
//! * **Everything else** (Windows, macOS, …) — [`console`]: the answer is
//!   printed to stdout. No GUI toolkit, no platform SDK, same behaviour.
//!
//! Both backends expose the same [`OverlayHandle`] / [`OverlayCtl`] pair, so
//! the rest of the app never learns which one is compiled in.

/// Default panel width in logical pixels (overlay backend).
pub const OVERLAY_W: u32 = 440;
/// Default panel height in logical pixels (overlay backend).
pub const OVERLAY_H: u32 = 150;
/// Default distance from the screen edges.
pub const MARGIN: i32 = 12;
#[cfg(target_os = "linux")]
mod wayland;
#[cfg(target_os = "linux")]
pub use wayland::{OverlayCtl, OverlayHandle};

#[cfg(not(target_os = "linux"))]
mod console;
#[cfg(not(target_os = "linux"))]
pub use console::{OverlayCtl, OverlayHandle};

/// Runtime-tunable panel configuration (CLI overrides).
#[derive(Debug, Clone, Copy)]
pub struct OverlayGeometry {
    pub width: u32,
    pub height: u32,
    /// Distance from the top edge (anchor TOP|RIGHT).
    pub margin_top: i32,
    /// Distance from the right edge.
    pub margin_right: i32,
    pub font_size: f32,
    /// Auto-hide the panel this many ms after each answer
    /// (`None` = stay visible). Shrinks the capture-exposure window.
    pub autohide_ms: Option<u64>,
}

impl Default for OverlayGeometry {
    fn default() -> Self {
        Self {
            width: OVERLAY_W,
            height: OVERLAY_H,
            margin_top: MARGIN,
            margin_right: MARGIN,
            font_size: 12.0,
            autohide_ms: None,
        }
    }
}

impl OverlayGeometry {
    /// Parse `--geometry WxH±X±Y [--font-size N]` style flags from args.
    ///
    /// Accepted forms:
    ///   --geometry 500x180        size only
    ///   --geometry 500x180+40+20  size + top/right margins
    ///   --font-size 18
    /// Unknown args are ignored so the flag list can grow.
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        let mut g = Self::default();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--geometry" => {
                    let v = args
                        .get(i + 1)
                        .ok_or("--geometry requires a value WxH[±X±Y]")?;
                    g.apply_geometry(v)?;
                    i += 1;
                }
                "--autohide-ms" => {
                    let v = args.get(i + 1).ok_or("--autohide-ms requires a number")?;
                    let n: u64 = v.parse().map_err(|_| format!("bad --autohide-ms '{v}'"))?;
                    g.autohide_ms = Some(n);
                    i += 1;
                }
                "--font-size" => {
                    let v = args.get(i + 1).ok_or("--font-size requires a number")?;
                    g.font_size = v.parse().map_err(|_| format!("bad --font-size '{v}'"))?;
                    if !(8.0..=72.0).contains(&g.font_size) {
                        return Err(format!("--font-size {} out of range 8..72", g.font_size));
                    }
                    i += 1;
                }
                _ => {}
            }
            i += 1;
        }
        Ok(g)
    }

    fn apply_geometry(&mut self, v: &str) -> Result<(), String> {
        let bad = || format!("bad --geometry '{v}' (want WxH or WxH+X+Y)");
        let (size, offsets) = v.split_once('+').unwrap_or((v, ""));
        let (w, h) = size.split_once('x').ok_or_else(bad)?;
        self.width = w.parse().map_err(|_| bad())?;
        self.height = h.parse().map_err(|_| bad())?;
        if self.width == 0 || self.width > 4096 || self.height == 0 || self.height > 4096 {
            return Err(bad());
        }
        if offsets.is_empty() {
            return Ok(());
        }
        let mut nums = offsets.splitn(2, '+');
        let x: i32 = nums.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
        self.margin_right = x;
        if let Some(y) = nums.next() {
            self.margin_top = y.parse().map_err(|_| bad())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn geometry_parsing() {
        let g = OverlayGeometry::from_args(&[]).unwrap();
        assert_eq!(g.width, OVERLAY_W);
        assert_eq!(g.margin_top, MARGIN);

        let g = OverlayGeometry::from_args(&args(&["--geometry", "500x180+40+20"])).unwrap();
        assert_eq!((g.width, g.height), (500, 180));
        assert_eq!((g.margin_right, g.margin_top), (40, 20));

        let g = OverlayGeometry::from_args(&args(&["--geometry", "300x100", "--font-size", "22"]))
            .unwrap();
        assert_eq!(g.font_size, 22.0);
        assert_eq!(g.margin_right, MARGIN);

        assert!(OverlayGeometry::from_args(&args(&["--geometry", "bogus"])).is_err());
        assert!(
            OverlayGeometry::from_args(&args(&["--font-size", "999"])).is_err(),
            "out of range must fail"
        );

        let g = OverlayGeometry::from_args(&args(&["--autohide-ms", "5000"])).unwrap();
        assert_eq!(g.autohide_ms, Some(5000));
        let g = OverlayGeometry::from_args(&[]).unwrap();
        assert_eq!(g.autohide_ms, None);
    }

    #[test]
    fn unknown_flags_are_ignored() {
        let g = OverlayGeometry::from_args(&args(&["--whatever", "1", "--geometry", "200x200"]))
            .unwrap();
        assert_eq!((g.width, g.height), (200, 200));
    }
}
