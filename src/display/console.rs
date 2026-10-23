//! Portable backend: print the answer to stdout.
//!
//! Used on every platform without the Wayland overlay (Windows, macOS).
//! Keeping the app alive and useful there — an interview answer in the
//! terminal beats a compile error on `cargo build` — without pulling in a
//! windowing toolkit or a platform SDK.
//!
//! The panel geometry (`--geometry`, `--font-size`, `--autohide-ms`) has no
//! meaning without a window and is ignored here.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use log::info;

use super::OverlayGeometry;

/// Clonable control endpoint — many owners can drive one display.
#[derive(Clone)]
pub struct OverlayCtl {
    ready: Arc<AtomicBool>,
}

impl OverlayCtl {
    pub fn set_status(&self, text: &str) {
        self.print(text);
    }

    pub fn push_answer(&self, text: &str) {
        self.print(text);
    }

    /// Stdout has no place to replace text — every entry just prints once.
    fn print(&self, text: &str) {
        if !self.ready.load(Ordering::SeqCst) {
            return;
        }
        println!("\n═══ LIhO ═══\n{text}\n");
    }
}

/// Owner-side handle. Printing is synchronous, so there is no background
/// thread — but the shape matches the overlay backend, keeping callers
/// identical on both platforms.
pub struct OverlayHandle {
    ctl: OverlayCtl,
    /// Always true: stdout needs no compositor handshake.
    pub configured: Arc<AtomicBool>,
}

impl OverlayHandle {
    pub fn launch(_geom: OverlayGeometry) -> Result<Self, String> {
        info!("console display enabled (no GUI backend on this platform)");
        Ok(Self {
            ctl: OverlayCtl {
                ready: Arc::new(AtomicBool::new(true)),
            },
            configured: Arc::new(AtomicBool::new(true)),
        })
    }

    pub fn ctl(&self) -> &OverlayCtl {
        &self.ctl
    }
}

impl Drop for OverlayHandle {
    fn drop(&mut self) {
        info!("console display stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle() -> OverlayHandle {
        OverlayHandle::launch(OverlayGeometry::default()).unwrap()
    }

    #[test]
    fn launch_is_immediately_configured() {
        let o = handle();
        assert!(o.configured.load(Ordering::Relaxed));
    }

    #[test]
    fn printing_is_always_safe() {
        let o = handle();
        o.ctl().set_status("думаю");
        o.ctl().push_answer("привет");
    }

    #[test]
    fn ctl_clone_shares_state() {
        let o = handle();
        let a = o.ctl().clone();
        let b = a.clone();
        a.push_answer("печатается");
        assert!(b.ready.load(Ordering::SeqCst));
    }
}
