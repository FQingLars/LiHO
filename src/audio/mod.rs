//! Audio capture: one stream per side of the conversation.
//!
//! | [`Source::User`] | microphone (cpal) | «Пользователь» |
//! | [`Source::Interviewer`] | system output, a.k.a. loopback | «Интервьюер» |
//!
//! Both streams are downmixed to mono, resampled to [`TARGET_RATE`] and
//! quantized to i16 — exactly what the T-one ONNX model consumes. The
//! loopback backend is platform-specific (PipeWire monitor on Linux, WASAPI
//! loopback on Windows); when it cannot be started the caller falls back to
//! the microphone alone.
//!
//! Chunks travel on a bounded channel and are **dropped, never blocked**, if
//! the consumer lags: a blocked audio callback is an audible glitch, a dropped
//! 20 ms chunk is not.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;

use log::info;

pub mod mic;

#[cfg(target_os = "linux")]
mod loopback_linux;
#[cfg(target_os = "linux")]
use loopback_linux as loopback;

#[cfg(target_os = "windows")]
mod loopback_windows;
#[cfg(target_os = "windows")]
use loopback_windows as loopback;

/// T-one sample rate (telephony wideband).
pub const TARGET_RATE: u32 = 8_000;

/// Capture rate we aim for: a cheap 2:1 step-down to [`TARGET_RATE`] while
/// still capturing the full band a wideband microphone delivers.
pub(crate) const COMFORTABLE_RATE: u32 = 16_000;

/// Which side of the conversation a stream carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The candidate — heard through the microphone.
    User,
    /// The interviewer — heard through the system output (call, browser).
    Interviewer,
}

impl Source {
    /// Transcript tag written in front of every phrase of this stream.
    pub fn label(self) -> &'static str {
        match self {
            Source::User => "Пользователь",
            Source::Interviewer => "Интервьюер",
        }
    }
}

/// Handle to a running capture. Dropping it stops the stream and joins the
/// backend thread.
pub struct CaptureHandle {
    source: Source,
    rx: Receiver<Vec<i16>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Backend-specific "please stop" action: a channel send for cpal/WASAPI,
    /// a PipeWire channel send for the Linux monitor.
    stop: Option<Box<dyn FnOnce() + Send>>,
    /// Chunks discarded because the consumer lagged (shared with the
    /// backend callback, so the count stays live).
    pub dropped_chunks: Arc<AtomicU64>,
}

impl CaptureHandle {
    /// Which conversation side this stream carries.
    pub fn source(&self) -> Source {
        self.source
    }

    /// Non-blocking receive of one 8 kHz mono i16 chunk.
    pub fn try_recv(&self) -> Option<Vec<i16>> {
        self.rx.try_recv().ok()
    }

    pub(crate) fn new(
        source: Source,
        rx: Receiver<Vec<i16>>,
        stop: Box<dyn FnOnce() + Send>,
        thread: std::thread::JoinHandle<()>,
        dropped_chunks: Arc<AtomicU64>,
    ) -> Self {
        Self {
            source,
            rx,
            thread: Some(thread),
            stop: Some(stop),
            dropped_chunks,
        }
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            stop();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        info!(
            "{} capture stopped ({} chunks dropped)",
            self.source.label(),
            self.dropped_chunks.load(Ordering::Relaxed)
        );
    }
}

/// Start capturing one conversation side.
///
/// `Source::User` uses the default microphone on every platform;
/// `Source::Interviewer` needs a loopback backend and may legitimately fail
/// (no output device, no permission, unsupported OS).
pub fn start_capture(source: Source) -> Result<CaptureHandle, String> {
    match source {
        Source::User => mic::start(),
        Source::Interviewer => loopback::start(),
    }
}

/// Linear-interpolating resampler with fractional phase carried across
/// chunks, so consecutive conversions form one continuous stream.
pub(crate) struct RateConverter {
    phase: f64,
}

impl RateConverter {
    pub(crate) fn new() -> Self {
        Self { phase: 0.0 }
    }

    pub(crate) fn convert(&mut self, input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        if src_rate == dst_rate || input.is_empty() {
            return input.to_vec();
        }
        let step = f64::from(src_rate) / f64::from(dst_rate);
        let mut out = Vec::with_capacity((input.len() as f64 / step) as usize + 2);
        let mut pos = self.phase;
        while (pos as usize) < input.len() {
            let i = pos as usize;
            let frac = (pos - i as f64) as f32;
            let s0 = input[i];
            let s1 = input.get(i + 1).copied().unwrap_or(s0);
            out.push(s0 + (s1 - s0) * frac);
            pos += step;
        }
        self.phase = pos - input.len() as f64;
        out
    }
}

/// Average channels of an interleaved buffer into mono.
pub(crate) fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Quantize to i16 with clamping — the ONNX model rejects out-of-range samples.
pub(crate) fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32_767.0) as i16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_the_agreed_speaker_tags() {
        assert_eq!(Source::User.label(), "Пользователь");
        assert_eq!(Source::Interviewer.label(), "Интервьюер");
    }

    #[test]
    fn downmix_stereo_averages_frames() {
        let stereo = [0.0_f32, 1.0, 0.5, -0.5, 1.0, 0.0];
        assert_eq!(downmix_to_mono(&stereo, 2), vec![0.5, 0.0, 0.5]);
    }

    #[test]
    fn downmix_mono_is_passthrough() {
        let m = [0.25_f32, -0.75];
        assert_eq!(downmix_to_mono(&m, 1), vec![0.25, -0.75]);
    }

    #[test]
    fn converter_identity_when_rates_equal() {
        let mut c = RateConverter::new();
        let input = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(c.convert(&input, TARGET_RATE, TARGET_RATE), input);
    }

    #[test]
    fn converter_48k_to_8k_produces_sixth_length() {
        let mut c = RateConverter::new();
        let input: Vec<f32> = (0..4800).map(|i| i as f32 / 4800.0).collect();
        let out = c.convert(&input, 48_000, TARGET_RATE);
        assert_eq!(out.len(), 800);
        assert!(out.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn converter_44k1_to_8k_keeps_rhythm() {
        let mut c = RateConverter::new();
        let input = vec![0.5_f32; 441];
        let out = c.convert(&input, 44_100, TARGET_RATE);
        // 44100/8000 = 5.5125 samples per output sample: 80 out (±1 rounding).
        assert!((out.len() as i64 - 80).abs() <= 1, "got {}", out.len());
        assert!(out.iter().all(|s| (s - 0.5).abs() < 1e-3));
    }

    #[test]
    fn converter_phase_carries_across_chunks() {
        let full: Vec<f32> = (0..9600).map(|i| i as f32).collect();
        let mut whole = RateConverter::new();
        let expect_len = whole.convert(&full, 48_000, 8_000).len();

        let mut split = RateConverter::new();
        let mut got = Vec::new();
        for part in full.chunks(1000) {
            got.extend(split.convert(part, 48_000, 8_000));
        }
        assert_eq!(got.len(), expect_len);
    }

    #[test]
    fn converter_silence_stays_silent() {
        let mut c = RateConverter::new();
        let silence = vec![0.0_f32; 480];
        assert!(c.convert(&silence, 48_000, 8_000).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn converter_empty_input_is_empty() {
        let mut c = RateConverter::new();
        assert!(c.convert(&[], 48_000, 8_000).is_empty());
    }

    #[test]
    fn to_i16_scales_and_clamps() {
        assert_eq!(to_i16(&[0.0, 1.0, -1.0]), vec![0, 32_767, -32_767]);
        assert_eq!(to_i16(&[2.0, -2.0]), vec![32_767, -32_767]);
        assert_eq!(to_i16(&[0.5])[0], 16_383);
    }
}
