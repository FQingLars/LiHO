//! Cross-platform microphone capture via **cpal**.
//!
//! One code path for every desktop OS: WASAPI (Windows), PulseAudio/ALSA
//! (Linux), CoreAudio (macOS). The previous PipeWire implementation could only
//! grab the sink monitor on Linux, which blocked Windows entirely.
//!
//! The device stream runs on its own thread; consumers read chunks through an
//! mpsc channel (chunks are dropped, never blocked, when the consumer lags).
//! Every chunk is already downmixed to mono, resampled to
//! [`TARGET_RATE`] and quantized to i16 — exactly what the T-one ONNX model
//! consumes (`signal` input, int16, 8 kHz).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{debug, error, info};

/// T-one sample rate (telephony wideband).
pub const TARGET_RATE: u32 = 8_000;
/// Capture rate we aim for: a cheap 2:1 step-down to [`TARGET_RATE`] while
/// still capturing the full band a wideband microphone delivers.
const COMFORTABLE_RATE: u32 = 16_000;

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

/// Handle to a running capture. Dropping it stops the stream and joins
/// the cpal thread.
pub struct CaptureHandle {
    stop_tx: Option<mpsc::Sender<()>>,
    rx: mpsc::Receiver<Vec<i16>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Chunks discarded because the consumer lagged (realtime policy:
    /// drop instead of blocking the audio thread).
    pub dropped_chunks: Arc<AtomicU64>,
}

impl CaptureHandle {
    /// Non-blocking receive of one 8 kHz mono i16 chunk.
    pub fn try_recv(&self) -> Option<Vec<i16>> {
        self.rx.try_recv().ok()
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        info!("audio capture stopped");
    }
}

/// Start capturing the default microphone into 8 kHz mono i16 chunks.
pub fn start_capture() -> Result<CaptureHandle, String> {
    // Handshake channel: thread reports either a fatal error (String)
    // or the stop-channel sender once everything is wired up.
    let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<mpsc::Sender<()>, String>>(1);
    // Sample data channel: bounded; producer drops on overflow (realtime).
    let (data_tx, data_rx) = mpsc::sync_channel::<Vec<i16>>(256);

    let dropped_chunks = Arc::new(AtomicU64::new(0));
    let dropped_in_thread = Arc::clone(&dropped_chunks);

    let thread = std::thread::Builder::new()
        .name("liho-audio".into())
        .spawn(move || {
            if let Err(e) = run_capture(ready_tx, data_tx, dropped_in_thread) {
                error!("audio thread failed: {e}");
            }
        })
        .map_err(|e| format!("spawn thread: {e}"))?;

    let stop_tx = ready_rx
        .recv()
        .map_err(|_| "audio thread died before signalling".to_string())??;

    info!("microphone capture started ({TARGET_RATE} Hz mono i16)");
    Ok(CaptureHandle {
        stop_tx: Some(stop_tx),
        rx: data_rx,
        thread: Some(thread),
        dropped_chunks,
    })
}

fn run_capture(
    ready_tx: mpsc::SyncSender<Result<mpsc::Sender<()>, String>>,
    data_tx: mpsc::SyncSender<Vec<i16>>,
    dropped: Arc<AtomicU64>,
) -> Result<(), String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("no default input device (plug in a microphone)")?;
    let device_name = device.name().unwrap_or_else(|_| "<unnamed>".to_string());
    let ranges: Vec<_> = device
        .supported_input_configs()
        .map_err(|e| format!("enumerate input configs of {device_name}: {e}"))?
        .collect();
    let (cfg, sample_format) = pick_config(
        ranges,
        device
            .default_input_config()
            .ok()
            .map(|c| c.sample_rate().0),
    )
    .ok_or_else(|| format!("{device_name} supports neither i16 nor f32 input"))?;

    let channels = cfg.channels as usize;
    let src_rate = cfg.sample_rate.0;
    info!("capture device: {device_name} ({src_rate} Hz, {channels} ch, {sample_format:?})");

    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let mut converter = RateConverter::new();
    let tx = data_tx.clone();
    let dropped_cb = Arc::clone(&dropped);

    let stream = device
        .build_input_stream_raw(
            &cfg,
            sample_format,
            move |data: &cpal::Data, _info: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = match data.sample_format() {
                    cpal::SampleFormat::I16 => data
                        .as_slice::<i16>()
                        .map(|s| s.iter().map(|v| *v as f32 / 32_768.0).collect())
                        .unwrap_or_default(),
                    cpal::SampleFormat::F32 => data
                        .as_slice::<f32>()
                        .map(|s| s.to_vec())
                        .unwrap_or_default(),
                    other => {
                        debug!("unsupported capture format {other:?}, buffer skipped");
                        return;
                    }
                };
                if samples.is_empty() {
                    return;
                }
                let mono = downmix_to_mono(&samples, channels);
                let resampled = converter.convert(&mono, src_rate, TARGET_RATE);
                if resampled.is_empty() {
                    return;
                }
                // Realtime policy: never block the audio thread.
                if tx.try_send(to_i16(&resampled)).is_err() {
                    dropped_cb.fetch_add(1, Ordering::Relaxed);
                }
            },
            move |e| error!("audio stream error: {e}"),
            None,
        )
        .map_err(|e| format!("build input stream: {e}"))?;

    stream
        .play()
        .map_err(|e| format!("start input stream: {e}"))?;

    // Everything is up — hand the stop channel to the owner.
    ready_tx.send(Ok(stop_tx)).map_err(|_| "owner gone")?;

    // Park until the owner drops the handle; dropping the stream stops audio.
    let _ = stop_rx.recv();
    drop(stream);
    Ok(())
}

/// Pick the input configuration to record from.
///
/// Ranking (higher wins):
///
/// 1. **format** — i16 needs no conversion on the way in, f32 is the fallback;
/// 2. **channels** — a microphone is 1-2 channels; the ALSA `default` device
///    happily offers 32 of them, and every extra channel is work per callback;
/// 3. **rate** — closest to [`COMFORTABLE_RATE`]. T-one consumes 8 kHz, so a
///    16 kHz capture needs a cheap 2:1 step-down, while exotic rates such as
///    the 384 kHz some PipeWire sinks advertise would mean 48x the resampling
///    work for identical results.
pub(crate) fn pick_config(
    ranges: Vec<cpal::SupportedStreamConfigRange>,
    _device_preferred_rate: Option<u32>,
) -> Option<(cpal::StreamConfig, cpal::SampleFormat)> {
    type Rank = (u8, u32, u32);

    fn rank(r: &cpal::SupportedStreamConfigRange) -> Option<Rank> {
        let fmt = match r.sample_format() {
            cpal::SampleFormat::I16 => 2u8,
            cpal::SampleFormat::F32 => 1u8,
            _ => return None,
        };
        let channels = r.channels() as u32;
        let rate = r.max_sample_rate().0;
        Some((
            fmt,
            10_000u32.saturating_sub(channels.saturating_mul(100)),
            100_000u32.saturating_sub(rate.abs_diff(COMFORTABLE_RATE)),
        ))
    }

    let best = ranges.iter().filter_map(rank).max()?;
    let r = ranges.iter().find(|r| rank(r) == Some(best))?;
    Some((
        cpal::StreamConfig {
            channels: r.channels(),
            sample_rate: cpal::SampleRate(r.max_sample_rate().0),
            buffer_size: cpal::BufferSize::Default,
        },
        if best.0 == 2 {
            cpal::SampleFormat::I16
        } else {
            cpal::SampleFormat::F32
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo_averages_frames() {
        let stereo = [0.0_f32, 1.0, 0.5, -0.5, 1.0, 0.0];
        let mono = downmix_to_mono(&stereo, 2);
        assert_eq!(mono, vec![0.5, 0.0, 0.5]);
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
        assert_eq!(c.convert(&input, 8_000, 8_000), input);
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

    fn range(
        fmt: cpal::SampleFormat,
        rate: u32,
        channels: u16,
    ) -> cpal::SupportedStreamConfigRange {
        cpal::SupportedStreamConfigRange::new(
            channels,
            cpal::SampleRate(rate),
            cpal::SampleRate(rate),
            cpal::SupportedBufferSize::Range {
                min: 128,
                max: 4096,
            },
            fmt,
        )
    }

    #[test]
    fn pick_config_prefers_i16() {
        let ranges = vec![
            range(cpal::SampleFormat::F32, 48_000, 2),
            range(cpal::SampleFormat::I16, 48_000, 2),
        ];
        let (cfg, fmt) = pick_config(ranges, None).unwrap();
        assert_eq!(fmt, cpal::SampleFormat::I16);
        assert_eq!(cfg.sample_rate.0, 48_000);
    }

    #[test]
    fn pick_config_falls_back_to_f32() {
        let ranges = vec![
            range(cpal::SampleFormat::I8, 48_000, 2),
            range(cpal::SampleFormat::U8, 48_000, 1),
            range(cpal::SampleFormat::F32, 44_100, 1),
        ];
        let (cfg, fmt) = pick_config(ranges, None).unwrap();
        assert_eq!(fmt, cpal::SampleFormat::F32);
        assert_eq!(cfg.channels, 1);
    }

    #[test]
    fn pick_config_prefers_rate_near_16k() {
        let ranges = vec![
            range(cpal::SampleFormat::I16, 48_000, 1),
            range(cpal::SampleFormat::I16, 16_000, 1),
        ];
        let (cfg, _fmt) = pick_config(ranges, None).unwrap();
        assert_eq!(cfg.sample_rate.0, 16_000);
    }

    #[test]
    fn pick_config_avoids_exotic_rates() {
        // PipeWire happily advertises 384 kHz; resampling that to 8 kHz is
        // 48x the work for no benefit.
        let ranges = vec![
            range(cpal::SampleFormat::I16, 48_000, 1),
            range(cpal::SampleFormat::I16, 384_000, 1),
        ];
        let (cfg, _fmt) = pick_config(ranges, None).unwrap();
        assert_eq!(cfg.sample_rate.0, 48_000);
    }

    #[test]
    fn pick_config_prefers_near_8k_rate_when_offered() {
        // A native 8 kHz capture needs no resampling at all and T-one is a
        // telephony model, so it wins over 44.1 kHz.
        let ranges = vec![
            range(cpal::SampleFormat::I16, 8_000, 1),
            range(cpal::SampleFormat::I16, 44_100, 1),
        ];
        let (cfg, _fmt) = pick_config(ranges, None).unwrap();
        assert_eq!(cfg.sample_rate.0, 8_000);
    }

    #[test]
    fn pick_config_prefers_fewer_channels() {
        // The ALSA `default` device offers 1..32 channels at one rate; taking
        // 32 would multiply the callback and downmix work by 32.
        let mut ranges: Vec<_> = (1..=8)
            .map(|ch| range(cpal::SampleFormat::I16, 48_000, ch))
            .collect();
        ranges.push(range(cpal::SampleFormat::I16, 48_000, 32));
        let (cfg, _fmt) = pick_config(ranges, None).unwrap();
        assert_eq!(cfg.channels, 1);
    }

    #[test]
    fn pick_config_prefers_mono_over_surround_even_at_other_rate() {
        let ranges = vec![
            range(cpal::SampleFormat::I16, 16_000, 6),
            range(cpal::SampleFormat::I16, 48_000, 1),
        ];
        let (cfg, _fmt) = pick_config(ranges, None).unwrap();
        assert_eq!(cfg.channels, 1);
    }

    #[test]
    fn pick_config_falls_back_when_only_exotic_rate_exists() {
        let ranges = vec![range(cpal::SampleFormat::I16, 384_000, 1)];
        let (cfg, _fmt) = pick_config(ranges, None).unwrap();
        assert_eq!(cfg.sample_rate.0, 384_000);
    }

    #[test]
    fn pick_config_rejects_unsupported_formats() {
        let ranges = vec![range(cpal::SampleFormat::I8, 48_000, 2)];
        assert!(pick_config(ranges, None).is_none());
        assert!(pick_config(Vec::new(), None).is_none());
    }

    /// Live check against the real default input device.
    /// Run manually: cargo test live_capture -- --ignored --nocapture
    #[test]
    #[ignore = "live capture: needs a microphone"]
    fn live_capture_delivers_samples() {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(true)
            .try_init();
        let cap = start_capture().expect("start capture");
        let mut total = 0usize;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline && total <= TARGET_RATE as usize / 4 {
            if let Some(chunk) = cap.try_recv() {
                total += chunk.len();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        println!(
            "LIVE CAPTURE: {total} samples (~{} ms @ {TARGET_RATE}), dropped={}",
            total * 1000 / TARGET_RATE as usize,
            cap.dropped_chunks.load(Ordering::Relaxed)
        );
        assert!(total > 0, "no audio samples captured in 5 s");
    }
}
