//! Microphone backend: the default input device through **cpal**.
//!
//! One code path for every desktop OS — WASAPI (Windows), PulseAudio/ALSA
//! (Linux), CoreAudio (macOS). The stream runs on its own thread; the owner
//! pulls 8 kHz mono i16 chunks from a bounded channel.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{debug, error, info};

use super::{
    COMFORTABLE_RATE, CaptureHandle, RateConverter, Source, TARGET_RATE, downmix_to_mono, to_i16,
};

/// Start capturing the default microphone.
pub(crate) fn start() -> Result<CaptureHandle, String> {
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
    info!("microphone: {device_name} ({src_rate} Hz, {channels} ch, {sample_format:?})");

    // Handshake channel: the thread reports either a fatal error (String) or
    // the stop-channel sender once everything is wired up.
    let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<Box<dyn FnOnce() + Send>, String>>(1);
    // Sample data channel: bounded; producer drops on overflow (realtime).
    let (data_tx, data_rx) = mpsc::sync_channel::<Vec<i16>>(256);
    let dropped = Arc::new(AtomicU64::new(0));
    let dropped_cb = Arc::clone(&dropped);

    let thread = std::thread::Builder::new()
        .name("liho-mic".into())
        .spawn(move || {
            let mut converter = RateConverter::new();
            let tx = data_tx.clone();
            let stream = match device.build_input_stream_raw(
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
                move |e| error!("microphone stream error: {e}"),
                None,
            ) {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("build microphone stream: {e}")));
                    return;
                }
            };

            if let Err(e) = stream.play() {
                let _ = ready_tx.send(Err(format!("start microphone stream: {e}")));
                return;
            }

            let (stop_tx, stop_rx) = mpsc::channel::<()>();
            let stop = Box::new(move || {
                let _ = stop_tx.send(());
            }) as Box<dyn FnOnce() + Send>;
            if ready_tx.send(Ok(stop)).is_err() {
                return; // owner gone
            }

            let _ = stop_rx.recv();
            drop(stream);
            info!("microphone stream stopped");
        })
        .map_err(|e| format!("spawn mic thread: {e}"))?;

    let stop = ready_rx
        .recv()
        .map_err(|_| "microphone thread died before signalling".to_string())??;

    Ok(CaptureHandle::new(
        Source::User,
        data_rx,
        stop,
        thread,
        dropped,
    ))
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
    fn pick_config_rejects_unsupported_formats() {
        let ranges = vec![range(cpal::SampleFormat::I8, 48_000, 2)];
        assert!(pick_config(ranges, None).is_none());
        assert!(pick_config(Vec::new(), None).is_none());
    }

    /// Live check against the real default input device.
    /// Run manually: cargo test live_mic -- --ignored --nocapture
    #[test]
    #[ignore = "live capture: needs a microphone"]
    fn live_microphone_capture_delivers_samples() {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(true)
            .try_init();
        let cap = start().expect("start capture");
        assert_eq!(cap.source(), Source::User);
        let mut total = 0usize;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline && total <= TARGET_RATE as usize / 4 {
            if let Some(chunk) = cap.try_recv() {
                total += chunk.len();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        println!(
            "LIVE MIC: {total} samples (~{} ms @ {TARGET_RATE}), dropped={}",
            total * 1000 / TARGET_RATE as usize,
            cap.dropped_chunks.load(Ordering::Relaxed)
        );
        assert!(total > 0, "no audio samples captured in 5 s");
    }
}
