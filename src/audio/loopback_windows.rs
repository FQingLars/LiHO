//! Windows loopback backend: the **WASAPI render device in loopback mode**.
//!
//! Initializing an `IAudioClient` on the default *render* endpoint with
//! `Direction::Render` makes WASAPI feed the mix of everything played on the
//! default output back into the client — the call in the browser, media
//! players, system sounds. That is the interviewer's side of the
//! conversation.
//!
//! Bindings: the `wasapi` crate (MIT). The capture thread runs its own
//! event-driven loop; the owner pulls 8 kHz mono i16 chunks from a bounded
//! channel and, as everywhere else, chunks are dropped rather than blocking
//! the audio engine.

use std::collections::VecDeque;
use std::convert::TryInto;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

use log::{debug, error, info};
use wasapi::{AudioClient, DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat};

use super::{CaptureHandle, RateConverter, Source, TARGET_RATE, downmix_to_mono, to_i16};

/// Requested mix format for the loopback client. `autoconvert` lets the audio
/// engine resample the device mix to it, so the callback always sees f32.
const LOOPBACK_RATE: u32 = 48_000;
const LOOPBACK_CHANNELS: u16 = 2;

/// Start capturing the system output (WASAPI loopback).
pub(crate) fn start() -> Result<CaptureHandle, String> {
    // COM apartments are per-thread; the capture thread initializes its own.
    let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<Box<dyn FnOnce() + Send>, String>>(1);
    let (data_tx, data_rx) = mpsc::sync_channel::<Vec<i16>>(256);
    let dropped = Arc::new(AtomicU64::new(0));
    let dropped_thread = Arc::clone(&dropped);

    let thread = std::thread::Builder::new()
        .name("liho-loopback".into())
        .spawn(move || {
            if let Err(e) = run_loopback(ready_tx, data_tx, dropped_thread) {
                error!("loopback thread failed: {e}");
            }
        })
        .map_err(|e| format!("spawn loopback thread: {e}"))?;

    let stop = ready_rx
        .recv()
        .map_err(|_| "loopback thread died before signalling".to_string())??;

    Ok(CaptureHandle::new(
        Source::Interviewer,
        data_rx,
        stop,
        thread,
        dropped,
    ))
}

fn run_loopback(
    ready_tx: mpsc::SyncSender<Result<Box<dyn FnOnce() + Send>, String>>,
    data_tx: mpsc::SyncSender<Vec<i16>>,
    dropped: Arc<AtomicU64>,
) -> Result<(), String> {
    wasapi::initialize_mta()
        .ok()
        .map_err(|e| format!("initialize COM (MTA): {e:?}"))?;

    let enumerator = DeviceEnumerator::new().map_err(|e| format!("WASAPI enumerator: {e}"))?;
    let device = enumerator
        // The client's direction is Render; asking for a Capture stream on it
        // is what makes WASAPI hand back the mix (see AUDCLNT_STREAMFLAGS_LOOPBACK).
        .get_default_device(&Direction::Render)
        .map_err(|e| format!("no default output device for loopback: {e}"))?;
    let device_name = device
        .get_friendlyname()
        .map_err(|e| format!("output device name: {e}"))?;
    let mut client: AudioClient = device
        .get_iaudioclient()
        .map_err(|e| format!("IAudioClient of {device_name}: {e}"))?;

    let format = WaveFormat::new(
        32,
        32,
        &SampleType::Float,
        LOOPBACK_RATE as usize,
        LOOPBACK_CHANNELS as usize,
        None,
    );
    let blockalign = format.get_blockalign() as usize;
    let (_def_time, min_time) = client
        .get_device_period()
        .map_err(|e| format!("device period: {e}"))?;
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: min_time,
    };
    client
        .initialize_client(&format, &Direction::Capture, &mode)
        .map_err(|e| format!("initialize loopback client: {e}"))?;
    let event = client
        .set_get_eventhandle()
        .map_err(|e| format!("loopback event handle: {e}"))?;
    let capture = client
        .get_audiocaptureclient()
        .map_err(|e| format!("IAudioCaptureClient for loopback: {e}"))?;

    client
        .start_stream()
        .map_err(|e| format!("start loopback stream: {e}"))?;
    info!(
        "WASAPI loopback on {device_name} ({LOOPBACK_RATE} Hz, {LOOPBACK_CHANNELS} ch) \
         -> {TARGET_RATE} Hz mono i16"
    );

    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let stop = Box::new(move || {
        let _ = stop_tx.send(());
    }) as Box<dyn FnOnce() + Send>;
    ready_tx.send(Ok(stop)).map_err(|_| "owner gone")?;

    let mut converter = RateConverter::new();
    let mut queue: VecDeque<u8> = VecDeque::with_capacity(16 * blockalign * 1024);

    loop {
        if stop_rx.try_recv().is_ok() {
            break;
        }
        capture
            .read_from_device_to_deque(&mut queue)
            .map_err(|e| format!("read loopback: {e}"))?;

        if queue.len() >= blockalign {
            let bytes: Vec<u8> = queue.drain(..).collect();
            let samples: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("4-byte f32 chunk")))
                .collect();
            let usable = samples.len() - samples.len() % LOOPBACK_CHANNELS as usize;
            let mono = downmix_to_mono(&samples[..usable], LOOPBACK_CHANNELS as usize);
            let out = converter.convert(&mono, LOOPBACK_RATE, TARGET_RATE);
            if !out.is_empty() && data_tx.try_send(to_i16(&out)).is_err() {
                dropped.fetch_add(1, Ordering::Relaxed);
            }
        }

        // 1000 (100 ns units) = 100 ms; short enough to notice a stop request.
        if event.wait_for_event(1_000).is_err() {
            debug!("loopback event timeout");
        }
    }

    client.stop_stream().ok();
    info!("WASAPI loopback stopped");
    Ok(())
}
