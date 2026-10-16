//! Linux loopback backend: the PipeWire **sink monitor**.
//!
//! `STREAM_CAPTURE_SINK=true` on a capture stream makes PipeWire mix every
//! stream playing on the default sink — the browser call, the music player,
//! everything audible — which is exactly the interviewer's side of the
//! conversation.
//!
//! Caveat: with speakers in the room the monitor also hears the microphone.
//! Headphones avoid the duplication; there is no echo cancellation here.

use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

use log::{debug, error, info};
use pipewire as pw;
use pw::spa::param::audio::{AudioFormat, AudioInfoRaw};
use pw::spa::param::format::{MediaSubtype, MediaType};
use pw::spa::param::format_utils;
use pw::spa::pod::Pod;

use super::{CaptureHandle, RateConverter, Source, TARGET_RATE, downmix_to_mono, to_i16};

/// Start capturing the system output (sink monitor).
pub(crate) fn start() -> Result<CaptureHandle, String> {
    pw::init();

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
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(|e| format!("mainloop: {e}"))?;
    let context =
        pw::context::ContextRc::new(&mainloop, None).map_err(|e| format!("context: {e}"))?;
    let core = context
        .connect_rc(None)
        .map_err(|e| format!("connect to PipeWire: {e}"))?;

    let (stop_tx, stop_rx) = pw::channel::channel::<()>();
    let quit_loop = mainloop.clone();
    let _attached = stop_rx.attach(mainloop.loop_(), move |_msg| {
        quit_loop.quit();
    });

    // Everything is up — hand the PipeWire stop channel to the owner.
    let stop = Box::new(move || {
        let _ = stop_tx.send(());
    }) as Box<dyn FnOnce() + Send>;
    ready_tx.send(Ok(stop)).map_err(|_| "owner gone")?;

    let props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::STREAM_CAPTURE_SINK => "true",
    };

    let stream = pw::stream::StreamBox::new(&core, "LIhO-interviewer", props)
        .map_err(|e| format!("monitor stream: {e}"))?;

    struct State {
        format: AudioInfoRaw,
        converter: RateConverter,
    }

    let state = std::cell::RefCell::new(State {
        format: AudioInfoRaw::new(),
        converter: RateConverter::new(),
    });

    let data_tx_proc = data_tx.clone();
    let dropped_proc = Arc::clone(&dropped);

    let _listener = stream
        .add_local_listener_with_user_data(state)
        .param_changed(|_, state, id, param| {
            let Some(param) = param else { return };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }
            let (media_type, media_subtype) = match format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }
            state
                .borrow_mut()
                .format
                .parse(param)
                .expect("parse audio format");
            let st = state.borrow();
            info!(
                "loopback format: {} Hz, {} ch",
                st.format.rate(),
                st.format.channels()
            );
        })
        .process(move |stream, state| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                debug!("out of buffers");
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            let (offset, size) = {
                let chunk = data.chunk();
                (chunk.offset() as usize, chunk.size() as usize)
            };
            let Some(all) = data.data() else { return };
            let Some(bytes) = all.get(offset..offset + size) else {
                return;
            };
            if bytes.is_empty() {
                return;
            }

            let mut st = state.borrow_mut();
            let channels = st.format.channels().max(1) as usize;
            let src_rate = st.format.rate().max(1);

            let (chunks, _) = bytes.as_chunks::<{ mem::size_of::<f32>() }>();
            let floats: Vec<f32> = chunks.iter().map(|b| f32::from_le_bytes(*b)).collect();
            let mono = downmix_to_mono(&floats, channels);
            let out = st.converter.convert(&mono, src_rate, TARGET_RATE);
            drop(st);

            if out.is_empty() {
                return;
            }
            // Realtime policy: never block the PipeWire process callback.
            if data_tx_proc.try_send(to_i16(&out)).is_err() {
                dropped_proc.fetch_add(1, Ordering::Relaxed);
            }
        })
        .register()
        .map_err(|e| format!("listener register: {e}"))?;

    let mut audio_info = AudioInfoRaw::new();
    audio_info.set_format(AudioFormat::F32LE);
    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| format!("pod serialize: {e}"))?
    .0
    .into_inner();
    let mut params = [Pod::from_bytes(&values).ok_or("bad pod")?];

    stream
        .connect(
            pw::spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|e| format!("monitor stream connect: {e}"))?;

    info!("PipeWire sink monitor started ({TARGET_RATE} Hz mono i16)");
    mainloop.run();
    Ok(())
}

/// Live check against the running PipeWire daemon.
/// Run manually: cargo test live_loopback -- --ignored --nocapture
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "live loopback: needs a running PipeWire session"]
    fn live_loopback_captures_system_output() {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(true)
            .try_init();
        let cap = start().expect("start loopback");
        assert_eq!(cap.source(), Source::Interviewer);
        let mut total = 0usize;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline && total <= TARGET_RATE as usize / 4 {
            if let Some(chunk) = cap.try_recv() {
                total += chunk.len();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        println!(
            "LIVE LOOPBACK: {total} samples (~{} ms @ {TARGET_RATE}), dropped={}",
            total * 1000 / TARGET_RATE as usize,
            cap.dropped_chunks.load(Ordering::Relaxed)
        );
        assert!(total > 0, "no system audio captured in 5 s");
    }
}
