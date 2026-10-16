use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use env_logger::Env;
use log::{error, info, warn};

mod audio;
mod config;
mod display;
mod llm;
mod pipeline;
mod pow;
mod tone;

#[tokio::main]
async fn main() {
    env_logger::init_from_env(Env::default().default_filter_or("info"));

    info!("LIhO starting — streaming ASR + LLM assistant");

    if let Err(e) = run().await {
        error!("Fatal error: {e}");
        std::process::exit(1);
    }
    info!("LIhO exited cleanly");
}

async fn run() -> Result<(), String> {
    // 1. Config
    let cfg = config::Config::load()?;
    info!(
        "Config loaded: token length {}, stt model '{}'",
        cfg.deepseek_token.chars().count(),
        cfg.stt_model.display()
    );

    // 2. Display first — it shows progress while the rest warms up.
    let geom = display::OverlayGeometry::from_args(&std::env::args().collect::<Vec<_>>())?;
    info!(
        "display config: {}x{} at right={}, top={}, font={} px, autohide={:?} ms",
        geom.width,
        geom.height,
        geom.margin_right,
        geom.margin_top,
        geom.font_size,
        geom.autohide_ms
    );
    let overlay = display::OverlayHandle::launch(geom)?;
    let overlay_ctl = overlay.ctl().clone();
    overlay_ctl.set_text("LIhO запускается…");
    for _ in 0..40 {
        if overlay.configured.load(Ordering::SeqCst) {
            info!("display ready");
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // 3. Audio + STT: one capture (and one streaming model) per speaker.
    let mut streams = Vec::new();
    for source in [audio::Source::User, audio::Source::Interviewer] {
        match audio::start_capture(source) {
            Ok(capture) => {
                // Streaming state is per audio source — a second T-one
                // instance costs memory, not accuracy.
                let transcriber = tone::Transcriber::load(&cfg.stt_model)?;
                info!("{} stream ready", source.label());
                streams.push((capture, transcriber));
            }
            Err(e) if source == audio::Source::User => {
                return Err(format!("микрофон недоступен: {e}"));
            }
            Err(e) => {
                // Loopback is a bonus: without it the app still works, it
                // just cannot separate the interviewer from the candidate.
                warn!("{source:?} stream unavailable ({e}) — continuing without it");
            }
        }
    }
    overlay_ctl.set_text("Модели загружены");

    // 4. LLM: persistent single-chat session
    let llm = Arc::new(llm::DeepSeekWeb::new(cfg.deepseek_token.clone())?);
    let session_id = llm.create_session().await?;
    info!("Chat session ready: {session_id} (fast mode: thinking disabled)");
    // Warm the PoW pool now so the first question gets an instant header.
    llm.schedule_refill();
    overlay_ctl.set_text("LIhO готов — слушаю вопросы");

    // 5. Ctrl-C (Windows/Linux) and SIGTERM stop the app.
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        if let Err(e) = ctrlc::set_handler(move || {
            info!("stop requested");
            stop.store(true, Ordering::SeqCst);
        }) {
            error!("cannot install Ctrl-C handler: {e}");
        }
    }

    // 6. The brain.
    let brain = pipeline::Pipeline::spawn(
        cfg,
        Arc::clone(&llm),
        session_id,
        streams,
        overlay_ctl,
        Arc::clone(&stop),
    );

    // Wait for a stop signal (Ctrl-C / SIGTERM).
    while !stop.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    let _ = brain.join();
    pipeline::Pipeline::join_worker();
    drop(llm);
    drop(overlay);
    Ok(())
}
