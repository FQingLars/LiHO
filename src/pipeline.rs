//! Pipeline glue: audio → streaming STT → LLM → display.
//!
//! Two threads keep listening and answering independent:
//!
//! **Listener**: pulls 8 kHz chunks from every capture (microphone and system
//! output), feeds each stream into its *own* T-one instance — streaming state
//! is per audio source, two sources cannot share one — and appends every
//! finalized phrase to a tagged transcript:
//!
//! ```text
//! Интервьюер: какой у вас опыт с kafka?
//! Пользователь: я три года возился с ней в продакшене
//! Интервьюер: а с какими версиями?
//! ```
//!
//! Everything the interviewer says is forwarded to the LLM together with the
//! dialogue history — deciding *whether* it was a question is the model's job,
//! not a fragile text heuristic's (T-one emits no punctuation at all, so "does
//! it end with '?'" can never fire). What remains on this side of the wire is
//! pacing: a minimum amount of new interviewer text and a minimum gap between
//! requests, so one monologue is one stream of requests instead of one per
//! breath.
//!
//! **Answer worker**: consumes the queue strictly in order (one DeepSeek chat
//! session must be threaded sequentially), drives the LLM through a dedicated
//! current-thread runtime and updates the display.
//!
//! `stop` flips an atomic that ends both loops promptly.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};

use crate::audio::{CaptureHandle, Source};
use crate::config::Config;
use crate::display::OverlayCtl;
use crate::llm::DeepSeekWeb;
use crate::tone::{CHUNK_SAMPLES, Transcriber};

/// Upper bound for the interviewer's accumulated speech; older text is cut so
/// a long monologue cannot grow without bound.
const MAX_INTERVIEWER_CHARS: usize = 2_000;
/// Upper bound for the whole tagged transcript sent to the LLM.
const MAX_TRANSCRIPT_CHARS: usize = 2_000;
/// Minimum NEW interviewer characters before another request is sent.
/// ~3 s of Russian speech at ~10 characters/s.
const MIN_NEW_CHARS: usize = 30;
/// Minimum wall-clock gap between two requests. The worker answers strictly in
/// order and a round-trip takes ~1-3 s; without this a fast talker would build
/// a backlog of stale questions.
const MIN_REQUEST_GAP: Duration = Duration::from_secs(3);

/// Decides when the interviewer's speech is worth sending.
///
/// Pure pacing, no semantics: enough new text since the last request **and**
/// enough time passed. `now` is injected so the rule stays testable.
pub struct RequestGate {
    min_new_chars: usize,
    min_gap: Duration,
    sent_at: Option<Instant>,
}

impl RequestGate {
    pub fn new(min_new_chars: usize, min_gap: Duration) -> Self {
        Self {
            min_new_chars,
            min_gap,
            sent_at: None,
        }
    }

    /// Is `new_chars` (interviewer characters accumulated since the last
    /// request) enough to ask again at `now`?
    pub fn allows(&self, new_chars: usize, now: Instant) -> bool {
        if new_chars < self.min_new_chars {
            return false;
        }
        match self.sent_at {
            None => true,
            Some(at) => now.duration_since(at) >= self.min_gap,
        }
    }

    /// Remember that a request went out at `now`.
    pub fn record_sent(&mut self, now: Instant) {
        self.sent_at = Some(now);
    }
}

impl Default for RequestGate {
    fn default() -> Self {
        Self::new(MIN_NEW_CHARS, MIN_REQUEST_GAP)
    }
}

/// Latest-wins inbox between the listener and the answer worker.
///
/// While the worker is busy with one request, a newer transcript **replaces**
/// the queued one: the model always answers the freshest dialogue, and a burst
/// of interviewer speech collapses into a single request instead of a backlog.
pub struct Inbox {
    slot: Mutex<Option<String>>,
    filled: Condvar,
}

impl Inbox {
    pub fn new() -> Self {
        Self {
            slot: Mutex::new(None),
            filled: Condvar::new(),
        }
    }

    /// Queue a message, superseding any not-yet-taken one.
    pub fn put(&self, message: String) {
        let mut slot = self.slot.lock().expect("inbox poisoned");
        *slot = Some(message);
        self.filled.notify_all();
    }

    /// Take the queued message, waiting up to `timeout` for one to arrive.
    pub fn take_timeout(&self, timeout: Duration) -> Option<String> {
        let slot = self.slot.lock().expect("inbox poisoned");
        let (mut slot, _) = self
            .filled
            .wait_timeout_while(slot, timeout, |queued| queued.is_none())
            .expect("inbox poisoned");
        slot.take()
    }
}

impl Default for Inbox {
    fn default() -> Self {
        Self::new()
    }
}

/// The tagged transcript plus the pacing gate — everything the listener needs
/// to decide *what and when* to send.
pub struct ListenState {
    transcript: String,
    interviewer: String,
    system_prompt: String,
    system_sent: bool,
}

impl ListenState {
    pub fn new(system_prompt: &str) -> Self {
        Self {
            transcript: String::new(),
            interviewer: String::new(),
            system_prompt: system_prompt.to_string(),
            system_sent: false,
        }
    }

    /// Length of the interviewer's accumulated speech so far.
    pub fn interviewer_chars(&self) -> usize {
        self.interviewer.chars().count()
    }

    /// Feed one finalized phrase into the tagged transcript.
    ///
    /// Returns nothing: *whether* to send is decided by the caller through
    /// [`Self::message`] plus the [`RequestGate`].
    pub fn on_phrase(&mut self, source: Source, phrase: &str) {
        let phrase = phrase.trim();
        if phrase.is_empty() {
            return;
        }
        self.transcript.push_str(source.label());
        self.transcript.push_str(": ");
        self.transcript.push_str(phrase);
        self.transcript.push('\n');
        self.trim_transcript();

        if source == Source::Interviewer {
            self.interviewer.push_str(phrase);
            self.interviewer.push(' ');
            if self.interviewer.chars().count() > MAX_INTERVIEWER_CHARS {
                self.interviewer = self
                    .interviewer
                    .chars()
                    .skip(MAX_INTERVIEWER_CHARS / 2)
                    .collect();
            }
        }
    }

    /// The message to send: the whole tagged dialogue, with the role prompt
    /// attached to the first request of the session.
    pub fn message(&mut self) -> String {
        if self.system_sent {
            self.transcript.clone()
        } else {
            self.system_sent = true;
            format!(
                "{}\n\nДиалог (реплики помечены спикером, новые идут в конце):\n{}",
                self.system_prompt,
                self.transcript.trim_end()
            )
        }
    }

    /// Keep the transcript bounded, dropping whole lines only.
    fn trim_transcript(&mut self) {
        let total = self.transcript.chars().count();
        if total <= MAX_TRANSCRIPT_CHARS {
            return;
        }
        let cut = total - MAX_TRANSCRIPT_CHARS;
        let rest: String = self.transcript.chars().skip(cut).collect();
        self.transcript = match rest.find('\n') {
            Some(i) => rest[i + 1..].to_string(),
            None => rest,
        };
    }
}

/// One conversation side: its capture, its T-one state and its buffer.
struct Stream {
    source: Source,
    capture: CaptureHandle,
    transcriber: Transcriber,
    pending: Vec<i16>,
}

pub struct Pipeline;

impl Pipeline {
    /// Spawn listener + answer-worker threads.
    ///
    /// `streams` pairs each capture with its own T-one instance; the speaker
    /// label comes from the capture itself.
    /// Set `stop`, then join the returned handle(s) via `join_worker`.
    pub fn spawn(
        cfg: Config,
        llm: Arc<DeepSeekWeb>,
        session_id: String,
        streams: Vec<(CaptureHandle, Transcriber)>,
        overlay: OverlayCtl,
        stop: Arc<AtomicBool>,
    ) -> std::thread::JoinHandle<()> {
        let inbox = Arc::new(Inbox::new());

        let worker_stop = Arc::clone(&stop);
        let llm_worker = Arc::clone(&llm);
        let overlay_worker = overlay.clone();
        let session_for_worker = session_id.clone();
        let worker_inbox = Arc::clone(&inbox);
        let worker = std::thread::Builder::new()
            .name("liho-answerer".into())
            .spawn(move || {
                answer_worker(
                    worker_stop,
                    llm_worker,
                    session_for_worker,
                    overlay_worker,
                    worker_inbox,
                )
            })
            .expect("spawn answerer");

        let listener = std::thread::Builder::new()
            .name("liho-listener".into())
            .spawn(move || listen(cfg, streams, overlay, inbox, stop))
            .expect("spawn listener");

        LISTENER_WORKER.lock().unwrap().replace(worker);
        listener
    }

    /// Join the answer-worker started by `spawn` (the listener is joined by
    /// the handle returned there).
    pub fn join_worker() {
        if let Some(h) = LISTENER_WORKER.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

static LISTENER_WORKER: std::sync::Mutex<Option<std::thread::JoinHandle<()>>> =
    std::sync::Mutex::new(None);

fn listen(
    cfg: Config,
    streams: Vec<(CaptureHandle, Transcriber)>,
    overlay: OverlayCtl,
    inbox: Arc<Inbox>,
    stop: Arc<AtomicBool>,
) {
    let mut streams: Vec<Stream> = streams
        .into_iter()
        .map(|(capture, transcriber)| Stream {
            source: capture.source(),
            capture,
            transcriber,
            pending: Vec::with_capacity(CHUNK_SAMPLES * 4),
        })
        .collect();
    let labels: Vec<&str> = streams.iter().map(|s| s.source.label()).collect();
    let mut state = ListenState::new(&cfg.system_prompt);
    let mut gate = RequestGate::default();
    let mut sent_chars = 0usize;
    let mut chunks_seen = 0u64;
    let mut stats_at = Instant::now();

    info!(
        "listener started: {} stream(s) [{}], T-one {CHUNK_SAMPLES}-sample slices",
        streams.len(),
        labels.join(", ")
    );

    while !stop.load(Ordering::SeqCst) {
        let mut got_audio = false;
        for stream in &mut streams {
            while let Some(chunk) = stream.capture.try_recv() {
                got_audio = true;
                chunks_seen += 1;
                stream.pending.extend_from_slice(&chunk);
            }
            if stream.pending.len() < CHUNK_SAMPLES {
                continue;
            }

            // ---- streaming STT: consume every full 300 ms slice ----
            while stream.pending.len() >= CHUNK_SAMPLES {
                let slice: Vec<i16> = stream.pending.drain(..CHUNK_SAMPLES).collect();
                let phrases = match stream.transcriber.forward(&slice, false) {
                    Ok(p) => p,
                    Err(e) => {
                        error!("STT failed on {}: {e}", stream.source.label());
                        break;
                    }
                };
                for p in phrases {
                    if p.text.trim().is_empty() {
                        continue;
                    }
                    info!("STT {}: “{}”", stream.source.label(), p.text.trim());
                    state.on_phrase(stream.source, &p.text);
                }
            }
        }

        // ---- pacing: forward the interviewer's speech to the LLM ----
        let now = Instant::now();
        let heard = state.interviewer_chars();
        if gate.allows(heard.saturating_sub(sent_chars), now) {
            gate.record_sent(now);
            sent_chars = heard;
            overlay.set_text("Формирую ответ…");
            inbox.put(state.message());
        }

        if !got_audio {
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }
        if stats_at.elapsed() > Duration::from_secs(10) {
            stats_at = Instant::now();
            let dropped: Vec<String> = streams
                .iter()
                .map(|s| {
                    format!(
                        "{}={}",
                        s.source.label(),
                        s.capture.dropped_chunks.load(Ordering::Relaxed)
                    )
                })
                .collect();
            debug!(
                "stats: chunks={chunks_seen}, dropped [{}]",
                dropped.join(", ")
            );
        }
    }

    info!("listener stopped");
}

fn answer_worker(
    stop: Arc<AtomicBool>,
    llm: Arc<DeepSeekWeb>,
    session_id: String,
    overlay: OverlayCtl,
    inbox: Arc<Inbox>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            error!("answerer: runtime build failed: {e}");
            return;
        }
    };
    let mut parent_msg: Option<i64> = None;

    loop {
        if stop.load(Ordering::SeqCst) {
            // Drain nothing on shutdown; exit promptly.
            break;
        }
        // Latest-wins: while we were busy a fresher transcript may have
        // replaced this one in the inbox.
        let Some(message) = inbox.take_timeout(Duration::from_millis(150)) else {
            continue;
        };
        overlay.set_text("Думаю над ответом…");
        let started = Instant::now();
        if !llm.has_ready_pow() {
            warn!("PoW pool cold — this answer may wait for a solve");
        }
        let result = rt.block_on(llm.ask(&session_id, parent_msg, &message));
        // Re-warm for the next question. tokio::spawn needs a runtime
        // context, and we are on a bare OS thread — enter ours.
        rt.block_on(async { llm.schedule_refill() });
        match result {
            Ok((answer, mid)) => {
                parent_msg = mid;
                overlay.set_text(&answer);
                info!(
                    "LLM answered in {:.1} s ({} chars)",
                    started.elapsed().as_secs_f32(),
                    answer.chars().count()
                );
            }
            Err(e) => {
                error!("LLM failed: {e}");
                overlay.set_text(&format!("Ошибка LLM:\n{e}"));
            }
        }
    }
    info!("answerer stopped");
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROMPT: &str = "Ты — ассистент на интервью.";

    fn state() -> ListenState {
        ListenState::new(PROMPT)
    }

    // ---- transcript --------------------------------------------------

    #[test]
    fn every_phrase_is_tagged_with_its_speaker() {
        let mut st = state();
        st.on_phrase(Source::Interviewer, "какой у вас опыт");
        st.on_phrase(Source::User, "я три года работал с kafka");
        assert_eq!(
            st.transcript,
            "Интервьюер: какой у вас опыт\nПользователь: я три года работал с kafka\n"
        );
    }

    #[test]
    fn empty_and_blank_phrases_are_ignored() {
        let mut st = state();
        st.on_phrase(Source::Interviewer, "");
        st.on_phrase(Source::Interviewer, "   ");
        st.on_phrase(Source::User, "\n");
        assert!(st.transcript.is_empty());
    }

    #[test]
    fn candidate_speech_is_never_counted_as_new_interviewer_text() {
        let mut st = state();
        st.on_phrase(Source::User, "я три года работал с kafka в продакшене");
        assert_eq!(st.interviewer_chars(), 0);
    }

    #[test]
    fn message_carries_the_whole_dialogue_with_the_prompt_once() {
        let mut st = state();
        st.on_phrase(Source::Interviewer, "какой у вас опыт с kafka");
        st.on_phrase(Source::User, "я три года работал с ним");
        let first = st.message();
        assert!(first.starts_with(PROMPT), "{first}");
        assert!(first.contains("Диалог"), "{first}");
        assert!(
            first.contains("Интервьюер: какой у вас опыт с kafka"),
            "{first}"
        );
        assert!(
            first.contains("Пользователь: я три года работал с ним"),
            "{first}"
        );

        st.on_phrase(Source::Interviewer, "а с какими версиями");
        let second = st.message();
        assert!(!second.contains(PROMPT), "prompt sent twice: {second}");
        assert!(second.contains("а с какими версиями"), "{second}");
    }

    #[test]
    fn transcript_is_bounded_and_line_aligned() {
        let mut st = state();
        for _ in 0..200 {
            st.on_phrase(
                Source::Interviewer,
                "разговор про распределённые системы и очереди сообщений",
            );
        }
        assert!(
            st.transcript.chars().count() <= MAX_TRANSCRIPT_CHARS,
            "transcript grew to {}",
            st.transcript.chars().count()
        );
        assert!(
            st.transcript.starts_with("Интервьюер:"),
            "trimmed mid-line: {:?}",
            &st.transcript[..40.min(st.transcript.len())]
        );
    }

    #[test]
    fn interviewer_buffer_is_bounded() {
        let mut st = state();
        for _ in 0..200 {
            st.on_phrase(Source::Interviewer, "очень длинная реплика про архитектуру");
        }
        assert!(st.interviewer_chars() <= MAX_INTERVIEWER_CHARS);
    }

    // ---- request pacing ----------------------------------------------

    #[test]
    fn first_request_goes_out_immediately() {
        let gate = RequestGate::default();
        assert!(gate.allows(MIN_NEW_CHARS, Instant::now()));
    }

    #[test]
    fn too_little_new_text_does_not_send() {
        let gate = RequestGate::default();
        assert!(!gate.allows(MIN_NEW_CHARS - 1, Instant::now()));
        assert!(!gate.allows(0, Instant::now()));
    }

    #[test]
    fn gap_between_requests_is_respected() {
        let t0 = Instant::now();
        let mut gate = RequestGate::default();
        gate.record_sent(t0);
        let too_soon = t0 + MIN_REQUEST_GAP - Duration::from_millis(50);
        assert!(!gate.allows(999, too_soon), "sent too soon");
        let late_enough = t0 + MIN_REQUEST_GAP + Duration::from_millis(10);
        assert!(gate.allows(999, late_enough), "gap elapsed");
    }

    #[test]
    fn both_conditions_must_hold() {
        let t0 = Instant::now();
        let mut gate = RequestGate::default();
        gate.record_sent(t0);
        // Enough time, not enough text.
        assert!(!gate.allows(3, t0 + MIN_REQUEST_GAP * 2));
        // Enough text, not enough time.
        assert!(!gate.allows(MIN_NEW_CHARS * 10, t0));
    }

    // ---- inbox -------------------------------------------------------

    const NOW: Duration = Duration::from_millis(0);

    #[test]
    fn inbox_hands_the_message_over() {
        let inbox = Inbox::new();
        assert_eq!(inbox.take_timeout(NOW), None, "empty inbox");
        inbox.put("первое".into());
        assert_eq!(inbox.take_timeout(NOW).as_deref(), Some("первое"));
        assert_eq!(inbox.take_timeout(NOW), None, "taken exactly once");
    }

    #[test]
    fn inbox_keeps_only_the_newest_message() {
        let inbox = Inbox::new();
        inbox.put("устаревшее".into());
        inbox.put("свежее".into());
        assert_eq!(inbox.take_timeout(NOW).as_deref(), Some("свежее"));
    }

    #[test]
    fn inbox_waits_for_a_message_and_wakes_on_put() {
        use std::sync::Arc;
        let inbox = Arc::new(Inbox::new());
        let waiter = Arc::clone(&inbox);
        let handle = std::thread::spawn(move || waiter.take_timeout(Duration::from_secs(5)));
        std::thread::sleep(Duration::from_millis(50));
        inbox.put("будильник".into());
        assert_eq!(handle.join().unwrap().as_deref(), Some("будильник"));
    }

    #[test]
    fn inbox_timeout_returns_none_when_idle() {
        let inbox = Inbox::new();
        assert_eq!(
            inbox.take_timeout(Duration::from_millis(20)),
            None,
            "must not block forever"
        );
    }
}
