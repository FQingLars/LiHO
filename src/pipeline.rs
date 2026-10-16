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
//! Only the interviewer stream can trigger an answer
//! ([crate::question::is_question]); the candidate's own words are context for
//! it. Answering starts while the interviewer is still talking.
//!
//! **Answer worker**: consumes the queue strictly in order (one DeepSeek chat
//! session must be threaded sequentially), drives the LLM through a dedicated
//! current-thread runtime and updates the display.
//!
//! `stop` flips an atomic that ends both loops promptly.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};

use crate::audio::{CaptureHandle, Source};
use crate::config::Config;
use crate::display::OverlayCtl;
use crate::llm::DeepSeekWeb;
use crate::question::is_question;
use crate::tone::{CHUNK_SAMPLES, Transcriber};

/// Minimum NEW characters before accumulated text may (re-)fire the LLM.
const MIN_NEW_CHARS: usize = 12;
/// Upper bound for the interviewer's accumulated text; older speech is cut.
const MAX_BUFFER_CHARS: usize = 2_000;
/// Upper bound for the whole tagged transcript sent to the LLM.
const MAX_TRANSCRIPT_CHARS: usize = 2_000;

/// Decides when the running transcript should trigger an answer.
///
/// T-one only emits a phrase after ~600 ms of silence, so every phrase is a
/// natural pause: that is the "the speaker stopped here" signal. Firing rules:
///
/// * the accumulated text (or the fresh phrase alone) looks like a question —
///   answer it, even mid-sentence, and mark the question as open;
/// * the interviewer keeps talking about an open question — fire once more so
///   the answer sees the rest of what was said;
/// * anything else — stay quiet.
///
/// Returns the transcript offset to mark as answered *after* the message built
/// from it is really queued (a failed send must not consume the text).
#[derive(Debug, Default)]
pub struct FirePolicy {
    answered_chars: usize,
    question_open: bool,
    followup_fired: bool,
}

impl FirePolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Text length already sent to the LLM.
    pub fn answered_chars(&self) -> usize {
        self.answered_chars
    }

    /// Re-base the offset after the text buffer was trimmed.
    pub fn set_answered_chars(&mut self, answered_chars: usize) {
        self.answered_chars = answered_chars;
    }

    /// Decide whether the freshly finalized `phrase` triggers an answer.
    ///
    /// `buffer` is the interviewer's accumulated text, `phrase` the new text.
    pub fn on_phrase(&mut self, buffer: &str, phrase: &str) -> Option<usize> {
        let n = buffer.chars().count();
        if n < self.answered_chars + MIN_NEW_CHARS {
            return None;
        }
        if is_question(buffer) || is_question(phrase) {
            self.question_open = true;
            self.followup_fired = false;
            return Some(n);
        }
        if self.question_open && !self.followup_fired {
            self.followup_fired = true;
            return Some(n);
        }
        None
    }

    /// Call once the message built from that offset reached the queue.
    pub fn mark_sent(&mut self, answered_chars: usize) {
        self.answered_chars = answered_chars;
    }
}

/// The tagged transcript plus the fire policy — everything the listener needs
/// to decide *what* to send.
pub struct ListenState {
    transcript: String,
    interviewer: String,
    policy: FirePolicy,
    system_prompt: String,
    system_sent: bool,
    /// Offset decided by the last `on_phrase`, confirmed by `mark_sent`
    /// once the message reached the queue.
    pending_answered: usize,
}

impl ListenState {
    pub fn new(system_prompt: &str) -> Self {
        Self {
            transcript: String::new(),
            interviewer: String::new(),
            policy: FirePolicy::new(),
            system_prompt: system_prompt.to_string(),
            system_sent: false,
            pending_answered: 0,
        }
    }

    /// Feed one finalized phrase. Returns the message to queue, if any.
    ///
    /// Every phrase is appended to the transcript; only the interviewer's
    /// phrases can trigger an answer.
    pub fn on_phrase(&mut self, source: Source, phrase: &str) -> Option<String> {
        let phrase = phrase.trim();
        if phrase.is_empty() {
            return None;
        }
        self.transcript.push_str(source.label());
        self.transcript.push_str(": ");
        self.transcript.push_str(phrase);
        self.transcript.push('\n');
        self.trim_transcript();

        if source != Source::Interviewer {
            return None;
        }

        self.interviewer.push_str(phrase);
        self.interviewer.push(' ');
        let (trimmed, survived) = trim_buffer(&self.interviewer, self.policy.answered_chars());
        self.interviewer = trimmed;
        self.policy.set_answered_chars(survived);

        let answered = self.policy.on_phrase(&self.interviewer, phrase)?;
        self.pending_answered = answered;
        if self.system_sent {
            Some(self.transcript.clone())
        } else {
            self.system_sent = true;
            Some(format!(
                "{}\n\nДиалог (реплики помечены спикером):\n{}",
                self.system_prompt,
                self.transcript.trim_end()
            ))
        }
    }

    /// Confirm that the message from the last `on_phrase` was queued.
    pub fn mark_sent(&mut self) {
        let answered = self.pending_answered;
        self.policy.mark_sent(answered);
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

/// Keep the interviewer's text bounded: cut off the oldest text, but never
/// before what the LLM has already been given (that would re-answer old
/// speech). Returns the new text plus how many chars of the answered prefix
/// survived.
pub fn trim_buffer(text: &str, answered_chars: usize) -> (String, usize) {
    let total = text.chars().count();
    if total <= MAX_BUFFER_CHARS {
        return (text.to_string(), answered_chars);
    }
    let cut = total - MAX_BUFFER_CHARS;
    let at = cut.max(answered_chars.min(total));
    let kept: String = text.chars().skip(at).collect();
    (kept, answered_chars.saturating_sub(at))
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
        let (q_tx, q_rx) = mpsc::channel::<String>();

        let worker_stop = Arc::clone(&stop);
        let llm_worker = Arc::clone(&llm);
        let overlay_worker = overlay.clone();
        let session_for_worker = session_id.clone();
        let worker = std::thread::Builder::new()
            .name("liho-answerer".into())
            .spawn(move || {
                answer_worker(
                    worker_stop,
                    llm_worker,
                    session_for_worker,
                    overlay_worker,
                    q_rx,
                )
            })
            .expect("spawn answerer");

        let listener = std::thread::Builder::new()
            .name("liho-listener".into())
            .spawn(move || listen(cfg, streams, overlay, q_tx, stop))
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
    q_tx: mpsc::Sender<String>,
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
                    if let Some(message) = state.on_phrase(stream.source, &p.text) {
                        info!("fire: “{}”", message.lines().last().unwrap_or("").trim());
                        overlay.set_text("Понял вопрос, отвечаю…");
                        if send_raw(&q_tx, message) {
                            state.mark_sent();
                        }
                    }
                }
            }
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

fn send_raw(q_tx: &mpsc::Sender<String>, msg: String) -> bool {
    match q_tx.send(msg) {
        Ok(()) => true,
        Err(e) => {
            error!("answer queue closed: {e}");
            false
        }
    }
}

fn answer_worker(
    stop: Arc<AtomicBool>,
    llm: Arc<DeepSeekWeb>,
    session_id: String,
    overlay: OverlayCtl,
    q_rx: mpsc::Receiver<String>,
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
        let Ok(message) = q_rx.recv_timeout(Duration::from_millis(150)) else {
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

    /// Feed phrases from one source and collect the messages that fired.
    fn run(sources: &[(Source, &str)]) -> Vec<String> {
        let mut st = ListenState::new(PROMPT);
        let mut out = Vec::new();
        for (source, phrase) in sources {
            if let Some(msg) = st.on_phrase(*source, phrase) {
                st.mark_sent();
                out.push(msg);
            }
        }
        out
    }

    fn transcript_of(sources: &[(Source, &str)]) -> String {
        let mut st = ListenState::new(PROMPT);
        for (source, phrase) in sources {
            st.on_phrase(*source, phrase);
        }
        st.transcript
    }

    // ---- fire policy -------------------------------------------------

    fn policy_after(chunks: &[&str]) -> (FirePolicy, Vec<usize>) {
        let mut p = FirePolicy::new();
        let mut buffer = String::new();
        let mut fires = Vec::new();
        for phrase in chunks {
            buffer.push_str(phrase);
            if let Some(answered) = p.on_phrase(&buffer, phrase) {
                p.mark_sent(answered);
                fires.push(answered);
            }
        }
        (p, fires)
    }

    #[test]
    fn plain_statement_does_not_fire() {
        let (p, f) = policy_after(&["я расскажу про свой опыт работы "]);
        assert!(f.is_empty());
        assert_eq!(p.answered_chars(), 0);
    }

    #[test]
    fn question_with_enough_new_chars_fires() {
        let (_, f) = policy_after(&["вы работали с kafka в продакшене? "]);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0], "вы работали с kafka в продакшене? ".chars().count());
    }

    #[test]
    fn too_few_new_chars_does_not_refire() {
        let mut p = FirePolicy::new();
        p.set_answered_chars("вы работали с kafka?".chars().count());
        assert!(p.on_phrase("вы работали с kafka? да", " да").is_none());
    }

    #[test]
    fn short_text_never_fires() {
        let mut p = FirePolicy::new();
        assert!(p.on_phrase("да?", "да?").is_none());
    }

    #[test]
    fn continuation_of_open_question_fires_once() {
        let (_, f) = policy_after(&[
            "какие у вас были обязанности? ",
            "я руководил командой из пяти человек и отвечал за найм ",
            "а ещё мы запускали еженедельные релизы ",
        ]);
        assert_eq!(f.len(), 2, "question + exactly one followup: {f:?}");
    }

    #[test]
    fn followup_after_new_question_fires_again() {
        let (_, f) = policy_after(&[
            "вы работали с kafka в продакшене? ",
            "а с какими версиями брокеров? ",
        ]);
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn monologue_after_question_stays_quiet() {
        let (_, f) = policy_after(&[
            "какие у вас были обязанности? ",
            "спасибо, очень подробно ",
            "кстати, в следующем раунде спросят про деплой ",
        ]);
        assert_eq!(f.len(), 2, "only the followup: {f:?}");
    }

    // ---- speaker tagging ---------------------------------------------

    #[test]
    fn every_phrase_is_tagged_with_its_speaker() {
        let t = transcript_of(&[
            (Source::Interviewer, "какой у вас опыт"),
            (Source::User, "я три года работал с kafka"),
        ]);
        assert!(t.contains("Интервьюер: какой у вас опыт\n"), "{t}");
        assert!(
            t.contains("Пользователь: я три года работал с kafka\n"),
            "{t}"
        );
    }

    #[test]
    fn candidate_speech_alone_never_fires() {
        let out = run(&[
            (Source::User, "вы работали с kafka в продакшене давно"),
            (Source::User, "а с какими версиями брокеров?"),
        ]);
        assert!(
            out.is_empty(),
            "candidate must not trigger answers: {out:?}"
        );
    }

    #[test]
    fn interviewer_question_carries_candidate_context() {
        let out = run(&[
            (Source::User, "я три года работал с kafka в продакшене"),
            (Source::Interviewer, "а с какими версиями брокеров?"),
        ]);
        assert_eq!(out.len(), 1, "{out:?}");
        let msg = &out[0];
        assert!(msg.contains("Пользователь: я три года работал"), "{msg}");
        assert!(msg.contains("Интервьюер: а с какими версиями"), "{msg}");
    }

    #[test]
    fn system_prompt_goes_out_only_once() {
        let out = run(&[
            (
                Source::Interviewer,
                "какой у вас опыт с kafka в продакшене?",
            ),
            (
                Source::Interviewer,
                "а с какими версиями брокеров вы работали?",
            ),
        ]);
        assert_eq!(out.len(), 2);
        assert!(out[0].starts_with(PROMPT), "{:?}", out[0]);
        assert!(!out[1].contains(PROMPT), "prompt repeated: {:?}", out[1]);
    }

    #[test]
    fn first_message_explains_the_tags() {
        let out = run(&[(Source::Interviewer, "какой у вас опыт с kafka?")]);
        assert!(out[0].contains("Диалог"), "{:?}", out[0]);
    }

    #[test]
    fn failed_send_retries_the_same_question() {
        let mut st = ListenState::new(PROMPT);
        let q = "вы работали с kafka в продакшене?";
        assert!(st.on_phrase(Source::Interviewer, q).is_some());
        // No mark_sent: the message was lost, the question must fire again.
        assert!(st.on_phrase(Source::Interviewer, q).is_some());
    }

    #[test]
    fn empty_and_blank_phrases_are_ignored() {
        let mut st = ListenState::new(PROMPT);
        assert!(st.on_phrase(Source::Interviewer, "").is_none());
        assert!(st.on_phrase(Source::Interviewer, "   ").is_none());
        assert!(st.transcript.is_empty());
    }

    // ---- bounds -------------------------------------------------------

    #[test]
    fn transcript_is_bounded_and_line_aligned() {
        let mut st = ListenState::new(PROMPT);
        for _ in 0..200 {
            st.on_phrase(
                Source::Interviewer,
                "разговор про распределённые системы и очереди",
            );
        }
        assert!(
            st.transcript.chars().count() <= MAX_TRANSCRIPT_CHARS,
            "transcript grew to {}",
            st.transcript.chars().count()
        );
        assert!(
            st.transcript.starts_with("Интервьюер:"),
            "{:?}",
            &st.transcript[..40]
        );
    }

    #[test]
    fn interviewer_buffer_is_bounded() {
        let mut st = ListenState::new(PROMPT);
        for _ in 0..200 {
            st.on_phrase(
                Source::Interviewer,
                "очень длинная реплика без вопроса вообще",
            );
        }
        assert!(st.interviewer.chars().count() <= MAX_BUFFER_CHARS);
    }

    #[test]
    fn trim_buffer_keeps_short_text_untouched() {
        let (t, a) = trim_buffer("короткий вопрос?", 0);
        assert_eq!(t, "короткий вопрос?");
        assert_eq!(a, 0);
    }

    #[test]
    fn trim_buffer_drops_old_text_and_rebases_answered_offset() {
        let text = "а ".repeat(MAX_BUFFER_CHARS * 2);
        let (t, a) = trim_buffer(&text, 0);
        assert_eq!(t.chars().count(), MAX_BUFFER_CHARS);
        assert_eq!(a, 0);
        assert!(t.starts_with("а а а"), "oldest text goes first");
    }

    #[test]
    fn trim_buffer_does_not_re_introduce_already_answered_text() {
        let tail = "новый вопрос про kafka?";
        let text = format!("{}{}", "старое ".repeat(MAX_BUFFER_CHARS), tail);
        let answered = MAX_BUFFER_CHARS * 7;
        let (t, a) = trim_buffer(&text, answered);
        assert_eq!(a, 0, "answered prefix is gone, offset must reset");
        assert!(t.ends_with(tail), "newest text kept: {t:?}");
    }
}
