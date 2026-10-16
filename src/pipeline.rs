//! Pipeline glue: audio → streaming STT → LLM → display.
//!
//! Two threads keep listening and answering independent:
//!
//! **Listener**: pulls 8 kHz chunks from the capture, feeds them to T-one in
//! 300 ms slices and accumulates the phrases it finalizes. As soon as the
//! running text looks like an interview question
//! ([`crate::question::is_question`]) and carries enough *new* characters, it
//! is queued for answering **immediately** — the speaker does not have to
//! finish. Whatever comes after the answered prefix becomes «Дополнение…».
//!
//! **Answer worker**: consumes the queue strictly in order (one DeepSeek
//! chat session must be threaded sequentially), drives the LLM through a
//! dedicated current-thread runtime and updates the display.
//!
//! `stop` flips an atomic that ends both loops promptly.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};

use crate::audio::CaptureHandle;
use crate::config::Config;
use crate::display::OverlayCtl;
use crate::llm::DeepSeekWeb;
use crate::question::is_question;
use crate::tone::{CHUNK_SAMPLES, Transcriber};

/// Minimum NEW characters before accumulated text may (re-)fire the LLM.
const MIN_NEW_CHARS: usize = 12;
/// Upper bound for the accumulated text; older speech is dropped.
const MAX_BUFFER_CHARS: usize = 2_000;

/// What to send to the LLM when the transcript crosses a threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Fire {
    /// Message for the DeepSeek session.
    pub message: String,
    /// Transcript length that is now considered answered.
    pub answered_chars: usize,
}

/// Decides when the running transcript should trigger an answer.
///
/// T-one only emits a phrase after ~600 ms of silence, so every phrase is a
/// natural pause: that is the "the speaker stopped here" signal. Firing rules:
///
/// * the accumulated text (or the fresh phrase alone) looks like a question —
///   answer it, even mid-sentence, and mark the question as open;
/// * the speaker keeps talking about an open question — send one
///   «Дополнение…» so the answer sees the rest of what was said;
/// * anything else — stay quiet.
#[derive(Debug, Default)]
pub struct FirePolicy {
    answered_chars: usize,
    question_open: bool,
    followup_sent: bool,
}

impl FirePolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Transcript length already sent to the LLM.
    pub fn answered_chars(&self) -> usize {
        self.answered_chars
    }

    /// Re-base the offset after the transcript buffer was trimmed.
    pub fn set_answered_chars(&mut self, answered_chars: usize) {
        self.answered_chars = answered_chars;
    }

    /// Decide whether the freshly finalized `phrase` triggers an answer.
    ///
    /// `buffer` is the whole transcript so far, `phrase` only the new text.
    pub fn on_phrase(&mut self, buffer: &str, phrase: &str) -> Option<Fire> {
        let n = buffer.chars().count();
        if n < self.answered_chars + MIN_NEW_CHARS {
            return None;
        }
        if is_question(buffer) || is_question(phrase) {
            self.question_open = true;
            self.followup_sent = false;
            return Some(Fire {
                message: buffer.trim().to_string(),
                answered_chars: n,
            });
        }
        if self.question_open && !self.followup_sent {
            self.followup_sent = true;
            return Some(Fire {
                message: format!("Дополнение к предыдущему вопросу: {}", phrase.trim()),
                answered_chars: n,
            });
        }
        None
    }

    /// Call after a [`Fire`] was actually queued (send failures must not
    /// advance the offset).
    pub fn mark_sent(&mut self, fire: &Fire) {
        self.answered_chars = fire.answered_chars;
    }
}

/// Keep the transcript bounded: cut off the oldest text, but never before
/// what the LLM has already been given (that would re-answer old speech).
///
/// Returns the new text plus how many chars of the old answered prefix
/// survived (the caller rebases [`FirePolicy::set_answered_chars`] with it).
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

pub struct Pipeline;

impl Pipeline {
    /// Spawn listener + answer-worker threads.
    /// Set `stop`, then join the returned handle(s) via [`Self::join`].
    pub fn spawn(
        cfg: Config,
        llm: Arc<DeepSeekWeb>,
        session_id: String,
        transcriber: Transcriber,
        capture: CaptureHandle,
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
            .spawn(move || listen(cfg, transcriber, capture, overlay, q_tx, stop))
            .expect("spawn listener");

        LISTENER_WORKER.lock().unwrap().replace(worker);
        listener
    }

    /// Join the answer-worker started by [`Pipeline::spawn`] (listener is
    /// joined by the handle returned there).
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
    mut transcriber: Transcriber,
    capture: CaptureHandle,
    overlay: OverlayCtl,
    q_tx: mpsc::Sender<String>,
    stop: Arc<AtomicBool>,
) {
    let mut pending: Vec<i16> = Vec::with_capacity(CHUNK_SAMPLES * 4);
    let mut text = String::new();
    let mut policy = FirePolicy::new();
    let mut system_sent = false;
    let mut chunks_seen = 0u64;
    let mut stats_at = Instant::now();

    info!("listener started: T-one streaming, {CHUNK_SAMPLES}-sample slices");

    while !stop.load(Ordering::SeqCst) {
        let Some(chunk) = capture.try_recv() else {
            std::thread::sleep(Duration::from_millis(20));
            continue;
        };
        chunks_seen += 1;
        pending.extend_from_slice(&chunk);

        if pending.len() < CHUNK_SAMPLES {
            if stats_at.elapsed() > Duration::from_secs(10) {
                stats_at = Instant::now();
                debug!(
                    "stats: chunks={chunks_seen}, dropped={}, buffered={}",
                    capture.dropped_chunks.load(Ordering::Relaxed),
                    pending.len()
                );
            }
            continue;
        }

        // ---- streaming STT: consume every full 300 ms slice we can ----
        let mut new_text = String::new();
        while pending.len() >= CHUNK_SAMPLES {
            let slice: Vec<i16> = pending.drain(..CHUNK_SAMPLES).collect();
            match transcriber.forward(&slice, false) {
                Ok(phrases) => {
                    for p in phrases {
                        new_text.push_str(&p.text);
                        new_text.push(' ');
                    }
                }
                Err(e) => {
                    error!("STT failed: {e}");
                    break;
                }
            }
        }

        if new_text.trim().is_empty() {
            continue;
        }
        info!("STT: “{}”", new_text.trim());

        // ---- accumulate and answer in parallel with the speaker ----
        text.push_str(&new_text);
        let (trimmed, survived) = trim_buffer(&text, policy.answered_chars());
        text = trimmed;
        policy.set_answered_chars(survived);

        let Some(fire) = policy.on_phrase(&text, new_text.trim()) else {
            continue;
        };
        info!("fire: “{}”", fire.message);
        overlay.set_text("Понял вопрос, отвечаю…");
        if send_raw(
            &q_tx,
            wrap_first(&cfg.system_prompt, &mut system_sent, &fire.message),
        ) {
            policy.mark_sent(&fire);
        }
    }

    info!("listener stopped");
}

fn wrap_first(system_prompt: &str, system_sent: &mut bool, text: &str) -> String {
    if *system_sent {
        text.to_string()
    } else {
        *system_sent = true;
        format!("{system_prompt}\n\nВопрос интервьюера: {text}")
    }
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

    fn policy_after(chunks: &[(&str, &str)]) -> (FirePolicy, Vec<Fire>) {
        let mut p = FirePolicy::new();
        let mut buffer = String::new();
        let mut fires = Vec::new();
        for (phrase, _expect) in chunks {
            buffer.push_str(phrase);
            if let Some(f) = p.on_phrase(&buffer, phrase) {
                p.mark_sent(&f);
                fires.push(f);
            }
        }
        (p, fires)
    }

    #[test]
    fn plain_statement_does_not_fire() {
        let (p, f) = policy_after(&[("я расскажу про свой опыт работы ", "")]);
        assert!(f.is_empty());
        assert_eq!(p.answered_chars(), 0);
    }

    #[test]
    fn question_with_enough_new_chars_fires() {
        let (_, f) = policy_after(&[("вы работали с kafka в продакшене? ", "")]);
        assert_eq!(f.len(), 1);
        assert!(f[0].message.contains("kafka"));
        assert_eq!(
            f[0].answered_chars,
            "вы работали с kafka в продакшене? ".chars().count()
        );
    }

    #[test]
    fn too_few_new_chars_does_not_refire() {
        let mut p = FirePolicy::new();
        p.set_answered_chars("вы работали с kafka?".chars().count());
        assert!(p.on_phrase("вы работали с kafka? да", " да").is_none());
    }

    #[test]
    fn short_text_never_fires() {
        assert!(FirePolicy::new().on_phrase("да?", "да?").is_none());
    }

    #[test]
    fn continuation_of_open_question_sends_one_followup() {
        let (_, f) = policy_after(&[
            ("какие у вас были обязанности? ", ""),
            (
                "я руководил командой из пяти человек и отвечал за найм ",
                "",
            ),
            ("а ещё мы запускали еженедельные релизы ", ""),
        ]);
        assert_eq!(f.len(), 2, "question + exactly one followup: {f:?}");
        assert!(
            f[1].message
                .starts_with("Дополнение к предыдущему вопросу:")
        );
        assert!(f[1].message.contains("руководил"));
    }

    #[test]
    fn followup_after_new_question_is_not_a_supplement() {
        let (_, f) = policy_after(&[
            ("вы работали с kafka в продакшене? ", ""),
            ("а с какими версиями брокеров? ", ""),
        ]);
        assert_eq!(f.len(), 2);
        assert!(
            !f[1].message.starts_with("Дополнение"),
            "{:?}",
            f[1].message
        );
    }

    #[test]
    fn monologue_after_answered_question_stays_quiet() {
        let (_, f) = policy_after(&[
            ("какие у вас были обязанности? ", ""),
            ("спасибо, очень подробно ", ""),
            ("кстати, в следующем раунде спросят про деплой ", ""),
        ]);
        assert_eq!(f.len(), 2, "only the followup is sent: {f:?}");
    }

    #[test]
    fn failed_send_does_not_advance_offset() {
        let mut p = FirePolicy::new();
        let fire = p
            .on_phrase(
                "вы работали с kafka в продакшене?",
                "вы работали с kafka в продакшене?",
            )
            .expect("question fires");
        // Simulate a closed queue: mark_sent is never called.
        assert_eq!(p.answered_chars(), 0);
        let again = p.on_phrase(
            "вы работали с kafka в продакшене? а с какими версиями?",
            " а с какими версиями?",
        );
        assert!(again.is_some(), "must retry after a failed send");
        p.mark_sent(&fire);
        assert_eq!(p.answered_chars(), fire.answered_chars);
    }

    #[test]
    fn trim_keeps_short_text_untouched() {
        let (t, a) = trim_buffer("короткий вопрос?", 0);
        assert_eq!(t, "короткий вопрос?");
        assert_eq!(a, 0);
    }

    #[test]
    fn trim_drops_old_text_and_rebases_answered_offset() {
        let text = "а ".repeat(MAX_BUFFER_CHARS * 2);
        let (t, a) = trim_buffer(&text, 0);
        assert_eq!(t.chars().count(), MAX_BUFFER_CHARS);
        assert_eq!(a, 0);
        assert!(t.starts_with("а а а"), "oldest text goes first");
    }

    #[test]
    fn trim_does_not_re_introduce_already_answered_text() {
        let tail = "новый вопрос про kafka?";
        let text = format!("{}{}", "старое ".repeat(MAX_BUFFER_CHARS), tail);
        let answered = MAX_BUFFER_CHARS * 7;
        let (t, a) = trim_buffer(&text, answered);
        assert_eq!(a, 0, "answered prefix is gone, offset must reset");
        assert!(t.ends_with(tail), "newest text kept: {t:?}");
    }
}
