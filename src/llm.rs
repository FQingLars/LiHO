//! DeepSeek web-chat client (unofficial `chat.deepseek.com/api/v0` endpoints).
//! Auth: browser localStorage userToken sent as Bearer + Cookie.
//! Context lives server-side in one persistent chat session ("single chat" rule).
//! Thinking is disabled by design (speed over reasoning).

use crate::pow::{PowChallenge, PowSolver, pow_response_header};
use log::{debug, info, warn};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::Ordering;

pub const BASE_URL: &str = "https://chat.deepseek.com";

const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";

const COMPLETION_PATH: &str = "/api/v0/chat/completion";

/// Browser-mimicking headers including auth. Pure, unit-tested.
/// `pow` is the base64 `x-ds-pow-response` value when solving is required.
pub(crate) fn build_headers(token: &str, pow: Option<&str>) -> Result<HeaderMap, String> {
    let mut h = HeaderMap::new();

    let mut insert = |key: &str, value: String| -> Result<(), String> {
        let name = HeaderName::from_bytes(key.as_bytes())
            .map_err(|e| format!("bad header name {key}: {e}"))?;
        let val = HeaderValue::from_str(&value)
            .map_err(|e| format!("bad header value for {key}: {e}"))?;
        h.insert(name, val);
        Ok(())
    };

    insert("Authorization", format!("Bearer {token}"))?;
    insert("Content-Type", "application/json".to_string())?;
    insert("Accept", "application/json".to_string())?;
    insert("User-Agent", USER_AGENT.to_string())?;
    insert("Origin", BASE_URL.to_string())?;
    insert("Referer", format!("{BASE_URL}/"))?;
    insert("Cookie", format!("token={token}"))?;
    if let Some(p) = pow {
        insert("x-ds-pow-response", p.to_string())?;
    }
    Ok(h)
}

/// Body for creating a new chat session.
pub(crate) fn create_session_body() -> Value {
    json!({ "agent": "chat" })
}

/// Body for one completion turn inside a session.
pub(crate) fn completion_body(
    session_id: &str,
    parent_message_id: Option<i64>,
    prompt: &str,
) -> Value {
    json!({
        "chat_session_id": session_id,
        "parent_message_id": parent_message_id,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": false,
        "search_enabled": false,
    })
}

/// Defensive session-id extraction: known paths first, fallback second.
pub(crate) fn extract_session_id(v: &Value) -> Option<String> {
    v.pointer("/data/biz_data/id")
        .or_else(|| v.pointer("/data/id"))
        .and_then(|x| x.as_str())
        .map(String::from)
}

fn preview(s: &str) -> String {
    s.chars().take(300).collect()
}

/// Incremental SSE parser for the DeepSeek web JSON-patch protocol.
/// Feed raw network chunks; it buffers blank-line-separated events and
/// accumulates answer content.
///
/// Protocol (captured live, see tests):
/// - `data:` lines carry JSON patches;
/// - `{"p":"<path>","o":"<op>","v":<val>}` — path persists until changed,
///   so continuations may omit `p`/`o` and carry only `v`;
/// - answer text lives under path `response/content`;
/// - stream ends when `response/status` receives `"FINISHED"`;
/// - `update_session` carries a full-state blob whose
///   `v.response.message_id` is needed for threading the next turn.
#[derive(Default)]
pub(crate) struct SseAccumulator {
    buf: Vec<u8>,
    text: String,
    finished: bool,
    last_message_id: Option<i64>,
    current_path: Option<String>,
}

impl SseAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk; returns `true` when the stream has finished.
    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Result<bool, String> {
        // Normalize CRLF/CR to LF so event boundaries always look like "\n\n".
        self.buf
            .extend(chunk.iter().map(|&b| if b == b'\r' { b'\n' } else { b }));
        self.process_buffer()
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn message_id(&self) -> Option<i64> {
        self.last_message_id
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }

    fn process_buffer(&mut self) -> Result<bool, String> {
        while let Some(pos) = self.buf.windows(2).position(|w| w == b"\n\n") {
            let event: Vec<u8> = self.buf.drain(..pos).collect();
            self.buf.drain(..2);
            let event_str = String::from_utf8_lossy(&event);

            for line in event_str.lines() {
                let Some(data) = line.trim().strip_prefix("data:") else {
                    continue; // `event:` names, comments, blanks
                };
                match serde_json::from_str::<Value>(data.trim()) {
                    Ok(v) => self.apply_event(&v),
                    Err(e) => warn!("SSE: unparsable data line ignored: {e}"),
                }
            }
        }
        Ok(self.finished)
    }

    fn apply_event(&mut self, v: &Value) {
        if let Some(p) = v.get("p").and_then(Value::as_str) {
            self.current_path = Some(p.to_string());
        }
        // Full-state blob from `update_session`.
        if let Some(mid) = v.pointer("/v/response/message_id").and_then(Value::as_i64) {
            self.last_message_id = Some(mid);
        }

        match self.current_path.as_deref() {
            Some("response/content") => {
                if let Some(s) = v.get("v").and_then(Value::as_str) {
                    match v.get("o").and_then(Value::as_str) {
                        Some("SET") => self.text.clear(),
                        _ => {} // APPEND or omitted
                    }
                    if !s.is_empty() {
                        self.text.push_str(s);
                    }
                }
            }
            Some("response/status") => {
                if v.get("v").and_then(Value::as_str) == Some("FINISHED") {
                    debug!("SSE: stream finished");
                    self.finished = true;
                }
            }
            _ => {} // token usage, thinking, unknown paths
        }
    }
}

/// Thin async wrapper over the web endpoints.
pub struct DeepSeekWeb {
    http: reqwest::Client,
    token: String,
    /// wasm solver state is mutable; lock held only during CPU-bound solve.
    pow: Arc<std::sync::Mutex<PowSolver>>,
    /// Pre-solved PoW headers ready to attach instantly. Solving costs
    /// ~20-30 s (wasmi), but challenges live 300 s — so we keep one warm
    /// at all times and answers pay zero PoW latency.
    ready_pow: std::sync::Mutex<Option<ReadyPow>>,
    /// Guards against two concurrent wasm solves (wasmi state + CPU burn).
    refill_in_flight: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Clone)]
struct ReadyPow {
    header: String,
    fetched_at: std::time::Instant,
}

/// A cached entry older than this is refreshed proactively
/// (challenge TTL is 300 s; leave a wide safety margin).
const POW_STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(150);

impl DeepSeekWeb {
    pub fn new(token: String) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(|e| format!("HTTP client build failed: {e}"))?;
        let pow = PowSolver::new()?;
        Ok(Self {
            http,
            token,
            pow: Arc::new(std::sync::Mutex::new(pow)),
            ready_pow: std::sync::Mutex::new(None),
            refill_in_flight: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// True when a fresh pre-solved header is available right now.
    pub fn has_ready_pow(&self) -> bool {
        self.ready_pow
            .lock()
            .unwrap()
            .as_ref()
            .map(|r| r.fetched_at.elapsed() < POW_STALE_AFTER)
            .unwrap_or(false)
    }

    /// Solve in background so the next question finds a warm header.
    /// MUST be called from within a Tokio runtime context (worker wraps it
    /// in `rt.block_on`). No-op while another solve is already running.
    pub fn schedule_refill(self: &Arc<Self>) {
        if self.refill_in_flight.swap(true, Ordering::SeqCst) {
            debug!("PoW refill already in flight");
            return;
        }
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let res = this.refill_pow().await;
            this.refill_in_flight.store(false, Ordering::SeqCst);
            if let Err(e) = res {
                warn!("PoW prefetch failed: {e}");
            }
        });
    }

    /// Wait (bounded) for an in-flight solve to land in the cache.
    async fn wait_for_inflight(&self) -> Result<String, String> {
        const MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(90);
        let started = std::time::Instant::now();
        loop {
            if let Some(ready) = self.ready_pow.lock().unwrap().take() {
                return Ok(ready.header);
            }
            if !self.refill_in_flight.load(Ordering::SeqCst) {
                return Err("refill finished without producing a header".into());
            }
            if started.elapsed() > MAX_WAIT {
                return Err("timed out waiting for in-flight PoW solve".into());
            }
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }
    }

    /// Solve a fresh challenge off-thread and cache the header.
    async fn refill_pow(&self) -> Result<(), String> {
        let challenge = self.create_pow_challenge().await?;
        let solved = {
            let solver = Arc::clone(&self.pow);
            let challenge = challenge.clone();
            tokio::task::spawn_blocking(move || {
                let mut s = solver.lock().map_err(|_| "PoW poisoned")?;
                s.solve(&challenge)
            })
            .await
            .map_err(|e| format!("PoW join: {e}"))?
        };
        let header = pow_response_header(&challenge, solved?);
        *self.ready_pow.lock().unwrap() = Some(ReadyPow {
            header,
            fetched_at: std::time::Instant::now(),
        });
        Ok(())
    }

    /// Instantly return a fresh cached header, or solve inline as fallback.
    async fn take_pow_header(&self) -> Result<String, String> {
        if let Some(ready) = self.ready_pow.lock().unwrap().take() {
            if ready.fetched_at.elapsed() < POW_STALE_AFTER {
                debug!("PoW: using prefetched header");
                return Ok(ready.header);
            }
            debug!("PoW: cached entry stale");
        }
        if self.refill_in_flight.load(Ordering::SeqCst) {
            debug!("PoW: waiting for in-flight solve");
            return self.wait_for_inflight().await;
        }
        self.refill_pow().await?;
        Ok(self
            .ready_pow
            .lock()
            .unwrap()
            .take()
            .expect("just filled")
            .header)
    }

    /// Fetch a PoW challenge for the completion endpoint.
    pub(crate) async fn create_pow_challenge(&self) -> Result<PowChallenge, String> {
        debug!("LLM: requesting PoW challenge");
        let url = format!("{BASE_URL}/api/v0/chat/create_pow_challenge");
        let resp = self
            .http
            .post(&url)
            .headers(build_headers(&self.token, None)?)
            .json(&json!({ "target_path": COMPLETION_PATH }))
            .send()
            .await
            .map_err(|e| format!("pow challenge request failed: {e}"))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        debug!("pow challenge [{status}]: {}", preview(&body));
        if !status.is_success() {
            return Err(format!(
                "pow challenge failed ({status}): {}",
                preview(&body)
            ));
        }
        let v: Value = serde_json::from_str(&body)
            .map_err(|e| format!("pow challenge bad JSON: {e}; body: {}", preview(&body)))?;
        let parsed: PowChallenge = serde_json::from_value(
            v.pointer("/data/biz_data/challenge")
                .cloned()
                .ok_or_else(|| format!("no challenge in response: {}", preview(&body)))?,
        )
        .map_err(|e| format!("challenge parse failed: {e}"))?;
        info!(
            "PoW challenge: difficulty={}, expire_at={}",
            parsed.difficulty, parsed.expire_at
        );
        Ok(parsed)
    }

    /// Create a persistent chat session; returns its id.
    pub async fn create_session(&self) -> Result<String, String> {
        info!("LLM: creating chat session");
        let url = format!("{BASE_URL}/api/v0/chat_session/create");
        let resp = self
            .http
            .post(&url)
            .headers(build_headers(&self.token, None)?)
            .json(&create_session_body())
            .send()
            .await
            .map_err(|e| format!("create session request failed: {e}"))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        debug!("create session [{status}]: {}", preview(&body));
        if !status.is_success() {
            return Err(format!(
                "create session failed ({status}): {} (is DEEPSEEK_TOKEN expired?)",
                preview(&body)
            ));
        }
        let v: Value = serde_json::from_str(&body)
            .map_err(|e| format!("create session bad JSON: {e}; body: {}", preview(&body)))?;
        extract_session_id(&v)
            .ok_or_else(|| format!("no session id in response: {}", preview(&body)))
    }

    /// Send one prompt into the session; streams the SSE answer.
    /// Returns (full answer text, assistant message_id for threading).
    ///
    /// The completion endpoint intermittently rejects even valid PoW with
    /// `40300 MISSING_HEADER` (observed flaky across identical requests;
    /// reference implementations retry). We retry up to 3 times with a
    /// **fresh** challenge each attempt. Failures after the stream started
    /// producing content are NOT retried (would duplicate the answer).
    pub async fn ask(
        &self,
        session_id: &str,
        parent_message_id: Option<i64>,
        prompt: &str,
    ) -> Result<(String, Option<i64>), String> {
        info!(
            "LLM: prompt to session {session_id} ({} chars)",
            prompt.chars().count()
        );
        const MAX_ATTEMPTS: usize = 3;
        let mut last_retriable = String::new();
        for attempt in 1..=MAX_ATTEMPTS {
            match self
                .attempt_completion(session_id, parent_message_id, prompt)
                .await
            {
                Ok(done) => return Ok(done),
                Err(AttemptError::Retriable(e)) => {
                    warn!("LLM attempt {attempt}/{MAX_ATTEMPTS} failed (retriable): {e}");
                    last_retriable = e;
                    tokio::time::sleep(std::time::Duration::from_millis(300 * attempt as u64))
                        .await;
                }
                Err(AttemptError::Fatal(e)) => return Err(e),
            }
        }
        Err(last_retriable)
    }

    async fn attempt_completion(
        &self,
        session_id: &str,
        parent_message_id: Option<i64>,
        prompt: &str,
    ) -> Result<(String, Option<i64>), AttemptError> {
        // PoW gate: use the prefetched header (0 ms) or solve inline.
        let pow_header = self.take_pow_header().await.map_err(|e| {
            warn!("PoW inline solve failed, retrying later: {e}");
            AttemptError::Retriable(e)
        })?;

        let url = format!("{BASE_URL}{COMPLETION_PATH}");
        let resp = self
            .http
            .post(&url)
            .headers(build_headers(&self.token, Some(&pow_header)).map_err(AttemptError::Fatal)?)
            .json(&completion_body(session_id, parent_message_id, prompt))
            .send()
            .await
            .map_err(|e| AttemptError::Retriable(format!("completion request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let t = resp.text().await.unwrap_or_default();
            debug!("completion [{status}]: {}", preview(&t));
            let msg = format!("completion failed ({status}): {}", preview(&t));
            // Auth problems will never heal on retry.
            return Err(if status.as_u16() == 401 || status.as_u16() == 403 {
                AttemptError::Fatal(format!("{msg} (is DEEPSEEK_TOKEN expired?)"))
            } else {
                AttemptError::Retriable(msg)
            });
        }

        use futures_util::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut acc = SseAccumulator::new();
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| AttemptError::Fatal(format!("stream read failed: {e}")))?;
            debug!(
                "SSE chunk ({} B): {}",
                chunk.len(),
                String::from_utf8_lossy(&chunk).trim_end()
            );
            let done = acc.feed(&chunk).map_err(AttemptError::Fatal)?;
            if done {
                break;
            }
        }
        if !acc.is_finished() {
            if acc.text().is_empty() {
                // Nothing streamed — server-side hiccup, safe to retry.
                return Err(AttemptError::Retriable(
                    "stream ended without finish marker and no content".into(),
                ));
            }
            return Err(AttemptError::Fatal(format!(
                "stream ended without finish marker; partial: {}",
                preview(acc.text())
            )));
        }
        let text = acc.text().to_string();
        info!(
            "LLM: answer ready ({} chars), message_id={:?}",
            text.chars().count(),
            acc.message_id()
        );
        Ok((text, acc.message_id()))
    }
}

/// Distinguishes errors that a fresh-challenge retry can plausibly fix
/// from those that cannot succeed on any attempt.
enum AttemptError {
    Retriable(String),
    Fatal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_carry_bearer_and_cookie() {
        let h = build_headers("tok123", None).unwrap();
        assert_eq!(
            h.get("authorization").unwrap().to_str().unwrap(),
            "Bearer tok123"
        );
        assert_eq!(h.get("cookie").unwrap().to_str().unwrap(), "token=tok123");
        assert_eq!(h.get("origin").unwrap().to_str().unwrap(), BASE_URL);
        assert!(
            h.get("user-agent")
                .unwrap()
                .to_str()
                .unwrap()
                .contains("Mozilla")
        );
        assert_eq!(
            h.get("content-type").unwrap().to_str().unwrap(),
            "application/json"
        );
    }

    #[test]
    fn headers_carry_pow_response_when_given() {
        let h = build_headers("tok123", Some("eyJhbnN3ZXIiOjF9")).unwrap();
        assert!(h.get("x-ds-pow-response").is_some());
    }

    #[test]
    fn headers_omit_pow_when_absent() {
        let h = build_headers("tok123", None).unwrap();
        assert!(h.get("x-ds-pow-response").is_none());
    }

    #[test]
    fn create_session_body_has_agent_chat() {
        assert_eq!(create_session_body()["agent"], json!("chat"));
    }

    #[test]
    fn completion_body_fields_without_parent() {
        let b = completion_body("sess1", None, "hi");
        assert_eq!(b["chat_session_id"], json!("sess1"));
        assert_eq!(b["prompt"], json!("hi"));
        assert_eq!(b["thinking_enabled"], json!(false));
        assert_eq!(b["search_enabled"], json!(false));
        assert_eq!(b["ref_file_ids"], json!([]));
        assert_eq!(b["parent_message_id"], Value::Null);
    }

    #[test]
    fn completion_body_fields_with_parent() {
        let b = completion_body("s", Some(42), "q");
        assert_eq!(b["parent_message_id"], json!(42));
    }

    #[test]
    fn extract_session_id_prefers_biz_data_path() {
        let v = json!({"success": true, "data": {"biz_data": {"id": "abc"}}});
        assert_eq!(extract_session_id(&v).as_deref(), Some("abc"));
    }

    #[test]
    fn extract_session_id_falls_back_to_data_id() {
        let v = json!({"data": {"id": "xyz"}});
        assert_eq!(extract_session_id(&v).as_deref(), Some("xyz"));
    }

    #[test]
    fn extract_session_id_none_when_absent() {
        assert_eq!(extract_session_id(&json!({"error": 1})), None);
    }

    // ---- SSE fixtures below are byte-for-byte captures from the live API ----

    const UPDATE_SESSION_BLOB: &str = concat!(
        "event: update_session\n",
        "data: {\"updated_at\":1787683446.889967}\n",
        "\n",
        "data: {\"v\":{\"response\":{\"message_id\":2,\"parent_id\":1,\"model\":\"\",",
        "\"role\":\"ASSISTANT\",\"thinking_enabled\":false,\"ban_edit\":false,",
        "\"ban_regenerate\":false,\"status\":\"WIP\",\"incomplete_message\":null,",
        "\"accumulated_token_usage\":0,\"files\":[],\"feedback\":null,",
        "\"inserted_at\":1787683446.8733249,\"search_enabled\":false,",
        "\"content\":\"\",\"thinking_content\":null,\"thinking_elapsed_secs\":null,",
        "\"search_status\":null,\"search_results\":null,\"tips\":[]}}}\n",
        "\n"
    );

    #[test]
    fn sse_full_live_stream_roundtrip() {
        let mut acc = SseAccumulator::new();
        let mut stream = String::from("event: ready\n\n");
        stream.push_str(UPDATE_SESSION_BLOB);
        stream.push_str("data: {\"p\":\"response/content\",\"o\":\"APPEND\",\"v\":\"Ч\"}\n\n");
        stream.push_str("data: {\"v\":\"еты\"}\n\n");
        stream.push_str("data: {\"v\":\"ре\"}\n\n");
        stream.push_str(
            "data: {\"p\":\"response/accumulated_token_usage\",\"o\":\"SET\",\"v\":51}\n\n",
        );
        stream.push_str("data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n");

        assert!(acc.feed(stream.as_bytes()).unwrap());
        assert_eq!(acc.text(), "Четыре");
        assert_eq!(acc.message_id(), Some(2));
        assert!(acc.is_finished());
    }

    #[test]
    fn sse_continuation_after_usage_path_does_not_pollute_text() {
        let mut acc = SseAccumulator::new();
        acc.feed(b"data: {\"p\":\"response/content\",\"o\":\"APPEND\",\"v\":\"ok\"}\n\n")
            .unwrap();
        acc.feed(b"data: {\"p\":\"response/accumulated_token_usage\",\"o\":\"SET\",\"v\":51}\n\n")
            .unwrap();
        acc.feed(b"data: {\"v\":7}\n\n").unwrap(); // usage continuation — not text
        assert_eq!(acc.text(), "ok");
    }

    #[test]
    fn sse_set_op_on_content_replaces_text() {
        let mut acc = SseAccumulator::new();
        acc.feed(b"data: {\"p\":\"response/content\",\"o\":\"APPEND\",\"v\":\"old\"}\n\n")
            .unwrap();
        acc.feed(b"data: {\"p\":\"response/content\",\"o\":\"SET\",\"v\":\"new\"}\n\n")
            .unwrap();
        assert_eq!(acc.text(), "new");
    }

    #[test]
    fn sse_handles_chunks_split_across_feed_boundaries() {
        let mut acc = SseAccumulator::new();
        let full = b"data: {\"p\":\"response/content\",\"o\":\"APPEND\",\"v\":\"split\"}\n\n\
                     data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n";
        let (a, b) = full.split_at(25);
        assert!(!acc.feed(a).unwrap());
        assert!(acc.feed(b).unwrap());
        assert_eq!(acc.text(), "split");
    }

    #[test]
    fn sse_crlf_boundaries_are_normalized() {
        let mut acc = SseAccumulator::new();
        let payload = b"data: {\"p\":\"response/content\",\"v\":\"crlf\"}\r\n\r\n\
                        data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\r\n\r\n";
        assert!(acc.feed(payload).unwrap());
        assert_eq!(acc.text(), "crlf");
    }

    #[test]
    fn sse_title_and_close_events_are_ignored() {
        let mut acc = SseAccumulator::new();
        let payload = b"event: title\ndata: {\"content\":\"2+2=4\"}\n\n\
                        event: close\ndata: {\"click_behavior\":\"none\",\"auto_resume\":false}\n\n";
        assert!(!acc.feed(payload).unwrap());
        assert_eq!(acc.text(), "");
    }

    #[test]
    fn sse_invalid_json_lines_are_ignored() {
        let mut acc = SseAccumulator::new();
        let payload =
            b": keep-alive\ndata: not-json{{{\ndata: {\"p\":\"response/content\",\"v\":\"ok\"}\n\n";
        assert!(!acc.feed(payload).unwrap());
        assert_eq!(acc.text(), "ok");
    }

    /// Mine real oracle tuples (challenge, prefix, difficulty → minimal
    /// nonce) for the threshold reverse-engineering. Solves run on a small
    /// thread pool (each thread owns its own wasm instance).
    /// Run: cargo test --release mine_oracles -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "mining: hits live API repeatedly"]
    async fn mine_oracle_tuples() {
        let count: usize = std::env::var("LIHO_MINE_COUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12);
        let _ = dotenvy::dotenv();
        let token = std::env::var("DEEPSEEK_TOKEN").expect("DEEPSEEK_TOKEN");
        let client = DeepSeekWeb::new(token).unwrap();

        // 1. Collect challenges politely.
        let mut challenges = Vec::with_capacity(count);
        for i in 0..count {
            let c = client.create_pow_challenge().await.expect("challenge");
            println!(
                "fetched {}/{}: d={} expire={}",
                i + 1,
                count,
                c.difficulty,
                c.expire_at
            );
            challenges.push(c);
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }

        // 2. Solve in parallel (threads × own solver instance).
        let jobs: Vec<std::sync::mpsc::Receiver<_>> = challenges
            .iter()
            .map(|c| {
                let c = c.clone();
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let mut s = crate::pow::PowSolver::new().expect("solver");
                    let _ = tx.send(s.solve(&c));
                });
                rx
            })
            .collect();

        let mut tuples = Vec::new();
        for (c, rx) in challenges.iter().zip(jobs) {
            let answer: f64 = rx.recv().expect("solver thread died").expect("solved");
            tuples.push(serde_json::json!({
                "challenge": c.challenge,
                "salt": c.salt,
                "expire_at": c.expire_at,
                "difficulty": c.difficulty,
                "signature": c.signature,
                "target_path": c.target_path,
                "answer": answer,
            }));
            println!("✓ n*={answer:.0} salt={}", c.salt);
        }
        std::fs::write(
            "/tmp/opencode/liho_oracles.json",
            serde_json::to_string_pretty(&tuples).unwrap(),
        )
        .unwrap();
        println!(
            "saved {} tuples → /tmp/opencode/liho_oracles.json",
            tuples.len()
        );
    }

    /// Live round-trip against chat.deepseek.com.
    /// Run manually: cargo test live -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "live API: needs DEEPSEEK_TOKEN in .env"]
    async fn live_ask_roundtrip() {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(true)
            .try_init();
        let _ = dotenvy::dotenv();
        let token = std::env::var("DEEPSEEK_TOKEN").expect("DEEPSEEK_TOKEN must be set");
        let client = Arc::new(DeepSeekWeb::new(token).unwrap());
        let sid = client.create_session().await.unwrap();
        // Cold: no cached header → inline solve dominates.
        let t0 = std::time::Instant::now();
        let (text1, mid) = client
            .ask(&sid, None, "Ответь ровно одним словом: сколько будет 2+2?")
            .await
            .unwrap();
        println!("COLD ASK: {:.1}s → {text1}", t0.elapsed().as_secs_f32());
        assert!(!text1.trim().is_empty());
        // Warm: header refilled after the first answer → stream-only latency.
        client.schedule_refill();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while !client.has_ready_pow() {
            assert!(
                std::time::Instant::now() < deadline,
                "prefetch never became ready"
            );
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        let t1 = std::time::Instant::now();
        let (text2, _) = client
            .ask(&sid, mid, "И ещё: 3+3? Одним словом.")
            .await
            .unwrap();
        println!("WARM ASK: {:.1}s → {text2}", t1.elapsed().as_secs_f32());
        assert!(!text2.trim().is_empty());
    }
}
