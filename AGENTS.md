# LIhO — AGENTS.md

## Project state
- Rust project (edition 2024). **End-to-end with parallel answering**, cross-platform and two-sided: mic capture («Пользователь») + system-output loopback («Интервьюер») → **T-one streaming ASR**, one model instance per stream → tagged transcript → pacing gate → answer worker (DeepSeek web, PoW+SSE, single chat) → answer on screen. Linux shows a Wayland overlay, Windows/macOS print to stdout.
- Entry point: `src/main.rs`. Build with `cargo build`, run with `cargo run` and speak; Ctrl-C (or SIGTERM on Unix) stops it.
- CLI flags: `--geometry WxH+X+Y` (size + right/top margins), `--font-size N` (8..72), `--autohide-ms N` (hide overlay N ms after each answer — Linux only). Unknown args ignored; parsing lives in `display::OverlayGeometry::from_args` with unit tests.

## User constraints (honor these)
- Target: interview assistant that captures audio, transcribes in real time, answers questions through the DeepSeek API without Rizoner, for speed.
- Must be OBS-undetectable (overlay mode on Linux), detailed logging for debugging.
- Must run on **Windows and Linux** (macOS works too, console output).
- Rules: KISS, TBC (Talk Before Coding), TDD.

## Build & test
- `cargo build` — compile (first build downloads ONNX Runtime binaries via `ort`)
- `cargo run` — run (needs a microphone, a Wayland session on Linux, `.env`)
- `cargo test` — unit tests
- `cargo test live -- --ignored --nocapture` — live round-trip: real model, real microphone, real DeepSeek chat
- `cargo check --target x86_64-pc-windows-msvc` — cross-compile check (windows-gnu has no ONNX Runtime prebuilds)
- CI: `.github/workflows/ci.yml`, matrix ubuntu-latest + windows-latest, steps = debug build, tests, `clippy -D warnings`, `fmt --check`, release build, artifact upload. Linux needs `libasound2-dev libpipewire-0.3-dev libwayland-dev libxkbcommon-dev` (pkg-config). The runner's clippy is **newer than the local toolchain** (1.99 vs 1.97), so `clippy -D warnings` can flag lints locally unknown — run the same clippy for both targets before pushing.
- `python3 scripts/reference_parity.py` — regenerates the upstream splitter oracle numbers (needs numpy)

## Architecture (src/)
| module | role |
|---|---|
| `audio/` | `mod.rs`: `Source` (User/Interviewer) + labels, resampler, `CaptureHandle` with a backend-specific stop action. `mic.rs`: cpal default input. `loopback_linux.rs`: PipeWire sink monitor. `loopback_windows.rs`: WASAPI loopback. Bounded channel, drop (never block) on overflow |
| `tone/` | T-one streaming ASR: `mod.rs` ONNX inference, `splitter.rs` logprob→phrases, `decoder.rs` greedy CTC |
| `pipeline.rs` | listener thread (audio→phrases→tagged transcript→pacing gate) + answer worker thread, `RequestGate`, latest-wins `Inbox` |
| `llm.rs` + `pow.rs` | DeepSeek web chat client, PoW gate (vendored site wasm), warm-header pool |
| `display/` | `mod.rs` geometry + backend selection, `wayland.rs` (Linux), `console.rs` (Windows/macOS) |
| `config.rs` | `.env` + env vars |

## Gotchas
- **T-one contract** (`t-tech/T-one`, Apache-2.0, 71M streaming CTC, Russian, telephony): input `signal` = int16 `[1, 2400, 1]` (300 ms @ 8 kHz) plus `state` = f16 `[1, 219729]`; outputs logprobs f32 `[1, 10, 35]` then the next state. Log-mel features and pre-emphasis are **inside** the ONNX graph — no feature code in Rust. Alphabet is 33 Russian letters + space (index 34 = CTC blank).
- `ort` quirks: `outputs.values()` yields `ValueRef`; use `try_extract_array::<f16>()` for the state (`try_extract_tensor::<f16>()` mis-resolves); `Session::run` needs `&mut self`; the `half` feature is required for f16 tensors; **one Environment per process** — `tone::shared_env()` keeps it in a `OnceLock` so both streams can own a session. ONNX Runtime's own logging is routed through a filter (Warning+) or it buries the app log.
- The splitter port is pinned to the upstream Python by `reference_parity` tests — `np.pad` on a **bool** array pads with `False`, which is what makes the first recorded silence boundary a *start* at `-MIN_SILENCE`. Do not "simplify" that away. Upstream asserts the stream starts with silence; our port returns nothing instead of panicking.
- Greedy decoding emits no spaces on its own — space is a normal label (index 33), a blank is not a word separator.
- Phrase = "the speaker paused ≥600 ms" (splitter), so **every phrase arrival is a natural pause** — `FirePolicy` relies on that: question → answer, at most one more fire for the rest of an open question, silence otherwise. The message sent is the whole tagged transcript, not a «Дополнение…» string (the dialogue format carries that context better).
- Capture config ranking: i16 > f32, then fewest channels, then rate closest to 16 kHz (T-one needs 8 kHz; PipeWire `default` may offer 384 kHz × 32 ch — 48× the resampling work for nothing).
- Loopback per OS: Linux = PipeWire monitor (`STREAM_CAPTURE_SINK=true`, F32LE, PipeWire main loop on the backend thread); Windows = WASAPI — client from the **Render** endpoint, then `initialize_client(..., Direction::Capture, ...)` is what sets `AUDCLNT_STREAMFLAGS_LOOPBACK`, and data is read with `AudioCaptureClient::read_from_device_to_deque`. Note `initialize_mta()` returns `HRESULT`, not `Result`.
- Loopback failure is not fatal: warn and run on the microphone alone.
- **No question detector** — it was deleted. T-one's alphabet is 33 Cyrillic letters + space, so a `text.ends_with('?')` rule can never fire and a starter-word rule gated on `.`/`!`/`…` was dead too; both were removed rather than patched. Everything the interviewer says is forwarded now, and whether it was a question is the model's call (see `LIHO_SYSTEM_PROMPT`). Only pacing stays local: `RequestGate` (≥30 new interviewer chars AND ≥3 s since the last request) and `Inbox` (latest-wins, so a busy worker never answers a stale question).
- DeepSeek SSE protocol (captured live, tests pinned to it): an answer can span **two** content paths — `response/content` for the first characters and `response/fragments/<i>/content` (index may be `-1`) for the rest. The opening `update_session` blob's `fragments[]` is a **partial snapshot** (live: "Кон" of a 150-word answer) and must never truncate the stream; it is a fallback for one-word replies that arrive complete inside the blob. `message_id` still comes from that blob.
- DeepSeek web API (`chat.deepseek.com/api/v0/*`) with browser `userToken` from localStorage; `DEEPSEEK_TOKEN` env or `.env`, never logged.
- Completion requires a PoW header (`POST /api/v0/chat/create_pow_challenge`, solved by vendored `assets/pow_solver.wasm` via wasmi, ~20-30 s). Prefetch: `DeepSeekWeb::schedule_refill()` keeps a warm 300 s header so an answer only streams (~1 s).
- LLM tokens expire → 401/403 are fatal; 40300 `MISSING_HEADER` is flaky and retried up to 3× (it arrives as the whole SSE body, so it must be handled before the stream parser).
- Display backends are cfg-gated in `display/mod.rs`; Wayland deps live in `[target.'cfg(target_os = "linux")'.dependencies]`. Overlay protocol: no buffer attach before the first configure+ack, else the connection dies.
- OBS-undetectable on Linux: Hyprland layer rule `hl.layer_rule{ match={namespace="^LIhO$"}, no_screen_share=true }` keeps the overlay out of share frames; `--autohide-ms` shrinks the exposure window further. There is no Windows equivalent — the console backend prints only when an answer arrives.

## Principles (strict)
- No hardcoded secrets — DeepSeek token via `DEEPSEEK_TOKEN` env var or `.env`.
- No костылей (hacks) — every crate and function must have a clear purpose; no dead code (unused API is deleted, not `#[allow]`ed).
- No architectural problems — single binary, well-separated modules (audio → STT → LLM → output).
- TDD first — write a test that compiles and fails before writing production code. Ported algorithms get parity tests against the reference implementation.