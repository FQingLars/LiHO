# LIhO

Real-time interview assistant: streaming Russian speech recognition on a
microphone, an LLM answer while the interviewer is still talking, and the
answer on screen.

```
microphone ──▶ cpal ─────────────┐
                                ├─▶ 8 kHz mono i16 ──▶ T-one (ONNX, streaming CTC)
system output ─▶ loopback ───────┘        (one model instance per stream)
                                                 │ phrases
                                                 ▼
                                   tagged transcript
                              «Интервьюер: …» / «Пользователь: …»
                                                 │
                                       question heuristic
                                       (interviewer only)
                                                 ▼
                             DeepSeek web chat ──▶ answer on screen
```

* **STT**: [T-one](https://huggingface.co/t-tech/T-one) (`t-tech/T-one`,
  Apache-2.0) — a 71M-parameter streaming CTC Conformer for Russian,
  specialised for telephony. Fed with raw 8 kHz int16 in 300 ms chunks; it
  emits finalized phrases ~600 ms after a pause. The Rust port lives in
  `src/tone/` (acoustic model via ONNX Runtime, logprob splitter, greedy CTC
  decoder). Streaming state is per audio source, so each speaker gets its own
  model instance.
* **LLM**: DeepSeek web chat (`chat.deepseek.com/api/v0/*`) with a browser
  `userToken`, a PoW gate and a single threaded chat session.
* **Display**: a Wayland layer-shell overlay on Linux, stdout on Windows and
  macOS.

## Platforms

| | microphone | system output (loopback) | answer output |
|---|---|---|---|
| Linux | `cpal` (PulseAudio/ALSA) | PipeWire sink monitor | Wayland overlay panel (top-right, click-through, `--autohide-ms`) |
| Windows | `cpal` (WASAPI) | WASAPI loopback | stdout |
| macOS | `cpal` (CoreAudio) | — (not implemented) | stdout |

The two audio sources are tagged in the transcript: the microphone is
«Пользователь» (the candidate), the system output is «Интервьюер» (the call).
Only the interviewer's words can trigger an answer; the candidate's speech is
context for it. If the loopback cannot start, LIhO logs a warning and keeps
running on the microphone alone.

## Build

Rust 1.88+ (edition 2024). The ONNX Runtime binaries are downloaded by the
`ort` crate on first build.

```bash
# Linux
cargo build --release

# Windows (MSVC toolchain; ONNX Runtime has no prebuilt windows-gnu)
cargo build --release
```

## Run

```bash
./scripts/fetch_model.sh      # downloads models/t-one.onnx (~138 MB)
export DEEPSEEK_TOKEN=<userToken from chat.deepseek.com localStorage>
cargo run --release
```

Speak; answers appear as soon as a question is recognizable. Flags:

| flag | meaning |
|---|---|
| `--geometry 500x180+40+20` | overlay size and top/right margins (Linux) |
| `--font-size 18` | overlay font size, 8..72 (Linux) |
| `--autohide-ms 4000` | hide the overlay N ms after each answer (Linux) |

Configuration (env or `.env`):

| variable | default | meaning |
|---|---|---|
| `DEEPSEEK_TOKEN` | — | **required**, DeepSeek `userToken` |
| `LIHO_STT_MODEL` | `models/t-one.onnx` | path to the T-one ONNX model |
| `LIHO_SYSTEM_PROMPT` | role prompt in `src/config.rs` | how the assistant should answer |

## Tests

```bash
cargo test                                    # unit tests, no hardware needed
cargo test live -- --ignored --nocapture       # real model, real microphone, real LLM
```

Live tests are `#[ignore]`d: `live_capture` needs a microphone, `live_tone*`
need `models/t-one.onnx`, `live_overlay` needs a Wayland session.

`src/tone/splitter.rs::reference_parity` pins the Rust port of the upstream
logprob splitter to the numbers produced by the original Python code;
`scripts/reference_parity.py` regenerates them (`python3`, needs `numpy`).

## Notes

* T-one is Russian-only and telephony-oriented.
* Use headphones: with speakers, the microphone hears the interviewer too and
  both streams transcribe the same words. There is no echo cancellation.
* The greedy decoder is used instead of the upstream KenLM beam search
  because KenLM has no Windows build.
* On Linux with Hyprland, adding
  `hl.layer_rule{ match={namespace="^LIhO$"}, no_screen_share=true }` to
  `hyprland.lua` makes the compositor skip the overlay in screen-share frames.

## License

Code in this repository is MIT-style (see LICENSE if present). T-one and its
weights are Apache-2.0 and belong to T-Software DC.