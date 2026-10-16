//! Streaming ASR: **T-one** (`t-tech/T-one`, Apache-2.0) — a 71M-parameter
//! streaming CTC Conformer for Russian, exported to ONNX.
//!
//! The pipeline is ported 1:1 from the reference Python implementation
//! (<https://github.com/voicekit-team/T-one>):
//!
//! ```text
//!   int16 PCM 8 kHz ──▶ [ ONNX acoustic model ] ──▶ logprobs [T, 35]
//!                                                     │
//!                        ┌────────────────────────────┘
//!                        ▼
//!               StreamingLogprobSplitter ──▶ phrases (logprobs + frames)
//!                        │
//!                        ▼
//!                GreedyCTCDecoder ──▶ text
//! ```
//!
//! Everything audio-feature related (log-mel filterbank, pre-emphasis, FFT) is
//! baked into the ONNX graph: the model consumes **raw int16 samples**, 2400 of
//! them per 300 ms call, plus a streaming `state` vector of 219729 f16 values.
//! Greedy decoding is used instead of the reference KenLM beam search because
//! KenLM has no Windows build.

pub mod decoder;
pub mod splitter;

use std::path::Path;

use half::f16;
use log::{debug, info};
use ort::session::Session;
use ort::value::Tensor;

pub use decoder::greedy_decode;
pub use splitter::LogprobSplitter;

use decoder::NUM_LABELS;

/// T-one sample rate (telephony wideband).
pub const SAMPLE_RATE: u32 = 8_000;
/// Samples per inference call: 300 ms at 8 kHz.
pub const CHUNK_SAMPLES: usize = 2_400;
/// Length of the streaming state vector.
const STATE_SIZE: usize = 219_729;
/// Seconds per acoustic frame.
const FRAME_SECONDS: f32 = 0.03;
/// Mean lookahead bias of the model, in seconds.
const MEAN_TIME_BIAS: f32 = 0.33;
/// Padding the reference pipeline adds on both sides (300 ms).
const PADDING_SAMPLES: usize = 2_400;

/// A decoded phrase with word-level-ish timings.
#[derive(Debug, Clone, PartialEq)]
pub struct TextPhrase {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
}

fn secs(frame: usize) -> f32 {
    (frame as f32 * FRAME_SECONDS - MEAN_TIME_BIAS - PADDING_SAMPLES as f32 / SAMPLE_RATE as f32)
        .max(0.0)
}

/// The model is strictly fixed-shape; anything else would be an ONNX error.
pub fn check_chunk_len(len: usize) -> Result<(), String> {
    if len != CHUNK_SAMPLES {
        return Err(format!(
            "T-one chunk must be {CHUNK_SAMPLES} samples, got {len}"
        ));
    }
    Ok(())
}

/// Streaming T-one transcriber. Feed it 8 kHz mono int16 audio in
/// 300 ms chunks; it returns finalized phrases as soon as it can.
pub struct Transcriber {
    session: Session,
    state: Vec<f16>,
    splitter: LogprobSplitter,
}

impl Transcriber {
    /// Load the ONNX acoustic model. Fail fast with an actionable message.
    pub fn load(model_path: &Path) -> Result<Self, String> {
        if !model_path.exists() {
            return Err(format!(
                "T-one model not found: {}. Run scripts/fetch_model.sh or point \
                 LIHO_STT_MODEL at t-one's model.onnx",
                model_path.display()
            ));
        }
        let env = ort::init()
            .with_name("liho")
            .build()
            .map_err(|e| format!("onnxruntime env: {e}"))?;
        let session = Session::builder(&env)
            .map_err(|e| format!("onnxruntime session builder: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| format!("onnxruntime optimization level: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("load T-one model {}: {e}", model_path.display()))?;
        info!(
            "STT model loaded: {} (T-one, {SAMPLE_RATE} Hz mono)",
            model_path.display()
        );
        Ok(Self {
            session,
            state: vec![f16::ZERO; STATE_SIZE],
            splitter: LogprobSplitter::new(),
        })
    }

    /// Feed one 300 ms chunk of 8 kHz mono int16 audio.
    ///
    /// Returns the phrases finalized by this chunk (usually empty).
    pub fn forward(&mut self, chunk: &[i16], is_last: bool) -> Result<Vec<TextPhrase>, String> {
        check_chunk_len(chunk.len())?;
        let started = std::time::Instant::now();
        let signal: Vec<i32> = chunk.iter().map(|s| i32::from(*s)).collect();
        let signal = Tensor::from_array((vec![1usize, CHUNK_SAMPLES, 1], signal))
            .map_err(|e| format!("signal tensor: {e}"))?;
        let state_in = Tensor::from_array((vec![1usize, STATE_SIZE], self.state.clone()))
            .map_err(|e| format!("state tensor: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs!["signal" => signal, "state" => state_in])
            .map_err(|e| format!("T-one inference: {e}"))?;
        let values: Vec<ort::value::ValueRef<'_>> = outputs.values().collect();
        let (_, logprobs) = values[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("logprobs: {e}"))?;
        let frames = logprobs.len() / NUM_LABELS;
        let next_state = values[1]
            .try_extract_array::<half::f16>()
            .map_err(|e| format!("next state: {e}"))?;
        self.state = next_state.iter().copied().collect();
        let phrases = self.splitter.forward(logprobs, is_last);
        let out: Vec<TextPhrase> = phrases
            .into_iter()
            .map(|p| TextPhrase {
                text: greedy_decode(&p.logprobs),
                start_time: secs(p.start_frame),
                end_time: secs(p.end_frame),
            })
            .collect();
        debug!(
            "T-one: {} frames in {:?} -> {} phrase(s)",
            frames,
            started.elapsed(),
            out.len()
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "models/t-one.onnx";

    #[test]
    fn chunk_geometry_matches_the_model() {
        // 300 ms at 8 kHz.
        assert_eq!(CHUNK_SAMPLES, SAMPLE_RATE as usize * 3 / 10);
    }

    #[test]
    fn missing_model_error_is_actionable() {
        let Err(err) = Transcriber::load(Path::new("models/definitely-missing.onnx")) else {
            panic!("missing model must be an error");
        };
        assert!(err.contains("fetch_model.sh"), "unhelpful: {err}");
    }

    #[test]
    fn wrong_chunk_length_is_rejected() {
        assert!(check_chunk_len(100).is_err());
        assert!(check_chunk_len(CHUNK_SAMPLES).is_ok());
        assert!(check_chunk_len(CHUNK_SAMPLES + 1).is_err());
    }

    /// Live: real Russian speech through the real ONNX model.
    /// `cargo test live_tone -- --ignored --nocapture`
    #[test]
    #[ignore = "live STT: needs models/t-one.onnx"]
    fn live_tone_transcribes_speech() {
        let wav = std::path::Path::new("models/t-one-sample.wav");
        if !wav.exists() || !Path::new(MODEL).exists() {
            panic!("run scripts/fetch_model.sh first");
        }
        let mut reader = hound::WavReader::open(wav).unwrap();
        assert_eq!(reader.spec().sample_rate, SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        let pcm: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        let mut t = Transcriber::load(Path::new(MODEL)).unwrap();
        let chunks: Vec<Vec<i16>> = pcm
            .chunks(CHUNK_SAMPLES)
            .map(|c| {
                let mut c = c.to_vec();
                c.resize(CHUNK_SAMPLES, 0);
                c
            })
            .collect();
        let mut text = String::new();
        for (i, c) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            for p in t.forward(c, last).unwrap() {
                text.push_str(&p.text);
                text.push(' ');
            }
        }
        println!("T-ONE TEXT: {text}");
        assert!(
            text.chars().filter(|c| c.is_alphabetic()).count() >= 5,
            "got '{text}'"
        );
    }

    /// Live: pure silence must not hallucinate speech.
    #[test]
    #[ignore = "live STT: needs models/t-one.onnx"]
    fn live_tone_silence_is_silent() {
        let mut t = Transcriber::load(Path::new(MODEL)).unwrap();
        let mut text = String::new();
        for i in 0..5 {
            for p in t.forward(&vec![0i16; CHUNK_SAMPLES], i == 4).unwrap() {
                text.push_str(&p.text);
            }
        }
        println!("SILENCE TEXT: '{text}'");
        assert!(text.chars().count() < 5, "hallucinated: '{text}'");
    }
}
