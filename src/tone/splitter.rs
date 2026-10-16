//! Port of `tone.logprob_splitter.StreamingLogprobSplitter` (Apache-2.0,
//! github.com/voicekit-team/T-one).
//!
//! The acoustic model emits frame-level logprobs every 300 ms. This splitter
//! decides where a *final* phrase ends (N consecutive non-speech frames) and
//! returns those phrases ready for decoding; the unfinished tail is carried in
//! the state for the next chunk.
//!
//! Deliberate deviation from the Python original: the upstream code asserts
//! that the stream starts with silence and crashes otherwise. Here an empty
//! silence list simply yields no phrase — the listener never dies.

use super::decoder::NUM_LABELS;

/// One finalized phrase: raw logprobs plus its frame range (for timestamps).
#[derive(Debug, Clone, PartialEq)]
pub struct Phrase {
    pub logprobs: Vec<f32>,
    pub start_frame: usize,
    pub end_frame: usize,
}

/// Streaming state: logprobs not yet consumed + global frame offset.
#[derive(Debug, Clone, Default)]
pub struct SplitterState {
    past: Vec<f32>,
    offset: usize,
}

/// Splits a logprob stream into phrases on silences.
pub struct LogprobSplitter {
    /// Sum of P(space) + P(blank) above which a frame counts as silence.
    silence_threshold: f32,
    /// Consecutive non-speech frames that close a phrase.
    min_silence: usize,
    /// Extra frames kept on both sides of a phrase (decoder warm-up).
    speech_expand: usize,
    /// Force-split a phrase longer than this many frames.
    max_phrase: usize,
    state: SplitterState,
}

impl Default for LogprobSplitter {
    fn default() -> Self {
        Self {
            silence_threshold: 0.9,
            min_silence: 20,
            speech_expand: 3,
            max_phrase: 2000,
            state: SplitterState::default(),
        }
    }
}

impl LogprobSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume one chunk of `[frames * NUM_LABELS]` logprobs.
    ///
    /// `is_last` forces the trailing silence so the final phrase is emitted
    /// without waiting for more audio.
    pub fn forward(&mut self, logprobs: &[f32], is_last: bool) -> Vec<Phrase> {
        let expand = self.speech_expand;
        let mut buf = std::mem::take(&mut self.state.past);
        buf.extend_from_slice(logprobs);
        let frames = buf.len() / NUM_LABELS;

        // P(space) + P(blank) — both live in the last two columns.
        let mut is_speech = vec![false; frames];
        for f in 0..frames {
            let row = &buf[f * NUM_LABELS..(f + 1) * NUM_LABELS];
            let ns = row[NUM_LABELS - 2].exp() + row[NUM_LABELS - 1].exp();
            is_speech[f] = ns <= self.silence_threshold;
        }

        let bounds = self.iterate_over_phrases(&is_speech, is_last);
        let mut phrases = Vec::with_capacity(bounds.len());
        let mut last_phrase = 0usize;
        for (start, end) in bounds {
            let from = start.saturating_sub(expand) * NUM_LABELS;
            let to = ((end + expand) * NUM_LABELS).min(buf.len());
            phrases.push(Phrase {
                logprobs: buf[from..to].to_vec(),
                start_frame: start + self.state.offset,
                end_frame: end + self.state.offset,
            });
            last_phrase = end.max(start);
        }

        // Keep the unfinished tail (plus warm-up context) for the next chunk.
        if !is_speech[last_phrase.min(frames)..].iter().any(|s| *s) {
            last_phrase = last_phrase.max(frames.saturating_sub(expand));
        }
        self.state.offset += last_phrase;
        self.state.past = buf[last_phrase.min(frames) * NUM_LABELS..].to_vec();
        phrases
    }

    /// Silence runs -> speech runs -> phrase boundaries, mirroring
    /// `StreamingLogprobSplitter._iterate_over_phrases`.
    ///
    /// The frame buffer is padded with `MIN_SILENCE` *non-speech* frames on
    /// both sides (the right pad only when flushing), which guarantees that
    /// every phrase has a closing boundary. Silence runs shorter than
    /// `MIN_SILENCE` do not separate phrases, and the last speech run is not
    /// emitted because the speaker may still continue.
    ///
    /// The Python original asserts that the stream starts with silence and
    /// crashes on an all-speech buffer; here such input simply yields nothing.
    fn iterate_over_phrases(&self, is_speech: &[bool], is_last: bool) -> Vec<(usize, usize)> {
        let min_sil = self.min_silence as i64;
        let speech_len = is_speech.len() as i64;

        // np.pad(~is_speech, (1, 1)) pads a bool array with False, i.e. with
        // "speech" — which is exactly why the first recorded boundary is a
        // silence *start* at -min_sil.
        let mut sil: Vec<bool> = Vec::with_capacity(is_speech.len() + 2 * self.min_silence + 2);
        sil.push(false);
        sil.extend(std::iter::repeat_n(true, self.min_silence)); // padded head
        sil.extend(is_speech.iter().map(|s| !s));
        if is_last {
            sil.extend(std::iter::repeat_n(true, self.min_silence)); // padded tail
        }
        sil.push(false);

        let mut starts: Vec<i64> = Vec::new();
        let mut ends: Vec<i64> = Vec::new();
        for k in 0..sil.len() - 1 {
            let d = i64::from(sil[k + 1]) - i64::from(sil[k]);
            let at = k as i64 - min_sil;
            if d == 1 {
                starts.push(at);
            } else if d == -1 {
                ends.push(at);
            }
        }

        // numpy pairs ends[i] with starts[i] and drops short runs.
        let keep: Vec<bool> = starts
            .iter()
            .zip(ends.iter())
            .map(|(s, e)| e - s >= min_sil)
            .collect();
        let kept_starts: Vec<i64> = starts
            .iter()
            .zip(keep.iter())
            .filter(|(_, k)| **k)
            .map(|(s, _)| *s)
            .collect();
        let speech_starts: Vec<i64> = ends
            .iter()
            .zip(keep.iter())
            .filter(|(_, k)| **k)
            .map(|(e, _)| *e)
            .collect();
        // The final silence start has no matching end: it bounds the last
        // (still open) speech run.
        let mut speech_ends: Vec<i64> = kept_starts[1..].to_vec();
        speech_ends.push(speech_len);

        let max_phrase = self.max_phrase as i64;
        let mut out = Vec::new();
        for i in 0..speech_starts.len().min(speech_ends.len()) {
            let mut start = speech_starts[i];
            let end = speech_ends[i];
            while end - start >= max_phrase {
                out.push((start as usize, (start + max_phrase) as usize));
                start += max_phrase;
            }
            if i + 1 < speech_ends.len() {
                out.push((start.max(0) as usize, end.max(0) as usize));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One frame of logprobs: `speech` decides P(space)+P(blank).
    fn frame(speech: bool) -> [f32; NUM_LABELS] {
        let mut f = [-20.0_f32; NUM_LABELS];
        if speech {
            f[0] = -0.01; // a letter wins
        } else {
            f[NUM_LABELS - 1] = -0.001; // blank ~= 1.0
            f[NUM_LABELS - 2] = -6.0; // space ~= 0.0025
        }
        f
    }

    fn frames(pattern: &[bool]) -> Vec<f32> {
        pattern.iter().flat_map(|s| frame(*s)).collect()
    }

    fn speech(n: usize) -> Vec<bool> {
        vec![true; n]
    }
    fn silence(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    /// The real stream opens with silence (the reference pipeline also pads
    /// 300 ms), so every test starts there.
    fn primed() -> LogprobSplitter {
        let mut s = LogprobSplitter::new();
        assert!(s.forward(&frames(&silence(30)), false).is_empty());
        s
    }

    #[test]
    fn silence_at_start_then_speech_produces_no_phrase() {
        let mut s = LogprobSplitter::new();
        assert!(s.forward(&frames(&silence(100)), false).is_empty());
    }

    #[test]
    fn speech_followed_by_long_silence_yields_one_phrase() {
        let mut s = primed();
        assert!(
            s.forward(&frames(&speech(60)), false).is_empty(),
            "unfinished speech must wait for silence"
        );

        let mut chunk = silence(40);
        chunk.extend(speech(10));
        let phrases = s.forward(&frames(&chunk), false);
        assert_eq!(phrases.len(), 1, "got {phrases:?}");
        assert!(phrases[0].end_frame > phrases[0].start_frame);
        assert!(!phrases[0].logprobs.is_empty());
    }

    #[test]
    fn short_gap_is_not_a_boundary() {
        let mut s = primed();
        s.forward(&frames(&speech(50)), false);
        // 5 silent frames (< MIN_SILENCE) then more speech.
        assert!(s.forward(&frames(&silence(5)), false).is_empty());
        assert!(
            s.forward(&frames(&speech(50)), false).is_empty(),
            "still one open phrase"
        );
    }

    #[test]
    fn is_last_flushes_open_speech() {
        let mut s = primed();
        s.forward(&frames(&speech(40)), false);
        let phrases = s.forward(&frames(&speech(40)), true);
        assert_eq!(phrases.len(), 1, "got {phrases:?}");
    }

    #[test]
    fn frame_offsets_grow_monotonically() {
        let mut s = primed();
        assert!(s.forward(&frames(&speech(40)), false).is_empty());
        let second = s.forward(&frames(&silence(40)), false);
        assert_eq!(second.len(), 1, "the pause closes the phrase");
        let b = second[0].end_frame;
        let third = s.forward(&frames(&speech(40)), false);
        assert!(third.is_empty());
        let fourth = s.forward(&frames(&silence(40)), false);
        assert!(
            fourth[0].end_frame > b,
            "global offsets must advance: {b} -> {}",
            fourth[0].end_frame
        );
    }

    #[test]
    fn two_phrases_from_one_chunk() {
        let mut s = primed();
        let mut pattern = speech(40);
        pattern.extend(silence(40));
        pattern.extend(speech(40));
        pattern.extend(silence(40));
        let phrases = s.forward(&frames(&pattern), false);
        assert_eq!(phrases.len(), 2, "two closed phrases: {phrases:?}");
        assert!(phrases[1].start_frame > phrases[0].end_frame);
    }

    #[test]
    fn state_does_not_grow_without_bound() {
        let mut s = LogprobSplitter::new();
        for _ in 0..200 {
            s.forward(&frames(&silence(10)), false);
            assert!(
                s.state.past.len() <= 64 * NUM_LABELS,
                "tail grew to {} frames",
                s.state.past.len() / NUM_LABELS
            );
        }
    }

    #[test]
    fn very_long_speech_is_force_split() {
        let mut s = primed();
        let phrases = s.forward(&frames(&speech(5000)), false);
        assert!(
            phrases.len() >= 2,
            "2000-frame cap must force a split, got {}",
            phrases.len()
        );
    }

    #[test]
    fn all_speech_chunk_without_is_last_is_safe() {
        // Upstream Python asserts here (IndexError) — we must not panic.
        let mut s = LogprobSplitter::new();
        assert!(s.forward(&frames(&speech(50)), false).is_empty());
    }

    #[test]
    fn empty_chunk_is_safe() {
        let mut s = LogprobSplitter::new();
        assert!(s.forward(&[], false).is_empty());
    }
}

#[cfg(test)]
mod reference_parity {
    //! Expected values produced by running the ORIGINAL Python implementation
    //! (`tone.logprob_splitter.StreamingLogprobSplitter`, commit 106f3b0) on
    //! the same synthetic logprobs. See scripts/reference_parity.py.
    use super::*;

    fn frame(speech: bool) -> [f32; NUM_LABELS] {
        let mut f = [-20.0_f32; NUM_LABELS];
        if speech {
            f[0] = -0.01;
        } else {
            f[NUM_LABELS - 1] = -0.001;
            f[NUM_LABELS - 2] = -6.0;
        }
        f
    }

    fn frames(pattern: &[bool]) -> Vec<f32> {
        pattern.iter().flat_map(|s| frame(*s)).collect()
    }

    fn run(chunks: &[Vec<bool>]) -> Vec<Vec<(usize, usize)>> {
        let mut s = LogprobSplitter::new();
        chunks
            .iter()
            .enumerate()
            .map(|(i, c)| {
                s.forward(&frames(c), i + 1 == chunks.len())
                    .iter()
                    .map(|p| (p.start_frame, p.end_frame))
                    .collect()
            })
            .collect()
    }

    fn sp(n: usize) -> Vec<bool> {
        vec![true; n]
    }
    fn si(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    #[test]
    fn parity_silence_only() {
        assert_eq!(run(&[si(60)]), vec![Vec::<(usize, usize)>::new()]);
    }

    #[test]
    fn parity_speech_then_silence_without_leading_silence() {
        let mut chunk = si(40);
        chunk.extend(sp(10));
        let out = run(&[sp(60), chunk]);
        // Reference: [(0, 60), (100, 110)] — note the trailing open run is
        // emitted here because the buffer ends inside a silence-padded chunk.
        assert_eq!(out[1], vec![(0, 60), (100, 110)]);
    }

    #[test]
    fn parity_primed_speech_then_silence() {
        let mut chunk = si(40);
        chunk.extend(sp(10));
        let out = run(&[si(30), sp(40), chunk]);
        // Reference: [(30, 70), (110, 120)] with global offsets applied.
        assert_eq!(out[2], vec![(30, 70), (110, 120)]);
    }

    #[test]
    fn parity_two_phrases_in_one_chunk() {
        let mut chunk = sp(40);
        chunk.extend(si(40));
        chunk.extend(sp(40));
        chunk.extend(si(40));
        let out = run(&[si(30), chunk]);
        assert_eq!(out[1], vec![(30, 70), (110, 150)]);
    }

    #[test]
    fn parity_is_last_flush() {
        let out = run(&[si(30), sp(40), sp(40)]);
        assert_eq!(out[2], vec![(30, 110)]);
    }

    #[test]
    fn parity_short_gap_is_not_a_boundary() {
        let out = run(&[si(30), sp(40), si(5), sp(40)]);
        assert!(out[1].is_empty() && out[2].is_empty(), "no early split");
        assert_eq!(out[3], vec![(30, 115)]);
    }

    #[test]
    fn parity_very_long_speech_is_force_split() {
        let out = run(&[si(30), sp(5000)]);
        assert_eq!(out[1], vec![(30, 2030), (2030, 4030), (4030, 5030)]);
    }

    #[test]
    fn parity_realistic_stream() {
        let out = run(&[si(30), sp(40), si(40), sp(40), si(40), sp(40), si(40)]);
        assert_eq!(out[2], vec![(30, 70)]);
        assert_eq!(out[4], vec![(110, 150)]);
        assert_eq!(out[6], vec![(190, 230)]);
    }
}
