//! Greedy CTC decoder for T-one (no KenLM — the beam-search decoder needs a
//! C++ library that does not build on Windows).
//!
//! CTC output is a per-frame argmax; decoding collapses repeated symbols and
//! drops the blank token, exactly like `tone.decoder.GreedyCTCDecoder`.

/// Output alphabet: 33 Russian letters + space. Index 34 is the CTC blank.
pub const LABELS: &str = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя ";
/// Number of logprob columns in the acoustic model output (incl. blank).
pub const NUM_LABELS: usize = 35;
/// Index of the CTC blank symbol.
pub const BLANK: usize = 34;

/// Decode a `[frames * NUM_LABELS]` logprob block into text.
///
/// * argmax per frame,
/// * collapse consecutive repeats,
/// * drop blanks,
/// * trim surrounding whitespace.
pub fn greedy_decode(logprobs: &[f32]) -> String {
    let mut out = String::new();
    let mut prev: Option<usize> = None;
    for frame in logprobs.chunks_exact(NUM_LABELS) {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, v) in frame.iter().enumerate() {
            if *v > best_v {
                best_v = *v;
                best = i;
            }
        }
        if best == BLANK || best >= LABELS.len() {
            // Blank resets the repeat-collapse state (standard CTC rule).
            prev = None;
            continue;
        }
        if prev == Some(best) {
            continue;
        }
        prev = Some(best);
        if let Some(c) = LABELS.chars().nth(best) {
            out.push(c);
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a logprob block from a token string; anything else gets -100.
    fn block(tokens: &[usize]) -> Vec<f32> {
        let mut v = vec![-100.0_f32; tokens.len() * NUM_LABELS];
        for (f, t) in tokens.iter().enumerate() {
            v[f * NUM_LABELS + t] = -0.01;
        }
        v
    }

    #[test]
    fn alphabet_is_russian_plus_space() {
        assert_eq!(LABELS.chars().count(), 34);
        assert!(LABELS.starts_with('а'));
        assert!(LABELS.ends_with(' '));
        assert_eq!(NUM_LABELS, LABELS.chars().count() + 1);
    }

    #[test]
    fn single_token_decodes_to_letter() {
        // 'а' is index 0.
        assert_eq!(greedy_decode(&block(&[0])), "а");
    }

    #[test]
    fn repeated_frames_collapse() {
        // 'а' 'а' 'б' 'б' 'б' -> "аб"
        assert_eq!(greedy_decode(&block(&[0, 0, 1, 1, 1])), "аб");
    }

    #[test]
    fn blanks_are_dropped_and_reset_repeats() {
        // 'а' blank 'а' -> "аа" (blank breaks the collapse run)
        assert_eq!(greedy_decode(&block(&[0, BLANK, 0])), "аа");
    }

    #[test]
    fn internal_space_survives_but_edges_are_trimmed() {
        // 'а' space(index 33) 'б' with padding spaces around
        assert_eq!(
            greedy_decode(&block(&[33, 0, 33, 1, 33])),
            "а б".trim().to_string()
        );
    }

    #[test]
    fn empty_input_gives_empty_text() {
        assert_eq!(greedy_decode(&[]), "");
    }

    #[test]
    fn realistic_phrase() {
        // "да" is д(4) а(0). Space is an explicit label (33), a blank is
        // not a word separator — same as the reference decoder.
        assert_eq!(greedy_decode(&block(&[4, 0])), "да");
        assert_eq!(greedy_decode(&block(&[4, 0, BLANK, 4, 0])), "дада");
        assert_eq!(greedy_decode(&block(&[4, 0, 33, 4, 0])), "да да");
    }
}
