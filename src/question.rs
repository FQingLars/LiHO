//! Heuristic "is this a complete interview question?" classifier.
//!
//! Works on live partial transcripts so answering can start while the
//! interviewer is still talking. Text semantics beat raw pitch-based
//! intonation detection for reliability across speakers (see module doc
//! in pipeline). Pure functions — fully unit-tested.

/// Words/bigrams that reliably OPEN interview questions in Russian.
/// Ultra-frequent bare «что/как/где/когда» are deliberately excluded:
/// mid-sentence they are almost always conjunctions («спасибо, что…»,
/// «такой, как…») and would false-positive; genuine questions with them
/// end in «?» and are caught by the punctuation rule below.
const STARTERS: &[&str] = &[
    // Imperative prompts (the most common interview openers)
    "расскажи",
    "расскажите",
    "опишите",
    "опиши",
    "описать",
    "покажите",
    "покажи",
    "показать",
    "объясните",
    "объясни",
    "объяснить",
    "перечислите",
    "перечисли",
    "назовите",
    "назови",
    // Strong interrogatives
    "какой",
    "какая",
    "какие",
    "каким",
    "каких",
    "какого",
    "почему",
    "зачем",
    "отчего",
    "сколько",
    "каков",
    "какова",
    "каковы",
    "есть ли",
    "был ли",
    "были ли",
    "умеете ли",
    "знаете ли",
    "приходилось",
    "удавалось",
    // Safe bigrams with generic heads
    "что такое",
    "что за",
    "как вы",
    "как ты",
    "как работает",
    "как работают",
    "как устроен",
    "как устроена",
    "как устроено",
    "как лучше",
    "как правильно",
    "как бы",
];

const MIN_WORDS: usize = 3;

fn normalize_tail(text: &str) -> &str {
    text.trim_end()
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// True when the transcript looks like a finished question worth answering.
pub fn is_question(text: &str) -> bool {
    let t = normalize_tail(text);
    if word_count(t) < MIN_WORDS {
        return false;
    }
    let lower = t.to_lowercase();

    if lower.ends_with('?') {
        return true;
    }

    // Question starter must open the utterance (first two words):
    // interview questions lead with the interrogative («Расскажите…»,
    // «Как вы…»), while mid-sentence «что»/«как» are usually conjunctions
    // («спасибо, что…», «такой, как…») and would false-positive.
    let ends_sentence =
        t.ends_with('.') || t.ends_with('!') || t.ends_with('…') || t.ends_with('"');
    let head_words: Vec<&str> = lower.split_whitespace().take(2).collect();
    let head = head_words.join(" ");
    let has_starter = STARTERS.iter().any(|w| find_word(&head, w));
    has_starter && ends_sentence
}

fn find_word(haystack: &str, needle: &str) -> bool {
    let mut search_from = 0usize;
    while let Some(pos) = haystack[search_from..].find(needle) {
        let abs = search_from + pos;
        let before_ok = abs == 0
            || !haystack[..abs]
                .chars()
                .next_back()
                .map(|c| c.is_alphanumeric())
                .unwrap_or(false);
        let after = abs + needle.len();
        let after_ok = after >= haystack.len()
            || !haystack[after..]
                .chars()
                .next()
                .map(|c| c.is_alphanumeric())
                .unwrap_or(false);
        if before_ok && after_ok {
            return true;
        }
        search_from = abs + needle.len();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_question_mark_always_wins() {
        assert!(is_question("Какие гарантии доставки вы использовали?"));
        assert!(is_question("стол или стул?"));
    }

    #[test]
    fn starter_with_sentence_end() {
        assert!(is_question("Расскажите о своём опыте работы с очередями."));
        assert!(is_question("Объясните, как работает ваш пайплайн."));
        assert!(is_question("Какой у вас был самый сложный проект!"));
    }

    #[test]
    fn mid_sentence_is_not_a_question() {
        assert!(!is_question(
            "Расскажите о своём опыте работы с распределёнными"
        ));
        assert!(!is_question(
            "Итак, мы обсудили архитектуру и переходим к практике"
        ));
    }

    #[test]
    fn too_short_is_rejected() {
        assert!(!is_question("Что?"));
        assert!(!is_question("Как дела."));
        assert!(!is_question(""));
    }

    #[test]
    fn statement_without_starter_rejected() {
        assert!(!is_question(
            "Спасибо, что поделились этим опытом подробнее."
        ));
        assert!(!is_question("Такой подход, как у вас, мы видели часто."));
        assert!(!is_question("Хорошо, давайте двигаться дальше."));
    }

    #[test]
    fn starter_must_be_whole_word() {
        // «Как» inside «Кактус» must not trigger.
        assert!(!is_question("У меня кактус зацвёл впервые."));
        assert!(is_question("Как вы масштабируете базу."));
    }

    #[test]
    fn english_basics_via_punct_only() {
        // English interviews rely on '?' presence; starters list is RU-first.
        assert!(is_question("Tell us about your experience with Kafka?"));
        assert!(!is_question(
            "Tell us about your experience with Kafka systems"
        ));
    }
}
