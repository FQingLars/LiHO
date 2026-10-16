"""Reference oracle for the Rust port of the T-one logprob splitter.

Runs the ORIGINAL Python implementation from
<https://github.com/voicekit-team/T-one> (`tone/logprob_splitter.py`,
commit 106f3b0) over synthetic logprobs and prints the phrase frame ranges it
produces. Those numbers are hard-coded in
`src/tone/splitter.rs::reference_parity` — run this script after touching the
splitter and update the tests if the numbers changed.

    python3 scripts/reference_parity.py          # needs numpy
"""

import numpy as np

MIN_SIL = 20
EXPAND = 3
MAX_PHRASE = 2000
THRESH = 0.9
NUM_LABELS = 35


class Splitter:
    """Verbatim port of StreamingLogprobSplitter.forward()."""

    def __init__(self):
        self.past = np.zeros((0, NUM_LABELS), dtype=np.float32)
        self.offset = 0

    def _iterate_over_phrases(self, is_speech, is_last=False):
        speech_len = len(is_speech)
        is_speech = np.pad(is_speech, (MIN_SIL, MIN_SIL if is_last else 0))
        changes = np.diff(np.pad(~is_speech, (1, 1)).astype(np.int32))
        starts = (changes == 1).nonzero()[0] - MIN_SIL
        ends = (changes == -1).nonzero()[0] - MIN_SIL
        mask = (ends - starts) >= MIN_SIL
        starts, ends = starts[mask], ends[mask]
        speech_starts = ends.tolist()
        speech_ends = starts.tolist()[1:] + [speech_len]
        out = []
        for i, (start, end) in enumerate(zip(speech_starts, speech_ends)):
            while end - start >= MAX_PHRASE:
                out.append((start, start + MAX_PHRASE))
                start += MAX_PHRASE
            if i < len(speech_ends) - 1:
                out.append((start, end))
        return out

    def forward(self, logprobs, is_last=False):
        logprobs = np.concatenate((self.past, logprobs), axis=-2)
        is_speech = np.exp(logprobs[..., -2:]).sum(axis=-1) <= THRESH
        phrases = []
        last_phrase = 0
        for start, end in self._iterate_over_phrases(is_speech, is_last):
            phrases.append((start + self.offset, end + self.offset))
            last_phrase = end
        if not len(np.nonzero(is_speech[last_phrase:])[0]):
            last_phrase = max(last_phrase, len(logprobs) - EXPAND)
        self.offset += last_phrase
        self.past = logprobs[last_phrase:]
        return phrases


def frames(pattern):
    """Frames where `speech=True` put a letter on top (P(blank)+P(space) ~ 0)."""
    out = np.full((len(pattern), NUM_LABELS), -20.0, dtype=np.float32)
    for i, speech in enumerate(pattern):
        if speech:
            out[i, 0] = -0.01
        else:
            out[i, NUM_LABELS - 1] = -0.001  # blank ~ 1.0
            out[i, NUM_LABELS - 2] = -6.0     # space ~ 0.0025
    return out


def sp(n):
    return [True] * n


def si(n):
    return [False] * n


def run(name, chunks):
    splitter = Splitter()
    print(f"--- {name}")
    for i, chunk in enumerate(chunks):
        phrases = splitter.forward(frames(chunk), is_last=i == len(chunks) - 1)
        print(f"  chunk {i} len={len(chunk)} -> {phrases}")


if __name__ == "__main__":
    run("silence only", [si(60)])
    run("speech then silence, no leading silence", [sp(60), si(40) + sp(10)])
    run("primed: speech then silence", [si(30), sp(40), si(40) + sp(10)])
    run("primed: two phrases in one chunk", [si(30), sp(40) + si(40) + sp(40) + si(40)])
    run("primed: is_last flush", [si(30), sp(40), sp(40)])
    run("primed: short gap", [si(30), sp(40), si(5), sp(40)])
    run("primed: very long speech", [si(30), sp(5000)])
    run("realistic stream", [si(30), sp(40), si(40), sp(40), si(40), sp(40), si(40)])