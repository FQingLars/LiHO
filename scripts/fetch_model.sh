#!/usr/bin/env bash
# Downloads the T-one streaming ASR model (ONNX) and the speech test fixture.
#
# T-one (t-tech/T-one, Apache-2.0) is a 71M-parameter streaming CTC model for
# Russian; only the ONNX acoustic model is needed — the decoder is greedy and
# lives in src/tone/decoder.rs.
#
# Usage: ./fetch_model.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

MODEL="models/t-one.onnx"
# The reference example audio, converted to the model's 8 kHz mono s16 input.
SAMPLE="models/t-one-sample.wav"

if [ ! -f "$MODEL" ]; then
  echo "Downloading T-one model.onnx (~138 MB)..."
  curl -L --fail --silent --show-error \
    -o "$MODEL.part" \
    "https://huggingface.co/t-tech/T-one/resolve/main/model.onnx"
  mv "$MODEL.part" "$MODEL"
fi

if [ ! -f "$SAMPLE" ]; then
  echo "Downloading the example audio fixture..."
  curl -L --fail --silent --show-error \
    -o "$SAMPLE.flac" \
    "https://raw.githubusercontent.com/voicekit-team/T-one/main/tone/demo/audio_examples/audio_short.flac"
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -i "$SAMPLE.flac" -ar 8000 -ac 1 -c:a pcm_s16le "$SAMPLE"
    rm -f "$SAMPLE.flac"
  else
    echo "WARNING: ffmpeg not found — kept $SAMPLE.flac."
    echo "         The live STT test needs ${SAMPLE} (8 kHz mono s16 WAV)."
  fi
fi

ls -la models/