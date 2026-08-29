#!/bin/bash
# Lancement MANUEL, sur le port 8090 pour ne jamais entrer en collision avec le
# service systemd qui occupe le 8089. Memes flags que le service. Ctrl+C pour arreter.
exec whisper-server \
  -m "$HOME/.local/share/whisper-cpp/ggml-medium-q5_0.bin" \
  -l fr \
  --host 127.0.0.1 --port 8090 \
  --request-path /v1 --inference-path /audio/transcriptions \
  -ml 100000 \
  --vad -vm "$HOME/.local/share/whisper-cpp/ggml-silero-v5.1.2.bin"
