#!/bin/bash
# Enregistre ta voix puis l'envoie au serveur. Usage : ./dictee.sh [duree_en_s]
SECS=${1:-5}
OUT=$(mktemp /tmp/dictee-XXXX.wav)
echo "Parle pendant ${SECS}s..."
arecord -q -f S16_LE -c1 -r16000 -d "$SECS" "$OUT" || exit 1
echo "Transcription..."
START=$(date +%s%N)
RESP=$(curl -s -X POST http://127.0.0.1:8089/v1/audio/transcriptions \
        -F "file=@$OUT" -F "language=fr" -F "temperature=0.0")
END=$(date +%s%N)
echo "--- brut (les \\n sont le piege a observer) ---"
echo "$RESP" | python -c "import json,sys; print(repr(json.load(sys.stdin)['text']))"
echo "--- lisible ---"
echo "$RESP" | python -c "import json,sys; print(json.load(sys.stdin)['text'].strip())"
echo "--- latence : $(( (END-START)/1000000 )) ms  (baseline int8 CPU : ~3940 ms) ---"
rm -f "$OUT"
