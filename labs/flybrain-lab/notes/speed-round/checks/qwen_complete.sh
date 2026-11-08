#!/usr/bin/env bash
# Qwen3.8-27B on the completion test's spots (qwen_complete.py): starts a llama-server on one card for the run (127.0.0.1:8099, the
# lab's CUDA build of llama.cpp with its CUDA 12 runtime; NOT the model server: no public port, stopped when the run ends), waits for
# it, runs the client, stops it. About a minute per ten spots on a 3090 (the prefix is up to 6 KB).
# usage: CUDA_VISIBLE_DEVICES=<uuid> qwen_complete.sh <heldout.bytes> <out.json> [client args: --spots 60 --ext gd --seed 0 ...]
#        NGL=0 for a CPU smoke test (slow: a minute a spot).
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID (or NGL=0 for the CPU)}"
IN=${1:?usage: qwen_complete.sh <heldout.bytes> <out.json> [client args]}; OUT=${2:?}; shift 2
MODEL=${MODEL:-/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf}
LL=/games/ailab/lab/llama.cpp-qwen4exp; PORT=${PORT:-8099}; NGL=${NGL:-99}
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH=$LL/.deps/cuda-12.8.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"$LL/build-cuda/bin/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c 4096 -b 2048 -ngl "$NGL" -t 8 --log-disable > "${OUT%.json}.server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 240); do curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; kill -0 $SRV 2>/dev/null || { echo "llama-server died: see ${OUT%.json}.server.log"; exit 1; }; sleep 2; done
curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "llama-server not up after 8 minutes"; exit 1; }
/games/ailab/lab/flybrain-lab/.venv/bin/python "$S/qwen_complete.py" --stream "$IN" --url "http://127.0.0.1:$PORT" --out "$OUT" "$@"
