#!/usr/bin/env bash
# 50M-class run: d=768, 12 layers, d_ff 1536 (hybrid 52.5M with 6 heads + local 512; transformer 57.1M), seq 2048,
# batch 4 x 32k steps (same 260M tokens as the 20M runs), lr 4e-4, warmup 2000, cosine, wd 0.1, fp32 like the 20M runs
# (bf16 autocast + 16-bit state cost 0.02 at 20M and gave no speedup; fp32 needs 14.4 GB for the hybrid, 9.4 GB for the transformer).
# usage: scripts/run_50m.sh <device> <hybrid|transformer> <seed> [steps] [linux|enwik8]
set -u
DEV=${1:-cuda:0}; ARM=${2:-hybrid}; S=${3:-0}; STEPS=${4:-32000}; CORPUS=${5:-linux}
C=${CORPORA:-$(cd "$(dirname "$0")/.." && pwd)/corpora}
cd "$(dirname "$0")/.."
if [ "$CORPUS" = linux ]; then TASK=code; FILE=$C/linux_full.bytes; TB=200000000; else TASK=wiki; FILE=$C/enwik8.bytes; TB=90000000; fi
[ -f "$FILE" ] || { echo "corpus not found: $FILE"; echo "build it with scripts/build_corpus.py (code corpora) or fetch enwik8 from mattmahoney.net, or set CORPORA=<dir>"; exit 2; }
COMMON="--task $TASK --corpus-file $FILE --train-bytes $TB --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0004 --decay 1.0 --d-model 768 --d-ff 1536 --n-layers 12 --batch 4 --seq-len 2048 --steps $STEPS --val-bytes 2000000 --save --seed $S --device $DEV --warmup 2000 --cosine --wd 0.1 --chunk 256"
case $ARM in
  hybrid)      NAME=m50_${CORPUS}_hybrid_s$S; EXTRA="--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4" ;;
  transformer) NAME=m50_${CORPUS}_tf_s$S;     EXTRA="--arms transformer --rope" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
[ -f results/$NAME.json ] && { echo "skip $NAME"; exit 0; }
echo "=== $NAME"; uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON $EXTRA 2>&1 | grep -v "^  "
