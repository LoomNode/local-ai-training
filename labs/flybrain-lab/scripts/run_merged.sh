#!/usr/bin/env bash
# One table (docs/superpowers/specs/2026-11-05-merged-table-design.md): the Run 41 20M winner (depth4local) and three variants.
#   base       : the winner as is
#   ngram      : + a separate Qwen-style hashed bigram/trigram embedding, 8192 rows per order, after layer 2   (~8.65M extra)
#   prior      : every memory head's table starts from a learned prior                                         (~8.40M extra)
#   prioraddr  : prior + each head's address shifted by a hash of the last 2 or 3 bytes (heads alternate)
# usage: scripts/run_merged.sh <device> <base|ngram|prior|prioraddr> <seed> [steps] [linux|enwik8]
set -u
DEV=${1:-cuda:0}; ARM=${2:-base}; S=${3:-0}; STEPS=${4:-4000}; CORPUS=${5:-linux}
C=${CORPORA:-$(cd "$(dirname "$0")/.." && pwd)/corpora}
cd "$(dirname "$0")/.."
if [ "$CORPUS" = linux ]; then TASK=code; FILE=$C/linux_full.bytes; TB=200000000; else TASK=wiki; FILE=$C/enwik8.bytes; TB=90000000; fi
[ -f "$FILE" ] || { echo "corpus not found: $FILE (set CORPORA=<dir>)"; exit 2; }
COMMON="--task $TASK --corpus-file $FILE --train-bytes $TB --normalize --key-conv 2 --top-k 4 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --steps $STEPS --val-bytes 2000000 --save --seed $S --device $DEV --warmup 1000 --cosine --wd 0.1 --chunk 256 --arms mingru_sparse_post --gate --amp --n-key 4096 --mem-every 2 --local-attn 512 --key-convs 2,4"
case $ARM in
  base)      EXTRA="" ;;
  ngram)     EXTRA="--ngram-embed 8192" ;;
  prior)     EXTRA="--mem-prior" ;;
  prioraddr) EXTRA="--mem-prior --ngram-addr 2,3" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
NAME=mt_${CORPUS}_${ARM}_${STEPS}_s$S
[ -f results/$NAME.json ] && { echo "skip $NAME"; exit 0; }
echo "=== $NAME"
uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON $EXTRA 2>&1 | grep -v "^  "
