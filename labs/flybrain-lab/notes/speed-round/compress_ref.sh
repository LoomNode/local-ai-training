#!/usr/bin/env bash
# Reference for the speed round: what do one-pass, no-gradient learners (classical compressors) get on the lab's corpus?
# Marginal bits/byte of a held-out region given the training bytes: 8 * (size(train + region) - size(train)) / len(region).
# The compressor keeps adapting inside the region, so this is not the model's cold-window protocol: read it as a reference only.
set -u -o pipefail
D=/tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/compress-ref
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
N=2097152   # 1,024 windows of 2,048 bytes
cd "$D"
head -c 200000000 "$C" > train.bin
tail -c +202117358 "$C" | head -c $N > dev.bin
tail -c +211576469 "$C" | head -c $N > test.bin
cat train.bin dev.bin > train_dev.bin
cat train.bin test.bin > train_test.bin
ls -l train.bin dev.bin test.bin train_dev.bin train_test.bin | awk '{print $5, $9}'
size() {  # size <tool> <file>: compressed bytes on stdout
  case $1 in
    xz)     xz -9e --lzma2=preset=9e,dict=512MiB -T1 -c "$2" | wc -c ;;
    zstd)   zstd --ultra -22 --long=31 -T1 -q -c "$2" | wc -c ;;
    ppmd6)  rm -f o.7z; 7z a -bso0 -bsp0 -m0=PPMd:o=6:mem=3900m o.7z "$2" >/dev/null && stat -c %s o.7z ;;
    ppmd8)  rm -f o.7z; 7z a -bso0 -bsp0 -m0=PPMd:o=8:mem=3900m o.7z "$2" >/dev/null && stat -c %s o.7z ;;
    ppmd16) rm -f o.7z; 7z a -bso0 -bsp0 -m0=PPMd:o=16:mem=3900m o.7z "$2" >/dev/null && stat -c %s o.7z ;;
  esac
}
echo "| tool | train bytes | seconds for train | dev bits/byte | test bits/byte | region alone: dev | region alone: test |"
echo "|---|---:|---:|---:|---:|---:|---:|"
for t in ppmd6 ppmd8 ppmd16 xz zstd; do
  t0=$(date +%s)
  a=$(size $t train.bin)
  t1=$(date +%s)
  b=$(size $t train_dev.bin)
  c=$(size $t train_test.bin)
  d=$(size $t dev.bin)
  e=$(size $t test.bin)
  python3 -c "
a,b,c,d,e,n=$a,$b,$c,$d,$e,$N
print(f'| $t | {a:,} | $((t1 - t0)) | {8*(b-a)/n:.3f} | {8*(c-a)/n:.3f} | {8*d/n:.3f} | {8*e/n:.3f} |')"
done
echo "done $(date +%T)"
