# Resume here

Lab: `/games/ailab/lab/flybrain-lab`  
Notebook: [`../notebook.md`](../notebook.md)

## Do not

- Load FlyWire / MaleCNS.
- Plant class claws / shared pattern-graph RNG.
- Call the coincidence mixer a better LM or a better bag classifier than MLP.
- Call the recurrent coincidence block (Run 19) better than the dense FFN or GRU; it is only cheaper per step.

## Current finding (2026-09-12)

- Run 17 fly graph unique-AND: fly 0.22 > shuffle 0.19, GRU ~0.24.
- **Run 18** stolen mixer (no connectome): 12-d bag mixer≈MLP≈GRU ~0.25; 48-d bag mixer **0.55** vs MLP **0.86** vs GRU **0.73**. Mixer ≫ chance, MLP wins. JSON: `results/mixer_bind_s{0,1}.json`, `results/mixer_bind_highd_s{0,1}.json`.

- **Run 19** coincidence block as a recurrent byte LM (`flybrain/model/seqmixer.py`): 600-step WikiText means mixer **3.36** / dense FFN **3.28** / GRU **3.16** (unigram 4.564, bigram 3.729). FLOPs/step 17.9k vs 27.2k vs 24.1k. Top‑k off is unstable; wider D, more claws, θ=0 do not help. JSON: `results/seqmix_wiki_s{0,1,2}{,_600}.json`.

- **Run 20** FLOP-matched d=48: mixer **3.09** vs dense **3.16** vs GRU **3.08** (600 steps, lr 0.003, 3 seeds); both-matched m=8: **3.11 / 3.21 / 3.11**. 11-seed both-matched: mixer 3.127 vs dense 3.168 (paired −0.041, se 0.020, 9/11) vs GRU 3.110. 5-seed 1500 steps: mixer 2.99 vs dense 2.95 vs GRU 2.92 (mixer behind). lr 0.01 diverges for all arms at this width. JSON: `results/seqmix_flops_d48_*.json`.

- **Run 21** fast weights on the sparse code (`flybrain/model/fastweight.py`): in-context copy 63–73 % at 600 steps, **91 %** at 1500; sparse-no-FW, dense+FW (D=74 FLOP-matched and D=1024 at 6.5× FLOPs) and FLOP-matched GRU all at chance (0.3–0.5 %; dense D=74 up to 27 %). WikiText unchanged (~3.1). JSON: `results/fw_copy_*.json`, `results/fw_wiki_s*.json`.

## Next test

Fast-weight line: modulator ablation, top‑k sweep on the key, adaptation-speed on a shifted corpus, tiny attention ceiling. Not MaleCNS.
