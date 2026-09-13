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

- **Run 20** FLOP-matched d=48: mixer **3.09** vs dense **3.16** vs GRU **3.08** (600 steps, lr 0.003, 3 seeds); both-matched m=8: **3.11 / 3.21 / 3.11**. At 1500 steps mixer 2.98 vs dense 2.87 vs GRU 2.85. lr 0.01 diverges for all arms at this width. JSON: `results/seqmix_flops_d48_*.json`.

## Next test

Late-training gap at 1500 steps: lr decay / lower lr after 600 vs a top‑k ceiling. Then per-FLOP at d=96. Not MaleCNS.
