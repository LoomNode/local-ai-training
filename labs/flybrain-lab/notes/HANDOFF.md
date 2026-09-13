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

- **Run 22** GLA baseline (learned key+query, elu+1 or identity, D=59/256/1024, lr 0.003/0.001/0.0003) at chance on copy; frozen Gaussian keys 27 % matched, chance wide; sparse k=8 **84 %**, cliff at k≥128; constant gate 1.0 = learned gate; keys-from-embedding does not rescue dense. JSON: `results/fw_copy_gla*.json`, `fw_copy_gauss*.json`, `fw_copy_topk*_s0.json`, `fw_copy_constgate*.json`.

- **Run 23** parallel form (`flybrain/model/sparsela.py`, GPU). Copy@64 lr 0.001: sparse >90 % by step 250, dense Gaussian 500, softmax 750, learned 750–1500, all end ~92 % (earlier 'dense at chance' was lr 0.003). Copy@512: only sparse top‑k learns (57–58 %, one-token-key ambiguity plateau); softmax/GLA/GRU at chance even at 10k steps. WikiText@512: GRU 2.59 ≪ all attention-style arms 3.24–3.48. Claws, nonnegativity, DA gate, delta rule all unnecessary. JSON: `results/sla_*.json`, `results/fw_copy_{gausstopk,signed,delta}*.json`.

- **Run 24** normalizer + bigram key: sparse top‑4 copies at 512 at **94 / 92 %** (Gaussian top‑k 91 / 93); learned, softmax, dense Gaussian, GRU all at chance with the same fixes. Copy@64: sparse >90 % by step 250, softmax 99 % by 1000, learned never. WikiText@512 fair: GRU **2.59** < learned 2.89 < softmax 2.93 < gauss 2.97 < sparse **3.16**. Sparse = exact retrieval, not text. JSON: `results/sla_copy512_fair_*.json`, `sla_copy64_fair_s0.json`, `sla_wiki512_fair_*.json`.

- **Run 25** six follow-ups at 4 blocks / d=128 / 9 MB: mixed (learned+sparse head) best fast-weight arm on wiki 2.335 and code 1.832, wins copyable bytes (wiki 1.43 vs learned 2.17); GRU still best overall (2.08 / 1.60). Capacity: with ≥6–12 % scored positions sparse learns at 1024 and 2048 (4096 units: 93–94.5 % up to 32 spans / 16 spans); 1024 units degrade gracefully (88→81 %) and hit cross-talk at 2048 (45 %). Softmax learns only at ~50 % signal. Cross-corpus: no adaptation. tied ≈ learned. Terminal corpus uninformative. JSON: `results/sla_big_*`, `sla_needle*`, `sla_cap1024_*`, `cross_wiki_to_code.json`; ckpts in `results/ckpt/`.

- **Run 26** GRU + sparse memory head (`flybrain/model/hybrid.py`): code 1.463/1.477 vs GRU 1.601/1.603, wide GRU 1.545/1.558, learned head 1.518/1.544; wiki 2.009 vs 2.079 / 2.038 / 2.038. Repeat bytes 0.5 vs 0.8. First bpb win over the best baseline. JSON: `results/hyb_*.json`.

## Next test

Hybrid at scale: d=256 / 50 MB, seq 2048, second wiki seed, and the same head on a transformer backbone. Not MaleCNS.
