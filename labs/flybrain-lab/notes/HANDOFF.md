# Resume here

Lab: `/games/ailab/lab/flybrain-lab`  
Summary of findings: [`RESULTS.md`](RESULTS.md)  
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

- **Run 27** confirmation: 3 seeds/corpus, wide GRU, lr sweep, 9k steps, leakage control (collapses to baseline), held-out prose (2.008 vs 2.071) and held-out mixed code (1.52/1.54 vs GRU 1.76/1.76, wide GRU 1.66/1.68). Run 26 stands. JSON: `results/conf_*.json`.

- **Run 28** backbones: minGRU (chunked fp64 scan) and gated linear attention (data-dependent decay) + sparse head: code −0.11…−0.24, prose −0.02…−0.07 on both seeds and both corpora; learned-key head a net loss on prose. Arms `mingru`, `gla`, `<bb>_<key>_<pre|post>`. JSON: `results/bb_*.json`.

- **Run 29** width ladder d=64/128/256/384 (`scripts/run_ladder.sh`, `scripts/ladder_summary.py`): minGRU gap stable (code 0.15→0.12, held-out 0.32→0.17, prose 0.04); GLA gap closes on code at d≥256 (learns its own recall), stays on prose; plain GLA seed-unstable at d=384. JSON: `results/ladder_*.json`.

- **Run 30** transformer control (`TransformerBackbone`, arms `transformer`, `transformer_<key>_<pre|post>`): at 3000 steps undertrained and unstable; at 9000 steps lr 0.001 it forms retrieval (code 1.41) and the head *hurts* by 0.1–0.3; at lr 3e-4 it never forms retrieval and the head helps ~0.1. JSON: `results/bb_transformer_*.json`.

- **Run 31** mask sweep (`scripts/mask_sweep.py`, needs `--save` checkpoints): head gains 0.3–0.4 on every matched bin down to 2-grams, costs 0.06 on unmatched bytes (54 % of code). JSON: `results/mask_*_masksweep.json`.

- **Run 32** equal budget 9000 steps, 3 seeds, code: GRU+head 1.378 < transformer 1.409 ≈ minGRU+head 1.41 < plain GLA 1.43–1.52. JSON: `results/eq9k_code_s{1,2}.json`.

- **Run 33** match-strength gate (`--gate`): 1.481 → 1.435 on code d=128; improves every matched bin, barely touches the unmatched cost. JSON: `results/gate_*.json`.

- **Run 34** scale (`scripts/run_scale.sh <device> <seed>`; corpora built by `scripts/build_corpus.py` from CPython 3.12.6, enwik8, Linux 6.10 into the session scratchpad — rebuild before reruns): d=256 / seq 2048 / 4096-unit head / 8000 steps, 2 seeds. CPython: minGRU 1.48, +head 1.16, +gated 1.11, transformer 1.22. enwik8: 1.85 / 1.68 / 1.63 / 1.84. Held-out kernel: 3.43 / 2.14 / 2.03 / 2.01. Ckpts in `results/ckpt/scale_*`. Cross-eval: `scripts/cross_eval.py --n-batches 16`.

- **Run 35** tuned recipe (`scripts/run_tuned.sh`; `--warmup --cosine --wd --rope`): transformer gains 0.2, recurrences 0.03. CPython: transformer 1.010 < head 1.084; enwik8: head 1.602 ≈ transformer 1.614. Run 34's "beats transformer" withdrawn.
- **Run 36** 4096 tokens (`scripts/run_ctx4096.sh`, index-form head `--chunk 256`): head gain 0.37 holds; transformer lead 0.07 → 0.04 on CPython.
- **Run 37** 20M class (`scripts/run_20m.sh`; corpora `linux_full.bytes` 220 MB, enwik8; CPython held out): kernel 1.30 / 0.91 / **0.78**; CPython held-out 2.42 / 1.68 / **1.46**; enwik8 1.54 / 1.38 / **1.36** (minGRU / +gated head / transformer). Head gain size-independent; transformer wins code, ties prose. Ckpts `results/ckpt/m20_*`.
- Engineering: `Block.chunk > 0` = exact linear-cost index form of the head (`_one_head_chunked`), minGRU scan batched; head overhead ~57 ms at d=256/2048/batch 8, plain minGRU ~80 ms, transformer ~190 ms.

## Next test

If continuing: a memory that grows with the model (n_key 16k, 2–4 heads, deeper value path) at the 20M setting, to see whether the code gap to the transformer is capacity. Otherwise write up. Previously: tuned-transformer control (done, Run 35) — the fairest remaining objection. Then 4096-token windows (does the gain keep growing with context?), and a fused/recurrent implementation for wall-clock. Previously proposed data/steps ladder on minGRU (does the gap close with training rather than width?); then the real scale run (d=256, 50 MB, seq 2048) on minGRU; real terminal transcripts. Not MaleCNS.
