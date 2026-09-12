# Resume here

Lab: `/games/ailab/lab/flybrain-lab`  
Notebook: [`../notebook.md`](../notebook.md)  
Directions: [`DIRECTIONS.md`](DIRECTIONS.md)  
Raw JSON: [`../results/`](../results/)

## Do not

- Load FlyWire / BANC / MaleCNS.
- Scale N.
- Plant class claws (explicit `class_claws=` **or** shared RNG seed with `make_sparse_patterns`).
- Freeze an identity decoder to fake a topology win.
- Call Run 13/14/16 a fly-wiring win.

## Current finding (2026-09-12)

- Flatten+ordered WikiText beats unigram; region shuffle ≥ fly.
- Flatten binding saturates 1.0 for all arms.
- **Run 16 unplanted** trained emit head: fly **0.241 / 0.277** vs shuffle **0.286 / 0.313** (seeds 0, 1). Fly > chance (0.0625), fly < shuffle. JSON: `results/bind_mb_n128_trained.json`, `_s1.json`. Graph seed offset `+10007`; tests fail if Kenyon[i]==pattern[i] for 16/16.

## Next test

The Kenyon prior is not earning a split at N=128 on this binding task without planting. Do not scale N to rescue it.
