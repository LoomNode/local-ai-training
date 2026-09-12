# Resume here

Lab: `/games/ailab/lab/flybrain-lab`  
Notebook: [`../notebook.md`](../notebook.md)  
Directions: [`DIRECTIONS.md`](DIRECTIONS.md)  
FLM comparison: [`FLM.md`](FLM.md)  
Raw JSON: [`../results/`](../results/)

## Do not

- Load FlyWire / BANC / MaleCNS.
- Scale N.
- Plant class claws or freeze an identity decoder to fake a topology win.
- Call Run 13/14 a fly-wiring win on WikiText.

## Current finding (2026-09-12)

- Flatten+ordered WikiText beats unigram; region shuffle ≥ fly (Runs 13–14).
- Flatten binding saturates 1.0 for all arms (Run 15).
- **Run 16** trained emit head (in=16, 1218 scalars), random unique claws, coincidence, collapse duplicate shuffle edges: fly **0.472 / 0.446** vs shuffle **0.275 / 0.288** (seeds 0, 1). Chance 0.0625. JSON: `results/bind_mb_n128_trained.json`, `results/bind_mb_n128_trained_s1.json`.

## Next test

Optional: unfreeze SK or drop WTA as ablations. Not MaleCNS. Not N>128.
