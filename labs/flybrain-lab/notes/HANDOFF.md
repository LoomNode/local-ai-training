# Resume here

Lab: `/games/ailab/lab/flybrain-lab`  
Notebook: [`../notebook.md`](../notebook.md)  
Directions: [`DIRECTIONS.md`](DIRECTIONS.md)  
FLM comparison: [`FLM.md`](FLM.md)  
Raw JSON: [`../results/`](../results/)

## Do not

- Load FlyWire / BANC / MaleCNS.
- Scale N.
- Call Run 13/14 a fly-wiring win on WikiText.
- Call Run 16 a language model (frozen anatomical classifier on 3-hot odors).
- Treat chat as the record.

## Current finding (2026-09-12)

- Flatten+ordered WikiText beats unigram; region shuffle ≥ fly (Runs 13–14).
- Flatten binding saturates 1.0 for all arms (Run 15).
- **Run 16** mushroom-body small head (`emit`, decoder in=16): fly **0.157 / 0.115** vs shuffle **0.092 / 0.060** on seeds 0 and 1. Chance 0.0625. JSON: `results/bind_mb_n128_emit.json`, `results/bind_mb_n128_emit_s1.json`.

## Next test

Optional: learned (unfrozen) KC→MBON on the same small head without identity decoder. Not MaleCNS. Not N>128.
