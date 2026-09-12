# Resume here

Lab: `/games/ailab/lab/flybrain-lab`  
Notebook: [`../notebook.md`](../notebook.md)

## Do not

- Load FlyWire / MaleCNS.
- Plant class claws / shared pattern-graph RNG.
- Call the coincidence mixer a better LM or a better bag classifier than MLP.

## Current finding (2026-09-12)

- Run 17 fly graph unique-AND: fly 0.22 > shuffle 0.19, GRU ~0.24.
- **Run 18** stolen mixer (no connectome): 12-d bag mixer≈MLP≈GRU ~0.25; 48-d bag mixer **0.55** vs MLP **0.86** vs GRU **0.73**. Mixer ≫ chance, MLP wins. JSON: `results/mixer_bind_s{0,1}.json`, `results/mixer_bind_highd_s{0,1}.json`.

## Next test

Only if F is too large for a dense bag vector (true fly-hash regime). Not WikiText flatten. Not MaleCNS.
