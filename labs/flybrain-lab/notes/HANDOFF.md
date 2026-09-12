# Resume here

Lab: `/games/ailab/lab/flybrain-lab`  
Notebook: [`../notebook.md`](../notebook.md)  
JSON: `results/bind_mb_n128_uniqueand.json`, `results/bind_mb_n128_uniqueand_s1.json`

## Do not

- Load FlyWire / BANC / MaleCNS.
- Scale N.
- Plant class claws or share pattern/graph RNG.
- Call Run 13/14 a WikiText motif win.

## Current finding (2026-09-12)

- Run 16 unplanted 3-claw: fly < shuffle.
- **Run 17** unique-PN coincidence (4 claws, θ=3.5, WTA 8): fly **0.219 / 0.210** vs shuffle **0.188 / 0.181** vs chance 0.0625. Decoder in=16, n=128, `--shuffle region`. Graph seed `task_seed+10007`.

## Next test

Ablate unique-PN incoming or WTA if you want to see which lesion matters. Not MaleCNS.
