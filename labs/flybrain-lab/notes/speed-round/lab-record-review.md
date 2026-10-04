> Compiled by a research subagent on 2026-10-04 for the speed round (see `notes/SPEED.md`), at Kyle's request to re-review the source material. Not checked against its sources line by line; what it marks unverified is unverified.

# flybrain-lab: the lab's own record, condensed for a training-speed design round

Prepared 2026-10-04, read-only. Nothing in the repository was changed, nothing under `.worktrees/` was opened, no GPU was used, no subagents.

**Sources read in full:** `notebook.md` (all 1,673 lines, Runs 0-49), `notes/RESULTS.md`, `notes/HANDOFF.md`, `notes/DIRECTIONS.md`, `README.md`; code `flybrain/model/sparsela.py`, `flybrain/model/hybrid.py`, `scripts/run_sparsela.py`.
**Extra material, always flagged "[outside]":** `notes/FLM.md`, `scripts/run_*.sh`, `results/m20_*.json` and `results/spine16_*.json` (timing, energy, memory, parameter and final-loss fields only), `docs/superpowers/specs/2026-10-03-group-learning-design.md`, read-only `git log` / `git show`, `scripts/cross_eval.py`, `tests/test_sparsela.py`.
**Four CPU-only checks I ran** (prefix as instructed, `PYTHONDONTWRITEBYTECODE=1`, scripts in this directory): `count_params.py` (parameter groups; total 16,066,824 matches `trainable_scalars` in `results/spine16_hybrid_s0.json`), `dtype_check.py` (dtypes under CPU bf16 autocast), `grad_check.py` (one forward/backward of the real config at B=1, T=512), `bf16_head_check.py` (whole-model bf16 cast at HEAD, tiny model).

**Working-tree note.** While I worked, someone else made comment, docstring and help-string edits (no line shifts) to `flybrain/model/hybrid.py`, `flybrain/model/state.py` and `scripts/run_sparsela.py` (07:43, branch `feat/group-learning`, uncommitted). All code line numbers below hold for the files as they are now; I read `hybrid.py` and `run_sparsela.py` just before those edits and re-checked that nothing else in them changed. `sparsela.py`, `notebook.md`, `notes/*`, `README.md` are untouched since 2026-10-03 / 2026-09-19.

**Conventions.** `N:123` = notebook.md line 123. `R:` RESULTS.md, `H:` HANDOFF.md, `D:` DIRECTIONS.md, `F:` FLM.md, `RM:` README.md. Code refs are `file:line` relative to the repo root. Numbers are quoted as the record gives them. Tags: "(record)" = stated in the sources; "[code]" = my reading of code; "[derived]" = my arithmetic on recorded numbers; "[outside]" = outside the four listed sources; "[CPU check]" = I ran it.

---

## 0. Ten-line summary

1. The lab's published wall-clock statement is a tie at 2048 bytes (6866 s vs 6657 s, fp32, unshared; N:1420, RM:71, R:143). Your 3x figure is a later measurement that is not in notebook.md; it is in `results/spine16_*.json` [outside]: 1.099 / 0.512 / 0.369 s per step (train_seconds / 16000), hybrid / zero-heads / transformer, mean power 150.5 / 150.1 / 149.7 W.
2. The same hybrid config under `--amp` took 4919 s = 0.307 s/step in Run 47 (N:1532). The spine hybrid with the same flags took 17586 s = 1.099 s/step. A 3.6x gap that the record does not explain (the notebook has no power-limit field). The transformer was never timed under amp in the notebook.
3. Heads are 53 % of the hybrid's step [derived]: (1.099 - 0.512) / 1.099. The zero-heads backbone (minGRU + one local-attention layer) is already 1.39x the transformer.
4. The head is fp32 by decision, not by finding: `sparsela.py:256-260`, added in commit 4ea1696 (2026-09-16), only rationale is the comment at `:257-258`. Run 42's whole-model bf16 eval ran the head in bf16 (my reading of commit fe91d82, which had no pin) and lost nothing (0.791 / 0.826 = fp32); that eval path no longer runs at HEAD.
5. The address is a dense matmul with a 0.78 %-dense buffer (`sparsela.py:92`), selected under `no_grad` (`:89-94`). The lab's FLOP formula costs it as 2 x claws x n_key = 32,768 (`hybrid.py:400`); the code does 4,194,304 FLOP/token/head.
6. No profiling of any kind is in the record. The only per-component timing is `H:85` (d=256: head ~57 ms, minGRU ~80 ms, transformer ~190 ms).
7. Every trained number in Runs 0-49 is Adam/AdamW + backprop. No component was ever trained by a local or one-shot rule in place of gradients. Closest data: a frozen random address (all of Runs 21-49), the in-context Hebbian write, and an accidental gradient cut through the memory state in Run 47 (+0.02 bpb).
8. Neither half of the hybrid beats the transformer alone: 4 heads 0.811 / 0.814 (N:1415), local attention without heads 0.808 (spine) vs transformer 0.781 / 0.789 (0.783 in the spine). The win needs both, each worth about 0.047.
9. Speed work already done: parallel T x T form -> exact chunked index form -> batched scan -> sliced no-grad top-k + sparse rescoring -> bf16 autocast (section 2.3).
10. The notebook's closing "Next" section (N:1651-1673) is stale (Runs 18-28) and has nothing on speed or gradient-free learning.

---

## 1. Mechanisms tried

"Regime" column: **BP-add** = trained by backprop alongside the backbone; **BP-frozen** = a component left untrained (frozen buffer) inside a backprop-trained model; **local-rule** = written by a forward-time local rule, writer trained by BPTT; **none** = not run here. Every row's training used Adam/AdamW (`run_sparsela.py:78-83`; V0 runs: "Adam lr=0.01", N:16).

### 1A. Fly anatomy and the V0 graph (Runs 1-17, CPU, N=128, 200-step protocol, unigram 4.564 / bigram 3.729 bpb)

| # | Mechanism (fly analogue the record names) | Run, notebook line | Compared against | Numbers as recorded | Verdict | Regime |
|---|---|---|---|---|---|---|
| A1 | Synthetic fly-like graph: sensory -> Kenyon -> core -> DA -> readout motifs, N=128, not FlyWire | Runs 1, 2, 13, 14: N:82, N:101, N:348, N:371; R:13 | degree-matched destination shuffle (Run 13); region-preserving shuffle (Run 14); matched GRU | Mean head d=4 (N:106-110): fly 4.600, shuffle 4.600, GRU 4.080. Flatten + ordered (N:356): fly 3.586 / dest-shuffle 3.621 / GRU 3.572. Region control (N:377): fly 3.600 / region-shuffle 3.577 / GRU 3.460 | Wash. "motifs do not beat a fair rewire" (N:382). R:13 "a wash on WikiText (3.60 vs 3.58) and on binding tasks". Runs 1-7 were I/O-limited, not tests of topology (D:5, D:79) | BP-frozen (topology fixed, synapses/cells trained, BPTT through graph steps) |
| A2 | Cell types: typed vs universal vs unique-per-neuron update programs | Runs 6-7: N:190, N:208; R:14 | one shared GRUCell; 4 type GRUCells; N unique GRUs | unique 4.596 / typed 4.600 / universal 4.600; shuffle 4.600-4.601 | Wash | BP-add |
| A3 | Synapse message width (scalar / 2 / 4 / projected) and sensory fan-in 10-50 % | Runs 3-5: N:121, N:140, N:170; R:15 | shuffle; matched GRU | all fly and shuffle arms 4.591-4.603 (unigram 4.564) | No effect. GRU rows move with `matched_gru` budget, not with the knob (N:153, N:181) | BP-add |
| A4 | Sign-locked E/I synapses (`w = sign*softplus(p)`, N:15) and nonnegative / signed keys | N:15; Run 23 N:633-636; Run 24 N:704; R:181 | signed vs unsigned sparse keys | copy@64, 600 steps (N:633): claws signed top-8 83 % / 82 % vs unsigned 84 % / 85 %. copy@512 (N:704): normalizer + signed 2-token key 94.4 % = unsigned 94.4 % | Not needed for keys. **Caveat:** R:181 says "sign constraints, claws, dopamine gating and structural plasticity were all tested"; the notebook has no run that drops the V0 sign-lock, and D:130 lists "Drop sign-lock" as a not-yet-run ablation | BP-add |
| A5 | I/O plumbing: decode sensory (Run 8), same-step skip (9), ordered hops (10), flatten head (12-13) | N:227-N:369 | each other | mean-head arms stay 4.594-4.601 (skip 4.594, ordered 4.597, sensory-decode fly 4.601 vs shuffle 4.421); flatten + ordered 3.586 | Not a fly mechanism; explains why Runs 1-7 sat at unigram | BP-add |
| A6 | Kenyon-style binding; claws; **unique Kenyon claws** ("count each PN once") | Runs 15-17: N:391, N:412, N:430 | region shuffle; GRU (chance 0.0625) | Run 15 (flatten head saturates): fly 1.000 at ~175 steps, shuffle ~125, GRU ~50. Run 16 (the 0.47 vs 0.28 split was a planted-claw RNG bug, N:421): fly 0.241 / 0.277 vs region-shuffle 0.286 / 0.313 vs GRU 0.271 / 0.225. Run 17 unique-PN 4-claw AND + WTA top-8: fly 0.219 / 0.210 vs region-shuffle 0.188 / 0.181 vs GRU 0.242 / 0.237 | Run 17 is the only fly > fair-rewire result in the anatomy block, and GRU is still ahead. Unique claws then "removed or varied with no loss" (R:15); "Claws are not needed" in the LM form (N:636) | BP-frozen (wiring random and fixed) |
| A7 | **Coincidence mixer** (stolen block, no connectome): frozen sparse +-1 expand -> count unique hits -> ReLU(z - theta) -> top-k -> thin read | Runs 18-20: N:448, N:475, N:508; closing notes N:1671, N:1673 | bag MLP and GRU (binding); dense GELU FFN and GRU at matched scalars / FLOPs (byte LM) | Binding 12 feat (N:454-457): mixer 0.250 / 0.231, MLP 0.253 / 0.265, GRU 0.253 / 0.238. 48 feat (N:461-464): mixer 0.547 / 0.564, MLP 0.854 / 0.865, GRU 0.724 / 0.727. Byte LM 600 steps (N:497): mixer 3.36, dense 3.28, GRU 3.16; FLOPs/step 17.9k / 27.2k / 24.1k. FLOP-matched d=48 (N:519): 3.09 / 3.16 / 3.08. 11 seeds both-matched (N:543): 3.127 +- 0.034 vs dense 3.168 +- 0.044 (paired -0.041, se 0.020, 9/11) vs GRU 3.110. 1500 steps, 5 seeds (N:544): 2.990 vs dense 2.948 vs GRU 2.917 | "real but small, and it is an early-training effect" (N:546); "cheaper per step, not better" (N:1671); "Do not scale it to claim a win". **"Wall-clock per step is 2-3x the dense arm in this Python loop (top-k and the per-token loop), unrelated to the FLOP count" (N:534)** | BP-frozen (expansion E untrained; theta and readout trained); replaces a dense FFN |
| A8 | **Connectome as reservoir** (nftechie/flm, MaleCNS 166,700 nodes / 25,582,938 edges, frozen graph + frozen 1.2B LM, 278,528-param adapter) | not run here: F:3; R:16; H:27 | their own direct-input (no graph) control | NLL nats/token (F:23-31): frozen backbone 1.3820; fly readout 1.3598; direct-input readout 1.3593; relabel 1.3813; no edges 1.3820; fly minus direct-input +0.000488 | "wiring participation != wiring helping" (F:50). HANDOFF "Do not: Load FlyWire" (H:27) | **Replacement** (frozen reservoir, only a small adapter trained); never run in this lab |
| A9 | Structural plasticity (V1 prune/grow) and V2 slot pool | "Path to V3" N:1603-1620 | none | N=64, drop 16 weakest edges, grow 16 random: E 239 -> 223 -> 239; loss (1 batch) 5.674 fly / 5.523 shuffle | Plumbing only; "Not a claim: plasticity improves bits/byte" (N:1613). R:181 nevertheless lists structural plasticity among things "tested" with no computational value; the notebook has no bpb test of it | none |

### 1B. Memory heads (Runs 21-49; byte LMs; loss in bpb unless "%" = copy accuracy)

| # | Mechanism (fly analogue the record names) | Run, notebook line | Compared against | Numbers as recorded | Verdict | Regime |
|---|---|---|---|---|---|---|
| B1 | **Sparse random address + top-k** (Kenyon expansion, RM:37) | Runs 21-25: N:552-811; 26-29: N:813-938; also R:24-25 | all-active dense code (D=74, D=1024); learned key+query (GLA); frozen dense Gaussian; softmax; GRU | copy@64, 600 steps lr 0.003 (N:567-571): sparse+FW 63 / 72 / 73 % vs sparse no-FW 0.3 % vs dense+FW D=74 18 / 0.9 / 27 % vs D=1024 0.3 % vs GRU 0.5 / 0.5 / 0.4 %; 1500 steps seed 0: 91 % vs 0.3 % (N:573). Learning speed (N:652, lr 0.001): >90 % at step 250 vs dense Gaussian 500, softmax 750, learned 750-1500; all end 91-93 %. copy@512 (N:706): sparse 94 % / 92 %; dense Gaussian, learned, softmax, GRU 0.3-0.4 %. Text, WikiText@512 (N:714-719): GRU 2.586 < learned 2.886 < softmax 2.925 < dense Gaussian 2.973 < Gaussian top-4 3.139 < sparse top-4 3.160. LM scale, minGRU (N:878-880): code 1.620 / 1.628 -> 1.480 / 1.499 with the head; prose 2.079 / 2.087 -> 2.042 / 2.045 | "Top-k sparsity is the only ingredient that survived every test" (N:636). "Sparse = exact retrieval, not text" (H:47). At 64 tokens it is "a learning-speed and robustness advantage ... not a capability gap" (N:654) | **BP-frozen**: `proj` is a `register_buffer` (`sparsela.py:40`) inside a backprop model; replaces learned key/query projections. Which units fire is outside autograd (`sparsela.py:89-94`) |
| B2 | **Trainable keys instead of the frozen one** (GLA learned key+query; `tied` trainable Gaussian; `mixed` learned + sparse head) | Runs 22-25, 26, 28, 29: N:586-N:811, N:813, N:869, N:902 | the frozen sparse key | Run 25 (N:740-745) wiki all bytes: GRU 2.079, mixed 2.335, learned 2.418, tied 2.466, softmax 2.443, sparse-only 2.908; wiki copyable: mixed 1.430 vs learned 2.169; code all bytes: GRU 1.602, mixed 1.832, learned 1.972, tied 2.019, sparse-only 2.558. `tied` ~ `learned` (N:783). On GRU code (N:823-825): sparse head 1.463 / 1.477 vs learned-key head 1.518 / 1.544. minGRU/GLA (N:891): learned-key head gets "a third to a half of the code gain and is a net loss on prose". GLA grows its own recall at d >= 256 (N:931) | Frozen sparse key = retrieval; trained keys = soft matching. Mixed is the best fast-weight arm but every attention-style arm loses to a GRU at ~500k scalars (N:786) | BP-add (the alternative to B1) |
| B3 | Active units / sparsity (top-k size) | Runs 19, 20, 22, 23, 48: N:499, N:532, N:609, N:658-660, N:1551-1559 | k sweeps | Run 22 (1024 units, copy@64, N:609): k=8 84 %, 16 83 %, 32 82 %, 64 63 %, 128 0.3 %, 256 0.3 %, 512 0.3 % ("cliff between 6 % and 12 % active"). Run 20 byte LM (N:532): k=0 3.263, 32 3.096, 64 3.105, 128 3.127, 256 3.296. Run 19: top-k off 4.567 / 3.411. Run 48 (d=256, N:1553-1557): top-2 0.979, top-4 0.966, top-8 0.965 | top-4 stays; "2 active units is worse" (N:1561) | BP-frozen |
| B4 | **Claws (+-1, 4 distinct input dims per unit) vs Gaussian projection** | Run 23 N:626-636; Run 24 N:706, N:718-719; Run 19 N:499 | Gaussian top-k | copy@64 600 steps k=8 of 1024 (N:630-634): claws 84 % / 85 %, Gaussian top-8 91 % / 91 %, Gaussian top-64 92 %. copy@512 seeds 0/1: sparse 94 % / 92 %, Gaussian top-k 91 % / 93 %. WikiText@512: Gaussian top-4 3.139 vs sparse 3.160. Run 19: 8 claws 3.420 / 3.707 | "Claws are not needed (a Gaussian projection with top-k is as good or better)" (N:636). **The shipped hybrid still uses claws = 4** (`run_sparsela.py:207`, `seqmixer.py:25-37`); a Gaussian-key hybrid at LM scale is not in the record (no such result file; the `gausstopk` files are copy runs and 2-block WikiText@512) | BP-frozen |
| B5 | **theta thresholds** (record's word: "coincidence"; theta learned per unit, N:482) | Run 19 N:499; Run 17-18 N:432, N:452; R:112 | theta init 0 | Run 19: theta init 0 gives 3.389 / 3.368 (600 steps, seeds 0/1; default-theta seeds 3.408 / 3.370, N:493-494). LM head: `KeyCode.theta` (`sparsela.py:41`), 16,384 params for 4 heads = "0.10 % of a 20M model" (R:112) | Learned vs frozen theta at LM scale: **not in the record**. (Runs 6-7 "theta too shared / too unique" in D:34 refer to V0 type programs, a different object) | BP-add (init 1.0, `run_sparsela.py:209`) |
| B6 | **Fast-weight write rule**: additive outer product key_{s-1} x v_s, decay 1.0 (KC->MBON plasticity, RM:38, N:554) | Run 21 N:552-582; Runs 22-24; LM Runs 26-49 | no FW; dense-key FW; GRU | See B1 (sparse+FW 63-73 % vs 0.3 % no-FW; 91 % at 1500 steps). Run 21 recurrence (N:557-561): F <- 0.95 F + sigmoid(G h_t) (V u_t) x key_{t-1}; "Gradients flow through the writes (BPTT, seq 64)" (N:563). WikiText (N:575): "Fast weights cost nothing on text at this length and buy nothing" | The mechanism that works. Dense keys fail because overlapping keys interfere (N:577) | **local-rule**: the write is a forward-time outer product, no gradient needed to write; the writer (val, key conv, backbone) is trained through it. Run 47 accident: cutting the gradient through S at chunk boundaries cost +0.02 bpb (row B16) |
| B7 | **Dopamine-like write gate** (learned modulator sigmoid(G h_t) on the write; DA) | Run 21 N:560; Run 22 N:607; R:15; RM:46 | constant write strength | constant 0.5 -> 0.3 % / 2 %; constant **1.0 -> 69 % / 71 %**; learned gate 63-73 % (N:607) | "The learned modulator is not doing anything a constant write strength does not" (N:607). Dropped; the LM form has constant strength (N:639). (The `--gate` flag in the LM is a different thing: B10) | BP-add |
| B8 | **Delta-rule write** | Run 23 N:633 | additive write | copy@64: delta 85 % / 85 % vs additive 84 % / 85 % | "buys nothing at this length" (N:636). Not tested at 512 or LM scale | BP-add |
| B9 | **Normalizer** (divide read by total match mass; "standard fix", R:27, no fly analogue named) | Run 24 N:695-704 | no normalizer | copy@512 1024 units k=4 (N:698-704): no normalizer 1-token key 58 %; no normalizer 2-token key 0.4 % (learned or frozen conv, 3 lrs); normalizer 1-token 56 %; **normalizer + 2-token key 94.4 %**; + signed 94.4 % | Essential at 512: nonnegative keys "accumulate coherent cross-talk" (N:696). It also amplifies weak matches, which the gate (B10) later corrects (N:1031) | BP-add (no params) |
| B10 | **Match-strength gate** `g = mass/(mass + softplus(tau))`, one scalar (no fly analogue named) | Run 33 N:1016-1035; Run 34 N:1058 | ungated head | d=128 code (N:1020-1028): minGRU 1.621, + head 1.481, + gated head 1.435; every matched bin improves 0.05-0.10; unmatched cost recovered 0.015 of 0.06 (predicted 0.03). Run 34 (N:1047-1049): CPython 1.162 / 1.157 -> 1.110 / 1.108; enwik8 1.671 / 1.680 -> 1.625 / 1.630; held-out kernel 2.173 / 2.099 -> 2.033 / 2.018 | Replicates: "-0.05 on code, -0.045 on prose, and -0.1 on the held-out kernel" (N:1058). In the best design | BP-add (`sparsela.py:206`) |
| B11 | **Key convolution** (causal depthwise conv so the key is a bigram / n-gram; mixed widths per head `--key-convs 2,4`) | Run 24 N:695, N:701-704; Run 41 N:1345, N:1358, N:1371; Run 48 N:1553-1561 | single-token key; widths 2,2 | Without the normalizer a bigram key fails (0.4 %). d=256 CPython (N:1357-1358): 2 heads 1.043 vs keys 2,4 1.037 (-0.006); ff640 1.024 vs ff640 + keys 1.019. Run 48 (N:1553-1555): keys 3,3 0.961, keys 2,3 0.965 vs 0.966 | Bigram key essential (with normalizer); width choice worth <= 0.006, "Closed" (N:1561). No fly analogue named | BP-add (trained `nn.Conv1d`, `sparsela.py:219`; `--freeze-conv` exists only on the `SparseKeyLM` path, `run_sparsela.py:63`) |
| B12 | **Signed keys** | Run 23 N:633; Run 24 N:704 | nonnegative keys | see A4 | Not needed. The index form rejects signed keys (`sparsela.py:86`) | BP-frozen |
| B13 | **Count head / next-byte count memory** (one-hot next bytes under the context key; "KC->MBON drives the output directly", N:1347) | Run 41 N:1347, N:1360, N:1369 | 2-head base; local-only | base 1.043; count 1.039 (-0.004, predicted -0.03); "3x slower. Out." (N:1360). local512 0.966 vs local + count + recency 0.972 (N:1369), "~2x slower" | Out. "the value heads already carry the continuation histogram implicitly" (N:1360). `CountHead` is `hybrid.py:231-259` | BP-add |
| B14 | **Learned recency / decay** (scalar per head, `--learn-decay --decay 0.999`; per-head `--decays 1.0,0.999`) | Run 41 N:1348, N:1361-1362, N:1372-1373; Runs 23-24 N:667, N:721 | decay 1.0 | decay 1.030 (-0.013; learned half-lives 662 / 435 bytes, init 693); archive + recent 1.028; recent + archive 1.032; count + decay 1.025. Decay 0.98 kills copy@512 (N:667); no change on text (N:721) | <= 0.013, "use archive + recent if at all" (N:1375). Recency as the fix for the ~0.1 long-window blending loss: "untested" (N:1328) | BP-add (1 scalar per head) |
| B15 | **Slots and capacity rules** (n_key units) | Run 25 addendum N:788-809; Run 38 N:1155; Run 39 N:1221-1259; Run 40 N:1263-1334 | 1024 / 4096 / 16k / 32k / 64k units; load in writes/slot | 2048 window 8x32 spans (N:801): 1024 units 45 % vs 4096 units 94.5 %. Run 39: 4 writes/slot costs 0.00-0.015, 8 costs 0.02-0.09, 16 costs 0.05 (N:1257). Rule "slots ~ window x active / 2" (4096 for 2048). Run 40 (N:1325-1326): 4096 / 32768 / 65536 slots at 32k: 0.826 / 0.772 / 0.756; at 64k 0.956 / 0.880 / 0.858. **Eval-time slot expansion hurts: 0.863 vs 0.826 at 32k** (N:1304). `slots16k` and `depth4x16k` arms cancelled as "predictable" (N:1155) | Slots follow the window (load <= ~4/slot), not the model; fixed at training time | BP-frozen (buffer sized at train time) |
| B16 | **Precision of the memory state** (fake_quant, per-row symmetric integer, `sparsela.py:121-132`) | Run 42 N:1381-1404 (eval-only); Run 47 N:1523-1543 (trained) | fp32 | Eval-only (N:1387-1397), 2048 / 32768: fp32 0.791 / 0.826; 16-bit 0.791 / 0.826; 8-bit 0.793 / 0.881; 6-bit 0.832 / 1.308; 8-bit on one head only: top 0.827, L6 0.830, L4 0.840, L2 0.845 (at 32768); layout L2 fp32 / L4 16 / L6 8 / top 8: 0.791 / 0.832; bf16 whole model 0.791 / 0.826. Trained (N:1529-1534): fp32 0.764 / 0.762 in 6866 s; bf16 alone 0.761 in 4919 s; 16-bit with the round() bug 0.784 (repeat bytes 0.197); 16-bit with STE 0.761; 8-bit with STE 0.839 | 16-bit lossless (eval and trained, after the fix); 8-bit +0.075 trained though +0.002 eval-only at 2048; 6-bit breaks. Details in section 2.6 | BP-frozen quantiser with straight-through gradient |
| B17 | **Heads in depth** (`--mem-every`) | Run 38 N:1144-1215 | 0 / 1 / 2 / 4 / 8 heads; transformer | bpb (N:1176-1183): 1.295 (12.9M) / 0.910 (13.4M) / 0.834 (13.95M) / **0.811 / 0.814 (15.0M)** / 0.802 (17.15M); transformer 0.781 / 0.789 (17.09M). "Gains per doubling of heads: 0.076, 0.022, 0.010" (N:1185) | One head per two layers is where the marginal head stops paying "for its 0.5M parameters and its step time" (N:1185). **All head-count wall-clocks were shared-GPU** (N:1163-1183) | BP-add |
| B18 | Head placement: on the input (`pre`) vs on the output (`post`) | Runs 26, 28: N:839, N:892 | each other | GRU code (N:823-824): input 1.463 / 1.477, output 1.480 / 1.487; prose output 2.009 vs input 2.050 (N:833-834) | "input-side is the cheaper win on code; output-side wins on prose and on repeat bytes" (N:839). Best design uses post | BP-add |
| B19 | **One sliding-window attention layer** (not a fly mechanism; the single biggest add) | Runs 41, 43, 44, 46: N:1338, N:1408, N:1451, N:1502 | 2 heads base; param-matched FFN; windows 128-1024; 1-2 layers | d=256 (N:1363): 0.966 at 2.25M vs 1.043 base, "-0.077", transformer 1.010. Param-matched FFN 640: 1.024 (N:1370). 20M (N:1417): 4 heads 0.811 / 0.814 -> 4 heads + local + keys 2,4 **0.764 / 0.762** vs transformer 0.781 / 0.789. Window (N:1457-1460): 128 0.984, 256 0.974, 512 0.966, 1024 0.971. Placement (N:1508-1513): after 1 0.984; 2 0.966; 3 0.964; 1+3 0.943 (2.52M) vs param-matched FFN 0.954 | One layer, mid-stack or later; window saturates at 512; second layer "not worth its parameters" (N:1516). Gain lands on unmatched bytes: 1.737 -> 1.656 (N:1468) | BP-add |
| B20 | Transformer backbone + head (negative control) | Run 30 N:940-968 | plain transformer | 9000 steps lr 0.001 (N:958): plain 1.412 / 1.409; + head 1.517 / 1.539 (input), 1.735 / 1.692 (output). lr 0.0003 (N:959): plain 1.953 / 1.939; + head 1.866 / 1.934, 1.877 / 1.877 | Once attention retrieves, "the head is interference" (N:961); head worse by 0.11-0.33 | BP-add |
| B21 | Fast-weight memory as an **in-context (gradient-free) learner** | Run 25 sec. 5 N:769-779; Run 27 N:856; Run 34 N:1057; Run 37 N:1132 | no memory | WikiText-trained -> Python code, no gradient (N:773-777): sparse 7.01 / 6.80, mixed 7.72 / 7.62, learned 8.06 / 8.40, softmax 8.40 / 8.31, GRU 10.08 / 9.83 ("No in-context domain adaptation", N:779). Later, held-out kernel at 1.85M (N:1057): plain minGRU first eighth 3.6 / last 3.4; memory arms and transformer 2.7 -> 1.9. Leakage control (N:856): keys from another sequence -> exactly the plain GRU (code 1.613 vs 1.601). Held-out CPython gap 0.74 "because the plain recurrence cannot adapt in-window and the head can" (N:1132) | In-context adaptation is "what the head buys a recurrence" (N:1057); null at ~0.5M scalars, real from 1.85M | local-rule at inference; no weight learning without gradients anywhere |

### 1C. What the record contains about learning without gradients

- **Nothing was trained without gradients.** Every trained arm is Adam / AdamW + backprop: `run_sparsela.py:78-83` (optimizer), `:297-298` (backward, clip at 1.0, step); V0 runs "Adam lr=0.01" (N:16).
- **Components left untrained inside a gradient-trained model:** the address `proj` (B1, B4, B12, B15), the coincidence-mixer expansion E (A7), the V0 topology (A1). Each was compared against its trained alternative (B2 for the address; matched dense FFN for A7).
- **Forward-time local write:** the Hebbian outer product (B6). It needs no gradient to write, but the writer is trained through it (Run 21: "Gradients flow through the writes (BPTT, seq 64)", N:563; chunk loop `sparsela.py:187-190`).
- **The one measurement of how much the write path needs gradient:** Run 47's accidental cut. `fake_quant` used `.round()` (zero gradient), "the heads learned to write with no gradient from anything older than the current 256-byte chunk" (N:1537): 0.784 vs 0.761 with the fix, repeat bytes 0.197 vs 0.156, non-repeat unchanged. So truncating the gradient through the carried state at 256-byte boundaries cost about 0.02 bpb, all on repeat bytes. It was a bug, not a designed test.
- **Reservoir / frozen-graph replacement:** only the nftechie/flm result (A8), not run here.
- **Stated hypothesis never isolated:** "gradient through F touches only k units per write, so credit assignment does not entangle every stored pair" (N:618).

---

## 2. Everything the record says about cost

### 2.1 Chronology of cost claims (what was said, and what was later withdrawn)

Power limit during Runs 0-49: **not in the record** (no field, no mention). The only power statement is [outside] the spec: "Both cards are now capped at 150 W. No existing timing is comparable." (`docs/superpowers/specs/2026-10-03-group-learning-design.md:35`).

| Where | Claim as recorded | Setting | Status |
|---|---|---|---|
| N:534 (Run 20) | "Wall-clock per step is 2-3x the dense arm in this Python loop (top-k and the per-token loop), unrelated to the FLOP count." | coincidence mixer, d=48 | stands; FLOP saving (34 %) not realised in time |
| N:640 (Run 23) | parallel form "reproduces the recurrent result (sparse 90 %) in 9 s on a 3090 instead of 5 min on CPU" | copy@64 | stands |
| N:933 (Run 29) | "+8 % params, +6 % arithmetic per byte" for one head at d=384; plain minGRU d=384 is "more than 2x the parameters and 2x the arithmetic" | analytic | stands (analytic only) |
| N:1056 (Run 34) | "at 2048 tokens the attention costs 16x more per window than at 512 while the head's cost is unchanged" | analytic | stands |
| N:1084 (Run 35) | "the recurrence + head runs a step in about 130 ms against about 190 ms for the transformer (Run 36 will measure this on an idle GPU)" | d=256, 4 layers, seq 2048, batch 8 | Run 36 did not measure on an idle GPU (N:1103 is "from this shared-GPU run"). H:85 repeats it: "head overhead ~57 ms at d=256/2048/batch 8, plain minGRU ~80 ms, transformer ~190 ms" |
| N:1103 (Run 36) | head arm 8000 steps 2200 s vs transformer 3435 s | seq 4096, batch 4, shared GPU | indicative only |
| N:1135 (Run 37) | shared GPUs: head 7137 s, plain minGRU 6000 s, transformer 11861 s; "trains in 60 % of the transformer's time" | 20M, shared | **withdrawn** (N:1200, R:97) |
| N:1163-1183 (Run 38) | shared-GPU wall-clock by head count: 0 heads 6000 s; 1 head 7137; 2 heads 9553; 4 heads 10620; 8 heads 17210; transformer 11861 | 20M, shared | all shared; the "efficient point" claim rests on bpb vs parameters, not on unshared step time |
| N:1200 (Run 38) | unshared transformer 6657 s; unshared 32768-slot 4-head arm 11152 s; head arm "trains slower than the transformer (about 1.5-1.7x)" | 20M | the 1.5-1.7x statement was itself superseded (next row) |
| N:1360, N:1369, N:1375 (Run 41) | count head "3x slower"; local + count + recency "~2x slower"; "local512 2302 s vs base 719 s unshared; not comparable, measure later" | d=256, 8000 steps | local512 vs base re-measured only indirectly in Run 44 |
| N:1417, N:1420 (Run 43) | 4 heads + local: **6866 s unshared** vs transformer 6657 s: "Wall-clock is a tie". The "~11000 s" figure "was extrapolated from shared runs and is withdrawn; with the sliced top-k of Run 40 the head arm costs about the same as attention at 2048" | 20M, fp32, 16000 steps, one seed each | the lab's standing published claim: RM:71, R:143, R:161, N:1447 |
| N:1455-1460 (Run 44) | unshared train seconds at d=256: window 128 848 s, 256 870 s, 1024 1069 s | 2 heads + local, 8000 steps | vs base 719 s unshared (N:1375): +18 % / +21 % / +49 % [derived] |
| N:1484, N:1494 (Run 45) | transformer trained at 8192: 17241 s vs 6657 s at 2048; hybrid "at 40 % of the training time" | 20M, fp32 | stands |
| N:1527-1536 (Run 47) | bf16 autocast 4919 s vs fp32 6866 s, "28 % faster". Hybrid only | 20M, 16000 steps | the transformer was never timed under amp (N:1527-1534 lists no transformer row) |
| N:1567 (Run 49) | hybrid 31237 s, transformer 24228 s at 50M | shared GPU | "not comparable" |
| H:72 | still carries "4-head ~11000 s (head arm trains slower at 2048)" | | contradicted by N:1420 |
| RM:42 / RM:71 | "per-token cost flat in window length" / "Wall-clock is a tie at 2048 bytes, not a win; the advantage is that per-token cost does not grow with the window" | | the standing claim |

### 2.2 Seconds per step from the recorded totals [derived; `results/m20_*.json` confirm the totals]

16,000 steps, batch 8, seq 2048 = 16,384 tokens/step, d 512, 8 layers.

| run | seconds | s/step | note |
|---|---:|---:|---|
| transformer fp32, unshared (Run 38 s1, N:1200) | 6657.45 | 0.416 | |
| hybrid fp32, unshared (Run 43 s1, N:1417) | 6866.20 | 0.429 | same config, shared GPU gave 18943.7 s (2.76x); transformer shared 11861 s vs 6657 s unshared is 1.78x |
| hybrid bf16 autocast (Run 47, N:1532) | 4918.55 | 0.307 | 28 % less than fp32 |
| hybrid autocast + 16-bit state, round() bug (N:1530) | 6935.88 | 0.434 | quantiser cost ate the amp gain |
| hybrid fp32 + 16-bit state, STE fix (N:1533) | 7657.82 | 0.479 | +792 s vs fp32 |
| hybrid fp32 + 8-bit state, STE fix (N:1534) | 7560.84 | 0.473 | |
| 4 heads, no local, 32768 slots, unshared (N:1200) | 11151.6 | 0.697 | pre-"sliced top-k" per N:1420 |

### 2.3 Speed work already done (so it is not repeated)

Chronology from the notebook and `git log` [outside for dates]:
1. Parallel T x T form on GPU (Run 23, N:638-640, commit c3e3f6a).
2. Exact linear-cost chunked index form, `--chunk` (commit d3c9655, 2026-09-14; H:85; used from Run 36, N:1094); minGRU scan batched "no per-chunk slicing" (same commit; `hybrid.py:43-46`).
3. Memory independent of slot count: "sliced no-grad top-k + sparse rescoring in `sparse_code`; index-equality intra-chunk scores; peak 10.7 GB at batch 8" (N:1304; commit 396875d, 2026-09-15; code at `sparsela.py:80-98`, `:166-168`). Run 43 credits it for closing the wall-clock gap (N:1420).
4. `--chunk 512` for the 32k / 64k slot arms: "chunk 512 halves the per-chunk state copies" (`scripts/run_capacity.sh:30` comment) [outside]; `--chunk 4096` at eval keeps 65536 slots x 128 chunks from 17 GB to "~2 GB per head" (N:1304).
5. bf16 autocast, head pinned fp32 (Run 47; commit 4ea1696, 2026-09-16).
6. Never proposed or tried in code or notes (grep of `flybrain/`, `scripts/`, `tests/`, notes): `torch.compile`, `allow_tf32` / `set_float32_matmul_precision`, fused AdamW, custom kernels, sparse application of the claws, activation checkpointing. PyTorch in the venv is 2.14.0+cu130 [CPU check]; its default `matmul precision` is "highest" and `cuda.matmul.allow_tf32` is False, so the fp32 baseline behind "28 % faster" is non-TF32 fp32.

### 2.4 What `--amp` does

- Flag: `scripts/run_sparsela.py:222`, help string "train and evaluate under bfloat16 autocast (backbone matmuls/attention in bf16; the memory head stays fp32)".
- Context manager: `_amp(args, device)` returns `torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(args.amp))` (`run_sparsela.py:92-94`).
- Training: forward and the loss are inside it (`:294-296`); the loss is `F.cross_entropy(logits.float()...)` (fp32). `loss.backward()`, `clip_grad_norm_(..., 1.0)` and the optimizer step are outside it (`:297-298`). No GradScaler (bf16). Master weights stay fp32.
- Evaluation: `eval_wiki` (`:119`) and `eval_fixed` (`:142`) use it; `eval_copy` (`:98-107`) does not; `scripts/cross_eval.py` has no autocast.
- What actually runs in bf16 in the current design [code + CPU check]: the minGRU `z` and `h` projections and FFN (`hybrid.py:20-22`, `:30`), local attention `qkv` / `out` / SDPA (`:111-112`, `:137`), the decoder (`:321`), and inside each head only `ln1` (fp32 out), the key conv and `val` projection (outputs bf16) and the output projection `outs` (output bf16).
- What stays out: embedding and residual stream (fp32); the head's whole internal computation (`sparsela.py:259-260`); the minGRU scan (fp64, `hybrid.py:39-47`, input `z`, `cand` are bf16-rounded values because autocast leaves `sigmoid` of a bf16 matmul in bf16 [CPU check: z_linear_out bf16, sigmoid_out bf16, a bf16, layer_out fp32, carry fp64]).
- Run 47's own accounting: "Autocast costs memory rather than saving it here (cached bf16 weight copies: the 50M hybrid needs 18.3 GB at batch 4 under autocast, 14.4 GB in fp32), and the minGRU scan (fp64) and the head (pinned fp32) stay out of it, so the 28 % is from the projections, FFNs and the one attention layer." (N:1538)
- Stale comment: `scripts/run_50m.sh:4` still says "bf16 autocast + 16-bit state cost 0.02 at 20M and gave no speedup" (written before the STE fix; the "no speedup" is the quantiser-ate-the-gain row in section 2.2).

### 2.5 Why the head is pinned to fp32, and what the record shows

- **Code:** `sparsela.py:256-260`:
  `def _one_head(...)`: comment (`:257-258`) "the fast-weight sums accumulate in fp32 whatever autocast does to the rest of the model; the state's precision is controlled explicitly by state_bits (fake-quantised between chunks)"; `with torch.autocast(device_type=xk.device.type, enabled=False): return self._one_head_fp32(code, xk.float(), v.float(), state)` (`:259-260`).
- **History [outside, git]:** the pin was introduced by commit 4ea1696 (2026-09-16 22:22), "Training under bf16 autocast (--amp) with the memory head pinned to fp32, and --state-bits ...", with unit test `tests/test_sparsela.py::test_bf16_autocast_keeps_the_memory_head_fp32_and_trains_with_16bit_state` (checks the head's read reaches its output projection in fp32 and that gradients exist). The commit has no measurement behind it; the comment is the only rationale.
- **Record:** Run 47 header (N:1523) and N:1538 describe the pin; H:81 repeats it. No sentence says bf16 hurt the head.
- **Run 42 is not the origin of the pin and is evidence against needing it** [outside, git]: Run 42's "bf16 weights (whole model)" row (N:1395, 0.791 / 0.826 = fp32) was produced by `cross_eval.py --bf16`, which does `m = m.to(torch.bfloat16)` (`cross_eval.py:126-127`; same at commit fe91d82). At fe91d82 (2026-09-15, before the pin) `sparse_memory_read` created `S` and `Ksum` with `xk.new_zeros(...)` (the input's dtype) and `_one_head` had no cast, so that eval ran the head, including S and Ksum accumulation, in bf16 and lost nothing at 2048 or 32768. The notebook labels the row "bf16 weights" and never says the head was in bf16.
- **That path no longer runs at HEAD** [CPU check, tiny model, `bf16_head_check.py`]: a whole-model `.to(torch.bfloat16)` forward raises for a head-only model ("expected m1 and m2 to have the same dtype, but got: float != BFloat16": the pin's `.float()` inputs against the bf16 `proj` buffer) and, for the full design, earlier in `LocalAttention`'s SDPA (rotary returns fp32 q and k while v is bf16). So `cross_eval.py --bf16`, the Run 42 measurement, cannot be repeated on any hybrid at HEAD. (Run 42's checkpoint was the 4-head arm without local attention, N:1383.)
- **Not in the record:** a bf16 head during training; fp16 anywhere; the head's step time in bf16; a head timing at all.

### 2.6 What precision the memory state was shown to need

"16-bit" and "8-bit" in Runs 42 and 47 mean `fake_quant` (`sparsela.py:121-132`): per-row symmetric integer quantisation with `qmax = 2**(bits-1) - 1`, applied to `S` and `Ksum` once per chunk boundary (`:191-192`), simulated on an fp32 tensor. Nothing is stored in 16 bits, so no memory or time is saved today. It is not fp16 or bf16 float storage.

| setting | 2048 | 32768 | source |
|---|---:|---:|---|
| fp32 | 0.791 | 0.826 | N:1387 |
| state 16 bits, all heads | 0.791 | 0.826 | N:1388 |
| state 8 bits, all heads | 0.793 | 0.881 | N:1389 |
| state 6 bits | 0.832 | 1.308 | N:1390 |
| 8 bits on top head / L6 / L4 / L2 only | 0.791 / 0.791 / 0.791 / 0.792 | 0.827 / 0.830 / 0.840 / 0.845 | N:1391-1394 |
| layout L2 fp32, L4 16, L6 8, top 8 | 0.791 | 0.832 | N:1397 |
| bf16 whole model (head in bf16 at that commit, see 2.5) | 0.791 | 0.826 | N:1395 |
| bf16 + 8-bit state | 0.793 | 0.877 | N:1396 |

Trained (Run 47, 20M kernel, 16000 steps; N:1529-1534): fp32 0.764 / 0.762; bf16 alone 0.761; 16-bit with `.round()` bug 0.784 (copyable 0.197); 16-bit with STE 0.761 (copyable 0.156); 8-bit with STE 0.839 (copyable 0.214). Reading in the record: "a coarse state hurts learning to write, not reading; 16 bits is the training budget" (R:175); early heads are most sensitive (8 bits: top 0.001, L6 0.004, L4 0.014, L2 0.019, N:1401); 8-bit eval damage grows with window because "a per-row absmax scale gets coarser as rows accumulate mass" (N:1400).

Nobody measured the head in fp16 / bf16 storage as a trained state, and nobody measured it in bf16 compute while training.

### 2.7 FLOPs, profiling, memory

- **FLOPs are analytic only.** `GRUHybrid.flops_per_step` (`hybrid.py:390-402`) and `SparseKeyLM.flops_per_token_recurrent` (`sparsela.py:341-353`); the runner writes `flops_per_token_recurrent` into every JSON (`run_sparsela.py:324-325`). Spine values [outside]: hybrid 38,182,912 vs transformer 33,816,576 per token (+12.9 %); zero-heads also 33,816,576 because the formula omits the scan, local attention and the attention O(T) term (`hybrid.py:394` comment "omits the O(T) attention term on purpose").
- **The formula and the implementation disagree on the address [code].** `hybrid.py:400` costs the key as `2 * claws * n_key` = 2 x 4 x 4096 = 32,768 FLOP/token/head (sparse claws). `sparsela.py:92` multiplies by the dense `[4096, 512]` buffer: 2 x 4096 x 512 = 4,194,304 FLOP/token/head, in fp32 with autocast off, under `no_grad`. Over 4 heads that is 16,646,144 FLOP/token more than the formula assumes (+49 % of the transformer's 33.8M). The buffer has exactly 4 nonzeros per row (16,384 of 2,097,152 entries, 0.78 %) [CPU check].
- **Profiling: none.** No profiler call, no per-op timing, no kernel breakdown anywhere in the notebook, notes, scripts or tests (grep "profil" finds nothing). Timers exist (`time.perf_counter`, `SplitClock`) but only give totals. The only per-component numbers: N:1084 / H:85 (d=256).
- **Energy / power:** not in the notebook. `--meter` (`run_sparsela.py:240`, `flybrain/meter.py`) exists since 2026-10-03; its fields land only in `results/spine16_*.json` [outside].
- **Memory in the record:** state is "One 4096 x d state matrix per head per sequence" (R:160), 8 MiB per sequence per head at d=512 fp32 [derived]; 65536 slots at batch 8 peaks 10.7 GB (N:1304); 50M fp32 14.4 GB hybrid vs 9.4 GB transformer (`run_50m.sh:4`); 50M autocast 18.3 GB (N:1538); two local layers "need ~3.5 GB at this size" (N:1517).
- **Code facts that bear on cost, unprofiled [code]:** the state is rebuilt in full each chunk (`sparsela.py:189`: `S * lam ** Lc + zeros_like(S).index_add(...)`, 64 MiB per head at B=8, 8 chunks per window); the scan carry is a Python loop of `nC - 1` = 63 iterations at T=2048 per layer (`hybrid.py:44-45`); the top-k runs in 4 slices of 512 positions per head (`sparsela.py:80`, `:91-93`); selected rows are gathered as `[B, T, 4, 512]` fp32 and multiplied again to recompute z with gradient (`:95-96`); the intra-chunk score tensor is `[B, 256, 256, 4, 4]` (`:167`). None of these has a recorded timing.

### 2.8 The origin of your 1.10 / 0.51 / 0.37 [outside: `results/spine16_{hybrid,zeroheads,tf}_s0.json`, written by `scripts/run_spine.sh`, flags `--amp --meter --val-fixed 128`, GPU 1, `power_limit_w` 150]

| arm | train_seconds / 16000 | joules (whole arm) | J/step [derived, joules / 16000] | mean_power_w | peak VRAM | final val_bpb | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 17585.6 s = 1.099 s | 2,667,984 | 166.7 | 150.46 | 7.96 GB | 0.7613 | 16,066,824 |
| zero-heads (minGRU + local 512) | 8188.7 s = 0.512 s | 1,235,520 | 77.2 | 150.07 | 5.03 GB | 0.8079 | 13,930,752 |
| transformer | 5901.0 s = 0.369 s | 889,055 | 55.6 | 149.74 | 2.77 GB | 0.7828 | 17,085,696 |

- All three sit at the cap, so seconds are about joules / 150. Hybrid / transformer: 2.98x in time, 3.00x in energy. Heads: 0.587 s and 89.5 J of the hybrid's step (53 %). Backbone excess over the transformer: 0.143 s (zero-heads / transformer = 1.39x).
- `joules` and `mean_power_w` cover the whole arm, evals and saves included, while `train_seconds` excludes them (the `--meter` help string, `run_sparsela.py:240`, says so in the current working copy). So J/step = joules / 16000 overstates the training step slightly: `eval_seconds` is 147.5 s (hybrid), 44.8 s (zero-heads), 36.7 s (transformer), about 0.6-0.8 % of each arm's wall-clock [derived]. The ratios quoted above are unaffected at the precision given.
- Per 1000 steps the hybrid is steady (1074.9 s at step 1000, 2173.4 at 2000, 3273.7 at 3000), so no slowdown over the run.
- The zero-heads arm is not reported in the notebook (Runs 0-49). Its runner was added 2026-09-19 as "the missing ablation" (commit a02c7b1) and had no result before the spine runs (spec line 36).
- Contention flag false for all three.

---

## 3. What is trained and what is frozen in the current best design

Flags: `--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4`, plus the common recipe of `run_capacity.sh` / `run_spine.sh`: d_model 512, 8 layers, d_ff 1024, n_key 4096, top_k 4, claws 4 (default, `run_sparsela.py:207`), theta init 1.0 (default, `:209`), `--normalize`, `--key-conv 2`, decay 1.0, `--chunk 256`, batch 8, seq 2048, lr 6e-4, warmup 1000, cosine, wd 0.1. Built by `build()` (`run_sparsela.py:49-59`) into `GRUHybrid` (`hybrid.py:266`). Counts below were produced on CPU and sum to **16,066,824** = `trainable_scalars` in the spine JSON.

### 3.1 Parameter groups

Everything that is a `parameter` has `requires_grad=True`; there is no explicit freeze. The only frozen state is the four `proj` buffers.

| group | where | count | frozen? | gets gradient? |
|---|---|---:|---|---|
| Embedding(256, 512) | `hybrid.py:309` | 131,072 | no | yes |
| 8 x minGRU layer | `hybrid.py:14-50` (ln1/ln2 `:19`, z `:20`, h `:21`, ff `:22`) | 12,615,680 (1,576,960 per layer: LN 2,048; `z` Linear(512,512)+bias 262,656; `h` Linear(512,512) no bias 262,144; FFN 512->1024->512 1,050,112) | no | yes |
| final LayerNorm `core.ln` | `hybrid.py:60` | 1,024 | no | yes |
| 1 local attention (after layer 4; 8 heads = d_model//64; window 512; RoPE) | `hybrid.py:103-112`, `:342-354` (`local_at = [4]`) | 1,051,648 (LN 1,024; qkv Linear(512,1536) 787,968; out Linear(512,512) 262,656) | no | yes |
| Decoder Linear(512,256), not tied | `hybrid.py:321` | 131,328 | no | yes |
| **4 memory heads**: top head `mem` (key_conv 2) after the final LN, inner heads `mems[0..2]` after layers 2, 4, 6 with key_conv 4, 2, 4 | `hybrid.py:327-339`, `:324`, `:337`, `:413-415`, `:382-383` | 2,136,072 total = 533,506 (key_conv 2) x 2 + 534,530 (key_conv 4) x 2; 13.3 % of the model | see below | see below |

One head (key_conv 2 / 4):

| piece | where | count | frozen? | gradient? |
|---|---|---:|---|---|
| `proj` buffer [4096, 512] fp32, +-1 with 4 distinct nonzeros per row, different seed per head (`seed + 1000*(i+1)`, `hybrid.py:337`), generated by `signed_expand_matrix` (`seqmixer.py:25-37`) | `sparsela.py:40` | 2,097,152 entries (8 MiB); 16,384 nonzeros | **frozen buffer**, not in `parameters()`, not in the optimizer | none (`requires_grad False`, `.grad` None) [CPU check] |
| `theta` [4096] per-unit threshold | `sparsela.py:41` | 4,096 | trained (no weight decay, `run_sparsela.py:82`) | yes, but only on the selected units: 599-851 of 4096 entries nonzero in one 512-byte sequence [CPU check] |
| `val` Linear(512,512), no bias | `sparsela.py:224` | 262,144 | trained | yes |
| `outs[0]` Linear(512,512), no bias | `sparsela.py:216` | 262,144 | trained | yes |
| key `conv` depthwise Conv1d(512,512,k,groups=512) | `sparsela.py:219` | 1,536 (k=2) / 2,560 (k=4) | trained | yes |
| `ln1` LayerNorm(512) | `sparsela.py:223` | 1,024 | trained | yes |
| `gate_tau` scalar | `sparsela.py:206` | 1 | trained | yes |
| `ln2` + placeholder FFN (`d_ff=1`: Linear(512,1), Linear(1,512)) | `sparsela.py:223`, `:225`; `d_ff=1` at `hybrid.py:328`, `:337` | 2,561 per head, 10,244 total | trained in name only | **never executed, never gets a gradient**: only `Block.forward` (`sparsela.py:299-301`) uses them and the hybrid calls `blk.memory` directly (`hybrid.py:368-370`, `:414-415`). Confirmed: 16,056,580 parameters get nonzero gradient and exactly these 10,244 get None [CPU check]; the repo's own test says "the heads' placeholder FFNs are unused" |

Totals: backbone 12,879,104 (80.2 %; embedding + 8 layers + final LN + decoder), local attention 1,051,648 (6.5 %), heads 2,136,072 (13.3 %). Frozen buffers 4 x 8 MiB = 32 MiB. Per-sequence state (not a parameter): `S` [B, 4096, 512] fp32 and `Ksum` [B, 4096] per head.
Optimizer: AdamW, weight decay 0.1 on tensors with dim >= 2 (includes embedding and conv weights), none on biases, LN, `theta`, `gate_tau` (`run_sparsela.py:78-83`); `clip_grad_norm_(params, 1.0)` each step (`:298`).

### 3.2 The head's forward pass: what is inside the autograd graph

Path: `GRUHybrid._head` (`hybrid.py:368-370`) or the top-head lines (`:413-415`) -> `Block.memory` (`sparsela.py:230-251`) -> `_one_head` (`:256-260`, fp32 pin) -> `_one_head_chunked` (`:253-254`) -> `sparse_memory_read` (`:135-197`).

| step | file:line | in autograd graph? | note |
|---|---|---|---|
| input LN | `hybrid.py:369` / `:414`; module `sparsela.py:223` | yes | trainable |
| key source and key conv | `sparsela.py:232`, `:234-240`; conv `:219` | yes | key source = the head's own LN'd input; bf16 conv output under `--amp` |
| value projection | `sparsela.py:224`, `:241` | yes | bf16 under `--amp` |
| fp32 pin | `sparsela.py:259-260` | n/a | `.float()` on xk and v, autocast off |
| **address selection**: dense matmul with `proj`, `relu(z - theta)`, top-k indices, in slices of 512 positions | `sparsela.py:89-94` (`code_slice = 512`, `:80`) | **NO** (`torch.no_grad()`) | discrete: which units fire is not differentiable |
| address values: gather the k selected frozen rows, recompute z for those k units, `val = relu(z_sel - theta[idx])` | `sparsela.py:95-97` | **yes**, wrt xk and wrt `theta[idx]` | gradient reaches the key source only through the magnitudes of the already-selected units, never through the selection |
| unit-normalise the code | `sparsela.py:147` | yes | |
| key shift (write key = previous position's key) | `sparsela.py:150-155` | yes (values); indices are integers | slot 0 reads nothing |
| intra-chunk scores by index equality, causal mask | `sparsela.py:166-171` | `eq` boolean: no; products of `val`: yes | decay is the constant 1.0 |
| inter-chunk read: gather rows of `S` and `Ksum` | `sparsela.py:175-179` | yes, wrt `S`, `Ksum` | gradient flows back into earlier chunks' writes |
| **normalizer** | `sparsela.py:181-182` (`out / (mass + 1e-3)`) | yes | |
| **match gate** | `sparsela.py:183-184`; `gate_tau` `:206` | yes | trained scalar |
| **write** (outer product into the state) | `sparsela.py:187-190`: `S = S * lam ** Lc + zeros_like(S).index_add(0, wflat, (vk_aged[..., None] * vc[:, :, None, :]).reshape(-1, d))`, same for `Ksum` | **yes** | BPTT across the 8 chunks of a 2048 window; this is the path Run 47's bug cut |
| state quantiser | `sparsela.py:191-192`; `fake_quant` `:121-132` | straight-through | off in the best-design flags |
| output projection and residual add | `sparsela.py:216`, `:245`; `hybrid.py:370` / `:415` | yes | bf16 under `--amp` |

Short version: everything in the head is differentiable except two discrete operations, the top-k selection of which slots fire and the index-equality mask. The read, normalizer, gate, write and state carry are all ordinary autograd, with gradient flowing through the state across chunks.

### 3.3 The minGRU scan (`hybrid.py:26-50`, `MinGRULayer.scan`)

- Recurrence h_t = (1 - z_t) h_{t-1} + z_t cand_t with `z = sigmoid(Linear(u))` (`:30`) and a **linear** candidate `cand = Linear(u)` (`:30`), `u = ln1(x)` (`:29`). No log-space (Heinsen) scan of the state; no associative scan; no custom kernel; no `torch.compile`.
- `a = (1 - z).clamp_min(1e-4)`, `b = z * cand` (`:31`). Under `--amp` these start as bf16-rounded values [CPU check]. Padded to a multiple of 32 with the identity (a=1, b=0) (`:36-37`); chunk length `L = 32` (`:24`).
- **float64 from `:39`** (`.double()`) through `:47`; the carry is fp64 (`:43`, state returned `h[:, -1].clone()` fp64, `:50`); cast back with `h.to(x.dtype)` at `:48` (fp32 residual stream; bf16 only in a whole-model bf16 cast).
- Inside each 32-step chunk: cumulative product in **log space**, `A = exp(cumsum(log a))` (`:40`); then the closed form `local = A * cumsum(b / A)` (`:41`), which is linear space (division by A, not a log-space accumulation of b). It is safe only because the chunk is short and `a >= 1e-4` (A >= 1e-128 stays inside fp64's range; this is why it is fp64). The record says the same: "chunked closed-form scan in float64" (N:873), "chunked fp64 scan" (H:55).
- Across chunks: a Python for-loop over the `nC - 1` = 63 chunk boundaries (T=2048), each step an elementwise fp64 update on a `[B, d]` tensor (`:44-45`: `carries.append(A_end[:, i] * carries[-1] + local_end[:, i])`), then `h = A * h0s + local` for all chunks at once (`:47`). All chunks are processed in one batched `[B, nC, L, d]` fp64 tensor (8 x 64 x 32 x 512 = 8.4M doubles, 67 MB per live tensor) [derived].
- Autocast leaves fp64 tensors alone, so the scan itself is the same with or without `--amp`; only its inputs `z` and `cand` differ (bf16-rounded under `--amp`, fp32 otherwise).

### 3.4 Other facts relevant to the speed round

- Data: `batch_windows(train, ...)` samples windows on CPU each step and `.to(device)` (`run_sparsela.py:292-293`); its cost is not measured in the record.
- Transformer baseline (`hybrid.py:193-228`, `RopeAttention` `:177-190`): pre-LN, RoPE, `F.scaled_dot_product_attention(..., is_causal=True)`, heads = d // 32 = 16 at d=512, 17,085,696 params; the lab never wrote a fused or compiled variant of it either.
- Head placement: one inner head after every second layer except the last (`n_inner = (n_layers - 1) // mem_every`, `hybrid.py:334`; here layers 2, 4, 6) plus the top head after the final LayerNorm; the local layer sits after `n_layers // 2` = layer 4 (`hybrid.py:348`).

---

## 4. Proposed in the record and never run (limited to training speed or learning without gradients)

**Sources checked for proposals:** `H:87-89` ("Next test"), `H:13-15` (killed runs), `N:1651-1673` ("Next"), `R:171` ("Unknown"), asides inside Runs, `D:` (V0 era), `F:58-60`.

**The notebook's closing "Next" section (N:1651-1673) is stale.** Its items date from Runs 18-28 (scale to d=256 / 50 MB / seq 2048, transformer backbone, real terminal transcripts, 2048 with 16k units / signed keys, multi-span at 512). All were done afterwards (Runs 34, 30, 32, 25 addendum, 23-24) except "real terminal transcripts" (not done; the synthetic terminal corpus was "uninformative", N:749). None bears on speed or gradient-free learning. H:89 "Next test" is about scale (enwik8 at 50M, a second 50M seed, 100M) and one speed item (item 3 below).

### Training speed

| # | Item | Where proposed | State in the record |
|---|---|---|---|
| S1 | Make `--amp` plus `--state-bits 16` the default for new runs | H:89 item (3) | The combination was only ever run with the buggy quantiser (0.784, 6935 s, N:1530). "**Not yet:** the two combined" (N:1543) with the STE fix: not in the record. The quantiser's own cost is visible but unexplained (fp32 + STE 7658 s vs fp32 6866 s; amp + buggy 6935 s vs amp 4919 s) |
| S2 | 50M under the amp recipe; unshared wall-clock at 50M | N:1543, N:1591, RM:68-69 | Run 49 was fp32 and shared; "wall-clock at 50M (both runs shared the GPU)" is listed under Not a claim |
| S3 | Transformer under amp | implied by N:1527-1534 (hybrid rows only) | not in the notebook (the spec, [outside], says "The transformer never ran under it") |
| S4 | 8-bit state with a scale tied to the row's own mass, or quantising the normalised read | N:1400 "Untested"; R:177 "fixable, untested" | not run |
| S5 | A trained mixed-precision layout (early heads 16-bit, late 8-bit) | N:1397, N:1401 (eval-only: 0.791 / 0.832) | Run 47 trained only uniform 16 and 8 |
| S6 | Integer weights; a trained-at-low-precision model; the count head's precision | N:1404, R:177 "Integer weights are untested here" | not run |
| S7 | Idle-GPU per-step timing | N:1084 "Run 36 will measure this on an idle GPU"; N:1375 "measure later" | Run 36 used a shared GPU (N:1103); local-512 cost re-measured only through Run 44 (848-1069 s vs base 719 s) |
| S8 | Unshared wall-clock for head counts other than 4 (1 / 2 / 8 heads) | N:1185 ("its step time") | all head-count timings are shared-GPU (N:1163-1183) |
| S9 | Chunk length at 2048 for speed (256 used throughout; 512 used for 32k / 64k "halves the per-chunk state copies") | `run_capacity.sh:30` comment [outside] | no sweep |
| S10 | Local window 128 (keeps three quarters of the gain, N:1462) as a cost lever at 20M | N:1462 | only at d=256 (Run 44) |
| S11 | Sparse (coincidence) block as a cheaper FFN at LM scale: "34 % fewer FLOPs" but "2-3x" slower wall-clock; open whether the late-training gap is an lr/schedule artefact or a top-k capacity ceiling | N:501, N:534, N:1669 | tried only at d=24-48 byte-LM scale (Runs 19-20); closed with "Do not scale it to claim a win" (N:1671) |
| S12 | Profile of any kind (head, scan, local attention, optimizer, data) | not proposed anywhere | none exists |
| S13 | TF32, `torch.compile`, fused kernels, sparse use of the claws, activation checkpointing | not proposed anywhere | never tried (section 2.3 item 6) |
| S14 | Long-window recency (decay < 1 in the chunked form) as the fix for the ~0.1 loss past 8k | N:1328, R:133 "recency weighting is the untested lever" | tested at 2048 only (Run 41) |

### Learning without gradients

| # | Item | Where proposed | State in the record |
|---|---|---|---|
| G1 | A reservoir slice: freeze the synthetic W, inject the current byte into all live neurons, pool all, train only a small readout vs a no-graph direct-input control | F:58-60, "Gate it behind an explicit ask; it is a new program" | never run |
| G2 | A deliberate test of how much gradient the memory write path needs (detach `S` at chunk boundaries or every k chunks) | not proposed | only the accidental Run 47 data point (+0.02, N:1537) |
| G3 | The credit-assignment hypothesis (gradient through F touches only k units per write) | N:618 | never isolated |
| G4 | A model trained to start from a loaded memory; commons / handoff across sequences | spec lines 49-53, 132-142 [outside] | the spec says nothing in the lab "has carried state across windows or shared memory across sequences" (spec line 42); the probe is "confirm-only" (spec line 142) |
| G5 | D:129-133 Tier-3 V0 ablations: residual update H <- H + GRU, drop sign-lock, LSTMCell / LRU at matched numel, attention readout over the population, lr sweep | D:127-133 | not run; the V0 graph line is closed (N:382, H:27-30) |

### Cross-lab note that closes one idea

N:1219: a sibling lab's low-precision dense-transformer deficit "is flat across match length rather than concentrated on repeatable bytes, so a recall head would not close it ... The merge idea is closed." Numbers were not recorded.

---

## 5. Places where the record conflicts with the framing in your first paragraph, or with itself

1. **Wall-clock.** The record's standing claim is a tie (N:1420, N:1447, RM:71, R:143, R:161: 6866 s vs 6657 s, fp32, unshared). Your 3x is only in `results/spine16_*.json` [outside]. Run 47's same-config amp hybrid (4919 s, 0.307 s/step, N:1532) is 3.6x faster than the spine hybrid (1.099 s/step); the record does not say why and has no power-limit field. The transformer's amp timing is not in the notebook. So the "28 % faster" number is hybrid-only.
2. **The cost is not only the heads.** zero-heads is 0.512 s vs the transformer's 0.369 s (1.39x); heads add 0.587 s (53 %) [outside, derived]. The fp64 scan (`hybrid.py:39-47`) and the local layer are in the backbone's excess, but nothing in the record isolates them.
3. **"Pinned to fp32" is a precaution, not a finding** (section 2.5). Run 42 ran the head in bf16 (whole-model cast, before the pin existed) and lost nothing; the record has no bf16-head training run. The pin also made Run 42's `--bf16` eval path unrunnable at HEAD.
4. **"16-bit state" is simulated int16, not 16-bit float storage**, and it costs time (section 2.2).
5. **"Frozen random expansion"** is accurate for `proj` but: theta (4096 per head, 16,384 total, 0.10 %) is trained (R:112); claws are not needed per Run 23 (N:636) yet are what ships; the "expansion" is applied as a dense matmul (`sparsela.py:92`), costed in the lab's FLOP formula as if sparse (`hybrid.py:400`); and which units fire is outside autograd (`sparsela.py:89-94`).
6. **"Beats a transformer per training step" needs both halves.** 4 heads alone: 0.811 / 0.814 (N:1415) vs transformer 0.781 / 0.789. Local attention without heads: 0.808 [outside, spine] vs transformer 0.783. Combined: 0.764 / 0.762 (N:1417); spine 0.7613. Each half is worth about 0.047. At 20M the local layer's gain lands on unmatched bytes (N:1468, N:1475), the heads' on matched bytes.
7. **Sensitivity to GPU sharing is lopsided** (section 2.2): hybrid 18943.7 s shared vs 6866.2 s unshared (2.76x), transformer 11861 s vs 6657 s (1.78x). Not explained in the record.
8. **Internal inconsistency:** H:72 still lists "4-head ~11000 s" which N:1420 withdraws; `run_50m.sh:4` still says amp + 16-bit state "gave no speedup" and "cost 0.02", which Run 47 attributes to a bug; R:181 says sign constraints and structural plasticity were "all tested" while D:130 lists the sign-lock ablation as not run and N:1613 says V1 plasticity was plumbing only.
9. **A `feat/speed-round` branch exists** (HEAD 3cee1fe, 2026-10-04 07:33, "Speed round: the list of three directions to test, with the bar and the expectations") [outside, `git branch`]. I did not open it, so I cannot say how its three directions overlap with the rows above.

---

## 6. Not in the record (explicit)

- Any per-op profile or time split inside the head, the scan, the local layer, the optimizer or data loading.
- The power limit, power draw or energy of any run in Runs 0-49.
- A transformer timed under bf16 autocast in the notebook.
- The head run in bf16 or fp16 during training; the head's step time in any precision; a bf16 / fp16 state storage format; TF32.
- `torch.compile`, fused AdamW, custom kernels, a sparse implementation of the claws.
- Unshared timings for 1 / 2 / 8 heads; unshared 50M timings; timings at 8k-32k windows for the hybrid or the transformer.
- A hybrid with Gaussian (non-claw) keys at LM scale; learned vs frozen theta at LM scale; a V0 sign-lock ablation.
- The zero-heads arm in notebook.md.
- A deliberate gradient truncation through the memory state; any weight learning without gradients; the credit-assignment hypothesis isolated.
- Why the Run 47 amp hybrid (0.307 s/step) and the spine hybrid (1.099 s/step) differ by 3.6x.
