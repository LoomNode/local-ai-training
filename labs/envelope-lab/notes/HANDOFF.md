# Handoff — envelope-lab / AILab (read this when returning)

**Last updated:** 2026-08-02 (~19:07 CDT)  
**Purpose:** enough context to resume without replaying the whole chat.

---

## What this lab is

**AILab** = unified local AI lab on a dual RTX 3090 box.

| Piece | Role |
|-------|------|
| `/games/ailab/ailab-cli/` | `ailab` CLI — one-pane console + scriptable commands |
| `/games/ailab/envelope-lab/` | Research subject: train small LMs under a **hardware envelope** |
| `/vault/ailab/envelope-lab/` | Cache, runs, journal, `runs.jsonl` (bulky artifacts) |
| `:8082` qwen-server | Local OpenAI-compat router; default chat model **KAT-Coder** on **GPU 1** |

**Not** trying to beat KAT at coding in this module. Trying to find **hardware-efficient architectures** (parameter reuse / density) that fit **one 3090** and clear a speed floor.

Lineage: Karpathy **autoresearch**, **fractalsearch** (AGENT.md, frozen harness, fixed budget, runs log), plus your theses on **parameter folding** and **tokens-as-parameters**.

---

## Hardware policy (important)

| Mode | Meaning |
|------|---------|
| **`solo` (default)** | **GPU 1 only** — dedicated AI card |
| **`dual` (turbo)** | **Both GPUs** — train on GPU 0, KAT/AI can stay on GPU 1 |

- Human toggles mode (`--mode solo|dual` / `ENVELOPE_MODE`).  
- Campaign 1 was intended **solo-first**.  
- **Conflict:** KAT fully loaded on GPU 1 (~23 GB) leaves ~no room for train. Solo research nights need **`systemctl --user stop qwen-server`** (or unload KAT), then start it again after.  
- GPU 0 often has display / other light use; dual train targets 0.

---

## Day-to-day workflow

```text
export PATH="/games/ailab/ailab-cli/bin:$PATH"
ailab                          # one pane (not five terminals)

  bare text  → chat with KAT (advisor only; does not run tools)
  /status /goals /metrics /results
  /cue [N]   → queue experiments FROM this chat (no train yet)
  /queue     → list pending
  /loop --mode solo --budget 300   → run queue (default = queue only)
  /loop --auto ...                 → free invent (optional)
  /trial ...                       → one-off train with current knobs
```

Scriptable:

```bash
ailab status
ailab queue list
ailab loop --mode solo --budget 300
ailab loop --auto --trials 5 --mode solo --budget 300
```

**Division of labor:** human = GPU mode + `/cue` + `/loop`/`/trial`. KAT = discuss + fill cues. Loop = apply knobs → train → keep/discard.

---

## Frozen vs editable

| Frozen (do not “game”) | Editable |
|------------------------|----------|
| `prepare.py` — data, eval, envelope limits | `train.py` knobs between `AGENT_KNOBS_BEGIN/END` |
| `lab_config.py` — solo/dual device routing | Architecture inside `train.py` (Block / MemoryBlock / future shared recurrent) |
| Time budget, tokenizer, val_bpb definition | `queue.jsonl`, `goals.md`, notes |

Envelope (in `prepare.py`): `MAX_VRAM_MB=22000`, `MIN_DECODE_TPS=15`, default train budget 300s.

---

## Architecture theses (read these)

| Doc | Thesis |
|-----|--------|
| [parameter-inefficiency-and-folding.md](parameter-inefficiency-and-folding.md) | Depth via **unique** layer weights is wasteful; reuse a **shared core** while **state/memory** changes |
| [tokens-as-parameters.md](tokens-as-parameters.md) | Shift capacity into **token-addressable** store + **small computer**; unify tokenizer-side capacity with weights |
| [TODO.md](TODO.md) | Concrete next tasks |
| [../CAMPAIGN.md](../CAMPAIGN.md) | Active experimental campaign definition |

**Campaign 1** only weakly tests the thesis (`USE_MEMORY_BLOCK` on still-stacked layers).  
**Real next test:** shared recurrent block (`N_UNIQUE × N_RECURRENCE`) + metrics for unique vs effective depth; then token-side capacity vs stack size.

**Not the same as:** quantization-only, Colibri expert paging, or “Mamba instead of Transformer” (orthogonal; can combine later).

---

## Campaign 1 (size efficiency, solo)

**Question:** Beat a plain small GPT on `val_bpb` under fixed 300s + envelope by architecture/capacity allocation (memory blocks, depth/width), not more wall-clock.

**Queue (seeded + run 2026-08-02) — FINISHED:**

| id | Idea | 300s result (approx) | status |
|----|------|----------------------|--------|
| base | plain DEPTH=6 N_EMBD=384 | bpb≈10.90 · 30.1M | discard* |
| m1 | memory + DEPTH=5 | bpb≈10.91 · 30.6M | discard* |
| m2 | memory + thin MLP | bpb≈10.90 · crash path | skipped/crash |
| m3 | plain deeper/narrower 8×320 | bpb≈**10.43** · best 300s | discard* |
| m4 | memory slots=32 MLP_MULT=3 | bpb≈10.84 | discard* |

\*All discarded vs obsolete **60s smoke keep (bpb≈6.92)** — **keep-logic bug**. Among fair 300s runs, **m3 (deeper/narrower plain) was best bpb**.

**When resuming Campaign 1 analysis:**
1. Fix budget-aware keep (same `TIME_BUDGET` only).  
2. Promote best 300s legal run (likely m3 or re-run base) to `train.best.py`.  
3. Then implement **shared recurrent** folding A/B.

**Corpus:** tiny Shakespeare via tiktoken GPT-2 (`prepare.py`) — fine for arch search, **not** a serious LM quality claim.

**Process state at handoff write:** campaign loop **done**; GPU1 free; `qwen-server` **inactive** (restart for KAT chat).

---

## Key paths

```text
/games/ailab/ailab-cli/bin/ailab          # entrypoint
/games/ailab/envelope-lab/train.py        # model + knobs
/games/ailab/envelope-lab/train.best.py   # last keep snapshot
/games/ailab/envelope-lab/prepare.py      # frozen law
/games/ailab/envelope-lab/queue.jsonl     # cued experiments
/games/ailab/envelope-lab/results.tsv     # keep/discard table
/games/ailab/envelope-lab/notes/          # theses + TODO + this file
/vault/ailab/envelope-lab/cache/          # train.bin / val.bin
/vault/ailab/envelope-lab/runs/           # run-*.log, journal.md, runs.jsonl, campaign1-loop.log
```

---

## Bugs / sharp edges fixed or known

| Item | Notes |
|------|--------|
| Chat “looping” on “hi” | Fixed: compact system prompt, `max_tokens=1024`, `enable_thinking=False` |
| KAT claiming it runs `/trial` | Prompt says KAT only chats; human runs slash commands |
| `tail -f` silent during train | Fixed for **new** trials: `python -u` / `PYTHONUNBUFFERED` in `run_trial.sh` |
| Keep vs mixed budgets | **Still broken** — see Campaign 1 note above |
| Solo + KAT loaded | OOM / no VRAM — stop qwen-server for solo train nights |

---

## Suggested resume checklist

1. `ailab queue list` / `cat results.tsv` / `tail journal.md` — see what finished.  
2. Fix budget-aware keep; set 300s plain as baseline keep if needed.  
3. `systemctl --user start qwen-server` if you want KAT on GPU1 again (after train done).  
4. Read `notes/parameter-inefficiency-and-folding.md` + `tokens-as-parameters.md`.  
5. Implement **shared recurrent** block; new campaign A/B under solo 300s.  
6. Optional: thin dashboard over `runs.jsonl` (fractalsearch-style spectator UI).

---

## One-line north star

> **Maximize quality under a single-GPU envelope by making each parameter do more work (reuse + state + token-addressable capacity)—not by stacking more one-use weight matrices or paging a giant unchanged stack.**
