# Experiment integrity audit and rerun tracker

> **Commit hash note (2026-09-13):** the corrected-source commit cited below as `5567c51`
> (full `5567c51d4c7e370183016532b878c1952f0d320e`) was rewritten to `1c6f3b6` before the push,
> changing only the author identity; the tree is identical (`79a2c497…`). Run manifests under
> `runs/` keep the original hash. Likewise the follow-up commit `a33b508` became `948ff62`
> (tree `dd204d2c…`).

Updated during the 2026-09-13 UTC dual-GPU trial. User authorized auditing all affected results and rerunning affected comparisons.
Keep this file current across sessions. Do not overwrite historical reports or run directories.

**Current status: complete.** All seven training runs finished (text8 at 2026-09-13 05:39:52 UTC;
follow-ups at 11:00:46 UTC). Artifact audit, both 1B generation modes, and the 1B exact split/resume
gate passed; results are written up in [provenance reruns](2026-09-13-provenance-reruns.md).
Earlier RUNNING/QUEUED entries below are historical snapshots, superseded by this status.

## Scope and decision rules

- Frozen-weight exposure: `weight_mode=frozen` after fused update commit `d11fe888` and before fix `5567c51`.
- Resume exposure: legacy continuation without verified selected seed, tokenizer identity, stochastic RNG,
  or pressure-leak phase. A missing provenance record is uncertainty, not proof of a wrong result.
- A fixed-scale baseline named `frozen5` is not necessarily a frozen-weight control. Inspect the protocol.
- Fresh uninterrupted trainable/QAT runs and isolated kernels are not invalidated merely by their date.
- Rebuild all comparison arms on corrected source when replacing an exposed comparison.
- GPU 1 only; preserve existing workloads, failed runs, old artifacts, and original token budgets.
- Record source/config/data/tokenizer hashes, run seeds, commands, final metrics and tensor checks.

## Work queue

- [x] Core Shakespeare: 15 fresh runs complete; see [report](2026-09-12-shakespeare-rerun.md).
- [x] Classify all 26 historical result reports against the two defect paths (table below).
- [x] Recover historical 1B config from `528946d`; queue a fresh replacement for missing launch provenance.
- [x] Text8 30k states curve: reconstruct protocol; old report explicitly resumes FP32/5/7 from 12k.
- [x] DONE: fresh text8 four-arm replacement.
  Rerun all four arms (FP32, codes 5/7/9), seed 1337, from scratch with corrected checkpoints.
- [x] Review other result reports: no additional confirmed defect-triggering runs found.
- [x] DONE: subword seed-1338 control and treatment; the later result commit lacks launch history.
- [x] DONE: historical 1B batch-96 screen; launch/resume provenance unavailable.
  These are precautionary replacements, not findings of numerical corruption.
- [x] Publish replacement result notes: [provenance reruns](2026-09-13-provenance-reruns.md).

## Inventory

Status `AUDIT` means not classified yet, not a finding of invalidity.

| Historical report | Exposure / evidence | Action / status |
| --- | --- | --- |
| [2026-06-20-controls](2026-06-20-controls.md) | Literal frozen controls recorded at `fd3900a` before `d11fe88`; six new frozen runs passed bitwise state checks. | DONE fresh corrected-code study; historical record predates bug |
| [2026-06-20-scaleup-25m](2026-06-20-scaleup-25m.md) | Pre-regression fresh FP32/ratchet arms; dropout enabled but no documented resume. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-20-smoke](2026-06-20-smoke.md) | Fresh CPU `lat compare`; pre-regression; no continuation. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-20-text8-states-curve](2026-06-20-text8-states-curve.md) | Explicit 12k→30k resume for FP32/5/7; seed 1337 stated but original checkpoint provenance unavailable. Dropout/leak off, char tokenizer; no identified RNG corruption mechanism. | DONE precautionary four-arm replacement; see [provenance reruns](2026-09-13-provenance-reruns.md) |
| [2026-06-20-tiny-shakespeare-full](2026-06-20-tiny-shakespeare-full.md) | Trainable arms recorded at `840bdfc` before regression; fresh 15-run replacement also complete. | DONE fresh corrected-code study; historical record predates bug |
| [2026-06-20-trainable-scale](2026-06-20-trainable-scale.md) | Pre-regression fresh trainable-scale arms; codes intentionally update. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-21-direction2-memory-decomposition](2026-06-21-direction2-memory-decomposition.md) | Fresh in-process memory probes; no checkpoint load or frozen mode. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-21-eager-throughput](2026-06-21-eager-throughput.md) | Fresh short learning-mode benchmarks; no continuation; inference discussion is not a frozen training control. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-21-forward-kernel-prototype](2026-06-21-forward-kernel-prototype.md) | Forward-only kernel microbenchmark; neither defect path. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-21-int8-activation-spike](2026-06-21-int8-activation-spike.md) | Activation/kernel precision and speed probes; neither defect path. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-21-int8-convergence-25m](2026-06-21-int8-convergence-25m.md) | BF16/int8 ratchet launches from scratch; int8 stopped at 5k. Documented commands have no resume. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-21-int8-tuned-kernel-reversal](2026-06-21-int8-tuned-kernel-reversal.md) | Bare GEMM/fused-linear/backward microbenchmarks; no continuation or frozen mode. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-21-packed-memory-scaling](2026-06-21-packed-memory-scaling.md) | Fresh three-step FP32/ratchet memory runs; intended updates. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-21-packed-training-memory-probe](2026-06-21-packed-training-memory-probe.md) | Fresh single/few-step memory probes; activation checkpointing is not training continuation. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-21-tuned-int8-gemm-bench](2026-06-21-tuned-int8-gemm-bench.md) | Standalone GEMM benchmark; neither defect path. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-22-corrected-memory-sweep](2026-06-22-corrected-memory-sweep.md) | Fresh three-step memory sweep and observability probes. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-22-int8-backward-working-set](2026-06-22-int8-backward-working-set.md) | Kernel/memory working-set probes; intended ratchet updates. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-22-int8-training-throughput](2026-06-22-int8-training-throughput.md) | Script constructs fresh seeded model and fixed random tokens; no checkpoint continuation. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-22-qat-deconfounding](2026-06-22-qat-deconfounding.md) | Explicitly all seven arms fresh 0→30k under one HEAD; modes FP32/QAT/ratchet. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-23-adaptive-scale-ratchet](2026-06-23-adaptive-scale-ratchet.md) | Fixed-scale baseline, not frozen weights; new trainable-scale arms and fresh QAT-study references. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-23-update-rule-momentum](2026-06-23-update-rule-momentum.md) | Plan driver explicitly uses `--weight-mode ratchet` even for `frozen5`; fresh 5k screens. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-24-int8-per-token-speed](2026-06-24-int8-per-token-speed.md) | Fresh throughput probes and separately launched 5k convergence arms; no documented continuation. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-24-int8-training-throughput-final](2026-06-24-int8-training-throughput-final.md) | Fresh throughput/memory probes; neither defect path. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-25-projection-oracle-state-count](2026-06-25-projection-oracle-state-count.md) | Fresh training screens; checkpoint projection is evaluation only, not resumed training. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-26-ratchet-embedding-ab](2026-06-26-ratchet-embedding-ab.md) | Spec explicitly requires both arms fresh, replacing an earlier resumed control; published A/B is fresh. | No defect-triggered rerun indicated by documented protocol |
| [2026-06-26-subword-sparse-embedding-ab](2026-06-26-subword-sparse-embedding-ab.md) | Seed 1337 plan launches fresh; seed 1338 added at `19b745e` without launch history, replacing a note about failed/interrupted attempts. | DONE seed-1338 control+treatment rerun; see [provenance reruns](2026-09-13-provenance-reruns.md) |

## Execution handoff

- Checkout: `/games/ailab/local-ai-training`, branch `fix/experiment-integrity`.
- Corrected training source commit: `5567c51d4c7e370183016532b878c1952f0d320e`.
- Python: `.venv/bin/python`; CLI: `.venv/bin/lat`.
- Existing new evidence: `runs/integrity-replication-2026-09-12/`.
- No historical experiment artifacts found in this fresh clone; a bounded directory search under
  `/games/ailab` found no second `local-ai-training` checkout or original run directory.
- Preserve uncommitted Shakespeare report and docs index changes from the prior task.

## Additional provenance gaps outside result reports

- [x] **1B feasibility protocol recovered; rerun QUEUED, not complete:** historical summary claims batch 96,
  LR 0.00075, 5k steps and validation 2.3648; today's named config has batch 224, LR 0.005,
  and a different token budget. Do not launch the current config as a replication. No resume is
  documented in the result, while the handoff requests a save/resume gate. Historical config recovered from `528946d` into
  `runs/integrity-followups-2026-09-12/historical-1b.toml`. It matches batch 96, LR 0.00075,
  5,000 steps and 122,880,000 sampled tokens. No evidence proves that the old run actually resumed.
- The embedding spec mentions an earlier resumed byte-level control; it explicitly replaces that
  control with fresh runs for the published A/B. No extra rerun for that discarded control.
- Packed-storage plans mention in-flight 60k runs, but there is no corresponding published result
  in the result inventory. This is not evidence that any listed 30k study continued to 60k.

## Audit limits and references

No published result is proven numerically corrupted by the identified defects. The text8 rerun
closes a known legacy-continuation provenance gap. The later QAT study already reran the same
state-count arms fresh, providing separate historical evidence; the new run adds corrected v2
checkpoint provenance. Dates alone do not invalidate studies.

The original artifacts are absent, so 'no rerun indicated' means the documented protocol avoids
the defect paths, not that this audit independently replayed each historical launch.

Key evidence: [integrity mechanics](2026-09-12-experiment-integrity.md),
[QAT fresh-run protocol](2026-06-22-qat-deconfounding.md),
[explicit ratchet-mode momentum driver](../superpowers/plans/2026-06-23-update-rule-momentum.md),
[fresh embedding requirement](../superpowers/specs/2026-06-26-ratchet-embedding-design.md).

## Active execution

- Output: `runs/integrity-text8-30k-2026-09-12/`.
- Driver: `run.py`; exact active command and hashes: `manifest.json`; status: `study.log`.
- Process tool session: `55257` (may not survive tool-session changes; inspect files before acting).
- Serial order: FP32, quinary, septenary, nonary. Seed 1337; 30,000 steps and
  491,520,000 sampled training tokens per arm (1,966,080,000 total).
- All launch directories must be absent. Do not rerun the driver against this directory: it refuses
  an existing manifest. If interrupted, inspect the v2 checkpoint and recorded command before
  deciding whether to resume the same arm; never quietly restart or discard it.
- At the last inspection FP32 had reached step 2,600 with finite loss. Full comparison is **not complete**.
  This is a historical progress snapshot; read `fp32.log` and `manifest.json` for current progress.

## Follow-up queue and review checklist

- Output: `runs/integrity-followups-2026-09-12/`; driver `queue.py`; live status `status.json`;
  log `queue.log`. Tool session `82077`. The initial waiting-only supervisor was restarted to add GPU
  availability and source-file hash checks; no training run was restarted. Its initial status/log
  are preserved as `status-initial-wait.json` and `queue-initial-wait.log`.
- The queue waits for the four text8 arms to finish successfully, then runs the two subword arms
  and historical 1B serially. It stops on a detected training failure and preserves logs.
- Both subword arms: seed 1338, codes 15, RMS EMA 0.9, no pressure leak, 30k steps,
  batch 64/context 256, LR 0.0003, pinned enwik8, one shared 8K BPE artifact.
  Only the treatment enables ratchet input embeddings. Each arm sees 491,520,000 tokens.
- 1B: seed 1337, codes 15, RMS EMA 0.9, ratchet embedding, historical config above;
  int8 forward, compiled updates, gradient checkpointing. No quiet batch/LR reductions on OOM.
- All seven queued runs together sample 3,072,000,000 training tokens.
- enwik8 and text8 downloads completed via the repository's SHA256-verified dataset helper.
  Original BPE artifact is unavailable; rebuilt tokenizer identity is recorded. Do not call
  this an exact historical-tokenizer replication or compare tiny cross-run deltas as causal.
- At launch: current source `5567c51`, GPU 1 free about 23.8 GiB. GPU 0 is not used.
- [x] After jobs finish, inspect terminal states and all complete metrics/checkpoints.
- [x] Compare text8 best/final losses at matched budgets (old nonary ended at 29,400, new at 30,000).
- [x] Compare subword last-four-evaluation means and shared tokenizer, verify master-weight audit.
- [x] For 1B, inspect loss trajectory, tensor audit, generation, and separately validate the requested
  save/resume gate before promoting to longer training. Training completion alone does not close that gate.
- [x] Write replacement results notes, link here, and update roadmap claims from actual evidence.
- [x] Mark this queue complete only after the analysis/gates above, not merely process exit.

Read-only progress command:

```bash
.venv/bin/python - <<'PY_STATUS'
import json
from pathlib import Path
for directory, name in [('integrity-text8-30k-2026-09-12', 'manifest.json'),
                        ('integrity-followups-2026-09-12', 'status.json')]:
    path = Path('runs') / directory
    state = json.loads((path / name).read_text())
    print(directory, state.get('state', state.get('active_arm', 'completed')))
    for log in sorted(path.glob('*.log')):
        lines = log.read_text(errors='replace').splitlines()
        print(log.name, lines[-1][:200] if lines else '(preparing)')
PY_STATUS
```


## Recovery after first text8 attempt

FP32 completed 30,000 steps: final validation 0.97290815, best 0.97272065.
Quinary failed before a checkpoint with CUDA OOM while another process held 20.28 GiB.
The dependent queue stopped. Subsequent live check found GPU 1 free (23,798 MiB).
Failed quinary directory/log and failure manifest are preserved; `recover.py` retains FP32
and retries quinary in `quinary-retry1`, followed by septenary/nonary with unchanged settings.
The follow-up queue is restarted after recovery; its failed status/log are archived.

## Monitoring upgrade (user authorized GPU 0 as well)

User requested active monitoring after GPU contention stopped a run. The selected policy keeps
one training job active at a time: repository history records a prior dual-3090-load crash.
GPU 1 remains preferred; GPU 0 may be selected if GPU 1 lacks headroom.

- Monitor helper: `runs/integrity-monitor-2026-09-13/gpu_guard.py`.
- Require three qualifying GPU-memory samples before each launch: 8,000 MiB free for 25M
  arms, 22,000 MiB for the historical 1B arm. Admission fails closed if telemetry is unavailable.
- A shared process lock prevents these monitored jobs from overlapping. It does not reserve
  hardware against unrelated programs; another program can still allocate VRAM mid-run.
- Heartbeats record actual selected physical GPU, managed PID, latest log progress, and other
  compute processes. Nonzero exits remain failures; no silent hyperparameter changes or retries.
- OOM recovery remains evidence-preserving: inspect the failed attempt, free device capacity,
  and v2 checkpoint compatibility before restarting/resuming.
- Text8 driver after handover: `monitored-recovery.py`; follow-ups: `monitored-queue.py`.
  The active quinary child is adopted without restarting training.
- GPU identifiers in errors are CUDA-local: local GPU 0 can mean physical GPU 1 after
  `CUDA_VISIBLE_DEVICES=1`. Use monitor UUID/index mapping for attribution.

Monitoring verified live: nine admission/parser/process-liveness tests passed. A CPU-only
live probe selected GPU 0 when GPU 1 lacked the 1B headroom threshold. Two live heartbeats
confirmed continued observation of original quinary PID 1512661 without a training restart.

Current supervisor sessions: monitored text8 `54355`; monitored follow-ups `1299`.
The old text8 supervisor PID 1512648 is deliberately paused to prevent duplicate launches;
new supervisor removes it after the adopted child exits. Do not resume that old supervisor.
Live status: `runs/integrity-monitor-2026-09-13/status-gpu-{0,1}.json`; historical GPU
selection/progress heartbeats retained in `heartbeats.jsonl` by `journal.py`.
Monitoring detects and records a failure but does not silently retry it, kill unrelated
processes, reserve all VRAM, or guarantee protection from new external allocations.

## Current execution policy: dual-GPU trial (supersedes serial policy above)

User explicitly authorized trying both cards after live inspection confirmed the existing caps.
No watt-limit service or Fleet configuration was changed.

- GPU 1: text8 septenary (existing PID 1592544, adopted without restart), then nonary.
  Driver `runs/integrity-text8-30k-2026-09-12/dual-recovery.py`; tool session `5314`.
- GPU 0: subword seed-1338 control, treatment, then historical 1B. Driver
  `runs/integrity-followups-2026-09-12/dual-queue.py`; tool session `37974`.
- Active thermal helper: `thermal_guard.py` (old `gpu_guard.py` preserved). Per-GPU locks
  and status files prevent these queues from overlapping on a card. No GPU fallback in this mode.
- Admission requires the memory reserve, three samples below 80 C, and cap <=300 W on GPU0
  or <=250 W on GPU1. Each job is monitored during execution. 80 C warns; 85 C stops only
  that managed PID after checking its process start time. Persistent missing telemetry for
  60 seconds also stops the managed child. TERM first, KILL after 30 seconds if necessary.
- Fleet's own temperature controls do not cover these directly launched jobs, hence this helper.
  NVIDIA memory-junction temperature is unavailable here; these thresholds use GPU core temperature.
- A stable pre-trial septenary checkpoint at step 3800 was copied to
  `runs/integrity-monitor-2026-09-13/pre-dual-septenary-step-3800/`.
- Old supervisor PID 1526184 is paused solely to preserve the current training child; the new
  supervisor removes it after that child terminates. Do not resume the old supervisor.
- Retired `queue.py` shadowed Python's standard-library module when launching on GPU0; moved
  unchanged to `legacy-supervisors/queue.py`. The pre-training launcher error is retained in
  `dual-queue-import-failure.log`; no training attempt was lost to that import error.
- Thermal guard plus original monitor tests: 20 passed. Includes adopted-process thermal
  protection, PID reuse refusal, per-GPU lock names, admission caps/temperatures, and bounded log reads.
- Full metrics and scientific review remain pending; starting both jobs is not completion.

### Initial dual-load observation result

Eleven samples spanning 300 seconds are retained in
`runs/integrity-monitor-2026-09-13/dual-trial-observation.json`. Both queues advanced throughout.
GPU0 core temperature ranged 77–79 C at approximately 299 W; GPU1 stayed at 75 C at
approximately 250 W. No thermal warning/stop event or matching kernel GPU-fault record was
observed during this window. Both queues remain running; this is an initial load observation,
not proof the entire multi-hour rerun has completed or a guarantee against future hardware faults.

## Training completion — 2026-09-13

All seven final v2 checkpoint metadata records are present with the expected step counts.

| Arm | Final validation | Best validation / last-four mean |
| --- | ---: | ---: |
| Text8 fp32 | 0.972908 | 0.972721 (best) |
| Text8 quinary | 1.236806 | 1.225254 (best) |
| Text8 septenary | 1.156952 | 1.151153 (best) |
| Text8 nonary | 1.104519 | 1.101661 (best) |
| subword-control-seed1338 | 2.450586 | 2.452436 (last-four mean) |
| subword-ratchet-seed1338 | 2.419195 | 2.418420 (last-four mean) |
| historical-1b-seed1337 | 2.368821 | 2.375096 (last-four mean) |

Subword seed-1338 control-minus-ratchet mean gap: 0.034017 nats. Do not conflate
training completion with the pending generation/resume and final scientific review gates.

## Final gates — 2026-09-13

- Artifact audit: PASS for all seven runs (`runs/integrity-final-checks-2026-09-13/audit.md`).
  Found the RMS-EMA byte-counter undercount; fixed in source after the resume gate finished.
- 1B generation gate: PASS, FP32 and native int8 paths (`generation-results.json`).
- 1B exact split/resume gate: PASS (120 model, 51 optimizer, 2 RNG tensors bitwise equal; metadata and metrics identical). Default-backend attempts failed because the
  flash SDPA backward is not repeatable at head size 128 (root cause and probes in
  [provenance reruns](2026-09-13-provenance-reruns.md)); fixed with the opt-in
  `deterministic_attention` setting and the gate rerun with it on every branch.
  The first attempt (`resume_checks.py`, GPU 1) finished its `continuous` branch and was killed
  mid `split-a` when the Codex session was interrupted at 12:15 UTC; its status is archived as
  `resume-status.interrupted-12-15Z.json`. A second GPU 1 attempt at 12:25 UTC never admitted:
  the standing qwen-server router had loaded a 20.7 GiB model there. The check was moved to GPU 0
  by `resume_checks_continue.py`, rerunning `continuous-gpu0`, `split-a`, and `split-b` on the
  same card so the comparison stays same-GPU; the GPU 1 `continuous` branch is retained for a
  cross-GPU determinism comparison.
