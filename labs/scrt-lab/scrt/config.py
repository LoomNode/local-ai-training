"""Experiment and model configuration for the SCRT lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    """Architecture knobs. Param budget is derived from concrete modules."""

    vocab_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    block_size: int = 64
    dropout: float = 0.0

    # Baseline: stack of independent blocks.
    n_layer: int = 4

    # Looped / SCRT: shared core reused this many times.
    n_loops: int = 4
    # Fixed residual-slot bank (LoRA-style adapters applied per loop).
    n_slots: int = 8
    lora_rank: int = 4
    # Born-merge form: rank of *per-loop* residual on top of shared LoRA.
    # Kept smaller than lora_rank so most capacity stays in the shared part.
    tiny_lora_rank: int = 2
    # Number of residual slots bound to each loop at init (round-robin rest free).
    slots_per_loop: int = 1
    # Exact number of slots occupied at init, round-robin across loops. Overrides
    # slots_per_loop when set. Occupancy is a buffer, not a parameter, so varying
    # this changes how much of the bank is in use at *identical* param count —
    # which is the only way to ask whether slot capacity binds at all.
    init_occupied: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.tiny_lora_rank < 1:
            raise ValueError("tiny_lora_rank must be >= 1")


@dataclass
class TrainConfig:
    """Training / consolidation knobs."""

    seed: int = 42
    batch_size: int = 32
    seq_len: int = 64
    steps: int = 200
    lr: float = 3e-3
    weight_decay: float = 0.01
    eval_every: int = 50
    consolidate_every: int = 25
    # Functional-similarity threshold for merge candidates (cosine).
    merge_sim_threshold: float = 0.85
    # Residual energy below this (relative) → free the slot.
    residual_free_threshold: float = 0.05
    # v2 consolidate throttle
    max_merges_per_cycle: int = 2
    max_regrow_per_cycle: int = 2
    # Frees were the only unthrottled op in the cycle; capacity-run logs show
    # ~1.7 frees/cycle against a 2-slot regrow budget, which starved the bank.
    max_frees_per_cycle: int = 2
    # Cycles a slot must survive before it is eligible to be freed. A regrown slot
    # starts with up == 0 (zero function, zero importance), so without a grace
    # period it is freed on the next cycle no matter how useful it would become.
    min_slot_age_cycles: int = 5
    keep_free_slots: int = 2
    # When False, skip free_weak_slots / regrow (merge-only reorganization).
    consolidate_enable_free: bool = True
    consolidate_enable_regrow: bool = True
    # Probe batch count for functional signatures.
    probe_batches: int = 2
    # GPU 1 is the dedicated research card on this box.
    device: str = "cuda:1"
    # Which arm: baseline | looped | scrt | scrt_static | shared_residual | modulated
    # scrt_static = residual bank, no consolidate (ablation)
    # shared_residual / modulated = born-consolidated (no merge)
    arm: str = "scrt"
    log_every: int = 25
    results_dir: str = "results"
    run_name: str = "mvp"
    # Data: synonym | structured | shakespeare | wikitext2
    corpus: str = "synonym"
    corpus_length: int = 80_000
    max_chars: int = 1_000_000
    # Char corpora only: fold all but the N-1 most frequent chars into one UNK id.
    # None = use every distinct char (WikiText-2 raw has 1013, mostly rare unicode).
    char_vocab_cap: Optional[int] = None
    # Only run consolidate while step <= steps * this fraction (1.0 = whole run).
    # v2 late thrash: free/regrow every cycle after peak — default half-run for scrt.
    consolidate_until_frac: float = 1.0
    # Do not consolidate before step > steps * this fraction (0.0 = from the start).
    # Saturation experiments set this near mid-run so merges happen on the plateau.
    consolidate_start_frac: float = 0.0
    # Plateau detection (eval history): no improvement > epsilon for this many evals.
    plateau_patience_evals: int = 5
    plateau_epsilon: float = 0.002
    # Ignore early training when declaring plateau (frac of total steps).
    plateau_min_frac: float = 0.25

    def __post_init__(self) -> None:
        _arms = {
            "baseline",
            "looped",
            "scrt",
            "scrt_static",
            "shared_residual",
            "modulated",
            "shared_plus_tiny",
        }
        if self.arm not in _arms:
            raise ValueError(f"unknown arm: {self.arm}")
        if not 0.0 <= self.consolidate_until_frac <= 1.0:
            raise ValueError("consolidate_until_frac must be in [0, 1]")
        if not 0.0 <= self.consolidate_start_frac <= 1.0:
            raise ValueError("consolidate_start_frac must be in [0, 1]")
        if self.consolidate_start_frac > self.consolidate_until_frac:
            raise ValueError("consolidate_start_frac must be <= consolidate_until_frac")


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"model": asdict(self.model), "train": asdict(self.train)}


def resolve_device(device: str) -> str:
    """Map bare ``cuda`` → ``cuda:1`` (dedicated research GPU on this box)."""
    if device == "cuda":
        return "cuda:1"
    return device


def tiny_mvp_config(device: str = "cuda:1", steps: int = 200) -> ExperimentConfig:
    """Tiny smoke config (easy synonym data)."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=48,
            n_embd=64,
            n_head=4,
            block_size=64,
            n_layer=4,
            n_loops=4,
            n_slots=8,
            lora_rank=4,
            slots_per_loop=1,
        ),
        train=TrainConfig(
            steps=steps,
            device=resolve_device(device),
            batch_size=32,
            seq_len=64,
            consolidate_every=25,
            eval_every=50,
            corpus="synonym",
            corpus_length=80_000,
        ),
    )


def stepped_config(
    device: str = "cuda:1",
    steps: int = 3000,
    *,
    match_params: bool = True,
) -> ExperimentConfig:
    """v1 stepped config (kept for comparison)."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=64,
            n_embd=128,
            n_head=4,
            block_size=128,
            dropout=0.0,
            n_layer=6,
            n_loops=8,
            n_slots=16,
            lora_rank=8,
            slots_per_loop=1,
        ),
        train=TrainConfig(
            seed=42,
            batch_size=64,
            seq_len=128,
            steps=steps,
            lr=1e-3,
            weight_decay=0.01,
            eval_every=100,
            consolidate_every=50,
            merge_sim_threshold=0.80,
            residual_free_threshold=0.05,
            max_merges_per_cycle=8,  # v1-like aggressive
            max_regrow_per_cycle=16,
            keep_free_slots=0,
            probe_batches=2,
            device=resolve_device(device),
            log_every=50,
            results_dir="results",
            run_name="stepped",
            corpus="synonym",
            corpus_length=500_000,
        ),
    )


def v2_config(device: str = "cuda:1", steps: int = 8000) -> ExperimentConfig:
    """Next real experiment: harder data + gentler residual consolidation.

    Improvements vs stepped v1:
      - structured synthetic (nested + delayed copy) — not pure bigrams
      - residual-preserving merge, max 2 merges/cycle, leave free slots
      - higher merge threshold (0.92)
      - longer horizon for plateau comparison
      - slightly larger width for headroom
    """
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=96,
            n_embd=192,
            n_head=6,
            block_size=128,
            dropout=0.0,
            n_layer=8,  # nominal; baseline re-matched by budget
            n_loops=8,
            n_slots=16,
            lora_rank=8,
            slots_per_loop=1,
        ),
        train=TrainConfig(
            seed=42,
            batch_size=64,
            seq_len=128,
            steps=steps,
            lr=8e-4,
            weight_decay=0.01,
            eval_every=200,
            consolidate_every=100,
            merge_sim_threshold=0.92,
            residual_free_threshold=0.08,
            max_merges_per_cycle=2,
            max_regrow_per_cycle=2,
            keep_free_slots=2,
            probe_batches=3,
            device=resolve_device(device),
            log_every=100,
            results_dir="results",
            run_name="v2",
            corpus="structured",
            corpus_length=800_000,
        ),
    )


def v2_char_config(
    device: str = "cuda:1",
    steps: int = 8000,
    source: str = "shakespeare",
) -> ExperimentConfig:
    """Char-level real text (vocab set at runtime from corpus)."""
    cfg = v2_config(device=device, steps=steps)
    cfg.train.corpus = source
    cfg.train.run_name = f"v2_{source}"
    cfg.train.max_chars = 1_000_000
    # vocab_size overwritten after corpus load
    cfg.model.vocab_size = 128  # placeholder
    return cfg


def ablation_config(device: str = "cuda:1", steps: int = 8000) -> ExperimentConfig:
    """4-arm ablation: baseline / looped / scrt_static / scrt.

    Same model/data budget as v2. Full SCRT consolidates only in the first half
    of training to avoid late free/regrow thrash observed in v2.
    """
    cfg = v2_config(device=device, steps=steps)
    cfg.train.run_name = "ablation"
    cfg.train.consolidate_until_frac = 0.5
    cfg.train.max_merges_per_cycle = 2
    cfg.train.max_regrow_per_cycle = 2
    cfg.train.keep_free_slots = 2
    cfg.train.residual_free_threshold = 0.10  # slightly less aggressive free
    return cfg


def saturation_config(device: str = "cuda:1", steps: int = 20000) -> ExperimentConfig:
    """Long-horizon run aimed at *post-saturation* consolidate benefit.

    Hypothesis: consolidation helps after capacity is the bottleneck, not early
    descent. Design:
      - long steps so baseline can flatten
      - consolidate only in the *second half* (start_frac=0.5, until=1.0)
      - gentler free/regrow so late merges are not pure thrash
      - same 4-arm matched budget as ablation/v2
    """
    cfg = v2_config(device=device, steps=steps)
    cfg.train.run_name = "saturation"
    cfg.train.steps = steps
    cfg.train.eval_every = 200
    cfg.train.log_every = 200
    cfg.train.consolidate_every = 100
    cfg.train.consolidate_start_frac = 0.5
    cfg.train.consolidate_until_frac = 1.0
    cfg.train.max_merges_per_cycle = 1
    cfg.train.max_regrow_per_cycle = 1
    cfg.train.keep_free_slots = 3
    cfg.train.residual_free_threshold = 0.12
    cfg.train.merge_sim_threshold = 0.90
    cfg.train.plateau_patience_evals = 6
    cfg.train.plateau_epsilon = 0.002
    cfg.train.plateau_min_frac = 0.3
    cfg.train.lr = 6e-4  # slightly lower for longer stability
    cfg.train.corpus = "structured"
    cfg.train.corpus_length = 1_000_000
    return cfg


def born_config(device: str = "cuda:1", steps: int = 20000) -> ExperimentConfig:
    """Born-consolidated suite: share by construction, no merge machinery.

    Arms: baseline / looped / shared_residual / modulated.
    Long horizon + light regularisation to read post-plateau behaviour.
    """
    cfg = v2_config(device=device, steps=steps)
    cfg.train.run_name = "born"
    cfg.train.steps = steps
    cfg.train.eval_every = 200
    cfg.train.log_every = 200
    cfg.train.consolidate_every = 0  # no consolidate on any arm
    cfg.train.lr = 5e-4
    cfg.train.weight_decay = 0.05
    cfg.model.dropout = 0.05
    cfg.train.corpus = "structured"
    cfg.train.corpus_length = 1_500_000
    cfg.train.plateau_patience_evals = 6
    cfg.train.plateau_epsilon = 0.002
    cfg.train.plateau_min_frac = 0.3
    return cfg


def saturation_merge_only_config(
    device: str = "cuda:1", steps: int = 20000
) -> ExperimentConfig:
    """Saturation follow-up: second-half consolidate with *merge only*.

    Prior saturation run overfit hard and free/regrow thrashed (many frees,
    few merges). This disables free+regrow, adds dropout/weight decay, and
    uses a longer stream so late eval is a fairer plateau test.
    """
    cfg = saturation_config(device=device, steps=steps)
    cfg.train.run_name = "sat_merge"
    cfg.train.consolidate_enable_free = False
    cfg.train.consolidate_enable_regrow = False
    cfg.train.max_merges_per_cycle = 2
    cfg.train.max_regrow_per_cycle = 0
    cfg.train.keep_free_slots = 0
    cfg.train.merge_sim_threshold = 0.88
    cfg.train.weight_decay = 0.1
    cfg.train.lr = 5e-4
    cfg.model.dropout = 0.1
    cfg.train.corpus_length = 1_500_000
    return cfg


def efficiency_config(device: str = "cuda:1", steps: int = 20000) -> ExperimentConfig:
    """Efficiency suite: native sizes, no ParamPad, multi-metric comparison.

    Same architecture family as born (baseline / looped / residual / modulated /
    shared_plus_tiny) but deliberately *does not* force equal param counts.
    Ranking is Pareto over (n_params, best_eval) plus eval-per-Mparam and wall time.

    Default corpus is **wikitext2** (capacity-bound). The structured synthetic
    corpus has entropy floor ~2.1405 nats/token; all arms pin near it so the
    loss axis is noise and a native-size Pareto collapses to "smaller wins by
    construction." Use ``corpus="structured"`` only for pipeline smoke tests.
    """
    cfg = born_config(device=device, steps=steps)
    cfg.train.run_name = "efficiency"
    cfg.train.consolidate_every = 0
    # Capacity-bound defaults (same as capacity_config char path).
    cfg.train.corpus = "wikitext2"
    cfg.train.max_chars = 11_000_000
    cfg.train.char_vocab_cap = 256
    cfg.model.vocab_size = 256  # re-read from corpus at load
    # Real text: lighter regularisation so capacity, not dropout, binds.
    cfg.train.lr = 8e-4
    cfg.train.weight_decay = 0.01
    cfg.model.dropout = 0.0
    return cfg


def capacity_config(device: str = "cuda:1", steps: int = 20000) -> ExperimentConfig:
    """Capacity-bound run: full char-level WikiText-2, one seed per invocation.

    Every prior run used the ``structured`` synthetic corpus, whose exact entropy
    rate is 2.1405 nats/token. All arms landed at 2.1827-2.1860 — inside 0.05 nats
    of the floor — so the arms were separated by ~0.003 nats of noise and nothing
    was capacity-bound. The saturation thesis is only testable when the *model*,
    not the data, is the binding constraint.

    Full WikiText-2 (10.9M chars) at ~1.4M params is comfortably in that regime:
    a model this size cannot memorise the corpus, so a flattening curve means
    capacity, and 16.7 epochs is mild enough that eval should track train.
    """
    cfg = v2_config(device=device, steps=steps)
    cfg.train.run_name = "capacity"
    cfg.train.steps = steps
    cfg.train.corpus = "wikitext2"
    cfg.train.max_chars = 11_000_000  # whole file (10,918,892)
    cfg.train.char_vocab_cap = 256
    cfg.model.vocab_size = 256  # placeholder; re-read from corpus at load
    cfg.train.eval_every = 200
    cfg.train.log_every = 200
    # Real text at 16 epochs does not overfit a 1.4M model — drop the heavy
    # regularisation the synthetic runs needed and let capacity be the limit.
    cfg.train.lr = 8e-4
    cfg.train.weight_decay = 0.01
    cfg.model.dropout = 0.0
    # Consolidate across the whole run, throttled (the born suite never fired it).
    cfg.train.consolidate_every = 100
    cfg.train.consolidate_start_frac = 0.0
    cfg.train.consolidate_until_frac = 1.0
    cfg.train.max_merges_per_cycle = 2
    cfg.train.max_regrow_per_cycle = 2
    # Frees were the only unthrottled op in the capacity run: 1351 frees vs
    # 50 merges collapsed 14 -> 4 active slots. Cap frees and give regrown
    # slots (up == 0, importance 0 -> maximally "weak" by construction) a
    # grace period to bootstrap before they become free candidates.
    cfg.train.max_frees_per_cycle = 2
    cfg.train.min_slot_age_cycles = 5
    cfg.train.keep_free_slots = 2
    cfg.train.merge_sim_threshold = 0.90
    cfg.train.residual_free_threshold = 0.12
    cfg.train.plateau_patience_evals = 6
    cfg.train.plateau_epsilon = 0.002
    cfg.train.plateau_min_frac = 0.3
    return cfg
