"""Copyable-mask split of validation loss for the 30k momentum-study checkpoints.

At 25M / text8 / 30k steps the ratchet ("plain") arm trails a matched QAT arm by
~0.073 nats of validation loss. This script asks whether that gap sits on bytes
the model could in principle copy from earlier in its 256-char window (the target
byte's context already occurred, verbatim, ending at an earlier position in the
same window) or on bytes it cannot. Pure scoring of existing checkpoints against
the text8 validation split; no training and no changes to src/.

Run as: .venv/bin/python -m scripts.copyable_mask_eval [flags]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from local_ai_training import data
from local_ai_training.generate import load_for_generation

REPO = Path(__file__).resolve().parent.parent

DEFAULT_ROOT = Path("/games/ailab/local-ai-training/runs/momentum-30k-2026-09-13")
DEFAULT_OUTPUT = Path("/games/ailab/local-ai-training/runs/copyable-mask-2026-09-14")
DEFAULT_DATASET_PATH = Path("/games/ailab/local-ai-training/data/text8")

# Pinned SHA-256 of the extracted text8 file (not the zip) -- asserted below so a
# corrupted or wrong-revision cache fails loudly instead of silently rescoring.
TEXT8_SHA256 = "6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7"

ARMS = ("plain", "momentum", "qat", "fp32")
SEEDS = (1337, 1338, 1339)
BIN_LABELS = ("0", "1", "2-3", "4-7", "8+")
COPYABLE8_LABELS = ("copyable8", "not_copyable8")
# (a, b) -> reported as gap "a-b" = nll(a) - nll(b).
GAP_PAIRS = (("plain", "qat"), ("momentum", "qat"), ("qat", "fp32"), ("plain", "fp32"))
GAP_ARMS = {arm for pair in GAP_PAIRS for arm in pair}


# --------------------------------------------------------------------------- #
# Pure tensor code: the match-length DP and its bins.
# --------------------------------------------------------------------------- #


def longest_earlier_match(x: torch.Tensor) -> torch.Tensor:
    """For x[B, T] (long ids), m[i] = longest earlier match ending before i.

    m[i] is the largest k such that the suffix x[i-k+1:i+1] also occurs ending at
    some earlier position j < i, i.e. x[j-k+1:j+1] == x[i-k+1:i+1]; m[0] = 0. This
    is the longest earlier n-gram match whose *next* byte is already known, so
    target x[i+1] is "copyable" with match length m[i].

    Batched diagonal DP: L[i, j] = L[i-1, j-1] + 1 if x[i] == x[j] else 0 (j < i),
    m[i] = max_{j<i} L[i, j]. `prev` holds the previous step's L[i-1, :] across the
    python loop over i (T steps, T == block_size); everything inside the loop is a
    vector op over batch and j, run on the same device as x.
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (batch, sequence)")
    batch_size, length = x.shape
    m = torch.zeros((batch_size, length), dtype=torch.long, device=x.device)
    if length <= 1:
        return m
    prev = torch.zeros((batch_size, length), dtype=torch.long, device=x.device)
    for i in range(length):
        shifted = torch.zeros((batch_size, length), dtype=torch.long, device=x.device)
        shifted[:, 1:] = prev[:, :-1]
        matches = x[:, i : i + 1] == x
        current = torch.where(matches, shifted + 1, torch.zeros_like(shifted))
        if i > 0:
            m[:, i] = current[:, :i].max(dim=1).values
        prev = current
    return m


def match_bin(m: torch.Tensor) -> torch.Tensor:
    """Bin match lengths into BIN_LABELS = ("0", "1", "2-3", "4-7", "8+")."""
    bins = torch.full_like(m, 4)
    bins = torch.where(m < 8, torch.full_like(m, 3), bins)
    bins = torch.where(m < 4, torch.full_like(m, 2), bins)
    bins = torch.where(m < 2, torch.full_like(m, 1), bins)
    bins = torch.where(m < 1, torch.zeros_like(m), bins)
    return bins


# --------------------------------------------------------------------------- #
# Small numeric helpers (pure, unit tested).
# --------------------------------------------------------------------------- #


def mean_and_sample_sd(values) -> tuple[float, float]:
    values = list(values)
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def gap_mass_per_bin(fractions: dict, gaps: dict) -> dict:
    """Per-bin contribution to the overall gap: fraction_bin * gap_bin.

    Summing the result over all bins of a complete partition reproduces the
    overall gap (fraction-weighted mean of the per-bin gaps).
    """
    return {label: fractions[label] * gaps[label] for label in fractions}


# --------------------------------------------------------------------------- #
# Dataset and batching.
# --------------------------------------------------------------------------- #


def load_validation_corpus(dataset_path: Path) -> tuple[data.CharCorpus, str]:
    text_path = data.download_text8(dataset_path)
    digest = hashlib.sha256(text_path.read_bytes()).hexdigest()
    if digest != TEXT8_SHA256:
        raise ValueError(f"text8 checksum mismatch: expected {TEXT8_SHA256}, got {digest}")
    corpus = data.build_char_corpus(text_path.read_text())
    return corpus, digest


def build_windows(
    validation_ids: torch.Tensor, *, block_size: int, limit_windows: int | None
) -> torch.Tensor:
    high = len(validation_ids) - block_size - 1
    starts = torch.arange(0, high, block_size)
    if limit_windows is not None:
        starts = starts[:limit_windows]
    return starts


def make_batches(
    validation_ids: torch.Tensor,
    starts: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for offset in range(0, len(starts), batch_size):
        chunk = starts[offset : offset + batch_size]
        inputs, targets = data.batch_from_starts(validation_ids, chunk, block_size=block_size)
        batches.append((inputs.to(device), targets.to(device)))
    return batches


def compute_bin_batches(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[torch.Tensor]:
    """Match-length bins depend only on the input windows, not the model -- compute once."""
    return [match_bin(longest_earlier_match(inputs)) for inputs, _ in batches]


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #


def checkpoint_path(root: Path, arm: str, seed: int) -> Path:
    return root / f"{arm}-seed{seed}" / "checkpoint"


def load_checkpoint_config(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text())


@torch.inference_mode()
def score_checkpoint(
    model,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    bin_batches: list[torch.Tensor],
    *,
    device: torch.device,
) -> dict:
    bin_sum = dict.fromkeys(BIN_LABELS, 0.0)
    bin_count = dict.fromkeys(BIN_LABELS, 0)
    copyable_sum = dict.fromkeys(COPYABLE8_LABELS, 0.0)
    copyable_count = dict.fromkeys(COPYABLE8_LABELS, 0)
    total_sum = 0.0
    total_count = 0
    copyable_index = len(BIN_LABELS) - 1  # "8+" is m >= 8, the copyable8 split.

    for (inputs, targets), bins in zip(batches, bin_batches, strict=True):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(inputs)
        vocab_size = logits.shape[-1]
        nll = F.cross_entropy(
            logits.float().view(-1, vocab_size), targets.reshape(-1), reduction="none"
        ).view(targets.shape)

        total_sum += float(nll.sum().item())
        total_count += nll.numel()

        for index, label in enumerate(BIN_LABELS):
            mask = bins == index
            bin_sum[label] += float(nll[mask].sum().item())
            bin_count[label] += int(mask.sum().item())

        copyable_mask = bins == copyable_index
        copyable_sum["copyable8"] += float(nll[copyable_mask].sum().item())
        copyable_count["copyable8"] += int(copyable_mask.sum().item())
        copyable_sum["not_copyable8"] += float(nll[~copyable_mask].sum().item())
        copyable_count["not_copyable8"] += int((~copyable_mask).sum().item())

    return {
        "overall_nll": total_sum / total_count,
        "bin_nll": {label: bin_sum[label] / bin_count[label] for label in BIN_LABELS},
        "bin_fraction": {label: bin_count[label] / total_count for label in BIN_LABELS},
        "copyable8_nll": {
            label: copyable_sum[label] / copyable_count[label] for label in COPYABLE8_LABELS
        },
        "copyable8_fraction": {
            label: copyable_count[label] / total_count for label in COPYABLE8_LABELS
        },
        "token_count": total_count,
    }


# --------------------------------------------------------------------------- #
# Aggregation across seeds: arm means and derived gaps.
# --------------------------------------------------------------------------- #


def arm_means(per_arm_seed: dict, arm: str, seeds: list[int]) -> dict:
    seed_keys = [str(seed) for seed in seeds]
    overall_mean, overall_sd = mean_and_sample_sd(
        per_arm_seed[arm][seed]["overall_nll"] for seed in seed_keys
    )
    bin_mean = {
        label: mean_and_sample_sd(
            per_arm_seed[arm][seed]["bin_nll"][label] for seed in seed_keys
        )[0]
        for label in BIN_LABELS
    }
    copyable_mean = {
        label: mean_and_sample_sd(
            per_arm_seed[arm][seed]["copyable8_nll"][label] for seed in seed_keys
        )[0]
        for label in COPYABLE8_LABELS
    }
    # Fractions are arm-independent (they depend only on the shared input windows);
    # take them from the first seed of this arm rather than recomputing.
    reference = per_arm_seed[arm][seed_keys[0]]
    return {
        "overall_mean": overall_mean,
        "overall_sd": overall_sd,
        "bin_mean": bin_mean,
        "copyable_mean": copyable_mean,
        "bin_fraction": reference["bin_fraction"],
        "copyable8_fraction": reference["copyable8_fraction"],
    }


def compute_gaps(per_arm_seed: dict, seeds: list[int]) -> dict:
    gaps = {}
    for a, b in GAP_PAIRS:
        key = f"{a}-{b}"
        per_seed = {}
        for seed in seeds:
            s = str(seed)
            ra, rb = per_arm_seed[a][s], per_arm_seed[b][s]
            overall_gap = ra["overall_nll"] - rb["overall_nll"]
            bin_gap = {label: ra["bin_nll"][label] - rb["bin_nll"][label] for label in BIN_LABELS}
            copyable_gap = {
                label: ra["copyable8_nll"][label] - rb["copyable8_nll"][label]
                for label in COPYABLE8_LABELS
            }

            bin_mass = gap_mass_per_bin(ra["bin_fraction"], bin_gap)
            copyable_mass = gap_mass_per_bin(ra["copyable8_fraction"], copyable_gap)
            for mass, name in ((bin_mass, "bin"), (copyable_mass, "copyable8")):
                total_mass = sum(mass.values())
                assert abs(total_mass - overall_gap) < 1e-6, (
                    f"{name} gap mass for {key} seed {seed} sums to {total_mass!r}, "
                    f"expected overall gap {overall_gap!r}"
                )

            per_seed[s] = {
                "overall_gap": overall_gap,
                "bin_gap": bin_gap,
                "copyable8_gap": copyable_gap,
                "bin_gap_mass": bin_mass,
                "copyable8_gap_mass": copyable_mass,
            }

        seed_keys = [str(seed) for seed in seeds]
        overall_mean, overall_sd = mean_and_sample_sd(
            per_seed[s]["overall_gap"] for s in seed_keys
        )
        bin_mean = {}
        bin_sd = {}
        for label in BIN_LABELS:
            bin_mean[label], bin_sd[label] = mean_and_sample_sd(
                per_seed[s]["bin_gap"][label] for s in seed_keys
            )
        copyable_mean = {}
        copyable_sd = {}
        for label in COPYABLE8_LABELS:
            copyable_mean[label], copyable_sd[label] = mean_and_sample_sd(
                per_seed[s]["copyable8_gap"][label] for s in seed_keys
            )

        gaps[key] = {
            "per_seed": per_seed,
            "overall_mean": overall_mean,
            "overall_sd": overall_sd,
            "bin_mean": bin_mean,
            "bin_sd": bin_sd,
            "copyable8_mean": copyable_mean,
            "copyable8_sd": copyable_sd,
        }
    return gaps


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #


def render_markdown_table(arm_stats: dict, gaps: dict) -> str:
    header = (
        "| bin | fraction | plain | momentum | qat | fp32 "
        "| plain-qat | qat-fp32 | gap share (plain-qat) |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|"
    plain_qat_overall = gaps["plain-qat"]["overall_mean"]
    rows = [header, separator]

    def row(
        label: str, fraction: float, values: dict, plain_qat_gap: float, qat_fp32_gap: float
    ) -> str:
        mass = fraction * plain_qat_gap
        share = mass / plain_qat_overall if plain_qat_overall else float("nan")
        cells = " | ".join(f"{values[arm]:.4f}" for arm in ARMS)
        return (
            f"| {label} | {fraction:.4f} | {cells} "
            f"| {plain_qat_gap:+.4f} | {qat_fp32_gap:+.4f} | {share:+.1%} |"
        )

    for label in BIN_LABELS:
        values = {arm: arm_stats[arm]["bin_mean"][label] for arm in ARMS}
        fraction = arm_stats["plain"]["bin_fraction"][label]
        rows.append(
            row(
                label,
                fraction,
                values,
                gaps["plain-qat"]["bin_mean"][label],
                gaps["qat-fp32"]["bin_mean"][label],
            )
        )

    overall_values = {arm: arm_stats[arm]["overall_mean"] for arm in ARMS}
    rows.append(
        row(
            "overall",
            1.0,
            overall_values,
            gaps["plain-qat"]["overall_mean"],
            gaps["qat-fp32"]["overall_mean"],
        )
    )

    copyable_rows = (
        ("copyable8", "copyable8 (m>=8)"),
        ("not_copyable8", "not copyable8 (m<8)"),
    )
    for label, display in copyable_rows:
        values = {arm: arm_stats[arm]["copyable_mean"][label] for arm in ARMS}
        fraction = arm_stats["plain"]["copyable8_fraction"][label]
        rows.append(
            row(
                display,
                fraction,
                values,
                gaps["plain-qat"]["copyable8_mean"][label],
                gaps["qat-fp32"]["copyable8_mean"][label],
            )
        )

    return "\n".join(rows)


def render_sanity_table(sanity_rows: list[tuple[str, int, float, float | None]]) -> str:
    header = "| arm | seed | eval overall NLL | recorded final | diff |"
    separator = "|---|---|---|---|---|"
    lines = [header, separator]
    for arm, seed, overall_nll, recorded_final in sanity_rows:
        if recorded_final is None:
            lines.append(f"| {arm} | {seed} | {overall_nll:.4f} | n/a | n/a |")
        else:
            diff = overall_nll - recorded_final
            lines.append(
                f"| {arm} | {seed} | {overall_nll:.4f} | {recorded_final:.4f} | {diff:+.4f} |"
            )
    return "\n".join(lines)


def git_commit(repo: Path = REPO) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-windows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    root = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    arms = list(args.arms)

    corpus, dataset_sha = load_validation_corpus(Path(args.dataset_path))
    validation_ids = corpus.validation_ids

    first_config = load_checkpoint_config(checkpoint_path(root, arms[0], seeds[0]))
    block_size = int(first_config["experiment_config"]["block_size"])

    starts = build_windows(validation_ids, block_size=block_size, limit_windows=args.limit_windows)
    window_count = len(starts)
    batches = make_batches(
        validation_ids, starts, block_size=block_size, batch_size=args.batch_size, device=device
    )
    bin_batches = compute_bin_batches(batches)

    baseline_path = root / "results.json"
    baseline_results = json.loads(baseline_path.read_text()) if baseline_path.is_file() else None

    per_arm_seed: dict[str, dict] = {}
    checkpoint_paths: dict[str, str] = {}
    sanity_rows: list[tuple[str, int, float, float | None]] = []

    for arm in arms:
        per_arm_seed[arm] = {}
        for seed in seeds:
            path = checkpoint_path(root, arm, seed)
            checkpoint_paths[f"{arm}-seed{seed}"] = str(path)
            config = load_checkpoint_config(path)
            checkpoint_block_size = int(config["experiment_config"]["block_size"])
            assert checkpoint_block_size == block_size, (
                f"{arm}-seed{seed} block_size {checkpoint_block_size} != {block_size}"
            )

            model, vocabulary = load_for_generation(path, device=device)
            assert vocabulary == corpus.vocabulary, (
                f"{arm}-seed{seed} checkpoint vocabulary does not match the corpus vocabulary"
            )

            result = score_checkpoint(model, batches, bin_batches, device=device)
            per_arm_seed[arm][str(seed)] = result

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            recorded_final = None
            if baseline_results is not None:
                recorded_final = (
                    baseline_results.get("per_seed", {})
                    .get(str(seed), {})
                    .get(arm, {})
                    .get("final")
                )
            sanity_rows.append((arm, seed, result["overall_nll"], recorded_final))

            print(
                f"scored {arm}-seed{seed}: overall_nll={result['overall_nll']:.4f} "
                f"recorded_final={recorded_final}"
            )

    arm_stats = {arm: arm_means(per_arm_seed, arm, seeds) for arm in arms}
    gaps = compute_gaps(per_arm_seed, seeds) if GAP_ARMS <= set(arms) else {}

    markdown_table = ""
    if gaps:
        markdown_table = render_markdown_table(arm_stats, gaps)
        print()
        print(markdown_table)

    print()
    sanity_table = render_sanity_table(sanity_rows)
    print(sanity_table)

    results = {
        "device": str(device),
        "torch_version": torch.__version__,
        "git_commit": git_commit(),
        "dataset_sha256": dataset_sha,
        "block_size": block_size,
        "batch_size": args.batch_size,
        "window_count": window_count,
        "root": str(root),
        "checkpoint_paths": checkpoint_paths,
        "per_arm_seed": per_arm_seed,
        "arm_stats": arm_stats,
        "gaps": gaps,
        "sanity": [
            {"arm": arm, "seed": seed, "overall_nll": overall_nll, "recorded_final": recorded_final}
            for arm, seed, overall_nll, recorded_final in sanity_rows
        ],
        "markdown_table": markdown_table,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
