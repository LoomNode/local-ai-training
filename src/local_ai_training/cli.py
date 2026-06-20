"""Command line interface for local ratchet experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .config import ExperimentConfig
from .data import build_char_corpus, download_tiny_shakespeare
from .model import build_seeded_model
from .ratchet import audit_no_master_weights
from .train import train_run

DEFAULT_CONFIG = Path("configs/ratchet_tiny.toml")


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lat", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", help="download pinned Tiny Shakespeare")
    dataset.add_argument("--cache-dir", type=Path, default=Path("data/huggingface"))

    train = subparsers.add_parser("train", help="train one ratchet arm")
    _add_config(train)
    train.add_argument("--codes", type=int, choices=(5, 7), default=5)
    train.add_argument("--seed", type=int)
    train.add_argument("--dataset-path", type=Path)
    train.add_argument("--cache-dir", type=Path, default=Path("data/huggingface"))
    train.add_argument("--output", type=Path, default=Path("runs/single"))
    train.add_argument("--resume", type=Path)

    compare = subparsers.add_parser("compare", help="run matched quinary/septenary arms")
    _add_config(compare)
    compare.add_argument("--dataset-path", type=Path)
    compare.add_argument("--cache-dir", type=Path, default=Path("data/huggingface"))
    compare.add_argument("--output", type=Path, default=Path("runs/comparison"))

    controls = subparsers.add_parser("controls", help="run FP32 and frozen control arms")
    _add_config(controls)
    controls.add_argument("--dataset-path", type=Path)
    controls.add_argument("--cache-dir", type=Path, default=Path("data/huggingface"))
    controls.add_argument("--output", type=Path, default=Path("runs/controls"))

    plot = subparsers.add_parser("plot", help="plot recursive experiment CSV files")
    plot.add_argument("run_dir", type=Path, nargs="?", default=Path("runs/comparison"))
    plot.add_argument("--output", type=Path)

    audit = subparsers.add_parser("audit", help="audit a configured model for master weights")
    audit.add_argument("--model", dest="config", type=Path, default=DEFAULT_CONFIG)
    audit.add_argument("--codes", type=int, choices=(5, 7), default=5)
    audit.add_argument("--vocab-size", type=int, default=65)
    return parser


def _corpus(dataset_path: Path | None, cache_dir: Path):
    path = dataset_path or download_tiny_shakespeare(cache_dir)
    text = path.read_text(encoding="utf-8")
    return build_char_corpus(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dataset":
        print(download_tiny_shakespeare(args.cache_dir))
        return 0
    if args.command == "plot":
        from .plotting import plot_comparison

        print(plot_comparison(args.run_dir, args.output))
        return 0
    config = ExperimentConfig.from_toml(args.config)
    if args.command == "audit":
        max_code = (args.codes - 1) // 2
        model = build_seeded_model(
            config.model_config(vocab_size=args.vocab_size),
            max_code=max_code,
            seed=config.seeds[0],
        )
        print(json.dumps(asdict(audit_no_master_weights(model, raise_on_violation=True)), indent=2))
        return 0
    corpus = _corpus(args.dataset_path, args.cache_dir)
    if args.command == "train":
        seed = args.seed if args.seed is not None else config.seeds[0]
        result = train_run(
            corpus=corpus,
            config=config,
            max_code=(args.codes - 1) // 2,
            seed=seed,
            run_dir=args.output,
            resume_from=args.resume,
        )
        print(json.dumps(asdict(result), default=str, indent=2))
        return 0
    if args.command == "compare":
        from .plotting import plot_comparison

        summaries = []
        for max_code, arm in ((2, "quinary"), (3, "septenary")):
            for seed in config.seeds:
                summaries.append(
                    train_run(
                        corpus=corpus,
                        config=config,
                        max_code=max_code,
                        seed=seed,
                        run_dir=args.output / arm / f"seed-{seed}",
                    )
                )
        plot_path = plot_comparison(args.output)
        print(json.dumps({"runs": len(summaries), "plot": str(plot_path)}, indent=2))
        return 0
    if args.command == "controls":
        from .plotting import plot_comparison

        summaries = []
        arms = (
            (None, "fp32", "fp32"),
            (2, "frozen-quinary", "frozen"),
            (3, "frozen-septenary", "frozen"),
        )
        for max_code, arm, weight_mode in arms:
            for seed in config.seeds:
                summaries.append(
                    train_run(
                        corpus=corpus,
                        config=config,
                        max_code=max_code,
                        seed=seed,
                        run_dir=args.output / arm / f"seed-{seed}",
                        weight_mode=weight_mode,
                    )
                )
        plot_path = plot_comparison(args.output)
        print(json.dumps({"runs": len(summaries), "plot": str(plot_path)}, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
