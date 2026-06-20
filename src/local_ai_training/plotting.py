"""Comparison plots for ratchet experiment CSV logs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_comparison(run_dir: str | Path, output: str | Path | None = None) -> Path:
    root = Path(run_dir)
    metric_files = sorted(root.glob("**/metrics.csv"))
    if not metric_files:
        raise ValueError(f"no metrics.csv files found under {root}")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for metrics_path in metric_files:
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        label = str(metrics_path.parent.relative_to(root)) or metrics_path.parent.name
        steps = [int(row["step"]) for row in rows]
        axes[0, 0].plot(steps, [float(row["validation_loss"]) for row in rows], label=label)
        train_rows = [row for row in rows if row["train_loss"].lower() != "nan"]
        axes[0, 1].plot(
            [int(row["step"]) for row in train_rows],
            [float(row["train_loss"]) for row in train_rows],
            label=label,
        )
        axes[1, 0].plot(steps, [float(row["zero_percent"]) for row in rows], label=label)
        axes[1, 1].plot(
            steps, [float(row["saturated_percent"]) for row in rows], label=label
        )
    for axis, title, ylabel in (
        (axes[0, 0], "Validation Loss", "loss"),
        (axes[0, 1], "Training Loss", "loss"),
        (axes[1, 0], "Zero Codes", "percent"),
        (axes[1, 1], "Saturated Codes", "percent"),
    ):
        axis.set_title(title)
        axis.set_xlabel("step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    output_path = Path(output) if output is not None else root / "comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path

