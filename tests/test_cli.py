import csv
from pathlib import Path

from local_ai_training import data
from local_ai_training.cli import build_parser
from local_ai_training.plotting import plot_comparison


def test_download_uses_pinned_hugging_face_file_without_remote_code(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "hub" / "input.txt"
    source.parent.mkdir()
    expected_text = "To be or not to be\n" * 100
    source.write_text(expected_text)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr(data, "hf_hub_download", fake_download)

    downloaded = data.download_tiny_shakespeare(tmp_path / "cache")

    assert downloaded.read_text() == expected_text
    assert calls == [
        {
            "repo_id": data.TINY_SHAKESPEARE_REPO,
            "repo_type": "dataset",
            "filename": "input.txt",
            "revision": data.TINY_SHAKESPEARE_REVISION,
            "cache_dir": str(tmp_path / "cache"),
        }
    ]


def test_parser_exposes_all_research_commands() -> None:
    parser = build_parser()

    for command in ("dataset", "train", "compare", "controls", "plot", "audit"):
        namespace = parser.parse_args([command, "--help"] if False else [command])
        assert namespace.command == command


def test_train_command_accepts_weight_mode_for_parallel_fp32_arms() -> None:
    parser = build_parser()

    default = parser.parse_args(["train"])
    assert default.weight_mode == "ratchet"

    fp32 = parser.parse_args(["train", "--weight-mode", "fp32"])
    assert fp32.weight_mode == "fp32"


def test_plot_comparison_reads_recursive_metrics_and_writes_png(tmp_path: Path) -> None:
    fieldnames = [
        "step",
        "train_loss",
        "validation_loss",
        "zero_percent",
        "saturated_percent",
        "code_moves",
    ]
    for arm, values in (("quinary", (2.0, 1.5)), ("septenary", (2.0, 1.3))):
        run = tmp_path / arm / "seed-1"
        run.mkdir(parents=True)
        with (run / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "step": 0,
                    "train_loss": "nan",
                    "validation_loss": values[0],
                    "zero_percent": 20,
                    "saturated_percent": 10,
                    "code_moves": 0,
                }
            )
            writer.writerow(
                {
                    "step": 10,
                    "train_loss": values[1],
                    "validation_loss": values[1],
                    "zero_percent": 18,
                    "saturated_percent": 12,
                    "code_moves": 4,
                }
            )

    output = plot_comparison(tmp_path)

    assert output == tmp_path / "comparison.png"
    assert output.stat().st_size > 0
