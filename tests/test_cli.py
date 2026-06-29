import csv
from pathlib import Path

import pytest

from local_ai_training import cli, data
from local_ai_training.cli import build_parser
from local_ai_training.plotting import plot_comparison
from local_ai_training.train import TrainResult


@pytest.mark.parametrize("codes", [3, 5, 7, 9, 11, 13, 15])
def test_train_accepts_extended_codes(codes):
    args = build_parser().parse_args(["train", "--codes", str(codes)])
    assert args.codes == codes


def test_train_trainable_scale_flag_defaults_off_and_can_enable():
    parser = build_parser()
    assert parser.parse_args(["train"]).trainable_scale is False
    assert parser.parse_args(["train", "--trainable-scale"]).trainable_scale is True


@pytest.mark.parametrize("codes", [3, 11, 13, 15])
def test_audit_accepts_extended_codes(codes):
    args = build_parser().parse_args(["audit", "--codes", str(codes)])
    assert args.codes == codes


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

    commands = (
        ["dataset"],
        ["shard", "fineweb-edu", "--target-tokens", "100"],
        ["train"],
        ["compare"],
        ["controls"],
        ["plot"],
        ["audit"],
    )
    for argv in commands:
        namespace = parser.parse_args(argv)
        assert namespace.command == argv[0]


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


def test_train_rms_ema_beta_flag_defaults_zero_and_parses():
    parser = build_parser()
    assert parser.parse_args(["train"]).rms_ema_beta == 0.0
    assert parser.parse_args(["train", "--rms-ema-beta", "0.9"]).rms_ema_beta == 0.9


def test_train_pressure_leak_period_flag_defaults_zero_and_parses():
    parser = build_parser()
    assert parser.parse_args(["train"]).pressure_leak_period == 0
    assert parser.parse_args(["train", "--pressure-leak-period", "4"]).pressure_leak_period == 4


def test_train_uses_config_tokenizer_for_corpus_when_cli_omits_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "subword.toml"
    config_path.write_text(
        "[training]\n"
        "tokenizer = 'subword'\n"
        "vocab_size = 300\n"
        "seeds = [123]\n"
        "device = 'cpu'\n"
    )
    dataset_path = tmp_path / "corpus.txt"
    dataset_path.write_text("hello world " * 10)
    calls: dict[str, object] = {}
    corpus = object()

    def fake_corpus(dataset_path_arg, cache_dir_arg, *, tokenizer, vocab_size):
        calls["corpus"] = (dataset_path_arg, cache_dir_arg, tokenizer, vocab_size)
        return corpus

    def fake_train_run(**kwargs):
        calls["train_config"] = kwargs["config"]
        calls["train_corpus"] = kwargs["corpus"]
        checkpoint = tmp_path / "checkpoint"
        return TrainResult(
            run_dir=tmp_path,
            metrics_csv=tmp_path / "metrics.csv",
            checkpoint=checkpoint,
            initial_validation_loss=0.0,
            final_validation_loss=0.0,
            total_code_moves=0,
        )

    monkeypatch.setattr(cli, "_corpus", fake_corpus)
    monkeypatch.setattr(cli, "train_run", fake_train_run)

    assert cli.main(
        [
            "train",
            "--config",
            str(config_path),
            "--dataset-path",
            str(dataset_path),
            "--output",
            str(tmp_path / "run"),
        ]
    ) == 0

    assert calls["corpus"] == (dataset_path, Path("data/huggingface"), "subword", 300)
    assert calls["train_corpus"] is corpus
    assert calls["train_config"].tokenizer == "subword"
    assert calls["train_config"].vocab_size == 300
    capsys.readouterr()


def test_train_cli_tokenizer_override_wins_over_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "subword.toml"
    config_path.write_text(
        "[training]\n"
        "tokenizer = 'subword'\n"
        "vocab_size = 300\n"
        "seeds = [123]\n"
        "device = 'cpu'\n"
    )
    calls: dict[str, object] = {}
    corpus = object()

    def fake_corpus(dataset_path_arg, cache_dir_arg, *, tokenizer, vocab_size):
        calls["corpus"] = (tokenizer, vocab_size)
        return corpus

    def fake_train_run(**kwargs):
        calls["train_config"] = kwargs["config"]
        return TrainResult(
            run_dir=tmp_path,
            metrics_csv=tmp_path / "metrics.csv",
            checkpoint=tmp_path / "checkpoint",
            initial_validation_loss=0.0,
            final_validation_loss=0.0,
            total_code_moves=0,
        )

    monkeypatch.setattr(cli, "_corpus", fake_corpus)
    monkeypatch.setattr(cli, "train_run", fake_train_run)

    assert cli.main(
        [
            "train",
            "--config",
            str(config_path),
            "--tokenizer",
            "char",
            "--vocab-size",
            "512",
        ]
    ) == 0

    assert calls["corpus"] == ("char", 512)
    assert calls["train_config"].tokenizer == "char"
    assert calls["train_config"].vocab_size == 512
    capsys.readouterr()


def test_shard_command_streams_fineweb_rows_to_token_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, object] = {}

    def fake_load_dataset(path, name, *, split, streaming, revision, trust_remote_code):
        calls["load_dataset"] = {
            "path": path,
            "name": name,
            "split": split,
            "streaming": streaming,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        return iter(
            [
                {"text": "alpha beta gamma " * 20},
                {"text": "delta epsilon zeta " * 20},
            ]
        )

    monkeypatch.setattr(cli, "load_dataset", fake_load_dataset)

    output = tmp_path / "fineweb-shard"
    assert cli.main(
        [
            "shard",
            "fineweb-edu",
            "--output",
            str(output),
            "--target-tokens",
            "32",
            "--vocab-size",
            "280",
            "--tokenizer-train-chars",
            "200",
        ]
    ) == 0

    assert calls["load_dataset"] == {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "streaming": True,
        "revision": cli.FINEWEB_EDU_REVISION,
        "trust_remote_code": False,
    }
    assert (output / "metadata.json").is_file()
    assert (output / "tokens.uint16").is_file()
    stdout = capsys.readouterr().out
    assert '"actual_tokens"' in stdout
