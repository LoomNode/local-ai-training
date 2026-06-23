from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

import local_ai_training.bitnet_eval as bitnet_eval
from local_ai_training.bitnet_eval import (
    BitNetConfig,
    apply_runtime_compatibility_patch,
    benchmark_cases,
    build_benchmark_command,
    build_chat_command,
    build_model_download_command,
    build_parser,
    build_smoke_command,
    build_toolchain_create_command,
    build_toolchain_install_command,
    download_verified,
    ensure_training_idle,
    extract_micromamba,
    load_config,
    main,
    new_run_directory,
    parse_benchmark_output,
    parse_time_metrics,
    require_ready,
    run_benchmark,
    run_setup,
    run_smoke,
    verify_model,
)


def config(tmp_path: Path) -> BitNetConfig:
    return BitNetConfig(
        repository_root=tmp_path,
        runtime_repository="https://github.com/microsoft/BitNet.git",
        runtime_commit="0fdaa16ae3b33876dbe8d79b2bbcf152f18560f8",
        model_repository="microsoft/bitnet-b1.58-2B-4T-gguf",
        model_revision="a1f2f1c765812aa8af3f6eda4a313707064bba15",
        model_filename="ggml-model-i2_s.gguf",
        model_size=4,
        model_sha256=hashlib.sha256(b"test").hexdigest(),
        prompt_tokens=(128, 512),
        generated_tokens=128,
        benchmark_threads=(1, 2, 4, 8, 16),
        benchmark_repetitions=5,
        chat_threads=8,
        chat_context=4096,
        chat_temperature=0.7,
    )


def test_config_rejects_malformed_immutable_pins(tmp_path: Path) -> None:
    values = config(tmp_path).__dict__ | {"model_sha256": "not-a-hash"}

    with pytest.raises(ValueError, match="model_sha256"):
        BitNetConfig(**values)


def test_verify_model_accepts_exact_size_and_hash(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"test")

    verify_model(model, expected_size=4, expected_sha256=hashlib.sha256(b"test").hexdigest())


@pytest.mark.parametrize(
    ("contents", "message"),
    [(b"bad", "size"), (b"xxxx", "SHA-256")],
)
def test_verify_model_rejects_mismatched_artifact(
    tmp_path: Path, contents: bytes, message: str
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(contents)

    with pytest.raises(RuntimeError, match=message):
        verify_model(model, expected_size=4, expected_sha256=hashlib.sha256(b"test").hexdigest())


def test_benchmark_cases_expand_prompt_and_thread_matrix(tmp_path: Path) -> None:
    assert benchmark_cases(config(tmp_path)) == [
        (prompt, threads) for prompt in (128, 512) for threads in (1, 2, 4, 8, 16)
    ]


def test_training_process_blocks_cpu_work_without_override() -> None:
    process_listing = "123 uv run lat train --config configs/ratchet_tiny.toml"

    with pytest.raises(RuntimeError, match="active training process"):
        ensure_training_idle(process_listing, allow_contention=False)

    ensure_training_idle(process_listing, allow_contention=True)


def test_unrelated_process_does_not_block_cpu_work() -> None:
    ensure_training_idle("123 python scripts/bitnet_eval.py benchmark", allow_contention=False)


def test_new_run_directory_is_timestamped_and_never_reused(tmp_path: Path) -> None:
    now = datetime(2026, 6, 21, 12, 34, 56, tzinfo=timezone.utc)
    first = new_run_directory(tmp_path, now=now)
    second = new_run_directory(tmp_path, now=now)

    assert first.name == "20260621T123456Z"
    assert second.name == "20260621T123456Z-01"


def test_chat_command_uses_cpu_conversation_defaults(tmp_path: Path) -> None:
    cfg = config(tmp_path)

    command = build_chat_command(cfg, system_prompt="Be concise.")

    assert command == [
        str(cfg.llama_cli),
        "-m",
        str(cfg.model_path),
        "-n",
        "-1",
        "-t",
        "8",
        "-p",
        "Be concise.",
        "-ngl",
        "0",
        "-c",
        "4096",
        "--temp",
        "0.7",
        "-b",
        "1",
        "-cnv",
    ]


def test_smoke_command_is_deterministic(tmp_path: Path) -> None:
    cfg = config(tmp_path)

    command = build_smoke_command(cfg, prompt="What is 84 * 3 / 2?")

    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "1337"
    assert command[command.index("--temp") + 1] == "0.0"
    assert command[command.index("-n") + 1] == "128"


def test_load_config_resolves_artifacts_beneath_repository_root(tmp_path: Path) -> None:
    config_path = tmp_path / "bitnet.toml"
    config_path.write_text(
        """
[runtime]
repository = "https://github.com/microsoft/BitNet.git"
commit = "0fdaa16ae3b33876dbe8d79b2bbcf152f18560f8"

[model]
repository = "microsoft/bitnet-b1.58-2B-4T-gguf"
revision = "a1f2f1c765812aa8af3f6eda4a313707064bba15"
filename = "ggml-model-i2_s.gguf"
size = 1187801280
sha256 = "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"

[benchmark]
prompt_tokens = [128, 512]
generated_tokens = 128
threads = [1, 2, 4, 8, 16]
repetitions = 5

[chat]
threads = 8
context = 4096
temperature = 0.7

[toolchain]
micromamba_url = "https://example.invalid/micromamba.tar.bz2"
micromamba_size = 99
micromamba_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
packages = ["python=3.11", "clang=18"]
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(config_path, repository_root=tmp_path)

    assert loaded.model_path == (
        tmp_path / "data/bitnet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    )
    assert loaded.runtime_commit == "0fdaa16ae3b33876dbe8d79b2bbcf152f18560f8"
    assert loaded.micromamba_size == 99
    assert loaded.toolchain_packages == ("python=3.11", "clang=18")


def test_benchmark_command_requests_machine_readable_repeated_cpu_run(tmp_path: Path) -> None:
    cfg = config(tmp_path)

    command = build_benchmark_command(cfg, prompt_tokens=512, threads=8)

    assert command == [
        str(cfg.llama_bench),
        "-m",
        str(cfg.model_path),
        "-n",
        "128",
        "-p",
        "512",
        "-t",
        "8",
        "-r",
        "5",
        "-ngl",
        "0",
        "-b",
        "1",
        "-o",
        "json",
    ]


def test_toolchain_command_installs_only_beneath_ignored_data(tmp_path: Path) -> None:
    cfg = config(tmp_path)

    command = build_toolchain_create_command(cfg)

    assert command[:6] == [
        str(cfg.micromamba_binary),
        "create",
        "--yes",
        "--prefix",
        str(cfg.toolchain_dir),
        "--channel",
    ]
    assert "clang=18" in command
    assert "clangxx=18" in command
    assert "cmake=3.31" in command
    assert cfg.toolchain_dir.is_relative_to(tmp_path / "data")


def test_toolchain_install_command_repairs_existing_environment(tmp_path: Path) -> None:
    cfg = config(tmp_path)

    command = build_toolchain_install_command(cfg)

    assert command[1] == "install"
    assert "clangxx=18" in command
    assert str(cfg.toolchain_dir) in command


def test_model_download_command_pins_revision_and_single_gguf(tmp_path: Path) -> None:
    cfg = config(tmp_path)

    command = build_model_download_command(cfg)

    assert cfg.hf_binary.name == "huggingface-cli"
    assert command == [
        str(cfg.hf_binary),
        "download",
        cfg.model_repository,
        cfg.model_filename,
        "--revision",
        cfg.model_revision,
        "--local-dir",
        str(cfg.model_dir),
    ]


def test_extract_micromamba_extracts_only_expected_binary(tmp_path: Path) -> None:
    archive = tmp_path / "micromamba.tar.bz2"
    payload = b"executable"
    with tarfile.open(archive, "w:bz2") as bundle:
        member = tarfile.TarInfo("bin/micromamba")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
        unwanted = tarfile.TarInfo("etc/unwanted")
        unwanted.size = 3
        bundle.addfile(unwanted, io.BytesIO(b"bad"))
    destination = tmp_path / "tools" / "bin" / "micromamba"

    extract_micromamba(archive, destination)

    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o111
    assert not (tmp_path / "etc/unwanted").exists()


def test_runtime_patch_makes_read_only_activation_pointer_const(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    source = cfg.runtime_dir / "src" / "ggml-bitnet-mad.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("        int8_t * y_col = y + col * by;\n", encoding="utf-8")

    apply_runtime_compatibility_patch(cfg)
    apply_runtime_compatibility_patch(cfg)

    assert source.read_text(encoding="utf-8") == (
        "        const int8_t * y_col = y + col * by;\n"
    )


def test_download_verified_reuses_valid_existing_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.write_bytes(b"test")
    opened = False

    def opener(_url: str):
        nonlocal opened
        opened = True
        raise AssertionError("valid artifact should not be downloaded again")

    download_verified(
        "https://example.invalid/artifact",
        destination,
        expected_size=4,
        expected_sha256=hashlib.sha256(b"test").hexdigest(),
        opener=opener,
    )

    assert not opened


def test_download_verified_replaces_partial_file_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.with_suffix(".part").write_bytes(b"partial")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    download_verified(
        "https://example.invalid/artifact",
        destination,
        expected_size=4,
        expected_sha256=hashlib.sha256(b"test").hexdigest(),
        opener=lambda _url: Response(b"test"),
    )

    assert destination.read_bytes() == b"test"
    assert not destination.with_suffix(".part").exists()


def test_parser_exposes_approved_workflow_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["doctor"]).command == "doctor"
    assert parser.parse_args(["setup"]).command == "setup"
    assert parser.parse_args(["smoke"]).command == "smoke"
    assert parser.parse_args(["benchmark"]).command == "benchmark"
    assert parser.parse_args(["chat"]).command == "chat"


def test_doctor_returns_failure_and_json_when_artifacts_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "bitnet.toml"
    config_path.write_text(
        """
[runtime]
repository = "https://github.com/microsoft/BitNet.git"
commit = "0fdaa16ae3b33876dbe8d79b2bbcf152f18560f8"
[model]
repository = "microsoft/bitnet-b1.58-2B-4T-gguf"
revision = "a1f2f1c765812aa8af3f6eda4a313707064bba15"
filename = "ggml-model-i2_s.gguf"
size = 1187801280
sha256 = "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
[benchmark]
prompt_tokens = [128, 512]
generated_tokens = 128
threads = [1, 2, 4, 8, 16]
repetitions = 5
[chat]
threads = 8
context = 4096
temperature = 0.7
""".strip(),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path), "doctor"], repository_root=tmp_path)

    assert exit_code == 1
    output = capsys.readouterr().out
    assert '"ready": false' in output
    assert '"model": false' in output


def test_run_setup_provisions_in_dependency_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(bitnet_eval, "provision_toolchain", lambda _cfg: calls.append("tools"))
    monkeypatch.setattr(bitnet_eval, "checkout_runtime", lambda _cfg: calls.append("runtime"))
    monkeypatch.setattr(bitnet_eval, "download_model", lambda _cfg: calls.append("model"))
    monkeypatch.setattr(bitnet_eval, "build_runtime", lambda _cfg: calls.append("build"))
    monkeypatch.setattr(bitnet_eval, "write_setup_manifest", lambda _cfg: calls.append("manifest"))
    monkeypatch.setattr(bitnet_eval, "doctor_report", lambda _cfg: {"ready": True})

    report = run_setup(cfg)

    assert report == {"ready": True}
    assert calls == ["tools", "runtime", "model", "build", "manifest"]


def test_require_ready_reports_failed_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bitnet_eval,
        "doctor_report",
        lambda _cfg: {"ready": False, "model": False, "llama_cli": True},
    )

    with pytest.raises(RuntimeError, match="model"):
        require_ready(config(tmp_path))


def test_parse_benchmark_output_accepts_official_json_array() -> None:
    rows = parse_benchmark_output(
        '[{"model_filename":"model.gguf","test":"pp512","avg_ts":42.5}]'
    )

    assert rows == [{"model_filename": "model.gguf", "test": "pp512", "avg_ts": 42.5}]


@pytest.mark.parametrize("output", ["not json", "{}", "[1]"])
def test_parse_benchmark_output_rejects_unusable_results(output: str) -> None:
    with pytest.raises(RuntimeError, match="benchmark JSON"):
        parse_benchmark_output(output)


def test_parse_time_metrics_extracts_peak_resident_memory() -> None:
    metrics = parse_time_metrics(
        "Elapsed (wall clock) time: 0:02.50\nMaximum resident set size (kbytes): 123456\n"
    )

    assert metrics == {"peak_rss_kib": 123456}


def test_smoke_preserves_fixed_prompts_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(tmp_path)
    monkeypatch.setattr(bitnet_eval, "require_ready", lambda _cfg: {"ready": True})
    monkeypatch.setattr(bitnet_eval, "_ensure_idle", lambda **_kwargs: None)
    monkeypatch.setattr(
        bitnet_eval,
        "_capture",
        lambda command: subprocess.CompletedProcess(command, 0, "answer", ""),
    )

    run_dir = run_smoke(
        cfg,
        allow_contention=False,
        now=datetime(2026, 6, 21, 12, 34, 56, tzinfo=timezone.utc),
    )

    document = __import__("json").loads((run_dir / "smoke.json").read_text())
    assert len(document["results"]) == 5
    assert all(row["stdout"] == "answer" for row in document["results"])
    assert document["claim_boundary"] == "qualitative generation smoke test"


def test_benchmark_aggregates_all_cases_to_json_and_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(tmp_path)
    monkeypatch.setattr(bitnet_eval, "require_ready", lambda _cfg: {"ready": True})
    monkeypatch.setattr(bitnet_eval, "_ensure_idle", lambda **_kwargs: None)
    monkeypatch.setattr(
        bitnet_eval,
        "_capture",
        lambda command: subprocess.CompletedProcess(
            command, 0, '[{"test":"tg128","avg_ts":12.5}]', "runtime info"
        ),
    )

    run_dir = run_benchmark(
        cfg,
        allow_contention=False,
        now=datetime(2026, 6, 21, 12, 34, 56, tzinfo=timezone.utc),
    )

    rows = __import__("json").loads((run_dir / "benchmark.json").read_text())
    assert len(rows) == 10
    assert {row["requested_threads"] for row in rows} == {1, 2, 4, 8, 16}
    assert (run_dir / "benchmark.csv").is_file()
    metadata = __import__("json").loads((run_dir / "metadata.json").read_text())
    assert metadata["model"]["matrix_weight_format"] == "I2_S packed ternary"
    assert metadata["model"]["artifact_bytes"] == 4
    assert set(metadata["toolchain"]) == {"clang", "cmake", "ninja", "python"}
