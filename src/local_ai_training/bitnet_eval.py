"""Reproducible orchestration for the external Microsoft BitNet runtime."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from .bitnet_artifacts import (
    BitNetConfig,
    download_verified,
    load_config,
    verify_model,
)
from .bitnet_runtime import (
    benchmark_cases,
    build_benchmark_command,
    build_chat_command,
    build_model_download_command,
    build_runtime,
    build_smoke_command,
    build_toolchain_create_command,
    checkout_runtime,
    doctor_report,
    download_model,
    ensure_training_idle,
    extract_micromamba,
    new_run_directory,
    provision_toolchain,
    write_setup_manifest,
)

__all__ = [
    "BitNetConfig",
    "benchmark_cases",
    "build_benchmark_command",
    "build_chat_command",
    "build_model_download_command",
    "build_parser",
    "build_smoke_command",
    "build_toolchain_create_command",
    "download_verified",
    "ensure_training_idle",
    "extract_micromamba",
    "load_config",
    "main",
    "new_run_directory",
    "parse_benchmark_output",
    "parse_time_metrics",
    "require_ready",
    "run_benchmark",
    "run_setup",
    "run_smoke",
    "verify_model",
]


def run_setup(config: BitNetConfig) -> dict[str, object]:
    provision_toolchain(config)
    checkout_runtime(config)
    download_model(config)
    build_runtime(config)
    write_setup_manifest(config)
    report = doctor_report(config)
    if not report["ready"]:
        raise RuntimeError(f"BitNet setup finished but doctor failed: {report}")
    return report


def require_ready(config: BitNetConfig) -> dict[str, object]:
    report = doctor_report(config)
    if not report["ready"]:
        failures = sorted(
            key for key, value in report.items() if isinstance(value, bool) and not value
        )
        raise RuntimeError(f"BitNet setup is not ready; failed checks: {', '.join(failures)}")
    return report


def parse_benchmark_output(output: str) -> list[dict[str, object]]:
    try:
        document = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("invalid benchmark JSON output") from error
    if not isinstance(document, list) or not all(isinstance(row, dict) for row in document):
        raise RuntimeError("benchmark JSON output must be an array of objects")
    return document


def parse_time_metrics(output: str) -> dict[str, int]:
    prefix = "Maximum resident set size (kbytes):"
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return {"peak_rss_kib": int(stripped.removeprefix(prefix).strip())}
    raise RuntimeError("time output contains no maximum resident set size")


SMOKE_PROMPTS = (
    "What is 84 * 3 / 2? Explain briefly.",
    "Explain quantum computing to a curious twelve-year-old in three sentences.",
    "Write a Python function that returns the first repeated item in a list.",
    "Return only JSON with keys name and value for the fact: alpha equals 7391.",
    (
        "Remember that the verification key is cobalt-fern-924. Explain why reproducible "
        "benchmarks matter, then finish by repeating the verification key exactly."
    ),
)


def _capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _ensure_idle(*, allow_contention: bool) -> None:
    listing = subprocess.run(
        ["ps", "-eo", "pid,args"], check=True, capture_output=True, text=True
    ).stdout
    ensure_training_idle(listing, allow_contention=allow_contention)


def _timestamped_run(config: BitNetConfig, category: str, now: datetime | None) -> Path:
    return new_run_directory(
        config.repository_root / "runs" / "bitnet" / category,
        now=now or datetime.now(timezone.utc),
    )


def run_smoke(
    config: BitNetConfig, *, allow_contention: bool, now: datetime | None = None
) -> Path:
    require_ready(config)
    _ensure_idle(allow_contention=allow_contention)
    run_dir = _timestamped_run(config, "smoke", now)
    results = []
    for index, prompt in enumerate(SMOKE_PROMPTS, start=1):
        command = build_smoke_command(config, prompt=prompt)
        completed = _capture(command)
        row = {
            "index": index,
            "prompt": prompt,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        results.append(row)
        (run_dir / f"prompt-{index:02d}.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (run_dir / f"prompt-{index:02d}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            (run_dir / "smoke.json").write_text(
                json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(
                f"BitNet smoke prompt {index} failed with exit code {completed.returncode}"
            )
    document = {
        "claim_boundary": "qualitative generation smoke test",
        "temperature": 0.0,
        "seed": 1337,
        "results": results,
    }
    (run_dir / "smoke.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def _benchmark_metadata(config: BitNetConfig) -> dict[str, object]:
    def version(command: list[str]) -> str | None:
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError:
            return None
        output = completed.stdout or completed.stderr
        return output.splitlines()[0] if completed.returncode == 0 and output else None

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "toolchain": {
            "clang": version([str(config.toolchain_bin / "clang"), "--version"]),
            "cmake": version([str(config.toolchain_bin / "cmake"), "--version"]),
            "ninja": version([str(config.toolchain_bin / "ninja"), "--version"]),
            "python": version([str(config.toolchain_python), "--version"]),
        },
        "runtime": {
            "repository": config.runtime_repository,
            "commit": config.runtime_commit,
            "backend": "CPU",
            "kernel": "I2_S",
        },
        "model": {
            "repository": config.model_repository,
            "revision": config.model_revision,
            "filename": config.model_filename,
            "artifact_bytes": config.model_size,
            "sha256": config.model_sha256,
            "matrix_weight_format": "I2_S packed ternary",
            "note": (
                "GGUF artifact bytes include metadata and non-matrix tensors; peak RSS is "
                "reported separately by /usr/bin/time"
            ),
        },
        "benchmark": {
            "prompt_tokens": list(config.prompt_tokens),
            "generated_tokens": config.generated_tokens,
            "threads": list(config.benchmark_threads),
            "repetitions": config.benchmark_repetitions,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def run_benchmark(
    config: BitNetConfig, *, allow_contention: bool, now: datetime | None = None
) -> Path:
    require_ready(config)
    _ensure_idle(allow_contention=allow_contention)
    run_dir = _timestamped_run(config, "benchmark", now)
    rows: list[dict[str, object]] = []
    for prompt_tokens, threads in benchmark_cases(config):
        stem = f"prompt-{prompt_tokens}-threads-{threads}"
        time_path = run_dir / f"{stem}.time.txt"
        command = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_path),
            *build_benchmark_command(
                config, prompt_tokens=prompt_tokens, threads=threads
            ),
        ]
        completed = _capture(command)
        (run_dir / f"{stem}.stdout.json").write_text(completed.stdout, encoding="utf-8")
        (run_dir / f"{stem}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"BitNet benchmark failed for prompt={prompt_tokens}, threads={threads}"
            )
        time_metrics = (
            parse_time_metrics(time_path.read_text(encoding="utf-8"))
            if time_path.is_file()
            else {}
        )
        for row in parse_benchmark_output(completed.stdout):
            rows.append(
                {
                    **row,
                    **time_metrics,
                    "requested_prompt_tokens": prompt_tokens,
                    "requested_generated_tokens": config.generated_tokens,
                    "requested_threads": threads,
                    "time_log": time_path.name,
                }
            )
    (run_dir / "benchmark.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(run_dir / "benchmark.csv", rows)
    (run_dir / "metadata.json").write_text(
        json.dumps(_benchmark_metadata(config), indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def run_chat(
    config: BitNetConfig,
    *,
    system_prompt: str,
    allow_contention: bool,
) -> int:
    require_ready(config)
    _ensure_idle(allow_contention=allow_contention)
    return subprocess.run(
        build_chat_command(config, system_prompt=system_prompt), check=False
    ).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bitnet_inference.toml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="verify the local BitNet runtime and model")
    subparsers.add_parser("setup", help="provision the pinned local runtime and model")
    for command in ("smoke", "benchmark"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--allow-contention", action="store_true")
    chat = subparsers.add_parser("chat", help="start an interactive BitNet conversation")
    chat.add_argument("--allow-contention", action="store_true")
    chat.add_argument("--system-prompt", default="You are a helpful assistant.")
    return parser


def main(
    argv: Sequence[str] | None = None, *, repository_root: Path | None = None
) -> int:
    args = build_parser().parse_args(argv)
    root = repository_root or Path(__file__).resolve().parents[2]
    config = load_config(args.config, repository_root=root)
    if args.command == "doctor":
        report = doctor_report(config)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1
    if args.command == "setup":
        print(json.dumps(run_setup(config), indent=2))
        return 0
    if args.command == "smoke":
        print(run_smoke(config, allow_contention=args.allow_contention))
        return 0
    if args.command == "benchmark":
        print(run_benchmark(config, allow_contention=args.allow_contention))
        return 0
    if args.command == "chat":
        return run_chat(
            config,
            system_prompt=args.system_prompt,
            allow_contention=args.allow_contention,
        )
    raise AssertionError(f"unhandled command: {args.command}")
