"""Provisioning and command construction for the pinned BitNet runtime."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

from .bitnet_artifacts import BitNetConfig, download_verified, verify_model


def benchmark_cases(config: BitNetConfig) -> list[tuple[int, int]]:
    return [
        (prompt_tokens, threads)
        for prompt_tokens in config.prompt_tokens
        for threads in config.benchmark_threads
    ]


def ensure_training_idle(process_listing: str, *, allow_contention: bool) -> None:
    training_active = any(
        "lat train" in line or "local_ai_training.cli train" in line
        for line in process_listing.splitlines()
    )
    if training_active and not allow_contention:
        raise RuntimeError(
            "active training process detected; wait for it to finish or pass --allow-contention"
        )


def new_run_directory(base: Path, *, now: datetime) -> Path:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    candidate = base / stamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base / f"{stamp}-{suffix:02d}"
    candidate.mkdir(parents=True)
    return candidate


def build_chat_command(config: BitNetConfig, *, system_prompt: str) -> list[str]:
    return [
        str(config.llama_cli),
        "-m",
        str(config.model_path),
        "-n",
        "-1",
        "-t",
        str(config.chat_threads),
        "-p",
        system_prompt,
        "-ngl",
        "0",
        "-c",
        str(config.chat_context),
        "--temp",
        str(config.chat_temperature),
        "-b",
        "1",
        "-cnv",
    ]


def build_smoke_command(config: BitNetConfig, *, prompt: str) -> list[str]:
    return [
        str(config.llama_cli),
        "-m",
        str(config.model_path),
        "-n",
        str(config.generated_tokens),
        "-t",
        str(config.chat_threads),
        "-p",
        prompt,
        "-ngl",
        "0",
        "-c",
        str(config.chat_context),
        "--temp",
        "0.0",
        "--seed",
        "1337",
        "-b",
        "1",
    ]


def build_benchmark_command(
    config: BitNetConfig, *, prompt_tokens: int, threads: int
) -> list[str]:
    return [
        str(config.llama_bench),
        "-m",
        str(config.model_path),
        "-n",
        str(config.generated_tokens),
        "-p",
        str(prompt_tokens),
        "-t",
        str(threads),
        "-r",
        str(config.benchmark_repetitions),
        "-ngl",
        "0",
        "-b",
        "1",
        "-o",
        "json",
    ]


def build_toolchain_create_command(config: BitNetConfig) -> list[str]:
    return [
        str(config.micromamba_binary),
        "create",
        "--yes",
        "--prefix",
        str(config.toolchain_dir),
        "--channel",
        "conda-forge",
        "--strict-channel-priority",
        *config.toolchain_packages,
    ]


def build_toolchain_install_command(config: BitNetConfig) -> list[str]:
    command = build_toolchain_create_command(config)
    command[1] = "install"
    return command


def build_model_download_command(config: BitNetConfig) -> list[str]:
    return [
        str(config.hf_binary),
        "download",
        config.model_repository,
        config.model_filename,
        "--revision",
        config.model_revision,
        "--local-dir",
        str(config.model_dir),
    ]


def extract_micromamba(archive: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as bundle:
        member = bundle.getmember("bin/micromamba")
        source = bundle.extractfile(member)
        if source is None:
            raise RuntimeError("micromamba archive contains no readable bin/micromamba")
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)
    destination.chmod(0o755)


def apply_runtime_compatibility_patch(config: BitNetConfig) -> None:
    source = config.runtime_dir / "src" / "ggml-bitnet-mad.cpp"
    original = "        int8_t * y_col = y + col * by;"
    corrected = "        const int8_t * y_col = y + col * by;"
    text = source.read_text(encoding="utf-8")
    if corrected in text:
        return
    if text.count(original) != 1:
        raise RuntimeError("pinned BitNet const compatibility patch no longer applies cleanly")
    source.write_text(text.replace(original, corrected), encoding="utf-8")


def doctor_report(config: BitNetConfig) -> dict[str, object]:
    checks: dict[str, bool] = {
        "architecture": platform.machine() == "x86_64",
        "toolchain_python": config.toolchain_python.is_file(),
        "clang": (config.toolchain_bin / "clang").is_file(),
        "cmake": (config.toolchain_bin / "cmake").is_file(),
        "ninja": (config.toolchain_bin / "ninja").is_file(),
        "hf": config.hf_binary.is_file(),
        "llama_cli": config.llama_cli.is_file(),
        "llama_bench": config.llama_bench.is_file(),
    }
    runtime_commit = None
    if (config.runtime_dir / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(config.runtime_dir), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        runtime_commit = result.stdout.strip() if result.returncode == 0 else None
    checks["runtime_commit"] = runtime_commit == config.runtime_commit
    try:
        verify_model(
            config.model_path,
            expected_size=config.model_size,
            expected_sha256=config.model_sha256,
        )
        checks["model"] = True
    except RuntimeError:
        checks["model"] = False
    return {
        "ready": all(checks.values()),
        **checks,
        "expected_runtime_commit": config.runtime_commit,
        "actual_runtime_commit": runtime_commit,
        "model_path": str(config.model_path),
    }


def _toolchain_environment(config: BitNetConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = f"{config.toolchain_bin}:{environment.get('PATH', '')}"
    environment["MAMBA_ROOT_PREFIX"] = str(config.data_dir / "mamba-root")
    return environment


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def provision_toolchain(config: BitNetConfig) -> None:
    if not config.micromamba_binary.is_file():
        download_verified(
            config.micromamba_url,
            config.micromamba_archive,
            expected_size=config.micromamba_size,
            expected_sha256=config.micromamba_sha256,
        )
        extract_micromamba(config.micromamba_archive, config.micromamba_binary)
    if not config.toolchain_python.is_file():
        _run(build_toolchain_create_command(config), env=_toolchain_environment(config))
    else:
        _run(build_toolchain_install_command(config), env=_toolchain_environment(config))
    required = (
        config.toolchain_python,
        config.toolchain_bin / "clang",
        config.toolchain_bin / "clang++",
        config.toolchain_bin / "cmake",
        config.toolchain_bin / "ninja",
        config.hf_binary,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"local BitNet toolchain is incomplete: {missing}")


def checkout_runtime(config: BitNetConfig) -> None:
    if not (config.runtime_dir / ".git").exists():
        config.runtime_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--recursive",
                config.runtime_repository,
                str(config.runtime_dir),
            ]
        )
    remote = subprocess.run(
        ["git", "-C", str(config.runtime_dir), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote.rstrip("/") != config.runtime_repository.rstrip("/"):
        raise RuntimeError(
            f"BitNet origin mismatch: expected {config.runtime_repository}, got {remote}"
        )
    _run(["git", "-C", str(config.runtime_dir), "fetch", "origin", config.runtime_commit])
    _run(["git", "-C", str(config.runtime_dir), "checkout", "--detach", config.runtime_commit])
    _run(
        [
            "git",
            "-C",
            str(config.runtime_dir),
            "submodule",
            "update",
            "--init",
            "--recursive",
        ]
    )
    apply_runtime_compatibility_patch(config)


def download_model(config: BitNetConfig) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    if not config.model_path.is_file():
        _run(build_model_download_command(config), env=_toolchain_environment(config))
    verify_model(
        config.model_path,
        expected_size=config.model_size,
        expected_sha256=config.model_sha256,
    )


def build_runtime(config: BitNetConfig) -> None:
    if config.llama_cli.is_file() and config.llama_bench.is_file():
        return
    environment = _toolchain_environment(config)
    _run(
        [
            str(config.toolchain_python),
            "-m",
            "pip",
            "install",
            "-r",
            str(config.runtime_dir / "requirements.txt"),
        ],
        cwd=config.runtime_dir,
        env=environment,
    )
    shutil.rmtree(config.runtime_dir / "build", ignore_errors=True)
    _run(
        [
            str(config.toolchain_python),
            "setup_env.py",
            "--model-dir",
            str(config.model_dir),
            "--log-dir",
            str(config.data_dir / "setup-logs"),
            "--quant-type",
            "i2_s",
        ],
        cwd=config.runtime_dir,
        env=environment,
    )
    if not config.llama_cli.is_file() or not config.llama_bench.is_file():
        raise RuntimeError("BitNet build completed without llama-cli and llama-bench")


def write_setup_manifest(config: BitNetConfig) -> None:
    environment = _toolchain_environment(config)
    explicit = subprocess.run(
        [
            str(config.micromamba_binary),
            "list",
            "--explicit",
            "--prefix",
            str(config.toolchain_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.splitlines()
    pip_freeze = subprocess.run(
        [str(config.toolchain_python), "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.splitlines()
    manifest = {
        "runtime": {
            "repository": config.runtime_repository,
            "commit": config.runtime_commit,
            "compatibility_patch": "make read-only y_col activation pointer const",
        },
        "model": {
            "repository": config.model_repository,
            "revision": config.model_revision,
            "filename": config.model_filename,
            "artifact_bytes": config.model_size,
            "sha256": config.model_sha256,
            "matrix_weight_format": "I2_S packed ternary",
            "note": (
                "GGUF artifact bytes include metadata and non-matrix tensors; runtime and "
                "KV-cache memory are excluded"
            ),
        },
        "toolchain": {
            "micromamba_url": config.micromamba_url,
            "micromamba_sha256": config.micromamba_sha256,
            "explicit_environment": explicit,
            "pip_freeze": pip_freeze,
        },
    }
    manifest_path = config.data_dir / "setup-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
