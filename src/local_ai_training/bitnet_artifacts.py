"""Pinned artifact configuration and integrity helpers for BitNet inference."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class BitNetConfig:
    repository_root: Path
    runtime_repository: str
    runtime_commit: str
    model_repository: str
    model_revision: str
    model_filename: str
    model_size: int
    model_sha256: str
    prompt_tokens: tuple[int, ...]
    generated_tokens: int
    benchmark_threads: tuple[int, ...]
    benchmark_repetitions: int
    chat_threads: int
    chat_context: int
    chat_temperature: float
    micromamba_url: str = (
        "https://api.anaconda.org/download/conda-forge/micromamba/2.8.1/"
        "linux-64/micromamba-2.8.1-0.tar.bz2"
    )
    micromamba_size: int = 6_881_647
    micromamba_sha256: str = "a934c3709c997feae403a27fd1e321c106d26ffa4f294800ffb11cbc9a3e8515"
    toolchain_packages: tuple[str, ...] = (
        "python=3.11",
        "clang=18",
        "cmake=3.31",
        "ninja=1.12",
        "pip=25.1",
        "huggingface_hub=0.33",
    )

    def __post_init__(self) -> None:
        hashes = {
            "runtime_commit": (self.runtime_commit, 40),
            "model_revision": (self.model_revision, 40),
            "model_sha256": (self.model_sha256, 64),
            "micromamba_sha256": (self.micromamba_sha256, 64),
        }
        for name, (value, length) in hashes.items():
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"{name} must be hexadecimal") from error
            if len(value) != length:
                raise ValueError(f"{name} must contain {length} hexadecimal characters")
        positive_values = (
            self.model_size,
            self.micromamba_size,
            self.generated_tokens,
            self.benchmark_repetitions,
            self.chat_threads,
            self.chat_context,
            *self.prompt_tokens,
            *self.benchmark_threads,
        )
        if min(positive_values) <= 0:
            raise ValueError(
                "BitNet sizes, token counts, repetitions, and threads must be positive"
            )
        if self.chat_temperature < 0:
            raise ValueError("chat temperature must be non-negative")

    @property
    def data_dir(self) -> Path:
        return self.repository_root / "data" / "bitnet"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir / "runtime"

    @property
    def tools_dir(self) -> Path:
        return self.data_dir / "tools"

    @property
    def micromamba_archive(self) -> Path:
        return self.tools_dir / "micromamba-2.8.1-0.tar.bz2"

    @property
    def micromamba_binary(self) -> Path:
        return self.tools_dir / "micromamba" / "bin" / "micromamba"

    @property
    def toolchain_dir(self) -> Path:
        return self.data_dir / "env"

    @property
    def toolchain_bin(self) -> Path:
        return self.toolchain_dir / "bin"

    @property
    def toolchain_python(self) -> Path:
        return self.toolchain_bin / "python"

    @property
    def hf_binary(self) -> Path:
        return self.toolchain_bin / "hf"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models" / "BitNet-b1.58-2B-4T"

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    @property
    def llama_cli(self) -> Path:
        return self.runtime_dir / "build" / "bin" / "llama-cli"

    @property
    def llama_bench(self) -> Path:
        return self.runtime_dir / "build" / "bin" / "llama-bench"


def load_config(path: Path, *, repository_root: Path) -> BitNetConfig:
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    runtime = document["runtime"]
    model = document["model"]
    benchmark = document["benchmark"]
    chat = document["chat"]
    toolchain = document.get("toolchain", {})
    fields = BitNetConfig.__dataclass_fields__
    return BitNetConfig(
        repository_root=repository_root.resolve(),
        runtime_repository=str(runtime["repository"]),
        runtime_commit=str(runtime["commit"]),
        model_repository=str(model["repository"]),
        model_revision=str(model["revision"]),
        model_filename=str(model["filename"]),
        model_size=int(model["size"]),
        model_sha256=str(model["sha256"]),
        prompt_tokens=tuple(int(value) for value in benchmark["prompt_tokens"]),
        generated_tokens=int(benchmark["generated_tokens"]),
        benchmark_threads=tuple(int(value) for value in benchmark["threads"]),
        benchmark_repetitions=int(benchmark["repetitions"]),
        chat_threads=int(chat["threads"]),
        chat_context=int(chat["context"]),
        chat_temperature=float(chat["temperature"]),
        micromamba_url=str(
            toolchain.get("micromamba_url", fields["micromamba_url"].default)
        ),
        micromamba_size=int(
            toolchain.get("micromamba_size", fields["micromamba_size"].default)
        ),
        micromamba_sha256=str(
            toolchain.get("micromamba_sha256", fields["micromamba_sha256"].default)
        ),
        toolchain_packages=tuple(
            str(value)
            for value in toolchain.get("packages", fields["toolchain_packages"].default)
        ),
    )


def verify_model(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"model file is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"model size mismatch: expected {expected_size}, got {actual_size}: {path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"model SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )


def download_verified(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    opener=urllib.request.urlopen,
) -> None:
    if destination.is_file():
        try:
            verify_model(
                destination,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            return
        except RuntimeError:
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with opener(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    verify_model(partial, expected_size=expected_size, expected_sha256=expected_sha256)
    partial.replace(destination)
