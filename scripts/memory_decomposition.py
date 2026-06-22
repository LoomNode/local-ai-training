"""Measure BF16/int8 training-memory attribution at the 2048-width sweep point.

The parent process launches every measured step in a fresh child process.  This is
measurement-only instrumentation: it wraps the ratchet module's imported quantization/GEMM
functions and uses saved-tensor hooks without changing training math or persistent state.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import local_ai_training.ratchet as ratchet_module
from local_ai_training.metrics import collect_ratchet_metrics
from local_ai_training.model import ModelConfig, build_seeded_model
from local_ai_training.ratchet import DiscreteRatchetLinear

MIB = 1024**2
DEFAULT_OUTPUT = Path("runs/memory-decomposition")
MODEL = {
    "vocab_size": 65,
    "block_size": 32,
    "n_layer": 16,
    "n_head": 16,
    "n_embd": 2048,
    "dropout": 0.0,
    "gradient_checkpointing": True,
}
BATCH_SIZE = 2
SEED = 1337


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def json_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def storage_key(tensor: Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _summary(values: list[int]) -> dict[str, int | float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one completed run is required")
    modes = {run["mode"] for run in runs}
    if len(modes) != 1:
        raise ValueError("runs must use one mode")
    result: dict[str, Any] = {"mode": modes.pop(), "run_count": len(runs)}
    numeric_keys = set.intersection(
        *[{key for key, value in run.items() if isinstance(value, (int, float))} for run in runs]
    )
    for key in sorted(numeric_keys):
        result[key] = _summary([run[key] for run in runs])
    return result


def _cuda_allocated() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def _tensor_inventory(model, optimizer) -> dict[str, Any]:
    groups: dict[tuple[str, str], int] = defaultdict(int)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    ratchet_names = {
        f"{module_name}.{name}" if module_name else name
        for module_name, module in model.named_modules()
        if isinstance(module, DiscreteRatchetLinear)
        for name, _ in module.named_buffers(recurse=False)
    }
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        state = "ratchet" if name in ratchet_names else "support"
        key = (state, str(tensor.dtype))
        groups[key] += tensor_bytes(tensor)
        counts[key] += 1
    for state in optimizer.state.values():
        for tensor in state.values():
            if isinstance(tensor, Tensor):
                key = ("optimizer", str(tensor.dtype))
                groups[key] += tensor_bytes(tensor)
                counts[key] += 1
    return {
        f"{state}:{dtype}": {
            "bytes": groups[(state, dtype)],
            "tensor_count": counts[(state, dtype)],
        }
        for state, dtype in sorted(groups)
    }


class SavedTensorRecorder:
    def __init__(self, model) -> None:
        self.active_block = "outside_blocks"
        self.seen: set[tuple[str, int, int]] = set()
        self.bytes: dict[str, int] = defaultdict(int)
        self.storages: dict[str, int] = defaultdict(int)
        self.handles = []
        self.original_checkpoint = torch.utils.checkpoint.checkpoint
        block_names = {id(block): f"block_{index:02d}" for index, block in enumerate(model.blocks)}

        def checkpoint(function, *args, **kwargs):
            previous = self.active_block
            self.active_block = block_names.get(id(function), "outside_blocks")
            try:
                return self.original_checkpoint(function, *args, **kwargs)
            finally:
                self.active_block = previous

        torch.utils.checkpoint.checkpoint = checkpoint
        for index, block in enumerate(model.blocks):
            name = f"block_{index:02d}"
            self.handles.append(block.register_forward_pre_hook(self._pre(name)))
            self.handles.append(block.register_forward_hook(self._post))

    def _pre(self, name):
        def hook(_module, _inputs):
            self.active_block = name
        return hook

    def _post(self, _module, _inputs, _output):
        self.active_block = "outside_blocks"

    def pack(self, tensor):
        key = storage_key(tensor)
        if key not in self.seen:
            self.seen.add(key)
            self.bytes[self.active_block] += key[2]
            self.storages[self.active_block] += 1
        return tensor

    @staticmethod
    def unpack(tensor):
        return tensor

    def close(self):
        torch.utils.checkpoint.checkpoint = self.original_checkpoint
        for handle in self.handles:
            handle.remove()

    def report(self):
        return {
            name: {"bytes": self.bytes[name], "unique_storages": self.storages[name]}
            for name in sorted(self.bytes)
        }


class OperationRecorder:
    def __init__(self) -> None:
        self.phase = "warmup"
        self.calls: list[dict[str, Any]] = []
        self.originals = {}
        self.measure_call_peaks = False

    def _wrap(self, name, function):
        def wrapped(*args, **kwargs):
            before = _cuda_allocated()
            if self.measure_call_peaks:
                torch.cuda.reset_peak_memory_stats()
            result = function(*args, **kwargs)
            after = _cuda_allocated()
            peak = torch.cuda.max_memory_allocated() if self.measure_call_peaks else None
            outputs = result if isinstance(result, tuple) else (result,)
            self.calls.append({
                "phase": self.phase,
                "operation": name,
                "input_shapes": [list(arg.shape) for arg in args if isinstance(arg, Tensor)],
                "output_shapes": [list(item.shape) for item in outputs if isinstance(item, Tensor)],
                "output_bytes": sum(
                    tensor_bytes(item) for item in outputs if isinstance(item, Tensor)
                ),
                "allocated_before": before,
                "allocated_after": after,
                "peak_bytes": peak,
                "transient_peak_delta_bytes": peak - before if peak is not None else None,
            })
            return result
        return wrapped

    def __enter__(self):
        for name in ("quantize_rows", "quantize_columns", "scaled_int8_mm"):
            original = getattr(ratchet_module, name)
            self.originals[name] = original
            setattr(ratchet_module, name, self._wrap(name, original))
        return self

    def __exit__(self, *_exc):
        for name, original in self.originals.items():
            setattr(ratchet_module, name, original)

    def report(self):
        measured = [call for call in self.calls if call["phase"] == "diagnostic"]
        by_operation = {}
        for name in ("quantize_rows", "quantize_columns", "scaled_int8_mm"):
            calls = [call for call in measured if call["operation"] == name]
            by_operation[name] = {
                "calls": len(calls),
                "output_bytes_sum": sum(call["output_bytes"] for call in calls),
                "max_output_bytes": max((call["output_bytes"] for call in calls), default=0),
                "max_transient_peak_delta_bytes": max(
                    (call["transient_peak_delta_bytes"] for call in calls), default=0
                ),
            }
        return {"summary": by_operation, "calls": measured}


def _support_gradient_bytes(model) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for parameter in model.parameters():
        if parameter.grad is not None:
            result[str(parameter.grad.dtype)] += tensor_bytes(parameter.grad)
    return dict(sorted(result.items()))


@contextmanager
def _saved_hooks(recorder):
    with torch.autograd.graph.saved_tensors_hooks(recorder.pack, recorder.unpack):
        yield


def run_child(mode: str, output: Path, run_index: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory decomposition")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = ModelConfig(**MODEL, matmul_mode=mode)
    model = build_seeded_model(config, max_code=2, seed=SEED).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    inputs = torch.randint(0, MODEL["vocab_size"], (BATCH_SIZE, MODEL["block_size"]), device="cuda")
    targets = torch.randint(
        0, MODEL["vocab_size"], (BATCH_SIZE, MODEL["block_size"]), device="cuda"
    )

    with OperationRecorder() as operations:
        # Warm a complete step, including Triton autotuning and Adam state allocation.
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        model.ratchet_update()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        initialization_and_warmup_peak = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()

        persistent_allocated = _cuda_allocated()
        inventory = _tensor_inventory(model, optimizer)
        saved = SavedTensorRecorder(model)
        torch.cuda.reset_peak_memory_stats()
        operations.phase = "forward"
        with _saved_hooks(saved):
            _, loss = model(inputs, targets)
            assert loss is not None
            forward_retained = _cuda_allocated()
            forward_peak = torch.cuda.max_memory_allocated()
            operations.phase = "backward"
            loss.backward()
        post_backward = _cuda_allocated()
        full_peak = torch.cuda.max_memory_allocated()
        support_gradients = _support_gradient_bytes(model)
        saved.close()

        model.ratchet_update()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_allocated = _cuda_allocated()

        # Reproduce the two non-training regions included in train.py's cumulative metric.
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model.eval()
            model(inputs, targets)
            model.train()
        evaluation_peak = torch.cuda.max_memory_allocated()
        evaluation_completed = _cuda_allocated()

        torch.cuda.reset_peak_memory_stats()
        collect_ratchet_metrics(model)
        observability_peak = torch.cuda.max_memory_allocated()
        observability_completed = _cuda_allocated()

        # Operation-local peaks require resetting CUDA's global peak counter.  Run a second,
        # explicitly diagnostic step only after the primary full-step measurements are final.
        operations.phase = "diagnostic"
        operations.measure_call_peaks = True
        _, diagnostic_loss = model(inputs, targets)
        assert diagnostic_loss is not None
        diagnostic_loss.backward()
        model.ratchet_update()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    saved_report = saved.report()
    saved_bytes = sum(item["bytes"] for item in saved_report.values())
    support_gradient_total = sum(support_gradients.values())
    result = {
        "complete": True,
        "mode": mode,
        "run_index": run_index,
        "seed": SEED,
        "configuration": {**MODEL, "batch_size": BATCH_SIZE, "max_code": 2},
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "uuid": json_value(torch.cuda.get_device_properties(0).uuid),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "persistent_allocated_bytes": persistent_allocated,
        "initialization_and_warmup_peak_bytes": initialization_and_warmup_peak,
        "persistent_inventory": inventory,
        "saved_tensors_by_block": saved_report,
        "saved_tensor_unique_bytes": saved_bytes,
        "support_gradient_by_dtype": support_gradients,
        "support_gradient_bytes": support_gradient_total,
        "forward_retained_bytes": forward_retained,
        "forward_peak_bytes": forward_peak,
        "backward_peak_bytes": full_peak,
        "post_backward_allocated_bytes": post_backward,
        "completed_step_allocated_bytes": completed_allocated,
        "evaluation_peak_bytes": evaluation_peak,
        "evaluation_completed_allocated_bytes": evaluation_completed,
        "observability_peak_bytes": observability_peak,
        "observability_completed_allocated_bytes": observability_completed,
        "peak_minus_persistent_bytes": full_peak - persistent_allocated,
        "residual_after_persistent_saved_final_grads_bytes": (
            full_peak - persistent_allocated - saved_bytes - support_gradient_total
        ),
        "operations": operations.report(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


def run_parent(output_dir: Path, repeats: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_runs = []
    for mode in ("bf16", "int8"):
        for run_index in range(repeats):
            path = output_dir / f"{mode}-run-{run_index + 1}.json"
            command = [
                sys.executable, __file__, "--child", "--mode", mode,
                "--run-index", str(run_index + 1), "--output", str(path),
            ]
            subprocess.run(command, check=True)
            result = json.loads(path.read_text())
            if not result.get("complete"):
                raise RuntimeError(f"incomplete measurement: {path}")
            all_runs.append(result)
    summary = {
        "configuration": {**MODEL, "batch_size": BATCH_SIZE, "seed": SEED, "repeats": repeats},
        "modes": {
            mode: aggregate_runs([run for run in all_runs if run["mode"] == mode])
            for mode in ("bf16", "int8")
        },
        "matched_peak_difference_bytes": {
            "median": statistics.median(
                run["backward_peak_bytes"] for run in all_runs if run["mode"] == "int8"
            ) - statistics.median(
                run["backward_peak_bytes"] for run in all_runs if run["mode"] == "bf16"
            )
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("bf16", "int8"), help=argparse.SUPPRESS)
    parser.add_argument("--run-index", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.child:
        if args.mode is None or args.output is None:
            raise SystemExit("--child requires --mode and --output")
        run_child(args.mode, args.output, args.run_index)
    else:
        if args.repeats < 2:
            raise SystemExit("--repeats must be at least 2")
        run_parent(args.output_dir, args.repeats)


if __name__ == "__main__":
    main()
