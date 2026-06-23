import importlib.util
import json
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location(
    "memory_decomposition", Path(__file__).parents[1] / "scripts" / "memory_decomposition.py"
)
assert SPEC is not None and SPEC.loader is not None
memory_decomposition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory_decomposition)
aggregate_runs = memory_decomposition.aggregate_runs
storage_key = memory_decomposition.storage_key
tensor_bytes = memory_decomposition.tensor_bytes


def test_tensor_bytes_uses_logical_tensor_size():
    tensor = torch.empty((3, 5), dtype=torch.bfloat16)

    assert tensor_bytes(tensor) == 30


def test_storage_key_deduplicates_views():
    tensor = torch.arange(12)

    assert storage_key(tensor) == storage_key(tensor.view(3, 4))


def test_aggregate_runs_reports_median_and_full_spread():
    runs = [
        {"mode": "int8", "peak_bytes": 90, "forward_retained_bytes": 30},
        {"mode": "int8", "peak_bytes": 110, "forward_retained_bytes": 40},
        {"mode": "int8", "peak_bytes": 100, "forward_retained_bytes": 35},
    ]

    aggregate = aggregate_runs(runs)

    assert aggregate["mode"] == "int8"
    assert aggregate["run_count"] == 3
    assert aggregate["peak_bytes"] == {"median": 100, "min": 90, "max": 110}
    assert aggregate["forward_retained_bytes"] == {"median": 35, "min": 30, "max": 40}


def test_json_value_converts_foreign_uuid_objects_to_strings():
    class DeviceUuid:
        def __str__(self):
            return "GPU-test-uuid"

    assert json.dumps({"uuid": memory_decomposition.json_value(DeviceUuid())}) == (
        '{"uuid": "GPU-test-uuid"}'
    )
