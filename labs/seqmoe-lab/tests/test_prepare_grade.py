"""scientific_grade must not promote a harvest/fallback cache to dump."""

from __future__ import annotations

from pathlib import Path

from seqmoe.data import prepare_scientific


def test_second_prepare_keeps_harvest_labels(tmp_path: Path):
    cache = tmp_path / "cache"
    first = prepare_scientific(cache_dir=cache)
    assert first["code_source"] in {"lab_python", "dump"}
    assert first["math_source"] in {"fallback", "dump"}
    second = prepare_scientific(cache_dir=cache)
    assert second["code_source"] == first["code_source"]
    assert second["math_source"] == first["math_source"]
    assert second["scientific_grade"] == first["scientific_grade"]
    if first["code_source"] != "dump" or first["math_source"] != "dump":
        assert second["scientific_grade"] != "full"


def test_sidecar_prevents_relabel_as_dump(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "code.txt").write_bytes(b"def foo():\n    return 1\n" * 80)
    (cache / "math.txt").write_bytes(b"1+1=2\n" * 80)
    (cache / "sources.json").write_text(
        '{"code_source": "lab_python", "math_source": "fallback"}\n'
    )
    meta = prepare_scientific(cache_dir=cache)
    assert meta["code_source"] == "lab_python"
    assert meta["math_source"] == "fallback"
    assert meta["scientific_grade"] != "full"
