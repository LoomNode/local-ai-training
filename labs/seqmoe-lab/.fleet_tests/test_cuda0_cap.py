"""cuda:0 is half a 3090 — default envelope 12 GiB, not the 22 GiB GPU1 envelope."""
from seqmoe.config import tiny_config, scientific_config


def test_tiny_cuda0_default_envelope_is_12_gib():
    cfg = tiny_config("cuda:0")
    assert cfg.train.device == "cuda:0"
    assert cfg.train.max_vram_mb == 12_288


def test_scientific_cuda0_default_envelope_is_12_gib():
    cfg = scientific_config("cuda:0")
    assert cfg.train.device == "cuda:0"
    assert cfg.train.max_vram_mb == 12_288


def test_cuda1_keeps_22_gib():
    cfg = tiny_config("cuda:1")
    assert cfg.train.max_vram_mb == 22_000
