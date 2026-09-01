"""Same-mix random FFN seeds: experts diverge; eval JSON carries diversity."""

from __future__ import annotations

from pathlib import Path

from seqmoe.assembly import assemble, save_expert, save_trunk
from seqmoe.config import (
    cpu_dump_seed_config,
    cpu_dump_skill_config,
    dump_seed_config,
    dump_skill_config,
    tiny_seed_config,
    tiny_skill_config,
)
from seqmoe.data import build_synth_data
from seqmoe.eval import expert_mlp_diversity
from seqmoe.model import SeqMoE
from seqmoe.train import train_expert


def test_tiny_seed_config_is_cpu_random_mix():
    cfg = tiny_seed_config(device="cuda:1")
    assert cfg.train.device == "cpu"
    assert cfg.train.expert_init == "random"
    assert cfg.train.expert_corpus == "mix"
    assert cfg.model.n_experts == 3
    assert cfg.model.top_k == 2


def test_random_init_same_mix_experts_diverge(tmp_path: Path):
    cfg = tiny_seed_config("cpu")
    cfg.model.n_experts = 2
    cfg.train.pretrain_steps = 2
    cfg.train.expert_steps = 4
    cfg.train.batch_size = 4
    cfg.train.seq_len = 32
    cfg.model.block_size = 32
    data = build_synth_data(seed=0)
    trunk = tmp_path / "trunk.pt"
    m = SeqMoE(cfg.model)
    save_trunk(m, trunk)
    paths = []
    for eid in (0, 1):
        model = SeqMoE.from_trunk(trunk)
        train_expert(model, data.mixed_train, cfg, eid, domain=f"seed{eid}")
        p = tmp_path / f"e{eid}.pt"
        save_expert(model, p, logical_id=eid, domain=f"seed{eid}")
        paths.append(p)
    moe = assemble(trunk, paths)
    div = expert_mlp_diversity(moe)
    assert div["pair_0_1_mean_abs"] > 1e-3
    assert abs(div["pair_0_1_cosine"]) < 0.999
    assert div["expert_0_vs_trunk_mean_abs"] > 1e-3
    assert div["expert_1_vs_trunk_mean_abs"] > 1e-3


def test_cpu_dump_configs():
    """Locked 120-step CPU tables stay CPU; they are not the GPU dump path."""
    sk = cpu_dump_skill_config()
    se = cpu_dump_seed_config()
    assert sk.train.device == se.train.device == "cpu"
    assert sk.train.pretrain_steps == se.train.pretrain_steps == 120
    assert sk.train.scientific is True and se.train.scientific is True
    assert sk.model.n_experts == se.model.n_experts == 3
    assert sk.train.expert_corpus == "domain" and sk.train.expert_init == "clone"
    assert se.train.expert_corpus == "mix" and se.train.expert_init == "random"


def test_dump_configs_honor_cuda_and_scientific_budget():
    sk = dump_skill_config(device="cuda:1")
    se = dump_seed_config(device="cuda:0")
    assert sk.train.device == "cuda:1"
    assert se.train.device == "cuda:0"
    assert dump_skill_config(device="cuda").train.device == "cuda:1"
    assert dump_skill_config(device="cpu").train.device == "cpu"
    for cfg in (sk, se):
        assert cfg.train.scientific is True
        assert cfg.model.n_embd == 256
        assert cfg.model.n_layer == 8
        assert cfg.model.n_experts == 3
        assert cfg.model.top_k == 2
        assert cfg.train.pretrain_steps == 5000
        assert cfg.train.expert_steps == 3000
        assert cfg.train.router_steps == 2000
        assert cfg.train.seq_len == 256
    assert sk.train.expert_init == "clone" and sk.train.expert_corpus == "domain"
    assert se.train.expert_init == "random" and se.train.expert_corpus == "mix"


def test_run_poc_dump_cli_does_not_rewrite_cuda_to_cpu():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_poc.py"
    spec = importlib.util.spec_from_file_location("seqmoe_run_poc_cli", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sk = mod.config_for_args(mod.parse_args(["--dump-skills", "--device", "cuda:1"]))
    se = mod.config_for_args(mod.parse_args(["--dump-seeds", "--device", "cuda:0"]))
    bare = mod.config_for_args(mod.parse_args(["--dump-skills", "--device", "cuda"]))
    defaulted = mod.config_for_args(mod.parse_args(["--dump-skills"]))
    assert sk.train.device == "cuda:1"
    assert se.train.device == "cuda:0"
    assert bare.train.device == defaulted.train.device == "cuda:1"
    assert sk.train.pretrain_steps == 5000
    assert sk.train.expert_init == "clone" and sk.train.expert_corpus == "domain"
    assert se.train.expert_init == "random" and se.train.expert_corpus == "mix"
    assert "stay on CPU" not in Path(path).read_text()


def test_tiny_skill_config_is_cpu_clone_domain():
    cfg = tiny_skill_config(device="cuda:0")
    assert cfg.train.device == "cpu"
    assert cfg.train.expert_init == "clone"
    assert cfg.train.expert_corpus == "domain"
    assert cfg.model.n_experts == 3
