"""Tiny SCRT train on GPU0. Param count must not grow; stay under half a 3090."""
import torch
from scrt.config import ExperimentConfig, ModelConfig, TrainConfig
from scrt.train import train_arm


def test_scrt_two_steps_on_cuda0():
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
    cfg = ExperimentConfig(
        model=ModelConfig(
            vocab_size=32,
            n_embd=32,
            n_head=4,
            block_size=16,
            n_layer=2,
            n_loops=2,
            n_slots=4,
            lora_rank=2,
            slots_per_loop=1,
        ),
        train=TrainConfig(
            seed=0,
            batch_size=4,
            seq_len=16,
            steps=2,
            lr=3e-3,
            eval_every=2,
            consolidate_every=2,
            log_every=2,
            probe_batches=1,
            device="cuda:0",
            arm="scrt",
            run_name="fleet_cuda0",
            results_dir="results",
        ),
    )
    r = train_arm(cfg, write_results=False)
    assert r.n_params > 0
    assert r.final_eval_loss > 0
    for entry in r.consolidate_log:
        assert entry["n_params"] == r.n_params
    assert torch.cuda.max_memory_allocated(0) < 12 * 1024 * 1024 * 1024
