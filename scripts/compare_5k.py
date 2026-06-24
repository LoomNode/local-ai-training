import os

# Force GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

from local_ai_training.config import ExperimentConfig
from local_ai_training.data import build_char_corpus, download_tiny_shakespeare
from local_ai_training.train import train_run


def main():
    # 1. Download and build corpus
    cache_dir = Path("data/huggingface")
    dataset_path = download_tiny_shakespeare(cache_dir)
    text = dataset_path.read_text(encoding="utf-8")
    corpus = build_char_corpus(text)

    # 2. Load base config and modify to 5000 steps
    config_path = Path("configs/ratchet_tiny.toml")
    config = ExperimentConfig.from_toml(config_path)
    # We use a dataclass replace to change steps
    from dataclasses import replace
    config = replace(config, steps=5000, eval_interval=500)

    # 3. Train bf16 baseline
    print("\n--- Training bf16 baseline for 5000 steps ---")
    config_bf16 = replace(config, matmul_mode="bf16")
    bf16_result = train_run(
        corpus=corpus,
        config=config_bf16,
        max_code=2,  # quinary ratchet
        seed=1337,
        run_dir=Path("runs/compare_5k_bf16"),
        weight_mode="ratchet"
    )

    # 4. Train int8 ACT mode
    print("\n--- Training int8 ACT for 5000 steps ---")
    config_int8 = replace(config, matmul_mode="int8")
    int8_result = train_run(
        corpus=corpus,
        config=config_int8,
        max_code=2,  # quinary ratchet
        seed=1337,
        run_dir=Path("runs/compare_5k_int8"),
        weight_mode="ratchet"
    )

    print("\n\n=== 5k Step Comparison Complete ===")
    print(f"bf16 Final Validation Loss: {bf16_result.final_validation_loss:.4f}")
    print(f"int8 Final Validation Loss: {int8_result.final_validation_loss:.4f}")

if __name__ == "__main__":
    main()
