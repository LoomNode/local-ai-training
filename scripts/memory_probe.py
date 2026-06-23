
import torch

from local_ai_training.config import ExperimentConfig
from local_ai_training.model import build_seeded_model


def main():
    device = torch.device("cuda")
    config = ExperimentConfig.from_toml("configs/ratchet_tiny.toml")
    
    print("Memory Probe:")
    print(f"{'Mode':<6} | {'Peak (MB)':>10} | {'Base (MB)':>10} | {'Fwd (MB)':>10} | {'Diff Peak-Fwd':>15}")
    print("-" * 65)

    for mode in ["fp32", "bf16", "int8"]:
        config_dict = config.to_dict()
        config_dict["matmul_mode"] = mode
        run_config = ExperimentConfig(**config_dict)
        
        model = build_seeded_model(
            run_config.model_config(vocab_size=65), max_code=3, seed=42
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        optimizer.step()
        
        B, T = run_config.batch_size, run_config.block_size
        inputs = torch.randint(0, 65, (B, T), device=device)
        targets = torch.randint(0, 65, (B, T), device=device)
        
        # Warmup step to clear JIT compilation peaks
        _, loss = model(inputs, targets)
        loss.backward()
        model.ratchet_update()
        optimizer.zero_grad()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        mem_base = torch.cuda.memory_allocated(device)
        
        _, loss = model(inputs, targets)
        mem_fwd = torch.cuda.memory_allocated(device)
        
        loss.backward()
        mem_peak = torch.cuda.max_memory_allocated(device)
        
        model.ratchet_update()
        
        base_mb = mem_base / 1024**2
        fwd_mb = mem_fwd / 1024**2
        peak_mb = mem_peak / 1024**2
        
        print(f"{mode:<6} | {peak_mb:>10.2f} | {base_mb:>10.2f} | {fwd_mb:>10.2f} | {peak_mb - fwd_mb:>15.2f}")

    print("\n--- Memory Probe (with Checkpointing) ---")
    for mode in ["fp32", "bf16", "int8"]:
        config_dict = config.to_dict()
        config_dict["matmul_mode"] = mode
        config_dict["gradient_checkpointing"] = True
        run_config = ExperimentConfig(**config_dict)
        
        model = build_seeded_model(
            run_config.model_config(vocab_size=65), max_code=3, seed=42
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        optimizer.step()
        
        B, T = run_config.batch_size, run_config.block_size
        inputs = torch.randint(0, 65, (B, T), device=device)
        targets = torch.randint(0, 65, (B, T), device=device)
        
        # Warmup step to clear JIT compilation peaks
        _, loss = model(inputs, targets)
        loss.backward()
        model.ratchet_update()
        optimizer.zero_grad()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        mem_base = torch.cuda.memory_allocated(device)
        
        _, loss = model(inputs, targets)
        mem_fwd = torch.cuda.memory_allocated(device)
        
        loss.backward()
        mem_peak = torch.cuda.max_memory_allocated(device)
        
        model.ratchet_update()
        
        base_mb = mem_base / 1024**2
        fwd_mb = mem_fwd / 1024**2
        peak_mb = mem_peak / 1024**2
        
        print(f"{mode + ' (ckpt)':<12} | {peak_mb:>10.2f} | {base_mb:>10.2f} | {fwd_mb:>10.2f} | {peak_mb - fwd_mb:>15.2f}")

if __name__ == '__main__':
    main()
