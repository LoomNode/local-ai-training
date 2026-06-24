import torch
from torch.profiler import profile, record_function, ProfilerActivity
from local_ai_training.model import ModelConfig, build_seeded_model

def run_profiler():
    config = ModelConfig(vocab_size=50257, block_size=256, n_layer=4, n_head=8, n_embd=512, matmul_mode="int8")
    model = build_seeded_model(config, max_code=2, seed=1337).cuda()
    
    batch = 64
    block_size = 256
    inputs = torch.randint(0, config.vocab_size, (batch, block_size), device='cuda')
    targets = torch.randint(0, config.vocab_size, (batch, block_size), device='cuda')
    
    # Warmup
    for _ in range(3):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    print("Starting profiler...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            optimizer.zero_grad()
            _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    run_profiler()
