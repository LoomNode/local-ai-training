# seqmoe-lab

Sequential Mixture-of-Experts training under limited VRAM.

**Hypothesis:** two expert FFNs, each trained alone against the same frozen trunk, can be assembled afterward and routed — so total specialist capacity is a storage problem, while training memory tracks one expert.

This is a new lab. It copies small pieces from `scrt-lab` (GPT block, AdamW loop, JSON results, GPU 1), `envelope-lab` (fail-closed VRAM envelope), and Colibri (logical expert ids). It does **not** import those trees, and it does **not** implement NVMe paging.

- Spec: [`docs/superpowers/specs/2026-08-30-sequential-moe-poc-design.md`](docs/superpowers/specs/2026-08-30-sequential-moe-poc-design.md)
- Default device: `cuda:1`
- Swap is not capacity

Implementation has not started. Do not add paging, LoRA experts, or a HuggingFace trunk until the spec’s success criteria are measured.
