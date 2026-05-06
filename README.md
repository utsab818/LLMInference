# LLM Inference Engine
A minimal, systems-focused implementation of LLM inference, built to understand how modern transformer-based models behave during generation.

This project is not about training large models. It focuses on how inference actually works under the hood — from attention mechanics to hardware-level performance behavior.

### Motivation
I am interested in the intersection of HPC, parallel computing, and AI inference. While learning LLMs, I realized that efficient inference is not just about model design, but about how computation is parallelized and mapped to hardware. Concepts like batching, KV cache, and attention directly relate to classical HPC ideas such as memory access patterns, compute vs memory bounds, and workload scheduling. This project is my way of understanding how modern AI systems behave from a systems and performance perspective, not just at the model level.

### Project Structure
```
attention/               -->    Transformer Mechanics (attention, GQA, FFN)
kv_cache/                -->    The Generation Loop (KV cache, prefill/decode)
gemm_gemv/               -->    Compute Characteristics (GEMM vs GEMV, roofline)
kernel_fundamentals/     -->    Kernel Fundamentals (GPU architecture, CUDA basics)
kernel_optimization/     -->    Kernel Optimization (coalescing, shared memory, Triton)
attention_optimization/  -->    FlashAttention (online softmax, tiled attention)
batching/                -->    Continuous Batching (Orca, radix cache, paged memory)
```
### Setup
This project uses uv for fast dependency management.

```
uv sync --extra full
```

### Run the test
```
# Run all tests
uv run pytest 

# Run specific modules
uv run pytest attention/test.py -v
uv run pytest kv_cache/test.py -v 
```

### Running Benchmarks
```
uv run python -m attention.benchmark
uv run python -m kv_cache.benchmark
```

### Outputs

```
Each module contains its own README with benchmark plots, observations, and hardware-specific insights.

All results are generated on:

Apple M4 Air (base variant, MPS backend)

and some required ones are generated in Tesla T4 GPU (Google Collab).
```

### Systems & Performance Focus
This project is intentionally systems-oriented. It focuses on how batching improves utilization, how KV cache changes memory access patterns, why inference workloads become memory-bound, and where optimizations like FlashAttention originate.

### Exploration topics

```
Transformer Mechanics (attention, GQA, FFN)
The Generation Loop (KV cache, prefill/decode)
Compute Characteristics (GEMM vs GEMV, roofline)
Kernel Fundamentals (GPU architecture, CUDA basics)
FlashAttention (online softmax, tiled attention)
Kernel Optimization (coalescing, shared memory, Triton)
Continuous Batching (Orca, radix cache, paged memory)
Advanced Scheduling (chunked prefill, CUDA graphs)
Mixture of Experts (MoE routing, expert caching)
Production Server (FastAPI, tokenizer pool, benchmarks)
```

### References
The Physics of LLM Inference by Elliot Arledge

### Note
This project is intended for learning and deep understanding of LLM inference, with a strong emphasis on systems behavior rather than high-level abstractions.