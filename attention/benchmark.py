"""Benchmarks for Chapter 01: Transformer Mechanics"""

import json
import time
from collections.abc import Callable

import torch

from .attention import MultiHeadAttention
from .ffn import FusedSwiGLUFFN, SwiGLUFFN
from .gqa import GroupedQueryAttention
from .transformer import TransformerBlock


def benchmark_fn(fn: Callable, device: str, warmup: int = 10, iterations: int = 100) -> dict:
    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    # Warmup
    for _ in range(warmup):
        fn()
    synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        synchronize()
        times.append((time.perf_counter() - start) * 1e6)

    mean = sum(times) / len(times)
    return {
        "mean_us": mean,
        "min_us": min(times),
        "max_us": max(times),
        "std_us": (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5,
    }

def run_benchmarks(device: str) -> list:
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    batch_size = 8
    seq_len = 512
    hidden_dim = 1024
    num_heads = 16
    num_kv_heads = 4
    intermediate_dim = int(8 / 3 * hidden_dim)

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    benchmarks = []

    # MHA
    mha = MultiHeadAttention(hidden_dim, num_heads).to(device, dtype)
    benchmarks.append({
        "name": "attention_mha",
        **benchmark_fn(lambda: mha(x), device),
    })

    # GQA
    gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads).to(device, dtype)
    benchmarks.append({
        "name": "attention_gqa",
        **benchmark_fn(lambda: gqa(x), device),
    })

    # SwiGLU unfused
    swiglu = SwiGLUFFN(hidden_dim, intermediate_dim).to(device, dtype)
    benchmarks.append({
        "name": "ffn_swiglu_unfused",
        **benchmark_fn(lambda: swiglu(x), device),
    })

    # SwiGLU fused
    fused_swiglu = FusedSwiGLUFFN(hidden_dim, intermediate_dim).to(device, dtype)
    benchmarks.append({
        "name": "ffn_swiglu_fused",
        **benchmark_fn(lambda: fused_swiglu(x), device),
    })

    # Full transformer block
    block = TransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_dim=intermediate_dim,
    ).to(device, dtype)
    benchmarks.append({
        "name": "transformer_block",
        **benchmark_fn(lambda: block(x), device),
    })

    return benchmarks


def main():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")

    all_results = {}
    for device in devices:
        print(f"Running on {device.upper()}...")
        all_results[device] = run_benchmarks(device)
        print(f"  Done.")

    # Build comparison table
    # Collect all benchmark names in order
    names = [b["name"] for b in all_results[devices[0]]]

    # Header
    col_w = 20
    num_w = 12
    header = f"{'Benchmark':<{col_w}}" + "".join(f"{d.upper():>{num_w}}" for d in devices)
    if len(devices) == 2 and "cpu" in devices and "mps" in devices:
        header += f"{'Speedup':>{num_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for name in names:
        # Look up mean_us for each device
        row_data = {}
        for device in devices:
            match = next(b for b in all_results[device] if b["name"] == name)
            row_data[device] = match["mean_us"]

        row = f"{name:<{col_w}}"
        for device in devices:
            row += f"{row_data[device]:>{num_w}.1f} us"

        # Speedup: CPU / MPS (higher = MPS is faster)
        if "cpu" in row_data and "mps" in row_data:
            speedup = row_data["cpu"] / row_data["mps"]
            row += f"{speedup:>{num_w}.2f}x"

        print(row)

    print("=" * len(header))
    print(f"{'(lower is faster)':>{col_w + num_w * len(devices)}}")

    # Full JSON output
    print("\n" + json.dumps(all_results, indent=2))
    return all_results


if __name__ == "__main__":
    main()