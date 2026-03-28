import time
from dataclasses import dataclass

import torch


@dataclass
class BenchmarkResult:
    device: str
    dtype: str
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    tflops: float
    memory_gbps: float


def sync_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()
    # CPU is synchronous, so no sync needed


def gemm_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def gemm_bytes(m: int, n: int, k: int, dtype: torch.dtype) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return (m * k + k * n + m * n) * element_size


def benchmark_gemm(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: str,
    warmup: int = 10,
    iterations: int = 50,
) -> BenchmarkResult:
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)

    for _ in range(warmup):
        torch.mm(a, b)
    sync_device(device)

    times = []
    for _ in range(iterations):
        sync_device(device)
        start = time.perf_counter()
        torch.mm(a, b)
        sync_device(device)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # microseconds

    times_tensor = torch.tensor(times)
    mean_us = times_tensor.mean().item()
    std_us = times_tensor.std(unbiased=False).item()
    min_us = times_tensor.min().item()
    max_us = times_tensor.max().item()

    flops = gemm_flops(m, n, k)
    tflops = flops / (mean_us * 1e-6) / 1e12

    bytes_moved = gemm_bytes(m, n, k, dtype)
    memory_gbps = bytes_moved / (mean_us * 1e-6) / 1e9

    return BenchmarkResult(
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        mean_us=mean_us,
        std_us=std_us,
        min_us=min_us,
        max_us=max_us,
        tflops=tflops,
        memory_gbps=memory_gbps,
    )


def try_benchmark(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: str,
    warmup: int = 10,
    iterations: int = 50,
) -> BenchmarkResult | None:
    try:
        return benchmark_gemm(
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
    except Exception as e:
        print(f"[SKIP] device={device}, dtype={dtype}: {e}")
        return None


def print_result(result: BenchmarkResult, m: int, n: int, k: int) -> None:
    print(
        f"M={m:6d}, N={n:6d}, K={k:6d} | "
        f"{result.device:>3s} | {result.dtype:>7s} | "
        f"{result.mean_us:10.1f} us | "
        f"{result.tflops:8.3f} TFLOPS | "
        f"{result.memory_gbps:8.3f} GB/s"
    )


def print_comparisons(results: list[BenchmarkResult]) -> None:
    by_key = {(r.device, r.dtype): r for r in results}

    def speedup(slower: BenchmarkResult, faster: BenchmarkResult) -> float:
        return slower.mean_us / faster.mean_us

    cpu_fp32 = by_key.get(("cpu", "float32"))
    cpu_fp16 = by_key.get(("cpu", "float16"))
    mps_fp32 = by_key.get(("mps", "float32"))
    mps_fp16 = by_key.get(("mps", "float16"))

    print("\nComparisons:")

    if cpu_fp32 and cpu_fp16:
        print(
            f"CPU float32 / float16 speedup = "
            f"{speedup(cpu_fp32, cpu_fp16):.2f}x "
            f"(>1 means CPU float16 is faster)"
        )

    if mps_fp32 and mps_fp16:
        print(
            f"MPS float32 / float16 speedup = "
            f"{speedup(mps_fp32, mps_fp16):.2f}x "
            f"(>1 means MPS float16 is faster)"
        )

    if cpu_fp32 and mps_fp32:
        print(
            f"float32 CPU / MPS speedup = "
            f"{speedup(cpu_fp32, mps_fp32):.2f}x "
            f"(>1 means MPS float32 is faster)"
        )

    if cpu_fp16 and mps_fp16:
        print(
            f"float16 CPU / MPS speedup = "
            f"{speedup(cpu_fp16, mps_fp16):.2f}x "
            f"(>1 means MPS float16 is faster)"
        )


if __name__ == "__main__":
    configs = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    modes = [
        ("cpu", torch.float32),
        ("cpu", torch.float16),
    ]

    if torch.mps.is_available():
        modes.extend(
            [
                ("mps", torch.float32),
                ("mps", torch.float16),
            ]
        )
    else:
        print("MPS not available, running CPU-only benchmarks.\n")

    for m, n, k in configs:
        print("=" * 100)
        print(f"GEMM Benchmark for shape: ({m}, {k}) x ({k}, {n})")
        print("=" * 100)

        results = []

        for device, dtype in modes:
            result = try_benchmark(
                m=m,
                n=n,
                k=k,
                dtype=dtype,
                device=device,
                warmup=10,
                iterations=50,
            )
            if result is not None:
                results.append(result)
                print_result(result, m, n, k)

        print_comparisons(results)
        print()