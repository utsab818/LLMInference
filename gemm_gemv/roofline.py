from dataclasses import dataclass


@dataclass
class HardwareSpec:
    peak_tflops: float
    memory_bandwidth_gbps: float
    name: str


RTX_3090 = HardwareSpec(
    peak_tflops=35.6,
    memory_bandwidth_gbps=936.0,
    name="RTX 3090",
)

RTX_4090 = HardwareSpec(
    peak_tflops=82.6,
    memory_bandwidth_gbps=1008.0,
    name="RTX 4090",
)

A100_80GB = HardwareSpec(
    peak_tflops=312.0,
    memory_bandwidth_gbps=2039.0,
    name="A100 80GB",
)

H100_SXM = HardwareSpec(
    peak_tflops=989.0,
    memory_bandwidth_gbps=3350.0,
    name="H100 SXM",
)

M4_AIR_BASE = HardwareSpec(
    peak_tflops=3.5,
    memory_bandwidth_gbps=120.0,
    name="MacBook Air M4 Base (8-core GPU)",
)


def roofline_throughput(ai: float, hw: HardwareSpec) -> float:
    memory_bound_throughput = ai * hw.memory_bandwidth_gbps / 1000
    return min(memory_bound_throughput, hw.peak_tflops)


def ridge_point(hw: HardwareSpec) -> float:
    return hw.peak_tflops * 1000 / hw.memory_bandwidth_gbps


def is_compute_bound(ai: float, hw: HardwareSpec) -> bool:
    return ai >= ridge_point(hw)


def gemm_arithmetic_intensity(
    m: int,
    n: int,
    k: int,
    bytes_per_element: int = 2,
) -> float:
    flops = 2 * m * n * k
    bytes_moved = (m * k + k * n + m * n) * bytes_per_element
    return flops / bytes_moved


def gemv_arithmetic_intensity(
    m: int,
    k: int,
    bytes_per_element: int = 2,
) -> float:
    flops = 2 * m * k
    bytes_moved = (m * k + k + m) * bytes_per_element
    return flops / bytes_moved


def batched_gemv_arithmetic_intensity(
    batch: int,
    m: int,
    k: int,
    bytes_per_element: int = 2,
) -> float:
    flops = 2 * batch * m * k
    bytes_moved = (m * k + batch * k + batch * m) * bytes_per_element
    return flops / bytes_moved


def plot_roofline(
    hw: HardwareSpec,
    points: list[tuple[str, float, float]] | None = None,
    save_path: str | None = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    ai_range = np.logspace(-2, 4, 1000)
    throughput = np.minimum(
        ai_range * hw.memory_bandwidth_gbps / 1000,
        hw.peak_tflops,
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.loglog(ai_range, throughput, linewidth=3, label="Roofline")

    ridge = ridge_point(hw)
    ax.axvline(x=ridge, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)

    ax.annotate(
        f"Ridge Point\nAI = {ridge:.1f}",
        xy=(ridge, hw.peak_tflops),
        xytext=(ridge * 1.08, hw.peak_tflops * 0.78),
        textcoords="data",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="-", color="gray", lw=1),
    )

    ax.fill_between(
        ai_range[ai_range < ridge],
        throughput[ai_range < ridge],
        0.01,
        alpha=0.18,
        color="red",
        label="Memory Bound",
    )
    ax.fill_between(
        ai_range[ai_range >= ridge],
        throughput[ai_range >= ridge],
        0.01,
        alpha=0.18,
        color="green",
        label="Compute Bound",
    )

    if points:
        # Hand-tuned offsets for clarity
        label_offsets = {
            "GEMV 4096": (10, 8),
            "Batch 1": (10, -2),
            "Batch 4": (10, 10),
            "Batch 16": (10, 10),
            "Batch 64": (12, 8),
            "Batch 256": (12, 8),
            "GEMM 4096": (10, 8),
        }

        for name, ai, measured_tflops in points:
            color = "green" if is_compute_bound(ai, hw) else "red"
            ax.scatter(
                [ai],
                [measured_tflops],
                s=280,
                c=color,
                edgecolors="white",
                linewidths=1.2,
                zorder=5,
            )

            dx, dy = label_offsets.get(name, (8, 8))
            ax.annotate(
                name,
                xy=(ai, measured_tflops),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8),
            )

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=13)
    ax.set_ylabel("Throughput (TFLOPS)", fontsize=13)
    ax.set_title(f"Roofline Model - {hw.name}", fontsize=16, pad=12)

    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, which="both", alpha=0.25)

    ax.set_xlim([0.01, 10000])
    ax.set_ylim([0.01, hw.peak_tflops * 1.6])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hw = M4_AIR_BASE

    print(f"Hardware: {hw.name}")
    print(f"Peak TFLOPS: {hw.peak_tflops}")
    print(f"Memory Bandwidth: {hw.memory_bandwidth_gbps} GB/s")
    print(f"Ridge Point: {ridge_point(hw):.1f} FLOP/Byte")
    print()

    gemm_ai = gemm_arithmetic_intensity(4096, 4096, 4096)
    gemv_ai = gemv_arithmetic_intensity(4096, 4096)

    print(f"GEMM (4096x4096x4096) AI: {gemm_ai:.1f} FLOP/Byte")
    print(f"  Compute bound: {is_compute_bound(gemm_ai, hw)}")
    print()

    print(f"GEMV (4096x4096) AI: {gemv_ai:.2f} FLOP/Byte")
    print(f"  Compute bound: {is_compute_bound(gemv_ai, hw)}")
    print()

    points = [
        ("GEMV 4096", gemv_ai, roofline_throughput(gemv_ai, hw) * 0.78),
        ("Batch 1", batched_gemv_arithmetic_intensity(1, 4096, 4096), roofline_throughput(batched_gemv_arithmetic_intensity(1, 4096, 4096), hw) * 0.95),
        ("Batch 4", batched_gemv_arithmetic_intensity(4, 4096, 4096), roofline_throughput(batched_gemv_arithmetic_intensity(4, 4096, 4096), hw) * 0.82),
        ("Batch 16", batched_gemv_arithmetic_intensity(16, 4096, 4096), roofline_throughput(batched_gemv_arithmetic_intensity(16, 4096, 4096), hw) * 0.88),
        ("Batch 64", batched_gemv_arithmetic_intensity(64, 4096, 4096), hw.peak_tflops * 0.79),
        ("Batch 256", batched_gemv_arithmetic_intensity(256, 4096, 4096), hw.peak_tflops * 0.78),
        ("GEMM 4096", gemm_ai, hw.peak_tflops * 0.78),
    ]

    for batch in [1, 4, 16, 64, 256]:
        ai = batched_gemv_arithmetic_intensity(batch, 4096, 4096)
        bound = "compute" if is_compute_bound(ai, hw) else "memory"
        print(f"Batched GEMV (batch={batch:3d}) AI: {ai:.2f} FLOP/Byte ({bound})")

    plot_roofline(hw, points=points, save_path="roofline.png")
    plt.show()