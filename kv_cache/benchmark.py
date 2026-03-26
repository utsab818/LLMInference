import json
import os
import time
import random
import torch

from .cached_generation import CachedTransformerModel, cached_generate


def synchronize(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()


def get_dtype_for_device(device: torch.device) -> torch.dtype:
    # Safer default on Apple Silicon unless you benchmark float16 separately.
    if device.type == 'cuda':
        return torch.float16
    if device.type == 'mps':
        return torch.float32
    return torch.float32


def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def naive_generate_same_model(
    model: CachedTransformerModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
):
    model.eval()
    generated = input_ids.clone()
    timings = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            synchronize(generated.device)
            start = time.perf_counter()

            logits = model(generated, caches=None, start_pos=0)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            synchronize(generated.device)
            timings.append((time.perf_counter() - start) * 1000)

    return generated, {
        'times_ms': timings,
        'total_ms': sum(timings),
        'mean_ms': sum(timings) / len(timings),
        'first_token_ms': timings[0],
        'last_token_ms': timings[-1],
    }


def benchmark_device(device_name: str, config: dict, prompt_lengths=(10, 50, 100), max_new_tokens: int = 50):
    device = torch.device(device_name)
    dtype = get_dtype_for_device(device)
    device_result = {
        'device': device_name,
        'dtype': str(dtype),
        'benchmarks': [],
    }

    print(f"\n=== Device: {device_name} | dtype: {dtype} ===")

    for prompt_len in prompt_lengths:
        print(f"\nPrompt length={prompt_len}, max_new_tokens={max_new_tokens}")
        set_seed(1234 + prompt_len)
        input_ids = torch.randint(0, config['vocab_size'], (1, prompt_len), device=device)

        # same weights for both methods on the same device
        set_seed(999)
        model = CachedTransformerModel(**config).to(device=device, dtype=dtype)
        model.eval()

        # warmup naive
        with torch.no_grad():
            _ = model(input_ids, caches=None, start_pos=0)
        synchronize(device)

        # naive
        _, naive_stats = naive_generate_same_model(model, input_ids, max_new_tokens)
        print(
            f"Naive: first={naive_stats['first_token_ms']:.2f} ms | "
            f"last={naive_stats['last_token_ms']:.2f} ms | "
            f"total={naive_stats['total_ms']:.2f} ms"
        )

        # fresh prompt for cached benchmark
        set_seed(1234 + prompt_len)
        input_ids_fresh = torch.randint(0, config['vocab_size'], (1, prompt_len), device=device)

        # warmup cached
        with torch.no_grad():
            warmup_caches = model.create_caches(1, prompt_len + 10, device, dtype)
            _ = model(input_ids_fresh, warmup_caches, start_pos=0)
        synchronize(device)

        _, cached_stats = cached_generate(model, input_ids_fresh, max_new_tokens)
        decode_avg = sum(cached_stats['decode_ms']) / len(cached_stats['decode_ms'])
        print(
            f"Cached: prefill={cached_stats['prefill_ms']:.2f} ms | "
            f"decode_avg={decode_avg:.2f} ms | total={cached_stats['total_ms']:.2f} ms"
        )

        speedup = naive_stats['total_ms'] / cached_stats['total_ms']
        print(f"Speedup: {speedup:.2f}x")

        device_result['benchmarks'].append({
            'prompt_len': prompt_len,
            'max_new_tokens': max_new_tokens,
            'naive': naive_stats,
            'cached': cached_stats,
            'speedup_cached_vs_naive': speedup,
        })

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return device_result


def compare_devices():
    config = {
        'vocab_size': 1000,
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'num_kv_heads': 2,
        'intermediate_dim': 1024,
    }

    available = ['cpu']
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available.insert(0, 'mps')
    if torch.cuda.is_available():
        available.insert(0, 'cuda')

    print('Available devices:', available)
    print('PyTorch version:', torch.__version__)

    results = {
        'torch_version': torch.__version__,
        'devices': [],
        'cross_device_summary': [],
    }

    for device_name in available:
        results['devices'].append(benchmark_device(device_name, config))

    by_name = {d['device']: d for d in results['devices']}
    if 'mps' in by_name and 'cpu' in by_name:
        mps_b = by_name['mps']['benchmarks']
        cpu_b = by_name['cpu']['benchmarks']
        for mps_entry, cpu_entry in zip(mps_b, cpu_b):
            results['cross_device_summary'].append({
                'prompt_len': mps_entry['prompt_len'],
                'max_new_tokens': mps_entry['max_new_tokens'],
                'naive_cpu_over_mps': cpu_entry['naive']['total_ms'] / mps_entry['naive']['total_ms'],
                'cached_cpu_over_mps': cpu_entry['cached']['total_ms'] / mps_entry['cached']['total_ms'],
                'cached_speedup_mps': mps_entry['speedup_cached_vs_naive'],
                'cached_speedup_cpu': cpu_entry['speedup_cached_vs_naive'],
            })

    print('\n=== JSON RESULTS ===')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    compare_devices()
