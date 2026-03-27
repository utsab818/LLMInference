# LLM Inference

### Run the test
```
uv run pytest 

uv run pytest attention/test.py -v
uv run pytest kv_cache/test.py -v 
```

### Run the benchmark
```
uv run python -m attention.benchmark
uv run python -m kv_cache.benchmark
```

### MPS vs CPU comparision in Apple M4 air

<h3>Attention</h3>

![alt text](attention/attention_benchmark.png)

<h3>KV Cache</h3>

<table align="center">
  <tr>
    <td align="center">
      <b>With MPS Float32</b><br/>
      <img src="kv_cache/kv_cache_benchmark.png" alt="KV Cache Benchmark - MPS Float32" width="400"/>
    </td>
    <td align="center">
      <b>With MPS Float16</b><br/>
      <img src="kv_cache/mps_float16.png" alt="KV Cache Benchmark - MPS Float16" width="400"/>
    </td>
  </tr>
</table>

<h3>GEMM_GEMV</h3>

GEMM in MPS (fp16) 
![alt text](gemm_gemv/gemm_bechmark_fp16.png)