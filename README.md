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

![alt text](outputs/attention_benchmark.png)

<h3>KV Cache</h3>

<table align="center">
  <tr>
    <td align="center">
      <b>With MPS Float32</b><br/>
      <img src="outputs/kv_cache_benchmark.png" alt="KV Cache Benchmark - MPS Float32" width="400"/>
    </td>
    <td align="center">
      <b>With MPS Float16</b><br/>
      <img src="outputs/mps_float16.png" alt="KV Cache Benchmark - MPS Float16" width="400"/>
    </td>
  </tr>
</table>

<h3>GEMM_GEMV</h3>

<table align="center">
  <tr>
    <td align="center">
      <b>GEMM with MPS Fp16</b><br/>
      <img src="outputs/gemm_benchmark.png" alt="GEMM Benchmark - MPS Fp16" width="400"/>
    </td>
    <td align="center">
      <b>CPU and MPS (Fp16, fp32)</b><br/>
      <img src="outputs/gemm_comparision.png" alt="GEMM Comparision Table" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>GEMV with MPS FP16</b><br/>
      <img src="outputs/gemv_benchmark.png" alt="GEMV Benchmark - MPS Fp16" width="400"/>
    </td>
    <td align="center">
      <b>CPU and MPS (Fp16, fp32)</b><br/>
      <img src="outputs/gemv_comparision.png" alt="GEMV Comparision Table" width="400"/>
    </td>
  </tr>
</table>

