# Outputs

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

<h3>Roofline</h3>
![alt text](outputs/roofline.png)


<h3>Batching for Float32 and Float16 respectively for MPS</h3>

Float32
![alt text](../outputs/batching_benchmark_32.png)

Float16
![alt text](../outputs/batching_benchmark_16.png)