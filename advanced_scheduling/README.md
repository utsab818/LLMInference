# Outputs

### Prefilled Chunk
Instead of waiting for single large prefill to process, we divide into chunks so that other get chances too and one with less prefill tokens gets chance to finish.

![alt text](../outputs/prefilled_chunk.png)

### Mixed Batch
Not only prefill, introducing decode process of finished chunk in single go makes it better.

![alt text](../outputs/mixed_batch.png)

### Cuda Graph
Reduce waiting game by capturing the graph for suitable batches.

![alt text](../outputs/cuda_graph.png)

### Overlap scheduling
Remove the waiting game between processes by parallel processing batching, scheduling and (prefill, decode).

![alt text](../outputs/overlap_scheduling.png)