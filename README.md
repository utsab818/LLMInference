# LLM Inference

### Run the test
```
uv run pytest 

uv run pytest attention/test.py -v
```

### Run the benchmark
```
uv run python -m attention.benchmark
```

### MPS vs CPU comparision in Apple M4 air

![alt text](attention/attention_benchmark.png)