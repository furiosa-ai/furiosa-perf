# Benchmark Methodology

This document explains how `furiosa-perf` measures LLM serving performance, why each design choice was made, and how to interpret the results.

---

## Background

`furiosa-perf` uses the [vLLM benchmark tool](https://github.com/vllm-project/vllm/tree/main/benchmarks) (`vllm bench serve`) to track serving performance across a fixed matrix of scenarios defined by:

- **Input token length** (`input_tokens`) — the length of each prompt
- **Output token length** (`output_tokens`) — the number of tokens to generate
- **Max concurrency** (`max_concurrency`) — the number of requests processed simultaneously, treated as the effective batch size (i.e. concurrent users)

Benchmarking against this fixed scenario matrix allows performance to be tracked consistently across hardware, firmware, and software versions.

### Isolated venv for the benchmark client

The vLLM benchmark tool is installed into an isolated Python venv managed automatically by `furiosa-perf` (under `bench_space/.<name>/`, e.g. `bench_space/.vllm/`). This avoids dependency conflicts between the benchmark client and the `furiosa-llm` serving environment, so no manual setup is needed.

---

## Benchmark design

### Workload construction

`furiosa-perf` invokes the vLLM benchmark tool via a **two-part command**: a shared base command applied to every scenario, followed by task-specific flags that vary by `task` type.

**Base command** (all tasks):

```bash
python -m vllm.entrypoints.cli.main bench serve \
  --percentile-metrics=ttft,tpot,itl,e2el \
  --metric-percentiles=25,50,75,90,95,99 \
  --max-concurrency=<max_concurrency> \
  --num-prompts=<num_prompts> \
  --request-rate=<request_rate> \
  --model=<model> \
  --ignore-eos \
  --save-result \
  --result-dir=<result_dir> \
  --ready-check-timeout-sec=0 \
  --trust-remote-code \
  [--host=<host>] [--port=<port>]
```

**Task-specific flags** (appended to the base command):

| Task | Additional flags |
|---|---|
| `offline` | `--backend=vllm --dataset-name=random --random-input-len=<ISL> --random-output-len=<OSL> --random-range-ratio=<ratio>` |
| `vl-offline` | `--backend=openai-chat --dataset-name=random-mm --random-input-len=<ISL> --random-output-len=<OSL> --random-range-ratio=<ratio> --random-mm-base-items-per-request=<N> --random-mm-bucket-config=<cfg> --random-mm-limit-mm-per-prompt=<cfg> --endpoint=/v1/chat/completions` |
| `reranker` | `--backend=vllm-rerank --dataset-name=random-rerank --random-input-len=<ISL> --random-batch-size=<N> --endpoint=/v1/rerank` |
| `embeddings` | `--backend=openai-embeddings --dataset-name=random --random-input-len=<ISL> --endpoint=/v1/embeddings` |

For example, a full `offline` scenario run looks like:

```bash
python -m vllm.entrypoints.cli.main bench serve \
  --percentile-metrics=ttft,tpot,itl,e2el \
  --metric-percentiles=25,50,75,90,95,99 \
  --max-concurrency=<max_concurrency> \
  --num-prompts=<max_concurrency * 3> \
  --request-rate=inf \
  --model=<model> \
  --ignore-eos \
  --save-result \
  --result-dir=<result_dir> \
  --ready-check-timeout-sec=0 \
  --trust-remote-code \
  --backend=vllm \
  --dataset-name=random \
  --random-input-len=<input_tokens> \
  --random-output-len=<output_tokens> \
  --random-range-ratio=0.0 \
  --port=<port>
```

### Why `num_prompts = max_concurrency × 3`

Setting `--num-prompts` to three times `--max-concurrency` (with `--request-rate inf`) creates a two-phase load pattern:

1. **Warm-up phase** — the first `max_concurrency` requests are sent to the server all at once, filling the concurrency slot immediately.
2. **Steady-state phase** — as each request completes, the next one from the remaining `2 × max_concurrency` pool is sent immediately, maintaining constant pressure.

This tests **prefill and decode hybrid batching** under sustained load at the target concurrency level, which is representative of a real-world scenario where `max_concurrency` users are simultaneously active. Running only a single batch (`num_prompts = max_concurrency`) would measure cold-start behavior; the extra two batches ensure the server is benchmarked in its steady operating state.

### `max_concurrency` as the concurrency dimension

`max_concurrency` controls the maximum number of in-flight requests. `furiosa-perf` sweeps this value across a list (e.g. `[1, 4, 8, 16, 32, 64, 128, 256]`) to build a throughput-latency curve. The highest `max_concurrency` at which TTFT and TPOT remain within acceptable thresholds gives the **maximum serviceable concurrent user count** for that model and hardware configuration.

---

## Metrics

`furiosa-perf` collects the following metrics for each scenario:

| Metric | Unit | Description |
|---|---|---|
| **TTFT** | seconds | Time to First Token — latency from request submission to the first generated token. Reported at P25/P50/P75/P90/P95/P99. |
| **TPOT** | ms/token | Time Per Output Token — average inter-token generation latency per request after the first token. Reported at P25/P50/P75/P90/P95/P99. |
| **ITL** | ms/token | Inter-Token Latency — per-token latency metric reported by `vllm bench serve`. Reported at P25/P50/P75/P90/P95/P99. |
| **E2EL** | seconds | End-to-End Latency — total time from request submission to the last generated token. Reported at P25/P50/P75/P90/P95/P99. |
| **Output Throughput** | tok/s | Output tokens generated per second across all concurrent requests. |
| **Total Throughput** | tok/s | Combined input + output tokens processed per second. |

In addition, per-scenario hardware metrics (NPU/GPU power, temperature, utilization; host CPU and memory) are sampled throughout each run and written to a monitoring CSV alongside the benchmark results.

### Reading the results

From TTFT, TPOT, and Output Throughput you can derive the per-user experience at a given concurrency level:

- **TTFT** directly represents how long a user waits before seeing the first token of the response.
- **TPOT** determines the perceived generation speed — lower TPOT means smoother streaming.
- **Output Throughput / max_concurrency** gives the effective single-user generation rate (tokens per second per user) at full concurrency.

---

## Scenario examples

The `examples/bench/llm_scenario.yaml` covers a wide range of input lengths at varying concurrency to characterise both short-context and long-context performance:

```yaml
name: vllm
task: offline
scenarios:
  - input_tokens: 128
    output_tokens: 128
    max_concurrency: [1, 4, 8, 16, 32, 64, 128, 256, 512]
  - input_tokens: 1024
    output_tokens: 1024
    max_concurrency: [1, 4, 8, 16, 32, 64, 128, 256]
  - input_tokens: 7168
    output_tokens: 1024
    max_concurrency: [1, 4, 8, 16, 32, 64]
  - input_tokens: 130048
    output_tokens: 1024
    max_concurrency: [1, 4]
```

Short-context scenarios (128 tokens in/out) sweep a wider concurrency range to find the throughput ceiling. Long-context scenarios (e.g. 130 k input) cap concurrency lower because each request occupies significantly more KV-cache.

---

## Output structure

Results are written under `./bench_space/`:

```
bench_space/
└── <DEVICE>_<NUM>_<BACKEND>_<VERSION>/
    └── <name>/<task>/<model>/
        ├── summary_<ISL>_<OSL>.md        # Markdown table (one per ISL/OSL pair)
        ├── summary_<ISL>_<OSL>.csv       # CSV summary
        └── <ISL>.<OSL>.<CONC>/
            ├── <DEVICE>_<NUM>_monitoring_log.csv   # hardware metrics during this scenario
            └── vllm-*.json                         # raw vLLM benchmark output
```

Each `summary_<ISL>_<OSL>` file contains one row per concurrency level, making it straightforward to spot where latency degrades as load increases.
