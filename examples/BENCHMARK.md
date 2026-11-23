# Benchmark Methodology

This document explains how `furiosa-perf` measures LLM serving performance, why each design choice was made, and how to interpret the results.

---

## Background

`furiosa-perf` uses the [vLLM benchmark tool](https://github.com/vllm-project/vllm/tree/main/benchmarks) (`vllm bench serve`) to track serving performance across a fixed matrix of shapes and concurrencies defined by:

- **Input token length** (`random-input-len`) — the length of each prompt
- **Output token length** (`random-output-len`) — the number of tokens to generate
- **Max concurrency** (`max-concurrency`) — the number of requests processed simultaneously, treated as the effective batch size (i.e. concurrent users)

Benchmarking against this fixed matrix allows performance to be tracked consistently across hardware, firmware, and software versions.

### Isolated venv for the benchmark client

The vLLM benchmark tool is installed on first use into its own Python venv, pinned to one vLLM version (`~/.cache/furiosa-perf/clients/vllm-<version>/`, shared by all runs on the host). This avoids dependency conflicts with the serving environment and keeps the client identical across hosts and dates, so results stay comparable. The version is printed in the summary header.

---

## Benchmark design

### Workload construction

Each point of a preset becomes one `vllm bench serve` call: the preset's shared `args`, overlaid with the point's matrix values, plus the target and result file that `furiosa-perf` sets itself. For a `baseline/text` point:

```bash
python -m vllm.entrypoints.cli.main bench serve \
  --model <served model> --host <host> --port <port> \
  --save-result --result-dir <point dir> --result-filename result.json \
  --ready-check-timeout-sec 0 \
  --backend vllm --dataset-name random --random-range-ratio 0 \
  --ignore-eos --temperature 0 --trust-remote-code \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 25,50,75,90,95,99 \
  --random-input-len <ISL> --random-output-len <OSL> \
  --max-concurrency <C> --num-prompts <3 x C>
```

The other bundled presets differ only in their `args` (see `furiosa_perf/presets/baseline/`): `image-text` sends one random 256x256 image per request through `/v1/chat/completions`, `embedding` and `rerank` use the pooling endpoints and have no output length.

### Why `num-prompts = max-concurrency × 3`

Setting `--num-prompts` to three times `--max-concurrency` (with `--request-rate inf`) creates a two-phase load pattern:

1. **Warm-up phase** — the first `max_concurrency` requests are sent to the server all at once, filling the concurrency slot immediately.
2. **Steady-state phase** — as each request completes, the next one from the remaining `2 × max_concurrency` pool is sent immediately, maintaining constant pressure.

This tests **prefill and decode hybrid batching** under sustained load at the target concurrency level, which is representative of a real-world case where `max_concurrency` users are simultaneously active. Running only a single batch (`num_prompts = max_concurrency`) would measure cold-start behavior; the extra two batches ensure the server is benchmarked in its steady operating state.

### `max_concurrency` as the concurrency dimension

`max_concurrency` controls the maximum number of in-flight requests. `furiosa-perf` sweeps this value across a list (e.g. `[1, 4, 8, 16, 32, 64, 128, 256]`) to build a throughput-latency curve. The highest `max_concurrency` at which TTFT and TPOT remain within acceptable thresholds gives the **maximum serviceable concurrent user count** for that model and hardware configuration.

---

## Metrics

`furiosa-perf` collects the following metrics for each point:

| Metric | Unit | Description |
|---|---|---|
| **TTFT** | seconds | Time to First Token — latency from request submission to the first generated token. Reported at P25/P50/P75/P90/P95/P99. |
| **TPOT** | ms/token | Time Per Output Token — average inter-token generation latency per request after the first token. Reported at P25/P50/P75/P90/P95/P99. |
| **ITL** | ms/token | Inter-Token Latency — per-token latency metric reported by `vllm bench serve`. Reported at P25/P50/P75/P90/P95/P99. |
| **E2EL** | seconds | End-to-End Latency — total time from request submission to the last generated token. Reported at P25/P50/P75/P90/P95/P99. |
| **Output Throughput** | tok/s | Output tokens generated per second across all concurrent requests. |
| **Total Throughput** | tok/s | Combined input + output tokens processed per second. |

In addition, the configured devices (power, temperatures, utilization, clocks, throttling), the server's `/metrics` (queue, KV-cache usage, token and prefix-cache counters) and server/client/host CPU and memory are sampled once per second for the whole run; each point's window is cut out of that log, and the full `/metrics` text is captured at every point start and end. Power and energy are taken over the window in which requests were running, so tokens per joule and tokens per second share one time base. See the README for the summary columns.

### Reading the results

From TTFT, TPOT, and Output Throughput you can derive the per-user experience at a given concurrency level:

- **TTFT** directly represents how long a user waits before seeing the first token of the response.
- **TPOT** determines the perceived generation speed — lower TPOT means smoother streaming.
- **Output Throughput / max_concurrency** gives the effective single-user generation rate (tokens per second per user) at full concurrency.

---

## Shape examples

The `baseline/text` preset covers a wide range of input lengths at varying concurrency to characterise both short-context and long-context performance (excerpt):

```yaml
matrix:
  - {random-input-len: 128, random-output-len: 128, max-concurrency: [1, 4, 8, 16, 32, 64, 128, 256, 512]}
  - {random-input-len: 1024, random-output-len: 1024, max-concurrency: [1, 4, 8, 16, 32, 64, 128, 256]}
  - {random-input-len: 7168, random-output-len: 1024, max-concurrency: [1, 4, 8, 16, 32, 64]}
  - {random-input-len: 130048, random-output-len: 1024, max-concurrency: [1, 4]}
```

Short-context shapes (128 tokens in/out) sweep a wider concurrency range to find the throughput ceiling. Long-context shapes (e.g. 130 k input) cap concurrency lower because each request occupies significantly more KV-cache.

---

## Output structure

Results are written under `./results/` (see the README for every file):

```
results/baseline/<DEVICE>_<NUM>[-<LAYOUT>]_<BACKEND>_<VERSION>/<task>/<org>--<model>/
├── run.json                      # the run: hardware, versions, server command, preset
├── summary_<ISL>_<OSL>.md|.csv   # one row per concurrency level
└── i<ISL>-o<OSL>-c<CONC>/
    ├── record.json               # the point, self-describing
    └── result.json               # raw vllm bench result
```

Each `summary_<ISL>_<OSL>` file contains one row per concurrency level, making it straightforward to spot where latency degrades as load increases.
