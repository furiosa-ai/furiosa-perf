# furiosa-perf

LLM serving performance benchmark CLI for Furiosa NPU and NVIDIA GPU. Launches an API server, drives [vLLM's benchmark tool](https://github.com/vllm-project/vllm) as the workload generator, collects per-scenario metrics, and writes Markdown/CSV summary reports. Optionally records per-second hardware telemetry (power, temperature, utilisation, KV-cache occupancy) alongside each run.

**Sections:**

- Setup: [Quick start](#quick-start), [Prerequisites](#prerequisites), [Installation](#installation)
- Run: [Running](#running), [Configuration](#configuration)
- Interpret: [Validated models](#validated-models), [Outputs](#outputs)
- Debug: [Troubleshooting](#troubleshooting), [Development](#development)

## Quick start

```bash
furiosa-perf run \
  --backend furiosa-llm \
  --hardware-type npu \
  --server-config examples/server/llama_3_1_8b_1.yaml \
  --benchmark-config examples/bench/benchmark_32k.yaml \
  --model-id "furiosa-ai/Llama-3.1-8B-Instruct"
```

Results are written to `./bench_space/`. Open `summary.md` or `summary.csv` for the per-scenario metrics table.

## Prerequisites
### System

- **OS**: Linux (Ubuntu 22.04 / 24.04 recommended)
- **Python**: 3.10+

### Backend

#### Furiosa NPU + `furiosa-llm`

```bash
pip install furiosa-llm
```

#### NVIDIA GPU + vLLM

```bash
pip install vllm

# Optional — GPU monitoring requires nvidia-smi on PATH
```

## Installation

```bash
git clone https://github.com/furiosa-ai/furiosa-perf.git
cd furiosa-perf
pip install .
```

Backend extras:

```bash
pip install ".[furiosa-llm]"   # Furiosa NPU backend
pip install ".[vllm]"          # NVIDIA GPU / vLLM backend
```

## Running

```bash
furiosa-perf run [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--model-id` | `LGAI-EXAONE/EXAONE-4.0-32B-FP8` | HuggingFace model ID or local path |
| `--hardware-type` | `npu` | Hardware type: `npu` or `gpu` |
| `--backend` | `furiosa-llm` | Serving backend: `furiosa-llm` or `vllm` |
| `--server-config` | *(required)* | Path to API server YAML config |
| `--benchmark-config` | *(required)* | Path to benchmark YAML config |

## Configuration

### API server config (`--server-config`)

Controls how the LLM API server is launched.

```yaml
# examples/server/llama_3_1_8b_1.yaml
devices: "0"
port: 8000
```

| Field | Default | Description |
|---|---|---|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8000` | Bind port |
| `devices` | — | Comma-separated device indices (e.g. `0,1,2,3`) |
| `tensor_parallel_size` | inferred from `devices` | TP degree |
| `served_model_name` | — | Model alias exposed on `/v1/models` |
| `no_enable_prefix_caching` | `true` | Disable prefix caching |

**NPU device indexing**: use the numbers from the `Device` column of `furiosa-smi info`:

```
$ furiosa-smi info
+------+--------+------------------------+---------+---------+--------------+
| Arch | Device | Firmware               | Temp.   | Power   | PCI-BDF      |
+------+--------+------------------------+---------+---------+--------------+
| rngd | npu0   | 2026.1.0(rc0), 5b172ac | ...     | ...     | 0000:03:00.0 |
+------+--------+------------------------+---------+---------+--------------+
| rngd | npu1   | 2026.1.0(rc0), 5b172ac | ...     | ...     | 0000:04:00.0 |
+------+--------+------------------------+---------+---------+--------------+
```

```yaml
devices: "0,1"
```

Backend-specific behaviour:
- **vLLM**: sets `CUDA_VISIBLE_DEVICES` from `devices`.
- **furiosa-llm**: converts `devices: 0,1` → `--devices npu:0,npu:1`, and scales `tensor_parallel_size` by 8 (PE count per NPU).

### Benchmark config (`--benchmark-config`)

Controls which scenarios to run.

```yaml
# examples/bench/benchmark_32k.yaml
name: vllm
task: offline
scenarios:
  - input_tokens: 1024
    output_tokens: 1024
    max_concurrency: [1, 4, 8, 16]
```

| Field | Description |
|---|---|
| `name` | Benchmark suite name (used in output directory path) |
| `task` | `offline`, `vl-offline`, `reranker`, or `embeddings` |
| `scenarios` | List of scenario objects; `max_concurrency` may be a list and is expanded automatically |

Scenario fields (for `offline` / `vl-offline`):

| Field | Default | Description |
|---|---|---|
| `input_tokens` | `1024` | Prompt length in tokens |
| `output_tokens` | `1024` | Generation length in tokens |
| `max_concurrency` | `1` | In-flight request cap (list expands to multiple runs) |
| `num_prompts` | `max_concurrency × 3` | Total requests to send |
| `request_rate` | `inf` | Target request rate (`inf` = saturate server) |

## Validated models

Models validated on **Furiosa RNGD**:

| HuggingFace model ID | TP (NPU count) | Base model |
|---|---:|---|
| `furiosa-ai/EXAONE-4.0-32B-FP8` | 4 | `LGAI-EXAONE/EXAONE-4.0-32B` |
| `furiosa-ai/Qwen3-32B-FP8` | 4 | `Qwen/Qwen2.5-32B-Instruct` |
| `furiosa-ai/Llama-3.3-70B-Instruct` | 4 | `meta-llama/Llama-3.3-70B-Instruct` |
| `furiosa-ai/Llama-3.1-8B-Instruct` | 1 | `meta-llama/Llama-3.1-8B-Instruct` |

## Outputs

Results are written under `./bench_space/` relative to the working directory.

```
bench_space/
└── <DEVICE>_<NUM>_<BACKEND>/
    └── <name>/<task>/<model>/
        ├── summary.md                             # aggregated table (all ISL/OSL groups)
        ├── summary.csv
        ├── summary_<ISL>_<OSL>.md                # per input/output-length table
        ├── summary_<ISL>_<OSL>.csv
        └── <ISL>.<OSL>.<CONCURRENCY>/
            ├── <DEVICE>_<NUM>_monitoring_log.csv  # per-second hardware telemetry
            └── vllm-*.json                        # raw vLLM bench output
```

Server logs are written to `./serve_logs/serve_<model>_<timestamp>.log`.

Monitoring CSV columns: `timestamp`, `power_consumption`, `peak_temperature`, `avg_utilization`, `kv_cache_usage_percentage`, `num_requests_running`, `num_requests_waiting`, `host_cpu_utils`, `host_memory_usage_gib`.

## Troubleshooting

**Monitoring CSV is empty or missing**
- NPU: `furiosa_smi_py` must be importable — install it and verify `furiosa-smi info` works.
- GPU: `nvidia-smi` must be on `PATH`.

**Server does not start within the timeout (30 min)**
Run the backend serve command directly and inspect the log at `./serve_logs/`. The startup timeout can be adjusted in `APIServerManager._wait_for_startup`.

**vLLM bench tool not found after `setup()`**
The benchmark runner installs the benchmark tool into an isolated venv under `./bench_space/`. Ensure `uv` is installed (`pip install uv`).

**`HF_TOKEN` is required**
```bash
export HF_TOKEN=<your_token>
```
Models gated on HuggingFace require a token with terms-of-use accepted for the target model.

**`tensor_parallel_size` mismatch error**
The number of entries in `devices` must equal `tensor_parallel_size`. Either set only `devices` (TP is inferred), or set both consistently.

## Development

```bash
git clone https://github.com/furiosa-ai/furiosa-perf.git
cd furiosa-perf
pip install -e ".[dev]"
pre-commit install
pytest -q
```

---

For third-party component attributions, see [LICENSE](LICENSE).
