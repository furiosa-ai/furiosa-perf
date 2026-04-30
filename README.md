## furiosa-perf

`furiosa-perf` is a performance benchmarking tool for **LLM serving**.

Its primary goal is to measure the performance of **Furiosa AI NPU** with **`furiosa-llm` (LLM API framework)** using the **vLLM benchmark tool** as the workload generator/metric collector.

For comparison, it also supports benchmarking **NVIDIA GPUs** with LLM frameworks (currently **vLLM**, with more backends planned).

It also provides an **experimental** `report` feature that aggregates benchmark results into a browsable **HTML dashboard**.

### Key features

- **End-to-end run**: launch an API server → run vLLM benchmark scenarios → save summaries
- **Hardware monitoring**: power/temperature/utilization + host CPU/memory + server `/metrics` into CSV logs
- **Experimental HTML report**: collect `*.csv` files and render a Plotly-based report

---

## Installation

### System requirements

- **OS**: Linux (recommended)
- **Python**: 3.10+
- **APT dependency**: `python3-venv` (required to create a local Python venv for the benchmark tool)

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

### Backend prerequisites

- **Furiosa NPU + `furiosa-llm`**
  - Install `furiosa-llm`
  - (Optional, for monitoring) `furiosa_smi_py` available (`furiosa-smi` installed and working)
- **NVIDIA GPU + vLLM**
  - Install `vllm`
  - (Optional, for monitoring) `nvidia-smi` available

### Install `furiosa-perf`

From the repository root:

```bash
git clone https://github.com/furiosa-ai/furiosa-perf.git
cd furiosa-perf
python -m pip install -U pip
python -m pip install .
```

Optional extras (backend-specific):

```bash
# For `furiosa-llm` backend
python -m pip install ".[furiosa-llm]"

# For vLLM backend
python -m pip install ".[vllm]"
```

---

## Quick start (run a benchmark)

Example config files:

- API server config: `examples/server/exaone.yaml`
- Benchmark config: `examples/vllm/exaone.yaml`

Run:

```bash
furiosa-perf run \
  --backend furiosa-llm \
  --hardware-type npu \
  --server-config /root/furiosa-perf/examples/server/exaone.yaml \
  --benchmark-config /root/furiosa-perf/examples/vllm/exaone.yaml \
  --model "LGAI-EXAONE/EXAONE-4.0-32B-FP8"
```

Options:

- `--full`: save full metrics (including p95/p99) into `summary.*`
- `--dev`: use Furiosa custom vLLM benchmark tools instead of the official vLLM tools
- `--save-api-log`: redirect API server stdout/stderr to `api_server.log` in the working directory (suppressed by default)

---

## Model List

The table below lists models validated on **Furiosa RNGD** (example).

| Model (HF ID) | TP (NPU count) | Base Model (HF ID) |
|---|---:|---|
| `furiosa-ai/EXAONE-4.0-32B-FP8` | 4 | LGAI-EXAONE/EXAONE-4.0-32B |
| `furiosa-ai/Qwen3-32B-FP8` | 4 | Qwen/Qwen2.5-32B-Instruct |
| `furiosa-ai/Llama-3.3-70B-Instruct` | 4 | meta-llama/Llama-3.3-70B-Instruct	|
| `furiosa-ai/Llama-3.1-8B-Instruct` | 1 | meta-llama/Llama-3.1-8B-Instruct	|

## Config formats

### 1) API server config (`--server-config`)

YAML file for API server launch settings (options differ by backend).

Example (`examples/server/exaone.yaml`):

```yaml
devices: "0,1,2,3"
```

Common keys (partial):

- `host` (default: `0.0.0.0`)
- `port` (default: `8000`)
- `devices`: comma-separated device indices (e.g. `0,1,2,3`)
- `tensor_parallel_size`: if omitted, inferred from `devices` length (backend-dependent)

Notes:

- vLLM backend sets `CUDA_VISIBLE_DEVICES` from `devices`.
- `furiosa-llm` backend converts `devices: 0,1,2,3` into `--devices npu:0,npu:1,npu:2,npu:3`.

- **RNGD device indexing**: for Furiosa RNGD, set `devices` using **only the device numbers shown in the `Device` column** of `furiosa-smi info` (e.g., `npu0..npu3` → `devices: 0,1,2,3`).

Example (`furiosa-smi info` → `devices`):

```sh
$ furiosa-smi info
+------+--------+------------------------+---------+---------+--------------+
| Arch | Device | Firmware               | Temp.   | Power   | PCI-BDF      |
+------+--------+------------------------+---------+---------+--------------+
| rngd | npu0   | 2026.1.0(rc0), 5b172ac | ...     | ...     | 0000:03:00.0 |
+------+--------+------------------------+---------+---------+--------------+
| rngd | npu1   | 2026.1.0(rc0), 5b172ac | ...     | ...     | 0000:04:00.0 |
+------+--------+------------------------+---------+---------+--------------+
| rngd | npu2   | 2026.1.0(rc0), 5b172ac | ...     | ...     | 0000:44:00.0 |
+------+--------+------------------------+---------+---------+--------------+
| rngd | npu3   | 2026.1.0(rc0), 5b172ac | ...     | ...     | 0000:45:00.0 |
+------+--------+------------------------+---------+---------+--------------+
```

```yaml
devices: 0,1,2,3
```

### 2) Benchmark config (`--benchmark-config`)

YAML file specifying the benchmark name, task, and scenarios.

Example (`examples/vllm/exaone.yaml`):

```yaml
name: vllm
task: offline
scenarios:
  - input_tokens: 1024
    output_tokens: 1024
    max_concurrency: [1, 4, 8, 16]
```

- `task`: currently `offline` is the primary supported task (others are planned/experimental)
- `scenarios`: `max_concurrency` can be a list or integer and will be expanded automatically.

---

## Outputs

By default, results are written under `./bench_space/` (relative to your current working directory).

Approximate structure:

- `bench_space/<name>/`
  - `venv/`: local virtual environment used to run the benchmark tool (vLLM bench)
  - `<DEVICE>_<NUM>_<BACKEND>_<VERSION>/`
    - `<name>/<task>/<model>/`
      - `summary_<ISL>_<OSL>.csv`, `summary_<ISL>_<OSL>.md`
      - `<ISL>.<OSL>.<CONC>/`
        - `*_monitoring_log.csv`
        - `vllm-*.json` (raw vLLM bench outputs)
- `api_server.log` (only when `--save-api-log` is passed): API server stdout/stderr, written to the working directory

Example:

- `bench_space/vllm/RNGD_4_furiosa-llm_2026.1.0rc2/vllm/offline/EXAONE-4.0-32B-FP8/summary_1k_1k.csv`

---

## Experimental: HTML report

This feature aggregates multiple `*.csv` files (benchmark summary files) and renders an HTML report.

```bash
furiosa-perf report \
  --benchmark-result-path /root/bench_space/vllm \
  --model-list EXAONE-4.0-32B-FP8 \
  --report-path /root/furiosa-perf-report
```

Open:

- `/root/furiosa-perf-report/index.html` in a browser.

---

## Troubleshooting / tips

- **Monitoring CSV is missing**
  - NPU: if `furiosa_smi_py` is not available, NPU monitoring is skipped.
  - GPU: if `nvidia-smi` is not available, GPU monitoring is skipped.
- **Server logs are not visible**
  - By default, server stdout/stderr is suppressed. Pass `--save-api-log` to capture them into `api_server.log` in the working directory. Alternatively, run the backend command directly to inspect startup errors.

---

## Development

```bash
cd /root/furiosa-perf
python -m pip install -e ".[dev]"
pytest -q
```