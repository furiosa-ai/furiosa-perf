# furiosa-perf

`furiosa-perf` is a performance benchmarking CLI for **LLM serving**.

It automates the full benchmark pipeline: launch an API server → run [vLLM benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks) scenarios → collect hardware metrics → save CSV/Markdown summaries.

**Supported backends:**

| Backend | Hardware |
|---|---|
| `furiosa-llm` | Furiosa RNGD (NPU) |
| `vllm` | NVIDIA GPU |

---

## Key features

- **End-to-end automation** — server launch, benchmark execution, and teardown in one command
- **Hardware monitoring** — power, temperature, utilization (NPU/GPU), host CPU/memory, and server `/metrics`, logged to CSV
- **Multiple tasks** — `offline` (LLM text), `vl-offline` (vision-language), `reranker`, `embeddings`
- **Flexible scenarios** — define token lengths and concurrency lists in a single YAML; `max_concurrency` can be a list and is auto-expanded

---

## Requirements

- **OS**: Linux
- **Python**: 3.12+
- **APT**: `python3.12-venv` (used internally to create a venv for the vLLM benchmark tool)

```bash
sudo apt-get install -y python3.12-venv
```

### Backend prerequisites

**Furiosa NPU + `furiosa-llm`**

```bash
pip install furiosa-llm
```

**NVIDIA GPU + vLLM**

```bash
pip install vllm
# nvidia-smi must be available for GPU monitoring
```

---

## Installation

```bash
git clone https://github.com/furiosa-ai/furiosa-perf.git
cd furiosa-perf

python3.12 -m venv .venv
source .venv/bin/activate

pip install -e .
```

Backend-specific extras:

```bash
pip install ".[furiosa-llm]"   # Furiosa NPU
pip install ".[vllm]"          # NVIDIA GPU
```

> **Note:** activate the venv (`source .venv/bin/activate`) before every session, or add it to your shell profile.

### HuggingFace token (gated models)

```bash
export HF_TOKEN=hf_...
# or
huggingface-cli login
```

---

## Quick start

```bash
furiosa-perf run \
  --backend furiosa-llm \
  --hardware-type npu \
  --server-config examples/server/exaone4.yaml \
  --benchmark-config examples/bench/llm_scenario.yaml \
  --model-id furiosa-ai/EXAONE-4.0-32B-FP8
```

See [`examples/script.sh`](examples/script.sh) for a complete working example.

### CLI options

| Option | Required | Default | Description |
|---|---|---|---|
| `--backend` | | `furiosa-llm` | Serving backend (`furiosa-llm` or `vllm`) |
| `--hardware-type` | | `npu` | Accelerator type (`npu` or `gpu`) |
| `--server-config` | ✓ | — | API server config YAML |
| `--benchmark-config` | ✓ | — | Benchmark scenario config YAML |
| `--model-id` | | `LGAI-EXAONE/EXAONE-4.0-32B-FP8` | HuggingFace model ID or local path |

---

## Config reference

### Server config (`--server-config`)

Controls how the API server is launched. Keys differ slightly by backend.

**Example** (`examples/server/exaone4.yaml`):

```yaml
served_model_name: furiosa-ai/EXAONE-4.0-32B-FP8
devices: 0,1,2,3
```

**Common keys:**

| Key | Default | Description |
|---|---|---|
| `host` | `0.0.0.0` | Server bind address |
| `port` | `8000` | Server port |
| `devices` | — | Comma-separated RNGD indices to use for serving (e.g. `0,1,2,3`) |
| `tensor_parallel_size` | inferred from `devices` | TP degree |
| `served_model_name` | — | Model alias exposed on `/v1/models` |

**Choosing `devices` (Furiosa RNGD)**

`devices` specifies which RNGD cards to use and how many to allocate for the model. The number of devices determines the tensor parallelism degree — larger models require more cards.

First, check how many RNGDs are available in your environment:

```bash
$ furiosa-smi info
+------+--------+-------------------+---------+---------+--------------+
| Arch | Device | Firmware          | Temp.   | Power   | PCI-BDF      |
+------+--------+-------------------+---------+---------+--------------+
| rngd | npu0   | 2026.2.0, a5db283 | 36.33°C | 39.00 W | 0000:03:00.0 |
+------+--------+-------------------+---------+---------+--------------+
| rngd | npu1   | 2026.2.0, a5db283 | 35.99°C | 38.00 W | 0000:04:00.0 |
+------+--------+-------------------+---------+---------+--------------+
| rngd | npu2   | 2026.2.0, a5db283 | 34.03°C | 39.00 W | 0000:44:00.0 |
+------+--------+-------------------+---------+---------+--------------+
| rngd | npu3   | 2026.2.0, a5db283 | 35.76°C | 39.00 W | 0000:45:00.0 |
+------+--------+-------------------+---------+---------+--------------+
```

Use the numbers from the **Device** column (`npu0` → `0`, `npu1` → `1`, …) as the `devices` value. For example, to use the first 4 cards: `devices: 0,1,2,3`. If two users share an 8-card server, one can use `0,1,2,3` and the other `4,5,6,7`.

**Supported models and recommended device count**

Furiosa AI supported models are available at [huggingface.co/furiosa-ai](https://huggingface.co/furiosa-ai).

| Model (HuggingFace ID) | Required RNGDs | Example `devices` |
|---|:---:|---|
| `furiosa-ai/EXAONE-4.0-32B-FP8` | 4 | `0,1,2,3` or `4,5,6,7` |
| `furiosa-ai/Llama-3.3-70B-Instruct` | 4 | `0,1,2,3` or `4,5,6,7` |
| `furiosa-ai/Qwen3-32B-FP8` | 4 | `0,1,2,3` or `4,5,6,7` |
| `furiosa-ai/Llama-3.1-8B-Instruct` | 1 | `0` or any device available in your environment |
| `furiosa-ai/Qwen2.5-0.5B-Instruct` | 1 | `0` or any device available in your environment |
| `furiosa-ai/Qwen3-Embedding-8B` | 1 | `0` or any device available in your environment |
| `furiosa-ai/Qwen3-Reranker-8B` | 1 | `0` or any device available in your environment |

**Notes:**

- `furiosa-llm` backend: `devices: 0,1,2,3` → `--devices npu:0,npu:1,npu:2,npu:3` (converted automatically)
- `vllm` backend: `devices` sets `CUDA_VISIBLE_DEVICES`

---

### Benchmark config (`--benchmark-config`)

Defines the benchmark name, task, and scenarios.

**Example** (`examples/bench/llm_scenario.yaml`):

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
```

`max_concurrency` can be a single integer or a list — lists are automatically expanded into separate runs.

**Supported tasks:**

| `task` | Description | Required extra fields |
|---|---|---|
| `offline` | LLM text generation | `output_tokens`, `random_range_ratio` |
| `vl-offline` | Vision-language generation | `output_tokens`, `random_mm_*` fields |
| `reranker` | Reranking | `random_batch_size` |
| `embeddings` | Text embeddings | — |

---

## Outputs

Results are written under `./bench_space/` relative to your working directory.

```
bench_space/
└── <DEVICE>_<NUM>_<BACKEND>_<VERSION>/
    └── <name>/<task>/<model>/
        ├── summary_<ISL>_<OSL>.md        # Markdown table
        ├── summary_<ISL>_<OSL>.csv       # CSV summary
        └── <ISL>.<OSL>.<CONC>/
            ├── <DEVICE>_<NUM>_monitoring_log.csv   # per-scenario hardware metrics
            └── vllm-*.json                         # raw vLLM benchmark output
```

Server logs are written to `./serve_logs/`.

**Example path:**

```
bench_space/RNGD_4_furiosa-llm_2026.2.0/vllm/offline/EXAONE-4.0-32B-FP8/summary_1k_1k.csv
```

---

## Troubleshooting

**Monitoring CSV is empty or missing**
- NPU: requires `furiosa_smi_py`. If not installed, NPU monitoring is silently skipped.
- GPU: requires `nvidia-smi`. If not found, GPU monitoring is silently skipped.

**Server fails to start**
- Check `./serve_logs/` for the server process output.
- Run the backend command directly to inspect startup errors.

**`HF_TOKEN` not set for gated models**
- Export `HF_TOKEN` or run `huggingface-cli login` before benchmarking.

---

## Development

```bash
pip install -e ".[dev]"

# Install lint tools (shfmt, shellcheck, ruff, mypy, yamlfmt, yamllint, actionlint)
make install-tools

# Run all linters
make lint

# Run tests
make test

# Clean caches
make clean
```
