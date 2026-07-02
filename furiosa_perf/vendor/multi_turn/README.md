# Vendored vLLM multi-turn benchmark

This directory contains files copied **verbatim** from the
[vLLM project](https://github.com/vllm-project/vllm), path
`benchmarks/multi_turn/`, licensed under **Apache-2.0**.

- **Upstream:** https://github.com/vllm-project/vllm
- **Path:** `benchmarks/multi_turn/`
- **Pinned commit:** `320c52b1342ad961091bb3333b867c0899907b06`
- **Retrieved:** 2026-07-02

## Vendored files

| File | Origin |
|------|--------|
| `benchmark_serving_multi_turn.py` | `benchmarks/multi_turn/benchmark_serving_multi_turn.py` |
| `bench_dataset.py` | `benchmarks/multi_turn/bench_dataset.py` |
| `bench_utils.py` | `benchmarks/multi_turn/bench_utils.py` |
| `requirements.txt` | `benchmarks/multi_turn/requirements.txt` |
| `data/pg1184.txt` | `benchmarks/multi_turn/pg1184.txt` (Project Gutenberg eBook #1184, public domain) |

The scripts import each other as **flat siblings**
(`from bench_dataset import ...`, `from bench_utils import ...`), so they must be
run with this directory on `PYTHONPATH` (see
`furiosa_perf/benchmark/vllm.py`, the `multi_turn` task branch).

## Maintenance

Do **not** edit the vendored `.py` files. To update, re-copy from the upstream
path at a new commit and update the pinned commit / date above. The Apache-2.0
SPDX headers at the top of each source file must be preserved. See the root
`NOTICE` file for attribution.
