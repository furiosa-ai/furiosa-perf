"""Tests for the vendored multi-turn benchmark integration (no live server needed)."""

import json
import sys
from importlib import resources
from pathlib import Path

import pytest
import yaml

from furiosa_perf.benchmark.multi_turn_input import (
    _peak_input_tokens,
    build_multi_turn_input,
)
from furiosa_perf.utils.config import MultiTurnScenarioConfig, PerformanceBenchConfigLoader

VENDOR_DIR = Path(str(resources.files("furiosa_perf.vendor.multi_turn")))
TEXT_FILE = VENDOR_DIR / "data" / "pg1184.txt"
EXAMPLE_YAML = Path(__file__).resolve().parents[1] / "examples" / "bench" / "multi_turn_scenario.yaml"


def _import_bench_dataset():
    """Import the vendored bench_dataset (flat sibling imports need the dir on path)."""
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    import bench_dataset  # noqa: PLC0415

    return bench_dataset


def _example_scenarios() -> list[MultiTurnScenarioConfig]:
    cfg = yaml.safe_load(EXAMPLE_YAML.read_text())
    return PerformanceBenchConfigLoader._create_config(cfg).scenarios


# ---- 1. Generated input JSON validates against the vendored parser ----------------


def test_build_input_validates_against_vendored_parser():
    bench_dataset = _import_bench_dataset()
    for scenario in _example_scenarios():
        doc = build_multi_turn_input(scenario, TEXT_FILE)
        # required top-level fields injected
        assert doc["filetype"] == "generate_conversations"
        assert doc["num_conversations"] >= scenario.max_concurrency
        assert Path(doc["text_files"][0]).is_absolute()
        # the vendored parser asserts on the schema; must not raise
        bench_dataset.parse_input_json_file(doc)


def test_scalar_and_dict_shape_specs():
    bench_dataset = _import_bench_dataset()
    scenario = MultiTurnScenarioConfig(
        max_concurrency=4,
        num_turns={"distribution": "uniform", "min": 24, "max": 40},
        prefix_num_tokens=1000,
        input_num_tokens=128,
        common_prefix_num_tokens=64,
        output_num_tokens={"distribution": "lognormal", "average": 100, "max": 200},
    )
    doc = build_multi_turn_input(scenario, TEXT_FILE)
    assert doc["prompt_input"]["num_turns"]["distribution"] == "uniform"
    assert doc["prompt_input"]["num_tokens"] == {"distribution": "constant", "value": 128}
    assert "common_prefix_num_tokens" in doc["prompt_input"]
    bench_dataset.parse_input_json_file(doc)


# ---- 2. Config expansion + validation ---------------------------------------------


def test_config_expansion_sweeps_clients():
    cfg = {
        "name": "vllm",
        "task": "multi_turn",
        "scenarios": [{"max_concurrency": [1, 8, 32], "num_conversations": 16}],
    }
    scenarios = PerformanceBenchConfigLoader._create_config(cfg).scenarios
    assert [s.max_concurrency for s in scenarios] == [1, 8, 32]
    # max_active_conversations defaults to num_clients; num_conversations bumped >= clients
    assert scenarios[0].max_active_conversations == 1
    assert scenarios[-1].num_conversations >= 32


def test_paired_limit_validation():
    with pytest.raises(ValueError):
        MultiTurnScenarioConfig(max_concurrency=1, limit_min_tokens=100)  # only one side
    with pytest.raises(ValueError):
        MultiTurnScenarioConfig(max_concurrency=1, limit_min_tokens=500, limit_max_tokens=100)  # min > max


# ---- 3. Input-context clamp -------------------------------------------------------


def test_context_clamp_fits_budget():
    scenario = MultiTurnScenarioConfig(
        max_concurrency=1,
        num_turns=128,
        prefix_num_tokens=45056,
        input_num_tokens=1024,
        output_num_tokens=256,
        limit_min_tokens=256,
        limit_max_tokens=256,
        max_input_context_tokens=32768,
    )
    doc = build_multi_turn_input(scenario, TEXT_FILE)
    pi = doc["prompt_input"]
    peak = _peak_input_tokens(
        pi["num_turns"]["value"], pi["prefix_num_tokens"]["value"], 0,
        pi["num_tokens"]["value"], doc["prompt_output"]["num_tokens"]["value"],
    )
    assert peak <= 32768 - 256 + 1


def test_context_clamp_noop_when_small():
    scenario = MultiTurnScenarioConfig(
        max_concurrency=1, num_turns=4, prefix_num_tokens=1000,
        input_num_tokens=500, output_num_tokens=100, max_input_context_tokens=100_000,
    )
    doc = build_multi_turn_input(scenario, TEXT_FILE)
    assert doc["prompt_input"]["prefix_num_tokens"] == {"distribution": "constant", "value": 1000}


# ---- 4. stats.json -> summary parsing ---------------------------------------------


def test_parse_multi_turn_results(tmp_path, monkeypatch):
    import furiosa_perf.benchmark.vllm as vllm_mod

    monkeypatch.setattr(vllm_mod, "WORKSPACE", str(tmp_path))
    from furiosa_perf.utils.config import PerformanceBenchConfig

    scenario = MultiTurnScenarioConfig(max_concurrency=4, num_conversations=8)
    cfg = PerformanceBenchConfig(
        name="vllm", task="multi_turn", model="m", device_name="RTX-PRO-6000",
        used_device_num=4, scenarios=[scenario],
    )
    bench = vllm_mod.VllmPerformanceBenchmark(cfg, backend="vllm", host="h", port=1)
    result_dir = bench.get_vllm_result_dir(bench._scenario_result_name(scenario))
    records = [
        {"ttft_ms": 100.0, "tpot_ms": 10.0, "latency_ms": 500.0, "start_time_ms": 0.0,
         "input_num_tokens": 1000, "output_num_tokens": 200},
        {"ttft_ms": 200.0, "tpot_ms": 20.0, "latency_ms": 700.0, "start_time_ms": 100.0,
         "input_num_tokens": 1100, "output_num_tokens": 210},
        {"ttft_ms": -1.0, "tpot_ms": 15.0, "latency_ms": 600.0, "start_time_ms": 50.0,
         "input_num_tokens": 1050, "output_num_tokens": 205},  # ttft=-1 dropped for TTFT
    ]
    (result_dir / "stats.json").write_text(json.dumps(records))
    stdout = "\x1b[32mAll clients finished, benchmark runtime: 12.500 sec (12500.000 ms), requests per second: 40.000\x1b[0m"

    row = bench._parse_multi_turn_results(stdout, scenario)
    assert row["Requests"] == 3
    assert row["Requests/sec"] == 40.0
    assert row["Runtime(s)"] == 12.5
    assert row["Concurrent"] == 4
    # TTFT percentiles ignore the -1 sentinel (only 100/200 ms remain)
    assert row["P50 TTFT(s)"] == pytest.approx(0.15, abs=1e-6)
    assert row["Mean TTFT(s)"] == pytest.approx(0.15, abs=1e-6)
    assert row["P90 TPOT(ms)"] > 0


def test_parse_multi_turn_recomputes_rps_without_stdout(tmp_path, monkeypatch):
    import furiosa_perf.benchmark.vllm as vllm_mod

    monkeypatch.setattr(vllm_mod, "WORKSPACE", str(tmp_path))
    from furiosa_perf.utils.config import PerformanceBenchConfig

    scenario = MultiTurnScenarioConfig(max_concurrency=2, num_conversations=4)
    cfg = PerformanceBenchConfig(name="vllm", task="multi_turn", model="m",
                                 device_name="d", used_device_num=1, scenarios=[scenario])
    bench = vllm_mod.VllmPerformanceBenchmark(cfg, backend="vllm", host="h", port=1)
    result_dir = bench.get_vllm_result_dir(bench._scenario_result_name(scenario))
    records = [{"ttft_ms": 10.0, "tpot_ms": 1.0, "latency_ms": 1000.0, "start_time_ms": 0.0,
                "input_num_tokens": 10, "output_num_tokens": 5}]
    (result_dir / "stats.json").write_text(json.dumps(records))
    row = bench._parse_multi_turn_results("no aggregate line here", scenario)
    assert row["Requests"] == 1
    assert row["Requests/sec"] > 0  # recomputed from timestamps


# ---- 6. Report does not pick up multi-turn output ---------------------------------


def test_report_ignores_multiturn_summary(tmp_path):
    from furiosa_perf.cli.report import collect_all_models, collect_all_tasks

    model_dir = tmp_path / "vllm" / "RTX_4_vllm" / "vllm" / "multi_turn" / "EXAONE"
    model_dir.mkdir(parents=True)
    (model_dir / "multiturn_summary.csv").write_text("a,b\n1,2\n")
    (model_dir / "nc8.conv16").mkdir()
    (model_dir / "nc8.conv16" / "stats.json").write_text("[]")

    assert collect_all_models(str(tmp_path)) == []
    assert collect_all_tasks(str(tmp_path)) == []
