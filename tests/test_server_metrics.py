"""Tests for parsing the serving framework's ``/metrics`` exposition."""

import pytest

from furiosa_perf.runner import server_metrics as sm

# Lines in the shape furiosa-llm 2026.4.0rc0 exposes them (captured from a live server).
FURIOSA_START = """
# TYPE furiosa_llm_prompt_tokens_total counter
furiosa_llm_prompt_tokens_total{model_name="m",engine="0"} 100
furiosa_llm_generation_tokens_total{model_name="m",engine="0"} 1000
furiosa_llm_request_success_total{model_name="m",engine="0",finished_reason="length"} 4
furiosa_llm_request_success_total{model_name="m",engine="0",finished_reason="stop"} 0
furiosa_llm_time_to_first_token_seconds_bucket{model_name="m",engine="0",le="0.1"} 2
furiosa_llm_time_to_first_token_seconds_bucket{model_name="m",engine="0",le="0.5"} 4
furiosa_llm_time_to_first_token_seconds_bucket{model_name="m",engine="0",le="+Inf"} 4
furiosa_llm_time_to_first_token_seconds_sum{model_name="m",engine="0"} 0.8
furiosa_llm_time_to_first_token_seconds_count{model_name="m",engine="0"} 4
"""
FURIOSA_END = """
furiosa_llm_prompt_tokens_total{model_name="m",engine="0"} 600
furiosa_llm_generation_tokens_total{model_name="m",engine="0"} 11000
furiosa_llm_request_success_total{model_name="m",engine="0",finished_reason="length"} 14
furiosa_llm_request_success_total{model_name="m",engine="0",finished_reason="stop"} 0
furiosa_llm_time_to_first_token_seconds_bucket{model_name="m",engine="0",le="0.1"} 2
furiosa_llm_time_to_first_token_seconds_bucket{model_name="m",engine="0",le="0.5"} 12
furiosa_llm_time_to_first_token_seconds_bucket{model_name="m",engine="0",le="+Inf"} 14
furiosa_llm_time_to_first_token_seconds_sum{model_name="m",engine="0"} 3.8
furiosa_llm_time_to_first_token_seconds_count{model_name="m",engine="0"} 14
furiosa_llm_kv_cache_usage_percent{model_name="m",engine="0"} 0.25
furiosa_llm_cache_config_info{engine="0",block_size="1",num_gpu_blocks="266078",enable_prefix_caching="true"} 1
"""


def test_parse_strips_prefix_and_total_and_keeps_labels() -> None:
    """Names lose the backend prefix and ``_total``; labels and values are kept; other prefixes are skipped."""
    samples = sm.parse(FURIOSA_END + 'vllm:num_requests_running{x="1"} 3\n', sm.FURIOSA_PREFIX)

    assert sm.Sample("prompt_tokens", {"model_name": "m", "engine": "0"}, 600.0) in samples
    info = next(s for s in samples if s.name == "cache_config_info")
    assert info.labels["num_gpu_blocks"] == "266078"
    assert all(not s.name.startswith("vllm") for s in samples)


def test_sample_columns_sums_counts_and_maxes_ratios() -> None:
    """Running requests add up across engines; KV usage takes the max engine and is scaled from 0-1 to 0-100."""
    text = """
furiosa_llm_num_requests_running{engine="0"} 3
furiosa_llm_num_requests_running{engine="1"} 5
furiosa_llm_kv_cache_usage_percent{engine="0"} 0.25
furiosa_llm_kv_cache_usage_percent{engine="1"} 0.75
furiosa_llm_prefix_cache_hits{engine="0"} 90
"""
    columns = sm.sample_columns(sm.parse(text, sm.FURIOSA_PREFIX))

    assert columns == {"num_requests_running": 8.0, "kv_cache_usage_pct": 75.0, "prefix_cache_hits": 90.0}


def test_sample_columns_accepts_older_vllm_names() -> None:
    """vLLM's pre-v1 ``gpu_cache_usage_perc`` fills the same KV column as the current name."""
    columns = sm.sample_columns(sm.parse("vllm:gpu_cache_usage_perc 0.5\nvllm:num_preemptions_total 2", sm.VLLM_PREFIX))

    assert columns == {"kv_cache_usage_pct": 50.0, "num_preemptions_total": 2.0}


def test_summarize_snapshots_reports_mean_latency_and_finish_reasons() -> None:
    """Over two snapshots: 10 requests took 3.0 s of TTFT in total, so 300 ms each; absent histograms are None."""
    summary = sm.summarize_snapshots(FURIOSA_START, FURIOSA_END, sm.FURIOSA_PREFIX)

    assert summary["server_ttft_mean_ms"] == pytest.approx(300.0)
    assert summary["server_itl_mean_ms"] is None
    assert summary["server_e2el_mean_ms"] is None
    assert summary["finished_length"] == 10.0
    assert summary["finished_stop"] == 0.0
    assert summary["finished_abort"] == 0.0


def test_summarize_snapshots_without_new_requests_has_no_latency() -> None:
    """Identical snapshots mean nothing finished in the window, so the means are None, not a division by zero."""
    summary = sm.summarize_snapshots(FURIOSA_START, FURIOSA_START, sm.FURIOSA_PREFIX)

    assert summary["server_ttft_mean_ms"] is None
    assert summary["finished_length"] == 0.0
