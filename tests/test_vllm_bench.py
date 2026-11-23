"""Tests for turning a vllm bench result into a validated summary row."""

from typing import Any

import pytest

from furiosa_perf.clients.vllm_bench import judge, summary_row

TEXT_POINT = {"random-input-len": 1024, "random-output-len": 128, "max-concurrency": 8, "num-prompts": 24}


def result(**overrides: Any) -> dict[str, Any]:
    """A vllm bench result JSON for TEXT_POINT where every request returned 128 tokens."""
    return {
        "num_prompts": 24,
        "completed": 24,
        "failed": 0,
        "total_output_tokens": 24 * 128,
        "output_throughput": 612.5,
        "total_token_throughput": 5512.5,
        "request_throughput": 4.785,
        "mean_ttft_ms": 250.0,
        "p99_ttft_ms": 731.0,
        "mean_tpot_ms": 12.29,
        "mean_e2el_ms": 1810.0,
        **overrides,
    }


@pytest.mark.parametrize(
    ("point", "res", "returncode", "reason"),
    [
        (TEXT_POINT, result(), 0, ""),
        (TEXT_POINT, result(failed=1, completed=23, total_output_tokens=23 * 128), 0, ""),  # 4 % failed
        (TEXT_POINT, result(failed=2, completed=22), 0, "2 of 24 requests failed"),
        (TEXT_POINT, result(total_output_tokens=0), 0, "no output tokens counted"),
        ({**TEXT_POINT, "ignore-eos": True}, result(total_output_tokens=1000), 0, "outputs shorter than requested"),
        (TEXT_POINT, result(total_output_tokens=1000), 0, ""),  # without ignore-eos short outputs are legitimate
        (TEXT_POINT, None, 1, "client exited with code 1 without a result file"),
        ({"random-input-len": 256, "max-concurrency": 1}, result(total_output_tokens=0), 0, ""),  # embeddings
    ],
)
def test_judge(point: dict[str, Any], res: dict[str, Any] | None, returncode: int, reason: str) -> None:
    """Failed requests above 5 %, uncounted or short outputs and client crashes invalidate a point."""
    got = judge(point, res, returncode)
    assert got.startswith(reason) if reason else got == ""


def test_summary_row_keeps_report_columns_and_units() -> None:
    """Column names and units are the ones the report reads: TTFT and E2EL in s, TPOT in ms."""
    row = summary_row(TEXT_POINT, result(), "")

    assert (row["Input Tokens"], row["Output Tokens"], row["Concurrent"]) == (1024, 128, 8)
    assert row["ok"] is True
    assert row["Output Throughput(tok/s)"] == 612.5
    assert row["Mean TTFT(s)"] == 0.25
    assert row["P99 TTFT(s)"] == 0.731
    assert row["Mean TPOT(ms)"] == 12.29
    assert row["Mean E2EL(s)"] == 1.81
    assert row["Median TTFT(s)"] is None  # absent in the result, so absent in the row


def test_summary_row_of_an_invalid_point_keeps_the_measured_values() -> None:
    """Invalid points are flagged, not blanked: the report decides how to show them."""
    row = summary_row(TEXT_POINT, result(total_output_tokens=0, output_throughput=0.0), "no output tokens counted")

    assert row["ok"] is False
    assert row["fail_reason"] == "no output tokens counted"
    assert row["Output Throughput(tok/s)"] == 0.0
    assert list(row) == list(summary_row(TEXT_POINT, None, "x"))  # same columns either way
