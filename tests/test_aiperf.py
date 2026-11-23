"""Tests for turning an AIPerf export into a validated, SLO-judged summary row."""

from pathlib import Path
from typing import Any

import pytest

from furiosa_perf.clients.aiperf import AiperfClient, judge, max_users_within_slo, summary_row
from furiosa_perf.spec import load_preset

SLO = {"ttft_p95_ms": 3000.0, "tpot_p95_ms": 50.0}
POINT = {"concurrency": 32, "streaming": True, "endpoint-type": "chat"}


def export(ttft_p95: float = 900.0, itl_p95: float = 31.0, **overrides: Any) -> dict[str, Any]:
    """An AIPerf JSON export in the shape aiperf 0.13.0 writes (metric -> unit and statistics)."""
    return {
        "completed_request_count": {"unit": "requests", "avg": 640.0},
        "request_error_rate": {"unit": "%", "avg": 0.0},
        "request_throughput": {"unit": "requests/sec", "avg": 2.13},
        "goodput": {"unit": "requests/sec", "avg": 2.01},
        "output_token_throughput": {"unit": "tokens/sec", "avg": 640.0},
        "time_to_first_token": {"unit": "ms", "avg": 400.0, "p50": 350.0, "p95": ttft_p95, "p99": 1200.0},
        "inter_token_latency": {"unit": "ms", "avg": 22.0, "p50": 21.0, "p95": itl_p95, "p99": 40.0},
        **overrides,
    }


@pytest.mark.parametrize(
    ("result", "returncode", "reason"),
    [
        (export(), 0, ""),
        (None, 1, "client exited with code 1 without a result file"),
        (export(completed_request_count={"avg": 0.0}), 0, "no request completed"),
        (export(request_error_rate={"avg": 7.5}), 0, "7.5 % of requests failed"),
        (export(time_to_first_token={"unit": "ms"}), 0, "no time to first token measured"),
    ],
)
def test_judge(result: dict[str, Any] | None, returncode: int, reason: str) -> None:
    """Client errors, no completed request, over 5 % errors and a streaming run without TTFT invalidate a point."""
    assert judge(POINT, result, returncode) == reason


def test_summary_row_judges_the_slo_on_p95() -> None:
    """A point meets the SLO when both p95 values are within the limits; values are kept in ms."""
    met = summary_row(POINT, export(), "", SLO)
    missed = summary_row(POINT, export(itl_p95=55.0), "", SLO)

    assert (met["Users"], met["ok"], met["slo_met"]) == (32, True, True)
    assert met["P95 TTFT(ms)"] == 900.0 and met["P95 TPOT(ms)"] == 31.0
    assert met["Goodput(req/s)"] == 2.01
    assert missed["slo_met"] is False


def test_summary_row_has_no_slo_verdict_for_an_invalid_point() -> None:
    """An invalid point neither meets nor misses the SLO, and keeps the same columns."""
    row = summary_row(POINT, None, "client exited with code 1", SLO)

    assert row["slo_met"] is None
    assert list(row) == list(summary_row(POINT, export(), "", SLO))


def row(users: int, ok: bool = True, slo_met: bool | None = True) -> dict[str, Any]:
    """A summary row with just the fields capacity looks at."""
    return {"Users": users, "ok": ok, "slo_met": slo_met}


def test_capacity_is_the_last_user_count_before_the_first_miss() -> None:
    """A count passing again after a miss is noise, so capacity stops at the first miss or invalid point."""
    assert max_users_within_slo([row(8), row(16), row(32, slo_met=False), row(64)]) == 16
    assert max_users_within_slo([row(16), row(8)]) == 16  # order of the points does not matter
    assert max_users_within_slo([row(8, ok=False, slo_met=None), row(16)]) is None
    assert max_users_within_slo([]) is None


def test_command_targets_the_server_and_passes_the_slo_as_goodput(tmp_path: Path) -> None:
    """furiosa-perf sets the target, artifact directory and tokenizer; the SLO also becomes --goodput."""
    preset = load_preset("chat-synthetic", "scenario")
    client = AiperfClient(preset, "org/m", "0.0.0.0", 8000, tmp_path, {}, tokenizer="org/m-tokenizer")
    client.env = Path("/env")
    command = client.command(preset.points[0], tmp_path)

    assert command[:2] == ["/env/bin/aiperf", "profile"]
    assert command[command.index("--url") + 1] == "http://0.0.0.0:8000"
    assert command[command.index("--artifact-dir") + 1] == str(tmp_path)
    assert command[command.index("--tokenizer") + 1] == "org/m-tokenizer"
    assert command[command.index("--goodput") + 1] == "time_to_first_token:3000.0 inter_token_latency:50.0"
    assert command[command.index("--extra-inputs") + 1] == '{"ignore_eos": true}'
    assert "--streaming" in command
