"""Run a preset's points with ``vllm bench serve`` and summarize them.

The client lives in its own virtual environment, pinned to
:data:`VLLM_BENCH_VERSION` so that results from different hosts and dates come
from the same client. Each point writes the client's result JSON next to its
log; the summary is built from that JSON, never from the console output.
"""

import json
from pathlib import Path
from typing import Any

from furiosa_perf.clients.base import BenchmarkClient
from furiosa_perf.runner.server import option_args

VLLM_BENCH_VERSION = "0.30.0"
RESULT_FILE = "result.json"

# A point is not trusted when more requests than this fraction failed.
MAX_FAILED_FRACTION = 0.05
# With ignore-eos every request must return exactly the requested output length;
# fewer tokens than this fraction of it means the client did not count them all.
MIN_OUTPUT_FRACTION = 0.9

# Statistics read from the result JSON for each latency metric.
LATENCY_STATS = ("mean", "median", "p25", "p50", "p75", "p90", "p95", "p99")
# result JSON metric -> (summary column suffix, divisor from ms, digits). The units match
# the earlier summary files, which the report reads.
LATENCY_COLUMNS = {"ttft": ("TTFT(s)", 1000.0, 4), "tpot": ("TPOT(ms)", 1.0, 3), "e2el": ("E2EL(s)", 1000.0, 4)}


def judge(point: dict[str, Any], result: dict[str, Any] | None, returncode: int) -> str:
    """Decide whether a point's numbers can be trusted.

    Args:
        point: Client options of the point.
        result: The client's result JSON, ``None`` if it wrote none.
        returncode: The client's exit code.

    Returns:
        ``""`` if the point is valid, otherwise the reason it is not.
    """
    if returncode != 0 or result is None:
        return f"client exited with code {returncode}" + ("" if result is not None else " without a result file")
    requested = int(result.get("num_prompts") or point.get("num-prompts") or 0)
    failed = int(result.get("failed") or 0)
    if requested and failed / requested > MAX_FAILED_FRACTION:
        return f"{failed} of {requested} requests failed"

    output_len = int(point.get("random-output-len") or 0)
    if output_len <= 0:
        return ""
    produced = int(result.get("total_output_tokens") or 0)
    if produced == 0:
        return "no output tokens counted"
    expected = int(result.get("completed") or 0) * output_len
    if point.get("ignore-eos") and produced < MIN_OUTPUT_FRACTION * expected:
        return f"outputs shorter than requested ({produced} of {expected} tokens)"
    return ""


def summary_row(point: dict[str, Any], result: dict[str, Any] | None, fail_reason: str) -> dict[str, Any]:
    """One summary row: the point's identity, its validity and the client's metrics.

    Column names and units are those of the earlier summary files, which the report
    reads. Metrics are kept as measured even when the point is invalid; ``ok`` and
    ``fail_reason`` say whether to trust them.

    Args:
        point: Client options of the point.
        result: The client's result JSON, ``None`` if it wrote none.
        fail_reason: Output of :func:`judge`.

    Returns:
        The row, with the same keys for every point.
    """
    result = result or {}

    def number(key: str, divisor: float = 1.0, digits: int = 2) -> float | None:
        value = result.get(key)
        return None if value is None else round(float(value) / divisor, digits)

    row: dict[str, Any] = {
        "Input Tokens": point.get("random-input-len"),
        "Output Tokens": point.get("random-output-len") or 0,
        "Concurrent": point.get("max-concurrency"),
        "ok": not fail_reason,
        "fail_reason": fail_reason,
        "completed": result.get("completed"),
        "failed": result.get("failed"),
        "Output Throughput(tok/s)": number("output_throughput"),
        "Total Throughput(tok/s)": number("total_token_throughput"),
        "Request Throughput(req/s)": number("request_throughput", digits=3),
    }
    for metric, (suffix, divisor, digits) in LATENCY_COLUMNS.items():
        for stat in LATENCY_STATS:
            row[f"{stat.title()} {suffix}"] = number(f"{stat}_{metric}_ms", divisor, digits)
    return row


class VllmBenchClient(BenchmarkClient):
    """``vllm bench serve``: fixed-length baseline points, one client run per point."""

    NAME = "vllm-bench"
    PACKAGE = "vllm"
    VERSION = VLLM_BENCH_VERSION
    EXECUTABLE = "vllm"

    def command(self, point: dict[str, Any], point_dir: Path) -> list[str]:
        """``vllm bench serve`` command line for one point."""
        command = [str(self.python), "-m", "vllm.entrypoints.cli.main", "bench", "serve"]
        command += ["--model", self.model, "--host", self.host, "--port", str(self.port)]
        # The server was already checked to be ready, so the client need not wait for it.
        command += ["--save-result", "--result-dir", str(point_dir), "--result-filename", RESULT_FILE]
        command += ["--ready-check-timeout-sec", "0"]
        for key, value in point.items():
            command += option_args(key, value)
        return command

    def read_result(self, point_dir: Path) -> dict[str, Any] | None:
        """The saved result JSON of a point."""
        path = point_dir / RESULT_FILE
        return json.loads(path.read_text()) if path.exists() else None

    def clear_result(self, point_dir: Path) -> None:
        """Remove a previous result JSON."""
        (point_dir / RESULT_FILE).unlink(missing_ok=True)

    def judge(self, point: dict[str, Any], result: dict[str, Any] | None, returncode: int) -> str:
        """See :func:`judge`."""
        return judge(point, result, returncode)

    def summary_row(self, point: dict[str, Any], result: dict[str, Any] | None, fail_reason: str) -> dict[str, Any]:
        """See :func:`summary_row`."""
        return summary_row(point, result, fail_reason)

    def concurrency(self, point: dict[str, Any]) -> int:
        """``max-concurrency`` of the point."""
        return int(point.get("max-concurrency") or 0)

    def group_key(self, row: dict[str, Any]) -> str | None:
        """``"<ISL>_<OSL>"`` such as ``"1k_1k"``: one extra summary file per length pair."""
        isl, osl = row["Input Tokens"], row["Output Tokens"]
        isl_key = f"{isl // 1024}k" if isl >= 1024 else str(isl)
        osl_key = f"{osl // 1024}k" if osl >= 1024 else str(osl)
        return f"{isl_key}_{osl_key}"
