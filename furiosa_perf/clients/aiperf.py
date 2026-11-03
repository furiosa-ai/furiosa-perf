"""Run service-scenario points with AIPerf and summarize them against the preset's SLO.

One point is one ``aiperf profile`` run. For chat presets a point's ``concurrency``
is the number of users in a conversation at the same time: AIPerf keeps a session's
slot during the user's think time between turns (checked against a logging mock
server), and each turn carries the conversation so far, including the responses
the server generated, so prefix caching is exercised as in a real chat service.
"""

import json
from pathlib import Path
from typing import Any

from furiosa_perf.clients.base import CLIENT_LOG, BenchmarkClient
from furiosa_perf.clients.traces import TraceWindow, fetch_trace, load_trace, scale_trace, warmup_rows, write_trace
from furiosa_perf.log import logger
from furiosa_perf.results.store import point_name
from furiosa_perf.runner.server import option_args
from furiosa_perf.spec import TRACE_POINT_KEYS

AIPERF_VERSION = "0.13.0"
RESULT_FILE = "profile_export_aiperf.json"
# AIPerf writes these next to RESULT_FILE; all are removed before a point reruns.
RESULT_FILE_PREFIX = "profile_export"

# A point is not trusted when more than this share of its requests failed.
MAX_ERROR_RATE_PCT = 5.0
# Preset SLO key -> (AIPerf metric, statistic) it limits.
SLO_METRICS = {"ttft_p95_ms": ("time_to_first_token", "p95"), "tpot_p95_ms": ("inter_token_latency", "p95")}
LATENCY_STATS = ("avg", "p50", "p90", "p95", "p99")
# Trace replay: file written per point, and wall-clock seconds a point lasts by default.
POINT_TRACE_FILE = "trace.jsonl"
DEFAULT_TRACE_DURATION_S = 600
# Directory, inside a point's, of the cache warm-up run.
WARMUP_DIR = "warmup"
# Point options that bound the measured run only; the warm-up runs to completion.
MEASURE_ONLY_KEYS = ("benchmark-duration", "benchmark-grace-period", "goodput")


def metric(result: dict[str, Any] | None, name: str, stat: str = "avg") -> float | None:
    """One statistic of an AIPerf metric (``{"unit": ..., "avg": ..., "p95": ...}``), ``None`` if absent."""
    entry = (result or {}).get(name)
    if not isinstance(entry, dict) or entry.get(stat) is None:
        return None
    return float(entry[stat])


def judge(point: dict[str, Any], result: dict[str, Any] | None, returncode: int) -> str:
    """Decide whether a point's numbers can be trusted.

    Args:
        point: AIPerf options of the point.
        result: AIPerf's JSON export, ``None`` if it wrote none.
        returncode: AIPerf's exit code.

    Returns:
        ``""`` if the point is valid, otherwise the reason it is not.
    """
    if returncode != 0 or result is None:
        return f"client exited with code {returncode}" + ("" if result is not None else " without a result file")
    completed = metric(result, "completed_request_count") or 0
    if completed == 0:
        return "no request completed"
    error_rate = metric(result, "request_error_rate") or 0.0
    if error_rate > MAX_ERROR_RATE_PCT:
        return f"{error_rate:.1f} % of requests failed"
    if point.get("streaming") and metric(result, "time_to_first_token", "p95") is None:
        return "no time to first token measured"
    return ""


def summary_row(
    point: dict[str, Any], result: dict[str, Any] | None, fail_reason: str, slo: dict[str, float]
) -> dict[str, Any]:
    """One summary row: users, validity, SLO verdict, throughput and latency distributions.

    Args:
        point: AIPerf options of the point.
        result: AIPerf's JSON export, ``None`` if it wrote none.
        fail_reason: Output of :func:`judge`.
        slo: The preset's SLO, e.g. ``{"ttft_p95_ms": 3000, "tpot_p95_ms": 50}``.

    Returns:
        The row, with the same keys for every point. ``slo_met`` is ``None`` for an
        invalid point or a missing metric.
    """

    def rounded(name: str, stat: str = "avg", digits: int = 2) -> float | None:
        value = metric(result, name, stat)
        return None if value is None else round(value, digits)

    slo_values = {key: metric(result, *SLO_METRICS[key]) for key in slo}
    slo_met = None
    if not fail_reason and slo and all(v is not None for v in slo_values.values()):
        slo_met = all(slo_values[key] <= limit for key, limit in slo.items())  # type: ignore[operator]

    row: dict[str, Any] = {
        "Users": point.get("concurrency"),
        "ok": not fail_reason,
        "fail_reason": fail_reason,
        "slo_met": slo_met,
        "completed": rounded("completed_request_count", digits=0),
        "Error Rate(%)": rounded("request_error_rate"),
        "Request Throughput(req/s)": rounded("request_throughput", digits=3),
        "Goodput(req/s)": rounded("goodput", digits=3),
        "Output Throughput(tok/s)": rounded("output_token_throughput"),
        "Total Throughput(tok/s)": rounded("total_token_throughput"),
        "Mean ISL": rounded("input_sequence_length", digits=1),
        "Mean OSL": rounded("output_sequence_length", digits=1),
    }
    for label, name in (("TTFT", "time_to_first_token"), ("TPOT", "inter_token_latency"), ("E2EL", "request_latency")):
        for stat in LATENCY_STATS:
            row[f"{stat.title()} {label}(ms)"] = rounded(name, stat)
    return row


def max_load_within_slo(rows: list[dict[str, Any]], column: str = "Users") -> float | None:
    """Largest load (``column``) whose point is valid and meets the SLO, ``None`` if none does.

    Points above the first one that misses the SLO are ignored: a higher load
    passing again after a failure is noise, not capacity.
    """
    best = None
    for row in sorted((r for r in rows if r.get(column) is not None), key=lambda r: r[column]):
        if not row["ok"] or not row["slo_met"]:
            break
        best = row[column]
    return best


# Kept for callers that ask about users specifically.
def max_users_within_slo(rows: list[dict[str, Any]]) -> float | None:
    """Largest user count whose point is valid and meets the SLO, see :func:`max_load_within_slo`."""
    return max_load_within_slo(rows, "Users")


class AiperfClient(BenchmarkClient):
    """``aiperf profile``: service-scenario points such as multi-turn chat or a trace replay."""

    NAME = "aiperf"
    PACKAGE = "aiperf"
    VERSION = AIPERF_VERSION
    EXECUTABLE = "aiperf"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """See :class:`BenchmarkClient`; adds the trace loaded on first use and the slices replayed."""
        super().__init__(*args, **kwargs)
        self._trace_rows: list[dict[str, Any]] | None = None
        self._windows: dict[str, TraceWindow] = {}
        # Timestamp (ms) in a point's trace file where its measured requests start.
        self._measure_offsets: dict[str, int] = {}

    def command(self, point: dict[str, Any], point_dir: Path) -> list[str]:
        """``aiperf profile`` command line for one point.

        The preset's SLO is also passed as ``--goodput`` (per-request thresholds) unless
        the point sets ``goodput`` itself. With a preset ``trace`` the point's slice
        of it is written to ``point_dir`` first and replayed at its recorded
        (stretched) timestamps; requests before ``--fixed-schedule-start-offset``
        are the cache warm-up, sent by :meth:`warm_up`.
        """
        command = self._base_command(point, point_dir)
        if self.preset.slo and "goodput" not in point:
            command += ["--goodput", " ".join(f"{SLO_METRICS[k][0]}:{v}" for k, v in self.preset.slo.items())]
        if self.preset.trace:
            self._write_point_trace(point, point_dir)
            offset = self._measure_offsets[point_name(point)]
            if offset:
                command += ["--fixed-schedule-start-offset", str(offset)]  # also the schedule's time zero
            if "benchmark-duration" not in point:
                # Stop at the end of the slice; the grace period then bounds the wait for late
                # replies, so an overloaded point cannot run on indefinitely.
                duration = point.get("trace-duration", DEFAULT_TRACE_DURATION_S)
                command += ["--benchmark-duration", str(duration)]
        for key, value in point.items():
            if key not in TRACE_POINT_KEYS:
                command += option_args(key, value)
        return command

    def warm_up(self, point: dict[str, Any], point_dir: Path) -> None:
        """Send a trace point's warm-up requests and wait for them, so the measured run starts on a warm cache.

        It replays the start of the same trace file as the measured run: AIPerf derives
        each hash id's block from the file's content and the random seed, so only the
        same file gives the measured requests the blocks the warm-up cached.

        Raises:
            RuntimeError: If the warm-up run fails; the point would be measured cold.
        """
        offset = self._measure_offsets.get(point_name(point), 0)
        if not offset:
            return
        warmup_dir = point_dir / WARMUP_DIR
        warmup_dir.mkdir(exist_ok=True)
        command = self._base_command(point, warmup_dir, trace_file=point_dir / POINT_TRACE_FILE)
        command += ["--fixed-schedule-end-offset", str(offset - 1)]
        for key, value in point.items():
            if key not in TRACE_POINT_KEYS and key not in MEASURE_ONLY_KEYS:
                command += option_args(key, value)
        returncode = self.run_logged(command, warmup_dir, warmup_dir / CLIENT_LOG)
        if returncode != 0:
            raise RuntimeError(f"the cache warm-up exited with code {returncode}, see {warmup_dir / CLIENT_LOG}")

    def _base_command(self, point: dict[str, Any], artifact_dir: Path, trace_file: Path | None = None) -> list[str]:
        """``aiperf profile`` with the target, artifact directory and, for a trace, its fixed-schedule replay."""
        command = [str(self.env / "bin" / "aiperf") if self.env else "aiperf", "profile"]
        command += ["--model", self.model, "--url", f"http://{self.host}:{self.port}"]
        command += ["--artifact-dir", str(artifact_dir), "--ui-type", "none"]
        if "tokenizer" not in point:
            command += ["--tokenizer", self.tokenizer]
        if self.preset.trace:
            trace_file = trace_file or artifact_dir / POINT_TRACE_FILE
            command += ["--custom-dataset-type", "mooncake_trace", "--input-file", str(trace_file)]
            command += ["--fixed-schedule"]  # send at the trace timestamps
        return command

    def _write_point_trace(self, point: dict[str, Any], point_dir: Path) -> None:
        """Write the point's trace file: warm-up requests first, then its slice stretched to its replay speed."""
        trace = self.preset.trace
        if self._trace_rows is None:
            self._trace_rows = load_trace(fetch_trace(str(trace["source"]), str(trace["sha256"])))
        max_total = trace.get("max-total-tokens") or self.preset.server.get("max-model-len")
        max_total = int(max_total) if max_total else None
        start = float(point.get("trace-start", 0))
        rows, window = scale_trace(
            self._trace_rows,
            int(point["trace-speed-pct"]),
            float(point.get("trace-duration", DEFAULT_TRACE_DURATION_S)),
            max_total,
            start_seconds=start,
        )
        warmup = warmup_rows(self._trace_rows, start, float(point.get("trace-warmup", 0)), max_total)
        offset = warmup[-1]["timestamp"] + 1 if warmup else 0
        write_trace(warmup + [{**r, "timestamp": r["timestamp"] + offset} for r in rows], point_dir / POINT_TRACE_FILE)
        window.warmup_requests = len(warmup)
        self._windows[point_name(point)] = window
        self._measure_offsets[point_name(point)] = offset
        logger.info(f"Trace slice for {point_name(point)}: {window}")

    def read_result(self, point_dir: Path) -> dict[str, Any] | None:
        """AIPerf's JSON export of a point."""
        path = point_dir / RESULT_FILE
        return json.loads(path.read_text()) if path.exists() else None

    def clear_result(self, point_dir: Path) -> None:
        """Remove every export file of a previous run."""
        for path in point_dir.glob(f"{RESULT_FILE_PREFIX}*"):
            path.unlink()

    def judge(self, point: dict[str, Any], result: dict[str, Any] | None, returncode: int) -> str:
        """See :func:`judge`."""
        return judge(point, result, returncode)

    def summary_row(self, point: dict[str, Any], result: dict[str, Any] | None, fail_reason: str) -> dict[str, Any]:
        """See :func:`summary_row`; trace points also say which slice of the trace they replayed."""
        row = summary_row(point, result, fail_reason, self.preset.slo)
        if self.preset.trace:
            window = self._windows.get(point_name(point))
            row = {
                "Trace Speed(%)": point.get("trace-speed-pct"),
                "Offered(req/s)": window.offered_req_per_s if window else None,
                "Trace Requests": window.requests if window else None,
                "Dropped Too Long": window.dropped_too_long if window else None,
                "Warm-up Requests": window.warmup_requests if window else None,
                **row,
            }
        return row

    def passes(self, row: dict[str, Any]) -> bool:
        """A point passes when it is valid and meets the SLO."""
        return bool(row["ok"]) and row["slo_met"] is True

    def run_summary(self) -> dict[str, Any]:
        """``capacity``: the SLO, the highest load that met it and, after a search, how it was found.

        The load is ``max_users`` for chat sessions, or ``max_speed_pct`` (percent of the
        recorded speed) for a trace replay, which also records the trace it replayed.
        """
        if not self.preset.slo:
            return {}
        rows = self.total_results["results"]
        capacity: dict[str, Any] = {"slo": self.preset.slo}
        if self.preset.trace:
            capacity["max_speed_pct"] = max_load_within_slo(rows, "Trace Speed(%)")
            capacity["trace"] = dict(self.preset.trace)
        else:
            capacity["max_users"] = max_load_within_slo(rows, "Users")
        if self.search_outcome is not None:
            capacity["search"] = self.search_outcome.to_dict()
        return {"capacity": capacity}

    def concurrency(self, point: dict[str, Any]) -> int:
        """Users of the point (``concurrency``); 0 for a trace replay, which has no fixed concurrency."""
        return int(point.get("concurrency") or 0)
