"""Hardware and server monitoring for benchmark runs.

A background process samples once per second, for the whole run:

* the accelerators the server was given (:mod:`furiosa_perf.runner.devices`), one row
  per device in ``devices.csv`` plus their sum/mean in ``monitor.csv``;
* the server's ``/metrics`` gauges and counters
  (:mod:`furiosa_perf.runner.server_metrics`);
* CPU and memory of the server, the benchmark client and the host.

Around every point the main process also appends the full ``/metrics`` text and
the device counters to ``metrics_snapshots.jsonl``. Per-point numbers are then
cut out of the CSV by timestamp (:func:`summarize_window`) and from the snapshot pair
(:func:`~furiosa_perf.runner.server_metrics.summarize_snapshots`).

Every probe is guarded so that one failure never stops the monitor, and
:meth:`HardwareMonitor.begin` restarts the process if it died anyway.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import requests

from furiosa_perf.log import logger, setup_logger
from furiosa_perf.runner import server_metrics
from furiosa_perf.runner.devices import (
    DEVICE_COLUMNS,
    DeviceSample,
    GpuSampler,
    NpuSampler,
    ProcessUsage,
    make_sampler,
    npu_busy_pct,
)

MONITOR_CSV = "monitor.csv"
DEVICES_CSV = "devices.csv"
SNAPSHOTS_JSONL = "metrics_snapshots.jsonl"

SAMPLE_INTERVAL_S = 1.0
METRICS_TIMEOUT_S = 2
# Fewer active samples than this and the point is flagged window_ok = False: RNGD
# power readings take ~3.5 s to follow a load change, and 1 Hz sampling cannot time a
# short point precisely, so power, energy and busy figures of such a window are rough.
MIN_ACTIVE_SAMPLES = 10
# Rows with at least this fraction of max_concurrency running count as steady state.
STEADY_RATIO = 0.9
MIN_STEADY_ROWS = 3
# A client process above this CPU share (100 = one core) is likely the bottleneck.
CLIENT_CPU_WARN_PCT = 95.0

HOST_COLUMNS = ["server_cpu_pct", "server_rss_gib", "client_cpu_pct", "host_cpu_pct", "host_mem_used_gib"]
MONITOR_COLUMNS = [
    "timestamp",
    "power_w",  # sum over the configured devices
    "peak_temp_c",
    "avg_util_pct",  # mean over the configured devices
    "throttle",  # OR of the devices' slowdown masks, idle bit removed
    *server_metrics.SAMPLE_COLUMNS,
    *HOST_COLUMNS,
]


@dataclass(frozen=True)
class MonitorConfig:
    """What to monitor and where to write it.

    Attributes:
        host: API server host exposing ``/metrics``.
        port: API server port.
        server_pid: PID of the server process, ``-1`` to skip process tracking.
        device_name: ``"RNGD"`` for furiosa-smi sampling, otherwise a GPU name.
        device_indices: Devices the server was launched on; empty means all.
        log_dir: Directory receiving the CSV and snapshot files.
    """

    host: str
    port: int
    server_pid: int
    device_name: str
    device_indices: list[int]
    log_dir: Path

    @property
    def metrics_url(self) -> str:
        """URL of the server's Prometheus endpoint."""
        return f"http://{self.host}:{self.port}/metrics"

    @property
    def metrics_prefix(self) -> str:
        """Metric name prefix of the serving framework on this device."""
        return server_metrics.FURIOSA_PREFIX if self.device_name == "RNGD" else server_metrics.VLLM_PREFIX


def _now() -> str:
    """Current UTC time in ISO 8601, the timestamp format of every monitor file."""
    return datetime.now(timezone.utc).isoformat()


def aggregate_devices(samples: list[DeviceSample]) -> dict[str, float | int | None]:
    """Combine per-device samples into the device columns of ``monitor.csv``.

    Args:
        samples: One sample per configured device.

    Returns:
        ``power_w`` (sum), ``peak_temp_c`` (max), ``avg_util_pct`` (mean) and
        ``throttle`` (bitwise OR); ``None`` where no device reported the value.
    """

    def present(name: str) -> list[Any]:
        return [v for s in samples if (v := getattr(s, name)) is not None]

    power, temps, utils, throttles = present("power_w"), present("temp_c"), present("util_pct"), present("throttle")
    mask = 0
    for t in throttles:
        mask |= t
    return {
        "power_w": round(sum(power), 2) if power else None,
        "peak_temp_c": round(max(temps), 2) if temps else None,
        "avg_util_pct": round(sum(utils) / len(utils), 2) if utils else None,
        "throttle": mask if throttles else None,
    }


def _monitor_loop(config: MonitorConfig) -> None:
    """Body of the monitor process. Never raises out of the loop; stopped by SIGTERM.

    The process is spawned, not forked: the parent has already initialized
    furiosa-smi (a Rust library with its own threads) while detecting the system, and
    a forked child that initializes it again deadlocks.

    No multiprocessing lock or event is shared with the parent on purpose: a child
    killed (e.g. by the OOM killer) while holding a lock would deadlock the parent.
    Every row is flushed, so terminating at any point loses nothing.
    """
    setup_logger("INFO")  # a spawned process starts with an unconfigured logger
    warned: set[str] = set()

    def warn_once(key: str, message: str) -> None:
        if key not in warned:
            warned.add(key)
            logger.warning(f"[monitor] {message}")

    sampler: NpuSampler | GpuSampler | None = None
    try:
        sampler = make_sampler(config.device_name, config.device_indices)
    except Exception as e:
        warn_once("sampler", f"device sampling is disabled; cause: {e}")
    processes = ProcessUsage(config.server_pid)

    config.log_dir.mkdir(parents=True, exist_ok=True)
    with (
        (config.log_dir / MONITOR_CSV).open("a", newline="") as monitor_file,
        (config.log_dir / DEVICES_CSV).open("a", newline="") as devices_file,
    ):
        monitor_writer = _csv_writer(monitor_file, MONITOR_COLUMNS)
        devices_writer = _csv_writer(devices_file, ["timestamp", *DEVICE_COLUMNS])
        while True:
            started = time.perf_counter()
            timestamp = _now()
            row: dict[str, Any] = {"timestamp": timestamp}

            devices: list[DeviceSample] = []
            if sampler is not None:
                try:
                    devices = sampler.sample()
                except Exception as e:
                    warn_once("device", f"failed to sample devices, continuing; cause: {e}")
            row.update(aggregate_devices(devices))

            try:
                response = requests.get(config.metrics_url, timeout=METRICS_TIMEOUT_S)
                response.raise_for_status()
                row.update(server_metrics.sample_columns(server_metrics.parse(response.text, config.metrics_prefix)))
            except requests.RequestException as e:
                warn_once("metrics", f"cannot read {config.metrics_url}, continuing; cause: {e}")

            try:
                row.update(processes.sample())
            except Exception as e:
                warn_once("host", f"failed to sample host usage, continuing; cause: {e}")

            monitor_writer.writerow(row)
            monitor_file.flush()
            for device in devices:
                devices_writer.writerow({"timestamp": timestamp, **asdict(device)})
            devices_file.flush()

            remaining = SAMPLE_INTERVAL_S - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


def _csv_writer(file: TextIO, columns: list[str]) -> csv.DictWriter[str]:
    """Return a ``DictWriter`` on ``file`` (opened for appending), writing the header if it is empty."""
    writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
    if file.tell() == 0:
        writer.writeheader()
    return writer


def _rows_between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows of ``df`` whose timestamp lies in ``[start, end]``."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return df[(ts >= start) & (ts <= end)]


def _number(value: Any) -> float | None:
    """``value`` as a float, ``None`` for missing or NaN."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


# Keys of summarize_window(), in the order they appear in summary files.
SUMMARY_KEYS = [
    "active_s",
    "monitor_rows",
    "active_samples",
    "steady_rows",
    "window_ok",
    "mean_power",
    "energy_j",
    "power_steady_w",
    "power_p95_w",
    "gen_tok_per_j",
    "total_tok_per_j",
    "avg_util_pct",
    "peak_temp_c",
    "throttled",
    "kv_cache_peak_pct",
    "waiting_peak",
    "preemptions",
    "prefix_hit_rate",
    "server_prompt_tps",
    "server_gen_tps",
    "client_cpu_peak_pct",
]


def summarize_window(df: pd.DataFrame, start: str, end: str, max_concurrency: int) -> dict[str, Any]:
    """Summarize the monitor rows of one point.

    Every rate and efficiency uses one *active* window so they agree with each other:
    it runs from the first to the last row with a running request between ``start``
    and ``end``, which excludes client start-up and teardown (the accelerator is idle
    then). Without server metrics it falls back to ``[start, end]`` and ``window_ok``
    is ``False``.

    ``mean_power`` is ``energy_j / active_s``, so ``gen_tok_per_j`` equals generated
    tokens per second divided by ``mean_power``.

    Args:
        df: ``monitor.csv`` covering ``[start, end]``.
        start: ISO timestamp taken just before the client started.
        end: ISO timestamp taken just after the client exited.
        max_concurrency: Point concurrency, for the steady-state cut.

    Returns:
        A dict with the same keys for every point; values that cannot be
        computed are ``None``.
    """
    summary: dict[str, Any] = dict.fromkeys(SUMMARY_KEYS)
    summary.update(monitor_rows=0, active_samples=0, steady_rows=0, window_ok=False)
    if df.empty:
        return summary

    start_ts, end_ts = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    window = _rows_between(df, start_ts, end_ts)
    summary["monitor_rows"] = len(window)
    if window.empty:
        return summary
    # Missing or unparsable cells become NaN so every column below is numeric.
    window = window.assign(**{c: pd.to_numeric(window[c], errors="coerce") for c in window.columns if c != "timestamp"})

    ts = pd.to_datetime(window["timestamp"], utc=True)
    active = window[window["num_requests_running"].fillna(0) > 0]
    from_server = not active.empty
    if not from_server:
        active = window
    active_ts = pd.to_datetime(active["timestamp"], utc=True)
    period_s = float(ts.diff().dt.total_seconds().median()) if len(ts) > 1 else SAMPLE_INTERVAL_S
    active_s = float((active_ts.iloc[-1] - active_ts.iloc[0]).total_seconds()) + period_s

    # Each sample stands for the period before it; the first one for one median period.
    dt = active_ts.diff().dt.total_seconds().fillna(period_s)
    energy_j = float((active["power_w"].fillna(0.0) * dt).sum())

    # Counter deltas run from the last row before the active window to its last row.
    before = window[ts < active_ts.iloc[0]]
    baseline = before.iloc[-1] if not before.empty else active.iloc[0]
    last = active.iloc[-1]

    def delta(column: str) -> float | None:
        # None when the server does not expose the counter. A server restart inside the
        # window resets counters; clamp to 0 instead of going negative.
        after, before = _number(last.get(column)), _number(baseline.get(column))
        if after is None or before is None:
            return None
        return max(0.0, after - before)

    def per(numerator: float | None, denominator: float, digits: int) -> float | None:
        return round(numerator / denominator, digits) if numerator is not None and denominator > 0 else None

    prompt, generated = delta("prompt_tokens_total"), delta("generation_tokens_total")
    hits, queries = delta("prefix_cache_hits"), delta("prefix_cache_queries")
    total = prompt + generated if prompt is not None and generated is not None else None

    steady = active
    if max_concurrency > 0:
        steady = active[active["num_requests_running"].fillna(0) >= STEADY_RATIO * max_concurrency]
    has_power = active["power_w"].notna().any()
    throttles = active["throttle"].dropna()

    summary.update(
        active_s=round(active_s, 1),
        active_samples=len(active),
        steady_rows=len(steady),
        window_ok=bool(from_server and len(active) >= MIN_ACTIVE_SAMPLES),
        mean_power=round(energy_j / active_s, 2) if has_power else None,
        energy_j=round(energy_j, 1) if has_power else None,
        power_steady_w=round(float(steady["power_w"].mean()), 2) if len(steady) >= MIN_STEADY_ROWS else None,
        power_p95_w=_number(round(active["power_w"].quantile(0.95), 2)) if has_power else None,
        gen_tok_per_j=per(generated, energy_j, 4),
        total_tok_per_j=per(total, energy_j, 4),
        avg_util_pct=_number(round(active["avg_util_pct"].mean(), 2)),
        peak_temp_c=_number(active["peak_temp_c"].max()),
        throttled=bool((throttles.astype(int) != 0).any()) if not throttles.empty else None,
        kv_cache_peak_pct=_number(round(active["kv_cache_usage_pct"].max(), 2)),
        waiting_peak=_number(active["num_requests_waiting"].max()),
        preemptions=delta("num_preemptions_total"),
        prefix_hit_rate=per(hits, queries or 0.0, 4),
        server_prompt_tps=per(prompt, active_s, 1),
        server_gen_tps=per(generated, active_s, 1),
        # Over the active window only, and only when the server located it: importing
        # the client's libraries at start-up briefly takes several cores and says
        # nothing about load generation.
        client_cpu_peak_pct=_number(active["client_cpu_pct"].max()) if from_server else None,
    )
    return summary


@dataclass
class PointWindow:
    """State captured by :meth:`HardwareMonitor.begin` for one point.

    Attributes:
        name: Point name, also used as the snapshot key.
        start: ISO timestamp taken just before the client started.
        metrics: ``/metrics`` text at the start (empty if unreachable).
        counters: Device counters at the start (empty for GPUs or on failure).
    """

    name: str
    start: str
    metrics: str = ""
    counters: dict[str, Any] = field(default_factory=dict)


class HardwareMonitor:
    """Owns the monitor process and turns its output into per-point summaries."""

    def __init__(self, config: MonitorConfig) -> None:
        """Store the configuration; call :meth:`start` to begin sampling.

        Args:
            config: What to monitor and where to write it.
        """
        self.config = config
        self._process: BaseProcess | None = None
        self._counter_source: NpuSampler | None = None

    def start(self) -> None:
        """Start the monitor process (no-op while it is alive)."""
        if self._process is not None and self._process.is_alive():
            return
        context = multiprocessing.get_context("spawn")  # see _monitor_loop for why not fork
        self._process = context.Process(target=_monitor_loop, args=(self.config,), daemon=True)
        self._process.start()
        logger.info(f"monitor started (pid {self._process.pid}) -> {self.config.log_dir}")

    def stop(self) -> None:
        """Stop the monitor process: SIGTERM, then SIGKILL if it does not exit."""
        if self._process is None:
            return
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2)
        self._process = None

    def restart(self, server_pid: int) -> None:
        """Restart sampling for a newly launched server process.

        Args:
            server_pid: PID of the new server, whose process tree is tracked from now on.
        """
        self.stop()
        self.config = replace(self.config, server_pid=server_pid)
        self.start()

    def begin(self, name: str) -> PointWindow:
        """Mark the start of a point; call right before launching the client.

        Restarts the monitor process if it died, so the point still gets data.

        Args:
            name: Point name.

        Returns:
            The window to pass to :meth:`end`.
        """
        if self._process is not None and not self._process.is_alive():
            logger.error(f"monitor process exited with code {self._process.exitcode}; restarting it")
            self.start()
        window = PointWindow(name=name, start=_now())
        window.metrics, window.counters = self._snapshot(name, "start")
        return window

    def end(self, window: PointWindow, max_concurrency: int, slice_dir: Path | None = None) -> dict[str, Any]:
        """Mark the end of a point and summarize it; call right after the client exits.

        Waits one sample interval so the last rows of the point are in the CSV.

        Args:
            window: Value returned by :meth:`begin`.
            max_concurrency: Point concurrency, for the steady-state cut.
            slice_dir: If given, the point's rows of ``monitor.csv`` and
                ``devices.csv`` are also written there.

        Returns:
            :func:`summarize_window` fields, server-side mean latencies and finish
            reasons, and ``npu_busy_pct`` on RNGD (``None`` unless ``window_ok``).
        """
        end = _now()
        end_metrics, end_counters = self._snapshot(window.name, "end")
        time.sleep(SAMPLE_INTERVAL_S)

        monitor_rows = _read_csv(self.config.log_dir / MONITOR_CSV)
        summary = summarize_window(monitor_rows, window.start, end, max_concurrency)
        summary.update(server_metrics.summarize_snapshots(window.metrics, end_metrics, self.config.metrics_prefix))
        summary["npu_busy_pct"] = (
            npu_busy_pct(window.counters, end_counters, summary["active_s"]) if summary["window_ok"] else None
        )

        if summary["monitor_rows"] == 0:
            logger.warning(f"monitor produced no rows between {window.start} and {end}")
        if summary["window_ok"] and (summary["client_cpu_peak_pct"] or 0.0) >= CLIENT_CPU_WARN_PCT:
            logger.warning(
                f"the benchmark client used {summary['client_cpu_peak_pct']}% CPU in point {window.name}; "
                "throughput may be limited by the client, not the server"
            )
        if slice_dir is not None:
            self._write_slice(window.start, end, slice_dir, monitor_rows)
        return summary

    def _snapshot(self, name: str, phase: str) -> tuple[str, dict[str, Any]]:
        """Append the full ``/metrics`` text and device counters to the snapshot log."""
        metrics = ""
        try:
            response = requests.get(self.config.metrics_url, timeout=METRICS_TIMEOUT_S)
            response.raise_for_status()
            metrics = response.text
        except requests.RequestException as e:
            logger.warning(f"cannot read {self.config.metrics_url} for the {phase} snapshot of {name}; cause: {e}")

        counters: dict[str, Any] = {}
        if self.config.device_name == "RNGD":
            try:
                if self._counter_source is None:
                    self._counter_source = NpuSampler(self.config.device_indices)
                counters = self._counter_source.counters()
            except Exception as e:
                logger.warning(f"failed to read NPU counters for the {phase} snapshot of {name}; cause: {e}")

        record = {"point": name, "phase": phase, "timestamp": _now(), "devices": counters, "metrics": metrics}
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.config.log_dir / SNAPSHOTS_JSONL).open("a") as f:
            f.write(json.dumps(record) + "\n")
        return metrics, counters

    def _write_slice(self, start: str, end: str, slice_dir: Path, monitor_rows: pd.DataFrame) -> None:
        """Copy the point's rows next to its results."""
        start_ts, end_ts = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        slice_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in ((MONITOR_CSV, monitor_rows), (DEVICES_CSV, _read_csv(self.config.log_dir / DEVICES_CSV))):
            if not rows.empty:
                _rows_between(rows, start_ts, end_ts).to_csv(slice_dir / name, index=False)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a monitor CSV, returning an empty frame if it is missing or empty."""
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError) as e:
        logger.warning(f"cannot read {path}; cause: {e}")
        return pd.DataFrame()
