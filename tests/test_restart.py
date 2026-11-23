"""Tests for restarting the server before every point (restart-server: per-point)."""

from pathlib import Path
from typing import Any

import pytest

from furiosa_perf.clients.base import BenchmarkClient
from furiosa_perf.results.store import RECORD_FILE, write_json
from furiosa_perf.runner.runner import BenchmarkRunner
from furiosa_perf.spec import load_preset, parse_preset


class FakeServer:
    """Stands in for APIServerManager: counts restarts and hands out a new PID each time."""

    def __init__(self) -> None:
        self.restarts = 0
        self.server_pid = 100

    def restart(self) -> str:
        self.restarts += 1
        self.server_pid += 1
        return "serve"


class FakeMonitor:
    """Stands in for HardwareMonitor: records the PIDs it was restarted with."""

    def __init__(self) -> None:
        self.pids: list[int] = []

    def restart(self, server_pid: int) -> None:
        self.pids.append(server_pid)


def test_restart_hook_keeps_the_fresh_server_for_the_first_point() -> None:
    """The run's own server serves the first point; every later point gets a new one and the monitor follows it."""
    server, monitor = FakeServer(), FakeMonitor()
    hook = BenchmarkRunner._restart_hook(server, monitor)  # type: ignore[arg-type]

    for _ in range(3):
        hook()

    assert server.restarts == 2
    assert monitor.pids == [101, 102]


class CountingClient(BenchmarkClient):
    """A client whose points are not run; it only records which points were measured."""

    NAME = PACKAGE = EXECUTABLE = "fake"
    VERSION = "0"

    def _run_point(self, point: dict[str, Any], monitor: Any) -> dict[str, Any]:
        return {"ok": True, "fail_reason": "", "Concurrent": point["max-concurrency"]}


def test_before_point_runs_only_for_measured_points(tmp_path: Path) -> None:
    """A point kept on --resume is not re-measured, so it costs no restart either."""
    preset = load_preset("text", "baseline")
    preset = type(preset)(**{**preset.__dict__, "points": preset.points[:3]})
    kept = preset.points[1]
    (tmp_path / "i128-o128-c4").mkdir()
    write_json(tmp_path / "i128-o128-c4" / RECORD_FILE, {"point": kept, "summary": {"ok": True, "fail_reason": ""}})
    calls: list[int] = []

    client = CountingClient(preset, "m", "h", 1, tmp_path, {}, resume=True)
    client.run(before_point=lambda: calls.append(1))

    assert len(calls) == 2
    assert len(client.total_results["results"]) == 3


def test_restart_policy_is_validated() -> None:
    """Only "never" and "per-point" are accepted; the chat preset restarts per point."""
    base = {"kind": "baseline", "task": "t", "client": "vllm-bench"}

    assert parse_preset(base).restart_server == "never"
    assert load_preset("chat-synthetic", "scenario").restart_server == "per-point"
    with pytest.raises(ValueError, match="restart-server value 'always' is unsupported"):
        parse_preset({**base, "restart-server": "always"})
