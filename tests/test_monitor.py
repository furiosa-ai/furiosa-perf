"""Tests for the hardware monitoring loop."""

import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from furiosa_perf.runner import monitor as monitor_module
from furiosa_perf.runner.monitor import HardwareMonitor
from furiosa_perf.runner.runner import BenchmarkRunner


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` that replays canned ``nvidia-smi`` outputs."""

    outputs: list[bytes] = []
    stop_event: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._output = _FakePopen.outputs.pop(0)
        if not _FakePopen.outputs:
            _FakePopen.stop_event.set()

    def communicate(self) -> tuple[bytes, None]:
        """Return the canned output."""
        return self._output, None


def test_gpu_monitor_survives_a_bad_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One unparsable sample must not end monitoring for the rest of the run (issue #5)."""
    stop_event = multiprocessing.Event()
    _FakePopen.stop_event = stop_event
    _FakePopen.outputs = [
        b"GPU,50,100.0,60\n",
        b"GPU,50,[N/A],60\n",
        b"GPU,70,120.0,65\n",
    ]
    monkeypatch.setattr(monitor_module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(HardwareMonitor, "_fetch_metrics_text", staticmethod(lambda host, port: None))
    monkeypatch.setattr(monitor_module.time, "sleep", lambda _: None)

    csv_path = tmp_path / "GPU_1_monitoring_log.csv"
    HardwareMonitor._monitor_gpu("localhost", 0, -1, csv_path, stop_event)

    rows = csv_path.read_text().splitlines()[1:]
    assert [row.split(",")[2] for row in rows] == ["100.0", "120.0"]


def _exit_with_error() -> None:
    raise SystemExit(1)


def test_stop_monitor_reports_a_monitor_that_died_early(caplog: pytest.LogCaptureFixture) -> None:
    """A monitor that exited before the run ended must be reported, not stopped silently."""
    proc = multiprocessing.Process(target=_exit_with_error)
    proc.start()
    proc.join()

    BenchmarkRunner._stop_monitor(None, proc, multiprocessing.Event())  # type: ignore[arg-type]

    assert "exited early" in caplog.text
