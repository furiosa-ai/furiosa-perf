"""Tests for replaying a request trace: download check, slicing and stretching, and the AIPerf command."""

import hashlib
from pathlib import Path

import pytest

from furiosa_perf.clients.aiperf import AiperfClient, max_load_within_slo
from furiosa_perf.clients.traces import fetch_trace, load_trace, scale_trace, warmup_rows
from furiosa_perf.spec import load_preset

ROWS = [
    {"timestamp": 0, "input_length": 6000, "output_length": 100, "hash_ids": [0, 1]},
    {"timestamp": 500, "input_length": 50000, "output_length": 100, "hash_ids": [0, 2]},
    {"timestamp": 1000, "input_length": 3000, "output_length": 50, "hash_ids": [0, 3]},
    {"timestamp": 4000, "input_length": 3000, "output_length": 50, "hash_ids": [0, 4]},
]


def test_scale_trace_cuts_the_slice_that_fits_and_stretches_it() -> None:
    """At 50 % for 2 s of wall time a point replays 1 s of trace, twice as slowly, minus over-long requests."""
    rows, window = scale_trace(ROWS, speed_pct=50, wall_seconds=2, max_total_tokens=40960)

    assert [r["timestamp"] for r in rows] == [0, 2000]  # the 50k-token request is dropped, 4 s is past the slice
    assert [r["hash_ids"] for r in rows] == [[0, 1], [0, 3]]  # prefix structure is kept as recorded
    assert (window.trace_seconds, window.requests, window.dropped_too_long) == (1.0, 2, 1)
    assert window.offered_req_per_s == 1.0


def test_scale_trace_at_full_speed_keeps_the_recorded_times() -> None:
    """100 % replays the recorded timestamps unchanged."""
    rows, _ = scale_trace(ROWS, speed_pct=100, wall_seconds=5, max_total_tokens=None)

    assert [r["timestamp"] for r in rows] == [0, 500, 1000, 4000]


def test_a_slice_can_start_mid_trace_after_its_warm_up() -> None:
    """From 1 s in, the slice is rebased to its start; the second before it is the warm-up, one output token each."""
    rows, window = scale_trace(ROWS, speed_pct=100, wall_seconds=5, max_total_tokens=40960, start_seconds=1)
    warmup = warmup_rows(ROWS, start_seconds=1, warmup_seconds=1, max_total_tokens=40960)

    assert [r["timestamp"] for r in rows] == [0, 3000]
    assert window.start_seconds == 1
    assert [(r["timestamp"], r["output_length"]) for r in warmup] == [(0, 1)]  # 50k-token request left out
    assert warmup[0]["hash_ids"] == [0, 1]


def test_fetch_trace_refuses_a_file_with_another_hash(tmp_path: Path) -> None:
    """A changed trace file never silently changes results: its SHA-256 must match the preset."""
    path = tmp_path / "trace.jsonl"
    path.write_text('{"timestamp": 0, "input_length": 1, "output_length": 1}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert fetch_trace(str(path), digest) == path
    assert load_trace(path)[0]["input_length"] == 1
    with pytest.raises(ValueError, match="the preset expects"):
        fetch_trace(str(path), "0" * 64)


def test_trace_points_replay_a_written_slice_at_fixed_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command replays the point's trace file on a fixed schedule bounded by the slice length."""
    preset = load_preset("chat-mooncake", "scenario")
    client = AiperfClient(preset, "m", "h", 1, tmp_path, {}, tokenizer="t")
    client.env = Path("/env")
    client._trace_rows = ROWS
    point = {**preset.points[0], "trace-speed-pct": 50, "trace-duration": 2, "trace-start": 0, "trace-warmup": 0}
    command = client.command(point, tmp_path)

    assert command[command.index("--custom-dataset-type") + 1] == "mooncake_trace"
    assert command[command.index("--input-file") + 1] == str(tmp_path / "trace.jsonl")
    assert "--fixed-schedule" in command
    assert command[command.index("--benchmark-duration") + 1] == "2"
    assert not [arg for arg in command if arg.startswith("--trace-")]
    assert "--fixed-schedule-start-offset" not in command  # no warm-up, nothing to skip
    assert len(load_trace(tmp_path / "trace.jsonl")) == 2


def test_capacity_of_a_trace_replay_is_a_speed() -> None:
    """For a trace the highest passing speed, not a user count, is the capacity."""
    rows = [
        {"Trace Speed(%)": 10, "ok": True, "slo_met": True},
        {"Trace Speed(%)": 40, "ok": True, "slo_met": False},
        {"Trace Speed(%)": 25, "ok": True, "slo_met": True},
    ]

    assert max_load_within_slo(rows, "Trace Speed(%)") == 25


def test_a_warmed_point_replays_one_file_in_two_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm-up and measured requests share one file (same hash-id blocks); offsets split them."""
    preset = load_preset("chat-mooncake", "scenario")
    client = AiperfClient(preset, "m", "h", 1, tmp_path, {}, tokenizer="t")
    client.env = Path("/env")
    client._trace_rows = ROWS
    point = {**preset.points[0], "trace-speed-pct": 100, "trace-duration": 5, "trace-start": 1, "trace-warmup": 1}
    command = client.command(point, tmp_path)
    runs: list[list[str]] = []
    monkeypatch.setattr(client, "run_logged", lambda c, cwd, log: runs.append(c) or 0)
    client.warm_up(point, tmp_path)

    written = load_trace(tmp_path / "trace.jsonl")
    assert [(r["timestamp"], r["output_length"]) for r in written] == [(0, 1), (1, 50), (3001, 50)]
    assert command[command.index("--fixed-schedule-start-offset") + 1] == "1"
    warmup = runs[0]
    assert warmup[warmup.index("--fixed-schedule-end-offset") + 1] == "0"
    assert warmup[warmup.index("--input-file") + 1] == str(tmp_path / "trace.jsonl")
    assert warmup[warmup.index("--artifact-dir") + 1] == str(tmp_path / "warmup")
    assert "--benchmark-grace-period" not in warmup and "--goodput" not in warmup
    assert warmup[warmup.index("--random-seed") + 1] == command[command.index("--random-seed") + 1]
