"""Tests for the capacity search: the bisection, its configuration and the loop that drives it."""

from pathlib import Path
from typing import Any

import pytest

from furiosa_perf.clients.aiperf import AiperfClient
from furiosa_perf.clients.search import bisect_capacity, max_probes
from furiosa_perf.spec import Search, load_preset, parse_preset, parse_search, with_search_range


def do_test_search(edge: int, minimum: int = 8, maximum: int = 256, step: int = 4) -> tuple[Any, list[int]]:
    """Bisect a server that passes up to ``edge`` users; return the outcome and the probed values."""
    probed: list[int] = []

    def passes(users: int) -> bool:
        probed.append(users)
        return users <= edge

    return bisect_capacity(minimum, maximum, step, passes), probed


@pytest.mark.parametrize("edge", [8, 9, 37, 100, 199, 255])
def test_bisection_lands_within_one_step_of_the_edge(edge: int) -> None:
    """The reported capacity passed, and a value at most one step above it failed."""
    outcome, probed = do_test_search(edge)

    assert outcome.bound == "exact"
    assert outcome.capacity <= edge < outcome.capacity + outcome.step + 1
    assert len(probed) <= max_probes(8, 256, 4) == 8
    assert probed[:2] == [8, 256]  # both ends are measured first


def test_a_range_that_misses_the_edge_is_reported_not_searched() -> None:
    """min failing gives no capacity; max passing gives a lower bound; neither probes the inside."""
    below, below_probes = do_test_search(edge=4)
    above, above_probes = do_test_search(edge=1000)

    assert (below.capacity, below.bound, below_probes) == (None, "below_min", [8])
    assert (above.capacity, above.bound, above_probes) == (256, "at_least_max", [8, 256])


def test_outcome_records_every_probe() -> None:
    """run.json keeps the probes in order with their verdicts, so a search can be audited."""
    outcome, _ = do_test_search(edge=100, minimum=10, maximum=200, step=50)

    assert outcome.to_dict()["probes"] == [
        {"value": 10, "passed": True},
        {"value": 200, "passed": False},
        {"value": 105, "passed": False},
        {"value": 57, "passed": True},
    ]
    assert outcome.capacity == 57


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"key": "request-rate", "min": 1, "max": 2}, "search key 'request-rate' is unsupported"),
        ({"key": "concurrency", "min": 0, "max": 2}, "search min '0' is invalid"),
        ({"key": "concurrency", "min": 8, "max": 8}, "range 8..8 is empty"),
        ({"key": "concurrency", "min": 1, "max": 9, "low": 1}, "search keys \\['low'\\] are unknown"),
    ],
)
def test_parse_search_rejects_invalid_ranges(raw: dict[str, Any], message: str) -> None:
    """Only the concurrency key, positive integer bounds and a non-empty range are accepted."""
    with pytest.raises(ValueError, match=message):
        parse_search(raw)


def test_a_search_preset_needs_an_slo_and_one_base_point() -> None:
    """Without an SLO nothing decides pass/fail; the base point must leave the searched key to the search."""
    preset = {
        "kind": "scenario",
        "task": "chat",
        "client": "aiperf",
        "search": {"key": "concurrency", "min": 1, "max": 9},
    }

    with pytest.raises(ValueError, match="needs an 'slo'"):
        parse_preset(preset)
    with pytest.raises(ValueError, match="one point without 'concurrency'"):
        parse_preset({**preset, "slo": {"ttft_p95_ms": 1}, "matrix": [{"concurrency": [1, 2]}]})


def test_search_range_can_be_overridden_per_run() -> None:
    """--search-min/--search-max replace one or both ends; an empty range or a preset without search is refused."""
    preset = load_preset("chat-synthetic", "scenario")

    assert with_search_range(preset, 16, None).search == Search("concurrency", 16, 256, 4)
    assert with_search_range(preset, None, 96).search == Search("concurrency", 8, 96, 4)
    assert with_search_range(preset, None, None) is preset
    with pytest.raises(ValueError, match="range 300..256 is empty"):
        with_search_range(preset, 300, None)
    with pytest.raises(ValueError, match="need a preset with a 'search' block"):
        with_search_range(load_preset("text", "baseline"), 1, 2)


class FakeAiperf(AiperfClient):
    """AiperfClient whose points are not run but answered by an SLO edge at ``edge`` users."""

    edge = 40

    def _run_point(self, point: dict[str, Any], monitor: Any) -> dict[str, Any]:
        users = point["concurrency"]
        return {"Users": users, "ok": True, "fail_reason": "", "slo_met": users <= self.edge}


def test_the_loop_runs_the_search_and_reports_it_as_capacity(tmp_path: Path) -> None:
    """With a search preset the client measures the probed user counts and run.json gets the search."""
    client = FakeAiperf(load_preset("chat-synthetic", "scenario"), "m", "h", 1, tmp_path, {})
    client.run()

    capacity = client.run_summary()["capacity"]
    assert 36 <= capacity["max_users"] <= 40
    assert capacity["search"]["bound"] == "exact"
    assert [row["Users"] for row in client.total_results["results"]] == [
        probe["value"] for probe in capacity["search"]["probes"]
    ]


class CrashingAiperf(FakeAiperf):
    """AiperfClient whose tool fails on the first point, as with an invalid option."""

    def _run_point(self, point: dict[str, Any], monitor: Any) -> dict[str, Any]:
        return {"Users": point["concurrency"], "ok": False, "fail_reason": "client exited with code 1", "slo_met": None}


def test_a_client_failure_stops_the_search_instead_of_counting_as_a_miss(tmp_path: Path) -> None:
    """A crashed tool says nothing about capacity, so no "below_min" verdict is drawn from it."""
    client = CrashingAiperf(load_preset("chat-synthetic", "scenario"), "m", "h", 1, tmp_path, {})

    with pytest.raises(RuntimeError, match="benchmark client failed at point u8"):
        client.run()
    assert client.search_outcome is None
