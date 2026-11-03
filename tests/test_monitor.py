"""Tests for the per-point summary cut out of ``monitor.csv``."""

import pandas as pd
import pytest

from furiosa_perf.runner.monitor import MONITOR_COLUMNS, SUMMARY_KEYS, summarize_window

T0 = pd.Timestamp("2026-01-01T00:00:00+00:00")


def iso(second: float) -> str:
    """ISO timestamp ``second`` seconds after T0."""
    return (T0 + pd.Timedelta(seconds=second)).isoformat()


def scenario_frame(num_active: int = 12, with_server_metrics: bool = True) -> pd.DataFrame:
    """A 1 Hz monitor log shaped like a real RNGD point.

    * 0-2 s: client start-up, nothing running, 40 W idle;
    * 3 s on: ``num_active`` rows with 4 requests running; power ramps 100 -> 200 W and
      then holds 300 W (RNGD power lags the load); 10 prompt and 100 generated tokens/s;
    * after that: nothing running while power decays 250 -> 50 W, then idle; the client
      runs at 30 % CPU throughout except for a 600 % import burst at start-up.
    """
    running = [0] * 3 + [4] * num_active + [0] * 10
    ramp = [100.0, 200.0] + [300.0] * (num_active - 2)
    power = [40.0] * 3 + ramp + [250.0, 200.0, 150.0, 100.0, 50.0] + [40.0] * 5
    rows = []
    prompt = generated = 0.0
    for second, (r, p) in enumerate(zip(running, power, strict=True)):
        if r:
            prompt, generated = prompt + 10.0, generated + 100.0
        row = dict.fromkeys(MONITOR_COLUMNS)
        row.update(
            timestamp=iso(second),
            power_w=p,
            peak_temp_c=40.0 + p / 100.0,
            avg_util_pct=80.0 if r else 0.0,
            throttle=0,
            client_cpu_pct=600.0 if second == 1 else 30.0,
        )
        if with_server_metrics:
            row.update(
                num_requests_running=r,
                num_requests_waiting=0,
                kv_cache_usage_pct=10.0 if r else 0.0,
                prompt_tokens_total=prompt,
                generation_tokens_total=generated,
                prefix_cache_hits=0.0,
                prefix_cache_queries=prompt,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, end_second: float, max_concurrency: int = 4) -> dict:
    """Summarize a point that started at T0 and whose client exited at ``end_second``."""
    return summarize_window(df, iso(0), iso(end_second), max_concurrency)


def test_active_window_excludes_client_startup_and_power_decay() -> None:
    """Active = the 12 running rows (12 s); the idle start-up and the decay after the load are excluded.

    Energy counts each row's power over the second before it: 100 + 200 + 10 x 300 = 3300 J.
    """
    s = summarize(scenario_frame(), end_second=20)

    assert s["active_s"] == 12.0
    assert s["active_samples"] == 12
    assert s["energy_j"] == pytest.approx(3300.0)
    assert s["mean_power"] == pytest.approx(3300.0 / 12)
    assert s["window_ok"] is True


def test_rates_and_efficiency_share_the_active_window() -> None:
    """Token rates and tok/J share the active window, so generated tok/s divided by mean power equals tok/J."""
    s = summarize(scenario_frame(), end_second=20)

    assert s["server_gen_tps"] == pytest.approx(100.0)
    assert s["server_prompt_tps"] == pytest.approx(10.0)
    assert s["gen_tok_per_j"] == pytest.approx(1200.0 / 3300.0, abs=1e-4)
    assert s["gen_tok_per_j"] == pytest.approx(s["server_gen_tps"] / s["mean_power"], abs=1e-4)
    assert s["total_tok_per_j"] == pytest.approx(1320.0 / 3300.0, abs=1e-4)


def test_steady_state_and_peaks() -> None:
    """Steady power averages the rows at full concurrency; peaks and flags come from the active rows."""
    s = summarize(scenario_frame(), end_second=20)

    assert s["steady_rows"] == 12
    assert s["power_steady_w"] == pytest.approx((100.0 + 200.0 + 300.0 * 10) / 12, abs=0.01)
    assert s["power_p95_w"] == 300.0
    assert s["kv_cache_peak_pct"] == 10.0
    assert s["throttled"] is False
    assert s["client_cpu_peak_pct"] == 30.0
    assert s["prefix_hit_rate"] == 0.0


def test_short_scenario_is_flagged() -> None:
    """Fewer active samples than MIN_ACTIVE_SAMPLES keep the numbers but set window_ok to False."""
    s = summarize(scenario_frame(num_active=4), end_second=7)

    assert s["active_samples"] == 4
    assert s["window_ok"] is False
    assert s["energy_j"] is not None


def test_without_server_metrics_the_whole_window_is_used_and_flagged() -> None:
    """With no /metrics columns, the active window falls back to [start, end] and window_ok is False."""
    s = summarize(scenario_frame(with_server_metrics=False), end_second=15)

    assert s["active_samples"] == 16
    assert s["window_ok"] is False
    assert s["gen_tok_per_j"] is None
    # The fallback window includes the client's start-up burst, so its CPU is not reported.
    assert s["client_cpu_peak_pct"] is None


def test_empty_inputs_keep_every_key() -> None:
    """Summary files need a stable header, so empty input still returns every key."""
    for df in (pd.DataFrame(), scenario_frame()):
        s = summarize_window(df, iso(3600), iso(3605), 4)
        assert list(s) == SUMMARY_KEYS
        assert s["monitor_rows"] == 0
        assert s["window_ok"] is False
