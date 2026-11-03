"""Tests for the device-side helpers that need no hardware."""

import pytest

from furiosa_perf.runner.devices import DeviceSample, npu_busy_pct, parse_gpu_value
from furiosa_perf.runner.monitor import aggregate_devices


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("power.draw", " 312.45", 312.45),
        ("memory.used", "2048", 2.0),  # MiB -> GiB
        ("pcie.link.gen.current", "5", 5),
        ("clocks_event_reasons.active", "0x0000000000000001", 0),  # idle only -> not throttled
        ("clocks_event_reasons.active", "0x0000000000000005", 4),  # idle + SW power cap -> power cap
        ("temperature.memory", "[N/A]", None),
        ("temperature.memory", "Not Supported", None),
        ("power.draw", "garbage", None),
    ],
)
def test_parse_gpu_value(field: str, raw: str, expected: float | int | None) -> None:
    """Each nvidia-smi cell becomes the unit DeviceSample uses; N/A and junk become None."""
    assert parse_gpu_value(field, raw) == expected


def test_aggregate_devices_sums_power_and_ors_throttle() -> None:
    """Power adds up, temperature takes the hottest device, utilization averages, throttle masks OR together."""
    samples = [
        DeviceSample(device=0, power_w=100.0, temp_c=40.0, util_pct=80.0, throttle=0),
        DeviceSample(device=1, power_w=120.0, temp_c=45.0, util_pct=60.0, throttle=4),
    ]

    assert aggregate_devices(samples) == {"power_w": 220.0, "peak_temp_c": 45.0, "avg_util_pct": 70.0, "throttle": 4}
    assert aggregate_devices([]) == {"power_w": None, "peak_temp_c": None, "avg_util_pct": None, "throttle": None}


def counters(cycles: int) -> dict[str, dict[str, float]]:
    """One RNGD with 8 PEs at 2000 MHz that has executed ``cycles`` cycles in total."""
    return {"0": {"exec_cycles": cycles, "num_pes": 8, "clock_mhz": 2000.0}}


def test_npu_busy_pct_divides_executed_cycles_by_active_capacity() -> None:
    """8 PEs x 2 GHz x 10 s = 160 G cycles of capacity; 120 G executed is 75 % busy."""
    assert npu_busy_pct(counters(0), counters(120_000_000_000), active_s=10.0) == 75.0


def test_npu_busy_pct_is_none_when_counters_reset_or_missing() -> None:
    """A counter that went backwards (device reopened), missing counters or no active time give None."""
    assert npu_busy_pct(counters(500), counters(100), active_s=10.0) is None
    assert npu_busy_pct({}, counters(100), active_s=10.0) is None
    assert npu_busy_pct(counters(0), counters(100), active_s=0.0) is None
