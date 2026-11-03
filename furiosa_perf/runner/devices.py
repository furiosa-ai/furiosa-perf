"""Per-device accelerator sampling and per-process host usage for the monitor.

RNGD is read through the ``furiosa_smi`` Python package (furiosa-smi 2026.4 or later) and
NVIDIA GPUs through ``nvidia-smi``. Both samplers return one :class:`DeviceSample` per
configured device so that imbalance between the devices of one tensor-parallel group
stays visible; the monitor also keeps their sum/mean per row.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from dataclasses import dataclass, fields
from typing import Any

import psutil

from furiosa_perf.log import logger

# Bit 0 of the throttle mask means "idle" for both RNGD (THROTTLE_REASON_IDLE) and
# NVIDIA (clocks_event_reasons GpuIdle); it is not a slowdown, so it is masked out.
THROTTLE_IDLE_BIT = 0x1
GIB = 1024**3


@dataclass
class DeviceSample:
    """One device at one instant. Fields a backend cannot report stay ``None``.

    Attributes:
        device: Device index as given in the server config ``devices``.
        power_w: Board power draw reported by the driver. RNGD reports a smoothed
            value that lags load changes by a few seconds.
        temp_c: Chip temperature (RNGD: SoC peak; GPU: core).
        ambient_c: RNGD ambient sensor.
        mem_temp_c: GPU memory temperature.
        util_pct: Compute utilization (RNGD: mean over PEs; GPU: ``utilization.gpu``).
        mem_used_gib: Device memory in use. RNGD preallocates nearly all of it at server
            start, so it shows that the server holds the device rather than load.
        clock_mhz: Core clock (RNGD: mean over PEs; GPU: SM clock).
        mem_clock_mhz: GPU memory clock (RNGD does not report one).
        throttle: Active slowdown reasons as a bit mask with the idle bit removed;
            ``0`` means running at full speed.
        pcie_gen: Current PCIe generation (GPU).
        pcie_width: Current PCIe link width (GPU).
    """

    device: int
    power_w: float | None = None
    temp_c: float | None = None
    ambient_c: float | None = None
    mem_temp_c: float | None = None
    util_pct: float | None = None
    mem_used_gib: float | None = None
    clock_mhz: float | None = None
    mem_clock_mhz: float | None = None
    throttle: int | None = None
    pcie_gen: int | None = None
    pcie_width: int | None = None


DEVICE_COLUMNS = [f.name for f in fields(DeviceSample)]


class NpuSampler:
    """Sample the configured RNGD devices through ``furiosa_smi`` (furiosa-smi 2026.4+)."""

    def __init__(self, indices: list[int]) -> None:
        """Open the devices.

        Args:
            indices: Device indices from the server config; empty means all devices.

        Raises:
            ValueError: If an index in ``indices`` does not exist on this host.
        """
        import furiosa_smi  # optional dependency, present on RNGD hosts only

        devices = {int(d.dev_node_index()): d for d in furiosa_smi.list_devices()}
        missing = sorted(set(indices) - set(devices))
        if missing:
            raise ValueError(f"the NPU indices {missing} are invalid; detected devices are {sorted(devices)}")
        self.devices = {i: devices[i] for i in (indices or sorted(devices))}
        # Per-PE cycle counters of the previous sample, for utilization since then.
        self._previous = {i: device.performance_counter() for i, device in self.devices.items()}

    def sample(self) -> list[DeviceSample]:
        """Read every configured device once.

        Utilization is measured over the time since the previous sample from the PE
        cycle counters, which only advance while a process holds the device.

        Returns:
            One sample per device, in index order.
        """
        samples = []
        for index, device in self.devices.items():
            temperature = device.device_temperature()
            counters = device.performance_counter()
            previous = self._previous[index]
            utilization = [counters[pe].utilization_since(previous[pe]) for pe in counters if pe in previous]
            self._previous[index] = counters
            clocks = list(device.core_frequency().values())
            samples.append(
                DeviceSample(
                    device=index,
                    power_w=device.power_consumption(),
                    temp_c=temperature.soc_peak(),
                    ambient_c=temperature.ambient(),
                    util_pct=100.0 * sum(utilization) / len(utilization) if utilization else None,
                    mem_used_gib=npu_memory_gib(device)[0],
                    clock_mhz=sum(clocks) / len(clocks) if clocks else None,
                    throttle=npu_throttle_mask(device.throttle_reason()),
                )
            )
        return samples

    def counters(self) -> dict[str, Any]:
        """Cumulative counters and static state, captured at point boundaries.

        ``task_execution_cycle`` counts PE cycles spent executing; its difference over
        a window divided by ``clock * seconds`` is the exact busy fraction, which a
        once-per-second utilization sample can only approximate.

        Returns:
            Per device: summed execution cycles, PE count, mean clock, DRAM in use and
            the governor profile.
        """
        per_device = {}
        for index, device in self.devices.items():
            pes = list(device.performance_counter().values())
            clocks = list(device.core_frequency().values())
            per_device[str(index)] = {
                "exec_cycles": sum(pe.task_execution_cycle() for pe in pes),
                "num_pes": len(pes),
                "clock_mhz": sum(clocks) / len(clocks) if clocks else 0.0,
                "dram_used_gib": npu_memory_gib(device)[0],
                "governor": str(device.governor_profile()),
            }
        return per_device


def npu_memory_gib(device: Any) -> tuple[float, float]:
    """``(in_use_gib, total_gib)`` of an RNGD's DRAM, summed over its per-PE ranges."""
    ranges = device.memory_usage().values()
    used = sum(r.in_use_bytes() for r in ranges)
    total = sum(r.total_bytes() for r in ranges)
    return round(used / GIB, 2), round(total / GIB, 2)


def npu_throttle_mask(reasons: list[Any]) -> int:
    """Active RNGD slowdown reasons as a bit mask, the idle bit removed.

    ``furiosa_smi`` reports a list of ``ThrottleReason`` values; bit ``int(reason)``
    is set for each (``Idle`` is 0, so it lines up with :data:`THROTTLE_IDLE_BIT`).
    """
    mask = 0
    for reason in reasons:
        mask |= 1 << int(reason)
    return mask & ~THROTTLE_IDLE_BIT


# nvidia-smi --query-gpu field -> DeviceSample field. Older drivers know the throttle
# mask as clocks_throttle_reasons.active, newer ones as clocks_event_reasons.active.
GPU_FIELDS: dict[str, str] = {
    "index": "device",
    "power.draw": "power_w",
    "temperature.gpu": "temp_c",
    "temperature.memory": "mem_temp_c",
    "utilization.gpu": "util_pct",
    "memory.used": "mem_used_gib",
    "clocks.sm": "clock_mhz",
    "clocks.mem": "mem_clock_mhz",
    "clocks_event_reasons.active": "throttle",
    "clocks_throttle_reasons.active": "throttle",
    "pcie.link.gen.current": "pcie_gen",
    "pcie.link.width.current": "pcie_width",
}
NVIDIA_SMI_TIMEOUT_S = 10


def _nvidia_smi(query: list[str], indices: list[int]) -> str:
    """Run one ``nvidia-smi --query-gpu`` call and return its CSV body."""
    command = ["nvidia-smi", f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"]
    if indices:
        # -i takes the same indices as CUDA_VISIBLE_DEVICES (default enumeration order).
        command += ["-i", ",".join(map(str, indices))]
    return subprocess.run(command, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S, check=True).stdout


def parse_gpu_value(field: str, raw: str) -> float | int | None:
    """Convert one ``nvidia-smi`` CSV cell to the unit :class:`DeviceSample` uses.

    Args:
        field: Query field name.
        raw: Cell text, e.g. ``"312.45"``, ``"0x0000000000000004"`` or ``"[N/A]"``.

    Returns:
        The value, or ``None`` for N/A, unsupported or unparsable cells.
    """
    raw = raw.strip()
    if not raw or raw.startswith("[") or raw in ("N/A", "Not Supported"):
        return None
    try:
        if field.endswith("reasons.active"):
            return int(raw, 16) & ~THROTTLE_IDLE_BIT
        if field in ("index", "pcie.link.gen.current", "pcie.link.width.current"):
            return int(raw)
        if field == "memory.used":  # MiB
            return float(raw) / 1024.0
        return float(raw)
    except ValueError:
        return None


class GpuSampler:
    """Sample the configured NVIDIA GPUs through ``nvidia-smi``."""

    def __init__(self, indices: list[int]) -> None:
        """Probe which query fields this driver accepts.

        ``nvidia-smi`` rejects the whole query if one field is unknown, so each field
        is tried once and only the accepted ones are queried afterwards.

        Args:
            indices: GPU indices from the server config; empty means all GPUs.
        """
        self.indices = indices
        self.query: list[str] = []
        for field in GPU_FIELDS:
            target = GPU_FIELDS[field]
            if target in (GPU_FIELDS[f] for f in self.query):
                continue  # an alias of this field is already accepted
            try:
                _nvidia_smi([field], indices)
            except subprocess.CalledProcessError:
                continue
            self.query.append(field)
        logger.info(f"[monitor] nvidia-smi fields: {', '.join(self.query)}")

    def sample(self) -> list[DeviceSample]:
        """Read every configured GPU once.

        Returns:
            One sample per GPU, in the order ``nvidia-smi`` lists them.
        """
        samples = []
        for line in _nvidia_smi(self.query, self.indices).strip().splitlines():
            cells = line.split(",")
            values = {GPU_FIELDS[f]: parse_gpu_value(f, c) for f, c in zip(self.query, cells, strict=True)}
            samples.append(DeviceSample(**values))  # type: ignore[arg-type]
        return samples

    def counters(self) -> dict[str, Any]:
        """GPUs have no cumulative counter used here; kept for interface parity.

        Returns:
            An empty dict.
        """
        return {}


def make_sampler(device_name: str, indices: list[int]) -> NpuSampler | GpuSampler:
    """Return the sampler for ``device_name`` (``"RNGD"`` or a GPU name).

    Args:
        device_name: Accelerator name detected for the run.
        indices: Device indices from the server config.

    Returns:
        An opened sampler.
    """
    if device_name == "RNGD":
        return NpuSampler(indices)
    return GpuSampler(indices)


def npu_busy_pct(start: dict[str, Any], end: dict[str, Any], active_s: float) -> float | None:
    """Fraction of PE cycles spent executing during the active part of a point.

    The counters only advance while PEs execute, so dividing by the active duration
    (not the whole point window, which includes client start-up) gives the busy
    fraction while requests were in flight.

    Args:
        start: :meth:`NpuSampler.counters` at the point start.
        end: :meth:`NpuSampler.counters` at the point end.
        active_s: Seconds during which the server had running requests.

    Returns:
        Percentage over all configured devices, or ``None`` if the counters are
        missing, were reset in between (the server released the device), or
        ``active_s`` is not positive.
    """
    if not start or not end or active_s <= 0:
        return None
    executed = 0.0
    capacity = 0.0
    for device, after in end.items():
        before = start.get(device)
        if before is None or after["exec_cycles"] < before["exec_cycles"]:
            return None
        executed += after["exec_cycles"] - before["exec_cycles"]
        capacity += after["num_pes"] * after["clock_mhz"] * 1e6 * active_s
    if capacity <= 0:
        return None
    return round(100.0 * executed / capacity, 1)


class ProcessUsage:
    """CPU and memory of the server and the benchmark client, plus the whole host.

    ``psutil.Process.cpu_percent`` reports usage since its previous call on the same
    object, so process objects are cached across samples.

    The client is not passed in: it is every process furiosa-perf (the monitor's
    parent) started, minus the server tree and the monitor itself. That stays correct
    as the client is relaunched for every point.
    """

    def __init__(self, server_pid: int) -> None:
        """Prime the host and server CPU counters.

        Args:
            server_pid: PID of the API server, or ``-1`` when it is not known.
        """
        self.server_pid = server_pid
        self._cache: dict[int, psutil.Process] = {}
        psutil.cpu_percent(None)

    def _cpu(self, pid: int) -> tuple[float, int]:
        """Return ``(cpu_percent, rss_bytes)`` of ``pid``, ``(0, 0)`` if it is gone."""
        try:
            process = self._cache.get(pid)
            if process is None:
                process = self._cache[pid] = psutil.Process(pid)
                process.cpu_percent(None)  # first call only primes the counter
            return process.cpu_percent(None), process.memory_info().rss
        except psutil.Error:
            self._cache.pop(pid, None)
            return 0.0, 0

    @staticmethod
    def _tree(pid: int) -> set[int]:
        """PIDs of ``pid`` and all its descendants; empty if ``pid`` is gone."""
        with contextlib.suppress(psutil.Error):
            root = psutil.Process(pid)
            return {root.pid, *(child.pid for child in root.children(recursive=True))}
        return set()

    def sample(self) -> dict[str, float]:
        """Take one reading.

        Returns:
            ``server_cpu_pct``/``server_rss_gib`` (process tree), ``client_cpu_pct``,
            ``host_cpu_pct`` (100 = all cores) and ``host_mem_used_gib``. Process CPU
            is per core: 100 = one core fully busy.
        """
        server = self._tree(self.server_pid) if self.server_pid > 0 else set()
        spawned = self._tree(os.getppid()) - {os.getppid()}
        client = spawned - server - self._tree(os.getpid())

        server_usage = [self._cpu(pid) for pid in server]
        client_usage = [self._cpu(pid) for pid in client]
        for stale in set(self._cache) - server - client:
            del self._cache[stale]
        return {
            "server_cpu_pct": round(sum(cpu for cpu, _ in server_usage), 1),
            "server_rss_gib": round(sum(rss for _, rss in server_usage) / GIB, 2),
            "client_cpu_pct": round(sum(cpu for cpu, _ in client_usage), 1),
            "host_cpu_pct": psutil.cpu_percent(None),
            "host_mem_used_gib": round(psutil.virtual_memory().used / GIB, 2),
        }
