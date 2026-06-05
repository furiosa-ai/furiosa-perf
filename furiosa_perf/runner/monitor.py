"""Hardware monitoring utilities for Furiosa Perf benchmarks.

This module periodically samples accelerator (NPU/GPU) metrics and server-side
runtime metrics (HTTP `/metrics`) during a benchmark run, then writes a
CSV-compatible log file.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import astuple, dataclass, fields
from datetime import datetime, timezone
from multiprocessing import Process
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import requests

from furiosa_perf.utils.logger import logger


@dataclass
class HardwareMonitorData:
    """One row of monitoring data written to the CSV log."""

    timestamp: str = ""
    used_device_num: int = 0
    power_consumption: float = 0.0
    peak_temperature: float = 0.0
    avg_utilization: float = 0.0
    kv_cache_usage_percentage: float = 0.0
    num_requests_running: float = 0.0
    num_requests_waiting: float = 0.0
    host_cpu_utils: float = 0.0
    host_memory_usage_gib: float = 0.0

    @classmethod
    def header(cls) -> str:
        """Return the CSV header line for the monitoring log."""
        return ",".join([f.name for f in fields(cls)]) + "\n"

    def __str__(self) -> str:
        """Serialize as one CSV line (including a trailing newline)."""
        self.avg_utilization = (
            round(self.avg_utilization / self.used_device_num, 2) if self.used_device_num > 0 else 0.0
        )
        self.power_consumption = round(self.power_consumption, 2)
        self.peak_temperature = round(self.peak_temperature, 2)

        return ",".join(str(v) for v in astuple(self)) + "\n"


class HardwareMonitor:
    """Run accelerator/host monitoring in a separate process."""

    @staticmethod
    def start_monitor(
        host: str,
        port: int,
        server_pid: int,
        device_name: str,
        used_device_num: int,
        result_dir_path: Path,
        stop_event: EventType,
    ) -> Process:
        """Start a monitoring process and return it.

        Args:
            host: Hostname/IP of the API server exposing `/metrics`.
            port: Port of the API server exposing `/metrics`.
            server_pid: PID of the benchmark server process (for host metrics).
            device_name: Hardware name (e.g., "RNGD" for Furiosa NPU).
            used_device_num: Number of devices used (included in output filename).
            result_dir_path: Directory to write the monitoring CSV.
            stop_event: Event used to stop the monitoring loop.
        """
        if device_name == "RNGD":
            target_function = HardwareMonitor._monitor_npu
        else:
            target_function = HardwareMonitor._monitor_gpu

        result_file_path = os.path.join(
            result_dir_path,
            f"{device_name}_{used_device_num}_monitoring_log.csv",
        )
        proc = Process(
            target=target_function,
            args=(host, port, server_pid, result_file_path, stop_event),
        )
        proc.start()
        return proc

    @staticmethod
    def _safe_sleep(interval: float, start_time: float) -> None:
        """Sleep the remaining time in the interval (never sleeps negative)."""
        remaining = interval - (time.perf_counter() - start_time)
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _fetch_metrics_text(host: str, port: int) -> str | None:
        """Fetch `/metrics` content, returning None on errors."""
        try:
            resp = requests.get(f"http://{host}:{port}/metrics", timeout=2)
            if resp.status_code != 200:
                return None
            return resp.text
        except requests.RequestException:
            return None

    @staticmethod
    def _get_process_tree_usage(parent: psutil.Process) -> tuple[float, float]:
        """Return (cpu_percent_sum, rss_gib_sum) for a process tree.

        Notes:
            - `psutil.Process.cpu_percent(None)` must be called once before
              sampling to prime internal counters, otherwise the first result
              is 0.0.
        """
        procs: list[psutil.Process] = [parent]
        try:
            procs += parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        tree_cpu = 0.0
        tree_rss = 0
        for p in procs:
            try:
                tree_cpu += p.cpu_percent(None)
                tree_rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        rss_gib = round(tree_rss / 1024 / 1024 / 1024, 2)
        return tree_cpu, rss_gib

    @staticmethod
    def _monitor_npu(
        host: str,
        port: int,
        server_pid: int,
        result_file_path: str,
        stop_event: EventType,
        interval: float = 1.0,
    ) -> None:
        """Monitor Furiosa NPU status and server metrics, writing CSV rows."""
        try:
            import furiosa_smi_py
        except ImportError:
            logger.warning("furiosa_smi_py is not installed. " "Please install it to enable Furiosa NPU monitoring.")
            return

        furiosa_smi_py.init()

        # Give the server a moment to initialize before sampling.
        time.sleep(10)

        devices = furiosa_smi_py.list_devices()
        observer = furiosa_smi_py.create_default_observer()

        if server_pid != -1:
            parent = psutil.Process(server_pid)
            parent.cpu_percent(None)

        try:
            with open(result_file_path, "w") as f:
                f.write(HardwareMonitorData.header())
                while not stop_event.is_set():
                    start_time = time.perf_counter()
                    data = HardwareMonitorData()
                    data.timestamp = datetime.now(timezone.utc).isoformat()

                    for device in devices:
                        power_consumption = device.power_consumption()
                        utilization = (
                            # Furiosa core utilization is reported per-PE;
                            # divide by 8 to get a device-level average.
                            sum(pe.pe_usage_percentage() for pe in observer.get_core_utilization(device))
                            / 8
                        )
                        temperature = device.device_temperature().soc_peak()
                        if utilization > 0:
                            data.used_device_num += 1
                            data.power_consumption += power_consumption
                            data.peak_temperature = max(data.peak_temperature, temperature)
                            data.avg_utilization += utilization

                    metrics_text = HardwareMonitor._fetch_metrics_text(host, port)
                    if metrics_text is None:
                        f.write(str(data))
                        f.flush()
                        time.sleep(interval)
                        continue

                    for line in metrics_text.splitlines():
                        if line.startswith("furiosa_llm_kv_cache_usage_perc"):
                            data.kv_cache_usage_percentage = float(line.split()[-1])
                        if line.startswith("furiosa_llm_num_requests_running"):
                            data.num_requests_running = float(line.split()[-1])
                        if line.startswith("furiosa_llm_num_requests_waiting"):
                            data.num_requests_waiting = float(line.split()[-1])

                    if server_pid != -1:
                        tree_cpu, rss_gib = HardwareMonitor._get_process_tree_usage(parent)
                    else:
                        tree_cpu = 0.0
                        rss_gib = 0.0

                    data.host_cpu_utils = tree_cpu
                    data.host_memory_usage_gib = rss_gib

                    f.write(str(data))
                    f.flush()
                    HardwareMonitor._safe_sleep(interval, start_time)
        except KeyboardInterrupt:
            logger.info("Furiosa NPU monitoring stopped by user.")

    @staticmethod
    def _monitor_gpu(
        host: str,
        port: int,
        server_pid: int,
        result_file_path: str,
        stop_event: EventType,
        interval: float = 1.0,
    ) -> None:
        """Monitor NVIDIA GPU status and server metrics, writing CSV rows."""
        if server_pid != -1:
            parent = psutil.Process(server_pid)
            parent.cpu_percent(None)

        try:
            with open(result_file_path, "w") as f:
                f.write(HardwareMonitorData.header())
                while not stop_event.is_set():
                    start_time = time.perf_counter()
                    data = HardwareMonitorData()
                    data.timestamp = datetime.now(timezone.utc).isoformat()

                    try:
                        nvidia_smi = subprocess.Popen(
                            [
                                "nvidia-smi",
                                "--query-gpu=name,utilization.gpu,power.draw," "temperature.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            stdout=subprocess.PIPE,
                        )
                        output, _ = nvidia_smi.communicate()
                    except FileNotFoundError:
                        logger.warning("nvidia-smi not found. Skipping NVIDIA GPU monitoring.")
                        return

                    devices_status = output.decode("utf-8").strip().split("\n")
                    for device_status in devices_status:
                        device_info = device_status.split(",")
                        utilization, power_consumption, temperature = map(
                            float,
                            device_info[1:4],
                        )
                        if utilization > 0:
                            data.used_device_num += 1
                            data.power_consumption += power_consumption
                            data.peak_temperature = max(
                                data.peak_temperature,
                                temperature,
                            )
                            data.avg_utilization += utilization

                    metrics_text = HardwareMonitor._fetch_metrics_text(host, port)
                    if metrics_text is None:
                        f.write(str(data))
                        f.flush()
                        time.sleep(interval)
                        continue

                    for line in metrics_text.splitlines():
                        if line.startswith("vllm:kv_cache_usage_perc"):
                            data.kv_cache_usage_percentage = float(line.split()[-1])
                        if line.startswith("vllm:num_requests_running"):
                            data.num_requests_running = float(line.split()[-1])
                        if line.startswith("vllm:num_requests_waiting"):
                            data.num_requests_waiting = float(line.split()[-1])

                    if server_pid != -1:
                        tree_cpu, rss_gib = HardwareMonitor._get_process_tree_usage(parent)
                    else:
                        tree_cpu = 0.0
                        rss_gib = 0.0

                    data.host_cpu_utils = tree_cpu
                    data.host_memory_usage_gib = rss_gib

                    f.write(str(data))
                    f.flush()
                    HardwareMonitor._safe_sleep(interval, start_time)
        except KeyboardInterrupt:
            logger.info("NVIDIA GPU monitoring stopped by user.")

    @staticmethod
    def get_benchmark_power_summary(
        csv_file_path: str,
        start_dt: str | None = None,
        end_dt: str | None = None,
        target_csv_file_path: str = "",
    ) -> dict[str, Any]:
        """Compute power summary statistics from the monitoring CSV for a benchmark window.

        Args:
            csv_file_path: Path to the monitoring data CSV file.
            start_dt: ISO-format start datetime (inclusive) for filtering rows.
                If ``None``, no lower bound is applied.
            end_dt: ISO-format end datetime (inclusive) for filtering rows.
                If ``None``, no upper bound is applied.
            target_csv_file_path: If non-empty, the filtered rows are written to
                this path as a CSV file.

        Returns:
            A dict with keys ``mean_power``, ``p95_power``, and ``p99_power``
            (all zero when the filtered window is empty or the file is unreadable).
        """
        df = pd.read_csv(
            csv_file_path,
            parse_dates=["timestamp"],
            keep_default_na=False,
        )
        power_metrics_info = {"mean_power": 0, "p95_power": 0, "p99_power": 0}
        mask = pd.Series(True, index=df.index)
        try:
            if start_dt is not None:
                mask &= df["timestamp"] >= pd.to_datetime(start_dt)
            if end_dt is not None:
                mask &= df["timestamp"] <= pd.to_datetime(end_dt)
            filtered_df = df[mask]
        except Exception as e:
            logger.error(f"Error filtering monitoring data: {e}")
            return power_metrics_info

        if target_csv_file_path:
            filtered_df.to_csv(target_csv_file_path, index=False, na_rep="N/A")

        if filtered_df.empty:
            return power_metrics_info

        power_consumptions = filtered_df["power_consumption"].tolist()

        power_metrics_info["mean_power"] = round(np.mean(power_consumptions), 2)
        power_metrics_info["p95_power"] = round(np.percentile(power_consumptions, 95), 2)
        power_metrics_info["p99_power"] = round(np.percentile(power_consumptions, 99), 2)

        return power_metrics_info
