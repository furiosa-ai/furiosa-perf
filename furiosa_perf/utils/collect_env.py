# Adopted from https://github.com/vllm-project/vllm/blob/main/vllm/collect_env.py
# ruff: noqa

import csv
import os
import subprocess
import sys
from collections import namedtuple
from typing import Any, Callable

from furiosa_perf.utils.logger import logger


def run_lambda(cmd: str) -> tuple[int, str, str]:
    """Run *cmd* as a subprocess and return (returncode, stdout, stderr)."""
    p = subprocess.run(cmd.split(), capture_output=True, text=True, check=False)
    return p.returncode, p.stdout, p.stderr


class SystemDetector:
    """Detect OS, Python version, framework version, and hardware info."""

    SystemEnv = namedtuple("SystemEnv", ["os", "python", "runtime", "hardware"])

    @staticmethod
    def check_system_compatibility(
        hardware_type: str, backend: str
    ) -> tuple[bool, "SystemDetector.SystemEnv"]:
        """Check whether the required hardware and backend are available.

        Args:
            hardware_type: Target accelerator type (e.g. ``"npu"`` or ``"gpu"``).
            backend: Serving framework name (e.g. ``"furiosa-llm"`` or ``"vllm"``).

        Returns:
            A tuple of ``(is_compatible, system_info)``. ``is_compatible`` is
            ``False`` when the hardware or framework is not found.
        """
        system_info = SystemDetector.detect_system_info(hardware_type, backend)
        hw = system_info.hardware.get(hardware_type)
        if hw is None or hw.get("name") == "Unknown":
            logger.warning(f"No {hardware_type.upper()} detected.")
            return False, system_info

        if system_info.runtime.split("_")[1] == "Unknown":
            logger.warning(f"Backend {backend!r} not found or version unknown.")
            return False, system_info

        return True, system_info

    @staticmethod
    def detect_system_info(hardware_type: str, backend: str) -> "SystemDetector.SystemEnv":
        """Collect OS, Python, framework, and hardware metadata.

        Args:
            hardware_type: Accelerator type to probe (``"npu"`` or ``"gpu"``).
            backend: Framework name used to look up the installed package version.

        Returns:
            A :class:`SystemEnv` namedtuple with all detected fields.
        """
        pv = sys.version_info
        return SystemDetector.SystemEnv(
            os=SystemDetector._get_os(),
            python=f"{pv.major}.{pv.minor}.{pv.micro}",
            runtime=SystemDetector._get_framework(backend),
            hardware=SystemDetector._get_hardware(hardware_type),
        )

    @staticmethod
    def _get_os() -> str:
        """Return a human-readable OS string (distro + kernel version)."""
        from platform import release

        rc, out, _ = run_lambda("lsb_release -a")
        if rc == 0:
            distro = out.split("Description:")[1].split("\n")[0].split("\t")[1].strip()
        else:
            distro = "Unknown"

        return f"{distro} {release()}"

    @staticmethod
    def _get_hardware(hardware_type: str) -> dict[str, Any]:
        """Collect CPU, memory, and accelerator information.

        Args:
            hardware_type: Which accelerator to probe (``"npu"`` or ``"gpu"``).

        Returns:
            A dict with keys ``"cpu"``, ``"memory"``, and *hardware_type*.
        """
        return {
            "cpu": SystemDetector._get_cpu_info(),
            "memory": SystemDetector._get_memory_info(),
            hardware_type: getattr(SystemDetector, f"_get_{hardware_type}_info")(),
        }

    @staticmethod
    def _get_cpu_info() -> dict[str, Any]:
        """Return CPU architecture, model name, and core count via ``lscpu``."""
        rc, out, _ = run_lambda("lscpu")
        if rc == 0:
            architecture = out.split("Architecture:")[1].strip().split("\n")[0]
            cores = int(out.split("CPU(s):")[1].strip().split("\n")[0])
            # Some architectures like ARM may not have "Model name:" field
            if "Model name:" in out:
                model_name = out.split("Model name:")[1].strip().split("\n")[0]
            else:
                model_name = architecture
        else:
            architecture = "Unknown"
            model_name = "Unknown"
            cores = 0

        return {"architecture": architecture, "model_name": model_name, "cores": cores}

    @staticmethod
    def _get_memory_info() -> str:
        """Return total host memory in GiB as a string (e.g. ``"256 GiB"``)."""
        rc, out, _ = run_lambda("free -g")
        if rc == 0:
            return "{} GiB".format(out.split("Mem:")[1].split("\n")[0].split()[0])
        return "Unknown"

    @staticmethod
    def _get_npu_info() -> dict[str, Any]:
        """Return Furiosa NPU name, memory, and per-device metadata.

        Requires ``furiosa_smi_py`` to be installed. Returns placeholder values
        with ``name="Unknown"`` if the package is unavailable or no device is found.
        """
        npus: dict[str, Any] = {"name": "Unknown", "memory": "0 GiB", "devices": []}
        try:
            import furiosa_smi_py

            furiosa_smi_py.init()
            for device in furiosa_smi_py.list_devices():
                device_info = device.device_info()
                mem_util = device.memory_utilization()
                arch = str(device_info.arch()).upper()
                if npus["name"] != arch:
                    npus["name"] = arch
                    total_bytes = sum(b.total_bytes() for b in mem_util.dram().memory())
                    npus["memory"] = f"{round(total_bytes / (1024 ** 3), 2)} GiB"
                npus["devices"].append(
                    {
                        "driver_version": str(furiosa_smi_py.driver_info()),
                        "firmware_version": str(device_info.firmware_version()),
                    }
                )
        except ImportError as e:
            logger.warning("No Furiosa NPU detected. Ensure 'furiosa-smi' is installed and the NPU is connected.")
            logger.warning(f"Furiosa NPU detection failed: {e}")

        return npus

    @staticmethod
    def _get_gpu_info() -> dict[str, Any]:
        """Return NVIDIA GPU name, memory, and per-device metadata via ``nvidia-smi``.

        Returns placeholder values with ``name="Unknown"`` if ``nvidia-smi`` is
        unavailable or returns a non-zero exit code.
        """

        def _normalize_name(raw: str) -> str:
            parts = raw.split()
            if len(parts) >= 3 and parts[1] == "RTX" and "6000" in parts:
                return "RTX-PRO-6000"
            return "-".join(parts[1:3]).upper()

        gpus: dict[str, Any] = {"name": "Unknown", "memory": "0 GiB", "devices": []}
        try:
            rc, out, err = run_lambda(
                "nvidia-smi --query-gpu=name,memory.total,driver_version,vbios_version --format=csv,nounits"
            )
            if rc != 0:
                logger.warning(f"NVIDIA GPU detection failed: {err}")
                return gpus

            for row in csv.DictReader(out.strip().split("\n")):
                name = _normalize_name(row["name"])
                if gpus["name"] != name:
                    gpus["name"] = name
                    mem_mib = int(row[" memory.total [MiB]"])
                    gpus["memory"] = f"{round(mem_mib / 1024, 2)} GiB"
                gpus["devices"].append(
                    {
                        "driver_version": row[" driver_version"],
                        "firmware_version": row[" vbios_version"],
                    }
                )
        except Exception as e:
            logger.warning(f"NVIDIA GPU detection failed: {e}")

        return gpus

    @staticmethod
    def _get_framework(backend: str) -> str:
        """Return ``"<backend>_<version>"`` or ``"<backend>_Unknown"`` if not installed.

        Args:
            backend: Package name to look up (e.g. ``"vllm"`` or ``"furiosa-llm"``).
        """
        from importlib.metadata import PackageNotFoundError, version

        try:
            return f"{backend}_{version(backend)}"
        except PackageNotFoundError:
            return f"{backend}_Unknown"
