# Adopted from https://github.com/vllm-project/vllm/blob/main/vllm/collect_env.py

# ruff: noqa
# code borrowed from https://github.com/pytorch/pytorch/blob/main/torch/utils/collect_env.py
import os
import sys
import csv
import subprocess

from typing import Any
from collections import namedtuple
from furiosa_perf.utils.logger import logger


def run_lambda(cmd: str):
    p = subprocess.run(cmd.split(), capture_output=True, text=True, check=False)
    return p.returncode, p.stdout, p.stderr


class SystemDetector:
    SystemEnv = namedtuple(
        "SystemEnv",
        [
            "os",
            "python",
            "runtime",
            "hardware",
        ],
    )

    @staticmethod
    def check_system_compatibility(hardware_type: str, backend: str) -> bool:
        system_info = SystemDetector.detect_system_info(hardware_type, backend)
        if system_info.hardware.get(hardware_type, None) is None:
            logger.warning(f"No {hardware_type.upper()} detected.")
            return False, system_info

        if system_info.runtime.split("_")[1] == "Unknown":
            logger.warning(f"No {hardware_type.upper()} detected.")
            return False, system_info

        return True, system_info

    @staticmethod
    def detect_system_info(hardware_type: str, backend: str) -> SystemEnv:
        python_version = sys.version_info

        return SystemDetector.SystemEnv(
            os=SystemDetector._get_os(),
            python=f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            runtime=SystemDetector._get_framework(backend),
            hardware=SystemDetector._get_hardware(hardware_type),
        )

    @staticmethod
    def _get_os():
        from platform import release

        rc, out, err = run_lambda("lsb_release -a")
        if rc == 0:
            ubuntu_os = out.split("Description:")[1].split("\n")[0].split("\t")[1]
        else:
            ubuntu_os = "Unknown"
        kernel_version = release()

        return "{} {}".format(ubuntu_os, kernel_version)

    @staticmethod
    def _get_hardware(hardware_type: str) -> dict[str, Any]:
        return {
            "cpu": SystemDetector._get_cpu_info(run_lambda),
            "memory": SystemDetector._get_memory_info(),
            f"{hardware_type}": getattr(SystemDetector, f"_get_{hardware_type}_info")(),
        }

    @staticmethod
    def _get_cpu_info(run_lambda) -> dict[str, Any]:
        rc, out, err = run_lambda("lscpu")

        if rc == 0:
            architecture = out.split("Architecture:")[1].strip().split("\n")[0]
            model_name = out.split("Model name:")[1].strip().split("\n")[0]
            cores = int(out.split("CPU(s):")[1].strip().split("\n")[0])
        else:
            architecture = "Unknown"
            model_name = "Unknown"
            cores = 0

        return {
            "architecture": architecture,
            "model_name": model_name,
            "cores": cores,
        }

    @staticmethod
    def _get_memory_info() -> str:
        rc, out, err = run_lambda("free -g")
        if rc == 0:
            memory = "{} GiB".format(out.split("Mem:")[1].split("\n")[0].split()[0])
        else:
            memory = "Unknown"
        return memory

    @staticmethod
    def _get_npu_info() -> dict[str, Any]:
        # We don't consider multiple NPU architectures in the same system for now
        npus: dict[str, Any] = {"name": "Unknown", "memory": "0 GiB", "devices": []}
        try:
            import furiosa_smi_py

            furiosa_smi_py.init()
            devices = furiosa_smi_py.list_devices()
            for device in devices:
                device_info = device.device_info()
                memory_utilization = device.memory_utilization()
                if npus["name"] != str(device_info.arch()).upper():
                    npus["name"] = str(device_info.arch()).upper()
                    npus["memory"] = "{} GiB".format(
                        round(
                            sum([mem_block.total_bytes() for mem_block in memory_utilization.dram().memory()])
                            / (1024 * 1024 * 1024),
                            2,
                        )
                    )

                npus["devices"].append(
                    {
                        "driver_version": str(furiosa_smi_py.driver_info()),
                        "firmware_version": str(device_info.firmware_version()),
                    }
                )
        except ImportError as e:
            logger.warning("No Furiosa NPU detected. Ensure 'furiosa-smi' is installed and the NPU is connected.")
            logger.warning(f"Furiosa NPU detection failed: {e}")
            pass

        return npus

    @staticmethod
    def _get_gpu_info() -> dict[str, Any]:
        def normalize_gpu_name(raw_name: str) -> str:
            name = raw_name.split(" ")
            if name[1] == "RTX" and "6000" in name:
                name = "RTX-PRO-6000"
            else:
                name = "-".join(name[1:3])
            return name.upper()

        # We don't consider multiple GPU architectures in the same system for now
        gpus: dict[str, Any] = {"name": "Unknown", "memory": "0 GiB", "devices": []}

        try:
            rc, out, err = run_lambda(
                "nvidia-smi --query-gpu=name,memory.total,driver_version,vbios_version --format=csv,nounits"
            )
            if rc == 0:
                reader = csv.DictReader(out.strip().split("\n"))
                for row in reader:
                    if gpus["name"] != normalize_gpu_name(row["name"]):
                        gpus["name"] = normalize_gpu_name(row["name"])
                        gpus["memory"] = "{} GiB".format(round(int(row[" memory.total [MiB]"]) / (1024 * 1024), 2))
                    gpus["devices"].append(
                        {"driver_version": row[" driver_version"], "firmware_version": row[" vbios_version"]}
                    )
            else:
                logger.warning(f"NVIDIA GPU detection failed: {err}")
        except Exception as e:
            logger.warning(f"NVIDIA GPU detection failed: {e}")
        return gpus

    @staticmethod
    def _get_framework(backend: str) -> str:
        from importlib.metadata import version, PackageNotFoundError

        try:
            return f"{backend}_{version(backend)}"
        except PackageNotFoundError:
            return f"{backend}_Unknown"
