"""Collect the environment a run depends on.

One collector serves three consumers: the checks ``run`` makes before launching
anything, the ``environment`` section of ``run.json``, and ``furiosa-perf env``.

Every probe keeps its raw output (written to ``env/`` of the run directory) and
only the fields results are compared by are parsed. A probe that fails records
its error and the rest carry on: a missing ``dpkg`` must not stop a benchmark.

The server's Python environment is inspected through its own interpreter
(:attr:`furiosa_perf.spec.Server.python`), never furiosa-perf's, so the
recorded backend version is the one that serves.
"""

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from furiosa_perf.log import logger
from furiosa_perf.runner.devices import npu_memory_gib
from furiosa_perf.spec import Server

PROBE_TIMEOUT_S = 60
GIB = 1024**3

# Environment variables recorded in run.json: the ones that change how the
# server, the client or the libraries under them behave.
ENV_VAR_PREFIXES = ("FURIOSA_", "NPU_", "RNGD_", "VLLM_", "CUDA_", "NCCL_", "OMP_", "HF_", "TORCH", "TOKENIZERS_")
# Variables whose name contains one of these have their value replaced.
SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
# Native (apt) packages recorded: drivers and runtimes shared by every venv on the host.
NATIVE_PACKAGE_MARKERS = ("furiosa", "nvidia", "cuda", "nccl")

# Terminal colour codes some tools print even into a pipe (nvidia-smi topo -m).
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

# furiosa_smi Arch (str() is "RNGD" from 2026.4, "Arch.Rngd" before) -> device name in result paths.
NPU_ARCH_NAMES = {"Rngd": "RNGD", "RngdS": "RNGD-S", "RngdPlus": "RNGD-PLUS", "RngdMax": "RNGD-MAX"}

GPU_QUERY = (
    "index,name,uuid,pci.bus_id,memory.total,driver_version,vbios_version,power.draw,"
    "power.limit,power.default_limit,power.max_limit,power.min_limit,pcie.link.gen.current,pcie.link.width.current"
)
# Fallback for drivers that reject a field of GPU_QUERY.
GPU_QUERY_MINIMAL = "index,name,memory.total,driver_version,vbios_version,power.limit,power.default_limit"


def npu_arch_name(arch: str) -> str:
    """Result-path name of an RNGD arch such as ``"Arch.RngdMax"`` or ``"Rngd"``."""
    variant = arch.split(".")[-1]
    return NPU_ARCH_NAMES.get(variant, variant.upper())


def gpu_product_name(raw: str) -> str:
    """Result-path name of an NVIDIA product, keeping every word that tells variants apart.

    ``"NVIDIA H100 80GB HBM3"`` -> ``"H100-80GB-HBM3"``,
    ``"NVIDIA RTX PRO 6000 Blackwell Server Edition"`` -> ``"RTX-PRO-6000-Blackwell-Server-Edition"``.
    """
    words = raw.strip().split()
    if words and words[0].upper() == "NVIDIA":
        words = words[1:]
    return "-".join("".join(c for c in word if c.isalnum()) for word in words if word)


def device_label(name: str, devices: list[dict[str, Any]]) -> str:
    """Device name for result paths, with a suffix only when a power setting is not the default.

    Args:
        name: Device name, e.g. ``"RNGD"`` or ``"H100-80GB-HBM3"``.
        devices: Metadata of the devices the server uses.

    Returns:
        ``name``, ``name@<W>W`` when a GPU power limit differs from its default
        (the lowest one if they differ), or ``name@powersave`` when an RNGD runs
        the PowerSave governor.
    """
    lowered = [
        d["power_limit_w"]
        for d in devices
        if d.get("power_limit_w")
        and d.get("default_power_limit_w")
        and d["power_limit_w"] != d["default_power_limit_w"]
    ]
    if lowered:
        return f"{name}@{round(min(lowered))}W"
    if any(str(d.get("governor", "")).lower() == "powersave" for d in devices):
        return f"{name}@powersave"
    return name


def probe(argv: list[str]) -> str:
    """Run a command and return its output; failures come back as text, never raise.

    Args:
        argv: Command to run.

    Returns:
        Standard output, or ``"error: ..."`` with the reason it could not be read.
    """
    if shutil.which(argv[0]) is None and not Path(argv[0]).exists():
        return f"error: {argv[0]} is not installed"
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"error: {e}"
    if done.returncode != 0:
        return f"error: {' '.join(argv)} exited with code {done.returncode}: {done.stderr.strip()[:500]}"
    return ANSI_ESCAPE.sub("", done.stdout)


def _number(raw: Any) -> float | None:
    """A numeric cell as float; ``None`` for empty, ``[N/A]`` or unparsable values."""
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def parse_key_values(text: str, separator: str) -> dict[str, str]:
    """``key<separator>value`` lines (``lscpu``, ``/etc/os-release``) as a dict with unquoted values."""
    pairs = {}
    for line in text.splitlines():
        key, found, value = line.partition(separator)
        if found:
            pairs[key.strip()] = value.strip().strip('"')
    return pairs


def recorded_env_vars(environ: dict[str, str]) -> dict[str, str]:
    """The environment variables worth recording, secrets masked.

    Args:
        environ: Usually ``os.environ``.

    Returns:
        Variables starting with :data:`ENV_VAR_PREFIXES`, sorted by name; values of
        names containing :data:`SECRET_MARKERS` are replaced by ``"***"``.
    """
    return {
        name: "***" if any(marker in name.upper() for marker in SECRET_MARKERS) else value
        for name, value in sorted(environ.items())
        if name.upper().startswith(ENV_VAR_PREFIXES)
    }


def native_packages(dpkg_output: str) -> dict[str, str]:
    """Driver and runtime packages from ``dpkg-query -W`` output (``name version`` per line)."""
    packages = {}
    for line in dpkg_output.splitlines():
        name, _, version = line.strip().partition(" ")
        if version and any(marker in name.lower() for marker in NATIVE_PACKAGE_MARKERS):
            packages[name] = version.strip()
    return packages


def parse_gpu_query(text: str, fields: str) -> list[dict[str, Any]]:
    """Rows of ``nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits`` as device dicts."""
    names = fields.split(",")
    devices = []
    for row in csv.reader(text.strip().splitlines()):
        cells = dict(zip(names, (cell.strip() for cell in row), strict=False))
        devices.append(
            {
                "index": int(cells["index"]),
                "name": gpu_product_name(cells["name"]),
                "product": cells["name"],
                "uuid": cells.get("uuid"),
                "pci_bus_id": cells.get("pci.bus_id"),
                "memory_gib": round((_number(cells.get("memory.total")) or 0.0) / 1024, 2),
                "driver_version": cells.get("driver_version"),
                "firmware_version": cells.get("vbios_version"),
                "idle_power_w": _number(cells.get("power.draw")),
                "power_limit_w": _number(cells.get("power.limit")),
                "default_power_limit_w": _number(cells.get("power.default_limit")),
                "max_power_limit_w": _number(cells.get("power.max_limit")),
                "min_power_limit_w": _number(cells.get("power.min_limit")),
                "pcie_gen": _number(cells.get("pcie.link.gen.current")),
                "pcie_width": _number(cells.get("pcie.link.width.current")),
            }
        )
    return devices


def gpu_devices() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Every NVIDIA GPU on the host and the raw ``nvidia-smi`` outputs.

    Returns:
        ``(devices, raw)``; devices is empty when ``nvidia-smi`` is unavailable.
    """
    fields = GPU_QUERY
    text = probe(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if text.startswith("error:"):
        fields = GPU_QUERY_MINIMAL
        text = probe(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    raw = {"topology.txt": probe(["nvidia-smi", "topo", "-m"]), "smi.txt": probe(["nvidia-smi", "-q"])}
    devices = [] if text.startswith("error:") else parse_gpu_query(text, fields)
    return devices, raw


def npu_devices() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Every RNGD on the host and the raw ``furiosa-smi`` outputs.

    RNGD has no default power limit to compare with, so the governor profile is what
    marks a non-default power setting; the current limit is recorded as well.

    Returns:
        ``(devices, raw)``; devices is empty when ``furiosa_smi`` is unavailable.
    """
    raw = {
        "topology.txt": probe(["furiosa-smi", "topo"]),
        "smi.txt": probe(["furiosa-smi", "info", "--format", "json"]) + probe(["furiosa-smi", "version"]),
    }
    try:
        import furiosa_smi  # optional dependency, present on RNGD hosts only

        handles = list(furiosa_smi.list_devices())
        driver = str(furiosa_smi.driver_info())
    except Exception as e:
        logger.warning(f"cannot read RNGD devices through furiosa_smi; cause: {e}")
        return [], raw

    devices = []
    for handle in handles:
        link = handle.pcie_info().link_info()
        devices.append(
            {
                "index": int(handle.dev_node_index()),
                "name": npu_arch_name(str(handle.arch())),
                "uuid": str(handle.uuid()),
                "serial": str(handle.serial()),
                "pci_bus_id": str(handle.bdf()),
                "numa_node": int(handle.numa_node()),
                "memory_gib": npu_memory_gib(handle)[1],
                "driver_version": driver,
                "firmware_version": str(handle.firmware_version()),
                "governor": str(handle.governor_profile()).split(".")[-1],
                "power_limit_w": handle.power_limit() / 1000,  # reported in mW
                "idle_power_w": float(handle.power_consumption()),
                "pcie_gen": int(link.pcie_gen_status()),
                "pcie_width": int(link.link_width_status()),
                "links": {
                    str(other.dev_node_index()): str(handle.device_to_device_link_type(other)).split(".")[-1]
                    for other in handles
                    if other is not handle
                },
            }
        )
    return sorted(devices, key=lambda d: d["index"]), raw


def host_info() -> tuple[dict[str, Any], str]:
    """Host facts results are compared by, and the raw text they were read from."""
    os_release = Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else ""
    lscpu = probe(["lscpu"])
    cpu = parse_key_values(lscpu, ":")
    cgroup = Path("/proc/1/cgroup").read_text() if Path("/proc/1/cgroup").exists() else ""
    info = {
        "hostname": socket.gethostname(),
        "os": parse_key_values(os_release, "=").get("PRETTY_NAME", platform.platform()),
        "kernel": platform.release(),
        "cpu_model": cpu.get("Model name", platform.processor()),
        "cpu_sockets": _number(cpu.get("Socket(s)")),
        "cpus": psutil.cpu_count(),
        "numa_nodes": _number(cpu.get("NUMA node(s)")),
        "memory_gib": round(psutil.virtual_memory().total / GIB, 1),
        # cgroup v2 hides the runtime ("0::/"), so a Kubernetes pod also counts as a container.
        "container": Path("/.dockerenv").exists()
        or "KUBERNETES_SERVICE_HOST" in os.environ
        or any(k in cgroup for k in ("docker", "containerd", "kubepods")),
        "kubernetes": "KUBERNETES_SERVICE_HOST" in os.environ,
    }
    raw = f"# /etc/os-release\n{os_release}\n# uname -a\n{probe(['uname', '-a'])}\n# lscpu\n{lscpu}"
    return info, raw


def python_env_info(python: Path, distribution: str) -> dict[str, Any]:
    """Python and package versions of an environment, read with its own interpreter.

    Args:
        python: Interpreter of the environment.
        distribution: Package to look up, e.g. ``"furiosa-llm"``.

    Returns:
        ``python``, ``python_version``, ``version`` and ``path`` (where the package is
        imported from); versions are ``"Unknown"`` when they cannot be read.
    """
    # The script gets the names as arguments so nothing is formatted into code.
    code = (
        "import importlib.metadata as m, importlib.util as u, json, platform, sys\n"
        "try:\n    version = m.version(sys.argv[1])\nexcept m.PackageNotFoundError:\n    version = 'Unknown'\n"
        "spec = u.find_spec(sys.argv[2])\n"
        "print(json.dumps({'python_version': platform.python_version(), 'version': version,"
        " 'path': spec.origin if spec else None}))\n"
    )
    out = probe([str(python), "-c", code, distribution, distribution.replace("-", "_")])
    try:
        info = json.loads(out)
    except ValueError:
        info = {"python_version": "Unknown", "version": "Unknown", "path": None, "error": out.strip()[:500]}
    return {"python": str(python), **info}


def freeze(python: Path) -> str:
    """``pip freeze`` of an environment: through uv (uv venvs have no pip), else pip."""
    text = probe(["uv", "pip", "freeze", "--python", str(python)])
    if text.startswith("error:"):
        text = probe([str(python), "-m", "pip", "freeze"])
    return text


def model_info(model: str) -> dict[str, Any]:
    """Which model files are served: the Hub revision, or a hash of a local ``config.json``."""
    path = Path(model).expanduser()
    if path.is_dir():
        config = path / "config.json"
        digest = hashlib.sha256(config.read_bytes()).hexdigest()[:16] if config.exists() else None
        return {"id": model, "local_path": str(path.resolve()), "config_sha256": digest}
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(model, local_files_only=True)
    except Exception:
        return {"id": model, "revision": None}
    return {"id": model, "revision": Path(snapshot).name}


def power_settings(hardware_type: str, indices: list[int]) -> list[dict[str, Any]]:
    """Current power settings of the used devices, to detect a change during a run.

    Args:
        hardware_type: ``npu`` or ``gpu``.
        indices: Devices the server uses.

    Returns:
        Per device, ``index`` and ``power_limit_w`` (GPU) or ``governor`` (RNGD).
    """
    devices = npu_devices()[0] if hardware_type == "npu" else gpu_devices()[0]
    keys = ("index", "governor") if hardware_type == "npu" else ("index", "power_limit_w")
    return [{key: d.get(key) for key in keys} for d in devices if d["index"] in indices]


@dataclass
class Environment:
    """What a run depends on, parsed for ``run.json`` and raw for ``env/``.

    Attributes:
        host: Host facts (OS, kernel, CPU, memory, container).
        devices: Every accelerator of the server's family on the host, by index.
        server_env: Python and backend versions of the server environment.
        native_packages: Driver/runtime apt packages shared by every environment.
        model: Hub revision or local-config hash of the served model.
        env_vars: Recorded environment variables, secrets masked.
        raw: File name -> raw probe output, written to ``env/``.
    """

    host: dict[str, Any]
    devices: list[dict[str, Any]]
    server_env: dict[str, Any]
    native_packages: dict[str, str]
    model: dict[str, Any]
    env_vars: dict[str, str]
    raw: dict[str, str] = field(default_factory=dict)

    def used_devices(self, indices: list[int]) -> list[dict[str, Any]]:
        """The devices with the given indices, in that order (missing ones skipped)."""
        by_index = {d["index"]: d for d in self.devices}
        return [by_index[i] for i in indices if i in by_index]

    def summary(self, indices: list[int]) -> dict[str, Any]:
        """The ``environment`` section of ``run.json``, limited to the used devices."""
        return {
            "host": self.host,
            "devices": self.used_devices(indices),
            "server_env": self.server_env,
            "native_packages": self.native_packages,
            "model": self.model,
            "env_vars": self.env_vars,
        }

    def write_raw(self, directory: Path) -> None:
        """Write every raw probe output, plus the parsed devices and variables, to ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in self.raw.items():
            (directory / name).write_text(text)
        (directory / "devices.json").write_text(json.dumps(self.devices, indent=2) + "\n")
        (directory / "env-vars.json").write_text(json.dumps(self.env_vars, indent=2) + "\n")


def collect(server: Server) -> Environment:
    """Collect the environment of ``server``: host, its devices, its Python environment, its model.

    Args:
        server: Server description; selects the device family, the environment to
            inspect and the model.

    Returns:
        The collected environment. Probes that failed leave ``None``/``"Unknown"``
        values and an ``error:`` line in their raw file.
    """
    host, host_raw = host_info()
    devices, device_raw = npu_devices() if server.hardware_type == "npu" else gpu_devices()
    dpkg = probe(["dpkg-query", "-W", "-f", "${Package} ${Version}\\n"])
    return Environment(
        host=host,
        devices=devices,
        server_env=python_env_info(server.python, server.backend),
        native_packages=native_packages(dpkg),
        model=model_info(server.model),
        env_vars=recorded_env_vars(dict(os.environ)),
        raw={
            "host.txt": host_raw,
            **device_raw,
            "packages.txt": dpkg
            if dpkg.startswith("error:")
            else "\n".join(f"{k} {v}" for k, v in native_packages(dpkg).items()),
            "server-freeze.txt": freeze(server.python),
        },
    )


def check(environment: Environment, server: Server) -> list[str]:
    """Problems that make a run pointless, found before launching anything.

    Args:
        environment: Output of :func:`collect`.
        server: The server about to be launched.

    Returns:
        One message per problem; empty when the run can start.
    """
    problems = []
    if environment.server_env.get("version") in (None, "Unknown"):
        problems.append(
            f"cannot find {server.backend} in the environment {server.env}; "
            f"set 'env' in the server config to the environment that has {server.backend} installed"
        )
    if not environment.devices:
        problems.append(f"no {server.hardware_type.upper()} devices are detected on this host")
    else:
        detected = {d["index"] for d in environment.devices}
        missing = [i for i in server.devices if i not in detected]
        if missing:
            problems.append(f"the device indices {missing} are invalid; detected devices are {sorted(detected)}")
    return problems
