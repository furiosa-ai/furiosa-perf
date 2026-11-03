"""Where a run's results go and what is written there.

Layout under the output directory::

    <kind>/<device>_<num devices>[-<layout>]_<backend>_<version>/<task>/<org>--<model>[__<variant>]/
        run.json                  what was run, on what, with which versions
        env/                      raw environment probes: host, devices, topology, packages, freezes
        server.log  monitor.csv  devices.csv  metrics_snapshots.jsonl
        summary.csv  summary.md   (and per ISL/OSL pair)
        <point>/
            record.json           one self-describing measurement point
            result.json  bench.log  monitor.csv  devices.csv

Every value needed to interpret a point is inside ``run.json`` / ``record.json``;
consumers should not parse paths, so the layout can change without breaking them.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Any

RUN_FILE = "run.json"
ENV_DIR = "env"  # raw environment probes (see furiosa_perf.runner.env)
RECORD_FILE = "record.json"
SERVER_LOG = "server.log"


# Load options of service-scenario points (AIPerf) -> name prefix.
SCENARIO_LOAD_PREFIXES = {
    "concurrency": "u",
    "user-centric-rate": "ucr",
    "num-users": "n",
    "request-rate": "r",
    "trace-speed-pct": "s",
}


def point_name(point: dict[str, Any]) -> str:
    """Directory name of a point.

    Fixed-length points: ``i1024-o128-c16``, ``i512-b5-c16`` (rerank) or ``i512-c16``
    (embedding). Scenario points are named by their load: ``u32`` for 32 users,
    ``ucr2.0-n64`` for a user-centric rate, ``r4.0`` for a request rate.

    Args:
        point: Client options of the point.

    Returns:
        The name.
    """
    load = [f"{prefix}{point[key]}" for key, prefix in SCENARIO_LOAD_PREFIXES.items() if key in point]
    if load and "random-input-len" not in point:
        return "-".join(load)
    parts = [f"i{point.get('random-input-len')}"]
    if point.get("random-output-len"):
        parts.append(f"o{point['random-output-len']}")
    elif "random-batch-size" in point:
        parts.append(f"b{point['random-batch-size']}")
    parts.append(f"c{point.get('max-concurrency')}")
    return "-".join(parts)


def check_unique_point_names(points: list[dict[str, Any]]) -> None:
    """Fail before running if two points would write to the same directory.

    Args:
        points: Expanded points of a preset.

    Raises:
        ValueError: If two points share a name, i.e. differ only in options the name leaves out.
    """
    seen: dict[str, int] = {}
    for index, point in enumerate(points):
        name = point_name(point)
        if name in seen:
            raise ValueError(
                f"points {seen[name]} and {index} would both be stored as '{name}'; "
                "they should differ in input length, output length, documents or concurrency"
            )
        seen[name] = index


def run_dir(
    output_dir: Path,
    kind: str,
    device: str,
    num_devices: int,
    layout: str,
    backend: str,
    task: str,
    model: str,
    variant: str = "",
) -> Path:
    """Directory of one run (one server configuration, one preset task, one model).

    Args:
        output_dir: Root given by ``--output-dir``.
        kind: Preset kind, e.g. ``baseline``.
        device: Device label, e.g. ``RNGD`` or ``H100-80GB-HBM3@500W``.
        num_devices: Devices the server uses.
        layout: :attr:`furiosa_perf.spec.Parallel.label`, ``""`` for pure tensor parallelism.
        backend: ``<backend>_<version>``, e.g. ``furiosa-llm_2026.4.0``.
        task: Preset task.
        model: Model ID the server serves; ``/`` becomes ``--``.
        variant: :func:`furiosa_perf.spec.variant_label` of the run's ``--server-arg`` options,
            so the same measurement with a server option changed gets its own directory.

    Returns:
        The run directory (not created).
    """
    layout_suffix = f"-{layout}" if layout else ""
    name = model.replace("/", "--") + ("__" + re.sub(r"[^\w.=+-]+", "-", variant) if variant else "")
    return output_dir / kind / f"{device}_{num_devices}{layout_suffix}_{backend}" / task / name


def prepare_run_dir(path: Path, resume: bool, overwrite: bool) -> None:
    """Create the run directory, refusing to mix two runs in one place.

    Args:
        path: Run directory.
        resume: Continue a run already there; points recorded as valid are kept.
        overwrite: Delete a run already there and start over.

    Raises:
        RuntimeError: If a run is already there and neither ``resume`` nor ``overwrite`` is set.
    """
    if (path / RUN_FILE).exists() or any(path.glob(f"*/{RECORD_FILE}")):
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise RuntimeError(
                f"the results directory {path} already holds a run; pass --resume to finish it or --overwrite to replace it"
            )
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as indented JSON, replacing the file only once it is complete."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file written by :func:`write_json`; ``None`` if it is missing or unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
