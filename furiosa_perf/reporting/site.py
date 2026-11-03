"""Build the report site: collect results into run files and copy the viewer next to them.

Site layout::

    index.html  style.css  *.js     the viewer (furiosa_perf/reporting/viewer), copied as is
    theme/                          optional look of the site (``--theme``): theme.css and its assets
    data/index.json                 every run: what it measured, on what, and how many points
    data/runs/<run id>.json         the points of one run, loaded by the viewer when it needs them,
                                    and how it was run (server command, environment, client command per point)

Runs come from the results tree (``run.json`` + ``<point>/record.json``) and from results
written before it (``bench_space``: ``<device>_<n>_<backend>_<version>/<tool>/<task>/<model>/summary.csv``).
"""

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from furiosa_perf.results.store import RECORD_FILE, RUN_FILE, read_json
from furiosa_perf.spec import variant_label

VIEWER = resources.files("furiosa_perf.reporting") / "viewer"

# Task names of results written before the presets (bench_space) -> preset task names.
LEGACY_TASKS = {"offline": "text", "vl-offline": "image-text", "embeddings": "embedding", "reranker": "rerank"}
# bench_space run directory: <device>_<n>_<backend>_<version>
LEGACY_RUN_DIR = re.compile(r"^(?P<hw>.+?)_(?P<n>\d+)_(?P<backend>[^_]+)_(?P<ver>.+)$")
# Server flags worth showing next to a run (feature switches, not plumbing like ports or devices).
FLAG = re.compile(r"--((?:enable|experimental|disable)[\w-]*)")
IGNORED_FLAGS = {"enable-payload-logging"}


# A bench_space point's client command, recovered from the run log (results from before record.json kept it).
COMMAND_FILE = "command.txt"
# What a bench_space point's vllm bench result records about how it was run.
LEGACY_PARAMS = ("max_concurrency", "num_prompts", "request_rate", "duration", "completed", "failed")
LEGACY_ENV = ("Ubuntu Version", "Python Version", "Framework Version", "Device Info", "Model Info")


def _point_key(isl: Any, osl: Any, concurrency: Any) -> str:
    """Key of a point in a run's details, as the viewer finds it: ``<isl>/<osl>/<concurrency>``."""
    return f"{isl}/{osl}/{concurrency}"


def server_flags(command: str) -> list[str]:
    """Feature flags of a server command line, e.g. ``enable-overlap-scheduling``."""
    return sorted({f for f in FLAG.findall(command or "") if f not in IGNORED_FLAGS})


def run_id(run: dict[str, Any]) -> str:
    """File-safe id of a run: what was measured plus when, so a re-run gets its own id."""
    layout = f"-{run['layout']}" if run["layout"] else ""
    when = re.sub(r"\D", "", run["date"] or "")[:12]
    parts = [
        f"{run['hw']}_{run['n']}{layout}_{run['backend']}_{run['ver']}",
        run["kind"],
        run["task"],
        run["model"],
        run["variant"],
        when,
    ]
    return re.sub(r"[^\w.+-]+", "-", "--".join(p for p in parts if p))


def _number(text: str) -> Any:
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


def _run(
    *,
    kind: str,
    task: str,
    model_id: str,
    hw: str,
    n: int,
    layout: str,
    backend: str,
    ver: str,
    date: str | None,
    command: str,
    source: Path,
    points: list[dict[str, Any]],
    variant: str = "",
    details: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    run = {
        "kind": kind,
        "task": task,
        "model": model_id.rsplit("/", 1)[-1],  # the name results of different organisations are compared by
        "model_id": model_id,
        "hw": hw,
        "n": n,
        "layout": layout,
        "backend": backend,
        "ver": ver,
        "date": date,
        "flags": server_flags(command),
        "variant": variant,  # --server-arg options of the run, e.g. no-enable-prefix-caching
        "source": str(source),
        **extra,
    }
    run["id"] = run_id(run)
    return run | {"points": points, "details": details or {}}


def load_run(run_file: Path) -> dict[str, Any] | None:
    """A run of the results tree: its ``run.json`` and the ``record.json`` of each point."""
    meta = read_json(run_file)
    if meta is None:
        return None
    records = [r for r in (read_json(p) for p in sorted(run_file.parent.glob(f"*/{RECORD_FILE}"))) if r]

    def key(record: dict[str, Any]) -> str:
        return _point_key(*(record.get("summary", {}).get(k) for k in ("Input Tokens", "Output Tokens", "Concurrent")))

    return _run(
        # results from before the renames
        kind={"simulation": "scenario"}.get(meta["kind"], meta["kind"]),
        task={"text-gen": "text", "vision": "image-text"}.get(meta["task"], meta["task"]),
        model_id=meta["model"],
        hw=meta.get("hardware_product") or meta["hardware"],
        n=meta["num_devices"],
        layout=meta.get("layout", ""),
        backend=meta["backend"],
        ver=meta["backend_version"],
        date=meta.get("started"),
        command=(meta.get("server") or {}).get("command", ""),
        source=run_file.parent,
        points=[r["summary"] for r in records if "summary" in r],
        details={
            "server_command": (meta.get("server") or {}).get("command", ""),
            "environment": meta.get("environment") or {},
            "commands": {key(r): r["command"] for r in records if r.get("command")},
            "params": {key(r): r["point"] for r in records if r.get("point")},
        },
        client=meta.get("client"),
        status=meta.get("status"),
        capacity=meta.get("capacity"),
        variant=variant_label(meta.get("variant") or {}),
    )


def load_legacy_run(summary: Path) -> dict[str, Any] | None:
    """A run written before the results tree: one ``summary.csv`` with ``* key: value`` header lines."""
    model_dir, task_dir, run_dir = summary.parent, summary.parent.parent, summary.parents[3]
    match = LEGACY_RUN_DIR.match(run_dir.name)
    if match is None or "__" in model_dir.name:  # "model__flag" folders were one-off flag experiments
        return None
    meta: dict[str, str] = {}
    body: list[str] = []
    for line in summary.read_text().splitlines():
        if line.startswith("* "):
            key, _, value = line[2:].partition(":")
            meta[key.strip()] = value.strip()
        elif line.strip():
            body.append(line)
    points = [{k: _number(v) for k, v in row.items() if k} for row in csv.DictReader(body)]
    date = meta.get("date")  # 20260708-045349
    if date:
        date = datetime.strptime(date, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc).isoformat()
    return _run(
        kind="baseline",
        task=LEGACY_TASKS.get(task_dir.name, task_dir.name),
        model_id=meta.get("Model Info") or model_dir.name,
        hw=match["hw"],
        n=int(match["n"]),
        layout="",
        backend=match["backend"],
        ver=match["ver"],
        date=date,
        command=meta.get("API Server Command", ""),
        source=summary.parent,
        points=points,
        details=_legacy_details(meta, model_dir),
    )


def _legacy_details(meta: dict[str, str], model_dir: Path) -> dict[str, Any]:
    """How a bench_space run was run: its summary.csv header, and per point (``<isl>.<osl>.<concurrency>/``)
    the recovered client command and what the vllm bench result recorded."""
    commands, params = {}, {}
    for point_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        parts = point_dir.name.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue
        key = _point_key(*parts)
        if (point_dir / COMMAND_FILE).exists():
            commands[key] = (point_dir / COMMAND_FILE).read_text().strip()
        result = next((r for r in map(read_json, sorted(point_dir.glob("*.json"))) if r), None)
        if result:
            params[key] = {k: result[k] for k in LEGACY_PARAMS if k in result}
    return {
        "server_command": meta.get("API Server Command", ""),
        "environment": {k: meta[k] for k in LEGACY_ENV if k in meta},
        "commands": commands,
        "params": params,
    }


def collect_runs(roots: list[Path]) -> list[dict[str, Any]]:
    """Every run under ``roots``, from the results tree or from bench_space."""
    runs: dict[str, dict[str, Any]] = {}
    for root in roots:
        for run_file in sorted(root.rglob(RUN_FILE)):
            run = load_run(run_file)
            if run is not None and run["points"]:
                runs[run["id"]] = run
        for summary in sorted(root.rglob("summary.csv")):
            if (summary.parent / RUN_FILE).exists():
                continue  # the results tree writes summary.csv too; its run.json was read above
            run = load_legacy_run(summary)
            if run is not None and run["points"]:
                runs[run["id"]] = run
    return sorted(runs.values(), key=lambda r: r["id"])


def _columns(points: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Points as columns and rows: every key in order of first appearance, missing values null."""
    cols: list[str] = []
    for point in points:
        cols.extend(k for k in point if k not in cols)
    return {"cols": cols, "rows": [[point.get(c) for c in cols] for point in points]}


def _shapes(points: list[dict[str, Any]]) -> list[str]:
    pairs = {(p["Input Tokens"], p["Output Tokens"]) for p in points if "Input Tokens" in p and "Output Tokens" in p}
    return [f"{i}/{o}" for i, o in sorted(pairs)]


def _failed(point: dict[str, Any]) -> bool:
    if "ok" in point:
        return not point["ok"]
    # Older results were not judged: a point failed when it measured no throughput
    # (output tokens for generation, input tokens for embedding and rerank, which have no output).
    throughput = point.get("Output Throughput(tok/s)", point.get("Total Throughput(tok/s)"))
    return not (throughput or 0) > 0


def write_data(runs: list[dict[str, Any]], site: Path) -> None:
    """Write ``data/index.json`` and one ``data/runs/<id>.json`` per run, replacing what was there."""
    data = site / "data"
    shutil.rmtree(data, ignore_errors=True)
    (data / "runs").mkdir(parents=True)
    index = []
    for run in runs:
        points = run["points"]
        meta = {k: v for k, v in run.items() if k not in ("points", "details")}
        index.append(meta | {"num_points": len(points), "failed": sum(map(_failed, points)), "shapes": _shapes(points)})
        body = _columns(points) | {"details": run["details"]}
        (data / "runs" / f"{run['id']}.json").write_text(json.dumps(body, separators=(",", ":")))
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (data / "index.json").write_text(json.dumps({"generated": generated, "runs": index}, separators=(",", ":")))


def write_viewer(site: Path, theme: Path | None) -> None:
    """Copy the viewer, and the theme directory (``theme.css`` and its assets) when given."""
    site.mkdir(parents=True, exist_ok=True)
    for item in VIEWER.iterdir():
        if item.is_file():
            (site / item.name).write_bytes(item.read_bytes())
    shutil.rmtree(site / "theme", ignore_errors=True)
    if theme is not None:
        shutil.copytree(theme, site / "theme")
    else:
        (site / "theme").mkdir()
        (site / "theme" / "theme.css").write_text(
            "/* no theme: run `furiosa-perf report --theme <dir>` to add one */\n"
        )
