"""Request traces replayed by AIPerf: download once, then cut and time-scale per point.

A trace in Mooncake format has one request per line: ``timestamp`` (ms),
``input_length``, ``output_length`` and ``hash_ids`` (one id per 512-token prompt
block; equal ids mean equal blocks). AIPerf turns every id into a fixed block of
tokens, so requests that shared a prompt prefix in production share it again and
hit the server's prefix cache the same way, without any original text.

A point replays the trace at a fraction of its recorded speed: timestamps are
stretched by ``100 / speed_pct``. To keep every point about as long, a point uses
only the part of the trace, from a chosen start, that fits its wall-clock duration
at that speed.

A trace starts with an empty cache, unlike a service that has been running. So a
point can start mid-trace and first warm the cache with the requests recorded just
before its start; their prompts fill the prefix cache as the service's history
would have.
"""

import hashlib
import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from furiosa_perf.log import logger

# Mooncake traces hash prompts in blocks of this many tokens.
MOONCAKE_BLOCK_SIZE = 512


def datasets_dir() -> Path:
    """Directory where downloaded traces are cached, shared by every run on this host."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "furiosa-perf" / "datasets"


def fetch_trace(source: str, sha256: str) -> Path:
    """Return a local copy of a trace, downloading it on first use and checking its hash.

    Args:
        source: URL or local path of the trace.
        sha256: Expected SHA-256 of the file; a mismatch is an error, so a changed
            upstream file never silently changes results.

    Returns:
        Path of the verified local file.

    Raises:
        ValueError: If the file's SHA-256 differs from ``sha256``.
    """
    path = Path(source)
    if not path.exists():
        path = datasets_dir() / f"{sha256[:16]}-{Path(source).name}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading trace {source} -> {path}")
            temporary = path.with_suffix(".part")
            urllib.request.urlretrieve(source, temporary)  # noqa: S310 - preset-defined https URL
            temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"the trace {source} has SHA-256 {digest}; the preset expects {sha256}")
    return path


@dataclass
class TraceWindow:
    """Which part of a trace a point replayed.

    Attributes:
        speed_pct: Replay speed in percent of the recorded speed.
        start_seconds: Recorded time, from the trace's first request, where the slice starts.
        trace_seconds: Recorded time covered, from ``start_seconds``.
        requests: Requests replayed.
        dropped_too_long: Requests in that window left out because input plus output
            exceeds the server's maximum length.
        offered_req_per_s: Mean request rate offered to the server.
        warmup_requests: Requests recorded before the slice, sent first to warm the cache.
    """

    speed_pct: int
    start_seconds: float
    trace_seconds: float
    requests: int
    dropped_too_long: int
    offered_req_per_s: float
    warmup_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        """The window as plain data for ``record.json``."""
        return asdict(self)


def _fits(row: dict[str, Any], max_total_tokens: int | None) -> bool:
    """Whether a request's input plus output fits the server's maximum length."""
    return max_total_tokens is None or row["input_length"] + row["output_length"] <= max_total_tokens


def scale_trace(
    rows: list[dict[str, Any]],
    speed_pct: int,
    wall_seconds: float,
    max_total_tokens: int | None,
    start_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], TraceWindow]:
    """Cut the part of a trace that lasts ``wall_seconds`` at ``speed_pct`` and stretch its timestamps.

    Args:
        rows: Trace requests sorted by ``timestamp`` (ms from the trace start).
        speed_pct: Replay speed in percent of the recorded speed (100 = as recorded).
        wall_seconds: How long the replay should last.
        max_total_tokens: Drop requests whose input plus output exceeds this
            (the server's maximum length); ``None`` keeps all.
        start_seconds: Recorded time, from the first request, where the slice starts.

    Returns:
        The requests to replay, with timestamps rebased to 0 and stretched, and a
        description of the window.
    """
    start = (rows[0]["timestamp"] if rows else 0) + start_seconds * 1000
    trace_ms = wall_seconds * 1000 * speed_pct / 100
    window = [r for r in rows if 0 <= r["timestamp"] - start <= trace_ms]
    kept = [r for r in window if _fits(r, max_total_tokens)]
    stretch = 100 / speed_pct
    scaled = [{**r, "timestamp": round((r["timestamp"] - start) * stretch)} for r in kept]
    info = TraceWindow(
        speed_pct=speed_pct,
        start_seconds=start_seconds,
        trace_seconds=round(trace_ms / 1000, 1),
        requests=len(scaled),
        dropped_too_long=len(window) - len(kept),
        offered_req_per_s=round(len(scaled) / wall_seconds, 3) if wall_seconds > 0 else 0.0,
    )
    return scaled, info


def warmup_rows(
    rows: list[dict[str, Any]], start_seconds: float, warmup_seconds: float, max_total_tokens: int | None
) -> list[dict[str, Any]]:
    """The requests recorded in the ``warmup_seconds`` before ``start_seconds``, to fill the cache.

    They keep their recorded spacing (full speed: the server queues what it cannot
    take at once, and only the resulting cache matters) and ask for one output
    token, since only their prompts are cached.

    Returns:
        The warm-up requests, timestamps rebased to 0.
    """
    start = (rows[0]["timestamp"] if rows else 0) + start_seconds * 1000
    begin = start - warmup_seconds * 1000
    window = [r for r in rows if begin <= r["timestamp"] < start and _fits(r, max_total_tokens)]
    return [{**r, "timestamp": round(r["timestamp"] - begin), "output_length": 1} for r in window]


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL trace, sorted by timestamp."""
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return sorted(rows, key=lambda r: r["timestamp"])


def write_trace(rows: list[dict[str, Any]], path: Path) -> None:
    """Write requests as the JSONL file AIPerf replays."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
