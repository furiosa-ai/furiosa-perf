"""Parse the serving framework's Prometheus ``/metrics`` exposition.

Two consumers share the parser:

* the monitor loop, which keeps a handful of gauges and counters per sample
  (:func:`sample_columns`), and
* the per-point summary, which compares the full exposition taken at the
  point start and end (:func:`summarize_snapshots`). Counters and histograms are
  cumulative, so the difference of two snapshots is exactly the point's share
  without having to decide up front which metrics matter.

Latency histograms are reduced to means (``_sum`` / ``_count``), not quantiles: their
buckets are too coarse (e.g. 10 ms then 25 ms for inter-token latency) for an
interpolated percentile to mean anything.

furiosa-llm names its metrics ``furiosa_llm_*`` and vLLM ``vllm:*``; the rest of the
name follows vLLM in both, so one table of names covers both backends.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

FURIOSA_PREFIX = "furiosa_llm_"
VLLM_PREFIX = "vllm:"

# One exposition line: name, optional {labels}, value (an optional timestamp is ignored).
_LINE = re.compile(r"^(?P<name>[A-Za-z_:][\w:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    """One line of the exposition.

    Attributes:
        name: Metric name with the backend prefix and a trailing ``_total`` removed,
            e.g. ``prompt_tokens`` for ``furiosa_llm_prompt_tokens_total``.
        labels: Label set of the line.
        value: Sample value.
    """

    name: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class Column:
    """How one monitor CSV column is derived from the exposition.

    Attributes:
        names: Accepted metric names (after prefix and ``_total`` removal); the first
            one present wins, which absorbs renames across backend versions.
        reduce: How label sets (one per engine) are combined: ``"sum"`` for counts and
            counters, ``"max"`` for ratios such as KV-cache usage.
        scale: Multiplier applied after reduction.
    """

    names: tuple[str, ...]
    reduce: str = "sum"
    scale: float = 1.0


# Monitor CSV column -> source metric. KV-cache usage is a 0-1 ratio in both backends
# (furiosa-llm calls it ``kv_cache_usage_percent`` regardless), stored here as 0-100.
SAMPLE_COLUMNS: dict[str, Column] = {
    "num_requests_running": Column(("num_requests_running",)),
    "num_requests_waiting": Column(("num_requests_waiting",)),
    "kv_cache_usage_pct": Column(
        ("kv_cache_usage_percent", "kv_cache_usage_perc", "gpu_cache_usage_perc"), reduce="max", scale=100.0
    ),
    "prefix_cache_hits": Column(("prefix_cache_hits",)),
    "prefix_cache_queries": Column(("prefix_cache_queries",)),
    "prompt_tokens_total": Column(("prompt_tokens",)),
    "generation_tokens_total": Column(("generation_tokens",)),
    "num_preemptions_total": Column(("num_preemptions",)),  # vLLM only; furiosa-llm does not expose it
    "wire_hit_rate": Column(("wire_hit_rate",), reduce="max"),  # furiosa-llm only
}

# Latency histograms summarized per point; vLLM renamed time_per_output_token to inter_token_latency.
LATENCY_HISTOGRAMS: dict[str, tuple[str, ...]] = {
    "ttft": ("time_to_first_token_seconds",),
    "itl": ("inter_token_latency_seconds", "time_per_output_token_seconds"),
    "e2el": ("e2e_request_latency_seconds",),
}
FINISHED_REASONS = ("stop", "length", "abort")


def parse(text: str, prefix: str) -> list[Sample]:
    """Parse the lines of ``text`` that start with ``prefix``.

    Comments, other prefixes and unparsable values are skipped.

    Args:
        text: Body of ``GET /metrics``.
        prefix: Backend metric prefix, :data:`FURIOSA_PREFIX` or :data:`VLLM_PREFIX`.

    Returns:
        The samples, in exposition order.
    """
    samples = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        match = _LINE.match(line)
        if match is None:
            continue
        try:
            value = float(match["value"])
        except ValueError:
            continue
        name = match["name"][len(prefix) :].removesuffix("_total")
        labels = dict(_LABEL.findall(match["labels"] or ""))
        samples.append(Sample(name, labels, value))
    return samples


def sample_columns(samples: list[Sample]) -> dict[str, float]:
    """Reduce the exposition to the monitor CSV columns of :data:`SAMPLE_COLUMNS`.

    Args:
        samples: Output of :func:`parse`.

    Returns:
        Column -> value for every column whose metric is present.
    """
    by_name: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        by_name[sample.name].append(sample.value)

    columns = {}
    for column, spec in SAMPLE_COLUMNS.items():
        name = next((n for n in spec.names if n in by_name), None)
        if name is None:
            continue
        values = by_name[name]
        reduced = max(values) if spec.reduce == "max" else sum(values)
        columns[column] = reduced * spec.scale
    return columns


def _counter_delta(start: list[Sample], end: list[Sample], name: str, **labels: str) -> float:
    """Sum of ``end - start`` over the label sets of ``name`` that match ``labels``."""

    def total(samples: list[Sample]) -> float:
        return sum(s.value for s in samples if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()))

    return max(0.0, total(end) - total(start))


def summarize_snapshots(start_text: str, end_text: str, prefix: str) -> dict[str, float | None]:
    """Server-side view of one point, from ``/metrics`` taken before and after it.

    Args:
        start_text: Exposition captured just before the client started.
        end_text: Exposition captured just after the client exited.
        prefix: Backend metric prefix.

    Returns:
        ``server_<ttft|itl|e2el>_mean_ms`` (``None`` when the histogram is missing or
        nothing was observed) and ``finished_<reason>`` request counts.
    """
    start, end = parse(start_text, prefix), parse(end_text, prefix)
    present = {s.name.removesuffix("_count") for s in end}

    summary: dict[str, float | None] = {}
    for key, candidates in LATENCY_HISTOGRAMS.items():
        histogram = next((h for h in candidates if h in present), None)
        count = _counter_delta(start, end, f"{histogram}_count") if histogram else 0.0
        seconds = _counter_delta(start, end, f"{histogram}_sum") / count if count > 0 else None
        summary[f"server_{key}_mean_ms"] = None if seconds is None else round(seconds * 1000.0, 2)
    for reason in FINISHED_REASONS:
        summary[f"finished_{reason}"] = _counter_delta(start, end, "request_success", finished_reason=reason)
    return summary
