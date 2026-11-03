"""Read and validate the two inputs of a run: the server config and the benchmark preset."""

import itertools
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from furiosa_perf.log import logger

# Serving backend -> accelerator family it runs on.
BACKENDS: dict[str, str] = {"furiosa-llm": "npu", "vllm": "gpu"}

# Server config keys furiosa-perf reads itself; every other key is passed through.
SERVER_FIELDS = ("backend", "model", "devices", "parallel", "env", "host", "port")

# Backend options furiosa-perf sets from ``devices`` and ``parallel``. They are not
# accepted as pass-through keys: their units differ between backends (furiosa-llm
# counts tensor parallelism in PEs), and two sources for one setting can disagree.
PARALLEL_SERVER_ARGS = ("tensor-parallel-size", "pipeline-parallel-size", "data-parallel-size")

# Mutually exclusive flag pairs: a preset that sets one removes the other from the
# server config instead of passing both.
EXCLUSIVE_SERVER_ARGS = (("enable-prefix-caching", "no-enable-prefix-caching"),)


@dataclass(frozen=True)
class Parallel:
    """Parallelism of the server, in units of devices (cards).

    ``tp * pp * dp`` equals the number of devices.

    Attributes:
        tp: Devices one model replica's tensors are split across. For furiosa-llm
            (2026.4+) this is fixed by the artifact and only checked, not passed.
        pp: Pipeline stages of one replica.
        dp: Independent replicas.
    """

    tp: int
    pp: int = 1
    dp: int = 1

    @property
    def label(self) -> str:
        """``""`` for pure tensor parallelism, else e.g. ``"tp2-dp2"`` for result names."""
        if self.pp == 1 and self.dp == 1:
            return ""
        parts = [f"tp{self.tp}"] + [f"{name}{size}" for name, size in (("pp", self.pp), ("dp", self.dp)) if size > 1]
        return "-".join(parts)


def parse_parallel(raw: Any, num_devices: int) -> Parallel:
    """Resolve ``parallel`` against the device count.

    Any one of ``tp``, ``pp`` and ``dp`` may be omitted and is then derived; ``pp``
    and ``dp`` default to 1, so an absent ``parallel`` is tensor parallelism over
    every device.

    Args:
        raw: Value of ``parallel`` in the server config (a mapping or ``None``).
        num_devices: Number of entries in ``devices``.

    Returns:
        The resolved parallelism.

    Raises:
        ValueError: If a key is unknown, a size is not a positive integer, or the
            sizes do not multiply to ``num_devices``.
    """
    raw = raw or {}
    if not isinstance(raw, dict) or set(raw) - {"tp", "pp", "dp"}:
        raise ValueError(f"the parallel setting '{raw}' is invalid; use a mapping with keys tp, pp and dp")
    sizes = {}
    for key, value in raw.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"the parallel size {key}={value!r} is invalid; use a positive integer")
        sizes[key] = value

    pp, dp = sizes.get("pp", 1), sizes.get("dp", 1)
    tp = sizes.get("tp")
    if tp is None:
        tp = num_devices // (pp * dp) if num_devices % (pp * dp) == 0 else 0
    elif "dp" not in sizes and num_devices % (tp * pp) == 0:
        dp = num_devices // (tp * pp)
    if tp < 1 or tp * pp * dp != num_devices:
        raise ValueError(
            f"the parallel setting {raw} does not fit {num_devices} devices; tp * pp * dp should equal the device count"
        )
    return Parallel(tp=tp, pp=pp, dp=dp)


@dataclass(frozen=True)
class Server:
    """How to launch the LLM API server.

    Attributes:
        backend: ``furiosa-llm`` or ``vllm``.
        model: Hugging Face model ID or local path to serve.
        devices: Accelerator indices to serve on.
        parallel: How the devices are split into tensor/pipeline/data parallelism.
        env: Root of the Python environment whose ``bin/<backend>`` is launched. Its
            interpreter is also where the backend version is read from, so the
            recorded version is the one that actually served.
        host: Address the server binds and the client connects to.
        port: Server port.
        args: Remaining options, passed to ``<backend> serve`` as ``--<key> <value>``.
    """

    backend: str
    model: str
    devices: list[int]
    parallel: Parallel
    env: Path
    host: str = "0.0.0.0"
    port: int = 8000
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def hardware_type(self) -> str:
        """Accelerator family of the backend: ``npu`` or ``gpu``."""
        return BACKENDS[self.backend]

    @property
    def executable(self) -> Path:
        """The backend's command-line entry point inside :attr:`env`."""
        return self.env / "bin" / self.backend

    @property
    def python(self) -> Path:
        """The interpreter of :attr:`env`."""
        return self.env / "bin" / "python"


def parse_devices(raw: Any) -> list[int]:
    """Read ``devices`` given as ``0``, ``"0,1"`` or ``[0, 1]``.

    Args:
        raw: Value of ``devices`` in the server config.

    Returns:
        The device indices in the given order.

    Raises:
        ValueError: If the value is empty, not a list of non-negative integers, or has duplicates.
    """
    items = raw if isinstance(raw, list) else str(raw).split(",")
    try:
        devices = [int(str(item).strip()) for item in items]
    except ValueError:
        devices = []
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        raise ValueError(f"the device list '{raw}' is invalid; use distinct indices such as 0,1,2,3")
    return devices


def parse_server(raw: dict[str, Any], default_env: Path) -> Server:
    """Build a :class:`Server` from the parsed server config.

    Keys may be written in snake_case or kebab-case; pass-through keys are
    normalised to kebab-case, the form both backends' command lines use. A key set
    to ``null`` or ``false`` is left off the command line.

    Args:
        raw: Mapping loaded from the server config YAML.
        default_env: Environment to use when ``env`` is not set.

    Returns:
        The validated server description.

    Raises:
        ValueError: If a required key is missing, the backend is unsupported, the
            parallelism does not fit the devices, or a pass-through key sets
            parallelism directly.
    """
    config = {key.replace("_", "-"): value for key, value in raw.items()}
    for key in ("backend", "model", "devices"):
        if config.get(key) in (None, ""):
            raise ValueError(f"the server config is missing '{key}'")
    backend = str(config["backend"])
    if backend not in BACKENDS:
        raise ValueError(f"the backend '{backend}' is unsupported; use one of {', '.join(BACKENDS)}")

    args = {key: value for key, value in config.items() if key not in SERVER_FIELDS}
    for key in args:
        if key in PARALLEL_SERVER_ARGS:
            raise ValueError(
                f"the server option '{key}' is not allowed; set 'parallel: {{tp, pp, dp}}' in units of devices instead"
            )
    devices = parse_devices(config["devices"])
    return Server(
        backend=backend,
        model=str(config["model"]),
        devices=devices,
        parallel=parse_parallel(config.get("parallel"), len(devices)),
        env=Path(config.get("env") or default_env).expanduser(),
        host=str(config.get("host", "0.0.0.0")),
        port=int(config.get("port", 8000)),
        args=args,
    )


def load_server(path: Path, default_env: Path) -> Server:
    """Load and validate a server config YAML.

    Args:
        path: Server config file.
        default_env: Environment to use when the config does not set ``env``.

    Returns:
        The validated server description.

    Raises:
        ValueError: If the file is not a mapping or fails :func:`parse_server`.
    """
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"the server config {path} should be a mapping of option names to values")
    try:
        server = parse_server(raw, default_env)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    logger.info(f"Loaded server config: {server}")
    return server


def apply_server_overrides(server: Server, overrides: dict[str, Any]) -> Server:
    """Return ``server`` with a preset's ``server`` options, or ``--server-arg``, applied on top.

    The preset wins over the server config: it states what its measurement needs (for
    example prefix caching off for the fixed-length baseline); ``--server-arg`` wins over
    both. ``null`` removes an option.

    Args:
        server: Server from the server config.
        overrides: Options to apply, kebab-case keys.

    Returns:
        The server with merged pass-through options.

    Raises:
        ValueError: If an override sets parallelism, which only ``parallel`` may.
    """
    for key in overrides:
        if key in PARALLEL_SERVER_ARGS:
            raise ValueError(
                f"the server option '{key}' cannot be overridden; parallelism comes from the server config"
            )
    args = dict(server.args)
    for pair in EXCLUSIVE_SERVER_ARGS:
        if any(key in overrides for key in pair):
            for key in pair:
                args.pop(key, None)
    args.update(overrides)
    return replace(server, args=args)


def parse_server_args(items: tuple[str, ...]) -> dict[str, Any]:
    """Server options given with ``--server-arg``: ``key`` for a flag, ``key=value`` otherwise.

    Values are read as YAML, so ``key=8192`` is a number and ``key=null`` removes the option.

    Raises:
        ValueError: If an item has no option name.
    """
    args: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.removeprefix("--").partition("=")
        if not key:
            raise ValueError(f"--server-arg '{item}' has no option name")
        args[key] = yaml.safe_load(value) if sep else True
    return args


def variant_label(args: dict[str, Any]) -> str:
    """Name of a run's ``--server-arg`` options, e.g. ``no-enable-prefix-caching``; ``""`` without any."""
    return "+".join(
        key if value is True else f"{key}={value if isinstance(value, str) else json.dumps(value)}"
        for key, value in sorted(args.items())
    )


# Client -> options furiosa-perf sets itself (target server, result location, UI).
MANAGED_CLIENT_ARGS: dict[str, tuple[str, ...]] = {
    "vllm-bench": ("model", "host", "port", "base-url", "save-result", "result-dir", "result-filename"),
    "aiperf": ("model", "model-names", "url", "artifact-dir", "output-artifact-dir", "ui-type", "ui"),
}
CLIENTS = tuple(MANAGED_CLIENT_ARGS)
# baseline: fixed-length sweep; scenario: service traffic such as multi-turn chat.
KINDS = ("baseline", "scenario")
# Service-level objectives a preset may set; each limits one latency percentile.
SLO_KEYS = ("ttft_p95_ms", "tpot_p95_ms")
# Load options a capacity search may vary: users talking at once, or the replay
# speed of a request trace in percent of its recorded speed.
SEARCH_KEYS = ("concurrency", "trace-speed-pct")
# Point options furiosa-perf itself consumes when replaying a trace (not passed to the client).
TRACE_POINT_KEYS = ("trace-speed-pct", "trace-duration", "trace-start", "trace-warmup")
# When the server is restarted: "never" (one server for the run) or "per-point"
# (a fresh server before every point, so no point inherits another's cache or state).
RESTART_POLICIES = ("never", "per-point")


@dataclass(frozen=True)
class Search:
    """Capacity search: bisect one load option for the largest value meeting the SLO.

    Attributes:
        key: Load option searched, e.g. ``concurrency`` (users).
        min: Smallest value tried; expected to meet the SLO.
        max: Largest value tried; expected to miss it.
        step: Resolution: stop when the passing and failing values are this close.
    """

    key: str
    min: int
    max: int
    step: int = 1


def parse_search(raw: dict[str, Any]) -> Search:
    """Validate a ``search`` mapping.

    Args:
        raw: ``{key, min, max, step}``; ``step`` defaults to 1.

    Returns:
        The search.

    Raises:
        ValueError: If a key is unknown, a bound is not a positive integer, or ``min >= max``.
    """
    unknown = set(raw) - {"key", "min", "max", "step"}
    if unknown:
        raise ValueError(f"the search keys {sorted(unknown)} are unknown; use key, min, max and step")
    if raw.get("key") not in SEARCH_KEYS:
        raise ValueError(f"the search key '{raw.get('key')}' is unsupported; use one of {', '.join(SEARCH_KEYS)}")
    values: dict[str, int] = {}
    for name in ("min", "max", "step"):
        value = raw.get(name, 1 if name == "step" else None)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"the search {name} '{value}' is invalid; use a positive integer")
        values[name] = value
    if values["min"] >= values["max"]:
        raise ValueError(f"the search range {values['min']}..{values['max']} is empty; min should be below max")
    return Search(key=str(raw["key"]), min=values["min"], max=values["max"], step=values["step"])


# Requests per point when the preset does not say: enough for every concurrent slot
# to be refilled twice after the first wave.
PROMPTS_PER_CONCURRENCY = 3


@dataclass(frozen=True)
class Preset:
    """What to measure and with which client.

    Attributes:
        kind: Measurement type; ``baseline`` is the fixed-length sweep.
        task: Label of what is measured (``text``, ``image-text``, ...), used in result paths.
        client: Benchmark client; ``vllm-bench`` runs ``vllm bench serve``.
        server: Server options the measurement requires, applied over the server config.
        points: One mapping of client options per measurement point, already expanded.
        slo: Latency limits a point must meet to count towards capacity, e.g.
            ``{"ttft_p95_ms": 3000, "tpot_p95_ms": 50}``; empty for no SLO.
        search: Capacity search replacing a fixed list of load values; ``points``
            then holds the single base point the searched value is added to.
        restart_server: ``"never"`` or ``"per-point"``, see :data:`RESTART_POLICIES`.
        trace: Request trace replayed instead of synthetic traffic: ``source`` (URL or
            path), ``sha256``, ``license``, and optionally ``max-total-tokens``.
    """

    kind: str
    task: str
    client: str
    server: dict[str, Any]
    points: list[dict[str, Any]]
    slo: dict[str, float] = field(default_factory=dict)
    search: Search | None = None
    restart_server: str = "never"
    trace: dict[str, Any] = field(default_factory=dict)


def _kebab(mapping: dict[str, Any]) -> dict[str, Any]:
    """``mapping`` with snake_case keys rewritten to kebab-case."""
    return {key.replace("_", "-"): value for key, value in mapping.items()}


def expand_matrix(args: dict[str, Any], matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand matrix rows into points: list values form a cartesian product.

    Each point is ``args`` overlaid with one combination of a row. ``num-prompts``
    defaults to :data:`PROMPTS_PER_CONCURRENCY` times ``max-concurrency``.

    Args:
        args: Client options shared by every point.
        matrix: Rows of options; a list value means one point per element.

    Returns:
        The points in row order, combinations in the order the keys appear.
    """
    points = []
    for row in matrix or [{}]:
        row = _kebab(row)
        keys = list(row)
        values = [v if isinstance(v, list) else [v] for v in row.values()]
        for combination in itertools.product(*values):
            point = {**args, **dict(zip(keys, combination, strict=True))}
            if "num-prompts" not in point and isinstance(point.get("max-concurrency"), int):
                point["num-prompts"] = PROMPTS_PER_CONCURRENCY * point["max-concurrency"]
            points.append(point)
    return points


def parse_preset(raw: dict[str, Any]) -> Preset:
    """Build a :class:`Preset` from the parsed preset YAML.

    Args:
        raw: Mapping with ``kind``, ``task``, ``client``, and optionally ``server``,
            ``args`` and ``matrix``.

    Returns:
        The validated preset.

    Raises:
        ValueError: If a key is missing or unknown, or ``args``/``matrix`` set an
            option furiosa-perf manages.
    """
    known = ("kind", "task", "client", "server", "restart-server", "trace", "args", "matrix", "slo", "search")
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"the preset keys {sorted(unknown)} are unknown; use {', '.join(known)}")
    restart = raw.get("restart-server", "never")
    if restart not in RESTART_POLICIES:
        raise ValueError(
            f"the restart-server value '{restart}' is unsupported; use one of {', '.join(RESTART_POLICIES)}"
        )
    for key in ("kind", "task", "client"):
        if not raw.get(key):
            raise ValueError(f"the preset is missing '{key}'")
    if raw["kind"] not in KINDS:
        raise ValueError(f"the preset kind '{raw['kind']}' is unsupported; use one of {', '.join(KINDS)}")
    if raw["client"] not in CLIENTS:
        raise ValueError(f"the client '{raw['client']}' is unsupported; use one of {', '.join(CLIENTS)}")

    slo = raw.get("slo") or {}
    unknown_slo = set(slo) - set(SLO_KEYS)
    if unknown_slo:
        raise ValueError(f"the SLO keys {sorted(unknown_slo)} are unknown; use {', '.join(SLO_KEYS)}")

    points = expand_matrix(_kebab(raw.get("args") or {}), raw.get("matrix") or [])
    search = parse_search(raw["search"]) if raw.get("search") else None
    if search is not None:
        if not slo:
            raise ValueError("a capacity search needs an 'slo' to decide which points pass")
        if len(points) != 1 or search.key in points[0]:
            raise ValueError(
                f"with 'search', args and matrix should describe one point without '{search.key}', which the search sets"
            )
    trace = _kebab(raw.get("trace") or {})
    if trace:
        missing = [key for key in ("source", "sha256") if not trace.get(key)]
        if missing:
            raise ValueError(f"the preset trace is missing {missing}; give its source and expected sha256")
        if raw["client"] != "aiperf":
            raise ValueError("a trace is replayed with the aiperf client; set 'client: aiperf'")
    managed = sorted({key for point in points for key in point if key in MANAGED_CLIENT_ARGS[raw["client"]]})
    if managed:
        raise ValueError(f"the client options {managed} are not allowed; furiosa-perf sets them from the server config")
    return Preset(
        kind=str(raw["kind"]),
        task=str(raw["task"]),
        client=str(raw["client"]),
        server=_kebab(raw.get("server") or {}),
        points=points,
        slo={key: float(value) for key, value in slo.items()},
        search=search,
        restart_server=str(restart),
        trace=trace,
    )


def with_search_range(preset: Preset, minimum: int | None, maximum: int | None) -> Preset:
    """``preset`` with its capacity-search range replaced where a bound is given.

    Args:
        preset: Loaded preset.
        minimum: New lower end, or ``None`` to keep the preset's.
        maximum: New upper end, or ``None`` to keep the preset's.

    Returns:
        The preset, unchanged when neither bound is given.

    Raises:
        ValueError: If a bound is given for a preset without ``search``, or the range is empty.
    """
    if minimum is None and maximum is None:
        return preset
    if preset.search is None:
        raise ValueError("--search-min/--search-max need a preset with a 'search' block")
    search = replace(
        preset.search,
        min=minimum if minimum is not None else preset.search.min,
        max=maximum if maximum is not None else preset.search.max,
    )
    if search.min >= search.max:
        raise ValueError(f"the search range {search.min}..{search.max} is empty; min should be below max")
    return replace(preset, search=search)


PRESETS_DIR = Path(__file__).parent / "presets"


def resolve_preset(name_or_path: str, kind: str) -> Path:
    """Find a preset file: a path, or the name of a bundled preset of ``kind`` such as ``text``.

    Args:
        name_or_path: File path, or a name relative to the bundled presets directory of ``kind``.
        kind: Measurement type the preset is for (``baseline`` or ``scenario``).

    Returns:
        Path of the preset file.

    Raises:
        ValueError: If neither exists.
    """
    path = Path(name_or_path).expanduser()
    if path.is_file():
        return path
    bundled = PRESETS_DIR / kind / f"{name_or_path}.yaml"
    if bundled.is_file():
        return bundled
    available = sorted(p.stem for p in (PRESETS_DIR / kind).glob("*.yaml"))
    raise ValueError(
        f"the {kind} preset '{name_or_path}' does not exist; give a file path or one of {', '.join(available)}"
    )


def load_preset(name_or_path: str, kind: str) -> Preset:
    """Load and validate a preset of ``kind`` by path or bundled name.

    Args:
        name_or_path: See :func:`resolve_preset`.
        kind: Measurement type the preset must declare.

    Returns:
        The validated preset.

    Raises:
        ValueError: If the preset cannot be found, is invalid, or is of another kind.
    """
    path = resolve_preset(name_or_path, kind)
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"the preset {path} should be a mapping")
    try:
        preset = parse_preset(raw)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    if preset.kind != kind:
        raise ValueError(f"{path} is a {preset.kind} preset; run it with `furiosa-perf run {preset.kind}`")
    logger.info(f"Loaded preset {path}: {preset.kind}/{preset.task}, {len(preset.points)} points")
    return preset
