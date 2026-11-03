"""What every benchmark client shares: its pinned environment and the point loop.

A client runs one external tool per point of a preset. This module owns the parts
that do not depend on the tool: installing the tool once per host at a pinned
version, iterating the points with the monitor around each, keeping valid points
on ``--resume``, writing ``record.json`` and the summary files. A subclass says how
to build the tool's command line, read its result file, judge a point and turn the
result into a summary row.
"""

import os
import shlex
import signal
import subprocess
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pytablewriter import MarkdownTableWriter

from furiosa_perf.clients.search import SearchOutcome, bisect_capacity, max_probes
from furiosa_perf.log import logger
from furiosa_perf.results.store import RECORD_FILE, check_unique_point_names, point_name, read_json, write_json
from furiosa_perf.runner.monitor import HardwareMonitor
from furiosa_perf.spec import Preset

CLIENT_PYTHON = "3.12"
CLIENT_LOG = "bench.log"
# Start of the fail_reason clients give when the tool itself exited with an error.
CLIENT_FAILURE = "client exited"


def client_env_dir(package: str, version: str) -> Path:
    """Directory of a pinned client environment, shared by every run on this host."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "furiosa-perf" / "clients" / f"{package}-{version}"


def ensure_client_env(package: str, version: str, executable: str) -> Path:
    """Create the client environment on first use and return its root.

    Args:
        package: Distribution to install, e.g. ``"vllm"`` or ``"aiperf"``.
        version: Exact version to pin.
        executable: Entry point whose presence means the install is complete.

    Returns:
        The environment root.

    Raises:
        RuntimeError: If creating the environment or installing the package fails.
    """
    env = client_env_dir(package, version)
    if (env / "bin" / executable).exists():
        return env
    logger.info(f"Installing the benchmark client {package}=={version} into {env}")
    for step in (
        ["uv", "venv", "--python", CLIENT_PYTHON, str(env)],
        ["uv", "pip", "install", "--python", str(env / "bin" / "python"), f"{package}=={version}"],
    ):
        done = subprocess.run(step, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(
                f"failed to set up the benchmark client ({' '.join(step)}); cause: {done.stderr.strip()}"
            )
    return env


class BenchmarkClient:
    """Run every point of a preset with one external tool against a running server.

    Subclasses set the class attributes and implement :meth:`command`,
    :meth:`read_result`, :meth:`judge` and :meth:`summary_row`.

    Attributes:
        NAME: Client name as written in presets (``client:``).
        PACKAGE: Distribution installed into the client environment.
        VERSION: Pinned version of :attr:`PACKAGE`.
        EXECUTABLE: Entry point in the environment's ``bin``.
    """

    NAME: ClassVar[str]
    PACKAGE: ClassVar[str]
    VERSION: ClassVar[str]
    EXECUTABLE: ClassVar[str]

    def __init__(
        self,
        preset: Preset,
        model: str,
        host: str,
        port: int,
        run_dir: Path,
        identity: dict[str, Any],
        resume: bool = False,
        tokenizer: str | None = None,
    ) -> None:
        """Store what to run and where results go.

        Args:
            preset: Points and task of the benchmark.
            model: Model ID the server reports on ``/v1/models``.
            host: API server host.
            port: API server port.
            run_dir: Directory of this run (see :mod:`furiosa_perf.results.store`).
            identity: Fields copied into every ``record.json`` so a point is
                self-describing: kind, task, model, hardware, parallelism, versions.
            resume: Keep points whose ``record.json`` is valid instead of rerunning them.
            tokenizer: Tokenizer for clients that count tokens themselves; the
                server config's ``model`` (a Hub ID or local path).

        Raises:
            ValueError: If two points of the preset would share a directory.
        """
        if preset.search is None:
            check_unique_point_names(preset.points)
        self.preset = preset
        self.task = preset.task
        self.model = model
        self.host = host
        self.port = port
        self.run_dir = run_dir
        self.identity = identity
        self.resume = resume
        self.tokenizer = tokenizer or model
        self.env: Path | None = None
        self.process: subprocess.Popen[str] | None = None
        self.total_results: dict[str, Any] = {"results": []}
        self.search_outcome: SearchOutcome | None = None
        self.before_point: Callable[[], None] | None = None

    @property
    def python(self) -> Path | None:
        """Interpreter of the client environment, once :meth:`setup` ran."""
        return self.env / "bin" / "python" if self.env is not None else None

    def setup(self) -> None:
        """Make sure the pinned client environment exists."""
        self.env = ensure_client_env(self.PACKAGE, self.VERSION, self.EXECUTABLE)
        logger.info(f"Benchmark client: {self.PACKAGE} {self.VERSION} ({self.env})")

    # The parts a client defines.

    def command(self, point: dict[str, Any], point_dir: Path) -> list[str]:
        """The tool's command line for one point, writing its result into ``point_dir``."""
        raise NotImplementedError

    def read_result(self, point_dir: Path) -> dict[str, Any] | None:
        """The tool's result for a point, ``None`` if it wrote none."""
        raise NotImplementedError

    def clear_result(self, point_dir: Path) -> None:
        """Remove a previous run's result so it can never be read as this run's."""
        raise NotImplementedError

    def judge(self, point: dict[str, Any], result: dict[str, Any] | None, returncode: int) -> str:
        """``""`` if the point's numbers can be trusted, otherwise the reason they cannot."""
        raise NotImplementedError

    def summary_row(self, point: dict[str, Any], result: dict[str, Any] | None, fail_reason: str) -> dict[str, Any]:
        """One summary row with the same keys for every point, invalid ones included."""
        raise NotImplementedError

    def warm_up(self, point: dict[str, Any], point_dir: Path) -> None:
        """Bring the server into the state a point is measured in; nothing by default.

        Runs after :meth:`command` and outside the monitor's window.
        """

    def concurrency(self, point: dict[str, Any]) -> int:
        """Concurrency of a point, for the monitor's steady-state cut."""
        return 0

    def group_key(self, row: dict[str, Any]) -> str | None:
        """Key splitting the summary into extra per-group files, ``None`` for no split."""
        return None

    def run_summary(self) -> dict[str, Any]:
        """Run-level results added to ``run.json`` after the last point."""
        return {}

    # The shared point loop.

    def passes(self, row: dict[str, Any]) -> bool:
        """Whether a point counts as passing in a capacity search; clients with an SLO refine this."""
        return bool(row["ok"])

    def run(
        self,
        monitor: HardwareMonitor | None = None,
        server_alive: Callable[[], bool] | None = None,
        before_point: Callable[[], None] | None = None,
    ) -> None:
        """Run the preset's points, or its capacity search, and collect the summary rows.

        An invalid point does not stop the run. If the server is found dead after
        one, the run stops, because every later point would fail as well.

        Args:
            monitor: Running hardware monitor; each point is bracketed with
                :meth:`HardwareMonitor.begin` / :meth:`HardwareMonitor.end`.
            server_alive: Returns whether the server still answers; checked after an
                invalid point.
            before_point: Called before every point that is measured (not before
                points kept on resume), e.g. to restart the server.

        Raises:
            RuntimeError: If the server stopped during the run.
        """
        self.before_point = before_point
        self.total_results = {"model": self.model, "task": self.task, "results": []}
        search = self.preset.search
        if search is None:
            for point in self.preset.points:
                self._measure(point, monitor, server_alive)
        else:
            base = self.preset.points[0]
            logger.info(
                f"Capacity search over {search.key} {search.min}..{search.max} (step {search.step}), "
                f"at most {max_probes(search.min, search.max, search.step)} points"
            )
            self.search_outcome = bisect_capacity(
                search.min,
                search.max,
                search.step,
                lambda value: self._search_probe({**base, search.key: value}, monitor, server_alive),
            )
            logger.info(f"Capacity search: {search.key} = {self.search_outcome.capacity} ({self.search_outcome.bound})")

        rows = self.total_results["results"]
        logger.info(f"{len(rows)} points run, {sum(not row['ok'] for row in rows)} invalid")

    def _search_probe(
        self, point: dict[str, Any], monitor: HardwareMonitor | None, server_alive: Callable[[], bool] | None
    ) -> bool:
        """Measure one search point and say whether it passed.

        Raises:
            RuntimeError: If the client itself failed: the point says nothing about the
                server's capacity, so the search cannot go on.
        """
        row = self._measure(point, monitor, server_alive)
        if str(row.get("fail_reason", "")).startswith(CLIENT_FAILURE):
            raise RuntimeError(
                f"the benchmark client failed at point {point_name(point)} ({row['fail_reason']}); "
                f"the capacity search stops, see {self.point_dir(point_name(point)) / CLIENT_LOG}"
            )
        return self.passes(row)

    def _measure(
        self, point: dict[str, Any], monitor: HardwareMonitor | None, server_alive: Callable[[], bool] | None
    ) -> dict[str, Any]:
        """Measure one point (or keep its valid record on resume) and append its summary row.

        Raises:
            RuntimeError: If the point is invalid and the server stopped answering.
        """
        kept = self._kept_row(point)
        if kept is not None:
            logger.info(f"point {point_name(point)} is already recorded as valid; keeping it")
            self.total_results["results"].append(kept)
            return kept
        if self.before_point is not None:
            self.before_point()
        row = self._run_point(point, monitor)
        self.total_results["results"].append(row)
        if not row["ok"]:
            logger.warning(f"point {point_name(point)} is invalid: {row['fail_reason']}")
            if server_alive is not None and not server_alive():
                raise RuntimeError(f"the server stopped answering during point {point_name(point)}; the run stops here")
        return row

    def _kept_row(self, point: dict[str, Any]) -> dict[str, Any] | None:
        """With ``resume``, the summary row of a point already recorded as valid."""
        if not self.resume:
            return None
        record = read_json(self.point_dir(point_name(point)) / RECORD_FILE)
        if record is None or record.get("point") != point or not record.get("summary", {}).get("ok"):
            return None
        return dict(record["summary"])

    def _run_point(self, point: dict[str, Any], monitor: HardwareMonitor | None) -> dict[str, Any]:
        """Run one point and return its summary row (plus monitor fields)."""
        name = point_name(point)
        point_dir = self.point_dir(name)
        self.clear_result(point_dir)
        command = self.command(point, point_dir)
        self.warm_up(point, point_dir)

        started = datetime.now(timezone.utc).isoformat()
        window = monitor.begin(name) if monitor is not None else None
        returncode = self.run_logged(command, point_dir, point_dir / CLIENT_LOG)

        result = self.read_result(point_dir)
        row = self.summary_row(point, result, self.judge(point, result, returncode))
        if monitor is not None and window is not None:
            row.update(monitor.end(window, self.concurrency(point), slice_dir=point_dir))
        record = {
            **self.identity,
            "client": self.NAME,
            "client_version": self.VERSION,
            "name": name,
            "point": point,
            "command": shlex.join(command),  # as run, so the report can show how the point was measured
            "started": started,
            "finished": datetime.now(timezone.utc).isoformat(),
            "summary": row,
        }
        write_json(point_dir / RECORD_FILE, record)
        logger.info(f"Parsed result: {row}")
        return row

    def run_logged(self, command: list[str], cwd: Path, log_path: Path) -> int:
        """Run a client command, copying its output to ``log_path`` and the log; return its exit code."""
        logger.info(f"Executing: {' '.join(command)}")
        with log_path.open("w") as log:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                bufsize=1,
                start_new_session=True,
            )
            assert self.process.stdout is not None, "stdout should be a pipe"
            for line in self.process.stdout:
                log.write(line)
                logger.info(line.rstrip())
            returncode = self.process.wait()
        self.process = None
        return returncode

    def finish_info(self, desc: str = "") -> None:
        """Write the Markdown and CSV summary files, overall and per :meth:`group_key`.

        Args:
            desc: Text prepended to the generated description block.
        """
        rows = self.total_results["results"]
        if not rows:
            return
        headers = list(rows[0])
        report_desc = (
            f"{desc}\n"
            f"* Performance Benchmark - {self.preset.kind} {self.task} with {self.PACKAGE} {self.VERSION}\n"
            f"* Model Info: {self.model}\n"
            f"* Device Info: {self.identity['hardware']} x {self.identity['num_devices']} {self.identity['layout']}\n"
            f"* date: {datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}\n"
            f"* Scenario Results:"
        )
        groups: dict[str, list[list[Any]]] = defaultdict(list)
        for row in rows:
            key = self.group_key(row)
            if key is not None:
                groups[key].append(list(row.values()))
        for key, values in groups.items():
            self._write_summary_files(headers, values, report_desc, suffix=f"_{key}", log=True)
        self._write_summary_files(headers, [list(row.values()) for row in rows], report_desc, suffix="", log=not groups)

    def _write_summary_files(
        self, headers: list[str], rows: list[list[Any]], desc: str, suffix: str, log: bool
    ) -> None:
        """Write ``summary<suffix>.md`` and ``.csv`` in the run directory; empty cells for missing values."""
        md_writer = MarkdownTableWriter()
        md_writer.headers = headers
        md_writer.value_matrix = rows
        csv_lines = [",".join(headers)] + [",".join("" if v is None else str(v) for v in row) for row in rows]
        if log:
            logger.info("\n".join([desc, md_writer.dumps()]))
        (self.run_dir / f"summary{suffix}.md").write_text(desc + "\n" + md_writer.dumps() + "\n")
        (self.run_dir / f"summary{suffix}.csv").write_text(desc + "\n" + "\n".join(csv_lines) + "\n")

    def point_dir(self, name: str) -> Path:
        """Return (and create) the directory of one point inside the run directory."""
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stop(self) -> None:
        """Terminate the running client process group, if any."""
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        self.process = None
