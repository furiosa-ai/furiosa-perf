"""Launch, health-check and stop the LLM API server process."""

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, TextIO

import psutil
import requests

from furiosa_perf.log import logger
from furiosa_perf.spec import Server

# furiosa-llm logs one line per replica with its pipeline stages, e.g.
# "DP entry DpId(1) → PP devices [TpGroupId(npu2), TpGroupId(npu3)]".
FURIOSA_REPLICA_LINE = re.compile(r"DP entry DpId\(\d+\) \S+ PP devices \[(.*?)\]")
FURIOSA_STAGE = re.compile(r"TpGroupId\(([^)]*)\)")
# Start of one launch in the appended server log (written by APIServerManager.start).
LAUNCH_MARKER = "=== "
STARTUP_TIMEOUT_S = 1800
STARTUP_POLL_S = 5
HEALTH_TIMEOUT_S = 5
LOG_TAIL_LINES = 20
# Seconds a stopping server gets to release its devices before it is killed.
STOP_TIMEOUT_S = 30
# Seconds a restart waits for the old server's port to be released.
PORT_RELEASE_TIMEOUT_S = 60
# Variables that would point the server at another Python environment than Server.env.
FOREIGN_PYTHON_VARS = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")


def option_args(key: str, value: Any) -> list[str]:
    """Render one pass-through option as command-line arguments.

    Args:
        key: Option name in kebab-case, without dashes.
        value: ``True`` for a bare flag, ``False``/``None`` to omit it, a list for
            several values, a dict for a JSON-valued option, anything else as text.

    Returns:
        The arguments to append, possibly none.
    """
    if value is None or value is False:
        return []
    if value is True:
        return [f"--{key}"]
    if isinstance(value, list):
        return [f"--{key}", *map(str, value)]
    if isinstance(value, dict):
        return [f"--{key}", json.dumps(value)]
    return [f"--{key}", str(value)]


def server_command(server: Server) -> tuple[list[str], dict[str, str]]:
    """Build the ``<backend> serve`` command line and its environment.

    The devices become ``--devices npu:<i>,...`` for furiosa-llm and
    ``CUDA_VISIBLE_DEVICES`` for vLLM. vLLM gets all three parallel sizes. furiosa-llm
    (2026.4+) gets the pipeline and data parallel sizes only: its tensor parallelism
    is fixed when the artifact is built, so ``parallel.tp`` must match the artifact
    and :func:`check_furiosa_layout` verifies the result after startup.

    Args:
        server: Server description.

    Returns:
        ``(argv, env)`` for :class:`subprocess.Popen`.
    """
    argv = [str(server.executable), "serve", server.model, "--host", server.host, "--port", str(server.port)]
    env = {key: value for key, value in os.environ.items() if key not in FOREIGN_PYTHON_VARS}
    env["PATH"] = f"{server.executable.parent}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"

    parallel = server.parallel
    if server.backend == "furiosa-llm":
        argv += ["--devices", ",".join(f"npu:{i}" for i in server.devices)]
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, server.devices))
        argv += ["--tensor-parallel-size", str(parallel.tp)]
    argv += ["--pipeline-parallel-size", str(parallel.pp), "--data-parallel-size", str(parallel.dp)]

    for key, value in server.args.items():
        argv += option_args(key, value)
    return argv, env


def furiosa_layout(log_text: str) -> list[list[set[int]]]:
    """Replicas furiosa-llm placed, from its startup log: per replica, the cards of each stage.

    A stage is a tensor-parallel group named after its cards, e.g. ``npu2`` or ``npu2-3``.

    Args:
        log_text: Server log of one launch.

    Returns:
        One list of stages per replica, each stage the set of its card indices;
        empty if the log has no placement lines.
    """
    replicas = []
    for stages in FURIOSA_REPLICA_LINE.findall(log_text):
        placed = []
        for group in FURIOSA_STAGE.findall(stages):
            bounds = [int(n) for n in re.findall(r"\d+", group)]
            placed.append(set(range(bounds[0], bounds[-1] + 1)) if bounds else set())
        replicas.append(placed)
    return replicas


def check_furiosa_layout(server: Server, log_text: str) -> None:
    """Fail if furiosa-llm placed the model differently from ``server.parallel``.

    furiosa-llm takes its tensor parallelism from the artifact and does not reject a
    mismatch: two cards with ``-pp 1 -dp 1`` and a one-card artifact silently use
    only the first card. The placement it logs is compared with the config instead.

    Raises:
        RuntimeError: If the replicas, stages or cards differ from the config.
    """
    replicas = furiosa_layout(log_text)
    if not replicas:
        logger.warning("cannot verify the parallel layout: furiosa-llm logged no placement")
        return
    parallel = server.parallel
    used = sorted(set().union(*(stage for replica in replicas for stage in replica)))
    stage_cards = {len(stage) for replica in replicas for stage in replica}
    expected = (parallel.dp, parallel.pp, {parallel.tp}, sorted(server.devices))
    actual = (len(replicas), len(replicas[0]), stage_cards, used)
    if actual != expected or any(len(replica) != parallel.pp for replica in replicas):
        tp_cards = ",".join(map(str, sorted(stage_cards)))
        raise RuntimeError(
            f"furiosa-llm placed {len(replicas)} replica(s) of {len(replicas[0])} stage(s) with {tp_cards} card(s) "
            f"each on npu {used}, not tp {parallel.tp} x pp {parallel.pp} x dp {parallel.dp} on npu "
            f"{sorted(server.devices)}; the artifact fixes the tensor parallelism, so set 'parallel' to match it"
        )


def pid_listening_on(port: int) -> int:
    """PID of the process listening on TCP ``port``, or ``-1`` if none or not visible."""
    try:
        connections = psutil.net_connections(kind="tcp")
    except psutil.Error:
        return -1
    for connection in connections:
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == port and connection.pid:
            return int(connection.pid)
    return -1


def _process_name(pid: int) -> str:
    """Name of process ``pid``, or ``"unknown"`` if it cannot be read."""
    try:
        return str(psutil.Process(pid).name())
    except psutil.Error:
        return "unknown"


class APIServerManager:
    """Manage the lifecycle of a vLLM or furiosa-llm API server subprocess."""

    def __init__(self, server: Server, log_path: Path) -> None:
        """Initialise the manager without starting the server.

        Args:
            server: What to launch.
            log_path: File the server's output is appended to.
        """
        self.server = server
        self.server_process: subprocess.Popen[str] | None = None
        self.server_pid = -1
        self._log_file: TextIO | None = None
        self._log_path = log_path

    @property
    def base_url(self) -> str:
        """URL prefix of the server's HTTP API."""
        return f"http://{self.server.host}:{self.server.port}"

    def start(self, reuse_running: bool = False) -> str:
        """Start the API server and wait until it is ready.

        Args:
            reuse_running: Use a server already listening on the port instead of
                failing. It must serve the configured model.

        Returns:
            The server command line, or ``""`` when an already running server is used.

        Raises:
            RuntimeError: If the port is taken (by a server without ``reuse_running``
                or by another program), the backend is missing from the environment,
                or the server exits or does not become ready in time.
        """
        running_model = self.served_model()
        if running_model is not None:
            return self._reuse(running_model, reuse_running)
        listener = pid_listening_on(self.server.port)
        if listener > 0:
            raise RuntimeError(
                f"port {self.server.port} is already in use by pid {listener} ({_process_name(listener)}); "
                "set another 'port' in the server config"
            )
        if not self.server.executable.exists():
            raise RuntimeError(
                f"cannot find {self.server.backend} at {self.server.executable}; "
                f"set 'env' in the server config to the environment that has {self.server.backend} installed"
            )

        argv, env = server_command(self.server)
        command = " ".join(argv)
        logger.info(f"Starting server: {command}")

        logger.info(f"Server log: {self._log_path}")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self._log_path, "a", buffering=1)  # noqa: SIM115, PTH123 - closed in stop()
        self._log_file.write(f"{LAUNCH_MARKER}{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {command}\n")

        self.server_process = subprocess.Popen(
            argv, env=env, stdout=self._log_file, stderr=self._log_file, text=True, start_new_session=True
        )
        self.server_pid = self.server_process.pid
        self._wait_for_startup()
        if self.server.backend == "furiosa-llm":
            try:
                check_furiosa_layout(self.server, self._launch_log())
            except RuntimeError:
                self.stop()
                raise
        logger.info("Server is ready.")
        return command

    def _reuse(self, running_model: str, reuse_running: bool) -> str:
        """Adopt the server already listening on the port, or refuse to."""
        port = self.server.port
        if not reuse_running:
            raise RuntimeError(
                f"port {port} already serves '{running_model}'; stop that server or pass --reuse-server to use it"
            )
        expected = str(self.server.args.get("served-model-name") or self.server.model)
        if running_model != expected:
            raise RuntimeError(f"the server on port {port} serves '{running_model}', not the configured '{expected}'")
        self.server_pid = pid_listening_on(port)
        logger.warning(f"Using the server already running on port {port} (pid {self.server_pid})")
        return ""

    def restart(self) -> str:
        """Stop the server and launch it again with the same configuration.

        Returns:
            The server command line.

        Raises:
            RuntimeError: If the old server keeps its port or the new one fails to start.
        """
        self.stop()
        deadline = time.time() + PORT_RELEASE_TIMEOUT_S
        while pid_listening_on(self.server.port) > 0:
            if time.time() > deadline:
                raise RuntimeError(
                    f"port {self.server.port} is still in use {PORT_RELEASE_TIMEOUT_S} s after the server stopped"
                )
            time.sleep(1)
        return self.start()

    def stop(self) -> None:
        """Terminate the server process group gracefully, falling back to SIGKILL."""
        if self.server_process is None:
            return
        if self.server_process.poll() is None:
            logger.info("Stopping server process…")
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=STOP_TIMEOUT_S)
                logger.info("Server process stopped successfully.")
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                logger.warning("Server process force-killed.")
        else:
            logger.info(f"Server already exited with code {self.server_process.poll()}.")

        self.server_process = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def served_model(self) -> str | None:
        """ID of the model served on the configured port, or ``None`` if nothing answers."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=HEALTH_TIMEOUT_S)
            response.raise_for_status()
            return str(response.json()["data"][0]["id"])
        except (requests.RequestException, KeyError, IndexError, ValueError):
            return None

    def model_card(self) -> dict[str, Any]:
        """The served model's ``/v1/models`` entry (furiosa-llm adds ``artifact_id``), empty if unavailable."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=HEALTH_TIMEOUT_S)
            response.raise_for_status()
            return dict(response.json()["data"][0])
        except (requests.RequestException, KeyError, IndexError, ValueError):
            return {}

    def _launch_log(self) -> str:
        """Server log of the latest launch (the log file is appended across restarts)."""
        text = self._log_path.read_text(errors="replace") if self._log_path.exists() else ""
        return text[text.rfind(LAUNCH_MARKER) :] if LAUNCH_MARKER in text else text

    def _log_tail(self) -> str:
        """Last lines of the server log, for error messages."""
        if not self._log_path.exists():
            return ""
        lines = self._log_path.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
        return "\n".join(lines)

    def _wait_for_startup(self) -> None:
        """Poll ``/v1/models`` until it answers, streaming the server log meanwhile.

        Raises:
            RuntimeError: If the server process exits or the timeout elapses.
        """
        start = time.time()
        log_pos = 0
        while True:
            if self._log_path.exists():
                with self._log_path.open() as f:
                    f.seek(log_pos)
                    for line in f:
                        logger.info(f"[server] {line.rstrip()}")
                    log_pos = f.tell()

            if self.server_process is not None and self.server_process.poll() is not None:
                raise RuntimeError(
                    f"the server exited during startup with code {self.server_process.poll()}; "
                    f"see {self._log_path}. Last lines:\n{self._log_tail()}"
                )
            if self.served_model() is not None:
                return
            if time.time() - start > STARTUP_TIMEOUT_S:
                raise RuntimeError(
                    f"the server did not become ready within {STARTUP_TIMEOUT_S} s; see {self._log_path}"
                )
            time.sleep(STARTUP_POLL_S)
