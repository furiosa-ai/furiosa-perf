"""Server lifecycle management and benchmark orchestration."""

import multiprocessing
import os
import signal
import subprocess
import time
from dataclasses import asdict
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import Any, TextIO

import psutil
import requests

from furiosa_perf.benchmark.vllm import VllmPerformanceBenchmark
from furiosa_perf.runner.monitor import HardwareMonitor
from furiosa_perf.utils.config import (
    APIServerConfig,
    FuriosaLLMServerConfig,
    PerformanceBenchConfig,
    VllmServerConfig,
)
from furiosa_perf.utils.logger import logger


class APIServerManager:
    """Manage the lifecycle of a vLLM or furiosa-llm API server subprocess.

    Handles starting, health-checking, and stopping the server process, and
    streams server logs to the logger while waiting for startup.
    """

    def __init__(
        self,
        config: APIServerConfig,
        model: str = "furiosa-ai/Llama-3.1-8B-Instruct",
    ) -> None:
        """Initialise the manager without starting the server.

        Args:
            config: Server configuration (host, port, devices, …).
            model: HuggingFace model ID or local path to serve.
        """
        self.model = model
        self.config = config
        self.server_process: subprocess.Popen[str] | None = None
        self.server_ready = False
        self.server_pid = -1
        self._log_file: TextIO | None = None
        self._log_path: Path | None = None

    def __del__(self) -> None:
        self.stop()

    def start(self) -> str:
        """Start the API server and wait until it is ready.

        Returns:
            The full server command as a space-joined string, or ``""`` if the
            server was already running.

        Raises:
            ValueError: If the command cannot be built from the config.
            RuntimeError: If the server does not become healthy within the timeout.
        """
        if self._is_api_server_ready(self.model):
            logger.warning(f"API server already running for model: {self.model}")
            self._get_opened_server_pid()
            return ""

        if self.server_process and self.server_process.poll() is None:
            logger.warning("Server already running.")
            self._get_opened_server_pid()
            return ""

        command, env = self._build_command()
        if not command:
            raise ValueError("Failed to build server command.")

        logger.info(f"Starting server: {' '.join(command)}")

        log_dir = Path("./serve_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = self.model.replace("/", "_")
        self._log_path = log_dir / f"serve_{safe_model_name}_{int(time.time())}.log"
        logger.info(f"Server log: {self._log_path}")
        self._log_file = open(self._log_path, "w", buffering=1)  # noqa: SIM115, PTH123

        env["PYTHONUNBUFFERED"] = "1"
        self.server_process = subprocess.Popen(
            command,
            env=env,
            stdout=self._log_file,
            stderr=self._log_file,
            text=True,
            start_new_session=True,
        )

        self._wait_for_startup(url=f"http://{self.config.host}:{self.config.port}/v1/models")
        self.server_ready = True
        self.server_pid = self.server_process.pid
        logger.info("Server is ready.")
        return " ".join(command)

    def stop(self) -> None:
        """Terminate the server process gracefully, falling back to SIGKILL."""
        if not self.server_process:
            return

        if self.server_process.poll() is None:
            logger.info("Stopping server process…")
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
                logger.info("Server process stopped successfully.")
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                logger.warning("Server process force-killed.")
        else:
            logger.info(f"Server already exited with code {self.server_process.poll()}.")

        self.server_process = None
        self.server_ready = False
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _to_kebab_case(self, s: str) -> str:
        """Convert an underscore-separated string to kebab-case."""
        return s.replace("_", "-")

    def _build_command(self) -> tuple[list[str], dict[str, str]]:
        """Construct the server CLI command and environment from the config.

        Returns:
            A tuple of ``(command_argv, env_dict)``.
        """
        env = os.environ.copy()
        command: list[str] = []

        if isinstance(self.config, VllmServerConfig):
            command = ["vllm", "serve", self.model]
            for key, value in asdict(self.config).items():
                if value is None:
                    continue
                cli_key = self._to_kebab_case(key)
                if isinstance(value, bool):
                    if value:
                        command.append(f"--{cli_key}")
                elif isinstance(value, int | str | float):
                    if isinstance(value, str) and cli_key == "devices":
                        env["CUDA_VISIBLE_DEVICES"] = value
                        continue
                    command.extend([f"--{cli_key}", str(value)])
                else:
                    logger.warning(f"Unsupported config type for {key}: {type(value)}")

        elif isinstance(self.config, FuriosaLLMServerConfig):
            command = ["furiosa-llm", "serve", self.model]
            for key, value in asdict(self.config).items():
                if value is None:
                    continue
                cli_key = self._to_kebab_case(key)
                if isinstance(value, bool):
                    if value:
                        command.append(f"--{cli_key}")
                elif isinstance(value, int | str | float):
                    if isinstance(value, str) and cli_key == "devices":
                        # Convert "0,1,2,3" → "npu:0,npu:1,npu:2,npu:3"
                        value = ",".join(f"npu:{idx}" for idx in value.split(","))
                    if cli_key == "tensor-parallel-size":
                        # furiosa-llm counts PEs (8 per NPU); vLLM counts NPUs.
                        value = value * 8
                    command.extend([f"--{cli_key}", str(value)])
                else:
                    logger.warning(f"Unsupported config type for {key}: {type(value)}")
        else:
            logger.error(f"Unsupported API server config type: {type(self.config)}")

        return command, env

    def _is_api_server_ready(self, expected_model_id: str) -> bool:
        """Return True when the server is already serving *expected_model_id*.

        Args:
            expected_model_id: Model ID to match against ``/v1/models`` response.
        """
        url = f"http://{self.config.host}:{self.config.port}/v1/models"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            model_id = resp.json()["data"][0]["id"]
            return model_id == expected_model_id  # type: ignore[no-any-return]
        except (requests.RequestException, KeyError, IndexError, ValueError):
            return False

    def _wait_for_startup(
        self, url: str = "http://localhost:8000/v1/models", timeout: int = 1800
    ) -> None:
        """Poll *url* until the server responds 200, streaming its log to the logger.

        Args:
            url: Health-check URL to poll.
            timeout: Maximum seconds to wait before raising.

        Raises:
            RuntimeError: If the server process exits early or the timeout elapses.
        """
        start = time.time()
        log_pos = 0

        while True:
            if self._log_path and self._log_path.exists():
                with self._log_path.open() as f:
                    f.seek(log_pos)
                    for line in f:
                        logger.info(f"[server] {line.rstrip()}")
                    log_pos = f.tell()

            if self.server_process and self.server_process.poll() is not None:
                raise RuntimeError(
                    f"Server exited unexpectedly (code {self.server_process.poll()}). "
                    f"See {self._log_path}"
                )

            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info("Server startup complete.")
                    return
            except requests.RequestException as e:
                logger.debug(f"Server not ready yet: {e}")

            if time.time() - start > timeout:
                raise RuntimeError(
                    f"Server did not start within {timeout}s. See {self._log_path}"
                )

            logger.info("Waiting for server launch — checking again in 30 s…")
            time.sleep(30)

    def _get_opened_server_pid(self) -> None:
        """Populate :attr:`server_pid` by scanning running processes by name."""
        if isinstance(self.config, VllmServerConfig):
            process_name = "vllm"
        elif isinstance(self.config, FuriosaLLMServerConfig):
            process_name = "furiosa-llm"
        else:
            return

        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] == process_name:
                self.server_pid = proc.info["pid"]


class BenchmarkRunner:
    """Orchestrate the full benchmark lifecycle: server → benchmark → teardown.

    Owns the :class:`APIServerManager` and :class:`VllmPerformanceBenchmark`
    instances and coordinates their startup, monitoring, and cleanup.
    """

    def __init__(
        self,
        system_info: Any,
        hardware_type: str,
        api_server_config: APIServerConfig,
        benchmark_config: PerformanceBenchConfig,
    ) -> None:
        """Store configuration for later use in :meth:`execute`.

        Args:
            system_info: :class:`SystemDetector.SystemEnv` namedtuple with OS/HW info.
            hardware_type: Accelerator type (``"npu"`` or ``"gpu"``).
            api_server_config: Parsed API server config dataclass.
            benchmark_config: Parsed benchmark config dataclass with scenarios.
        """
        self.system_info = system_info
        self.hardware_type = hardware_type
        self.api_server_config = api_server_config
        self.benchmark_config = benchmark_config

    def execute(self, model: str) -> list[Any]:
        """Run the full benchmark pipeline and return per-scenario results.

        Sequence:
            1. Start the API server.
            2. Query ``/v1/models`` to resolve the served model ID.
            3. Set up and run the vLLM benchmark.
            4. Start hardware monitoring alongside the benchmark.
            5. On completion or error, stop all components in reverse order.

        Args:
            model: HuggingFace model ID or local path to pass to the server.

        Returns:
            List of per-scenario result dicts from :attr:`VllmPerformanceBenchmark.total_results`.

        Raises:
            Exception: Re-raises any exception from server startup or benchmark execution.
        """
        api_server = APIServerManager(model=model, config=self.api_server_config)
        benchmark: VllmPerformanceBenchmark | None = None
        monitoring_proc: multiprocessing.Process | None = None
        stop_monitor_event: EventType | None = None
        desc = ""

        try:
            logger.info(f"Starting server for {model}")
            server_command = api_server.start()
            desc = (
                f"* Ubuntu Version: {self.system_info.os}\n"
                f"* Python Version: {self.system_info.python}\n"
                f"* Framework Version: {self.system_info.runtime}\n"
                f"* API Server Command: {server_command}"
            )

            pretrained_id = self._fetch_pretrained_id(api_server)
            self._configure_benchmark(pretrained_id)

            benchmark = VllmPerformanceBenchmark(
                config=self.benchmark_config,
                backend=self.system_info.runtime,
                host=api_server.config.host,
                port=api_server.config.port,
                env=os.environ.copy(),
            )
            benchmark.setup()

            stop_monitor_event = multiprocessing.Event()
            monitoring_proc = HardwareMonitor.start_monitor(
                api_server.config.host,
                api_server.config.port,
                api_server.server_pid,
                benchmark.device_name,
                benchmark.used_device_num,
                benchmark.base_dir,
                stop_monitor_event,
            )
            benchmark.run()

        except Exception:
            logger.exception("Benchmark execution failed.")
            raise
        finally:
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                self._cleanup(benchmark, monitoring_proc, stop_monitor_event, api_server, desc)
            finally:
                signal.signal(signal.SIGINT, old_sigint)

        return benchmark.total_results["results"]  # type: ignore[no-any-return]

    def _fetch_pretrained_id(self, api_server: APIServerManager) -> str:
        """Query ``/v1/models`` and return the first model's ID.

        Args:
            api_server: Running server manager whose host/port to query.

        Returns:
            The ``id`` field of the first model returned by the server.

        Raises:
            requests.HTTPError: If the server responds with a non-2xx status.
        """
        resp = requests.get(f"http://{api_server.config.host}:{api_server.config.port}/v1/models", timeout=5)
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]  # type: ignore[no-any-return]

    def _configure_benchmark(self, pretrained_id: str) -> None:
        """Populate benchmark_config fields resolved at runtime.

        Args:
            pretrained_id: Model ID returned by the running API server.
        """
        self.benchmark_config.model = pretrained_id
        self.benchmark_config.device_name = self.system_info.hardware[self.hardware_type]["name"]
        self.benchmark_config.used_device_num = len((self.api_server_config.devices or "").split(","))

    def _stop_monitor(
        self, proc: multiprocessing.Process, event: EventType
    ) -> None:
        """Stop the monitoring process with graceful → terminate → kill escalation.

        Args:
            proc: The monitoring :class:`multiprocessing.Process` to stop.
            event: The stop event to signal first.
        """
        event.set()
        proc.join(timeout=5)
        if proc.is_alive():
            logger.warning("Monitoring process did not stop gracefully; terminating.")
            proc.terminate()
            proc.join(timeout=2)
        if proc.is_alive():
            logger.warning("Monitoring process still alive; killing.")
            proc.kill()
            proc.join(timeout=2)

    def _cleanup(
        self,
        benchmark: VllmPerformanceBenchmark | None,
        monitoring_proc: multiprocessing.Process | None,
        stop_monitor_event: EventType | None,
        api_server: APIServerManager | None,
        desc: str,
    ) -> None:
        """Stop all components in reverse start order, logging failures individually.

        Args:
            benchmark: Benchmark instance to stop and finalise (may be ``None``).
            monitoring_proc: Monitoring subprocess to stop (may be ``None``).
            stop_monitor_event: Event used to signal the monitoring loop.
            api_server: API server to stop (may be ``None``).
            desc: Passed through to :meth:`VllmPerformanceBenchmark.finish_info`.
        """
        if benchmark is not None:
            try:
                benchmark.stop()
            except Exception:
                logger.exception("Failed to stop benchmark.")

        if monitoring_proc is not None and stop_monitor_event is not None:
            try:
                self._stop_monitor(monitoring_proc, stop_monitor_event)
            except Exception:
                logger.exception("Failed to stop monitoring process.")

        if api_server is not None:
            try:
                api_server.stop()
            except Exception:
                logger.exception("Failed to stop API server.")

        if benchmark is not None:
            try:
                benchmark.finish_info(desc)
            except Exception:
                logger.exception("Failed to finalise benchmark info.")
