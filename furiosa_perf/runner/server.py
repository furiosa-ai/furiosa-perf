import os
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import psutil
import requests

from furiosa_perf.configs.settings import APIServerConfig, FuriosaLLMServerConfig, VllmServerConfig
from furiosa_perf.utils.logger import logger


class APIServerManager:
    def __init__(
        self,
        config: APIServerConfig,
        model: str = "furiosa-ai/Llama-3.1-8B-Instruct",
        save_api_log: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.save_api_log = save_api_log
        self.server_process: subprocess.Popen[str] | None = None
        self.server_ready = False
        self.server_pid = -1
        self._log_file = None
        self._log_path: Path | None = None

    def __del__(self) -> None:
        self.stop()

    def start(self) -> str:
        if self._is_api_server_ready():
            logger.warning(f"API server is already running for model: {self.model}")
            self._get_opened_server_pid()
            return ""

        if self.server_process and self.server_process.poll() is None:
            logger.warning("Server already running")
            self._get_opened_server_pid()
            return ""

        command, env = self._build_command()
        if not command:
            raise ValueError("Failed to build server command.")

        logger.info(f"Starting server: {' '.join(command)}")

        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        if self.save_api_log:
            log_dir = Path("./serve_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_model_name = self.model.replace("/", "_")
            self._log_path = log_dir / f"serve_{safe_model_name}_{int(time.time())}.log"
            logger.info(f"Server log: {self._log_path}")
            self._log_file = self._log_path.open("w", buffering=1)  # noqa: SIM115
            stdout = self._log_file
            stderr = self._log_file
            env["PYTHONUNBUFFERED"] = "1"

        self.server_process = subprocess.Popen(  # noqa: S603
            command,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )

        self._wait_for_startup()
        self.server_ready = True
        self.server_pid = self.server_process.pid
        logger.info("Server is ready")
        return " ".join(command)

    def stop(self) -> None:
        if not self.server_process:
            return

        if self.server_process.poll() is None:
            logger.info("Stopping server process")
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
                logger.info("Server process stopped")
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                logger.warning("Server process force killed")
        else:
            logger.info(f"Server already exited with code {self.server_process.poll()}")

        self.server_process = None
        self.server_ready = False
        self.server_pid = -1
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _to_kebab_case(self, value: str) -> str:
        return value.replace("_", "-")

    def _build_command(self) -> tuple[list[str], dict[str, str]]:
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
                        value = ",".join(f"npu:{idx}" for idx in value.split(","))
                    if cli_key == "tensor-parallel-size":
                        value *= 8
                    command.extend([f"--{cli_key}", str(value)])
                else:
                    logger.warning(f"Unsupported config type for {key}: {type(value)}")
        else:
            logger.error(f"Unsupported API Server config: {self.config}")
            return [], env

        return command, env

    def _is_api_server_ready(self) -> bool:
        url = f"http://{self.config.host}:{self.config.port}/v1/models"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()["data"][0]["id"] == self.model
        except (requests.RequestException, KeyError, IndexError, ValueError):
            return False

    def _wait_for_startup(self, timeout: int = 1800) -> None:
        url = f"http://{self.config.host}:{self.config.port}/v1/models"
        start = time.time()
        log_pos = 0

        while True:
            if self._log_path and self._log_path.exists():
                with self._log_path.open() as file:
                    file.seek(log_pos)
                    for line in file:
                        logger.info(f"[server] {line.rstrip()}")
                    log_pos = file.tell()

            if self.server_process and self.server_process.poll() is not None:
                message = f"Server process exited unexpectedly (code {self.server_process.poll()})"
                if self._log_path:
                    message = f"{message}. See {self._log_path}"
                raise RuntimeError(message)

            try:
                if requests.get(url, timeout=5).status_code == 200:
                    logger.info("Server startup complete")
                    return
            except requests.RequestException as exc:
                logger.debug(f"Server not ready yet: {exc}")

            if time.time() - start > timeout:
                message = f"Server did not start within {timeout} seconds"
                if self._log_path:
                    message = f"{message}. See {self._log_path}"
                raise RuntimeError(message)

            logger.info("Waiting for server launch. Check after 30 seconds")
            time.sleep(30)

    def _get_opened_server_pid(self) -> None:
        process_name = ""
        if isinstance(self.config, VllmServerConfig):
            process_name = "vllm"
        elif isinstance(self.config, FuriosaLLMServerConfig):
            process_name = "furiosa-llm"
        else:
            return

        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] == process_name:
                self.server_pid = proc.info["pid"]
                return
