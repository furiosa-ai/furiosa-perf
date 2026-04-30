import os
import signal
import subprocess
import time
from dataclasses import asdict

import requests
import psutil

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
        self.server_process: subprocess.Popen[str] | None = None
        self.server_proc: psutil.Process | None = None
        self.log_output = "api_server.log" if save_api_log else subprocess.DEVNULL

    def __del__(self) -> None:
        self.stop()

    def start(self) -> str:
        if self._is_api_server_ready():
            logger.warning(f"API server already running for model: {self.model}")
            self.server_proc = self._find_process_by_port(self.config.port)
            return ""

        if self.server_process and self.server_process.poll() is None:
            logger.warning("Server process already running but not yet ready")
            return ""

        command, env = self._build_command()
        if not command:
            raise ValueError("Failed to build server command.")

        logger.info(f"Starting server: {' '.join(command)}")

        self.server_process = subprocess.Popen(
            command,
            env=env,
            stdout=self.log_output,
            stderr=self.log_output,
            text=True,
            start_new_session=True,
        )

        self._wait_for_startup()
        self.server_proc = psutil.Process(self.server_process.pid)
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
        self.server_proc = None

    def _build_command(self) -> tuple[list[str], dict[str, str]]:
        env = os.environ.copy()

        if isinstance(self.config, VllmServerConfig):
            command = ["vllm", "serve", self.model]
            for key, value in asdict(self.config).items():
                if value is None:
                    continue
                cli_key = key.replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{cli_key}"])
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
                cli_key = key.replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{cli_key}"])
                elif isinstance(value, int | str | float):
                    if isinstance(value, str) and cli_key == "devices":
                        # e.g. "0,1" → "npu:0,npu:1"
                        value = ",".join(f"npu:{idx}" for idx in value.split(","))
                    if cli_key == "tensor-parallel-size":
                        # Furiosa tensor-parallel-size is per-PE; multiply by 8 PEs per NPU to match vllm semantics
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

    def _wait_for_startup(self, timeout: int = 600) -> None:
        url = f"http://{self.config.host}:{self.config.port}/v1/models"
        start = time.time()
        while True:
            try:
                if requests.get(url, timeout=5).status_code == 200:
                    return
            except requests.RequestException as e:
                logger.debug(f"Server not ready yet: {e}")

            if time.time() - start > timeout:
                raise RuntimeError(f"Server did not start within {timeout} seconds")

            logger.info("Waiting for server launch. Check after 30 seconds")
            time.sleep(30)

    def _find_process_by_port(self, port: int) -> psutil.Process | None:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.status == "LISTEN" and conn.pid:
                try:
                    return psutil.Process(conn.pid)
                except psutil.NoSuchProcess:
                    return None
        return None
