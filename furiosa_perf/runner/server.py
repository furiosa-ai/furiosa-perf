import os
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import requests
import psutil

from furiosa_perf.configs.settings import APIServerConfig, FuriosaLLMServerConfig, VllmServerConfig
from furiosa_perf.utils.logger import logger


class APIServerManager:
    def __init__(
        self,
        config: APIServerConfig,
        model: str = "furiosa-ai/Llama-3.1-8B-Instruct",
    ) -> None:
        self.model = model
        self.config = config
        self.server_process: subprocess.Popen[str] | None = None
        self.server_ready = False
        self.server_pid = -1
        self._log_file = None
        self._log_path: Path | None = None

    def __del__(self) -> None:
        self.stop()

    def start(self) -> None:
        if self._is_api_server_ready(self.model):
            logger.warning(f"API server is already running for model: {self.model}")
            self._get_opened_server_pid()
            return

        if self.server_process and self.server_process.poll() is None:
            logger.warning("Server already running")
            self._get_opened_server_pid()
            return

        command, env = self._build_command()
        if not command:
            raise ValueError("Failed to build server command.")

        logger.info(f" Starting server: {' '.join(command)}")

        log_dir = Path("./serve_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = self.model.replace("/", "_")
        self._log_path = log_dir / f"serve_{safe_model_name}_{int(time.time())}.log"
        logger.info(f"Server log: {self._log_path}")
        self._log_file = open(self._log_path, "w", buffering=1)

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
        logger.info("Server is ready")
        return " ".join(command)

    def stop(self) -> None:
        if not self.server_process:
            return

        if self.server_process.poll() is None:
            logger.info(" Stopping server process")
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
                logger.info("Server process stopped successfully ")
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                logger.warning("Server process force killed")

        else:
            exit_code = self.server_process.poll()
            logger.info(f"Server already exited with code {exit_code}")

        self.server_process = None
        self.server_ready = False
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _to_kebab_case(self, s: str) -> str:
        return s.replace("_", "-")

    def _build_command(self) -> tuple[list[str], dict[str, str]]:
        env = os.environ.copy()
        command: list[str] = []

        if isinstance(self.config, VllmServerConfig):
            command = [
                "vllm",
                "serve",
                self.model,
            ]
            # additional options for vllm

            for key, value in asdict(self.config).items():
                if value is None:
                    continue

                cli_key = self._to_kebab_case(key)
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{cli_key}"])
                elif isinstance(value, int | str | float):
                    if isinstance(value, str) and cli_key == "devices":
                        env["CUDA_VISIBLE_DEVICES"] = str(value)
                        continue
                    command.extend([f"--{cli_key}", str(value)])
                else:
                    logger.warning(f"Unsupported config type for {key}: {type(value)}")
                    continue

        elif isinstance(self.config, FuriosaLLMServerConfig):
            command = [
                "furiosa-llm",
                "serve",
                self.model,
            ]
            # additional options for furiosa-llm
            for key, value in asdict(self.config).items():
                if value is None:
                    continue

                cli_key = self._to_kebab_case(key)
                if isinstance(value, bool):
                    if value:
                        command.extend([f"--{cli_key}"])
                elif isinstance(value, int | str | float):
                    if isinstance(value, str) and cli_key == "devices":
                        # convert to furiosa-llm devices argument format. e.g. "npu:0,npu:1"
                        value = ",".join(map(lambda idx: f"npu:{idx}", value.split(",")))

                    if cli_key == "tensor-parallel-size":
                        # Furiosa NPU uses the term 'tensor-parallel-size' in terms of number of PEs in a NPU,
                        # while vllm uses it in terms of number of GPUs. Assuming each Furiosa NPU has 8 PEs,
                        # we multiply the value by 8 to align with vllm's interpretation.
                        value *= 8
                    command.extend([f"--{cli_key}", str(value)])
                else:
                    logger.warning(f"Unsupported config type for {key}: {type(value)}")
                    continue
        else:
            logger.error(f"Unsupported API Server config: {self.config}")
        return command, env

    def _is_api_server_ready(self, expected_model_id: str) -> bool:
        model_url = f"http://{self.config.host}:{self.config.port}/v1/models"
        try:
            resp = requests.get(model_url, timeout=5)
            resp.raise_for_status()
            payload = resp.json()
            model_id = payload["data"][0]["id"]
            return bool(model_id == expected_model_id)
        except (requests.RequestException, KeyError, IndexError, ValueError):
            return False

    def _wait_for_startup(self, url: str = "http://localhost:8000/v1/models", timeout: int = 600) -> None:
        start = time.time()
        log_pos = 0
        while True:
            if self._log_path and self._log_path.exists():
                with open(self._log_path) as f:
                    f.seek(log_pos)
                    for line in f:
                        logger.info(f"[server] {line.rstrip()}")
                    log_pos = f.tell()

            if self.server_process and self.server_process.poll() is not None:
                raise RuntimeError(f"Server process exited unexpectedly (code {self.server_process.poll()}). See {self._log_path}")

            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info("Server startup complete")
                    return
            except requests.RequestException as e:
                logger.debug(f"Server not ready yet: {e}")

            if time.time() - start > timeout:
                raise RuntimeError(f"Server did not start within {timeout} seconds. See {self._log_path}")

            logger.info("Waiting for server launch. Check after 30 seconds")
            time.sleep(30)

    def _get_opened_server_pid(self):
        process_name = ""
        if isinstance(self.config, VllmServerConfig):
            process_name = "vllm"
        elif isinstance(self.config, FuriosaLLMServerConfig):
            process_name = "furiosa-llm"
        else:
            return 

        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] == process_name:
                self.server_pid = proc.info['pid']