import os
import time
import signal
import requests
import subprocess

from dataclasses import asdict
from furiosa_perf.configs.settings import (
    APIServerConfig, 
    FuriosaLLMServerConfig, 
    VllmServerConfig
)
from furiosa_perf.utils.logger import logger

class APIServerManager:
    def __init__(
        self,
        config: APIServerConfig,
        model: str = "furiosa-ai/EXAONE-4.0-32B-FP8",
    ) -> None:
        self.model = model
        self.config = config
        self.server_process: subprocess.Popen[str] | None = None

    def __del__(self) -> None:
        self.stop()
        
    def start(self) -> None:
        
        command, env = self._build_command()
        self.server_process = subprocess.Popen(
            command,
            env = env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        self._wait_for_startup(url=f"http://{self.config.host}:{self.config.port}/v1/models")
        self.server_ready = True
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

    def _build_command(self) -> tuple[list[str], dict[str, str]]:
        env = os.environ.copy()
        command: list[str] = []

        if isinstance(self.config, FuriosaLLMServerConfig):
            command = [
                "furiosa-llm",
                "serve",
                self.model
            ]

            for key, value in asdict(self.config).items():
                if value is None:
                    continue

                opt_param = key.replace("_", "-")
                if isinstance(value, bool) and value:
                    command.extend([f"--{opt_param}"])
                elif isinstance(value, int | str | float):
                    pass
    

    def _wait_for_startup(self, timeout: int = 600) -> None:
        start_time = time.perf_counter()
        while True:
            try:
                resp = requests.get(f"http://{self.config.host}:{self.config.port}/v1/models")
                if resp.status_code == 200:
                    logger.info("Server startup complete")
                    return
            except requests.RequestException as e:
                logger.debug(f"Server not ready yet: {e}")

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Time: server did not start within {timeout} seconds")

            logger.info("Waiting for server launch. Check after 30 seconds")
            time.sleep(30)