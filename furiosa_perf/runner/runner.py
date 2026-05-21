import multiprocessing
import os
import signal
from pathlib import Path
from typing import Any

import yaml

from furiosa_perf.benchmark.vllm import VllmPerformanceBenchmark
from furiosa_perf.configs.settings import (
    APIServerConfig,
    APIServerConfigLoader,
    PerformanceBenchConfig,
    PerformanceBenchConfigLoader,
)
from furiosa_perf.runner.monitor import HardwareMonitor
from furiosa_perf.runner.server import APIServerManager
from furiosa_perf.utils.logger import logger


class BenchmarkRunner:
    def __init__(
        self,
        system_info: dict[str, Any],
        hardware_type: str,
        save_api_log: bool = False,
        debug: bool = False,
        log_all: bool = False,
    ) -> None:
        self.save_api_log = save_api_log
        self.debug = debug
        self.log_all = log_all
        self.hardware_type = hardware_type
        self.results: list[Any] = []

        self.system_info = system_info
        self.api_server_config: APIServerConfig
        self.benchmark_config: PerformanceBenchConfig

    def api_server_setup(self, backend: str, api_server_config_path: Path) -> None:
        if not api_server_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {api_server_config_path}")

        with api_server_config_path.open() as file:
            api_server_config_data = yaml.safe_load(file)

        self.api_server_config = APIServerConfigLoader.create_config(
            backend=backend,
            configs=api_server_config_data,
        )
        logger.info(f"Loaded server config: {self.api_server_config}")

    def benchmark_config_setup(self, benchmark_config_path: Path) -> None:
        if not Path(benchmark_config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {benchmark_config_path}")

        with benchmark_config_path.open() as file:
            benchmark_config_data = yaml.safe_load(file)

        self.benchmark_config = PerformanceBenchConfigLoader.create_config(
            configs=benchmark_config_data,
        )
        logger.info(f"Loaded benchmark config: {self.benchmark_config}")

    def execute(self, model: str, full: bool, dev: bool) -> list[Any]:
        monitoring_proc = None
        stop_monitor_event = None
        api_server = None
        benchmark = None
        desc = ""

        try:
            api_server = APIServerManager(model=model, config=self.api_server_config, save_api_log=self.save_api_log)
            logger.info(f"Starting server for {model}")
            server_command = api_server.start()
            desc = (
                f"* Ubuntu Version: {self.system_info.os}\n"
                f"* Python Version: {self.system_info.python}\n"
                f"* Framework Version: {self.system_info.runtime}\n"
                f"* API Server Command: {server_command}"
            )

            self.benchmark_config.model = api_server.model
            self.benchmark_config.device_name = self.system_info.hardware[self.hardware_type]["name"]
            self.benchmark_config.used_device_num = len(self.api_server_config.devices.split(","))
            benchmark = VllmPerformanceBenchmark(
                config=self.benchmark_config,
                backend=self.system_info.runtime,
                dev=dev,
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
            logger.exception("Benchmark execution failed")
            raise
        finally:
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                if benchmark is not None:
                    try:
                        benchmark.stop()
                    except Exception:
                        logger.exception("Failed to stop benchmark")

                if monitoring_proc is not None and stop_monitor_event is not None:
                    try:
                        stop_monitor_event.set()
                        monitoring_proc.join(timeout=5)

                        if monitoring_proc.is_alive():
                            logger.warning("Monitoring process did not stop gracefully; terminating")
                            monitoring_proc.terminate()
                            monitoring_proc.join(timeout=2)

                        if monitoring_proc.is_alive():
                            logger.warning("Monitoring process still alive; killing")
                            monitoring_proc.kill()
                            monitoring_proc.join(timeout=2)
                    except Exception:
                        logger.exception("Failed to stop monitoring process")

                if api_server is not None:
                    try:
                        api_server.stop()
                    except Exception:
                        logger.exception("Failed to stop API server")

                if benchmark is not None:
                    try:
                        benchmark.finish_info(full, desc)
                    except Exception:
                        logger.exception("Failed to finalize benchmark info")

            finally:
                signal.signal(signal.SIGINT, old_sigint)

        return self.results
