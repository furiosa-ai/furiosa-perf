"""CLI ``run`` command — orchestrates server launch, benchmark, and teardown."""

from pathlib import Path

import click

from furiosa_perf.runner.runner import BenchmarkRunner
from furiosa_perf.utils.collect_env import SystemDetector
from furiosa_perf.utils.config import APIServerConfigLoader, PerformanceBenchConfigLoader
from furiosa_perf.utils.logger import logger, setup_logger


@click.command()
@click.option(
    "--model-id",
    type=str,
    default="LGAI-EXAONE/EXAONE-4.0-32B-FP8",
    show_default=True,
    help="Local path to model directory, or HuggingFace model ID.",
)
@click.option(
    "--hardware-type",
    type=str,
    default="npu",
    show_default=True,
    help="Hardware type to benchmark (e.g. npu, gpu).",
)
@click.option(
    "--backend",
    type=str,
    default="furiosa-llm",
    show_default=True,
    help="Backend framework to use (e.g. furiosa-llm, vllm).",
)
@click.option(
    "--server-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="LLM API server launch configuration file (YAML).",
)
@click.option(
    "--benchmark-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Benchmark scenario configuration file (YAML).",
)
def run(
    model_id: str,
    hardware_type: str,
    backend: str,
    server_config: Path,
    benchmark_config: Path,
) -> None:
    """Launch an API server, run benchmark scenarios, and save results."""
    setup_logger("INFO")
    logger.info("Starting furiosa-perf")
    logger.info(f"Model:         {model_id}")
    logger.info(f"Hardware type: {hardware_type}")
    logger.info(f"Backend:       {backend}")

    system_valid, system_info = SystemDetector.check_system_compatibility(hardware_type, backend)
    if not system_valid:
        logger.error("Hardware compatibility check failed. Exiting.")
        raise SystemExit(1)

    api_server_config = APIServerConfigLoader.api_server_setup(backend, server_config)
    benchmark_config_obj = PerformanceBenchConfigLoader.benchmark_config_setup(benchmark_config)

    runner = BenchmarkRunner(system_info, hardware_type, api_server_config, benchmark_config_obj)
    logger.info("Benchmarking started.")
    runner.execute(model_id)
    logger.info("Benchmarking completed.")
