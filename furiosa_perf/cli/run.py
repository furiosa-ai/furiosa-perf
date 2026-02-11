import click

from pathlib import Path
from furiosa_perf.runner.runner import BenchmarkRunner
from furiosa_perf.utils.logger import logger, setup_logger
from furiosa_perf.utils.collect_env import SystemDetector

@click.command()
@click.option(
    "--model",
    type=str,
    default="LGAI-EXAONE/EXAONE-4.0-32B-FP8",
    show_default=True,
    help=("MODEL Local path to model directory, or Hugging Face model id (e.g., LGAI-EXAONE/EXAONE-4.0-32B-FP8).")
)
@click.option(
    "--hardware-type",
    type=str,
    default="npu",
    show_default=True,
    help=("Hardware type to benchmark (e.g., npu, gpu)."),
)
@click.option(
    "--backend",
    type=str,
    default="furiosa-llm",
    show_default=True,
    help=("Backend framework to use for the benchmark (e.g., furiosa-llm, vllm)."),
)
@click.option(
    "--server-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help=(
        "LLM API Server launch configuration file for the benchmark. "
        "Supports YAML (.yaml/.yml) the format is inferred from the file extension."
    ),
)
@click.option(
    "--benchmark-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help=(
        "Benchmark configuration file for the benchmark. "
        "Supports YAML (.yaml/.yml) the format is inferred from the file extension."
    ),
)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Save full benchmark results (default: False).",
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help="Use furiosa-custom vllm benchmark tools instead of official vllm tools (default: False).",
)
def run(model: str, hardware_type: str, backend: str, server_config: Path, benchmark_config: Path, full: bool, dev: bool) -> None:
    setup_logger("INFO")
    logger.info("Starting FURIOSA-BENCH")
    logger.info(f"Model: {model}")
    logger.info(f"Hardware Type: {hardware_type}")
    logger.info(f"Backend: {backend}")
    logger.info("System Information")

    system_valid, system_info = SystemDetector.check_system_compatibility(hardware_type, backend)
    if not system_valid:
        logger.error("Hardware compatibility check failed. Exiting.")
        exit(1)

    runner = BenchmarkRunner(system_info, hardware_type)
    runner.api_server_setup(backend, server_config)
    runner.benchmark_config_setup(benchmark_config)
    logger.info("Benchmarking started.")
    runner.execute(model, full, dev)
    logger.info("Benchmarking completed.")