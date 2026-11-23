"""Entry point for the furiosa-perf CLI: the ``run``, ``env`` and ``report`` commands."""

import warnings

# requests emits RequestsDependencyWarning when urllib3 or chardet/charset-normalizer
# versions are newer than the range it was tested against. Suppress at the CLI boundary
# since the functionality is unaffected.
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

import json  # noqa: E402
import shlex  # noqa: E402
import sys  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import click  # noqa: E402

from furiosa_perf.clients.search import max_probes  # noqa: E402
from furiosa_perf.log import logger, setup_logger  # noqa: E402
from furiosa_perf.reporting.command import report  # noqa: E402
from furiosa_perf.runner import env as environment  # noqa: E402
from furiosa_perf.runner.runner import BenchmarkRunner  # noqa: E402
from furiosa_perf.runner.server import server_command  # noqa: E402
from furiosa_perf.spec import (  # noqa: E402
    Preset,
    Server,
    apply_server_overrides,
    load_preset,
    load_server,
    parse_server_args,
    with_search_range,
)


def _run_options(f: Callable[..., None]) -> Callable[..., None]:
    """Options shared by every kind of run."""
    for option in reversed(
        [
            click.argument("preset"),
            click.option(
                "--server-config",
                required=True,
                type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
                help="API server config (YAML): backend, model, devices, env, and options passed to `<backend> serve`.",
            ),
            click.option(
                "--server-arg",
                "server_args",
                multiple=True,
                metavar="KEY[=VALUE]",
                help="Server option to set on top of the server config and preset, e.g. no-enable-prefix-caching "
                "or max-num-seqs=64 (repeatable; KEY=null removes one). The run is stored as a variant next to "
                "the run without it, for comparing a feature off and on.",
            ),
            click.option(
                "--output-dir",
                default="results",
                show_default=True,
                type=click.Path(file_okay=False, path_type=Path),
                help="Root of the results tree; each run goes to <kind>/<device>/<task>/<model> under it.",
            ),
            click.option(
                "--resume",
                is_flag=True,
                help="Finish a run already in the results directory, keeping its valid points.",
            ),
            click.option(
                "--overwrite", is_flag=True, help="Delete a run already in the results directory and start over."
            ),
            click.option(
                "--reuse-server",
                is_flag=True,
                help="Benchmark a server already running on the configured port instead of launching one.",
            ),
            click.option(
                "--check",
                is_flag=True,
                help="Validate the server config and preset, print the server command and point count, then exit.",
            ),
        ]
    ):
        f = option(f)
    return f


@click.group()
def run() -> None:
    """Launch an API server, run a benchmark preset against it, and save results."""


@run.command()
@_run_options
def baseline(**options: Any) -> None:
    """Fixed-length sweep: PRESET is text, image-text, embedding, rerank, or a preset file."""
    _run("baseline", **options)


@run.command()
@_run_options
@click.option(
    "--search-min",
    type=click.IntRange(min=1),
    help="Override the lower end of the preset's capacity search (e.g. users).",
)
@click.option(
    "--search-max",
    type=click.IntRange(min=1),
    help="Override the upper end of the preset's capacity search.",
)
def scenario(search_min: int | None, search_max: int | None, **options: Any) -> None:
    """Service scenario: PRESET is chat-synthetic, chat-mooncake, or a preset file."""
    _run("scenario", search_min=search_min, search_max=search_max, **options)


def _run(
    kind: str,
    preset: str,
    server_config: Path,
    server_args: tuple[str, ...],
    output_dir: Path,
    resume: bool,
    overwrite: bool,
    reuse_server: bool,
    check: bool,
    search_min: int | None = None,
    search_max: int | None = None,
) -> None:
    setup_logger("WARNING" if check else "INFO")  # --check prints only the plan
    if resume and overwrite:
        raise click.UsageError("--resume and --overwrite are mutually exclusive")
    try:
        benchmark = with_search_range(load_preset(preset, kind), search_min, search_max)
        variant = parse_server_args(server_args)
        server = apply_server_overrides(load_server(server_config, default_env=Path(sys.prefix)), benchmark.server)
        server = apply_server_overrides(server, variant)
    except ValueError as e:
        raise click.UsageError(str(e)) from None
    if reuse_server and benchmark.restart_server == "per-point":
        raise click.UsageError("--reuse-server cannot be combined with a preset that restarts the server per point")

    if check:
        _print_check(server, benchmark)
        return

    logger.info(f"Starting furiosa-perf: {server.backend} serving {server.model} on devices {server.devices}")
    env = environment.collect(server)
    problems = environment.check(env, server)
    if problems:
        raise click.UsageError("; ".join(problems))

    runner = BenchmarkRunner(env, server, benchmark, output_dir, resume=resume, overwrite=overwrite, variant=variant)
    logger.info("Benchmarking started.")
    runner.execute(reuse_server=reuse_server)
    logger.info("Benchmarking completed.")


def _print_check(server: Server, preset: Preset) -> None:
    """Print what ``run`` would launch, for validating configs in scripts and CI."""
    argv, env = server_command(server)
    if "CUDA_VISIBLE_DEVICES" in env:
        click.echo(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} \\")
    click.echo(shlex.join(argv))
    if preset.restart_server == "per-point":
        click.echo("the server is restarted before every point")
    search = preset.search
    if search is None:
        click.echo(f"{len(preset.points)} points")
    else:
        click.echo(
            f"capacity search over {search.key} {search.min}..{search.max} (step {search.step}): "
            f"at most {max_probes(search.min, search.max, search.step)} points"
        )
    if not server.executable.exists():
        click.echo(f"warning: {server.executable} does not exist on this host", err=True)


@click.command(name="env")
@click.option(
    "--server-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Server config whose environment, devices and model are inspected.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    help="Also write the raw probe outputs to this directory (the env/ of a run).",
)
def env_command(server_config: Path, output: Path | None) -> None:
    """Show the environment a run with this server config would record, without running it."""
    setup_logger("WARNING")
    try:
        server = load_server(server_config, default_env=Path(sys.prefix))
    except ValueError as e:
        raise click.UsageError(str(e)) from None
    env = environment.collect(server)
    click.echo(json.dumps(env.summary(server.devices), indent=2, default=str))
    for problem in environment.check(env, server):
        click.echo(f"warning: {problem}", err=True)
    if output is not None:
        env.write_raw(output)
        click.echo(f"raw probe outputs written to {output}", err=True)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """furiosa-perf: LLM serving performance benchmark CLI."""


cli.add_command(run)
cli.add_command(env_command)
cli.add_command(report)

if __name__ == "__main__":
    cli()
