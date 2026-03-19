import click

from furiosa_perf.cli.run import run
from furiosa_perf.cli.report import report


@click.group(context_settings={"help_option_names": ["-h", "--help"]})  # type: ignore[misc]
def cli() -> None:
    """furiosa-perf: performance CLI"""
    pass


cli.add_command(run)
cli.add_command(report)

if __name__ == "__main__":
    cli()
