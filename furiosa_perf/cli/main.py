"""Entry point for the furiosa-perf CLI."""

import warnings

# requests emits RequestsDependencyWarning when urllib3 or chardet/charset-normalizer
# versions are newer than the range it was tested against. Suppress at the CLI boundary
# since the functionality is unaffected.
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

import click

from furiosa_perf.cli.run import run


@click.group(context_settings={"help_option_names": ["-h", "--help"]})  # type: ignore[misc]
def cli() -> None:
    """furiosa-perf: LLM serving performance benchmark CLI."""


cli.add_command(run)

if __name__ == "__main__":
    cli()
