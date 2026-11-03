"""The ``report`` command: build the report site from result directories."""

from pathlib import Path

import click

from furiosa_perf.reporting.site import collect_runs, write_data, write_viewer


@click.command()
@click.argument("results", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    default="site",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Site directory; serve it with any static web server (e.g. python -m http.server -d site).",
)
@click.option(
    "--data-only",
    is_flag=True,
    help="Rewrite only data/ and keep the viewer and theme already in the site directory.",
)
@click.option(
    "--theme",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory with a theme.css (fonts, colours, logo) copied into the site as theme/.",
)
def report(results: tuple[Path, ...], output: Path, data_only: bool, theme: Path | None) -> None:
    """Build a static report site from RESULTS directories (results trees or bench_space)."""
    runs = collect_runs(list(results))
    if not runs:
        raise click.UsageError(f"no results found under {', '.join(map(str, results))}")
    if data_only and not (output / "index.html").exists():
        raise click.UsageError(f"--data-only needs a site in {output}; build it once without --data-only")
    if not data_only:
        write_viewer(output, theme)
    write_data(runs, output)
    points = sum(len(r["points"]) for r in runs)
    click.echo(f"{len(runs)} runs, {points} points -> {output}")
