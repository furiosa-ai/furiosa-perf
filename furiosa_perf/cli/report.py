import json
import shutil
from importlib import resources
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import click
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape

from furiosa_perf.reporting.charts import (
    plot_interactive_user_chart,
    plot_line_chart,
    plot_rack_performance_chart,
    plot_table_chart,
    plot_ttft_or_tpot_chart,
)
from furiosa_perf.reporting.schemas import BenchmarkMetricLoader
from furiosa_perf.reporting.theme import TABLE_COLUMNS


def fig_to_lazy_html(fig: go.Figure, div_id: str, config: dict[str, Any] | None = None) -> str:
    """Serialize a Plotly figure into a placeholder div + an inert JSON spec.

    Unlike ``fig.to_html()`` — which emits a ``<script>`` that calls ``Plotly.newPlot``
    immediately on page load — this embeds the figure spec inside a
    ``<script type="application/json">`` tag that the browser does *not* execute.
    The front-end (report.js) renders each chart lazily, only when its group becomes
    visible, so a report with hundreds of charts no longer blocks on rendering all of
    them up front.

    Args:
        fig (go.Figure): The Plotly figure to serialize.
        div_id (str): HTML id for the placeholder div and (suffixed ``-spec``) its
            JSON spec.
        config (dict[str, Any] | None): Optional Plotly config dict passed to
            ``Plotly.newPlot``.

    Returns:
        str: An HTML fragment — the placeholder ``<div>`` followed by its inert JSON
        ``<script>`` spec.
    """
    spec = json.loads(pio.to_json(fig))  # {"data": ..., "layout": ...}, numpy-safe
    spec["config"] = config or {}
    payload = json.dumps(spec).replace("</", "<\\/")  # keep </script> from closing the tag
    return (
        f'<div id="{div_id}" class="js-plotly-plot lazy-plot"></div>'
        f'<script type="application/json" id="{div_id}-spec">{payload}</script>'
    )


def collect_all_models(raw_data_path: str) -> list[str]:
    base = Path(raw_data_path)
    raw_data_files = base.rglob("*/summary.csv")
    return sorted({p.parent.name for p in raw_data_files})


def collect_all_tasks(raw_data_path: str) -> list[str]:
    base = Path(raw_data_path)
    raw_data_files = base.rglob("*/summary.csv")
    return sorted({p.parent.parent.name for p in raw_data_files})


def collect_and_build_report_html(
    raw_data_files: list[Path], target_model: str, task: str = "offline"
) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    print(target_model)
    for raw_data_file in raw_data_files:
        df = BenchmarkMetricLoader.load_offline_benchmark_metric(raw_data_file)
        frames.append(df)

    report_charts = []

    total_df = pd.concat(frames, ignore_index=True)

    s = total_df["device"].dropna()
    latest_furiosa_version = s.str.extract(r"furiosa-llm_(.*)")[0].dropna().max()
    for (input_tokens, output_tokens), group in total_df.groupby(["ISL", "OSL"]):
        tokens = f"{input_tokens}/{output_tokens}"

        pat = latest_furiosa_version
        table_group = group.iloc[0:0] if pd.isna(pat) else group[group["device"].str.contains(pat, na=False)]

        report_charts.append(
            {
                "tokens": tokens,
                "html": [
                    fig_to_lazy_html(
                        plot_table_chart(table_group, TABLE_COLUMNS),
                        f"{target_model}-{task}-{tokens}-rngd-{latest_furiosa_version}-table",
                        config={"responsive": True},
                    ),
                    fig_to_lazy_html(
                        plot_interactive_user_chart(group),
                        f"{target_model}-{tokens}-{task}-interactive",
                        config={
                            "displayModeBar": True,
                            "editable": True,
                            "edits": {"shapePosition": False, "annotationPosition": False},
                        },
                    ),
                    fig_to_lazy_html(
                        plot_rack_performance_chart(group),
                        f"{target_model}-{tokens}-{task}-rack",
                        config={"responsive": True},
                    ),
                    fig_to_lazy_html(
                        plot_ttft_or_tpot_chart(group, "ttft"),
                        f"{target_model}-{tokens}-{task}-ttft",
                        config={"responsive": True},
                    ),
                    fig_to_lazy_html(
                        plot_ttft_or_tpot_chart(group, "tpot"),
                        f"{target_model}-{tokens}-{task}-tpot",
                        config={"responsive": True},
                    ),
                    fig_to_lazy_html(
                        plot_line_chart(group, "Concurrent", "TPS/Watt"),
                        f"{target_model}-{tokens}-{task}-tps-watt",
                        config={"responsive": True},
                    ),
                ],
            }
        )

    report_contents = {
        "model": target_model,
        "title": f"{target_model} Performance Analysis",
        "content": {f"{task}": {"charts": report_charts, "key": "ISL / OSL"}},
        "version": latest_furiosa_version,
    }
    return report_contents


def save_report_html(report_contents: dict[str, Any], manifest_data: list[dict[str, Any]], report_path: Path) -> None:
    ENV = Environment(
        loader=PackageLoader("furiosa_perf.reporting", "template"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    TEMPLATE = ENV.get_template("report.html")
    html = TEMPLATE.render(**report_contents)

    out_html = report_path / "index.html"
    out_html.write_text(html, encoding="utf-8")

    for asset in ("static", "template"):
        src = Path(str(resources.files("furiosa_perf.reporting").joinpath(asset)))
        dst = report_path / asset
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    with (report_path / "manifest.json").open("w") as f:
        json.dump(manifest_data, f)
    click.echo(f"Report generated successfully in {report_path}.")


@click.command()
@click.option(
    "--benchmark-result-path",
    type=str,
    required=True,
    help=(
        "The path to the result directory for the benchmark. "
        "We currently support the topology-based result directory."
        "e.g., <benchmark_result_path>/vllm/<hardware_name>_<used_number_of_devices>_<runtime_version>/<tool_name>/<task_name>/<model_name>/*.csv"
    ),
)
@click.option(
    "--model-list",
    type=str,
    required=True,
    default="all",
    help=(
        "The list of models to be included in the benchmark report (comma-separated list of model names)."
        " If 'all' is specified, all models will be included."
    ),
)
@click.option(
    "--task-list",
    type=str,
    required=True,
    default="all",
    help=(
        "The list of tasks to be included in the benchmark report (comma-separated list of task names)."
        " If 'all' is specified, all tasks will be included."
    ),
)
@click.option(
    "--report-contents",
    type=str,
    required=False,
    default="",
    help=(
        "The path of the report contents .yaml file path."
        "If not specified, the report contents will be generated from the benchmark result."
    ),
)
@click.option(
    "--report-path",
    type=str,
    default="./report",
    help="The path to the output directory for the benchmark report (.html)",
)
def report(
    benchmark_result_path: str,
    model_list: str,
    task_list: str,
    report_contents: str,
    report_path: str,
) -> None:
    models = collect_all_models(benchmark_result_path) if model_list == "all" else model_list.split(",")

    tasks = collect_all_tasks(benchmark_result_path) if task_list == "all" else task_list.split(",")

    out_dir = Path(report_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)

    model_items: dict[str, dict[str, Any]] = {}
    manifest_data = []
    for model in models:
        for task in tasks:
            raw_data_files = list(Path(benchmark_result_path).rglob(f"{task}/{model}/summary.csv"))
            if len(raw_data_files) == 0:
                print(f"No summary.csv file found for {model} in {benchmark_result_path}")
                continue
            print(raw_data_files)
            task_report = collect_and_build_report_html(raw_data_files, model, task)

            if model not in model_items:
                model_items[model] = {
                    "model": model,
                    "title": task_report["title"],
                    "content": {},
                    "version": task_report["version"],
                }
            model_items[model]["content"][task] = task_report["content"][task]

            csv_bundle = []
            for raw_data_file in raw_data_files:
                new_file_name = f"{model}_{raw_data_file.parents[3].name}_{task}.csv"
                shutil.copy(raw_data_file, f"{out_dir / 'csv'}/{new_file_name}")
                csv_bundle.append(f"{out_dir / 'csv'}/{new_file_name}")

            zip_file_name = f"{out_dir}/csv/{model}_{task}.zip"
            with ZipFile(zip_file_name, "w", compression=ZIP_DEFLATED) as zf:
                for path in csv_bundle:
                    p = Path(path)
                    zf.write(p, arcname=p.name)
                    p.unlink()

            manifest_data.append({"model": model, "task": task, "zip_file": zip_file_name})

    items = list(model_items.values())

    contents: dict[str, Any]
    if report_contents:
        with Path(report_contents).open() as f:
            contents = yaml.safe_load(f)
    else:
        contents = {
            "title": "FuriosaAI's RNGD with furiosa-llm Benchmark Report",
            "abstract": (
                """
                In alignment with the mass production of the RNGD chip in January 2026, we have continuously optimized the SDK end-to-end to enable fast, reliable serving of major LLMs such as EXAONE 4.0, Qwen 3, and Llama 3.3 in real-world production workloads. We fundamentally renewed the compiler architecture, moving from a whole-block compilation approach to a composable-kernel design that enables Sarathi Serve–style online scheduling, including true mixed prefill/decode batching. Concretely, the compiler factorizes execution into reusable building blocks: batch-agnostic tokenwise kernels for shared per-token compute and attention-bucket kernels for batch- and KV-cache–dependent work. These blocks can then be composed at runtime to match the current request mix and avoid prefill-driven disruption of in-flight decode. This shift was complemented by end-to-end optimization across kernels, runtime, and the serving stack. In parallel, we refined an analysis framework that translates these performance gains into actionable customer purchasing and operations decision metrics, including SLO-constrained peak concurrent user capacity, scalability under power and datacenter space constraints, and total cost of ownership (TCO) considerations.
                \n
                This report shows that SDK 2026.1 shifts competitiveness from headline throughput under matched settings to scalable, SLO-compliant capacity under realistic operating constraints. We evaluate the changes since SDK 2025.3 (July 2025) using both standard serving metrics (TTFT, TPOT, power efficiency) and an operations-oriented methodology: for each model, we sweep the feasible configuration space (device count, topology and parallelization, serving and scheduling options, and bucket settings) and compute (i) the maximum concurrency that satisfies target SLOs and (ii) the associated cost efficiency. This framing enables apples-to-apples comparisons across platforms by answering a single practical question: which system delivers the most admissible load per dollar and per rack under the service’s target SLO regime?
                """
            ),
            "model_list": models,
            "task_list": tasks,
            "items": items,
            "theme": "dark",
        }

    save_report_html(contents, manifest_data, out_dir)
