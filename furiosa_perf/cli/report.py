import glob
import json
import os
import shutil
from importlib import resources
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import click
import pandas as pd
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape

from furiosa_perf.reporting.charts import (
    plot_interactive_user_chart, plot_rack_performance_chart, plot_table_chart, plot_ttft_or_tpot_chart, plot_line_chart
)
from furiosa_perf.reporting.schemas import BenchmarkMetricLoader
from furiosa_perf.reporting.theme import TABLE_COLUMNS

def collect_all_models(raw_data_path: str) -> list[str]:
    base = Path(raw_data_path)
    raw_data_files = base.rglob(f"*/summary.csv")
    models = set(sorted({p.parent.name for p in raw_data_files}))
    return models

def collect_all_tasks(raw_data_path: str) -> list[str]:
    base = Path(raw_data_path)
    raw_data_files = base.rglob(f"*/summary.csv")
    tasks = set(sorted({p.parent.parent.name for p in raw_data_files}))
    return tasks


def collect_and_build_report_html(
    raw_data_files: list[str],
    target_model: str,
    task: str = "offline"
) -> dict[str, Any]:
    total_df: list[pd.DataFrame] = []
    print(target_model)
    for raw_data_file in raw_data_files:
        df = BenchmarkMetricLoader.load_offline_benchmark_metric(raw_data_file)
        total_df.append(df)

    report_charts = []

    total_df = pd.concat(total_df, ignore_index=True)
    
    s = total_df["device"].dropna()
    latest_furiosa_version = s.str.extract(r'furiosa-llm_(.*)')[0].dropna().max()
    for (input_tokens, output_tokens), group in total_df.groupby(["ISL", "OSL"]):
        tokens = f"{input_tokens}/{output_tokens}"

        pat = latest_furiosa_version
        if pd.isna(pat):
            table_group = group.iloc[0:0]
        else:
            table_group = group[group["device"].str.contains(pat, na=False)]

        report_charts.append(
            {
                "tokens": tokens,
                "html": [
                    plot_table_chart(table_group, TABLE_COLUMNS).to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config=dict(responsive=True),
                        div_id=f"{target_model}-{task}-{tokens}-rngd-{latest_furiosa_version}-table",
                    ),
                    plot_interactive_user_chart(
                        group,
                    ).to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config=dict(
                            displayModeBar=True,
                            editable=True,
                            edits={"shapePosition": False, "annotationPosition": False},
                        ),
                        div_id=f"{target_model}-{tokens}-{task}-interactive",
                    ),
                    plot_rack_performance_chart(
                        group,
                    ).to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config=dict(responsive=True),
                        div_id=f"{target_model}-{tokens}-{task}-rack",
                    ),
                    plot_ttft_or_tpot_chart(group, "ttft").to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config=dict(responsive=True),
                        div_id=f"{target_model}-{tokens}-{task}-ttft",
                    ),
                    plot_ttft_or_tpot_chart(group, "tpot").to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config=dict(responsive=True),
                        div_id=f"{target_model}-{tokens}-{task}-tpot",
                    ),
                    plot_line_chart(group, "Concurrent", "TPS/Watt").to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config=dict(responsive=True),
                        div_id=f"{target_model}-{tokens}-{task}-tps-watt",
                    ),
                ],
            }
        )

    report_contents = {
        "model": target_model,
        "title": f"{target_model} Performance Analysis",
        "content": {
            f"{task}": {
                "charts":report_charts,
                "key": "ISL / OSL"
            }
        },
        "version": latest_furiosa_version,
    }
    return report_contents


def save_report_html(report_contents: dict[str, Any], manifest_data: list[dict[str, Any]], report_path: str) -> None:
    ENV = Environment(
        loader=PackageLoader("furiosa_perf.reporting", "template"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    TEMPLATE = ENV.get_template("report.html")
    html = TEMPLATE.render(**report_contents)

    out_html = report_path / "index.html"
    out_html.write_text(html, encoding="utf-8")

    src = resources.files("furiosa_perf.reporting").joinpath("static")
    dst = report_path / "static"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    src = resources.files("furiosa_perf.reporting").joinpath("template")
    dst = report_path / "template"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    with open(f"{report_path}/manifest.json", "w") as f:
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
    )
)
@click.option(
    "--model-list",
    type=str,
    required=True,
    default = "all",
    help=(
        "The list of models to be included in the benchmark report (comma-separated list of model names)."
        " If 'all' is specified, all models will be included."
    )
)
@click.option(
    "--task-list",
    type=str,
    required=True,
    default="all",
    help=(
        "The list of tasks to be included in the benchmark report (comma-separated list of task names)."
        " If 'all' is specified, all tasks will be included."
    )
)
@click.option(
    "--report-contents",
    type=str,
    required=False,
    default="",
    help=(
        "The path of the report contents .yaml file path."
        "If not specified, the report contents will be generated from the benchmark result."
    )
)
@click.option(
    "--report-path",
    type=str,
    default="./report",
    help="The path to the output directory for the benchmark report (.html)"
)
def report(
    benchmark_result_path: str,
    model_list: str,
    task_list: str,
    report_contents: str,
    report_path: str,
) -> None:

    if model_list == "all":
        model_list = list(collect_all_models(benchmark_result_path))
    else:
        model_list = model_list.split(",")
    
    if task_list == "all":
        task_list = list(collect_all_tasks(benchmark_result_path))
    else:
        task_list = task_list.split(",")

    if not Path(f"{report_path}/csv").exists():
        os.makedirs(f"{report_path}/csv", exist_ok=True)

    out_dir = Path(report_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(out_dir / "csv", exist_ok=True)

    items = []
    manifest_data = []
    for model in model_list:
        for task in task_list:
            raw_data_files = list(Path(benchmark_result_path).rglob(f"{task}/{model}/summary.csv"))
            if len(raw_data_files) == 0:
                print(f"No summary.csv file found for {model} in {benchmark_result_path}")
                continue
            
            items.append(
                collect_and_build_report_html(
                    raw_data_files,
                    model, 
                    task
                )
            )

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
                    os.remove(path)

            manifest_data.append({"model": model, "task": task, "zip_file": zip_file_name})

    if report_contents:
        with open(report_contents, "r") as f:
            report_contents = yaml.load(f)
    else:
        report_contents = {
            "title": "FuriosaAI's RNGD with furiosa-llm Benchmark Report",
            "abstract": (
                """
                In alignment with the mass production of the RNGD chip in January 2026, we have continuously optimized the SDK end-to-end to enable fast, reliable serving of major LLMs such as EXAONE 4.0, Qwen 3, and Llama 3.3 in real-world production workloads. We fundamentally renewed the compiler architecture, moving from a whole-block compilation approach to a composable-kernel design that enables Sarathi Serve–style online scheduling, including true mixed prefill/decode batching. Concretely, the compiler factorizes execution into reusable building blocks: batch-agnostic tokenwise kernels for shared per-token compute and attention-bucket kernels for batch- and KV-cache–dependent work. These blocks can then be composed at runtime to match the current request mix and avoid prefill-driven disruption of in-flight decode. This shift was complemented by end-to-end optimization across kernels, runtime, and the serving stack. In parallel, we refined an analysis framework that translates these performance gains into actionable customer purchasing and operations decision metrics, including SLO-constrained peak concurrent user capacity, scalability under power and datacenter space constraints, and total cost of ownership (TCO) considerations.
                \n
                This report shows that SDK 2026.1 shifts competitiveness from headline throughput under matched settings to scalable, SLO-compliant capacity under realistic operating constraints. We evaluate the changes since SDK 2025.3 (July 2025) using both standard serving metrics (TTFT, TPOT, power efficiency) and an operations-oriented methodology: for each model, we sweep the feasible configuration space (device count, topology and parallelization, serving and scheduling options, and bucket settings) and compute (i) the maximum concurrency that satisfies target SLOs and (ii) the associated cost efficiency. This framing enables apples-to-apples comparisons across platforms by answering a single practical question: which system delivers the most admissible load per dollar and per rack under the service’s target SLO regime?
                """
            ),
            "model_list": model_list,
            "task_list": task_list,
            "items": items,
            "theme": "dark",
        }

    save_report_html(report_contents, manifest_data, out_dir)
    return