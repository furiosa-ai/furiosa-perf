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
from jinja2 import Environment, PackageLoader, select_autoescape

from furiosa_perf.reporting.charts import (
    plot_interactive_user_chart, plot_rack_performance_chart, plot_table_chart, plot_ttft_or_tpot_chart, plot_line_chart
)
from furiosa_perf.reporting.schemas import BenchmarkMetricLoader
from furiosa_perf.reporting.theme import TABLE_COLUMNS


def export_csv_bundle_zip(
    raw_data_path: str,
    target_model_list: list[str],
    task: str,
    export_path: str,
) -> None:
    zip_files = []
    manifest_data = []
    for target_model in target_model_list:
        summary_csv_files = glob.glob(f"{raw_data_path}/*/*/{task}/{target_model}/*.csv")
        if len(summary_csv_files) == 0:
            continue

        csv_bundle = []
        for summary_csv_file in summary_csv_files:
            info = summary_csv_file.split("/")[-5]
            new_name = f"{target_model}_{info}_{task}.csv"
            shutil.copy(summary_csv_file, f"{export_path}/{new_name}")
            csv_bundle.append(f"{export_path}/{new_name}")

        zip_file_name = f"{export_path}{target_model}_{task}.zip"
        zip_files.append(zip_file_name)
        with ZipFile(zip_file_name, "w", compression=ZIP_DEFLATED) as zf:
            for path in csv_bundle:
                p = Path(path)
                zf.write(p, arcname=p.name)
                os.remove(path)

        manifest_data.append({"model": target_model, "task": task, "zip_file": zip_file_name})

    with open(f"{export_path}/manifest.json", "w") as f:
        json.dump(manifest_data, f)


def collect_and_build_plotly_report_html(
    raw_data_path: str,
    target_model: str,
    version: str = "2026.1.0",
    task: str = "offline",
) -> dict[str, Any]:
    total_df: list[pd.DataFrame] = []

    summary_csv_files = glob.glob(f"{raw_data_path}/*/*/{task}/{target_model}/*.csv")
    if len(summary_csv_files) == 0:
        return {"model": target_model, "title": f"{target_model} Performance Analysis", "content": {f"{task}": []}}

    for summary_csv_file in summary_csv_files:
        df = BenchmarkMetricLoader.load_offline_benchmark_metric(summary_csv_file)
        total_df.append(df)

    total_df = pd.concat(total_df, ignore_index=True)

    report_charts = []
    for (input_tokens, output_tokens), group in total_df.groupby(["ISL", "OSL"]):
        tokens = f"{input_tokens}/{output_tokens}"
        report_charts.append(
            {
                "tokens": tokens,
                "html": [
                    plot_table_chart(group[group["device"].str.contains(version, na=False)], TABLE_COLUMNS).to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config=dict(responsive=True),
                        div_id=f"{target_model}-{task}-{tokens}-rngd-{version}-table",
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
    }

    return report_contents


@click.command()  # type: ignore[misc]
@click.option("--result-path", type=str, required=True)
@click.option(
    "--model-list",
    type=str,
    required=True,
    default="EXAONE-4.0-32B-FP8,Qwen3-32B-FP8,Llama-3.1-8B-Instruct,Llama-3.1-70B-Instruct",
)
@click.option("--output-dir", type=str, required=True, default="./")
@click.option("--version", type=str, required=True, default="2026.1.0")
def report(result_path: str, model_list: str, output_dir: str, version: str = "2026.1.0") -> None:
    # TODO: summary 로직
    model_list = model_list.split(",")

    items = []
    task_list = ["offline"]
    for model in model_list:
        for task in task_list:
            items.append(collect_and_build_plotly_report_html(result_path, model, version, task))

    context = {
        "title": "FuriosaAI's RNGD with furiosa-llm Benchmark Report",
        "description": (
            """
        This report summarizes the AI inference performance of various devices as measured in July 2025 using FuriosaAI SDK version 2025.4.
        """
        ),
        "model_list": model_list,
        "task_list": task_list,
        "items": items,
        "theme": "dark",
        "css_path": "./static/custom.css",
        "version": "2026.1.0rc0",
    }

    ENV = Environment(
        loader=PackageLoader("furiosa_bench.report", "template"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    TEMPLATE = ENV.get_template("report.html")
    html = TEMPLATE.render(**context)

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    html = TEMPLATE.render(**context)
    out_html = out_dir / "index.html"
    out_html.write_text(html, encoding="utf-8")

    # static 복사: furiosa_bench.report/_static -> out_dir/static
    src = resources.files("furiosa_bench.report").joinpath("static")
    dst = out_dir / "static"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    src = resources.files("furiosa_bench.report").joinpath("template")
    dst = out_dir / "template"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    if not (out_dir / "csv").exists():
        os.makedirs(out_dir / "csv", exist_ok=True)
    export_csv_bundle_zip(result_path, model_list, task, f"{out_dir}/csv/")

    click.echo(f"Report generated successfully in {out_dir}.")
