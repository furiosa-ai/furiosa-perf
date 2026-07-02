import json
import shutil
from importlib import resources
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import click
import numpy as np
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
from furiosa_perf.reporting.theme import (
    SUMMARY_GPU_BACKEND,
    SUMMARY_GPU_FAMILY,
    SUMMARY_METRICS,
    SUMMARY_NPU_BACKEND,
    SUMMARY_NPU_FAMILY,
    SUMMARY_REFERENCE_RATIO,
    SUMMARY_SCENARIO_ISL_OSL,
    TABLE_COLUMNS,
    _device_family,
    build_model_color_map,
)


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
        "dataframe": total_df,
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


def _endpoint_label(hardware: str, backend: str, version: str) -> str:
    """Human-readable label for a ``(hardware, backend, version)`` endpoint."""
    parts = [p for p in (hardware, backend, version) if p]
    return " ".join(parts)


def _select_summary_scenario(task: str, tdf: pd.DataFrame, isl: int, osl: int) -> pd.DataFrame:
    """Rows for one task's summary scenario.

    Most tasks use the exact ``isl/osl`` scenario. ``vl-offline`` carries image
    tokens, so its text ISL rarely lands exactly on the nominal value — for it we
    take the largest ISL strictly below ``isl`` (same OSL) and still present it as the
    ``isl/osl`` scenario. Falls back to the exact scenario when no below-target row
    exists.

    Args:
        task (str): Task name.
        tdf (pd.DataFrame): Rows for that task.
        isl (int): Nominal input sequence length for the summary scenario.
        osl (int): Nominal output sequence length for the summary scenario.

    Returns:
        pd.DataFrame: The selected subset (possibly empty).
    """
    if task == "vl-offline":
        cand = tdf[(tdf["OSL"] == osl) & (tdf["ISL"] < isl)]
        if not cand.empty:
            target_isl = int(cand["ISL"].max())
            return tdf[(tdf["OSL"] == osl) & (tdf["ISL"] == target_isl)]
    return tdf[(tdf["ISL"] == isl) & (tdf["OSL"] == osl)]


def build_summary_data(frames: list[pd.DataFrame]) -> dict[str, Any] | None:
    """Build the compact data blob powering the interactive Overview page.

    The Overview page lets the user pick which two endpoints to compare (two RNGD
    versions, or two device+version endpoints) and computes the A/B performance
    ratios *in the browser*. So instead of pre-rendering charts server-side, this
    emits a compact JSON structure the front-end reduces on the fly.

    For the configured ISL/OSL scenario each measured value is pre-reduced per
    ``(task, model, endpoint, metric, concurrency, num_devices)`` in the metric's
    ``better`` direction (max for throughput-like, min for latency-like), with
    zero/NaN values dropped — mirroring the filtering the old Python matching logic
    did before ``report.js`` ports ``_best_per_concurrency`` /
    ``_closest_per_concurrency`` over this data.

    Args:
        frames (list[pd.DataFrame]): Per-(model, task) frames, each tagged with
            ``model`` and ``task`` columns.

    Returns:
        dict[str, Any] | None: The blob (see keys below), or ``None`` when the
        scenario produced no data.
    """
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["family"] = df["hardware"].map(_device_family)
    df["version"] = df["version"].fillna("")

    isl, osl = SUMMARY_SCENARIO_ISL_OSL

    # Per-task scenario selection (see _select_summary_scenario). Each task's rows are
    # pre-filtered to its summary scenario; vl-offline uses the largest sub-1k ISL but
    # is still presented as the 1k/1k scenario.
    task_frames: dict[str, pd.DataFrame] = {}
    for task in sorted(df["task"].dropna().unique()):
        selected = _select_summary_scenario(task, df[df["task"] == task], isl, osl)
        if not selected.empty:
            task_frames[task] = selected.copy()
    if not task_frames:
        return None

    scenario_df = pd.concat(task_frames.values(), ignore_index=True)
    models = sorted(scenario_df["model"].dropna().unique())
    color_map = build_model_color_map(models)

    # All distinct endpoints present in the scenario, plus the RNGD/furiosa-llm
    # subset used to populate the "Version comparison" dropdowns.
    endpoints: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, r in scenario_df[["hardware", "backend", "version", "family"]].drop_duplicates().iterrows():
        hw, backend, ver, family = r["hardware"], r["backend"], r["version"], r["family"]
        key = f"{hw}|{backend}|{ver}"
        if key in seen:
            continue
        seen.add(key)
        endpoints.append(
            {
                "key": key,
                "hardware": hw,
                "backend": backend,
                "version": ver,
                "family": family,
                "label": _endpoint_label(hw, backend, ver),
            }
        )
    endpoints.sort(key=lambda e: (e["family"], e["hardware"], e["backend"], e["version"]))
    version_endpoints = sorted(
        (e for e in endpoints if e["family"] == SUMMARY_NPU_FAMILY and e["backend"] == SUMMARY_NPU_BACKEND),
        key=lambda e: e["version"],
    )

    # data[task][model][endpoint_key][metric_key][concurrency][num_devices] = value
    data: dict[str, Any] = {}
    for task, tdf in task_frames.items():
        task_block: dict[str, Any] = {}
        for model in models:
            mdf = tdf[tdf["model"] == model]
            if mdf.empty:
                continue
            model_block: dict[str, Any] = {}
            for endpoint in endpoints:
                edf = mdf[
                    (mdf["hardware"] == endpoint["hardware"])
                    & (mdf["backend"] == endpoint["backend"])
                    & (mdf["version"] == endpoint["version"])
                ]
                if edf.empty:
                    continue
                metric_block: dict[str, Any] = {}
                for metric in SUMMARY_METRICS:
                    column, better = metric["column"], metric["better"]
                    if column not in edf.columns:
                        continue
                    # Drop NaN, ±inf, and zero: inf arises e.g. from TPS/Watt when a
                    # run has no power reading (throughput / 0). A single inf/NaN would
                    # otherwise serialize to `Infinity`/`NaN`, which is invalid JSON and
                    # breaks the entire Overview blob in the browser.
                    s = edf.dropna(subset=[column])
                    s = s[np.isfinite(s[column])]
                    s = s[s[column] != 0]
                    if s.empty:
                        continue
                    conc_block: dict[str, dict[str, float]] = {}
                    for (conc, ndev), grp in s.groupby(["Concurrent", "num_devices"]):
                        vals = grp[column]
                        value = float(vals.max() if better == "high" else vals.min())
                        if value == 0 or not np.isfinite(value):
                            continue
                        conc_block.setdefault(str(int(conc)), {})[str(int(ndev))] = round(value, 6)
                    if conc_block:
                        metric_block[metric["key"]] = conc_block
                if metric_block:
                    model_block[endpoint["key"]] = metric_block
            if model_block:
                task_block[model] = model_block
        if task_block:
            data[task] = task_block

    if not data:
        return None

    return {
        "scenario": f"{isl}/{osl}",
        "tasks": sorted(data),
        "metrics": [{"key": m["key"], "label": m["label"], "better": m["better"]} for m in SUMMARY_METRICS],
        "referenceRatio": SUMMARY_REFERENCE_RATIO,
        "endpoints": endpoints,
        "versionEndpoints": version_endpoints,
        "colorMap": color_map,
        "npuFamily": SUMMARY_NPU_FAMILY,
        "npuBackend": SUMMARY_NPU_BACKEND,
        "gpuFamily": SUMMARY_GPU_FAMILY,
        "gpuBackend": SUMMARY_GPU_BACKEND,
        "data": data,
    }


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
    summary_frames: list[pd.DataFrame] = []
    for model in models:
        for task in tasks:
            raw_data_files = list(Path(benchmark_result_path).rglob(f"{task}/{model}/summary.csv"))
            if len(raw_data_files) == 0:
                print(f"No summary.csv file found for {model} in {benchmark_result_path}")
                continue
            print(raw_data_files)
            task_report = collect_and_build_report_html(raw_data_files, model, task)

            frame = task_report["dataframe"].copy()
            frame["model"] = model
            frame["task"] = task
            summary_frames.append(frame)

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

    summary_data = build_summary_data(summary_frames)
    if summary_data is not None:
        # allow_nan=False: fail loudly if any non-finite value slipped through rather
        # than emit `Infinity`/`NaN` (invalid JSON that would break the Overview blob).
        blob_json = json.dumps(summary_data, allow_nan=False).replace("</", "<\\/")
        contents["summary"] = {
            "scenario": summary_data["scenario"],
            "metrics": summary_data["metrics"],
            "blob_json": blob_json,
        }
    else:
        contents["summary"] = None

    save_report_html(contents, manifest_data, out_dir)
