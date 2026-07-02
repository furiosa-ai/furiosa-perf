"""vLLM-based performance benchmark implementation."""

import json
import os
import re
import signal
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from pytablewriter import MarkdownTableWriter

from furiosa_perf.benchmark.multi_turn_input import build_multi_turn_input
from furiosa_perf.runner.monitor import HardwareMonitor
from furiosa_perf.utils.config import PerformanceBenchConfig, ScenarioConfig
from furiosa_perf.utils.logger import logger

WORKSPACE = "./bench_space"

# Extra Python deps the vendored multi-turn benchmark needs in the tool venv
# (on top of ``vllm``). Kept as version floors so uv resolves them against vllm's pins.
MULTI_TURN_DEPS = [
    "numpy>=1.24",
    "pandas>=2.0",
    "aiohttp>=3.10",
    "transformers>=4.46",
    "xlsxwriter>=3.2.1",
    "tqdm>=4.66",
]


class VllmPerformanceBenchmark:
    """Run vLLM benchmark scenarios against a running API server.

    Manages an isolated Python venv for the benchmark tool, drives one
    ``vllm bench serve`` subprocess per scenario, parses the output, and
    writes Markdown/CSV summary files.
    """

    ENV: dict[str, Any] = {
        "SETUP_COMMANDS": [
            ("create_venv", ["uv", "venv", "--python", "3.12", "{venv_dir}"]),
            ("install_package", ["uv", "pip", "install", "--python", "{python_exe}", "vllm"]),
        ],
    }

    COMMAND: list[str] = [
        "{exe}",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--percentile-metrics=ttft,tpot,itl,e2el",
        "--metric-percentiles=25,50,75,90,95,99",
        "--max-concurrency={max_concurrency}",
        "--num-prompts={num_prompts}",
        "--request-rate={request_rate}",
        "--model={model}",
        "--ignore-eos",
        "--save-result",
        "--result-dir={result_dir}",
        "--ready-check-timeout-sec=0",
        "--trust-remote-code",
    ]

    VLLM_COMMANDS: dict[str, list[str]] = {
        "offline": [
            "--backend=vllm",
            "--dataset-name=random",
            "--random-input-len={input_tokens}",
            "--random-output-len={output_tokens}",
            "--random-range-ratio={random_range_ratio}",
        ],
        "vl-offline": [
            "--backend=openai-chat",
            "--dataset-name=random-mm",
            "--random-input-len={input_tokens}",
            "--random-output-len={output_tokens}",
            "--random-range-ratio={random_range_ratio}",
            "--random-mm-base-items-per-request={random_mm_base_items_per_request}",
            "--random-mm-bucket-config={random_mm_bucket_config}",
            "--random-mm-limit-mm-per-prompt={random_mm_limit_mm_per_prompt}",
            "--endpoint=/v1/chat/completions",
        ],
        "reranker": [
            "--backend=vllm-rerank",
            "--dataset-name=random-rerank",
            "--random-input-len={input_tokens}",
            "--random-batch-size={random_batch_size}",
            "--endpoint=/v1/rerank",
        ],
        "embeddings": [
            "--backend=openai-embeddings",
            "--dataset-name=random",
            "--random-input-len={input_tokens}",
            "--endpoint=/v1/embeddings",
        ],
    }

    def __init__(
        self,
        config: PerformanceBenchConfig,
        backend: str,
        host: str | None = None,
        port: int | None = None,
        env: dict[str, Any] | None = None,
    ) -> None:
        """Initialise benchmark paths, commands, and environment.

        Args:
            config: Benchmark configuration (name, task, scenarios, …).
            backend: Serving framework name used in result-directory paths.
            host: API server hostname (forwarded to the benchmark command).
            port: API server port (forwarded to the benchmark command).
            env: Environment variables passed to the benchmark subprocess
                (must include ``HF_TOKEN`` if model download is needed).
        """
        self.name = config.name
        self.model = config.model
        self.device_name = config.device_name
        self.used_device_num = config.used_device_num
        self.task = config.task
        self.scenarios = config.scenarios
        self.backend = backend
        self.total_results: dict[str, Any] = {}
        self.host = host
        self.port = port
        self.bench_process: subprocess.Popen[str] | None = None
        self.env: dict[str, Any] = env or {}

        self.base_dir = self.get_base_dir()
        self.venv_dir = self.base_dir / f".{self.name}"
        self.python_exe = (self.venv_dir / "bin" / "python").absolute()

        # Vendored multi-turn benchmark lives inside the package; resolve its on-disk
        # location so the tool venv can run it (its modules import as flat siblings).
        self._mt_script_dir: Path | None = None
        self._mt_text_file: Path | None = None
        if self.task == "multi_turn":
            self._mt_script_dir = Path(str(resources.files("furiosa_perf.vendor.multi_turn")))
            self._mt_text_file = self._mt_script_dir / "data" / "pg1184.txt"

        self.SETUP_COMMANDS = self._build_setup_commands()
        self.BASE_COMMAND = [arg.replace("{exe}", str(self.python_exe)) for arg in self.COMMAND]
        self.log_init()

    def log_init(self) -> None:
        """Log resolved paths at startup for traceability."""
        logger.info(f"base_dir={self.base_dir}")
        logger.info(f"venv_dir={self.venv_dir}")
        logger.info(f"python_exe={self.python_exe}")
        logger.info("=" * 100)

    def _build_setup_commands(self) -> list[tuple[str, list[str]]]:
        """Expand format placeholders in ENV setup commands.

        Returns:
            A list of ``(step_name, command_argv)`` tuples ready to execute.
        """
        fmt = {
            "python_exe": self.python_exe,
            "venv_dir": str(self.venv_dir),
        }
        setup_commands = list(self.ENV["SETUP_COMMANDS"])
        if self.task == "multi_turn":
            setup_commands.append(
                ("install_multiturn_deps", ["uv", "pip", "install", "--python", "{python_exe}", *MULTI_TURN_DEPS])
            )
        return [
            (key, [arg.format(**fmt) for arg in arg_list])
            for key, arg_list in setup_commands
        ]

    def start_info(self) -> None:
        """Log a human-readable summary of what is about to run."""
        logger.info(
            f"Starting performance test — {self.name} + {self.task} | "
            f"Model: {self.model} | "
            f"Device: {self.device_name} x {self.used_device_num}"
        )

    def setup(self) -> None:
        """Create the venv and install the benchmark tool inside it."""
        logger.info(f"Setting up {self.name} benchmark environment…")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._execute_setup_commands()
        logger.info(f"{self.name} setup completed successfully.")

    def _execute_setup_commands(self) -> None:
        """Run each setup step sequentially, raising on failure."""
        for step_name, command in self.SETUP_COMMANDS:
            logger.info(f"Setup step: {step_name}")
            if step_name == "create_venv" and self.venv_dir.exists():
                logger.info(f"Skipping {step_name} — venv already exists at {self.venv_dir}.")
                continue
            if step_name == "install_package" and (self.venv_dir / "bin" / "vllm").exists():
                logger.info(f"Skipping {step_name} — vllm already installed at {self.venv_dir}.")
                continue
            if step_name == "install_multiturn_deps" and self.python_exe.exists():
                check = subprocess.run(
                    [str(self.python_exe), "-c", "import aiohttp, pandas, tqdm, transformers, xlsxwriter"],
                    capture_output=True,
                )
                if check.returncode == 0:
                    logger.info(f"Skipping {step_name} — multi-turn deps already present.")
                    continue
            p = subprocess.run(command, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"Setup step {step_name!r} failed: {p.stderr}")

    @staticmethod
    def _scenario_output_tokens(scenario: ScenarioConfig) -> int:
        """Output-token count for a scenario, or ``0`` for tasks that don't generate.

        ``reranker`` and ``embeddings`` produce no output tokens, so their scenario
        configs omit the field entirely; treat those as ``0``.
        """
        return getattr(scenario, "output_tokens", 0)

    def _scenario_result_name(self, scenario: ScenarioConfig) -> str:
        """Unique per-scenario result sub-directory name.

        The generative tasks vary output length, reranker varies batch size, and
        embeddings vary neither — so the middle component is task-dependent to keep
        every scenario's directory (and monitoring log) distinct:

        * ``offline`` / ``vl-offline`` → ``"<isl>.<osl>.<concurrency>"``
        * ``reranker`` → ``"<isl>.b<batch>.<concurrency>"``
        * ``embeddings`` → ``"<isl>.<concurrency>"``
        """
        if self.task == "multi_turn":
            # max_concurrency is reused as the client count for multi_turn.
            return f"nc{scenario.max_concurrency}.conv{scenario.num_conversations}"
        parts = [str(scenario.input_tokens)]
        if hasattr(scenario, "output_tokens"):
            parts.append(str(scenario.output_tokens))
        elif hasattr(scenario, "random_batch_size"):
            parts.append(f"b{scenario.random_batch_size}")
        parts.append(str(scenario.max_concurrency))
        return ".".join(parts)

    def _get_format_args(self, scenario: ScenarioConfig) -> dict[str, Any]:
        """Build the format-arg dict for filling command placeholders.

        Args:
            scenario: The scenario whose fields are merged into the dict.

        Returns:
            A dict suitable for ``str.format(**args)`` on command templates.
        """
        return {
            **asdict(scenario),
            "model": self.model,
            "result_dir": self.get_vllm_result_dir(self._scenario_result_name(scenario)),
        }

    def _get_command_for_scenario(self, scenario: ScenarioConfig) -> list[str]:
        """Build the full benchmark argv for a single scenario.

        Args:
            scenario: Scenario whose parameters are embedded in the command.

        Returns:
            List of strings ready to pass to :class:`subprocess.Popen`.
        """
        if self.task == "multi_turn":
            return self._get_multi_turn_command(scenario)
        fmt = self._get_format_args(scenario)
        command = [arg.format(**fmt) for arg in (self.BASE_COMMAND + self.VLLM_COMMANDS[self.task])]
        if self.host:
            command.append(f"--host={self.host}")
        if self.port:
            command.append(f"--port={self.port}")
        return command

    def _get_multi_turn_command(self, scenario: ScenarioConfig) -> list[str]:
        """Build the argv for the vendored multi-turn benchmark client.

        Assembles the ``generate_multi_turn.json`` input from the scenario, writes it
        into the scenario's result directory, and returns the command to run the
        vendored ``benchmark_serving_multi_turn.py`` in the tool venv.

        Args:
            scenario: A :class:`MultiTurnScenarioConfig` (``max_concurrency`` == the
                number of parallel clients).

        Returns:
            List of strings ready to pass to :class:`subprocess.Popen`.
        """
        assert self._mt_script_dir is not None and self._mt_text_file is not None
        result_dir = self.get_vllm_result_dir(self._scenario_result_name(scenario))
        input_path = result_dir / "generate_multi_turn.json"
        doc = build_multi_turn_input(scenario, self._mt_text_file)
        input_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

        command = [
            str(self.python_exe),
            str(self._mt_script_dir / "benchmark_serving_multi_turn.py"),
            f"--model={self.model}",
            f"--url=http://{self.host}:{self.port}",
            f"--input-file={input_path}",
            f"--num-clients={scenario.max_concurrency}",
            f"--max-active-conversations={scenario.max_active_conversations}",
            f"--request-rate={scenario.request_rate}",
            f"--seed={scenario.seed}",
            f"--warmup-percentages={scenario.warmup_percentages}",
            f"--stats-json-output={result_dir / 'stats.json'}",
            "--excel-output",
        ]
        if scenario.limit_min_tokens >= 1 and scenario.limit_max_tokens >= 1:
            command += [
                f"--limit-min-tokens={scenario.limit_min_tokens}",
                f"--limit-max-tokens={scenario.limit_max_tokens}",
            ]
        if scenario.max_num_requests is not None:
            command.append(f"--max-num-requests={scenario.max_num_requests}")
        if scenario.trust_remote_code:
            command.append("--trust-remote-code")
        return command

    def run(self) -> None:
        """Run all scenarios sequentially and store results in :attr:`total_results`."""
        cwd: Path = self.base_dir.resolve()
        self.total_results = {
            "model": self.model,
            "task": self.task,
            "device_name": self.device_name,
            "used_device_num": self.used_device_num,
            "results": [],
        }
        for scenario in self.scenarios:
            self.total_results["results"].append(self._run_scenario(scenario, cwd))

        assert len(self.total_results["results"]) > 0, "No results collected from benchmarks"
        logger.info(f"Total results collected: {len(self.total_results['results'])}")

    def _run_scenario(self, scenario: ScenarioConfig, cwd: Path) -> dict[str, Any]:
        """Execute one benchmark scenario and return the parsed result dict.

        Args:
            scenario: Scenario configuration to run.
            cwd: Working directory for the benchmark subprocess.

        Returns:
            Dict containing parsed metrics and hardware power summary.

        Raises:
            RuntimeError: If the benchmark subprocess exits with a non-zero code.
        """
        command = self._get_command_for_scenario(scenario)
        logger.info(f"Executing: {' '.join(command)} | cwd: {cwd}")

        start_timestamp = datetime.now(UTC).isoformat()
        env = os.environ.copy()
        if hf_token := self.env.get("HF_TOKEN"):
             env["HF_TOKEN"] = hf_token
        if self.task == "multi_turn":
            # The vendored scripts import each other as flat siblings, and the tool
            # writes its timestamped xlsx into cwd — run inside the result dir with the
            # vendored dir on PYTHONPATH so imports resolve and outputs land together.
            cwd = self.get_vllm_result_dir(self._scenario_result_name(scenario))
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self._mt_script_dir), env.get("PYTHONPATH", "")]))
        self.bench_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
            bufsize=1,
            start_new_session=True,
            env=env,
        )

        output_lines: list[str] = []
        if self.bench_process.stdout is not None:
            for line in self.bench_process.stdout:
                logger.info(line.strip())
                output_lines.append(line)

        self.bench_process.wait()
        if self.bench_process.returncode != 0:
            raise RuntimeError(
                f"Benchmark failed — subprocess exited with code {self.bench_process.returncode}"
            )
        self.bench_process = None
        end_timestamp = datetime.now(UTC).isoformat()

        result = self._parse_results("".join(output_lines), scenario)
        result.update(
            HardwareMonitor.get_benchmark_power_summary(
                csv_file_path=self.base_dir / f"{self.device_name}_{self.used_device_num}_monitoring_log.csv",
                start_dt=start_timestamp,
                end_dt=end_timestamp,
                target_csv_file_path=self.get_vllm_result_dir(self._scenario_result_name(scenario))
                / f"{self.device_name}_{self.used_device_num}_monitoring_log.csv",
            )
        )
        logger.info(f"Parsed result: {result}")
        return result

    def _parse_results(self, stdout: str, scenario: ScenarioConfig) -> dict[str, Any]:
        """Parse vLLM bench stdout into a structured result dict.

        Args:
            stdout: Full standard output captured from the benchmark subprocess.
            scenario: Scenario whose token counts seed the result dict.

        Returns:
            Dict of metric name → value, keyed with display-friendly names.
        """
        if self.task == "multi_turn":
            return self._parse_multi_turn_results(stdout, scenario)
        results: dict[str, Any] = {
            "Input Tokens": scenario.input_tokens,
            "Output Tokens": self._scenario_output_tokens(scenario),
            "Concurrent": scenario.max_concurrency,
        }
        for line in stdout.strip().split("\n"):
            try:
                metric, *_, score = line.split()
            except ValueError as e:
                logger.debug(f"Skipping unparseable line: {e}")
                continue
            if "Total token throughput (tok/s):" in line or "Total Token throughput (tok/s):" in line:
                results["Total Throughput(tok/s)"] = float(score)
            if "Output token throughput (tok/s):" in line:
                results["Output Throughput(tok/s)"] = float(score)
            if "TTFT" in line:
                results[f"{metric.title()} TTFT(s)"] = round(float(score) / 1000.0, 3)
            if "TPOT" in line:
                results[f"{metric.title()} TPOT(ms)"] = float(score)
            if "E2EL" in line:
                results[f"{metric.title()} E2EL(s)"] = round(float(score) / 1000.0, 2)
        return results

    def _parse_multi_turn_results(self, stdout: str, scenario: ScenarioConfig) -> dict[str, Any]:
        """Aggregate the multi-turn tool's per-request ``stats.json`` into one row.

        Reads the ``stats.json`` array written by the vendored benchmark, computes
        latency-metric percentiles, and scrapes the aggregate throughput lines from
        stdout (recomputing from timestamps if the scrape fails).

        Args:
            stdout: Captured subprocess stdout (may contain ANSI color codes).
            scenario: The multi-turn scenario just executed.

        Returns:
            Dict of column name → value for one ``multiturn_summary.csv`` row.
        """
        result_dir = self.get_vllm_result_dir(self._scenario_result_name(scenario))
        stats_path = result_dir / "stats.json"
        records: list[dict[str, Any]] = []
        if stats_path.exists():
            records = json.loads(stats_path.read_text())

        def _arr(key: str, keep_negative: bool = True) -> np.ndarray:
            vals = [r[key] for r in records if r.get(key) is not None and (keep_negative or r[key] >= 0)]
            return np.asarray(vals, dtype=float)

        ttft = _arr("ttft_ms", keep_negative=False)  # -1 marks "no first token"
        tpot = _arr("tpot_ms")
        latency = _arr("latency_ms")
        input_toks = _arr("input_num_tokens")
        output_toks = _arr("output_num_tokens")

        def _p(arr: np.ndarray, pct: float, scale: float = 1.0, ndigits: int = 3) -> float:
            return round(float(np.percentile(arr, pct)) * scale, ndigits) if arr.size else 0.0

        # Aggregate throughput is stdout-only; strip ANSI then scrape, else recompute.
        clean = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
        rps_match = re.search(r"requests per second:\s*([\d.]+)", clean)
        runtime_match = re.search(r"benchmark runtime:\s*([\d.]+)\s*sec", clean)
        runtime_sec = float(runtime_match.group(1)) if runtime_match else 0.0
        if rps_match:
            requests_per_sec = float(rps_match.group(1))
        elif records:
            starts = _arr("start_time_ms")
            span_sec = (float(starts.max()) + float(latency.max()) - float(starts.min())) / 1000.0 if starts.size else 0.0
            requests_per_sec = round(len(records) / span_sec, 3) if span_sec > 0 else 0.0
            runtime_sec = runtime_sec or round(span_sec, 3)
        else:
            requests_per_sec = 0.0

        return {
            # keys consumed by the power-summary merge / display parity
            "Input Tokens": int(round(float(input_toks.mean()))) if input_toks.size else 0,
            "Output Tokens": int(round(float(output_toks.mean()))) if output_toks.size else 0,
            "Concurrent": scenario.max_concurrency,
            # multi-turn specific columns
            "Num Clients": scenario.max_concurrency,
            "Max Active Conv": scenario.max_active_conversations,
            "Num Conversations": scenario.num_conversations,
            "Requests": len(records),
            "Requests/sec": requests_per_sec,
            "Runtime(s)": runtime_sec,
            "Mean TTFT(s)": round(float(ttft.mean()) / 1000.0, 3) if ttft.size else 0.0,
            "P50 TTFT(s)": _p(ttft, 50, scale=1 / 1000.0),
            "P90 TTFT(s)": _p(ttft, 90, scale=1 / 1000.0),
            "P99 TTFT(s)": _p(ttft, 99, scale=1 / 1000.0),
            "P50 TPOT(ms)": _p(tpot, 50),
            "P90 TPOT(ms)": _p(tpot, 90),
            "P99 TPOT(ms)": _p(tpot, 99),
            "P50 Latency(s)": _p(latency, 50, scale=1 / 1000.0),
            "P90 Latency(s)": _p(latency, 90, scale=1 / 1000.0),
            "P99 Latency(s)": _p(latency, 99, scale=1 / 1000.0),
        }

    def finish_info(self, desc: str = "") -> None:
        """Write Markdown and CSV summary files for all scenario results.

        Args:
            desc: Prefix text prepended to the auto-generated report description.
        """
        if self.task == "multi_turn":
            self._finish_info_multiturn(desc)
            return
        scenario_results = self.total_results["results"]
        report_desc = self._build_report_desc(desc)
        headers = self._prepare_result_headers(scenario_results)

        isl_osl_result: dict[str, list[Any]] = defaultdict(list)
        for sub_result in scenario_results:
            key = self._to_isl_osl_result(sub_result["Input Tokens"], sub_result["Output Tokens"])
            isl_osl_result[key].append(list(sub_result.values()))

        total_rows: list[Any] = []
        for isl_osl, rows in isl_osl_result.items():
            total_rows.extend(rows)
            self._write_summary_files(
                self.get_vllm_result_dir(""), headers, rows, report_desc,
                suffix=f"_{isl_osl}", log=True,
            )

        self._write_summary_files(self.get_vllm_result_dir(""), headers, total_rows, report_desc, suffix="")

    def _finish_info_multiturn(self, desc: str) -> None:
        """Write the multi-turn roll-up as ``multiturn_summary.csv`` / ``.md``.

        Deliberately named differently from ``summary.csv`` so ``furiosa-perf report``
        (which globs ``*/summary.csv``) never picks up multi-turn results.

        Args:
            desc: Prefix text prepended to the auto-generated report description.
        """
        scenario_results = self.total_results["results"]
        if not scenario_results:
            logger.warning("No multi_turn results to summarise.")
            return
        report_desc = self._build_report_desc(desc)
        headers = list(scenario_results[0].keys())
        rows = [list(r.values()) for r in scenario_results]

        md_writer = MarkdownTableWriter()
        md_writer.headers = headers
        md_writer.value_matrix = rows
        logger.info("\n".join([report_desc, md_writer.dumps()]))

        out_dir = self.get_vllm_result_dir("")
        csv_lines = [",".join(headers)] + [",".join(str(v) for v in row) for row in rows]
        with Path.open(out_dir / "multiturn_summary.md", "w") as f:
            f.write(report_desc + "\n" + md_writer.dumps() + "\n")
        with Path.open(out_dir / "multiturn_summary.csv", "w") as f:
            f.write(report_desc + "\n" + "\n".join(csv_lines) + "\n")

    def _build_report_desc(self, prefix: str) -> str:
        """Build the human-readable description block prepended to summary files.

        Args:
            prefix: Text to place before the auto-generated metadata lines.

        Returns:
            Multi-line description string.
        """
        return (
            f"{prefix}\n"
            f"* Performance Benchmark - {self.name} for {self.task}\n"
            f"* Model Info: {self.model}\n"
            f"* Device Info: {self.device_name} x {self.used_device_num}\n"
            f"* date: {datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}\n"
            f"* Scenario Results:"
        )

    def _prepare_result_headers(self, scenario_results: list[dict[str, Any]]) -> list[str]:
        """Optionally strip P95/P99 keys and return the column header list.

        Args:
            scenario_results: List of result dicts.

        Returns:
            List of column header strings derived from the first result dict.
        """
        return list(scenario_results[0].keys()) if scenario_results else []

    def _write_summary_files(
        self,
        result_dir: Path,
        headers: list[str],
        rows: list[Any],
        desc: str,
        suffix: str,
        log: bool = False,
    ) -> None:
        """Write a Markdown and a CSV summary file for *rows*.

        Args:
            result_dir: Directory to write ``summary<suffix>.md`` and ``.csv``.
            headers: Column header strings.
            rows: List of row value lists (one per scenario result).
            desc: Description block prepended before the table in each file.
            suffix: Appended to ``"summary"`` to form the filename (e.g. ``"_1k_1k"``).
            log: When ``True``, also emit the Markdown table to the logger.
        """
        md_writer = MarkdownTableWriter()
        md_writer.headers = headers
        md_writer.value_matrix = rows

        csv_lines = [",".join(headers)] + [",".join(str(v) for v in row) for row in rows]

        if log:
            logger.info("\n".join([desc, md_writer.dumps()]))

        with Path.open(result_dir / f"summary{suffix}.md", "w") as f:
            f.write(desc + "\n" + md_writer.dumps() + "\n")

        with Path.open(result_dir / f"summary{suffix}.csv", "w") as f:
            f.write(desc + "\n" + "\n".join(csv_lines) + "\n")

    def _to_isl_osl_result(self, input_tokens: int, output_tokens: int) -> str:
        """Return a compact ``"<ISL>_<OSL>"`` key for grouping results.

        Args:
            input_tokens: Input sequence length in tokens.
            output_tokens: Output sequence length in tokens.

        Returns:
            String like ``"1k_1k"`` or ``"512_256"``.
        """
        isl = f"{input_tokens // 1024}k" if input_tokens >= 1024 else str(input_tokens)
        osl = f"{output_tokens // 1024}k" if output_tokens >= 1024 else str(output_tokens)
        return f"{isl}_{osl}"

    def get_base_dir(self) -> Path:
        """Return (and create) the top-level workspace directory.

        Returns:
            Resolved :class:`~pathlib.Path` to ``./bench_space/``.
        """
        workspace = Path(WORKSPACE)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def get_vllm_result_dir(self, result_name: str) -> Path:
        """Return (and create) the per-scenario result directory.

        Args:
            result_name: Sub-path appended after the model directory
                (e.g. ``"1024.1024.8"`` or ``""`` for the summary root).

        Returns:
            Resolved absolute :class:`~pathlib.Path` to the result directory.
        """
        task_name = self.task
        result_dir = (
            self.base_dir
            / f"{self.device_name}_{self.used_device_num}_{self.backend}"
            / self.name
            / task_name
            / self.model.split("/")[-1]
            / result_name
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir.resolve()

    def stop(self) -> None:
        """Terminate the running benchmark subprocess if one is active."""
        if self.bench_process is None:
            return

        if self.bench_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.bench_process.pid), signal.SIGTERM)
                self.bench_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.bench_process.pid), signal.SIGKILL)

        self.bench_process = None
