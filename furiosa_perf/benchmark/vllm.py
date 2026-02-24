import os
import venv
import subprocess

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from pathlib import Path
from pytablewriter import MarkdownTableWriter
from furiosa_perf.utils.logger import logger
from furiosa_perf.runner.monitor import HardwareMonitor
from furiosa_perf.configs.settings import PerformanceBenchConfig, ScenarioConfig

WORKSPACE = "./bench_space"

class VllmPerformanceBenchmark:
    ENV = {
        "dev": {
            "REPO_URL": "https://github.com/furiosa-ai/vllm.git",
            "BRANCH": "furiosa-custom-0.13",
            "REPO_NAME": "custom-vllm",
            "SETUP_COMMANDS": [
                ("down_benchmark", ["git", "clone", "{REPO_URL}", "-b", "{BRANCH}", "{REPO_NAME}"]),
                ("install_package", ['{python_exe}', '-m', 'pip', 'install', '-r', '{REPO_NAME}/requirements/common.txt'])
            ]
        },
        'official': {
            "REPO_URL": "https://github.com/vllm-project/vllm.git",
            "BRANCH": "releases/v0.13.0",
            "REPO_NAME": "official-vllm",
            "SETUP_COMMANDS": [
                ("down_benchmark", ["git", "clone", "{REPO_URL}", "-b", "{BRANCH}", "{REPO_NAME}"]),
                ("install_package", ['{python_exe}', '-m', 'pip', 'install', 'vllm==0.13.0'])
            ]
        }
    }
    COMMAND: list[str] = [
        "{exe}", "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--percentile-metrics=ttft,tpot,itl,e2el",
        "--metric-percentiles=25,50,75,90,95,99",
        "--max-concurrency={max_concurrency}",
        "--num-prompts={num_prompts}",
        "--request-rate={request_rate}",
        "--model={model}",
        "--ignore-eos",
        "--save-result",
        "--result-dir={result_dir}",
        "--ready-check-timeout-sec=0"
    ]
    VLLM_COMMANDS: dict[str, list[str]] = {
        "offline": [
            "--backend=vllm",
            "--dataset-name=random",
            "--random-input-len={input_tokens}",
            "--random-output-len={output_tokens}",
            "--random-range-ratio={random_range_ratio}",
        ],
        "reranker": [
            "--bakcend=vllm-rerank",
            "--endpoint=/v1/rerank",
            "--dataset-name=random-rerank",
        ],
        "embeddings": [],
        "prefix-cache": [],
    }

    def __init__(
        self,
        config: PerformanceBenchConfig,
        backend: str,
        host: str | None = None,
        port: int | None = None,
        dev: bool = False,
    ) -> None:
        self.name = config.name
        self.model = config.model
        self.device_name = config.device_name
        self.used_device_num = config.used_device_num
        self.task = config.task
        self.scenarios = config.scenarios
        self.dev = 'dev' if dev else 'official'
        self.backend = backend
        self.total_results: dict[str, Any] = {}
        self.host = host
        self.port = port

        self.branch = self.ENV[self.dev]['BRANCH']
        self.repo_name = self.ENV[self.dev]['REPO_NAME']

        self.base_dir =  self.get_base_dir()
        self.repo_dir = self.base_dir / self.repo_name
        self.venv_dir = self.base_dir / f"{config.name}_venv"
        self.python_exe = (self.venv_dir / "bin" / "python").absolute()

        self.SETUP_COMMANDS = [
            (key, [arg.format(
                REPO_URL=self.ENV[self.dev]['REPO_URL'], 
                BRANCH=self.branch, 
                REPO_NAME=self.ENV[self.dev]['REPO_NAME'], 
                python_exe=self.python_exe) for arg in arg_list])
            for key, arg_list in self.ENV[self.dev]['SETUP_COMMANDS']
        ]

        self.BASE_COMMAND = [
            arg.replace("{exe}", f"{str(self.python_exe)}")
            for arg in self.COMMAND
        ]
        self.log_init()

    def start_info(self) -> None:
        logger.info(
            f"Starting performance test - {self.name} + {self.task} | "
            f"Model: {self.model} | "
            f"Device: {self.device_name} x {self.used_device_num} | "
        )

    def log_init(self) -> None:
        logger.info(f"base_dir={self.base_dir}")
        logger.info(f"repo_dir={self.repo_dir}")
        logger.info(f"venv_dir={self.venv_dir}")
        logger.info(f"python_exe={self.python_exe}")
        logger.info("=" * 100)

    def setup(self) -> None:
        """Setup the benchmark environment."""
        logger.info(f"Setting up {self.name} benchmark environment...")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._execute_setup_commands()
        logger.info(f"{self.name} setup completed successfully")

    def _execute_setup_commands(self) -> None:
        cwd = self.base_dir
        venv.create(self.venv_dir, with_pip=True)

        for step_name, command in self.SETUP_COMMANDS:
            logger.info(f"Setup step : {step_name}")
            if self.repo_dir.exists() and step_name == "down_benchmark":
                logger.info(f"Skipping {step_name}, {self.repo_name} already exists.")
                continue
            p = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"Setup step: {step_name} failed")

    def _get_format_args(self, scenario: ScenarioConfig) -> dict[str, Any]:
        return {
            "model": self.model,
            **asdict(scenario),
            "result_dir": self.get_vllm_result_dir(
                f"{scenario.input_tokens}.{scenario.output_tokens}.{scenario.max_concurrency}"
            ),
        }

    def _get_command_for_scenario(self, scenario: ScenarioConfig) -> list[str]:
        format_args = self._get_format_args(scenario)

        command = [arg.format(**format_args) for arg in (self.BASE_COMMAND + self.VLLM_COMMANDS[self.task])]
        if self.host:
            command.append(f"--host={self.host}")
        if self.port:
            command.append(f"--port={self.port}")
        return command

    def run(self) -> None:
        cwd: Path = self.repo_dir.resolve()

        self.total_results = {
            "model": self.model,
            "task": self.task,
            "device_name": self.device_name,
            "used_device_num": self.used_device_num,
            "results": [],
        }

        for scenario in self.scenarios:
            command = self._get_command_for_scenario(scenario)
            logger.info(f"Executing: {' '.join(command)} | cwd: {cwd}")

            start_timestamp = datetime.now(timezone.utc).isoformat()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                bufsize=1,
            )
            output_lines = []
            if process.stdout is not None:
                for line in process.stdout:
                    logger.info(line.strip())
                    output_lines.append(line)

            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"Benchmark execution failed - Command failed with code {process.returncode}")

            end_timestamp = datetime.now(timezone.utc).isoformat()
            result = self._parse_results("".join(output_lines), scenario)   

            result.update(
                HardwareMonitor.get_benchmark_power_summary(
                    csv_file_path=os.path.join(
                        self.base_dir, f"{self.device_name}_{self.used_device_num}_monitoring_log.csv"
                    ),
                    start_dt=start_timestamp,
                    end_dt=end_timestamp,
                    target_csv_file_path=os.path.join(
                        self.get_vllm_result_dir(
                            f"{scenario.input_tokens}.{scenario.output_tokens}.{scenario.max_concurrency}"
                        ),
                        f"{self.device_name}_{self.used_device_num}_monitoring_log.csv",
                    ),
                )
            )

            logger.info(f"Parsed result: {result}")
            self.total_results["results"].append(result)

        assert len(self.total_results["results"]) > 0, "No results collected from benchmarks"
        logger.info(f"Total results collected: {len(self.total_results)}")

    def _parse_results(self, stdout: str, scenario: ScenarioConfig) -> dict[str, Any]:
        """Parse results"""
        results: dict[str, Any] = {
            "Input Tokens": scenario.input_tokens,
            "Output Tokens": scenario.output_tokens,
            "Concurrent": scenario.max_concurrency,
        }

        lines = stdout.strip().split("\n")
        for line in lines:
            try:
                metric, *_, score = line.split()
            except ValueError as e:
                logger.debug(f"Error occured: {e}")
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


    def finish_info(self, full: bool = False, desc: str = "") -> None:
        scenario_results = self.total_results["results"]

        desc = (
            f"{desc}\n"
            f"* Performance Benchmark - {self.name} for {self.task}\n"
            f"* Model Info: {self.model}\n"
            f"* Device Info: {self.device_name} x {self.used_device_num}\n"
            f"* date: {datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}\n"
            f"* Scenario Results:"
        )

        md_writer = MarkdownTableWriter()
        for scenario_result in scenario_results:
            if not full:
                keys_to_remove = [key for key in scenario_result if "P95" in key or "P99" in key]
                for key in keys_to_remove:
                    del scenario_result[key]

            md_writer.headers = list(scenario_result.keys())

        isl_osl_result = defaultdict[Any, list](list)
        for sub_result in scenario_results:
            isl_osl = self._to_isl_osl_result(sub_result["Input Tokens"], sub_result["Output Tokens"])
            isl_osl_result[isl_osl].append(list(sub_result.values()))
        
        total_md_contents = []
        total_csv_contents = []
        for isl_osl, result in isl_osl_result.items():
            md_contents = []
            csv_contents = [",".join(md_writer.headers)]
            for sub_result in result:
                md_contents.append(sub_result)
                csv_contents.append(",".join([str(value) for value in sub_result]))
                total_md_contents.append(sub_result)
                total_csv_contents.append(",".join([str(value) for value in sub_result]))
            
            md_writer.value_matrix = md_contents
            logger.info(
                "\n".join(
                    [
                        desc,
                        md_writer.dumps(),
                    ]
                )
            )

            with Path.open(self.get_vllm_result_dir("") / f"summary_{isl_osl}.md", "w") as f:
                f.write(desc + "\n" + md_writer.dumps() + "\n")

            with Path.open(self.get_vllm_result_dir("") / f"summary_{isl_osl}.csv", "w") as f:
                f.write(desc + "\n" + "\n".join(csv_contents) + "\n")

        md_writer.value_matrix = total_md_contents
        with Path.open(self.get_vllm_result_dir("") / f"summary.md", "w") as f:
            f.write(desc + "\n" + md_writer.dumps() + "\n")

        with Path.open(self.get_vllm_result_dir("") / f"summary.csv", "w") as f:
            f.write(desc + "\n" + "\n".join(total_csv_contents) + "\n")

    def _to_isl_osl_result(self, input_tokens: int, output_tokens: int) -> str:
        isl = f"{input_tokens//1024}k" if input_tokens >= 1024 else str(input_tokens)
        osl = f"{output_tokens//1024}k" if output_tokens >= 1024 else str(output_tokens)
        return f"{isl}_{osl}"

    def get_base_dir(self) -> Path:
        workspace = Path(f"{WORKSPACE}")

        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def get_vllm_result_dir(self, result_name: str) -> Path:
        """Return the benchmark result directory."""
        task_name = f"dev-{self.task}" if self.dev == "dev" else self.task
        result_dir = Path(
            self.base_dir
            / f"{self.device_name}_{self.backend}_{self.used_device_num}"
            / self.name
            / task_name
            / self.model.split("/")[-1]
            / result_name
        )
        if not result_dir.exists():
            result_dir.mkdir(parents=True)

        return result_dir.resolve()