"""Configuration dataclasses and loaders for API server and benchmark runs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import yaml

from furiosa_perf.utils.logger import logger


@dataclass
class APIServerConfig:
    """Base configuration for any LLM API server."""

    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int | None = None
    enable_expert_parallel: bool | None = None
    data_parallel_size: int = 1
    devices: str | None = None  # comma-separated device indices
    served_model_name: str | None = None
    no_enable_prefix_caching: bool | None = True


@dataclass
class VllmServerConfig(APIServerConfig):
    """vLLM-specific API server configuration.

    Infers ``tensor_parallel_size`` from ``devices`` when omitted, and vice
    versa. Raises ``ValueError`` if both are provided but inconsistent.
    """

    def __post_init__(self) -> None:
        """Resolve and validate tensor_parallel_size / devices consistency."""
        if self.tensor_parallel_size is None:
            if self.devices is not None:
                self.tensor_parallel_size = len(self.devices.split(","))
            else:
                self.tensor_parallel_size = 1
                self.devices = ",".join(map(str, range(self.tensor_parallel_size)))
        elif self.devices is None:
            self.devices = ",".join(map(str, range(self.tensor_parallel_size)))

        if not self._check_compatibility():
            raise ValueError(
                f"devices={self.devices!r} is not compatible with "
                f"tensor_parallel_size={self.tensor_parallel_size}"
            )

    def _check_compatibility(self) -> bool:
        """Return True when device count matches tensor_parallel_size."""
        used = len(self.devices.split(",")) if self.devices is not None else 1
        return used == self.tensor_parallel_size


@dataclass
class FuriosaLLMServerConfig(APIServerConfig):
    """furiosa-llm-specific API server configuration."""

    fxb: str | None = None

    def __post_init__(self) -> None:
        """Infer tensor_parallel_size from devices when omitted."""
        if self.tensor_parallel_size is None and self.devices is not None:
            self.tensor_parallel_size = len(self.devices.split(","))

        if not self._check_compatibility():
            raise ValueError(
                f"devices={self.devices!r} is not compatible with "
                f"tensor_parallel_size={self.tensor_parallel_size}"
            )

    def _check_compatibility(self) -> bool:
        """Return True when device count matches tensor_parallel_size."""
        used = len(self.devices.split(",")) if self.devices is not None else 1
        return used == self.tensor_parallel_size


class APIServerConfigLoader:
    """Load and construct an :class:`APIServerConfig` from a YAML file."""

    CONFIG_REGISTRY: dict[str, type[APIServerConfig]] = {
        "vllm": VllmServerConfig,
        "furiosa-llm": FuriosaLLMServerConfig,
    }

    @classmethod
    def _create_config(cls, backend: str, configs: dict[str, Any]) -> APIServerConfig:
        """Instantiate the correct config subclass for *backend*.

        Args:
            backend: Backend name (e.g. ``"vllm"`` or ``"furiosa-llm"``).
            configs: Raw YAML key/value pairs forwarded to the dataclass.

        Returns:
            A fully initialised :class:`APIServerConfig` subclass instance.

        Raises:
            ValueError: If *backend* is not registered in :attr:`CONFIG_REGISTRY`.
        """
        config_class = cls.CONFIG_REGISTRY.get(backend)
        if config_class is None:
            raise ValueError(f"No server config registered for backend {backend!r}.")
        return config_class(**configs)

    @classmethod
    def api_server_setup(cls, backend: str, api_server_config_path: Path) -> APIServerConfig:
        """Load a server config YAML and return the parsed config object.

        Args:
            backend: Backend name used to select the config class.
            api_server_config_path: Path to the YAML configuration file.

        Returns:
            Parsed :class:`APIServerConfig` instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not api_server_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {api_server_config_path}")

        with open(api_server_config_path) as f:
            api_server_config = cls._create_config(backend=backend, configs=yaml.safe_load(f))

        logger.info(f"Loaded server config: {api_server_config}")
        return api_server_config


@dataclass
class BaseScenarioConfig:
    """Fields shared by every vLLM ``bench serve`` scenario.

    These map onto the placeholders in ``VllmPerformanceBenchmark.COMMAND``
    (``--max-concurrency``, ``--num-prompts``, ``--request-rate``) plus the
    input/output token counts used for result-directory naming and parsing.
    """

    input_tokens: int = 1024
    max_concurrency: int = 1
    num_prompts: int | None = None
    request_rate: int | str = "inf"
    model: str = ""


@dataclass
class LLMScenarioConfig(BaseScenarioConfig):
    """Scenario config for the ``offline`` task."""

    random_range_ratio: float = 0.0
    output_tokens: int = 1024

    def __post_init__(self) -> None:
        """Default num_prompts to max_concurrency × 3 when not set."""
        if self.num_prompts is None:
            self.num_prompts = self.max_concurrency * 3


@dataclass
class VLScenarioConfig(LLMScenarioConfig):
    """Scenario config for the ``vl-offline`` (vision-language) task."""

    random_range_ratio: float = 0.0
    random_mm_base_items_per_request: int = 1
    random_mm_bucket_config: str = "{(256, 256, 1): 1.0}"
    random_mm_limit_mm_per_prompt: str = '{"image": 1}'


@dataclass
class RerankerScenarioConfig(BaseScenarioConfig):
    """Scenario config for the ``reranker`` task."""

    random_batch_size: int = 5

    def __post_init__(self) -> None:
        """Default num_prompts to request_rate × 3 when not set."""
        if self.num_prompts is None:
            self.num_prompts = self.request_rate * 3  # type: ignore[operator]


@dataclass
class EmbeddingScenarioConfig(BaseScenarioConfig):
    """Scenario config for the ``embeddings`` task."""

    def __post_init__(self) -> None:
        """Default num_prompts to request_rate × 3 when not set."""
        if self.num_prompts is None:
            self.num_prompts = self.request_rate * 3  # type: ignore[operator]


ScenarioConfig: TypeAlias = (
    LLMScenarioConfig | VLScenarioConfig | RerankerScenarioConfig | EmbeddingScenarioConfig
)


@dataclass
class PerformanceBenchConfig:
    """Top-level benchmark configuration holding all scenarios to run."""

    name: str
    task: str
    model: str = ""
    device_name: str = ""
    used_device_num: int = 1
    scenarios: list[ScenarioConfig] = field(default_factory=list)


class PerformanceBenchConfigLoader:
    """Load and construct a :class:`PerformanceBenchConfig` from a YAML file."""

    CONFIG_REGISTRY: dict[str, type[ScenarioConfig]] = {
        "offline": LLMScenarioConfig,
        "vl-offline": VLScenarioConfig,
        "reranker": RerankerScenarioConfig,
        "embeddings": EmbeddingScenarioConfig,
    }

    @classmethod
    def _create_config(cls, configs: dict[str, Any]) -> PerformanceBenchConfig:
        """Parse raw YAML data into a :class:`PerformanceBenchConfig`.

        Args:
            configs: Raw YAML dict containing ``name``, ``task``, and ``scenarios``.

        Returns:
            Fully expanded :class:`PerformanceBenchConfig` with one
            :class:`ScenarioConfig` per concurrency level.

        Raises:
            ValueError: If the task name is not registered.
        """
        task = configs.get("task")
        scenario_class = cls.CONFIG_REGISTRY.get(task)  # type: ignore[arg-type]
        if scenario_class is None:
            raise ValueError(f"No scenario config registered for task {task!r}.")

        base = {k: v for k, v in configs.items() if k != "scenarios"}
        expanded: list[ScenarioConfig] = []
        for scenario in configs.get("scenarios", []):
            concurrencies = scenario["max_concurrency"]
            if isinstance(concurrencies, list):
                base_fields = {k: v for k, v in scenario.items() if k != "max_concurrency"}
                expanded.extend(scenario_class(**base_fields, max_concurrency=c) for c in concurrencies)
            else:
                expanded.append(scenario_class(**scenario))

        return PerformanceBenchConfig(**base, scenarios=expanded)

    @classmethod
    def benchmark_config_setup(cls, benchmark_config_path: Path) -> PerformanceBenchConfig:
        """Load a benchmark config YAML and return the parsed config object.

        Args:
            benchmark_config_path: Path to the YAML configuration file.

        Returns:
            Parsed :class:`PerformanceBenchConfig` instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not benchmark_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {benchmark_config_path}")

        with open(benchmark_config_path) as f:
            benchmark_config = cls._create_config(yaml.safe_load(f))

        logger.info(f"Loaded benchmark config: {benchmark_config}")
        return benchmark_config
