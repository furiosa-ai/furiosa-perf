from dataclasses import dataclass, field
from typing import Any, TypeAlias

from furiosa_perf.utils.logger import logger

@dataclass
class APIServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int | None = None
    enable_expert_parallel: bool | None = None
    data_parallel_size: int = 1
    devices: str | None = None  # comma separated list of deviceps IDs

@dataclass
class VllmServerConfig(APIServerConfig):
    no_enable_prefix_caching: bool | None = True

    def __post_init__(self) -> None:
        if self.tensor_parallel_size is None:
            if self.devices is not None:
                self.tensor_parallel_size = len(self.devices.split(","))
            else:
                self.tensor_parallel_size = 1
                self.devices = ",".join(map(str, range(self.tensor_parallel_size)))

        elif self.devices is None:
            self.devices = ",".join(map(str, range(self.tensor_parallel_size)))

        if not self.check_compatibility():
            raise ValueError(
                f"Devices: {self.devices} is not compatible with tensor_parallel_size: {self.tensor_parallel_size}"
            )

    def check_compatibility(self) -> bool:
        used_device_num = len(self.devices.split(",")) if self.devices is not None else 1
        return used_device_num == self.tensor_parallel_size


@dataclass
class FuriosaLLMServerConfig(APIServerConfig):
    enable_prefix_caching: bool | None = False
    expected_average_seq_length: int | None = None
    max_concurrency: int | None = None
    max_num_batched_tokens: int | None = None
    max_num_prompt_tokens: int | None = None
    prefix_cache_lookahead_requests: int | None = None

    def __post_init__(self) -> None:
        # case 1 -> tp is None, devices is None       
        if self.tensor_parallel_size is None:
            if self.devices is not None:
                self.tensor_parallel_size = len(self.devices.split(","))

        if not self._check_compatibility():
            raise ValueError(
                f"Devices: {self.devices} is not compatible with tensor_parallel_size: {self.tensor_parallel_size}"
            )

    def _check_compatibility(self) -> bool:
        used_device_num = len(self.devices.split(",")) if self.devices is not None else 1
        return used_device_num == self.tensor_parallel_size


class APIServerConfigLoader:
    # TODO: add more server configurations
    CONFIG_REGISTRY: dict[str, type[APIServerConfig]] = {
        "vllm": VllmServerConfig,
        "furiosa-llm": FuriosaLLMServerConfig,
    }

    @classmethod
    def create_config(cls, backend: str, configs: dict[str, Any]) -> APIServerConfig:
        config_class = cls.CONFIG_REGISTRY.get(backend, None)

        if config_class is None:
            logger.error(f"Server configuration for serving_framework {backend} not found.")
            raise ValueError(f"Server configuration for serving_framework {backend} not found.")
        return config_class(**configs)


@dataclass
class LLMScenarioConfig:
    input_tokens: int = 1024
    output_tokens: int = 1024
    max_concurrency: int = 1
    num_prompts: int | None = None
    request_rate: int | str = "inf"
    random_range_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.num_prompts is None:
            self.num_prompts = self.max_concurrency * 3

@dataclass
class ShareGPTConfig:
    input_tokens: int = 1024
    output_tokens: int = 1024
    max_concurrency: int = 1


@dataclass
class PrefixCacheScenarioConfig:
    # TODO(JW): need to add prefix cache benchmark commands
    input_tokens: int = 1024
    output_tokens: int = 1024
    max_concurrency: int = 1


@dataclass
class RerankerScenarioConfig:
    # TODO(JW): need to add reranker benchmark commands
    input_tokens: int = 1024
    output_tokens: int = 1024
    max_concurrency: int = 1


@dataclass
class EmbeddingScenarioConfig:
    # TODO(JW): need to add embedding benchmark commands
    input_tokens: int = 1024
    output_tokens: int = 1024
    max_concurrency: int = 1


ScenarioConfig: TypeAlias = (
    LLMScenarioConfig | 
    RerankerScenarioConfig | 
    PrefixCacheScenarioConfig | 
    EmbeddingScenarioConfig |
    ShareGPTConfig
)

@dataclass
class PerformanceBenchConfig:
    name: str
    task: str
    model: str = ""
    device_name: str = ""
    used_device_num: int = 1
    scenarios: list[ScenarioConfig] = field(default_factory=list)

class PerformanceBenchConfigLoader:
    CONFIG_REGISTRY: dict[str, type[ScenarioConfig]] = {
        "offline": LLMScenarioConfig,
        "prefix-cache": PrefixCacheScenarioConfig,
        "reranker": RerankerScenarioConfig,
        "embedding": EmbeddingScenarioConfig,
        "sharegpt": ShareGPTConfig,
    }

    @classmethod
    def create_config(cls, configs: dict[str, Any]) -> PerformanceBenchConfig:
        scenario_class = cls.CONFIG_REGISTRY.get(configs.get("task", None), None)
        if scenario_class is None:
            logger.error(f"Performance benchmark configuration for task {configs['task']} not found.")
            raise ValueError(f"Performance benchmark configuration for task {configs['task']} not found.")

        base = {k: v for k, v in configs.items() if k not in ["scenarios"]}
        expanded_scenarios = []
        for scenario in configs.get("scenarios", []):
            if isinstance(scenario["max_concurrency"], list):
                scenario_base = {k: v for k, v in scenario.items() if k != "max_concurrency"}
                expanded_scenarios.extend([
                    scenario_class(**scenario_base, max_concurrency=max_concurrency) for max_concurrency in scenario["max_concurrency"]
                ])
            else:
                expanded_scenarios.append(scenario_class(**scenario))
        return PerformanceBenchConfig(**base, scenarios=expanded_scenarios)