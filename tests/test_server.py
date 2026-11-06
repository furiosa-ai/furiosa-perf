"""Tests for building the API server command line."""

from pathlib import Path

import pytest

from furiosa_perf.runner.server import check_furiosa_layout, furiosa_layout, option_args, server_command
from furiosa_perf.spec import Server, parse_server

ENV = Path("/opt/serving-env")


def command(backend: str, devices: str, **extra: object) -> tuple[list[str], dict[str, str]]:
    """``server_command`` for a parsed config with the given backend and devices."""
    raw = {"backend": backend, "model": "org/model", "devices": devices, "enable-prefix-caching": True, **extra}
    return server_command(parse_server(raw, ENV))


def test_furiosa_llm_gets_pipeline_and_data_parallelism_only() -> None:
    """Two RNGD cards as two replicas: devices as npu:<i>, pp and dp explicit, tp left to the artifact."""
    argv, env = command("furiosa-llm", "2,3", parallel={"dp": 2})

    assert argv[:3] == [str(ENV / "bin" / "furiosa-llm"), "serve", "org/model"]
    assert argv[argv.index("--devices") + 1] == "npu:2,npu:3"
    assert "--tensor-parallel-size" not in argv  # removed from furiosa-llm serve in 2026.4
    assert argv[argv.index("--data-parallel-size") + 1] == "2"
    assert argv[argv.index("--pipeline-parallel-size") + 1] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in env


# Placement lines as furiosa-llm 2026.4 logs them (ANSI colour codes trimmed).
TWO_REPLICAS = """INFO furiosa_generator::next_gen::generator: DP entry DpId(0) \u2192 PP devices [TpGroupId(npu0)]
INFO furiosa_generator::next_gen::generator: DP entry DpId(1) \u2192 PP devices [TpGroupId(npu1)]"""
ONE_PIPELINE = "INFO ...: DP entry DpId(0) \u2192 PP devices [TpGroupId(npu0), TpGroupId(npu1)]"


def test_furiosa_layout_reads_replicas_and_stages_from_the_log() -> None:
    """Each logged replica lists its stages; a multi-card group like npu2-3 is a range."""
    assert furiosa_layout(TWO_REPLICAS) == [[{0}], [{1}]]
    assert furiosa_layout(ONE_PIPELINE) == [[{0}, {1}]]
    assert furiosa_layout("DP entry DpId(0) \u2192 PP devices [TpGroupId(npu2-3)]") == [[{2, 3}]]
    assert furiosa_layout("DP entry DpId(0) \u2192 PP devices [TpGroupId(npu0pe0-3)]") == [[{0}]]


def test_check_furiosa_layout_rejects_a_silently_dropped_card() -> None:
    """Two cards with dp 1 and a one-card artifact use only npu0; the run must not go on."""

    def two_cards(parallel: dict[str, int]) -> Server:
        return parse_server({"backend": "furiosa-llm", "model": "m", "devices": "0,1", "parallel": parallel}, ENV)

    check_furiosa_layout(two_cards({"dp": 2}), TWO_REPLICAS)
    check_furiosa_layout(two_cards({"tp": 1, "pp": 2}), ONE_PIPELINE)
    with pytest.raises(RuntimeError, match="set 'parallel' to match it"):
        check_furiosa_layout(two_cards({"tp": 2}), "DP entry DpId(0) \u2192 PP devices [TpGroupId(npu0)]")


def test_vllm_selects_gpus_through_the_environment() -> None:
    """vLLM sees only the configured GPUs and counts tensor parallelism in GPUs."""
    argv, env = command("vllm", "4,5,6,7", parallel={"tp": 2})

    assert env["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert "--devices" not in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--data-parallel-size") + 1] == "2"


def test_server_runs_from_its_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env's bin comes first on PATH and variables pointing at another environment are dropped."""
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/else")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else/lib")
    _, env = command("vllm", "0")

    assert env["PATH"].startswith(str(ENV / "bin"))
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONPATH" not in env


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, ["--enforce-eager"]),
        (False, []),
        (None, []),
        (4096, ["--enforce-eager", "4096"]),
        (["a", "b"], ["--enforce-eager", "a", "b"]),
        ({"k": 1}, ["--enforce-eager", '{"k": 1}']),
    ],
)
def test_option_args(value: object, expected: list[str]) -> None:
    """Flags, omitted values, scalars, lists and JSON objects render as the backends' argparse expects."""
    assert option_args("enforce-eager", value) == expected
