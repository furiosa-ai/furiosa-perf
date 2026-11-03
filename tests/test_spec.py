"""Tests for reading and validating the server config."""

from pathlib import Path

import pytest

from furiosa_perf.spec import Parallel, parse_devices, parse_parallel, parse_server

ENV = Path("/opt/serving-env")


def server_config(**overrides: object) -> dict[str, object]:
    """A minimal valid server config with ``overrides`` applied."""
    return {"backend": "furiosa-llm", "model": "furiosa-ai/Qwen3-8B-FP8", "devices": "0,1", **overrides}


@pytest.mark.parametrize(
    ("raw", "expected"), [(0, [0]), ("0,1,2,3", [0, 1, 2, 3]), ([4, 5], [4, 5]), (" 1 , 3", [1, 3])]
)
def test_parse_devices_accepts_int_string_and_list(raw: object, expected: list[int]) -> None:
    """YAML gives ``devices: 0`` as an int and ``devices: 0,1`` as a string; both and lists are read."""
    assert parse_devices(raw) == expected


@pytest.mark.parametrize("raw", ["", "a,b", "0,0", "-1", []])
def test_parse_devices_rejects_empty_garbage_duplicates_and_negatives(raw: object) -> None:
    """Anything that is not a list of distinct non-negative indices is a config error."""
    with pytest.raises(ValueError, match="device list"):
        parse_devices(raw)


@pytest.mark.parametrize(
    ("raw", "num_devices", "expected"),
    [
        (None, 4, Parallel(tp=4)),  # default: tensor parallelism over every device
        ({"dp": 2}, 4, Parallel(tp=2, dp=2)),  # tp derived
        ({"tp": 1}, 4, Parallel(tp=1, dp=4)),  # dp derived
        ({"tp": 1, "pp": 2}, 2, Parallel(tp=1, pp=2)),
        ({"tp": 2, "pp": 1, "dp": 2}, 4, Parallel(tp=2, dp=2)),
    ],
)
def test_parse_parallel_derives_the_missing_size(raw: object, num_devices: int, expected: Parallel) -> None:
    """One omitted size is derived from the device count; pp and dp default to 1."""
    assert parse_parallel(raw, num_devices) == expected


@pytest.mark.parametrize(
    "raw", [{"tp": 3}, {"dp": 3}, {"tp": 2, "dp": 3}, {"tp": 0}, {"tp": "2"}, {"tp": True}, {"ep": 2}, [2]]
)
def test_parse_parallel_rejects_sizes_that_do_not_fit(raw: object) -> None:
    """Sizes must be positive integers whose product is the device count (4 here)."""
    with pytest.raises(ValueError, match="parallel"):
        parse_parallel(raw, 4)


def test_parallel_label_is_empty_only_for_pure_tensor_parallelism() -> None:
    """Pure TP keeps the existing result names; other layouts are named after their sizes."""
    assert Parallel(tp=4).label == ""
    assert Parallel(tp=2, dp=2).label == "tp2-dp2"
    assert Parallel(tp=1, pp=2).label == "tp1-pp2"


def test_parse_server_normalises_keys_and_passes_the_rest_through() -> None:
    """snake_case keys become kebab-case and unknown keys are kept for the backend, nothing else added."""
    server = parse_server(server_config(served_model_name="m", max_model_len=4096, parallel={"dp": 2}), ENV)

    assert server.devices == [0, 1]
    assert server.parallel == Parallel(tp=1, dp=2)
    assert server.env == ENV
    assert server.executable == ENV / "bin" / "furiosa-llm"
    assert server.args == {"served-model-name": "m", "max-model-len": 4096}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"backend": "tgi"}, "backend 'tgi' is unsupported"),
        ({"model": ""}, "missing 'model'"),
        ({"tensor_parallel_size": 2}, "'tensor-parallel-size' is not allowed"),
        ({"data-parallel-size": 2}, "'data-parallel-size' is not allowed"),
    ],
)
def test_parse_server_rejects_invalid_configs(overrides: dict[str, object], message: str) -> None:
    """Each error names the offending key so the config can be fixed without reading code."""
    with pytest.raises(ValueError, match=message):
        parse_server(server_config(**overrides), ENV)


def test_system_python_finds_the_backend_in_local_bin(tmp_path: Path) -> None:
    """pip on a Debian/Ubuntu system Python (sys.prefix /usr) installs scripts to /usr/local/bin."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python3").touch()
    (tmp_path / "local" / "bin").mkdir(parents=True)
    (tmp_path / "local" / "bin" / "furiosa-llm").touch()
    server = parse_server(server_config(), tmp_path)

    assert server.executable == tmp_path / "local" / "bin" / "furiosa-llm"
    assert server.python == tmp_path / "bin" / "python3"
