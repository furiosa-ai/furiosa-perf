"""Tests for benchmark presets: matrix expansion, validation and server overrides."""

from pathlib import Path

import pytest

from furiosa_perf.spec import (
    PRESETS_DIR,
    apply_server_overrides,
    expand_matrix,
    load_preset,
    parse_preset,
    parse_server,
    parse_server_args,
    variant_label,
)


def preset(**overrides: object) -> dict[str, object]:
    """A minimal valid preset mapping with ``overrides`` applied."""
    return {"kind": "baseline", "task": "text", "client": "vllm-bench", **overrides}


def test_expand_matrix_takes_the_cartesian_product_of_list_values() -> None:
    """Two lengths x two concurrencies give four points, each with the shared args and a default prompt count."""
    points = expand_matrix(
        {"dataset-name": "random"},
        [{"random-input-len": [128, 512], "random_output_len": 128, "max-concurrency": [1, 8]}],
    )

    assert [(p["random-input-len"], p["max-concurrency"], p["num-prompts"]) for p in points] == [
        (128, 1, 3),
        (128, 8, 24),
        (512, 1, 3),
        (512, 8, 24),
    ]
    assert all(p["dataset-name"] == "random" and p["random-output-len"] == 128 for p in points)


def test_expand_matrix_keeps_an_explicit_prompt_count_and_runs_args_alone_without_a_matrix() -> None:
    """num-prompts given anywhere wins over the default; no matrix means one point from args."""
    assert expand_matrix({}, [{"max-concurrency": 4, "num-prompts": 100}])[0]["num-prompts"] == 100
    assert expand_matrix({"max-concurrency": 2}, []) == [{"max-concurrency": 2, "num-prompts": 6}]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "trace"}, "kind 'trace' is unsupported"),
        ({"client": "guidellm"}, "client 'guidellm' is unsupported"),
        ({"slo": {"ttft_p99_ms": 1}}, "SLO keys \\['ttft_p99_ms'\\] are unknown"),
        ({"client": "aiperf", "args": {"url": "x"}}, "'url'\\] are not allowed"),
        ({"task": ""}, "missing 'task'"),
        ({"scenarios": []}, "keys \\['scenarios'\\] are unknown"),
        ({"args": {"result-dir": "out"}}, "'result-dir'\\] are not allowed"),
        ({"matrix": [{"model": "x"}]}, "'model'\\] are not allowed"),
    ],
)
def test_parse_preset_rejects_invalid_presets(overrides: dict[str, object], message: str) -> None:
    """Unsupported kinds and clients, missing keys, old-format keys and managed client options are config errors."""
    with pytest.raises(ValueError, match=message):
        parse_preset(preset(**overrides))


@pytest.mark.parametrize("path", sorted(PRESETS_DIR.rglob("*.yaml")), ids=lambda p: p.stem)
def test_every_bundled_preset_loads(path: Path) -> None:
    """The bundled presets are valid and every point has the fields a summary row is keyed by."""
    loaded = load_preset(path.stem, path.parent.name)

    assert loaded.points
    if loaded.kind == "baseline":
        assert all("random-input-len" in p and isinstance(p["max-concurrency"], int) for p in loaded.points)
    elif loaded.search is not None:
        assert len(loaded.points) == 1 and loaded.search.key not in loaded.points[0]
    else:
        assert all(isinstance(p["concurrency"], int) for p in loaded.points)


def test_preset_server_options_win_and_null_removes_an_option() -> None:
    """The preset turns prefix caching off even if the server config enabled it, and null drops the parser."""
    server = parse_server(
        {
            "backend": "vllm",
            "model": "m",
            "devices": 0,
            "enable-prefix-caching": True,
            "reasoning-parser": "qwen3",
            "max-model-len": 4096,
        },
        Path("/env"),
    )
    merged = apply_server_overrides(server, {"no-enable-prefix-caching": True, "reasoning-parser": None})

    assert merged.args == {"max-model-len": 4096, "reasoning-parser": None, "no-enable-prefix-caching": True}


def test_preset_server_options_cannot_set_parallelism() -> None:
    """Parallelism belongs to the server config, so a preset cannot change it behind its back."""
    server = parse_server({"backend": "vllm", "model": "m", "devices": 0}, Path("/env"))
    with pytest.raises(ValueError, match="parallelism comes from the server config"):
        apply_server_overrides(server, {"tensor-parallel-size": 2})


def test_server_args_are_typed_and_named_for_the_run_directory() -> None:
    """--server-arg reads values as YAML; the label lists the options in a stable order."""
    args = parse_server_args(("--no-enable-prefix-caching", "max-num-seqs=64", "reasoning-parser=null"))

    assert args == {"no-enable-prefix-caching": True, "max-num-seqs": 64, "reasoning-parser": None}
    assert variant_label(args) == "max-num-seqs=64+no-enable-prefix-caching+reasoning-parser=null"
    assert variant_label({}) == ""
    with pytest.raises(ValueError, match="has no option name"):
        parse_server_args(("=1",))


def test_preset_of_another_kind_is_refused() -> None:
    """`run baseline` with a scenario preset file stops before anything runs."""
    with pytest.raises(ValueError, match="run it with `furiosa-perf run scenario`"):
        load_preset(str(PRESETS_DIR / "scenario" / "chat-synthetic.yaml"), "baseline")
    with pytest.raises(ValueError, match="baseline preset 'chat-synthetic' does not exist"):
        load_preset("chat-synthetic", "baseline")
