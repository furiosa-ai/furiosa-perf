"""Tests for environment collection: device names, parsing of probe output and pre-run checks."""

from pathlib import Path

import pytest

from furiosa_perf.runner.env import (
    GPU_QUERY,
    Environment,
    check,
    device_label,
    gpu_product_name,
    model_info,
    native_packages,
    npu_arch_name,
    parse_gpu_query,
    parse_key_values,
    probe,
    recorded_env_vars,
)
from furiosa_perf.spec import parse_server


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NVIDIA H100 80GB HBM3", "H100-80GB-HBM3"),
        ("NVIDIA H100 NVL", "H100-NVL"),
        ("NVIDIA H100 PCIe", "H100-PCIe"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX-PRO-6000-Blackwell-Server-Edition"),
        ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "RTX-PRO-6000-Blackwell-MaxQ-Workstation-Edition"),
    ],
)
def test_gpu_product_name_keeps_the_words_that_tell_variants_apart(raw: str, expected: str) -> None:
    """SXM/NVL/PCIe H100s and the RTX PRO 6000 editions get distinct names."""
    assert gpu_product_name(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [("RNGD", "RNGD"), ("Arch.Rngd", "RNGD"), ("Rngd", "RNGD"), ("Arch.RngdMax", "RNGD-MAX")])
def test_npu_arch_name(raw: str, expected: str) -> None:
    """RNGD variants are named after their arch, whether or not the enum prefix is present."""
    assert npu_arch_name(raw) == expected


def test_device_label_marks_only_non_default_power_settings() -> None:
    """Default limits and the Performance governor add nothing; a lowered limit or PowerSave is shown."""
    default = {"power_limit_w": 700.0, "default_power_limit_w": 700.0}
    lowered = {"power_limit_w": 500.0, "default_power_limit_w": 700.0}

    assert device_label("H100-80GB-HBM3", [default, default]) == "H100-80GB-HBM3"
    assert device_label("H100-80GB-HBM3", [default, lowered]) == "H100-80GB-HBM3@500W"
    assert device_label("RNGD", [{"governor": "Performance"}]) == "RNGD"
    assert device_label("RNGD", [{"governor": "PowerSave"}]) == "RNGD@powersave"


def test_recorded_env_vars_keep_relevant_prefixes_and_mask_secrets() -> None:
    """Only variables that change server or library behaviour are kept; tokens never reach run.json."""
    environ = {"HF_TOKEN": "hf_abc", "HF_HOME": "/hf", "FURIOSA_LOG": "info", "HOME": "/root", "VLLM_API_KEY": "k"}

    assert recorded_env_vars(environ) == {
        "FURIOSA_LOG": "info",
        "HF_HOME": "/hf",
        "HF_TOKEN": "***",
        "VLLM_API_KEY": "***",
    }


def test_native_packages_keep_drivers_and_runtimes() -> None:
    """Shared apt packages that decide device behaviour are recorded; unrelated ones are not."""
    dpkg = "bash 5.1-6\nfuriosa-smi 2026.1.1-3\nlibnccl2 2.21.5-1+cuda12.4\nnvidia-driver-570 570.86\n"

    assert native_packages(dpkg) == {
        "furiosa-smi": "2026.1.1-3",
        "libnccl2": "2.21.5-1+cuda12.4",
        "nvidia-driver-570": "570.86",
    }


def test_parse_gpu_query_reads_power_limits_and_tolerates_na() -> None:
    """A real H100 row parses into the device fields; [N/A] cells become None instead of failing."""
    row = (
        "0, NVIDIA H100 80GB HBM3, GPU-1234, 00000000:18:00.0, 81559, 595.91.07, 96.00.89.00.01, "
        "69.91, 700.00, 700.00, 700.00, 200.00, 5, [N/A]"
    )
    (device,) = parse_gpu_query(row, GPU_QUERY)

    assert device["index"] == 0
    assert device["name"] == "H100-80GB-HBM3"
    assert device["memory_gib"] == 79.65
    assert (device["power_limit_w"], device["default_power_limit_w"], device["min_power_limit_w"]) == (
        700.0,
        700.0,
        200.0,
    )
    assert device["idle_power_w"] == 69.91
    assert device["pcie_gen"] == 5.0
    assert device["pcie_width"] is None


def test_parse_key_values_reads_lscpu_and_os_release() -> None:
    """Both "key: value" and key="value" formats are read, quotes removed."""
    assert parse_key_values("Model name:   AMD EPYC 9454\nSocket(s): 2\n", ":")["Socket(s)"] == "2"
    assert parse_key_values('PRETTY_NAME="Ubuntu 22.04.5 LTS"\n', "=") == {"PRETTY_NAME": "Ubuntu 22.04.5 LTS"}


def environment(devices: list[dict[str, object]], version: str = "2026.4.0") -> Environment:
    """An Environment with the given devices and backend version, everything else empty."""
    return Environment(
        host={}, devices=devices, server_env={"version": version}, native_packages={}, model={}, env_vars={}
    )


def test_check_reports_a_missing_backend_and_missing_devices() -> None:
    """A run is refused before launch when the backend is not in env or a configured device does not exist."""
    server = parse_server({"backend": "furiosa-llm", "model": "m", "devices": "0,5", "env": "/venv"}, Path("/x"))

    assert check(environment([{"index": 0}, {"index": 5}]), server) == []
    problems = check(environment([{"index": 0}, {"index": 1}], version="Unknown"), server)
    assert any("cannot find furiosa-llm in the environment /venv" in p for p in problems)
    assert any("device indices [5] are invalid" in p for p in problems)
    assert check(environment([]), server) == ["no NPU devices are detected on this host"]


def test_summary_keeps_only_the_used_devices_in_config_order() -> None:
    """run.json describes the devices the server used, in the order the config lists them."""
    env = environment([{"index": 0}, {"index": 1}, {"index": 2}])

    assert env.summary([2, 0])["devices"] == [{"index": 2}, {"index": 0}]


def test_model_info_hashes_a_local_config(tmp_path: Path) -> None:
    """A local model directory is identified by the hash of its config.json instead of a Hub revision."""
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')

    info = model_info(str(tmp_path))
    assert info["local_path"] == str(tmp_path.resolve())
    assert len(info["config_sha256"]) == 16


def test_probe_strips_terminal_colour_codes() -> None:
    """Raw probe files stay plain text even when a tool colours its output (nvidia-smi topo -m does)."""
    assert probe(["printf", "\\033[4mGPU0\\033[0m NV18"]) == "GPU0 NV18"
    assert probe(["definitely-not-installed-tool"]).startswith("error: definitely-not-installed-tool is not installed")
