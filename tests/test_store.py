"""Tests for the results layout: names, the run directory policy and JSON files."""

from pathlib import Path

import pytest

from furiosa_perf.results.store import (
    RECORD_FILE,
    RUN_FILE,
    check_unique_point_names,
    point_name,
    prepare_run_dir,
    read_json,
    run_dir,
    write_json,
)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ({"random-input-len": 1024, "random-output-len": 128, "max-concurrency": 8}, "i1024-o128-c8"),
        ({"random-input-len": 512, "random-batch-size": 5, "max-concurrency": 4}, "i512-b5-c4"),
        ({"random-input-len": 256, "max-concurrency": 1}, "i256-c1"),
        ({"concurrency": 32, "isl": 150}, "u32"),
        ({"user-centric-rate": 2.0, "num-users": 64}, "ucr2.0-n64"),
    ],
)
def test_point_name_says_what_each_number_is(point: dict[str, object], expected: str) -> None:
    """Baseline points are named by input, output or documents and concurrency; scenario points by their load."""
    assert point_name(point) == expected


def test_points_that_would_share_a_directory_are_rejected() -> None:
    """Two points differing only in an option the name leaves out would overwrite each other."""
    point = {"random-input-len": 128, "random-output-len": 128, "max-concurrency": 1}
    with pytest.raises(ValueError, match="points 0 and 1 would both be stored as 'i128-o128-c1'"):
        check_unique_point_names([point, {**point, "random-range-ratio": 0.5}])


def test_run_dir_layout() -> None:
    """kind / device_count[-layout]_backend_version / task / org--model."""
    path = run_dir(Path("out"), "baseline", "H100-80GB-HBM3", 2, "tp1-dp2", "vllm_0.28.0", "text", "Qwen/Qwen3-8B")

    assert path == Path("out/baseline/H100-80GB-HBM3_2-tp1-dp2_vllm_0.28.0/text/Qwen--Qwen3-8B")
    assert run_dir(Path("out"), "baseline", "RNGD", 4, "", "furiosa-llm_2026.4.0", "text", "m").parts[2] == (
        "RNGD_4_furiosa-llm_2026.4.0"
    )
    variant = run_dir(Path("out"), "scenario", "RNGD", 1, "", "furiosa-llm_2026.4.0", "chat", "org/m", 'c={"a": 1}')
    assert variant.name == "org--m__c=-a-1-"  # next to org--m, file-safe


def do_test_existing_run(tmp_path: Path, resume: bool, overwrite: bool) -> Path:
    """Create a run directory holding one recorded point, then prepare it again."""
    path = tmp_path / "run"
    (path / "i1-c1").mkdir(parents=True)
    write_json(path / "i1-c1" / RECORD_FILE, {"summary": {"ok": True}})
    prepare_run_dir(path, resume=resume, overwrite=overwrite)
    return path


def test_an_existing_run_is_refused_by_default(tmp_path: Path) -> None:
    """Without --resume or --overwrite a second run never mixes into the first."""
    with pytest.raises(RuntimeError, match="already holds a run; pass --resume"):
        do_test_existing_run(tmp_path, resume=False, overwrite=False)


def test_resume_keeps_and_overwrite_deletes_an_existing_run(tmp_path: Path) -> None:
    """--resume leaves the recorded points in place; --overwrite starts from an empty directory."""
    kept = do_test_existing_run(tmp_path / "a", resume=True, overwrite=False)
    replaced = do_test_existing_run(tmp_path / "b", resume=False, overwrite=True)

    assert (kept / "i1-c1" / RECORD_FILE).exists()
    assert replaced.is_dir() and not any(replaced.iterdir())


def test_a_fresh_directory_is_created(tmp_path: Path) -> None:
    """The first run of a configuration just creates its directory."""
    path = tmp_path / "baseline" / "x" / "text" / "m"
    prepare_run_dir(path, resume=False, overwrite=False)

    assert path.is_dir()
    assert not (path / RUN_FILE).exists()


def test_json_round_trip_and_unreadable_files(tmp_path: Path) -> None:
    """write_json leaves no temporary file behind; read_json returns None for missing or broken files."""
    write_json(tmp_path / "a.json", {"x": 1, "p": Path("/p")})

    assert read_json(tmp_path / "a.json") == {"x": 1, "p": "/p"}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
    (tmp_path / "b.json").write_text("{not json")
    assert read_json(tmp_path / "b.json") is None
    assert read_json(tmp_path / "missing.json") is None
