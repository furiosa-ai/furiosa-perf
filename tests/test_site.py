"""Tests for collecting results into the report site's data files."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from furiosa_perf.reporting.site import collect_runs, write_data, write_viewer

LEGACY_SUMMARY = """* Framework Version: furiosa-llm_2026.3.0
* API Server Command: furiosa-llm serve furiosa-ai/m --port 8000 --enable-overlap-scheduling --enable-payload-logging
* Model Info: furiosa-ai/m
* date: 20260708-045349
* Scenario Results:
Input Tokens,Output Tokens,Concurrent,Output Throughput(tok/s),mean_power
128,128,1,70.77,231.0
128,128,4,0,99.9
"""


def _results(tmp_path: Path) -> Path:
    legacy = tmp_path / "bench_space" / "RNGD_4_furiosa-llm_2026.3.0" / "vllm" / "offline" / "m"
    legacy.mkdir(parents=True)
    (legacy / "summary.csv").write_text(LEGACY_SUMMARY)
    point = legacy / "128.128.1"
    point.mkdir()
    (point / "vllm-infqps-concurrency1-m.json").write_text(
        json.dumps({"max_concurrency": 1, "num_prompts": 3, "ttfts": [0.1]})
    )
    (point / "command.txt").write_text("vllm bench serve --max-concurrency=1 --num-prompts=3\n")
    flag_experiment = legacy.parent / "m__overlap"
    flag_experiment.mkdir()
    (flag_experiment / "summary.csv").write_text(LEGACY_SUMMARY)

    run = tmp_path / "results" / "baseline" / "H100_2-tp1-dp2_vllm" / "text" / "org--m"
    (run / "i1024-o128-c1").mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "kind": "baseline",
                "task": "text",
                "model": "org/m",
                "hardware": "H100",
                "hardware_product": "H100-80GB-HBM3",
                "num_devices": 2,
                "layout": "tp1-dp2",
                "backend": "vllm",
                "backend_version": "0.30.0",
                "started": "2026-10-18T02:28:08+00:00",
                "server": {"command": "vllm serve org/m --no-enable-prefix-caching"},
                "environment": {"host": {"os": "Ubuntu 22.04"}},
                "variant": {"max-num-seqs": 64},
            }
        )
    )
    summary = {
        "Input Tokens": 1024,
        "Output Tokens": 128,
        "Concurrent": 1,
        "ok": False,
        "Output Throughput(tok/s)": 5.0,
    }
    record = {
        "summary": summary,
        "point": {"isl": 1024, "osl": 128, "concurrency": 1},
        "command": "aiperf profile --concurrency 1",
    }
    (run / "i1024-o128-c1" / "record.json").write_text(json.dumps(record))
    (run / "summary.csv").write_text("Input Tokens\n1024\n")  # read through run.json, not as a legacy run
    return tmp_path


def test_both_result_layouts_become_runs(tmp_path: Path) -> None:
    """Legacy and new runs carry the same fields; flag experiments and the new tree's summary.csv are skipped."""
    new, legacy = collect_runs([_results(tmp_path)])  # sorted by id

    assert (legacy["task"], legacy["model"], legacy["hw"], legacy["n"], legacy["ver"]) == (
        "text",
        "m",
        "RNGD",
        4,
        "2026.3.0",
    )
    assert legacy["flags"] == ["enable-overlap-scheduling"]
    assert legacy["date"] == "2026-07-08T04:53:49+00:00"
    assert legacy["points"][0]["Output Throughput(tok/s)"] == 70.77
    assert (new["model"], new["model_id"], new["hw"], new["layout"], new["flags"]) == (
        "m",
        "org/m",
        "H100-80GB-HBM3",
        "tp1-dp2",
        [],
    )
    assert new["variant"] == "max-num-seqs=64" and legacy["variant"] == ""
    assert new["id"] == "H100-80GB-HBM3_2-tp1-dp2_vllm_0.30.0--baseline--text--m--max-num-seqs-64--202610180228"


def test_runs_say_how_they_were_run(tmp_path: Path) -> None:
    """Server command, environment, and per point the client command and its parameters, keyed isl/osl/concurrency."""
    new, legacy = collect_runs([_results(tmp_path)])

    assert legacy["details"] == {
        "server_command": "furiosa-llm serve furiosa-ai/m --port 8000 --enable-overlap-scheduling --enable-payload-logging",
        "environment": {"Framework Version": "furiosa-llm_2026.3.0", "Model Info": "furiosa-ai/m"},
        "commands": {"128/128/1": "vllm bench serve --max-concurrency=1 --num-prompts=3"},
        "params": {"128/128/1": {"max_concurrency": 1, "num_prompts": 3}},
    }
    assert new["details"] == {
        "server_command": "vllm serve org/m --no-enable-prefix-caching",
        "environment": {"host": {"os": "Ubuntu 22.04"}},
        "commands": {"1024/128/1": "aiperf profile --concurrency 1"},
        "params": {"1024/128/1": {"isl": 1024, "osl": 128, "concurrency": 1}},
    }


def test_site_has_an_index_and_one_file_per_run(tmp_path: Path) -> None:
    """index.json describes each run and counts failed points; each run file has its points as columns."""
    runs = collect_runs([_results(tmp_path)])
    site = tmp_path / "site"
    write_viewer(site, theme=None)
    write_data(runs, site)

    index = json.loads((site / "data" / "index.json").read_text())["runs"]
    assert [(r["num_points"], r["failed"], r["shapes"]) for r in index] == [
        (1, 1, ["1024/128"]),
        (2, 1, ["128/128"]),
    ]
    points = json.loads((site / "data" / "runs" / f"{index[1]['id']}.json").read_text())
    assert "details" not in index[1] and points["details"]["commands"]
    assert points["cols"][:4] == ["Input Tokens", "Output Tokens", "Concurrent", "Output Throughput(tok/s)"]
    assert points["rows"][1][3] == 0
    assert (site / "index.html").exists() and (site / "theme" / "theme.css").exists()


FORMULA_CHECK = """
const vm = require('vm'), fs = require('fs'), assert = require('assert');
const c = { PAGES: {}, Math, Object }; vm.createContext(c);
vm.runInContext(fs.readFileSync('common.js', 'utf8') + fs.readFileSync('explorer.js', 'utf8') + ';this.cf = compileFormula;', c);
const ev = (s, v) => { const r = c.cf(s); return JSON.stringify([r.f(v), r.params]); };
assert.strictEqual(ev('out / n * 8 / a', { out: 800, n: 4, a: 2 }), '[800,["a"]]');
assert.strictEqual(ev('a + b * 3 + c', { a: 1, b: 2, c: 3 }), '[10,["a","b","c"]]');
assert.strictEqual(ev('-2^2', {}), '[-4,[]]');
assert.strictEqual(ev('max(out, 1e3) / (conc - 1)', { out: 5, conc: 3 }), '[500,[]]');
for (const bad of ['out +', '(out', 'alert(1)', 'constructor(1)', 'a.constructor', '']) assert.throws(() => c.cf(bad), undefined, bad);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_custom_formula_is_parsed_not_evaluated():
    """The Explorer's custom Y formula comes from the URL: arithmetic and known functions only, never code."""
    viewer = Path(__file__).parents[1] / "furiosa_perf" / "reporting" / "viewer"
    subprocess.run(["node", "-e", FORMULA_CHECK], cwd=viewer, check=True)


PORTABLE_CHECK = """
const vm = require('vm'), fs = require('fs'), assert = require('assert');
const c = { PAGES: {}, document: { addEventListener() {} } }; vm.createContext(c);
vm.runInContext(fs.readFileSync('run.js', 'utf8') + ';this.portable = portable;', c);
const t = (raw, shown) => assert.strictEqual(c.portable(raw), shown);
t('/root/b/.vllm/bin/python -m vllm.entrypoints.cli.main bench serve --max-concurrency=64 --result-dir=/root/b/x --port=8000',
  'vllm bench serve --max-concurrency=64 --port=8000');
t('/root/venv/bin/vllm serve org/m --host 0.0.0.0', 'vllm serve org/m --host 0.0.0.0');
t('furiosa-llm serve org/m --devices npu:0', 'furiosa-llm serve org/m --devices npu:0');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_commands_are_shown_without_the_hosts_paths() -> None:
    """Run details name the tool rather than the measuring host's interpreter path, and drop its result directory."""
    viewer = Path(__file__).parents[1] / "furiosa_perf" / "reporting" / "viewer"
    subprocess.run(["node", "-e", PORTABLE_CHECK], cwd=viewer, check=True)
