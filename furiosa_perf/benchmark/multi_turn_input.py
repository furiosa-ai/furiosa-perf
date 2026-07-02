"""Build the vendored multi-turn benchmark's input document from a scenario.

The vendored ``benchmark_serving_multi_turn.py`` consumes a
``generate_multi_turn.json`` describing a synthetic-conversation workload. Rather
than ship separate JSON files, furiosa-perf assembles that document at run time
from the inline fields of :class:`~furiosa_perf.utils.config.MultiTurnScenarioConfig`.

This module is intentionally free of any subprocess / server dependency so it can
be unit-tested and its output validated against the vendored
``bench_dataset.parse_input_json_file``.
"""

from math import ceil, floor
from pathlib import Path
from typing import Any

from furiosa_perf.utils.config import MultiTurnScenarioConfig
from furiosa_perf.utils.logger import logger

_ALLOWED_DISTRIBUTIONS = {"constant", "uniform", "lognormal", "zipf", "poisson"}


def _dist(spec: int | float | dict[str, Any]) -> dict[str, Any]:
    """Normalise a shape spec into a bench_dataset distribution dict.

    Args:
        spec: A scalar (interpreted as a ``constant`` distribution) or a dict already
            describing a distribution (passed through after validation).

    Returns:
        dict[str, Any]: A distribution dict suitable for ``generate_multi_turn.json``.

    Raises:
        ValueError: If ``spec`` is a dict without a supported ``distribution`` key.
    """
    if isinstance(spec, dict):
        distribution = spec.get("distribution")
        if distribution not in _ALLOWED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution {distribution!r}; expected one of {sorted(_ALLOWED_DISTRIBUTIONS)}"
            )
        return spec
    return {"distribution": "constant", "value": spec}


def _representative(spec: int | float | dict[str, Any]) -> float:
    """Best-effort single-value estimate of a shape spec (for context sizing).

    Args:
        spec: A scalar or distribution dict.

    Returns:
        float: A representative value — the constant value, the uniform midpoint, the
        lognormal average, or the alpha of zipf/poisson (capped by ``max`` when given).
    """
    if not isinstance(spec, dict):
        return float(spec)
    dist = spec.get("distribution")
    if dist == "constant":
        return float(spec["value"])
    if dist == "uniform":
        return (float(spec["min"]) + float(spec["max"])) / 2.0
    if dist == "lognormal":
        return float(spec.get("average", spec.get("mean", 0.0)))
    if dist in ("zipf", "poisson"):
        val = float(spec.get("alpha", 0.0))
        if spec.get("max") is not None:
            val = min(val, float(spec["max"]))
        return val
    return 0.0


def _peak_input_tokens(
    num_turns: float, prefix: float, common_prefix: float, per_turn_input: float, per_turn_output: float
) -> float:
    """Estimate the peak request input size (tokens) at the final user turn.

    The tool resends the full chat history each turn, so the last user turn carries
    the unique prefix, the optional shared prefix, and every prior user/assistant
    message.

    Args:
        num_turns: Total number of messages in the conversation.
        prefix: Per-conversation unique prefix length in tokens.
        common_prefix: Shared prefix length in tokens.
        per_turn_input: Tokens per user turn.
        per_turn_output: Tokens per assistant turn.

    Returns:
        float: Estimated peak input tokens for the conversation.
    """
    num_user = ceil(num_turns / 2)
    num_asst = floor(num_turns / 2)
    return common_prefix + prefix + num_user * per_turn_input + num_asst * per_turn_output


def build_multi_turn_input(scenario: MultiTurnScenarioConfig, text_file_path: Path) -> dict[str, Any]:
    """Assemble the complete ``generate_multi_turn.json`` document for a scenario.

    Injects the required top-level fields the vendored parser asserts on
    (``filetype`` / ``num_conversations`` / ``text_files``), builds the
    ``prompt_input`` / ``prompt_output`` sections from the scenario's shape fields,
    and — when :attr:`MultiTurnScenarioConfig.max_input_context_tokens` is set —
    clamps the workload down to fit that context budget.

    Args:
        scenario: The multi-turn scenario describing the workload inline.
        text_file_path: Path to the token-source corpus (vendored ``pg1184.txt``);
            written into the document as an absolute path so it resolves regardless
            of the subprocess working directory.

    Returns:
        dict[str, Any]: The document, ready to ``json.dump`` and pass via
        ``--input-file``.
    """
    prompt_input: dict[str, Any] = {
        "num_turns": _dist(scenario.num_turns),
        "prefix_num_tokens": _dist(scenario.prefix_num_tokens),
        "num_tokens": _dist(scenario.input_num_tokens),
    }
    if scenario.common_prefix_num_tokens is not None:
        prompt_input["common_prefix_num_tokens"] = _dist(scenario.common_prefix_num_tokens)

    doc: dict[str, Any] = {
        "filetype": "generate_conversations",
        "num_conversations": scenario.num_conversations,
        "text_files": [str(text_file_path.resolve())],
        "print_stats": False,
        "prompt_input": prompt_input,
        "prompt_output": {"num_tokens": _dist(scenario.output_num_tokens)},
    }

    if scenario.max_input_context_tokens is not None:
        _clamp_input_context(doc, scenario)

    return doc


def _clamp_input_context(doc: dict[str, Any], scenario: MultiTurnScenarioConfig) -> None:
    """Shrink the workload in place so the peak input fits the context budget.

    Reduces, in order and recomputing after each step: the per-conversation prefix,
    then the number of turns (kept even and ``>= 2``), then the per-turn input size.
    Clamped fields are rewritten as ``constant`` distributions (a clamp intentionally
    collapses a distribution to keep the guarantee). Only the constant path is exact;
    for other distributions the representative value is used and a warning is logged.

    Args:
        doc: The document produced by :func:`build_multi_turn_input` (mutated).
        scenario: The scenario carrying ``max_input_context_tokens`` and the output cap.
    """
    cap = scenario.max_input_context_tokens
    assert cap is not None

    turns = _representative(scenario.num_turns)
    prefix = _representative(scenario.prefix_num_tokens)
    per_in = _representative(scenario.input_num_tokens)
    per_out = _representative(scenario.output_num_tokens)
    common = _representative(scenario.common_prefix_num_tokens) if scenario.common_prefix_num_tokens is not None else 0.0
    output_cap = float(scenario.limit_max_tokens) if scenario.limit_max_tokens >= 1 else per_out

    budget = cap - output_cap  # leave room for the answer being generated
    peak = _peak_input_tokens(turns, prefix, common, per_in, per_out)
    if peak <= budget:
        return

    non_constant = any(
        isinstance(s, dict) and s.get("distribution") != "constant"
        for s in (scenario.num_turns, scenario.prefix_num_tokens, scenario.input_num_tokens, scenario.output_num_tokens)
    )
    if non_constant:
        logger.warning(
            "multi_turn: input-context clamp on a non-constant distribution is best-effort; "
            "clamped fields are collapsed to constants."
        )

    # 1) shrink the unique prefix toward zero.
    prefix = max(0.0, budget - common - ceil(turns / 2) * per_in - floor(turns / 2) * per_out)
    prefix = min(prefix, _representative(scenario.prefix_num_tokens))

    # 2) if still over budget, drop turns (keep even, >= 2).
    while _peak_input_tokens(turns, prefix, common, per_in, per_out) > budget and turns > 2:
        turns = max(2, (int(turns) - 2) // 2 * 2 if int(turns) % 2 else int(turns) - 2)

    # 3) if still over, shrink per-turn input tokens (floor 1).
    if _peak_input_tokens(turns, prefix, common, per_in, per_out) > budget:
        num_user = max(1, ceil(turns / 2))
        room = budget - common - prefix - floor(turns / 2) * per_out
        per_in = max(1.0, room / num_user)

    final_peak = _peak_input_tokens(turns, prefix, common, per_in, per_out)
    logger.info(
        "multi_turn: clamped workload to context cap %d (peak %d -> %d): "
        "num_turns=%d prefix_num_tokens=%d input_num_tokens=%d",
        cap, int(peak), int(final_peak), int(turns), int(prefix), int(per_in),
    )
    assert final_peak <= budget + 1, f"clamp failed: peak {final_peak} > budget {budget}"

    doc["prompt_input"]["num_turns"] = {"distribution": "constant", "value": int(turns)}
    doc["prompt_input"]["prefix_num_tokens"] = {"distribution": "constant", "value": int(prefix)}
    doc["prompt_input"]["num_tokens"] = {"distribution": "constant", "value": int(per_in)}
