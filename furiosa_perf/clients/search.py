"""Find the largest load that still passes, by bisection.

Used for capacity: the most users a server holds while every point stays valid and
meets the SLO. It assumes passing is monotonic in the load (more users never makes
latency better); a noisy point near the edge can shift the answer by one step.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SearchOutcome:
    """Result of :func:`bisect_capacity`.

    Attributes:
        capacity: Largest value measured as passing, ``None`` if even ``minimum`` failed.
        bound: ``"exact"`` when the edge lies within ``step`` of ``capacity``;
            ``"below_min"`` when ``minimum`` already failed; ``"at_least_max"`` when
            ``maximum`` passed, so the real capacity may be higher.
        minimum: Lower end of the searched range.
        maximum: Upper end of the searched range.
        step: Resolution: the search stops when the passing and failing values are this close.
        probes: Every measured value with its verdict, in order.
    """

    capacity: int | None
    bound: str
    minimum: int
    maximum: int
    step: int
    probes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The outcome as plain data for ``run.json``."""
        return asdict(self)


def max_probes(minimum: int, maximum: int, step: int) -> int:
    """Upper bound on the measurements :func:`bisect_capacity` makes for a range."""
    count, gap = 2, maximum - minimum
    while gap > step:
        gap = (gap + 1) // 2
        count += 1
    return count


def bisect_capacity(minimum: int, maximum: int, step: int, passes: Callable[[int], bool]) -> SearchOutcome:
    """Bisect ``[minimum, maximum]`` for the largest value where ``passes`` holds.

    ``minimum`` and ``maximum`` are measured first, so a range that does not contain
    the edge is reported as such instead of being searched.

    Args:
        minimum: Smallest value to try; expected to pass.
        maximum: Largest value to try; expected to fail.
        step: Stop when the largest passing and smallest failing values are at most this far apart.
        passes: Measures one value and says whether it passed.

    Returns:
        The outcome, with every probe recorded.
    """
    outcome = SearchOutcome(capacity=None, bound="below_min", minimum=minimum, maximum=maximum, step=step)

    def probe(value: int) -> bool:
        passed = passes(value)
        outcome.probes.append({"value": value, "passed": passed})
        return passed

    if not probe(minimum):
        return outcome
    if probe(maximum):
        outcome.capacity, outcome.bound = maximum, "at_least_max"
        return outcome
    low, high = minimum, maximum  # low passed, high failed
    while high - low > step:
        middle = (low + high) // 2
        if probe(middle):
            low = middle
        else:
            high = middle
    outcome.capacity, outcome.bound = low, "exact"
    return outcome
