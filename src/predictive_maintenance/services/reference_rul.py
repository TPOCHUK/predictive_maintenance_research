"""Offline FD001 labels for demonstration only; never used as model inputs."""

from functools import lru_cache
from pathlib import Path

from predictive_maintenance.services.simulation import dataset


@lru_cache(maxsize=1)
def failure_cycles() -> dict[int, int]:
    path = Path(__file__).resolve().parents[3] / "data" / "RUL_FD001.txt"
    labels = [int(value) for value in path.read_text().split()]
    last_cycles = dataset().groupby("unit_id").cycle.max().sort_index()
    if len(labels) != len(last_cycles):
        raise ValueError("FD001 labels do not match the test engine count")
    return {
        int(unit): int(last) + remaining
        for (unit, last), remaining in zip(last_cycles.items(), labels)
    }


def reference_rul(unit_id: int, cycle: int) -> int | None:
    failure = failure_cycles().get(unit_id)
    if failure is None or cycle <= 0 or cycle > failure:
        return None
    return failure - cycle
