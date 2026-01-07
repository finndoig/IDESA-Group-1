"""
planner.py
Exact route planning for a small number of targets (≈5).

We brute-force all permutations and pick the shortest path length.
This is feasible for small N because 5! = 120 permutations.
"""

from __future__ import annotations
import itertools
import math
from typing import List, Tuple, Optional

Point = Tuple[float, float]

# function to compute distance between two points
def _dist(a: Point, b: Point) -> float:
    """Euclidean distance in world units (metres)."""
    return math.hypot(b[0] - a[0], b[1] - a[1])

# function to plan optimal order of targets
def plan_optimal_order(
    targets: List[Point],
    start: Point,
    end: Optional[Point] = None
) -> List[Point]:
    """
    Returns the targets reordered to minimise total path length.

    Path definition:
      start -> targets (in some order) -> end (optional)

    If end is None, route ends at the last target.
    """
    if not targets:
        return []

    best_order = None
    best_cost = float("inf")

    for perm in itertools.permutations(targets):
        cost = 0.0

        # start -> first
        cost += _dist(start, perm[0])

        # between targets
        for a, b in zip(perm[:-1], perm[1:]):
            cost += _dist(a, b)

        # last -> end (if required)
        if end is not None:
            cost += _dist(perm[-1], end)

        if cost < best_cost:
            best_cost = cost
            best_order = perm

    return list(best_order)
