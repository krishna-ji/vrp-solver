"""NSGA-II engine — non-dominated sorting and crowding distance."""

from __future__ import annotations
import random
from src.models import Solution


def fast_non_dominated_sort(population: list[Solution]) -> list[list[Solution]]:
    """Assign Pareto rank to each solution. Returns list of fronts."""
    n = len(population)
    domination_count: list[int] = [0] * n
    dominated_set: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[Solution]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(population[i], population[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _dominates(population[j], population[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

    # First front
    front_indices = [i for i in range(n) if domination_count[i] == 0]
    for i in front_indices:
        population[i].rank = 0
    fronts.append([population[i] for i in front_indices])

    k = 0
    while front_indices:
        next_front: list[int] = []
        for i in front_indices:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = k + 1
                    next_front.append(j)
        k += 1
        if next_front:
            fronts.append([population[j] for j in next_front])
        front_indices = next_front

    return fronts


def crowding_distance(front: list[Solution]) -> None:
    """Assign crowding distance to each solution in a front (in-place)."""
    n = len(front)
    if n <= 2:
        for s in front:
            s.crowding_distance = float("inf")
        return

    for s in front:
        s.crowding_distance = 0.0

    n_obj = len(front[0].objectives)
    for m in range(n_obj):
        front.sort(key=lambda s: s.objectives[m])
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        obj_range = front[-1].objectives[m] - front[0].objectives[m]
        if obj_range == 0:
            continue
        for i in range(1, n - 1):
            front[i].crowding_distance += (
                (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
            )


def tournament_select(population: list[Solution], k: int = 2) -> Solution:
    """Binary tournament selection on rank then crowding distance."""
    candidates = random.sample(population, k)
    candidates.sort(key=lambda s: (s.rank, -s.crowding_distance))
    return candidates[0]


def _dominates(a: Solution, b: Solution) -> bool:
    """True if a dominates b (all objectives <=, at least one <). Minimization."""
    dominated = False
    for ai, bi in zip(a.objectives, b.objectives, strict=True):
        if ai > bi:
            return False
        if ai < bi:
            dominated = True
    return dominated
