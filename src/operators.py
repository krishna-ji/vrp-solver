"""VRP-specific genetic operators: crossover and mutation."""

from __future__ import annotations
import random
from src.models import Order


def order_crossover(parent_a: list[Order], parent_b: list[Order]) -> list[Order]:
    """Order crossover (OX1) preserving relative order."""
    n = len(parent_a)
    if n < 2:
        return list(parent_a)
    start, end = sorted(random.sample(range(n), 2))
    child: list[Order | None] = [None] * n
    child[start : end + 1] = parent_a[start : end + 1]
    placed = set(o.order_id for o in parent_a[start : end + 1])

    pos = (end + 1) % n
    for order in parent_b[end + 1 :] + parent_b[: end + 1]:
        if order.order_id not in placed:
            child[pos] = order
            placed.add(order.order_id)
            pos = (pos + 1) % n

    return [o for o in child if o is not None]


def pmx_crossover(parent_a: list[Order], parent_b: list[Order]) -> list[Order]:
    """Partially Mapped Crossover (PMX) — preserves absolute positions.

    Better than OX1 when position within route matters (e.g., time-window
    sensitive food deliveries where sequence directly affects lateness).
    Falls back to OX1 when parents have different lengths.
    """
    n = len(parent_a)
    if n < 2:
        return list(parent_a)
    if len(parent_b) != n:
        return order_crossover(parent_a, parent_b)

    start, end = sorted(random.sample(range(n), 2))

    # Build mapping from the crossover segment
    child: list[Order | None] = [None] * n
    child[start : end + 1] = parent_a[start : end + 1]

    mapping: dict[str, str] = {}
    for i in range(start, end + 1):
        mapping[parent_a[i].order_id] = parent_b[i].order_id
        mapping[parent_b[i].order_id] = parent_a[i].order_id

    placed_ids = {o.order_id for o in parent_a[start : end + 1]}
    id_to_order = {o.order_id: o for o in parent_a + parent_b}
    all_orders = list(parent_a)  # full order set for fallback

    for i in list(range(0, start)) + list(range(end + 1, n)):
        candidate = parent_b[i]
        # Follow the mapping chain to find an unplaced order
        seen: set[str] = set()
        while candidate.order_id in placed_ids:
            seen.add(candidate.order_id)
            mapped_id = mapping.get(candidate.order_id, "")
            mapped_order = id_to_order.get(mapped_id)
            if mapped_order is None or mapped_order.order_id in seen:
                # Chain broke or looped — pick any unplaced order
                candidate = next(
                    (o for o in all_orders if o.order_id not in placed_ids),
                    candidate,
                )
                break
            candidate = mapped_order
        child[i] = candidate
        placed_ids.add(candidate.order_id)

    return [o for o in child if o is not None]


def swap_mutation(sequence: list[Order], prob: float = 0.1) -> list[Order]:
    """Swap two random orders with given probability."""
    seq = list(sequence)
    if random.random() < prob and len(seq) >= 2:
        i, j = random.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]
    return seq


def inversion_mutation(sequence: list[Order], prob: float = 0.05) -> list[Order]:
    """Reverse a random subsequence — the mutation counterpart of 2-opt."""
    seq = list(sequence)
    if random.random() < prob and len(seq) >= 3:
        i, j = sorted(random.sample(range(len(seq)), 2))
        seq[i : j + 1] = seq[i : j + 1][::-1]
    return seq


def route_transfer_mutation(
    routes: list[list[Order]], prob: float = 0.05
) -> list[list[Order]]:
    """Move one or more orders between routes, or swap orders across routes."""
    routes = [list(r) for r in routes]
    if random.random() >= prob:
        return routes

    non_empty = [i for i, r in enumerate(routes) if r]
    if len(non_empty) < 2:
        return routes

    action = random.random()

    if action < 0.5:
        # Single transfer (original behavior)
        src = random.choice(non_empty)
        dst = random.choice([i for i in range(len(routes)) if i != src])
        order = routes[src].pop(random.randrange(len(routes[src])))
        routes[dst].insert(random.randrange(len(routes[dst]) + 1), order)

    elif action < 0.8:
        # Multi-order transfer: move 1-3 consecutive orders
        src = random.choice(non_empty)
        dst = random.choice([i for i in range(len(routes)) if i != src])
        n_move = min(random.randint(1, 3), len(routes[src]))
        start_idx = random.randrange(len(routes[src]) - n_move + 1)
        moved = routes[src][start_idx : start_idx + n_move]
        del routes[src][start_idx : start_idx + n_move]
        insert_pos = random.randrange(len(routes[dst]) + 1)
        routes[dst][insert_pos:insert_pos] = moved

    else:
        # Swap: exchange one order between two routes
        src = random.choice(non_empty)
        other_non_empty = [i for i in non_empty if i != src]
        if other_non_empty:
            dst = random.choice(other_non_empty)
            si = random.randrange(len(routes[src]))
            di = random.randrange(len(routes[dst]))
            routes[src][si], routes[dst][di] = routes[dst][di], routes[src][si]

    return routes
