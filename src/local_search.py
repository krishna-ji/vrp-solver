"""Local search operators for route improvement (hybrid optimization)."""

from __future__ import annotations
import random
from src.models import Route, Order
from src.distance import haversine
from src.constraints import check_time_window_feasibility, check_capacity


def two_opt(route: Route) -> Route:
    """Intra-route 2-opt: reverse a segment to reduce distance.

    Tries all (i, j) reversal pairs, applies the best improvement.
    Standard VRP local search move.
    """
    seq = route.order_sequence
    if len(seq) < 3:
        return route

    best_seq = list(seq)
    best_cost = _route_cost(route.driver, best_seq)
    improved = True

    while improved:
        improved = False
        for i in range(len(best_seq) - 1):
            for j in range(i + 2, len(best_seq)):
                new_seq = best_seq[:i] + best_seq[i : j + 1][::-1] + best_seq[j + 1 :]
                new_cost = _route_cost(route.driver, new_seq)
                if new_cost < best_cost - 1e-6:
                    candidate = Route(driver=route.driver, order_sequence=new_seq)
                    if check_time_window_feasibility(candidate):
                        best_seq = new_seq
                        best_cost = new_cost
                        improved = True

    new_route = Route(driver=route.driver, order_sequence=best_seq)
    return new_route


def or_opt(route: Route) -> Route:
    """Or-opt: relocate a subsequence of 1-3 orders to a better position."""
    seq = list(route.order_sequence)
    if len(seq) < 3:
        return route

    best_seq = list(seq)
    best_cost = _route_cost(route.driver, best_seq)

    for seg_len in (1, 2, 3):
        for i in range(len(best_seq) - seg_len + 1):
            segment = best_seq[i : i + seg_len]
            remaining = best_seq[:i] + best_seq[i + seg_len :]
            for j in range(len(remaining) + 1):
                candidate_seq = remaining[:j] + segment + remaining[j:]
                c = _route_cost(route.driver, candidate_seq)
                if c < best_cost - 1e-6:
                    candidate_route = Route(driver=route.driver, order_sequence=candidate_seq)
                    if check_time_window_feasibility(candidate_route):
                        best_seq = candidate_seq
                        best_cost = c

    return Route(driver=route.driver, order_sequence=best_seq)


def inter_route_relocate(routes: list[Route]) -> list[Route]:
    """Move an order from overloaded/long route to a shorter route if beneficial."""
    if len(routes) < 2:
        return routes

    routes = [Route(driver=r.driver, order_sequence=list(r.order_sequence)) for r in routes]

    # Try all pairs, not just longest→shortest
    best_improvement = 0.0
    best_move: tuple[int, int, int, int] | None = None  # (src_idx, order_idx, dst_idx, insert_pos)

    for si, src in enumerate(routes):
        if not src.order_sequence:
            continue
        src_cost = _route_cost(src.driver, src.order_sequence)

        for oi, order in enumerate(src.order_sequence):
            new_src_seq = src.order_sequence[:oi] + src.order_sequence[oi + 1:]
            new_src_cost = _route_cost(src.driver, new_src_seq)
            src_saving = src_cost - new_src_cost

            for di, dst in enumerate(routes):
                if di == si:
                    continue
                dst_weight = sum(o.weight_kg for o in dst.order_sequence)
                if dst_weight + order.weight_kg > dst.driver.capacity_kg:
                    continue
                if len(dst.order_sequence) >= dst.driver.max_orders:
                    continue

                old_dst_cost = _route_cost(dst.driver, dst.order_sequence)
                for pos in range(len(dst.order_sequence) + 1):
                    new_dst_seq = dst.order_sequence[:pos] + [order] + dst.order_sequence[pos:]
                    new_dst_cost = _route_cost(dst.driver, new_dst_seq)
                    dst_increase = new_dst_cost - old_dst_cost

                    improvement = src_saving - dst_increase
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (si, oi, di, pos)

    if best_move is not None:
        si, oi, di, pos = best_move
        order = routes[si].order_sequence.pop(oi)
        routes[di].order_sequence.insert(pos, order)
        # Verify feasibility
        if not (check_time_window_feasibility(routes[si])
                and check_time_window_feasibility(routes[di])
                and check_capacity(routes[di])):
            routes[di].order_sequence.remove(order)
            routes[si].order_sequence.insert(oi, order)

    return routes


def apply_local_search(routes: list[Route]) -> list[Route]:
    """Apply all local search operators to improve a solution."""
    # Intra-route: 2-opt + or-opt
    improved = [two_opt(r) for r in routes]
    improved = [or_opt(r) for r in improved]
    # Inter-route: relocate (try multiple passes)
    for _ in range(3):
        improved = inter_route_relocate(improved)
    return improved


def _route_cost(driver: object, orders: list[Order]) -> float:
    """Multi-objective cost for a single route.

    Weighted combination of distance + lateness + freshness, so local search
    improves all objectives instead of only distance.
    """
    if not orders:
        return 0.0
    from src.models import Location, OrderType
    from src.fitness import speed_at_time, freshness_penalty

    current = driver.start_location  # type: ignore[union-attr]
    current_time: float = driver.shift_start  # type: ignore[union-attr]
    dist = 0.0
    lateness = 0.0
    fresh_pen = 0.0

    for order in orders:
        d_pickup = haversine(current, order.pickup)  # type: ignore[arg-type]
        speed = speed_at_time(current_time)
        current_time += (d_pickup / speed) * 60
        dist += d_pickup

        ready_time = order.earliest_pickup + order.prep_time_min
        effective_earliest = max(order.earliest_pickup, ready_time)
        if current_time < effective_earliest:
            current_time = effective_earliest

        d_dropoff = haversine(order.pickup, order.dropoff)
        speed = speed_at_time(current_time)
        current_time += (d_dropoff / speed) * 60
        dist += d_dropoff

        delivery_time = current_time
        if delivery_time > order.latest_delivery:
            lateness += delivery_time - order.latest_delivery

        fresh_pen += freshness_penalty(order, delivery_time)
        current = order.dropoff

    # Weighted sum: distance + lateness (high weight) + freshness
    return dist + lateness * 2.0 + fresh_pen * 1.5
