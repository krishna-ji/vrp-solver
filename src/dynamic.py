"""Dynamic re-optimization: insert new orders into existing routes."""

from __future__ import annotations
from src.models import Order, Route, Location
from src.distance import haversine


def cheapest_insertion(order: Order, routes: list[Route]) -> list[Route]:
    """Insert a new order into the best position across all routes.

    For each route, finds the position that increases total distance the least.
    Respects capacity constraints. Used for real-time order insertion
    between full NSGA-II optimization runs.
    """
    best_route_idx = -1
    best_position = 0
    best_cost = float("inf")

    for ri, route in enumerate(routes):
        # Check capacity
        total_weight = sum(o.weight_kg for o in route.order_sequence) + order.weight_kg
        if total_weight > route.driver.capacity_kg:
            continue

        # Check max orders
        if len(route.order_sequence) >= route.driver.max_orders:
            continue

        # Try every insertion position
        for pos in range(len(route.order_sequence) + 1):
            cost = _insertion_cost(route, order, pos)
            if cost < best_cost:
                best_cost = cost
                best_route_idx = ri
                best_position = pos

    if best_route_idx == -1:
        # No feasible insertion — add to least loaded route (allow overflow)
        loads = [(sum(o.weight_kg for o in r.order_sequence), i) for i, r in enumerate(routes)]
        loads.sort()
        best_route_idx = loads[0][1]
        best_position = len(routes[best_route_idx].order_sequence)

    # Create new routes with insertion
    new_routes = []
    for i, route in enumerate(routes):
        if i == best_route_idx:
            new_seq = list(route.order_sequence)
            new_seq.insert(best_position, order)
            new_routes.append(Route(driver=route.driver, order_sequence=new_seq))
        else:
            new_routes.append(Route(driver=route.driver, order_sequence=list(route.order_sequence)))

    return new_routes


def batch_insert(new_orders: list[Order], routes: list[Route]) -> list[Route]:
    """Insert multiple new orders one by one using cheapest insertion.

    Orders are sorted by urgency (earliest deadline first) before insertion.
    This provides a warm-start for the next NSGA-II optimization window.
    """
    sorted_orders = sorted(new_orders, key=lambda o: o.latest_delivery)
    current_routes = routes
    for order in sorted_orders:
        current_routes = cheapest_insertion(order, current_routes)
    return current_routes


def _insertion_cost(route: Route, order: Order, position: int) -> float:
    """Extra distance (km) caused by inserting order at position in route."""
    seq = route.order_sequence
    start_loc = route.driver.start_location

    if not seq:
        # Empty route: cost = start→pickup + pickup→dropoff
        return (
            haversine(start_loc, order.pickup)
            + haversine(order.pickup, order.dropoff)
        )

    # Location before insertion point
    if position == 0:
        prev_loc = start_loc
    else:
        prev_loc = seq[position - 1].dropoff

    # Location after insertion point
    if position < len(seq):
        next_loc = seq[position].pickup
    else:
        next_loc = seq[-1].dropoff  # end of route

    # Old cost: prev → next
    old_cost = haversine(prev_loc, next_loc) if position < len(seq) else 0.0

    # New cost: prev → pickup → dropoff → next
    new_cost = (
        haversine(prev_loc, order.pickup)
        + haversine(order.pickup, order.dropoff)
        + (haversine(order.dropoff, next_loc) if position < len(seq) else 0.0)
    )

    return new_cost - old_cost
