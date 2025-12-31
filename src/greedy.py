"""Greedy nearest-neighbor baseline for comparison."""

from __future__ import annotations
from src.models import Order, Driver, Route, Location
from src.distance import haversine


def greedy_nearest_neighbor(
    orders: list[Order], drivers: list[Driver]
) -> list[Route]:
    """Assign orders to drivers greedily by nearest pickup.

    Drivers are sorted by capacity (largest first) so heavy parcels
    go to vehicles that can carry them.
    """
    remaining = list(orders)
    sorted_drivers = sorted(drivers, key=lambda d: d.capacity_kg, reverse=True)
    routes: list[Route] = []

    for driver in sorted_drivers:
        route = Route(driver=driver)
        current_loc = driver.start_location
        current_weight = 0.0

        while remaining:
            if len(route.order_sequence) >= driver.max_orders:
                break
            # Find nearest feasible order
            best_idx = -1
            best_dist = float("inf")
            for i, order in enumerate(remaining):
                if current_weight + order.weight_kg > driver.capacity_kg:
                    continue
                d = haversine(current_loc, order.pickup)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            if best_idx == -1:
                break

            order = remaining.pop(best_idx)
            route.order_sequence.append(order)
            current_weight += order.weight_kg
            current_loc = order.dropoff

        routes.append(route)

    # Distribute remaining orders across drivers that can still accept them
    for order in list(remaining):
        best_route = None
        best_dist = float("inf")
        for route in routes:
            weight = sum(o.weight_kg for o in route.order_sequence)
            if (len(route.order_sequence) >= route.driver.max_orders
                    or weight + order.weight_kg > route.driver.capacity_kg):
                continue
            loc = route.order_sequence[-1].dropoff if route.order_sequence else route.driver.start_location
            d = haversine(loc, order.pickup)
            if d < best_dist:
                best_dist = d
                best_route = route
        if best_route is not None:
            best_route.order_sequence.append(order)
            remaining.remove(order)

    # Last resort: spread across drivers with fewest orders (infeasible, but fair)
    if remaining:
        for order in remaining:
            route = min(routes, key=lambda r: len(r.order_sequence))
            route.order_sequence.append(order)

    return routes
