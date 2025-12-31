"""Multi-objective fitness evaluation for VRP solutions.

Objectives (all minimized):
  1. Total fleet distance (km)
  2. Max delivery lateness (min)
  3. Total idle time (min)
  4. Workload unfairness: max_route_duration - min_route_duration (min)
"""

from __future__ import annotations
from src.models import Solution, Route, Driver, Order, Location, OrderType
from src.distance import haversine
from src.constraints import total_penalty as _route_penalty


def speed_at_time(time_min: float) -> float:
    """Time-of-day speed (km/h) modeling Kathmandu traffic.

    Peak hours:  8-10 AM (480-600), 5-7 PM (1020-1140) → 12 km/h
    Off-peak:    10 PM - 6 AM (1320-360) → 35 km/h
    Normal:      everything else → 25 km/h
    """
    hour = (time_min / 60) % 24
    if 8 <= hour < 10 or 17 <= hour < 19:
        return 12.0
    if 22 <= hour or hour < 6:
        return 35.0
    return 25.0


def freshness_penalty(order: Order, delivery_time: float) -> float:
    """Food quality degradation penalty.

    Food should arrive ASAP after prep completion.
    Linear penalty for first 30 min, quadratic growth after that.
    Non-food orders: 0 penalty.
    """
    if order.order_type != OrderType.FOOD:
        return 0.0
    elapsed = delivery_time - order.earliest_pickup - order.prep_time_min
    if elapsed <= 0:
        return 0.0
    if elapsed <= 30:
        return elapsed * 0.05  # mild linear penalty: 0 → 1.5 over 30 min
    return 1.5 + (elapsed - 30) ** 2 * 0.1  # quadratic after 30 min


def evaluate_solution(routes: list[Route]) -> tuple[float, float, float, float]:
    """Compute (total_distance, max_lateness, total_idle, unfairness).

    Constraint violations (capacity, max orders) are added as penalties
    to distance and lateness so infeasible solutions are dominated.
    """
    total_dist = 0.0
    max_lateness = 0.0
    total_idle = 0.0
    total_freshness = 0.0
    route_durations: list[float] = []
    total_constraint_penalty = 0.0

    for route in routes:
        dist, lateness, idle, fresh, duration = _evaluate_route(route)
        total_dist += dist
        max_lateness = max(max_lateness, lateness)
        total_idle += idle
        total_freshness += fresh
        route_durations.append(duration)
        total_constraint_penalty += _route_penalty(route)

    # Workload fairness: minimize gap between busiest and least busy driver
    active_durations = [d for d in route_durations if d > 0]
    if len(active_durations) >= 2:
        unfairness = max(active_durations) - min(active_durations)
    else:
        unfairness = 0.0

    # Fold freshness into lateness (food-aware lateness)
    effective_lateness = max_lateness + total_freshness

    # Add constraint penalties so infeasible solutions are dominated
    total_dist += total_constraint_penalty
    effective_lateness += total_constraint_penalty

    return total_dist, effective_lateness, total_idle, unfairness


def _evaluate_route(route: Route) -> tuple[float, float, float, float, float]:
    """Evaluate a single route.

    Returns (distance_km, max_lateness_min, idle_min, freshness, duration_min).
    """
    if not route.order_sequence:
        idle = route.driver.shift_end - route.driver.shift_start
        return 0.0, 0.0, idle, 0.0, 0.0

    driver = route.driver
    current_loc = driver.start_location
    current_time = driver.shift_start
    start_time = current_time
    total_dist = 0.0
    max_lateness = 0.0
    total_idle = 0.0
    total_freshness = 0.0

    for order in route.order_sequence:
        # Drive to pickup
        d_pickup = haversine(current_loc, order.pickup)
        speed = speed_at_time(current_time)
        travel_time = (d_pickup / speed) * 60
        arrival_time = current_time + travel_time
        total_dist += d_pickup

        # Wait if arrived early or food not ready
        ready_time = order.earliest_pickup + order.prep_time_min
        effective_earliest = max(order.earliest_pickup, ready_time)
        if arrival_time < effective_earliest:
            total_idle += effective_earliest - arrival_time
            arrival_time = effective_earliest

        current_time = arrival_time

        # Drive to dropoff
        d_dropoff = haversine(order.pickup, order.dropoff)
        speed = speed_at_time(current_time)
        travel_time = (d_dropoff / speed) * 60
        delivery_time = current_time + travel_time
        total_dist += d_dropoff

        # Lateness
        if delivery_time > order.latest_delivery:
            max_lateness = max(max_lateness, delivery_time - order.latest_delivery)

        # Freshness penalty for food
        total_freshness += freshness_penalty(order, delivery_time)

        current_time = delivery_time
        current_loc = order.dropoff

    # Remaining shift time is idle
    if current_time < driver.shift_end:
        total_idle += driver.shift_end - current_time

    duration = current_time - start_time
    return total_dist, max_lateness, total_idle, total_freshness, duration


# --- Extended metrics for benchmarking ---

# CO₂ emission factors (kg CO₂ per km) — IPCC urban estimates
_CO2_PER_KM = {
    "bike": 0.021,   # motorcycle
    "car": 0.170,    # small car, urban driving
}


def detailed_metrics(routes: list[Route]) -> dict[str, float]:
    """Compute extended evaluation metrics from a set of routes.

    Returns a dict with all metrics for benchmark comparison.
    """
    from src.distance import haversine

    total_orders = sum(len(r.order_sequence) for r in routes)
    active_routes = [r for r in routes if r.order_sequence]
    n_active = len(active_routes)
    total_drivers = len(routes)

    # Per-route stats
    on_time = 0
    food_fresh = 0
    food_total = 0
    delivery_times: list[float] = []
    route_distances: list[float] = []
    route_durations: list[float] = []
    total_co2 = 0.0
    first_pickup = float("inf")
    last_delivery = 0.0
    total_shift = 0.0

    for route in routes:
        total_shift += route.driver.shift_end - route.driver.shift_start
        vtype = route.driver.vehicle_type.value  # "bike" or "car"

        if not route.order_sequence:
            continue

        current_loc = route.driver.start_location
        current_time = route.driver.shift_start
        start_time = current_time
        route_dist = 0.0

        for order in route.order_sequence:
            # Pickup
            d_pickup = haversine(current_loc, order.pickup)
            speed = speed_at_time(current_time)
            current_time += (d_pickup / speed) * 60
            route_dist += d_pickup

            ready_time = order.earliest_pickup + order.prep_time_min
            effective_earliest = max(order.earliest_pickup, ready_time)
            if current_time < effective_earliest:
                current_time = effective_earliest

            first_pickup = min(first_pickup, current_time)
            pickup_time = current_time

            # Dropoff
            d_dropoff = haversine(order.pickup, order.dropoff)
            speed = speed_at_time(current_time)
            current_time += (d_dropoff / speed) * 60
            route_dist += d_dropoff

            delivery_time = current_time
            last_delivery = max(last_delivery, delivery_time)
            delivery_times.append(delivery_time - pickup_time)

            # On-time?
            if delivery_time <= order.latest_delivery:
                on_time += 1

            # Food freshness (within 30 min of prep)
            if order.order_type == OrderType.FOOD:
                food_total += 1
                elapsed = delivery_time - order.earliest_pickup - order.prep_time_min
                if elapsed <= 30:
                    food_fresh += 1

            current_loc = order.dropoff

        route_distances.append(route_dist)
        route_durations.append(current_time - start_time)
        total_co2 += route_dist * _CO2_PER_KM.get(vtype, 0.1)

    total_dist = sum(route_distances) if route_distances else 0.0
    total_active_time = sum(route_durations) if route_durations else 0.0

    return {
        "on_time_rate": (on_time / total_orders * 100) if total_orders else 0.0,
        "fleet_utilization": (total_active_time / total_shift * 100) if total_shift else 0.0,
        "food_freshness_rate": (food_fresh / food_total * 100) if food_total else 0.0,
        "avg_delivery_time": (sum(delivery_times) / len(delivery_times)) if delivery_times else 0.0,
        "makespan": (last_delivery - first_pickup) if first_pickup < float("inf") else 0.0,
        "active_drivers": n_active,
        "total_drivers": total_drivers,
        "avg_orders_per_driver": (total_orders / n_active) if n_active else 0.0,
        "max_route_distance": max(route_distances) if route_distances else 0.0,
        "co2_kg": total_co2,
    }
