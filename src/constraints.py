"""Constraint checking for VRP solutions."""

from __future__ import annotations
from src.models import Route, Order, OrderType


def check_capacity(route: Route) -> bool:
    """True if total weight of orders <= driver capacity."""
    total = sum(o.weight_kg for o in route.order_sequence)
    return total <= route.driver.capacity_kg


def check_max_orders(route: Route) -> bool:
    """True if number of orders <= driver's max_orders."""
    return len(route.order_sequence) <= route.driver.max_orders


def check_time_window_feasibility(route: Route) -> bool:
    """True if driver can physically visit all stops within shift."""
    from src.fitness import speed_at_time
    from src.distance import haversine

    if not route.order_sequence:
        return True

    current_time = route.driver.shift_start
    current_loc = route.driver.start_location

    for order in route.order_sequence:
        d = haversine(current_loc, order.pickup)
        speed = speed_at_time(current_time)
        current_time += (d / speed) * 60
        ready_time = order.earliest_pickup + order.prep_time_min
        current_time = max(current_time, ready_time)
        d2 = haversine(order.pickup, order.dropoff)
        speed = speed_at_time(current_time)
        current_time += (d2 / speed) * 60
        current_loc = order.dropoff

    return current_time <= route.driver.shift_end


def check_food_freshness(route: Route, max_elapsed_min: float = 45.0) -> bool:
    """True if all food orders delivered within max_elapsed_min of prep completion."""
    from src.fitness import speed_at_time
    from src.distance import haversine

    if not route.order_sequence:
        return True

    current_time = route.driver.shift_start
    current_loc = route.driver.start_location

    for order in route.order_sequence:
        d = haversine(current_loc, order.pickup)
        speed = speed_at_time(current_time)
        current_time += (d / speed) * 60
        ready_time = order.earliest_pickup + order.prep_time_min
        current_time = max(current_time, ready_time)
        d2 = haversine(order.pickup, order.dropoff)
        speed = speed_at_time(current_time)
        delivery_time = current_time + (d2 / speed) * 60

        if order.order_type == OrderType.FOOD:
            elapsed = delivery_time - ready_time
            if elapsed > max_elapsed_min:
                return False

        current_time = delivery_time
        current_loc = order.dropoff

    return True


def total_penalty(route: Route) -> float:
    """Penalty score for constraint violations (0 = feasible)."""
    penalty = 0.0

    # Capacity violation
    total_weight = sum(o.weight_kg for o in route.order_sequence)
    if total_weight > route.driver.capacity_kg:
        penalty += (total_weight - route.driver.capacity_kg) * 100.0

    # Max orders violation
    excess_orders = len(route.order_sequence) - route.driver.max_orders
    if excess_orders > 0:
        penalty += excess_orders * 50.0

    return penalty
