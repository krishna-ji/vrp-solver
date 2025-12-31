"""Tests for VRP solver core components."""

from __future__ import annotations
import random

import pytest

from src.models import Location, Order, Driver, Route, Solution, OrderType, VehicleType
from src.fitness import evaluate_solution, freshness_penalty, speed_at_time
from src.operators import order_crossover, pmx_crossover, swap_mutation, inversion_mutation, route_transfer_mutation
from src.constraints import check_capacity, check_max_orders, check_time_window_feasibility, total_penalty
from src.local_search import two_opt, or_opt, inter_route_relocate, apply_local_search
from src.greedy import greedy_nearest_neighbor
from src.dynamic import cheapest_insertion, batch_insert
from src.distance import haversine


# ── Fixtures ─────────────────────────────────────────────────────────────

KTM = Location(27.70, 85.32)  # Kathmandu center


def _make_order(oid: str, plat: float = 27.70, plon: float = 85.32,
                dlat: float = 27.71, dlon: float = 85.33,
                weight: float = 2.0, otype: OrderType = OrderType.PARCEL,
                earliest: float = 480, latest: float = 960,
                prep: float = 0.0) -> Order:
    return Order(
        order_id=oid,
        pickup=Location(plat, plon),
        dropoff=Location(dlat, dlon),
        weight_kg=weight,
        earliest_pickup=earliest,
        latest_delivery=latest,
        order_type=otype,
        prep_time_min=prep,
    )


def _make_driver(did: str = "D1", cap: float = 10.0, max_ord: int = 5,
                 vtype: VehicleType = VehicleType.BIKE) -> Driver:
    return Driver(
        driver_id=did,
        start_location=KTM,
        capacity_kg=cap,
        shift_start=480,
        shift_end=960,
        vehicle_type=vtype,
        max_orders=max_ord,
    )


def _sample_orders(n: int = 6) -> list[Order]:
    random.seed(42)
    orders = []
    for i in range(n):
        orders.append(_make_order(
            f"O{i+1}",
            plat=27.68 + random.random() * 0.04,
            plon=85.30 + random.random() * 0.04,
            dlat=27.68 + random.random() * 0.04,
            dlon=85.30 + random.random() * 0.04,
        ))
    return orders


def _sample_drivers(n: int = 2) -> list[Driver]:
    return [_make_driver(f"D{i+1}") for i in range(n)]


# ── Distance ─────────────────────────────────────────────────────────────

class TestDistance:
    def test_haversine_zero(self):
        assert haversine(KTM, KTM) == 0.0

    def test_haversine_positive(self):
        a = Location(27.70, 85.32)
        b = Location(27.71, 85.33)
        d = haversine(a, b)
        assert 1.0 < d < 2.0  # roughly 1.4 km

    def test_haversine_symmetric(self):
        a, b = Location(27.70, 85.32), Location(27.75, 85.40)
        assert abs(haversine(a, b) - haversine(b, a)) < 1e-6


# ── Fitness ──────────────────────────────────────────────────────────────

class TestFitness:
    def test_empty_routes(self):
        d = _make_driver()
        routes = [Route(driver=d)]
        objs = evaluate_solution(routes)
        assert objs[0] == 0.0  # zero distance
        assert objs[1] == 0.0  # zero lateness

    def test_objectives_tuple_length(self):
        orders = _sample_orders(4)
        drivers = _sample_drivers(2)
        routes = [Route(driver=drivers[0], order_sequence=orders[:2]),
                  Route(driver=drivers[1], order_sequence=orders[2:])]
        objs = evaluate_solution(routes)
        assert len(objs) == 4

    def test_speed_peak_hours(self):
        assert speed_at_time(510) == 12.0  # 8:30 AM
        assert speed_at_time(1080) == 12.0  # 6:00 PM

    def test_speed_off_peak(self):
        assert speed_at_time(1380) == 35.0  # 11 PM
        assert speed_at_time(180) == 35.0   # 3 AM

    def test_speed_normal(self):
        assert speed_at_time(720) == 25.0  # noon

    def test_freshness_no_penalty_non_food(self):
        o = _make_order("X", otype=OrderType.PARCEL)
        assert freshness_penalty(o, 600) == 0.0

    def test_freshness_linear_under_30(self):
        o = _make_order("X", otype=OrderType.FOOD, earliest=480, prep=10)
        # delivery at 510 → elapsed = 510 - 480 - 10 = 20 min
        p = freshness_penalty(o, 510)
        assert 0 < p < 1.5  # linear region

    def test_freshness_quadratic_over_30(self):
        o = _make_order("X", otype=OrderType.FOOD, earliest=480, prep=10)
        # delivery at 560 → elapsed = 560 - 480 - 10 = 70 min
        p = freshness_penalty(o, 560)
        assert p > 10  # quadratic region, should be large

    def test_constraint_penalty_in_fitness(self):
        d = _make_driver(cap=5.0, max_ord=1)
        orders = [_make_order("O1", weight=3.0), _make_order("O2", weight=3.0)]
        routes = [Route(driver=d, order_sequence=orders)]
        objs = evaluate_solution(routes)
        # Over capacity (6 > 5) and over max orders (2 > 1) → penalty added
        assert objs[0] > 0  # distance includes penalty


# ── Constraints ──────────────────────────────────────────────────────────

class TestConstraints:
    def test_capacity_ok(self):
        d = _make_driver(cap=10.0)
        r = Route(driver=d, order_sequence=[_make_order("O1", weight=5.0)])
        assert check_capacity(r)

    def test_capacity_exceeded(self):
        d = _make_driver(cap=5.0)
        r = Route(driver=d, order_sequence=[
            _make_order("O1", weight=3.0), _make_order("O2", weight=3.0)
        ])
        assert not check_capacity(r)

    def test_max_orders_ok(self):
        d = _make_driver(max_ord=3)
        r = Route(driver=d, order_sequence=[_make_order(f"O{i}") for i in range(3)])
        assert check_max_orders(r)

    def test_max_orders_exceeded(self):
        d = _make_driver(max_ord=2)
        r = Route(driver=d, order_sequence=[_make_order(f"O{i}") for i in range(3)])
        assert not check_max_orders(r)

    def test_total_penalty_feasible(self):
        d = _make_driver(cap=10.0, max_ord=5)
        r = Route(driver=d, order_sequence=[_make_order("O1", weight=2.0)])
        assert total_penalty(r) == 0.0

    def test_total_penalty_infeasible(self):
        d = _make_driver(cap=3.0, max_ord=1)
        orders = [_make_order("O1", weight=2.0), _make_order("O2", weight=2.0)]
        r = Route(driver=d, order_sequence=orders)
        assert total_penalty(r) > 0.0


# ── Operators ────────────────────────────────────────────────────────────

class TestOperators:
    def test_ox1_preserves_all_orders(self):
        random.seed(1)
        orders = _sample_orders(6)
        child = order_crossover(orders[:6], list(reversed(orders[:6])))
        assert sorted(o.order_id for o in child) == sorted(o.order_id for o in orders)

    def test_ox1_no_duplicates(self):
        random.seed(2)
        orders = _sample_orders(8)
        child = order_crossover(orders, list(reversed(orders)))
        ids = [o.order_id for o in child]
        assert len(ids) == len(set(ids))

    def test_pmx_preserves_all_orders(self):
        random.seed(3)
        orders = _sample_orders(6)
        child = pmx_crossover(orders, list(reversed(orders)))
        assert sorted(o.order_id for o in child) == sorted(o.order_id for o in orders)

    def test_pmx_no_duplicates(self):
        random.seed(4)
        for _ in range(20):  # run many times to exercise edge cases
            orders = _sample_orders(6)
            pa = list(orders)
            pb = list(orders)
            random.shuffle(pa)
            random.shuffle(pb)
            child = pmx_crossover(pa, pb)
            ids = [o.order_id for o in child]
            assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

    def test_pmx_different_lengths_falls_back(self):
        orders = _sample_orders(6)
        child = pmx_crossover(orders[:4], orders[:3])
        # Falls back to OX1 for different lengths; OX1 result length
        # equals parent_a but only fills slots that have unique orders
        assert len(child) <= len(orders[:4])
        # No duplicate IDs
        ids = [o.order_id for o in child]
        assert len(ids) == len(set(ids))

    def test_swap_mutation_preserves(self):
        orders = _sample_orders(4)
        mutated = swap_mutation(orders, prob=1.0)
        assert sorted(o.order_id for o in mutated) == sorted(o.order_id for o in orders)

    def test_inversion_mutation_preserves(self):
        orders = _sample_orders(5)
        mutated = inversion_mutation(orders, prob=1.0)
        assert sorted(o.order_id for o in mutated) == sorted(o.order_id for o in orders)

    def test_route_transfer_preserves_total(self):
        random.seed(5)
        orders = _sample_orders(6)
        routes = [orders[:3], orders[3:]]
        result = route_transfer_mutation(routes, prob=1.0)
        all_ids = sorted(o.order_id for r in result for o in r)
        expected = sorted(o.order_id for o in orders)
        assert all_ids == expected


# ── Local Search ─────────────────────────────────────────────────────────

class TestLocalSearch:
    def test_two_opt_improves_or_same(self):
        d = _make_driver()
        orders = _sample_orders(5)
        route = Route(driver=d, order_sequence=orders)
        improved = two_opt(route)
        # Cost should not get worse
        from src.local_search import _route_cost
        old_c = _route_cost(d, orders)
        new_c = _route_cost(d, improved.order_sequence)
        assert new_c <= old_c + 1e-6

    def test_or_opt_preserves_orders(self):
        d = _make_driver()
        orders = _sample_orders(5)
        route = Route(driver=d, order_sequence=orders)
        improved = or_opt(route)
        assert sorted(o.order_id for o in improved.order_sequence) == sorted(o.order_id for o in orders)

    def test_apply_local_search_preserves(self):
        drivers = _sample_drivers(2)
        orders = _sample_orders(6)
        routes = [Route(driver=drivers[0], order_sequence=orders[:3]),
                  Route(driver=drivers[1], order_sequence=orders[3:])]
        improved = apply_local_search(routes)
        all_ids = sorted(o.order_id for r in improved for o in r.order_sequence)
        expected = sorted(o.order_id for o in orders)
        assert all_ids == expected


# ── Greedy ───────────────────────────────────────────────────────────────

class TestGreedy:
    def test_assigns_all_orders(self):
        orders = _sample_orders(6)
        drivers = _sample_drivers(2)
        routes = greedy_nearest_neighbor(orders, drivers)
        all_ids = sorted(o.order_id for r in routes for o in r.order_sequence)
        expected = sorted(o.order_id for o in orders)
        assert all_ids == expected

    def test_returns_route_per_driver(self):
        routes = greedy_nearest_neighbor(_sample_orders(4), _sample_drivers(3))
        assert len(routes) == 3


# ── Dynamic Insertion ────────────────────────────────────────────────────

class TestDynamic:
    def test_cheapest_insertion_adds_order(self):
        d = _make_driver(cap=20.0, max_ord=5)
        existing = [Route(driver=d, order_sequence=[_make_order("O1")])]
        new_order = _make_order("O_NEW")
        result = cheapest_insertion(new_order, existing)
        all_ids = [o.order_id for r in result for o in r.order_sequence]
        assert "O_NEW" in all_ids
        assert len(all_ids) == 2

    def test_batch_insert_respects_urgency(self):
        d = _make_driver(cap=50.0, max_ord=10)
        routes = [Route(driver=d)]
        new_orders = [
            _make_order("LATE", latest=900),
            _make_order("URGENT", latest=500),
        ]
        result = batch_insert(new_orders, routes)
        all_ids = [o.order_id for r in result for o in r.order_sequence]
        assert "URGENT" in all_ids
        assert "LATE" in all_ids
        assert len(all_ids) == 2


# ── API models ───────────────────────────────────────────────────────────

class TestAPIValidation:
    def test_order_in_rejects_bad_lat(self):
        from serve import OrderIn
        with pytest.raises(Exception):
            OrderIn(order_id="X", pickup_lat=200, pickup_lon=85, dropoff_lat=27, dropoff_lon=85)

    def test_order_in_rejects_negative_weight(self):
        from serve import OrderIn
        with pytest.raises(Exception):
            OrderIn(order_id="X", pickup_lat=27, pickup_lon=85, dropoff_lat=27, dropoff_lon=85, weight_kg=-1)

    def test_driver_in_rejects_empty_id(self):
        from serve import DriverIn
        with pytest.raises(Exception):
            DriverIn(driver_id="", start_lat=27, start_lon=85)
