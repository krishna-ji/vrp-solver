"""FastAPI service — exposes VRP solver as a REST API."""

from __future__ import annotations
import csv
import random
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.models import Location, Order, Driver, Route, OrderType, VehicleType
from src.fitness import evaluate_solution
from src.nsga2 import fast_non_dominated_sort, crowding_distance, tournament_select
from src.operators import order_crossover, pmx_crossover, swap_mutation, inversion_mutation, route_transfer_mutation
from src.local_search import apply_local_search
from src.dynamic import cheapest_insertion, batch_insert
from src.demand_forecast import DemandTracker, SpatialGrid, load_historical_orders
from src.preposition import assign_prepositions

app = FastAPI(
    title="VRP Solver API",
    description="Multi-objective Vehicle Routing for Kathmandu delivery networks",
    version="0.2.0",
)


# ── Request / Response models ───────────────────────────────────────────

class OrderIn(BaseModel):
    order_id: str
    pickup_lat: float = Field(ge=-90, le=90)
    pickup_lon: float = Field(ge=-180, le=180)
    dropoff_lat: float = Field(ge=-90, le=90)
    dropoff_lon: float = Field(ge=-180, le=180)
    weight_kg: float = Field(default=1.0, ge=0, le=500)
    earliest_pickup: float = Field(default=0, ge=0)
    latest_delivery: float = Field(default=1440, ge=0)
    order_type: str = "parcel"
    prep_time_min: float = Field(default=0, ge=0)
    priority: int = Field(default=1, ge=1, le=5)
    is_fragile: bool = False


class DriverIn(BaseModel):
    driver_id: str = Field(min_length=1)
    start_lat: float = Field(ge=-90, le=90)
    start_lon: float = Field(ge=-180, le=180)
    capacity_kg: float = Field(default=10.0, ge=1, le=1000)
    shift_start: float = Field(default=480, ge=0)
    shift_end: float = Field(default=960, ge=0)
    vehicle_type: str = "bike"
    max_orders: int = Field(default=3, ge=1, le=100)


class OptimizeRequest(BaseModel):
    orders: list[OrderIn]
    drivers: list[DriverIn]
    pop_size: int = Field(default=80, ge=10, le=500)
    generations: int = Field(default=100, ge=10, le=1000)
    seed: Optional[int] = 42


class Assignment(BaseModel):
    driver_id: str
    vehicle_type: str
    sequence: int
    order_id: str
    order_type: str
    pickup: list[float]
    dropoff: list[float]


class RouteOut(BaseModel):
    driver_id: str
    vehicle_type: str
    n_orders: int
    assignments: list[Assignment]


class OptimizeResponse(BaseModel):
    objectives: dict[str, float]
    routes: list[RouteOut]
    pareto_front_size: int


class InsertRequest(BaseModel):
    new_orders: list[OrderIn]
    current_routes: list[dict]  # simplified — driver_id + order_ids


class InsertResponse(BaseModel):
    updated_routes: list[RouteOut]
    inserted_count: int
    objectives: dict[str, float]


# ── Helpers ──────────────────────────────────────────────────────────────

def _to_order(o: OrderIn) -> Order:
    return Order(
        order_id=o.order_id,
        pickup=Location(o.pickup_lat, o.pickup_lon),
        dropoff=Location(o.dropoff_lat, o.dropoff_lon),
        weight_kg=o.weight_kg,
        earliest_pickup=o.earliest_pickup,
        latest_delivery=o.latest_delivery,
        order_type=OrderType(o.order_type),
        prep_time_min=o.prep_time_min,
        priority=o.priority,
        is_fragile=o.is_fragile,
    )


def _to_driver(d: DriverIn) -> Driver:
    return Driver(
        driver_id=d.driver_id,
        start_location=Location(d.start_lat, d.start_lon),
        capacity_kg=d.capacity_kg,
        shift_start=d.shift_start,
        shift_end=d.shift_end,
        vehicle_type=VehicleType(d.vehicle_type),
        max_orders=d.max_orders,
    )


def _run_nsga2(
    orders: list[Order],
    drivers: list[Driver],
    pop_size: int,
    generations: int,
) -> list[Route]:
    """Minimal NSGA-II loop returning best-compromise routes."""
    from src.models import Solution

    population: list[Solution] = []
    for _ in range(pop_size):
        shuffled = list(orders)
        random.shuffle(shuffled)
        routes = [Route(driver=d) for d in drivers]
        for i, o in enumerate(shuffled):
            routes[i % len(routes)].order_sequence.append(o)
        objs = evaluate_solution(routes)
        population.append(Solution(routes=routes, objectives=objs))

    for gen in range(generations):
        offspring: list[Solution] = []
        while len(offspring) < pop_size:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child_routes = []
            for r1, r2 in zip(p1.routes, p2.routes):
                xover = pmx_crossover if random.random() < 0.3 else order_crossover
                seq = xover(r1.order_sequence, r2.order_sequence)
                seq = swap_mutation(seq)
                seq = inversion_mutation(seq)
                child_routes.append(Route(driver=r1.driver, order_sequence=seq))
            seqs = route_transfer_mutation([r.order_sequence for r in child_routes])
            for r, s in zip(child_routes, seqs):
                r.order_sequence = s
            objs = evaluate_solution(child_routes)
            offspring.append(Solution(routes=child_routes, objectives=objs))

        if (gen + 1) % 10 == 0:
            combined_temp = population + offspring
            tf = fast_non_dominated_sort(combined_temp)
            if tf:
                for sol in tf[0][:3]:
                    improved = apply_local_search([r.order_sequence for r in sol.routes])
                    for r, s in zip(sol.routes, improved):
                        r.order_sequence = s
                    sol.objectives = evaluate_solution(sol.routes)

        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)
        new_pop: list[Solution] = []
        for front in fronts:
            crowding_distance(front)
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                front.sort(key=lambda s: -s.crowding_distance)
                new_pop.extend(front[: pop_size - len(new_pop)])
                break
        population = new_pop

    front0 = [s for s in population if s.rank == 0]
    best = min(front0, key=lambda s: sum(s.objectives))
    return best.routes, len(front0), best.objectives


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    """Run full NSGA-II optimization."""
    if req.seed is not None:
        random.seed(req.seed)

    orders = [_to_order(o) for o in req.orders]
    drivers = [_to_driver(d) for d in req.drivers]

    routes, front_size, objs = _run_nsga2(orders, drivers, req.pop_size, req.generations)

    route_out = []
    for route in routes:
        assignments = []
        for seq, order in enumerate(route.order_sequence):
            assignments.append(Assignment(
                driver_id=route.driver.driver_id,
                vehicle_type=route.driver.vehicle_type.value,
                sequence=seq,
                order_id=order.order_id,
                order_type=order.order_type.value,
                pickup=[order.pickup.lat, order.pickup.lon],
                dropoff=[order.dropoff.lat, order.dropoff.lon],
            ))
        route_out.append(RouteOut(
            driver_id=route.driver.driver_id,
            vehicle_type=route.driver.vehicle_type.value,
            n_orders=len(route.order_sequence),
            assignments=assignments,
        ))

    return OptimizeResponse(
        objectives={
            "total_distance_km": round(objs[0], 2),
            "lateness_min": round(objs[1], 2),
            "idle_time_min": round(objs[2], 2),
            "unfairness_min": round(objs[3], 2),
        },
        routes=route_out,
        pareto_front_size=front_size,
    )


@app.get("/health")
def health():
    return {"status": "ok", "solver": "NSGA-II", "objectives": 4,
            "features": ["forecast", "preposition", "dynamic_insert"]}


# ── Demand Forecast Endpoints ────────────────────────────────────────────

# Initialize tracker on startup with historical data if available
_tracker: Optional[DemandTracker] = None


def _get_tracker() -> DemandTracker:
    global _tracker
    if _tracker is None:
        _tracker = DemandTracker()
        hist_path = Path("data/historical_orders.csv")
        if hist_path.exists():
            orders = load_historical_orders(hist_path)
            # Group by day
            day_orders: dict[int, list[tuple[Location, float]]] = {}
            with open(hist_path, newline="") as f:
                for row in csv.DictReader(f):
                    day = int(row["day"])
                    loc = Location(float(row["dropoff_lat"]), float(row["dropoff_lon"]))
                    t = float(row["time_min"])
                    day_orders.setdefault(day, []).append((loc, t))
            for day in sorted(day_orders):
                _tracker.record_day(day_orders[day])
    return _tracker


class ForecastRequest(BaseModel):
    start_min: float = Field(description="Start of prediction window (minutes from midnight)")
    end_min: float = Field(description="End of prediction window (minutes from midnight)")
    top_k: int = Field(default=5, ge=1, le=20)


class HotspotOut(BaseModel):
    lat: float
    lon: float
    predicted_orders: float
    confidence: float
    zone: str


class ForecastResponse(BaseModel):
    hotspots: list[HotspotOut]
    window: str


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """Predict demand hotspots for a future time window."""
    tracker = _get_tracker()
    hotspots = tracker.predict_window(req.start_min, req.end_min, top_k=req.top_k)

    return ForecastResponse(
        hotspots=[
            HotspotOut(
                lat=h.centroid.lat,
                lon=h.centroid.lon,
                predicted_orders=h.predicted_orders,
                confidence=h.confidence,
                zone=f"({h.zone.row},{h.zone.col})",
            )
            for h in hotspots
        ],
        window=f"{int(req.start_min)}–{int(req.end_min)} min",
    )


# ── Pre-position Endpoint ───────────────────────────────────────────────

class PrepositionRequest(BaseModel):
    idle_drivers: list[DriverIn]
    start_min: float = Field(description="Start of target window (minutes from midnight)")
    end_min: float = Field(description="End of target window (minutes from midnight)")
    max_reposition_km: float = 5.0
    top_k: int = 5


class DirectiveOut(BaseModel):
    driver_id: str
    target_lat: float
    target_lon: float
    distance_km: float
    predicted_demand: float
    reason: str


class PrepositionResponse(BaseModel):
    directives: list[DirectiveOut]
    hotspots_considered: int


@app.post("/preposition", response_model=PrepositionResponse)
def preposition(req: PrepositionRequest):
    """Suggest where to send idle drivers based on predicted demand."""
    tracker = _get_tracker()
    hotspots = tracker.predict_window(req.start_min, req.end_min, top_k=req.top_k)
    drivers = [_to_driver(d) for d in req.idle_drivers]

    directives = assign_prepositions(
        idle_drivers=drivers,
        hotspots=hotspots,
        max_reposition_km=req.max_reposition_km,
    )

    return PrepositionResponse(
        directives=[
            DirectiveOut(
                driver_id=d.driver.driver_id,
                target_lat=d.target.lat,
                target_lon=d.target.lon,
                distance_km=d.distance_km,
                predicted_demand=d.target_zone_demand,
                reason=d.reason,
            )
            for d in directives
        ],
        hotspots_considered=len(hotspots),
    )


# ── Dynamic Insert Endpoint ─────────────────────────────────────────────

@app.post("/insert", response_model=InsertResponse)
def insert_orders(req: InsertRequest):
    """Insert new orders into existing routes using cheapest insertion."""
    # Rebuild current routes from the simplified dict format
    # Each dict: {"driver_id": ..., "driver": DriverIn, "order_ids": [...], "orders": [OrderIn]}
    routes: list[Route] = []
    for rd in req.current_routes:
        driver_in = DriverIn(**rd["driver"])
        driver = _to_driver(driver_in)
        order_seq = [_to_order(OrderIn(**o)) for o in rd.get("orders", [])]
        routes.append(Route(driver=driver, order_sequence=order_seq))

    new_orders = [_to_order(o) for o in req.new_orders]
    updated = batch_insert(new_orders, routes)
    objs = evaluate_solution(updated)

    route_out = []
    for route in updated:
        assignments = []
        for seq, order in enumerate(route.order_sequence):
            assignments.append(Assignment(
                driver_id=route.driver.driver_id,
                vehicle_type=route.driver.vehicle_type.value,
                sequence=seq,
                order_id=order.order_id,
                order_type=order.order_type.value,
                pickup=[order.pickup.lat, order.pickup.lon],
                dropoff=[order.dropoff.lat, order.dropoff.lon],
            ))
        route_out.append(RouteOut(
            driver_id=route.driver.driver_id,
            vehicle_type=route.driver.vehicle_type.value,
            n_orders=len(route.order_sequence),
            assignments=assignments,
        ))

    return InsertResponse(
        updated_routes=route_out,
        inserted_count=len(new_orders),
        objectives={
            "total_distance_km": round(objs[0], 2),
            "lateness_min": round(objs[1], 2),
            "idle_time_min": round(objs[2], 2),
            "unfairness_min": round(objs[3], 2),
        },
    )
