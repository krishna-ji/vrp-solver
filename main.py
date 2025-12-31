"""CLI entry point for VRP Solver — food, parcel, and ride-sharing optimization."""

from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path

from src.models import Location, Order, Driver, Route, Solution, OrderType, VehicleType
from src.fitness import evaluate_solution
from src.nsga2 import fast_non_dominated_sort, crowding_distance, tournament_select
from src.operators import order_crossover, pmx_crossover, swap_mutation, inversion_mutation, route_transfer_mutation
from src.local_search import apply_local_search
from src.greedy import greedy_nearest_neighbor
from src.visualization import plot_pareto_front, plot_routes_map


def load_orders(path: Path) -> list[Order]:
    orders = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            orders.append(Order(
                order_id=row["order_id"],
                pickup=Location(float(row["pickup_lat"]), float(row["pickup_lon"])),
                dropoff=Location(float(row["dropoff_lat"]), float(row["dropoff_lon"])),
                weight_kg=float(row["weight_kg"]),
                earliest_pickup=float(row["earliest_pickup"]),
                latest_delivery=float(row["latest_delivery"]),
                order_type=OrderType(row.get("order_type", "parcel")),
                prep_time_min=float(row.get("prep_time_min", 0)),
                priority=int(row.get("priority", 1)),
                is_fragile=row.get("is_fragile", "false").lower() == "true",
            ))
    return orders


def load_drivers(path: Path) -> list[Driver]:
    drivers = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vtype = VehicleType(row["vehicle_type"]) if "vehicle_type" in row else VehicleType.BIKE
            drivers.append(Driver(
                driver_id=row["driver_id"],
                start_location=Location(float(row["start_lat"]), float(row["start_lon"])),
                capacity_kg=float(row["capacity_kg"]),
                shift_start=float(row["shift_start"]),
                shift_end=float(row["shift_end"]),
                vehicle_type=vtype,
                max_orders=int(row.get("max_orders", 3)),
            ))
    return drivers


def random_assignment(orders: list[Order], drivers: list[Driver]) -> list[Route]:
    """Random partition of orders across drivers."""
    shuffled = list(orders)
    random.shuffle(shuffled)
    routes = [Route(driver=d) for d in drivers]
    for i, order in enumerate(shuffled):
        routes[i % len(routes)].order_sequence.append(order)
    return routes


def deadline_sorted_assignment(orders: list[Order], drivers: list[Driver]) -> list[Route]:
    """Assign orders sorted by deadline urgency — food first, then by latest_delivery."""
    from src.distance import haversine

    type_priority = {OrderType.FOOD: 0, OrderType.RIDE: 1, OrderType.PARCEL: 2}
    sorted_orders = sorted(orders, key=lambda o: (type_priority.get(o.order_type, 2), o.latest_delivery))

    routes = [Route(driver=d) for d in drivers]

    for order in sorted_orders:
        # Find best driver: closest with capacity and fewest orders
        best_route = None
        best_score = float("inf")
        for route in routes:
            weight = sum(o.weight_kg for o in route.order_sequence)
            if weight + order.weight_kg > route.driver.capacity_kg:
                continue
            if len(route.order_sequence) >= route.driver.max_orders:
                continue
            loc = route.order_sequence[-1].dropoff if route.order_sequence else route.driver.start_location
            d = haversine(loc, order.pickup)
            # Score: distance + penalty for load imbalance
            score = d + len(route.order_sequence) * 2.0
            if score < best_score:
                best_score = score
                best_route = route

        if best_route is None:
            # Fallback: least loaded route
            best_route = min(routes, key=lambda r: len(r.order_sequence))
        best_route.order_sequence.append(order)

    return routes


def encode_solution(routes: list[Route]) -> Solution:
    """Evaluate routes and wrap in Solution."""
    objs = evaluate_solution(routes)
    return Solution(routes=list(routes), objectives=objs)


def evolve(
    orders: list[Order],
    drivers: list[Driver],
    pop_size: int = 100,
    generations: int = 200,
) -> list[Solution]:
    """Run NSGA-II with local search and return final population."""
    print(f"NSGA-II: pop={pop_size}, gen={generations}, orders={len(orders)}, drivers={len(drivers)}")

    # --- Smart initialization: seed with heuristic solutions ---
    population = []

    # Seed 1: greedy nearest-neighbor
    greedy_routes = greedy_nearest_neighbor(orders, drivers)
    population.append(encode_solution(greedy_routes))

    # Seed 2-4: deadline-sorted (urgency-aware)
    for _ in range(3):
        dl_routes = deadline_sorted_assignment(orders, drivers)
        population.append(encode_solution(dl_routes))

    # Seed 5-10: greedy with local search applied
    for _ in range(min(6, pop_size - len(population))):
        routes = greedy_nearest_neighbor(orders, drivers)
        routes = apply_local_search(routes)
        population.append(encode_solution(routes))

    # Fill rest with random
    while len(population) < pop_size:
        routes = random_assignment(orders, drivers)
        population.append(encode_solution(routes))

    # Initial sort
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        crowding_distance(front)

    for gen in range(generations):
        # --- Adaptive mutation rate: increase when diversity drops ---
        front0_count = sum(1 for s in population if s.rank == 0)
        diversity_ratio = front0_count / pop_size
        # When >80% of pop is in front-0, diversity is low → boost mutation
        mutation_boost = 1.0 + max(0.0, diversity_ratio - 0.5) * 3.0

        # Create offspring
        offspring: list[Solution] = []
        while len(offspring) < pop_size:
            p1 = tournament_select(population)
            p2 = tournament_select(population)

            # Crossover per route — randomly choose OX1 or PMX
            child_routes = []
            for r1, r2 in zip(p1.routes, p2.routes):
                xover = pmx_crossover if random.random() < 0.3 else order_crossover
                child_seq = xover(r1.order_sequence, r2.order_sequence)
                child_seq = swap_mutation(child_seq, prob=0.15 * mutation_boost)
                child_seq = inversion_mutation(child_seq, prob=0.08 * mutation_boost)
                child_routes.append(Route(driver=r1.driver, order_sequence=child_seq))

            # Inter-route mutation (boosted rate)
            seqs = route_transfer_mutation(
                [r.order_sequence for r in child_routes],
                prob=0.08 * mutation_boost,
            )
            for r, seq in zip(child_routes, seqs):
                r.order_sequence = seq

            offspring.append(encode_solution(child_routes))

        # --- More aggressive local search ---
        ls_interval = 5 if gen < 50 else 10
        if (gen + 1) % ls_interval == 0:
            all_combined = population + offspring
            temp_fronts = fast_non_dominated_sort(all_combined)
            if temp_fronts:
                # Apply LS to top-10 (or more early on)
                n_ls = 15 if gen < 50 else 10
                for sol in temp_fronts[0][:n_ls]:
                    improved = apply_local_search(sol.routes)
                    sol.routes = improved
                    sol.objectives = evaluate_solution(sol.routes)

        # Combine and select
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

        if (gen + 1) % 50 == 0 or gen == 0:
            best = min(population, key=lambda s: s.objectives[0])
            print(
                f"  Gen {gen+1:4d} | dist={best.objectives[0]:.1f}km "
                f"late={best.objectives[1]:.1f}min idle={best.objectives[2]:.1f}min "
                f"fair={best.objectives[3]:.1f}min"
                f" | Front-0={sum(1 for s in population if s.rank == 0)}"
                f" | mut={mutation_boost:.2f}x"
            )

    return population


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Objective VRP Solver (NSGA-II) — Kathmandu Delivery")
    parser.add_argument("--orders", type=Path, default=Path("data/kathmandu_mixed.csv"))
    parser.add_argument("--drivers", type=Path, default=Path("data/kathmandu_drivers.csv"))
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    orders = load_orders(args.orders)
    drivers = load_drivers(args.drivers)

    food = sum(1 for o in orders if o.order_type == OrderType.FOOD)
    parcel = sum(1 for o in orders if o.order_type == OrderType.PARCEL)
    ride = sum(1 for o in orders if o.order_type == OrderType.RIDE)
    bikes = sum(1 for d in drivers if d.vehicle_type == VehicleType.BIKE)
    cars = sum(1 for d in drivers if d.vehicle_type == VehicleType.CAR)
    print(f"Orders: {len(orders)} ({food} food, {parcel} parcel, {ride} ride)")
    print(f"Drivers: {len(drivers)} ({bikes} bike, {cars} car)")

    population = evolve(orders, drivers, args.pop_size, args.generations)

    # Best compromise solution (min distance from front-0)
    front0 = [s for s in population if s.rank == 0]
    best = min(front0, key=lambda s: sum(s.objectives))
    print(f"\nBest compromise:")
    print(f"  Distance:  {best.objectives[0]:.1f} km")
    print(f"  Lateness:  {best.objectives[1]:.1f} min (includes freshness penalty)")
    print(f"  Idle time: {best.objectives[2]:.1f} min")
    print(f"  Unfairness:{best.objectives[3]:.1f} min (max-min route duration)")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    plot_pareto_front(front0, output_path=args.output / "pareto_front.png")
    plot_routes_map(best.routes, output_path=args.output / "routes_map.html")

    # Save assignments CSV
    with open(args.output / "assignments.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["driver_id", "vehicle_type", "sequence", "order_id", "order_type",
                     "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon",
                     "weight_kg", "priority"])
        for route in best.routes:
            for seq, order in enumerate(route.order_sequence):
                w.writerow([route.driver.driver_id, route.driver.vehicle_type.value,
                            seq, order.order_id, order.order_type.value,
                            order.pickup.lat, order.pickup.lon,
                            order.dropoff.lat, order.dropoff.lon,
                            order.weight_kg, order.priority])

    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
