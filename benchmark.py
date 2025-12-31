"""Benchmark: NSGA-II vs Greedy baseline — multi-metric comparison."""

from __future__ import annotations
import argparse
import random
from pathlib import Path

from main import load_orders, load_drivers, evolve, encode_solution
from src.greedy import greedy_nearest_neighbor
from src.fitness import evaluate_solution, detailed_metrics
from src.visualization import plot_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VRP Solver vs Greedy")
    parser.add_argument("--orders", type=Path, default=Path("data/kathmandu_mixed.csv"))
    parser.add_argument("--drivers", type=Path, default=Path("data/kathmandu_drivers.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    orders = load_orders(args.orders)
    drivers = load_drivers(args.drivers)

    # Greedy baseline
    greedy_routes = greedy_nearest_neighbor(orders, drivers)
    greedy_obj = evaluate_solution(greedy_routes)
    greedy_ext = detailed_metrics(greedy_routes)

    # NSGA-II
    population = evolve(orders, drivers, pop_size=100, generations=200)
    front0 = [s for s in population if s.rank == 0]
    best = min(front0, key=lambda s: sum(s.objectives))
    nsga2_routes = best.routes
    nsga2_obj = best.objectives
    nsga2_ext = detailed_metrics(nsga2_routes)

    # --- Print 4 core objectives ---
    print("=" * 65)
    print("CORE OBJECTIVES (NSGA-II optimization targets)")
    print("=" * 65)
    obj_names = ["Total Distance (km)", "Lateness (min)", "Idle Time (min)", "Unfairness (min)"]
    for i, name in enumerate(obj_names):
        g, n = greedy_obj[i], nsga2_obj[i]
        pct = ((n - g) / g * 100) if g > 0 else 0.0
        print(f"  {name:<25s}  Greedy: {g:>9.1f}  NSGA-II: {n:>9.1f}  ({pct:+.1f}%)")

    # --- Print extended metrics ---
    print()
    print("=" * 65)
    print("EXTENDED METRICS")
    print("=" * 65)
    ext_labels = [
        ("On-Time Rate (%)", "on_time_rate", "%"),
        ("Fleet Utilization (%)", "fleet_utilization", "%"),
        ("Food Freshness Compliance (%)", "food_freshness_rate", "%"),
        ("Avg Delivery Time (min)", "avg_delivery_time", "min"),
        ("Makespan (min)", "makespan", "min"),
        ("Active Drivers", "active_drivers", ""),
        ("Avg Orders / Driver", "avg_orders_per_driver", ""),
        ("Max Route Distance (km)", "max_route_distance", "km"),
        ("CO₂ Emissions (kg)", "co2_kg", "kg"),
    ]
    for label, key, unit in ext_labels:
        g, n = greedy_ext[key], nsga2_ext[key]
        if g > 0:
            pct = ((n - g) / g * 100)
            pct_str = f"({pct:+.1f}%)"
        else:
            pct_str = ""
        print(f"  {label:<32s}  Greedy: {g:>9.1f}  NSGA-II: {n:>9.1f}  {pct_str}")

    # --- Save plots ---
    args.output.mkdir(parents=True, exist_ok=True)
    plot_comparison(
        greedy_obj, nsga2_obj,
        greedy_ext, nsga2_ext,
        output_path=args.output / "comparison.png",
    )


if __name__ == "__main__":
    main()
