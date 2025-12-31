"""Visualization — Pareto front plots and route maps."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from src.models import Solution, Route


def plot_pareto_front(
    solutions: list[Solution],
    labels: tuple[str, ...] = ("Total Distance (km)", "Lateness + Freshness (min)", "Idle Time (min)", "Unfairness (min)"),
    output_path: Path = Path("results/pareto_front.png"),
) -> None:
    """4-objective Pareto front as 2D scatter matrix (6 pairwise plots)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    objs = [s.objectives for s in solutions if s.rank == 0]
    if not objs:
        print("No Pareto-optimal solutions to plot.")
        return

    n_obj = len(objs[0])
    pairs = [(i, j) for i in range(n_obj) for j in range(i + 1, n_obj)]
    n_plots = len(pairs)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    for idx, (i, j) in enumerate(pairs):
        ax = axes_flat[idx]
        ax.scatter([o[i] for o in objs], [o[j] for o in objs], s=20, alpha=0.7, c="#3498db")
        ax.set_xlabel(labels[i] if i < len(labels) else f"Obj {i}")
        ax.set_ylabel(labels[j] if j < len(labels) else f"Obj {j}")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Pareto Front — Non-Dominated Solutions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Pareto front saved to {output_path}")


def plot_routes_map(
    routes: list[Route],
    output_path: Path = Path("results/routes_map.html"),
) -> None:
    """Interactive map of routes using folium."""
    try:
        import folium
    except ImportError:
        print("folium not installed — skipping map.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Center on average location
    all_lats = []
    all_lons = []
    for route in routes:
        all_lats.append(route.driver.start_location.lat)
        all_lons.append(route.driver.start_location.lon)
        for o in route.order_sequence:
            all_lats.extend([o.pickup.lat, o.dropoff.lat])
            all_lons.extend([o.pickup.lon, o.dropoff.lon])

    if not all_lats:
        return

    center = (sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons))
    m = folium.Map(location=center, zoom_start=13)

    colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]
    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        coords = [(route.driver.start_location.lat, route.driver.start_location.lon)]
        for o in route.order_sequence:
            coords.append((o.pickup.lat, o.pickup.lon))
            coords.append((o.dropoff.lat, o.dropoff.lon))

        folium.PolyLine(coords, color=color, weight=3, opacity=0.7).add_to(m)
        folium.Marker(
            coords[0],
            popup=f"Driver {route.driver.driver_id}",
            icon=folium.Icon(color=color, icon="car", prefix="fa"),
        ).add_to(m)

    m.save(str(output_path))
    print(f"Route map saved to {output_path}")


def plot_comparison(
    greedy_obj: tuple[float, ...],
    nsga2_obj: tuple[float, ...],
    greedy_ext: dict[str, float] | None = None,
    nsga2_ext: dict[str, float] | None = None,
    output_path: Path = Path("results/comparison.png"),
) -> None:
    """Bar chart comparing greedy vs NSGA-II on core objectives + extended metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_ext = greedy_ext is not None and nsga2_ext is not None

    if has_ext:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))

    # --- Core objectives (left panel) ---
    labels = ["Total Distance\n(km)", "Lateness\n(min)", "Idle Time\n(min)", "Unfairness\n(min)"]
    n = min(len(greedy_obj), len(nsga2_obj), len(labels))
    x = range(n)
    width = 0.35

    bars_g = ax1.bar([i - width / 2 for i in x], greedy_obj[:n], width, label="Greedy", color="#e74c3c")
    bars_n = ax1.bar([i + width / 2 for i in x], nsga2_obj[:n], width, label="NSGA-II", color="#2ecc71")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels[:n])
    ax1.legend()
    ax1.set_title("Core Objectives (lower is better)")
    ax1.grid(axis="y", alpha=0.3)

    # Add % change labels
    for i in range(n):
        g, ns = greedy_obj[i], nsga2_obj[i]
        if g > 0:
            pct = (ns - g) / g * 100
            color = "#27ae60" if pct <= 0 else "#c0392b"
            ax1.annotate(f"{pct:+.0f}%", xy=(i + width / 2, ns),
                         ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)

    if has_ext:
        # --- Extended metrics (right panel) ---
        ext_keys = [
            ("On-Time\nRate", "on_time_rate"),
            ("Fleet\nUtilization", "fleet_utilization"),
            ("Food Fresh\nCompliance", "food_freshness_rate"),
            ("Avg Delivery\nTime (min)", "avg_delivery_time"),
            ("Makespan\n(min)", "makespan"),
        ]
        ext_labels = [lbl for lbl, _ in ext_keys]
        g_vals = [greedy_ext[k] for _, k in ext_keys]
        n_vals = [nsga2_ext[k] for _, k in ext_keys]
        x2 = range(len(ext_keys))

        ax2.bar([i - width / 2 for i in x2], g_vals, width, label="Greedy", color="#e74c3c")
        ax2.bar([i + width / 2 for i in x2], n_vals, width, label="NSGA-II", color="#2ecc71")
        ax2.set_xticks(list(x2))
        ax2.set_xticklabels(ext_labels, fontsize=8)
        ax2.legend()
        ax2.set_title("Extended Metrics")
        ax2.grid(axis="y", alpha=0.3)

        # % change labels
        for i, (g, ns) in enumerate(zip(g_vals, n_vals)):
            if g > 0:
                pct = (ns - g) / g * 100
                # For rates (%), higher is better
                higher_better = ext_keys[i][1].endswith("_rate") or ext_keys[i][1] == "fleet_utilization"
                color = "#27ae60" if (pct >= 0) == higher_better else "#c0392b"
                ax2.annotate(f"{pct:+.0f}%", xy=(i + width / 2, ns),
                             ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)

    fig.suptitle("Greedy vs NSGA-II Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {output_path}")
