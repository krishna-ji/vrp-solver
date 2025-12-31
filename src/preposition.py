"""Vehicle pre-positioning — move idle drivers toward predicted demand hotspots.

Given a demand forecast (list of Hotspot) and a set of drivers with current
locations, assigns idle drivers to reposition toward high-demand zones
before orders arrive.  This reduces first-mile pickup time for the next
batch of orders.
"""

from __future__ import annotations
from dataclasses import dataclass
from src.models import Driver, Location
from src.distance import haversine
from src.demand_forecast import Hotspot


@dataclass(slots=True)
class RepositionDirective:
    """Instruction for one driver to move toward a hotspot."""
    driver: Driver
    target: Location
    target_zone_demand: float
    distance_km: float
    reason: str


def assign_prepositions(
    idle_drivers: list[Driver],
    hotspots: list[Hotspot],
    max_reposition_km: float = 5.0,
    min_confidence: float = 0.1,
) -> list[RepositionDirective]:
    """Greedily assign idle drivers to hotspot centroids.

    Algorithm:
    1. Filter hotspots by confidence threshold.
    2. Sort hotspots by predicted_orders descending.
    3. For each hotspot, find the nearest idle driver within max_reposition_km.
    4. Assign driver → hotspot centroid, remove both from pools.

    Args:
        idle_drivers: drivers with no active route / finished deliveries.
        hotspots: predicted demand points from DemandTracker.predict().
        max_reposition_km: don't send a driver further than this.
        min_confidence: ignore hotspots below this confidence.

    Returns:
        List of RepositionDirective, one per assigned driver.
    """
    viable = [h for h in hotspots if h.confidence >= min_confidence]
    viable.sort(key=lambda h: -h.predicted_orders)

    available = list(idle_drivers)
    directives: list[RepositionDirective] = []

    for hs in viable:
        if not available:
            break

        # Find nearest idle driver
        best_driver = None
        best_dist = float("inf")
        for drv in available:
            d = haversine(drv.start_location, hs.centroid)
            if d < best_dist and d <= max_reposition_km:
                best_dist = d
                best_driver = drv

        if best_driver is not None:
            available.remove(best_driver)
            directives.append(RepositionDirective(
                driver=best_driver,
                target=hs.centroid,
                target_zone_demand=hs.predicted_orders,
                distance_km=round(best_dist, 2),
                reason=f"Zone({hs.zone.row},{hs.zone.col}) predicted "
                       f"{hs.predicted_orders} orders, conf={hs.confidence}",
            ))

    return directives


def should_preposition(
    driver: Driver,
    current_time_min: float,
    route_end_time_min: float,
    idle_threshold_min: float = 15.0,
) -> bool:
    """Check if a driver will be idle long enough to justify repositioning.

    A driver finishing their current route with > idle_threshold_min
    remaining in shift is a candidate for pre-positioning.
    """
    remaining_shift = driver.shift_end - route_end_time_min
    if remaining_shift < idle_threshold_min:
        return False
    idle_gap = route_end_time_min - current_time_min
    return idle_gap <= 5.0  # only if they're finishing soon
