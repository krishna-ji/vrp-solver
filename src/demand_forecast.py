"""Demand forecasting — grid-based zone prediction using exponential smoothing.

Divides the service area into a spatial grid, tracks historical order counts
per zone per time slot, and predicts future demand to enable proactive
vehicle pre-positioning.  Works with any OrderType but is most impactful
for parcel delivery where repeat-customer patterns are strong.
"""

from __future__ import annotations
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.models import Location


# ── Grid Zone ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Zone:
    """A rectangular cell in the spatial grid, identified by row/col index."""
    row: int
    col: int


@dataclass(slots=True)
class ZoneDemand:
    """Historical demand counts for one zone across time slots."""
    zone: Zone
    centroid: Location
    # counts[slot_index] = number of orders observed in that slot
    counts: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Hotspot:
    """A predicted future demand point."""
    zone: Zone
    centroid: Location
    predicted_orders: float
    confidence: float  # 0-1, how stable the signal is


# ── Spatial Grid ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class SpatialGrid:
    """Partitions a bounding box into rows x cols rectangular cells.

    Kathmandu valley default: ~12 km N-S x ~15 km E-W → 500 m cells.
    """
    min_lat: float = 27.62
    max_lat: float = 27.78
    min_lon: float = 85.25
    max_lon: float = 85.42
    n_rows: int = 32     # ~500 m per cell vertically
    n_cols: int = 34     # ~500 m per cell horizontally

    @property
    def lat_step(self) -> float:
        return (self.max_lat - self.min_lat) / self.n_rows

    @property
    def lon_step(self) -> float:
        return (self.max_lon - self.min_lon) / self.n_cols

    def locate(self, loc: Location) -> Zone:
        """Map a lat/lon to its grid zone."""
        row = int((loc.lat - self.min_lat) / self.lat_step)
        col = int((loc.lon - self.min_lon) / self.lon_step)
        row = max(0, min(row, self.n_rows - 1))
        col = max(0, min(col, self.n_cols - 1))
        return Zone(row, col)

    def centroid(self, zone: Zone) -> Location:
        """Return the center point of a zone cell."""
        lat = self.min_lat + (zone.row + 0.5) * self.lat_step
        lon = self.min_lon + (zone.col + 0.5) * self.lon_step
        return Location(lat, lon)


# ── Demand Tracker ───────────────────────────────────────────────────────

TIME_SLOT_MIN = 30  # each slot covers 30 minutes
SLOTS_PER_DAY = 48  # 24 h * 60 / 30


def _slot_index(time_min: float) -> int:
    """Convert minutes-from-midnight to slot index."""
    return min(int(time_min / TIME_SLOT_MIN), SLOTS_PER_DAY - 1)


class DemandTracker:
    """Accumulates order history and forecasts per-zone demand.

    Uses double exponential smoothing (Holt's method) for each zone's
    time-slot series to capture both level and trend.
    """

    def __init__(self, grid: Optional[SpatialGrid] = None, alpha: float = 0.3,
                 beta: float = 0.1):
        self.grid = grid or SpatialGrid()
        self.alpha = alpha   # level smoothing
        self.beta = beta     # trend smoothing
        # zone → slot → list of daily counts
        self._history: dict[Zone, dict[int, list[int]]] = {}
        # Smoothed state: zone → slot → (level, trend)
        self._state: dict[Zone, dict[int, tuple[float, float]]] = {}

    def record_order(self, dropoff: Location, time_min: float) -> None:
        """Record one order delivery for tracking."""
        zone = self.grid.locate(dropoff)
        slot = _slot_index(time_min)
        self._history.setdefault(zone, {}).setdefault(slot, []).append(1)

    def record_day(self, orders: list[tuple[Location, float]]) -> None:
        """Record a full day's orders then update smoothing state.

        Args:
            orders: list of (dropoff_location, delivery_time_min) pairs.
        """
        # Count orders per zone per slot for this day
        day_counts: dict[Zone, dict[int, int]] = {}
        for loc, t in orders:
            zone = self.grid.locate(loc)
            slot = _slot_index(t)
            day_counts.setdefault(zone, {}).setdefault(slot, 0)
            day_counts[zone][slot] += 1
            # Also track in _history for confidence calculation
            self._history.setdefault(zone, {}).setdefault(slot, []).append(1)

        # Update Holt smoothing for each zone/slot
        for zone, slots in day_counts.items():
            if zone not in self._state:
                self._state[zone] = {}
            for slot, count in slots.items():
                if slot not in self._state[zone]:
                    # Initialize: level = count, trend = 0
                    self._state[zone][slot] = (float(count), 0.0)
                else:
                    prev_level, prev_trend = self._state[zone][slot]
                    level = self.alpha * count + (1 - self.alpha) * (prev_level + prev_trend)
                    trend = self.beta * (level - prev_level) + (1 - self.beta) * prev_trend
                    self._state[zone][slot] = (level, trend)

        # Also init zero for zones that existed before but got no orders today
        for zone in self._state:
            for slot in list(self._state[zone]):
                if zone not in day_counts or slot not in day_counts[zone]:
                    prev_level, prev_trend = self._state[zone][slot]
                    level = self.alpha * 0 + (1 - self.alpha) * (prev_level + prev_trend)
                    trend = self.beta * (level - prev_level) + (1 - self.beta) * prev_trend
                    self._state[zone][slot] = (level, trend)

    def predict(self, slot: int, horizon: int = 1, top_k: int = 5) -> list[Hotspot]:
        """Predict demand for a future time slot.

        Args:
            slot: target slot index (0-47).
            horizon: how many periods ahead (1 = next occurrence of this slot).
            top_k: return this many highest-demand zones.

        Returns:
            List of Hotspot sorted by predicted_orders descending.
        """
        predictions: list[tuple[Zone, float, float]] = []

        for zone, slots in self._state.items():
            if slot not in slots:
                continue
            level, trend = slots[slot]
            forecast = level + horizon * trend
            if forecast < 0.5:
                continue

            # Confidence: fraction of days this zone/slot had orders
            n_data_points = len(self._history.get(zone, {}).get(slot, []))
            confidence = min(1.0, n_data_points / 7.0)  # saturates at 1 week of data

            predictions.append((zone, max(0.0, forecast), confidence))

        # Fill gaps: for zones with no data in this slot, estimate from neighbors
        observed_zones = {z for z, _, _ in predictions}
        for zone, slots in self._state.items():
            if zone in observed_zones:
                continue
            # Average neighbor zones that DO have data for this slot
            neighbor_vals: list[float] = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nz = Zone(zone.row + dr, zone.col + dc)
                    if nz in self._state and slot in self._state[nz]:
                        nl, nt = self._state[nz][slot]
                        nf = nl + horizon * nt
                        if nf >= 0.5:
                            neighbor_vals.append(nf)
            if neighbor_vals:
                avg_forecast = sum(neighbor_vals) / len(neighbor_vals) * 0.5  # dampen
                if avg_forecast >= 0.5:
                    predictions.append((zone, avg_forecast, 0.1))  # low confidence

        # Sort by predicted orders descending
        predictions.sort(key=lambda x: -x[1])

        return [
            Hotspot(
                zone=z,
                centroid=self.grid.centroid(z),
                predicted_orders=round(pred, 1),
                confidence=round(conf, 2),
            )
            for z, pred, conf in predictions[:top_k]
        ]

    def predict_window(self, start_min: float, end_min: float,
                       top_k: int = 5) -> list[Hotspot]:
        """Predict aggregate demand across a time window."""
        start_slot = _slot_index(start_min)
        end_slot = _slot_index(end_min)

        zone_totals: dict[Zone, tuple[float, float]] = {}  # zone → (total, max_conf)

        for slot in range(start_slot, end_slot + 1):
            for hs in self.predict(slot, top_k=self.grid.n_rows * self.grid.n_cols):
                prev_total, prev_conf = zone_totals.get(hs.zone, (0.0, 0.0))
                zone_totals[hs.zone] = (
                    prev_total + hs.predicted_orders,
                    max(prev_conf, hs.confidence),
                )

        ranked = sorted(zone_totals.items(), key=lambda x: -x[1][0])
        return [
            Hotspot(
                zone=z,
                centroid=self.grid.centroid(z),
                predicted_orders=round(total, 1),
                confidence=round(conf, 2),
            )
            for z, (total, conf) in ranked[:top_k]
        ]


# ── CSV loader ───────────────────────────────────────────────────────────

def load_historical_orders(path: Path) -> list[tuple[Location, float]]:
    """Load historical order data: CSV with dropoff_lat, dropoff_lon, time_min."""
    orders = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            loc = Location(float(row["dropoff_lat"]), float(row["dropoff_lon"]))
            t = float(row["time_min"])
            orders.append((loc, t))
    return orders
