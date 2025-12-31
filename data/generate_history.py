"""Generate synthetic historical order data for demand forecasting.

Simulates 14 days of Kathmandu delivery patterns with:
- Restaurant clusters (Thamel, New Road, Jhamsikhel) with lunch/dinner peaks
- Parcel warehouse hubs with morning-heavy distribution
- Random background demand across the valley

Run: python -m data.generate_history
"""

from __future__ import annotations
import csv
import random
from pathlib import Path

# Kathmandu demand clusters (lat, lon, label, peak_slots, avg_orders_per_slot)
CLUSTERS = [
    # Food clusters — lunch (720-840 min) and dinner (1080-1260 min) peaks
    (27.7150, 85.3120, "Thamel restaurants",    [24, 25, 26, 27, 36, 37, 38, 39, 40, 41], 4),
    (27.7172, 85.3240, "New Road eateries",     [24, 25, 26, 27, 36, 37, 38, 39, 40, 41], 3),
    (27.6950, 85.3150, "Jhamsikhel cafes",      [24, 25, 26, 27, 36, 37, 38, 39, 40, 41], 2),
    (27.7300, 85.3200, "Lazimpat dining",       [24, 25, 26, 27, 36, 37, 38, 39, 40, 41], 2),
    (27.7080, 85.3400, "Baneshwor food court",  [24, 25, 26, 27, 36, 37, 38, 39, 40, 41], 3),
    # Parcel clusters — morning to midday (480-780 min)
    (27.7000, 85.3500, "Koteshwor warehouse",   [16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 5),
    (27.7250, 85.3400, "Chabahil distribution", [16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 3),
    (27.7100, 85.3050, "Kalanki hub",           [16, 17, 18, 19, 20, 21, 22, 23], 2),
]

# Delivery radius (degrees, ~1-3 km)
DELIVERY_RADIUS = 0.015


def _jitter(val: float, spread: float = 0.003) -> float:
    return val + random.gauss(0, spread)


def generate_day(day_index: int) -> list[dict]:
    """Generate one day of orders with realistic spatial-temporal patterns."""
    rows = []
    order_counter = 0

    for lat, lon, label, peak_slots, avg_per_slot in CLUSTERS:
        for slot in range(48):  # 0-47, each = 30 min
            # Base rate: 0-1 orders in non-peak, avg_per_slot in peak
            if slot in peak_slots:
                n_orders = max(0, int(random.gauss(avg_per_slot, avg_per_slot * 0.3)))
            else:
                n_orders = 1 if random.random() < 0.15 else 0

            # Weekend boost (+30%)
            if day_index % 7 >= 5:
                n_orders = int(n_orders * 1.3)

            for _ in range(n_orders):
                order_counter += 1
                time_min = slot * 30 + random.uniform(0, 30)
                dropoff_lat = _jitter(lat + random.uniform(-DELIVERY_RADIUS, DELIVERY_RADIUS))
                dropoff_lon = _jitter(lon + random.uniform(-DELIVERY_RADIUS, DELIVERY_RADIUS))
                rows.append({
                    "day": day_index,
                    "order_id": f"H{day_index:02d}_{order_counter:04d}",
                    "dropoff_lat": round(dropoff_lat, 6),
                    "dropoff_lon": round(dropoff_lon, 6),
                    "time_min": round(time_min, 1),
                })

    return rows


def main() -> None:
    random.seed(42)
    output = Path("data/historical_orders.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for day in range(14):
        all_rows.extend(generate_day(day))

    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["day", "order_id", "dropoff_lat", "dropoff_lon", "time_min"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"Generated {len(all_rows)} historical orders across 14 days → {output}")

    # Quick stats
    from collections import Counter
    days = Counter(r["day"] for r in all_rows)
    for d in sorted(days):
        print(f"  Day {d}: {days[d]} orders")


if __name__ == "__main__":
    main()
