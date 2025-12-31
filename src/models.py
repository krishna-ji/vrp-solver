"""Domain models for VRP — supports food delivery, parcel, and ride-sharing."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class OrderType(Enum):
    FOOD = "food"
    PARCEL = "parcel"
    RIDE = "ride"


class VehicleType(Enum):
    BIKE = "bike"
    CAR = "car"


# Vehicle type constraints
VEHICLE_CAPACITY: dict[VehicleType, float] = {
    VehicleType.BIKE: 10.0,   # kg
    VehicleType.CAR: 50.0,
}
VEHICLE_MAX_ORDERS: dict[VehicleType, int] = {
    VehicleType.BIKE: 3,
    VehicleType.CAR: 8,
}


@dataclass(frozen=True, slots=True)
class Location:
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    pickup: Location
    dropoff: Location
    weight_kg: float
    earliest_pickup: float      # minutes from midnight
    latest_delivery: float      # minutes from midnight
    order_type: OrderType = OrderType.PARCEL
    prep_time_min: float = 0.0  # restaurant prep time (food only)
    priority: int = 1           # 1=normal, 2=express, 3=urgent
    is_fragile: bool = False


@dataclass(frozen=True, slots=True)
class Driver:
    driver_id: str
    start_location: Location
    capacity_kg: float
    shift_start: float          # minutes from midnight
    shift_end: float            # minutes from midnight
    vehicle_type: VehicleType = VehicleType.BIKE
    max_orders: int = 3


@dataclass(slots=True)
class Route:
    driver: Driver
    order_sequence: list[Order] = field(default_factory=list)
    total_distance_km: float = 0.0
    total_time_min: float = 0.0
    max_lateness_min: float = 0.0
    idle_time_min: float = 0.0
    freshness_penalty: float = 0.0  # food quality degradation


@dataclass(slots=True)
class Solution:
    routes: list[Route] = field(default_factory=list)
    objectives: tuple[float, ...] = ()  # (distance, lateness, idle, fairness)
    rank: int = 0
    crowding_distance: float = 0.0
