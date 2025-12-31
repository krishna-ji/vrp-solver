"""Distance computations — Haversine for now, OSRM-ready."""

import math
import numpy as np
from numpy.typing import NDArray
from src.models import Location


_R_EARTH_KM = 6371.0


def haversine(a: Location, b: Location) -> float:
    """Great-circle distance in km between two lat/lon points."""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _R_EARTH_KM * math.asin(math.sqrt(h))


def build_distance_matrix(locations: list[Location]) -> NDArray[np.float64]:
    """NxN distance matrix (km) for a list of locations."""
    n = len(locations)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(locations[i], locations[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat
