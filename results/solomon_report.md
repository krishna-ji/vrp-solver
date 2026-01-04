# Solomon VRPTW Benchmark Report

## Dataset Description

The **Solomon benchmark** (1987) is the gold standard for evaluating VRPTW algorithms. It contains 56 instances with 100 customers each, divided into six categories:

| Category | Customer Distribution | Time Windows | Scheduling Horizon |
|----------|----------------------|--------------|-------------------|
| **C1** | Clustered | Narrow | Short (~230) |
| **C2** | Clustered | Wide | Long (~1000) |
| **R1** | Random (uniform) | Narrow | Short (~230) |
| **R2** | Random (uniform) | Wide | Long (~1000) |
| **RC1** | Mixed (cluster + random) | Narrow | Short (~230) |
| **RC2** | Mixed (cluster + random) | Wide | Long (~1000) |

Each instance specifies a depot, 100 customers with coordinates, demand, time windows, and service time, plus a homogeneous fleet with capacity constraints.

**Adaptation for our solver:** Solomon instances use Euclidean coordinates and a single-depot pickup model. We project coordinates onto an 11 km × 10 km area in Kathmandu and model each delivery as depot→customer (warehouse distribution). Our Haversine distance is proportional to Solomon's Euclidean distance within this projection.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Population size | 60 |
| Generations | 100 |
| Local search | Multi-objective (every 5/10 gen) |
| Initialization | Smart (greedy + deadline + LS seeds) |
| Mutation | Adaptive (diversity-scaled) |
| Seed | 42 |

---

## Results

### Per-Instance Results

| Instance | Cat | Greedy Dist | NSGA-II Dist | Δ% | Greedy Late | NSGA-II Late | Δ% | NSGA-II On-Time | Vehicles Used | Known Best | Time (s) |
|----------|-----|------------|-------------|-----|------------|-------------|-----|----------------|--------------|-----------|----------|
| C101 | C1 | 575.0 | 601.2 | +4.6% | 1091.8 | 70.0 | -93.6% | 0.0% | 25/25 | 828.9 | 70.2 |
| C102 | C1 | 575.0 | 599.3 | +4.2% | 1091.8 | 71.0 | -93.5% | 18.6% | 25/25 | 828.9 | 74.9 |
| C105 | C1 | 575.0 | 545.1 | -5.2% | 1023.8 | 38.1 | -96.3% | 82.8% | 25/25 | 828.9 | 68.5 |
| C201 | C2 | 609.9 | 5.8 | -99.1% | 3077.2 | 0.0 | -100.0% | 100.0% | 5/25 | 591.6 | 34.4 |
| C202 | C2 | 609.9 | 9.5 | -98.4% | 3077.2 | 0.0 | -100.0% | 100.0% | 4/25 | 591.6 | 23.4 |
| C205 | C2 | 609.9 | 19.4 | -96.8% | 2814.6 | 67.3 | -97.6% | 88.9% | 7/25 | 588.9 | 43.3 |
| R101 | R1 | 502.7 | 534.7 | +6.4% | 256.7 | 11.7 | -95.4% | 0.0% | 25/25 | 1650.8 | 91.6 |
| R102 | R1 | 502.7 | 517.5 | +2.9% | 250.3 | 16.7 | -93.3% | 15.8% | 25/25 | 1486.1 | 94.1 |
| R105 | R1 | 502.7 | 412.8 | -17.9% | 232.7 | 0.8 | -99.7% | 98.8% | 25/25 | 1377.1 | 79.3 |
| R201 | R2 | 519.4 | 24.2 | -95.3% | 1345.8 | 0.0 | -100.0% | 100.0% | 3/25 | 1252.4 | 244.3 |
| R202 | R2 | 519.4 | 38.4 | -92.6% | 1307.0 | 19.3 | -98.5% | 90.0% | 2/25 | 1191.7 | 417.1 |
| R205 | R2 | 519.4 | 27.1 | -94.8% | 1210.8 | 29.8 | -97.5% | 50.0% | 2/25 | 994.4 | 61.1 |
| RC101 | RC1 | 663.6 | 692.5 | +4.4% | 242.7 | 5.2 | -97.9% | 94.8% | 25/25 | 1697.0 | 40.8 |
| RC102 | RC1 | 663.6 | 541.1 | -18.5% | 208.2 | 0.3 | -99.9% | 98.8% | 25/25 | 1554.8 | 46.2 |
| RC105 | RC1 | 663.6 | 602.5 | -9.2% | 211.0 | 15.2 | -92.8% | 66.7% | 25/25 | 1629.4 | 44.9 |
| RC201 | RC2 | 689.4 | 60.0 | -91.3% | 1410.9 | 168.8 | -88.0% | 50.0% | 2/25 | 1406.9 | 78.7 |
| RC202 | RC2 | 689.4 | 29.6 | -95.7% | 1353.3 | 17.4 | -98.7% | 83.3% | 3/25 | 1365.7 | 117.6 |
| RC205 | RC2 | 689.4 | 76.4 | -88.9% | 1382.7 | 281.3 | -79.7% | 54.5% | 2/25 | 1297.7 | 53.2 |

### Category Averages

| Category | Avg Greedy Dist | Avg NSGA-II Dist | Avg Dist Δ% | Avg Greedy Late | Avg NSGA-II Late | Avg Late Δ% | Avg On-Time | Avg Time (s) |
|----------|----------------|-----------------|-------------|----------------|-----------------|-------------|------------|-------------|
| **C1** | 575.0 | 581.9 | +1.2% | 1069.1 | 59.7 | -94.4% | 33.8% | 71.2 |
| **C2** | 609.9 | 11.6 | -98.1% | 2989.7 | 22.4 | -99.2% | 96.3% | 33.7 |
| **R1** | 502.7 | 488.3 | -2.8% | 246.6 | 9.7 | -96.1% | 38.2% | 88.3 |
| **R2** | 519.4 | 29.9 | -94.2% | 1287.8 | 16.4 | -98.7% | 80.0% | 240.8 |
| **RC1** | 663.6 | 612.1 | -7.8% | 220.7 | 6.9 | -96.9% | 86.8% | 44.0 |
| **RC2** | 689.4 | 55.3 | -92.0% | 1382.3 | 155.8 | -88.7% | 62.6% | 83.2 |

### Overall Summary

- **Instances evaluated:** 18
- **Total customers served:** 1800
- **Average distance change (NSGA-II vs Greedy):** -49.0%
- **Average lateness reduction:** -95.7%
- **Average on-time rate:** 66.3%
- **Total compute time:** 1683.7s

---

## Analysis

### Category Characteristics

**C-type (Clustered):** Customers are geographically clustered, making route planning easier. Both greedy and NSGA-II perform well, but NSGA-II's multi-objective optimization provides better time window compliance. Narrow windows (C1) are more constrained than wide windows (C2).

**R-type (Random):** Uniformly distributed customers create longer routes with more crossings. Greedy nearest-neighbor struggles with time windows because proximity doesn't correlate with urgency. NSGA-II's lateness-aware optimization shines here.

**RC-type (Mixed):** A realistic mix of clustered and random customers. These instances are generally the hardest because algorithms must handle both dense clusters and isolated customers efficiently.

### Narrow vs Wide Time Windows

- **Type-1 (narrow):** Tight windows force more vehicles and longer routes. Lateness reduction is critical — even small delays cascade.
- **Type-2 (wide):** Relaxed windows allow fewer vehicles and more consolidation. The algorithm has more freedom to optimize distance without violating time constraints.

### Comparison with Known Best Results

The known best results use exact methods or highly specialized metaheuristics tuned specifically for single-objective VRPTW (minimize vehicles, then distance). Our solver optimizes 4 objectives simultaneously (distance, lateness, idle time, fairness), which produces different tradeoffs. Direct distance comparison is therefore not apples-to-apples, but provides a reference point for solution quality.

---

## Methodology Notes

1. **Coordinate projection:** Solomon XY → Kathmandu lat/lon (11×10 km area). Haversine distances are proportional to Euclidean within this small projection.
2. **Single-depot model:** All orders originate from the depot (warehouse distribution). This matches Solomon's assumption.
3. **Vehicle fleet:** Homogeneous (all cars, capacity per Solomon spec). No bike/car mix since Solomon doesn't distinguish vehicle types.
4. **Time units:** Solomon uses abstract time units for windows and service times. We treat these directly as minutes.
5. **Multi-objective:** Unlike standard Solomon benchmarks that minimize (vehicles, distance), we optimize (distance, lateness, idle time, workload fairness) simultaneously via NSGA-II.

---

*Report generated by solomon_benchmark.py*
*Solomon instances: M.M. Solomon (1987), Operations Research 35(2)*
