from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from ortools.linear_solver import pywraplp

from inventory_transfer_env import (
    InventoryTransferAction,
    InventoryTransferEnv,
    InventoryTransferObservation,
    Transfer,
)


def _greedy_plan(obs) -> InventoryTransferAction:
    products = obs.products

    transfers: list[Transfer] = []

    remaining_budget = float("inf") if obs.budget is None else float(obs.budget)
    remaining_out = None if obs.outbound_capacity is None else {k: int(v) for k, v in obs.outbound_capacity.items()}
    remaining_in = None if obs.inbound_capacity is None else {k: int(v) for k, v in obs.inbound_capacity.items()}

    lane_remaining = None
    if obs.lane_capacity is not None:
        lane_remaining = {}
        for i, row in obs.lane_capacity.items():
            for j, by_p in row.items():
                for p, cap in by_p.items():
                    lane_remaining[(i, j, p)] = int(cap)

    sku_remaining = None
    if obs.sku_capacity is not None:
        sku_remaining = {}
        for w in obs.warehouses:
            by_p = obs.sku_capacity.get(w.id, {})
            for p, cap in by_p.items():
                sku_remaining[(w.id, p)] = int(cap) - int(w.inventory.get(p, 0))

    lots = {} if obs.min_transfer_lot is None else {p: int(v) for p, v in obs.min_transfer_lot.items()}

    fixed_cost = {} if obs.lane_fixed_cost is None else {
        (i, j): float(c)
        for i, row in obs.lane_fixed_cost.items()
        for j, c in row.items()
    }
    activated_lanes: set[tuple[str, str]] = set()

    # Greedy: for each product, send from surplus to deficit by cheapest lane
    for p in products:
        surplus: dict[str, int] = {}
        deficit: dict[str, int] = {}
        for w in obs.warehouses:
            inv_units = int(w.inventory.get(p, 0))
            dem_units = int(w.demand.get(p, 0))
            if inv_units > dem_units:
                surplus[w.id] = inv_units - dem_units
            elif dem_units > inv_units:
                deficit[w.id] = dem_units - inv_units

        while surplus and deficit and remaining_budget > 1e-9:
            # pick best (i,j) lane cost
            best: tuple[str, str, float] | None = None
            for i in surplus:
                for j in deficit:
                    c = float(obs.transfer_cost[i][j])
                    if not math.isfinite(c) or c < 0:
                        continue

                    if lane_remaining is not None and lane_remaining.get((i, j, p), 0) <= 0:
                        continue

                    # Prefer reusing an already activated lane to avoid paying fixed cost again.
                    effective_cost = c
                    if (i, j) not in activated_lanes:
                        effective_cost += float(fixed_cost.get((i, j), 0.0))

                    if best is None or effective_cost < best[2]:
                        best = (i, j, effective_cost)
            if best is None:
                break
            i, j, _ = best

            q = min(surplus[i], deficit[j])
            if remaining_out is not None:
                q = min(q, max(0, remaining_out.get(i, 0)))
            if remaining_in is not None:
                q = min(q, max(0, remaining_in.get(j, 0)))

            if lane_remaining is not None:
                q = min(q, max(0, lane_remaining.get((i, j, p), 0)))

            if sku_remaining is not None:
                q = min(q, max(0, sku_remaining.get((j, p), 0)))

            lot = int(lots.get(p, 0) or 0)
            if lot > 0:
                q = (int(q) // lot) * lot

            lane_cost = float(obs.transfer_cost[i][j])
            lane_fixed = 0.0
            if (i, j) not in activated_lanes:
                lane_fixed = float(fixed_cost.get((i, j), 0.0))

            if lane_cost > 0:
                if math.isfinite(remaining_budget):
                    q = min(q, int(remaining_budget // lane_cost))
            else:
                q = int(q)

            if lot > 0:
                q = (int(q) // lot) * lot

            # Ensure we can afford the fixed activation if this is a new lane.
            if lane_fixed > 0 and math.isfinite(remaining_budget) and remaining_budget <= lane_fixed + 1e-9:
                break

            if q <= 0:
                break
            transfers.append(Transfer(from_warehouse=i, to_warehouse=j, product=p, quantity=int(q)))

            if (i, j) not in activated_lanes and lane_fixed > 0:
                remaining_budget -= lane_fixed
                activated_lanes.add((i, j))

            remaining_budget -= lane_cost * q
            if remaining_out is not None:
                remaining_out[i] = int(remaining_out.get(i, 0) - q)
            if remaining_in is not None:
                remaining_in[j] = int(remaining_in.get(j, 0) - q)

            if lane_remaining is not None:
                lane_remaining[(i, j, p)] = int(lane_remaining.get((i, j, p), 0) - q)

            if sku_remaining is not None:
                sku_remaining[(j, p)] = int(sku_remaining.get((j, p), 0) - q)

            surplus[i] -= q
            deficit[j] -= q
            if surplus[i] == 0:
                surplus.pop(i)
            if deficit[j] == 0:
                deficit.pop(j)

    return InventoryTransferAction(transfers=transfers)


def _candidate_lane_seeds(obs, top_k: int = 10) -> list[InventoryTransferAction]:
    wh_ids = [w.id for w in obs.warehouses]
    products = list(obs.products)
    inv0 = {(w.id, p): int(w.inventory.get(p, 0)) for w in obs.warehouses for p in products}
    dem0 = {(w.id, p): int(w.demand.get(p, 0)) for w in obs.warehouses for p in products}

    fixed_cost = {} if obs.lane_fixed_cost is None else {
        (i, j): float(c) for i, row in obs.lane_fixed_cost.items() for j, c in row.items()
    }
    lots = {} if obs.min_transfer_lot is None else {p: int(v) for p, v in obs.min_transfer_lot.items()}

    scored: list[tuple[float, str, str, str]] = []
    for p in sorted(products):
        lot = int(lots.get(p, 0) or 0)
        q0 = lot if lot > 0 else 1
        for i in sorted(wh_ids):
            s = max(0, int(inv0[(i, p)]) - int(dem0[(i, p)]))
            if s <= 0:
                continue
            for j in sorted(wh_ids):
                if i == j:
                    continue
                d = max(0, int(dem0[(j, p)]) - int(inv0[(j, p)]))
                if d <= 0:
                    continue
                if obs.lane_capacity is not None and obs.lane_capacity.get(i, {}).get(j, {}).get(p) is not None:
                    if int(obs.lane_capacity[i][j][p]) <= 0:
                        continue
                ship = float(obs.transfer_cost[i][j])
                fixed = float(fixed_cost.get((i, j), 0.0))
                # Net benefit per unit (using penalty as marginal value of meeting demand).
                net = float(obs.penalty_per_unit_shortage) - ship
                # Approximate fixed activation amortized over a min-lot-sized shipment.
                if q0 > 0:
                    net -= fixed / float(q0)
                if net <= 1e-9:
                    continue
                scored.append((net, p, i, j))

    scored.sort(reverse=True)
    seeds: list[InventoryTransferAction] = []
    for net, p, i, j in scored[:top_k]:
        _ = net
        lot = int(lots.get(p, 0) or 0)
        q = lot if lot > 0 else 1
        seeds.append(
            InventoryTransferAction(
                transfers=[Transfer(from_warehouse=i, to_warehouse=j, product=p, quantity=int(q))]
            )
        )
    return seeds


def _multi_start_improve(obs, base_action: InventoryTransferAction, *, seeds_k: int = 10) -> InventoryTransferAction:
    # Only do multi-start when it likely matters (fixed-charge / larger instances).
    if len(obs.warehouses) < 6 and not obs.lane_fixed_cost:
        return _polish_action(obs, base_action, max_iters=50)

    candidate_seeds = _candidate_lane_seeds(obs, top_k=seeds_k)

    actions: list[InventoryTransferAction] = []
    actions.append(base_action)
    actions.extend(candidate_seeds)

    # Add pairwise seeds for stronger exploration (bounded).
    for a in candidate_seeds[: min(6, len(candidate_seeds))]:
        for b in candidate_seeds[: min(6, len(candidate_seeds))]:
            if a is b:
                continue
            actions.append(InventoryTransferAction(transfers=list(a.transfers) + list(b.transfers)))

    best_action = InventoryTransferAction(transfers=[])
    best_cost = float("inf")

    for act in actions:
        repaired = _repair_action(obs, act)
        polished = _polish_action(obs, repaired, max_iters=60)
        cost, _fr, ok = _simulate_total_cost(obs, polished)
        if ok and cost < best_cost - 1e-9:
            best_cost = cost
            best_action = polished

    if best_cost == float("inf"):
        return _repair_action(obs, base_action)
    return best_action


def _repair_action(obs, action: InventoryTransferAction) -> InventoryTransferAction:
    wh_ids = [w.id for w in obs.warehouses]
    products = list(obs.products)

    inv = {(w.id, p): int(w.inventory.get(p, 0)) for w in obs.warehouses for p in products}

    remaining_budget = float("inf") if obs.budget is None else float(obs.budget)
    remaining_out = None if obs.outbound_capacity is None else {k: int(v) for k, v in obs.outbound_capacity.items()}
    remaining_in = None if obs.inbound_capacity is None else {k: int(v) for k, v in obs.inbound_capacity.items()}

    lane_remaining = None
    if obs.lane_capacity is not None:
        lane_remaining = {}
        for i, row in obs.lane_capacity.items():
            for j, by_p in row.items():
                for p, cap in by_p.items():
                    lane_remaining[(i, j, p)] = int(cap)

    sku_remaining = None
    if obs.sku_capacity is not None:
        sku_remaining = {}
        for w in obs.warehouses:
            by_p = obs.sku_capacity.get(w.id, {})
            for p, cap in by_p.items():
                sku_remaining[(w.id, p)] = int(cap) - int(w.inventory.get(p, 0))

    lots = {} if obs.min_transfer_lot is None else {p: int(v) for p, v in obs.min_transfer_lot.items()}
    fixed_cost = {} if obs.lane_fixed_cost is None else {
        (i, j): float(c) for i, row in obs.lane_fixed_cost.items() for j, c in row.items()
    }
    activated_lanes: set[tuple[str, str]] = set()

    repaired: list[Transfer] = []
    for t in action.transfers:
        if t.from_warehouse not in wh_ids or t.to_warehouse not in wh_ids:
            continue
        if t.product not in products:
            continue
        if t.from_warehouse == t.to_warehouse:
            continue

        q = int(t.quantity)
        if q <= 0:
            continue

        q = min(q, inv[(t.from_warehouse, t.product)])
        if remaining_out is not None:
            q = min(q, max(0, remaining_out.get(t.from_warehouse, 0)))
        if remaining_in is not None:
            q = min(q, max(0, remaining_in.get(t.to_warehouse, 0)))
        if lane_remaining is not None:
            q = min(q, max(0, lane_remaining.get((t.from_warehouse, t.to_warehouse, t.product), 0)))
        if sku_remaining is not None:
            q = min(q, max(0, sku_remaining.get((t.to_warehouse, t.product), 0)))

        lot = int(lots.get(t.product, 0) or 0)
        if lot > 0:
            q = (q // lot) * lot

        if q <= 0:
            continue

        lane_cost = float(obs.transfer_cost[t.from_warehouse][t.to_warehouse])
        is_new_lane = (t.from_warehouse, t.to_warehouse) not in activated_lanes
        lane_fixed = float(fixed_cost.get((t.from_warehouse, t.to_warehouse), 0.0)) if is_new_lane else 0.0

        # Ensure we can afford fixed activation + variable cost.
        if math.isfinite(remaining_budget):
            if lane_fixed > 0 and remaining_budget <= lane_fixed + 1e-9:
                continue

            avail_for_variable = remaining_budget - lane_fixed
            if avail_for_variable < 0:
                continue

            if lane_cost > 0:
                q = min(q, int(avail_for_variable // lane_cost))
            if lot > 0:
                q = (q // lot) * lot
            if q <= 0:
                continue

            # Final safety: total spend must fit (after lot rounding).
            if lane_fixed + lane_cost * q > remaining_budget + 1e-9:
                if lane_cost <= 0:
                    continue
                q = int((avail_for_variable // lane_cost) // max(1, lot) * max(1, lot)) if lot > 0 else int(avail_for_variable // lane_cost)
                if q <= 0:
                    continue

        # Apply spend
        if is_new_lane and lane_fixed > 0:
            remaining_budget -= lane_fixed
            activated_lanes.add((t.from_warehouse, t.to_warehouse))
        remaining_budget -= lane_cost * q

        inv[(t.from_warehouse, t.product)] -= q
        inv[(t.to_warehouse, t.product)] += q

        if remaining_out is not None:
            remaining_out[t.from_warehouse] = int(remaining_out.get(t.from_warehouse, 0) - q)
        if remaining_in is not None:
            remaining_in[t.to_warehouse] = int(remaining_in.get(t.to_warehouse, 0) - q)
        if lane_remaining is not None:
            lane_remaining[(t.from_warehouse, t.to_warehouse, t.product)] = int(
                lane_remaining.get((t.from_warehouse, t.to_warehouse, t.product), 0) - q
            )
        if sku_remaining is not None:
            sku_remaining[(t.to_warehouse, t.product)] = int(sku_remaining.get((t.to_warehouse, t.product), 0) - q)

        repaired.append(
            Transfer(
                from_warehouse=t.from_warehouse,
                to_warehouse=t.to_warehouse,
                product=t.product,
                quantity=int(q),
            )
        )

    return InventoryTransferAction(transfers=repaired)


def _apply_action_to_state(obs, action: InventoryTransferAction):
    wh_ids = [w.id for w in obs.warehouses]
    products = list(obs.products)

    inv = {(w.id, p): int(w.inventory.get(p, 0)) for w in obs.warehouses for p in products}
    dem = {(w.id, p): int(w.demand.get(p, 0)) for w in obs.warehouses for p in products}

    remaining_budget = float("inf") if obs.budget is None else float(obs.budget)
    remaining_out = None if obs.outbound_capacity is None else {k: int(v) for k, v in obs.outbound_capacity.items()}
    remaining_in = None if obs.inbound_capacity is None else {k: int(v) for k, v in obs.inbound_capacity.items()}

    lane_remaining = None
    if obs.lane_capacity is not None:
        lane_remaining = {}
        for i, row in obs.lane_capacity.items():
            for j, by_p in row.items():
                for p, cap in by_p.items():
                    lane_remaining[(i, j, p)] = int(cap)

    sku_remaining = None
    if obs.sku_capacity is not None:
        sku_remaining = {}
        for w in obs.warehouses:
            by_p = obs.sku_capacity.get(w.id, {})
            for p, cap in by_p.items():
                sku_remaining[(w.id, p)] = int(cap) - int(w.inventory.get(p, 0))

    lots = {} if obs.min_transfer_lot is None else {p: int(v) for p, v in obs.min_transfer_lot.items()}
    fixed_cost = {} if obs.lane_fixed_cost is None else {
        (i, j): float(c) for i, row in obs.lane_fixed_cost.items() for j, c in row.items()
    }
    activated_lanes: set[tuple[str, str]] = set()

    transfer_out = {w: 0 for w in wh_ids}
    transfer_in = {w: 0 for w in wh_ids}

    for t in action.transfers:
        if t.from_warehouse == t.to_warehouse:
            continue
        q = int(t.quantity)
        if q <= 0:
            continue
        if (t.from_warehouse, t.product) not in inv:
            continue
        if inv[(t.from_warehouse, t.product)] < q:
            continue

        lot = int(lots.get(t.product, 0) or 0)
        if lot > 0 and (q % lot) != 0:
            continue

        if remaining_out is not None and transfer_out.get(t.from_warehouse, 0) + q > int(remaining_out.get(t.from_warehouse, 0)):
            continue
        if remaining_in is not None and transfer_in.get(t.to_warehouse, 0) + q > int(remaining_in.get(t.to_warehouse, 0)):
            continue
        if lane_remaining is not None and lane_remaining.get((t.from_warehouse, t.to_warehouse, t.product), 0) < q:
            continue
        if sku_remaining is not None and sku_remaining.get((t.to_warehouse, t.product), 0) < q:
            continue

        lane_cost = float(obs.transfer_cost[t.from_warehouse][t.to_warehouse])
        is_new_lane = (t.from_warehouse, t.to_warehouse) not in activated_lanes
        lane_fixed = float(fixed_cost.get((t.from_warehouse, t.to_warehouse), 0.0)) if is_new_lane else 0.0

        spend = lane_fixed + lane_cost * q
        if math.isfinite(remaining_budget) and spend > remaining_budget + 1e-9:
            continue

        remaining_budget -= spend
        if is_new_lane and lane_fixed > 0:
            activated_lanes.add((t.from_warehouse, t.to_warehouse))
        if is_new_lane and lane_fixed == 0:
            activated_lanes.add((t.from_warehouse, t.to_warehouse))

        inv[(t.from_warehouse, t.product)] -= q
        inv[(t.to_warehouse, t.product)] += q
        transfer_out[t.from_warehouse] += q
        transfer_in[t.to_warehouse] += q

        if remaining_out is not None:
            remaining_out[t.from_warehouse] = int(remaining_out.get(t.from_warehouse, 0) - q)
        if remaining_in is not None:
            remaining_in[t.to_warehouse] = int(remaining_in.get(t.to_warehouse, 0) - q)
        if lane_remaining is not None:
            lane_remaining[(t.from_warehouse, t.to_warehouse, t.product)] = int(
                lane_remaining.get((t.from_warehouse, t.to_warehouse, t.product), 0) - q
            )
        if sku_remaining is not None:
            sku_remaining[(t.to_warehouse, t.product)] = int(sku_remaining.get((t.to_warehouse, t.product), 0) - q)

    return {
        "wh_ids": wh_ids,
        "products": products,
        "inv": inv,
        "dem": dem,
        "remaining_budget": remaining_budget,
        "remaining_out": remaining_out,
        "remaining_in": remaining_in,
        "lane_remaining": lane_remaining,
        "sku_remaining": sku_remaining,
        "lots": lots,
        "fixed_cost": fixed_cost,
        "activated_lanes": activated_lanes,
    }


def _polish_action(obs, base_action: InventoryTransferAction, max_iters: int = 50) -> InventoryTransferAction:
    base_action = _repair_action(obs, base_action)
    st = _apply_action_to_state(obs, base_action)

    wh_ids = st["wh_ids"]
    products = st["products"]
    inv = st["inv"]
    dem = st["dem"]
    remaining_budget = st["remaining_budget"]
    remaining_out = st["remaining_out"]
    remaining_in = st["remaining_in"]
    lane_remaining = st["lane_remaining"]
    sku_remaining = st["sku_remaining"]
    lots = st["lots"]
    fixed_cost = st["fixed_cost"]
    activated_lanes = st["activated_lanes"]

    transfers = list(base_action.transfers)

    for _ in range(max_iters):
        if not math.isfinite(remaining_budget) or remaining_budget <= 1e-9:
            break

        best = None
        for p in sorted(products):
            deficit = {w: max(0, int(dem[(w, p)]) - int(inv[(w, p)])) for w in wh_ids}
            surplus = {w: max(0, int(inv[(w, p)]) - int(dem[(w, p)])) for w in wh_ids}
            if sum(deficit.values()) <= 0 or sum(surplus.values()) <= 0:
                continue

            lot = int(lots.get(p, 0) or 0)
            step_q = lot if lot > 0 else 1
            for i in sorted(wh_ids):
                if surplus.get(i, 0) <= 0:
                    continue
                for j in sorted(wh_ids):
                    if i == j or deficit.get(j, 0) <= 0:
                        continue
                    if lane_remaining is not None and lane_remaining.get((i, j, p), 0) <= 0:
                        continue
                    lane_cost = float(obs.transfer_cost[i][j])
                    is_new_lane = (i, j) not in activated_lanes
                    lane_fixed = float(fixed_cost.get((i, j), 0.0)) if is_new_lane else 0.0

                    if math.isfinite(remaining_budget) and remaining_budget < lane_fixed + lane_cost * step_q - 1e-9:
                        continue

                    net = float(obs.penalty_per_unit_shortage) - lane_cost
                    if is_new_lane and step_q > 0:
                        net -= lane_fixed / float(step_q)

                    if net <= 1e-9:
                        continue
                    cand = (net, p, i, j, lane_fixed)
                    if best is None or cand > best:
                        best = cand

        if best is None:
            break

        _net, p, i, j, lane_fixed = best
        q = min(
            max(0, int(inv[(i, p)]) - int(dem[(i, p)])),
            max(0, int(dem[(j, p)]) - int(inv[(j, p)])),
        )

        if remaining_out is not None:
            q = min(q, max(0, int(remaining_out.get(i, 0))))
        if remaining_in is not None:
            q = min(q, max(0, int(remaining_in.get(j, 0))))
        if lane_remaining is not None:
            q = min(q, max(0, int(lane_remaining.get((i, j, p), 0))))
        if sku_remaining is not None:
            q = min(q, max(0, int(sku_remaining.get((j, p), 0))))

        lot = int(lots.get(p, 0) or 0)
        if lot > 0:
            q = (q // lot) * lot

        if q <= 0:
            break

        lane_cost = float(obs.transfer_cost[i][j])
        spend = lane_fixed + lane_cost * q if (i, j) not in activated_lanes else lane_cost * q
        if math.isfinite(remaining_budget) and spend > remaining_budget + 1e-9:
            if lane_cost <= 0:
                break
            q = int(max(0.0, remaining_budget - (lane_fixed if (i, j) not in activated_lanes else 0.0)) // lane_cost)
            if lot > 0:
                q = (q // lot) * lot
            if q <= 0:
                break
            spend = lane_fixed + lane_cost * q if (i, j) not in activated_lanes else lane_cost * q

        if (i, j) not in activated_lanes and lane_fixed > 0:
            remaining_budget -= lane_fixed
            activated_lanes.add((i, j))
        elif (i, j) not in activated_lanes:
            activated_lanes.add((i, j))

        remaining_budget -= lane_cost * q
        inv[(i, p)] -= q
        inv[(j, p)] += q
        if remaining_out is not None:
            remaining_out[i] = int(remaining_out.get(i, 0) - q)
        if remaining_in is not None:
            remaining_in[j] = int(remaining_in.get(j, 0) - q)
        if lane_remaining is not None:
            lane_remaining[(i, j, p)] = int(lane_remaining.get((i, j, p), 0) - q)
        if sku_remaining is not None:
            sku_remaining[(j, p)] = int(sku_remaining.get((j, p), 0) - q)

        transfers.append(Transfer(from_warehouse=i, to_warehouse=j, product=p, quantity=int(q)))

    return InventoryTransferAction(transfers=transfers)


def _reset_once(base_url: str, task_id: str):
    last: Exception | None = None
    for attempt in range(6):
        try:
            with InventoryTransferEnv(base_url=base_url).sync() as env:
                r = env.reset(task_id=task_id)
                return r.observation
        except Exception as e:
            last = e
            msg = str(e)
            retryable = (
                "CAPACITY_REACHED" in msg
                or "Server at capacity" in msg
                or "received 1000" in msg
                or "ConnectionClosed" in msg
            )
            if not retryable or attempt >= 5:
                raise
            time.sleep(0.5 * (2**attempt))
    raise last  # type: ignore[misc]


def _step_once(base_url: str, task_id: str, action: InventoryTransferAction):
    # Use a fresh connection for the step to avoid WS idle timeouts while the LLM is thinking.
    last: Exception | None = None
    for attempt in range(6):
        try:
            with InventoryTransferEnv(base_url=base_url).sync() as env:
                _ = env.reset(task_id=task_id)
                return env.step(action)
        except Exception as e:
            last = e
            msg = str(e)
            retryable = (
                "CAPACITY_REACHED" in msg
                or "Server at capacity" in msg
                or "received 1000" in msg
                or "ConnectionClosed" in msg
            )
            if not retryable or attempt >= 5:
                raise
            time.sleep(0.5 * (2**attempt))
    raise last  # type: ignore[misc]


def _run_multi_step_episode(base_url: str, task_id: str, mip_time_limit_ms: int):
    """Run a multi-step episode using a single WebSocket connection.

    Each step re-solves the MIP on the current observation (receding-horizon),
    which sees the updated inventory state returned by the previous step.
    """
    last: Exception | None = None
    for attempt in range(6):
        try:
            with InventoryTransferEnv(base_url=base_url).sync() as env:
                reset_result = env.reset(task_id=task_id)
                current_obs = reset_result.observation
                max_steps = getattr(current_obs, "max_steps", 1)

                step_result = None
                for _s in range(max_steps):
                    step_action = _mip_plan(current_obs, time_limit_ms=mip_time_limit_ms)
                    _, _, raw_ok = _simulate_total_cost(current_obs, step_action)
                    if not raw_ok or not step_action.transfers:
                        step_action = _greedy_plan(current_obs)
                    step_result = env.step(step_action)
                    if not step_result.done:
                        current_obs = step_result.observation
                if step_result is None:
                    raise RuntimeError("Episode produced no step results")
                return step_result
        except Exception as e:
            last = e
            msg = str(e)
            retryable = (
                "CAPACITY_REACHED" in msg
                or "Server at capacity" in msg
                or "received 1000" in msg
                or "ConnectionClosed" in msg
            )
            if not retryable or attempt >= 5:
                raise
            time.sleep(0.5 * (2**attempt))
    raise last  # type: ignore[misc]


def _simulate_total_cost(obs, action: InventoryTransferAction) -> tuple[float, float, bool]:
    wh_ids = [w.id for w in obs.warehouses]
    products = list(obs.products)
    inv = {(w.id, p): int(w.inventory.get(p, 0)) for w in obs.warehouses for p in products}
    dem = {(w.id, p): int(w.demand.get(p, 0)) for w in obs.warehouses for p in products}

    transfer_out = {w: 0 for w in wh_ids}
    transfer_in = {w: 0 for w in wh_ids}

    lane_used: dict[tuple[str, str, str], int] = {}
    transfer_cost_total = 0.0
    violations: list[str] = []

    lots = {} if obs.min_transfer_lot is None else {p: int(v) for p, v in obs.min_transfer_lot.items()}

    for t in action.transfers:
        if t.from_warehouse == t.to_warehouse:
            continue
        q = int(t.quantity)
        if q < 0:
            violations.append("negative quantity")
            continue
        if t.from_warehouse not in wh_ids or t.to_warehouse not in wh_ids:
            violations.append("unknown warehouse")
            continue
        if t.product not in products:
            violations.append("unknown product")
            continue

        lot = int(lots.get(t.product, 0) or 0)
        if lot > 0 and q > 0 and (q % lot) != 0:
            violations.append("min transfer lot violation")
            continue

        if inv[(t.from_warehouse, t.product)] < q:
            violations.append("insufficient inventory")
            continue

        transfer_cost_total += float(obs.transfer_cost[t.from_warehouse][t.to_warehouse]) * q
        inv[(t.from_warehouse, t.product)] -= q
        inv[(t.to_warehouse, t.product)] += q
        transfer_out[t.from_warehouse] += q
        transfer_in[t.to_warehouse] += q
        lane_used[(t.from_warehouse, t.to_warehouse, t.product)] = lane_used.get(
            (t.from_warehouse, t.to_warehouse, t.product), 0
        ) + q

    lane_activation_cost_total = 0.0
    if obs.lane_fixed_cost:
        activated = {(i, j) for (i, j, _p), used in lane_used.items() if used > 0}
        for (i, j) in activated:
            lane_activation_cost_total += float(obs.lane_fixed_cost.get(i, {}).get(j, 0.0))

    # Constraints
    if obs.outbound_capacity:
        for w, cap in obs.outbound_capacity.items():
            if transfer_out.get(w, 0) > int(cap):
                violations.append("outbound cap exceeded")

    if obs.inbound_capacity:
        for w, cap in obs.inbound_capacity.items():
            if transfer_in.get(w, 0) > int(cap):
                violations.append("inbound cap exceeded")

    if obs.lane_capacity:
        for (i, j, p), used in lane_used.items():
            cap = obs.lane_capacity.get(i, {}).get(j, {}).get(p)
            if cap is not None and used > int(cap):
                violations.append("lane cap exceeded")

    if obs.sku_capacity:
        for w in wh_ids:
            by_p = obs.sku_capacity.get(w, {})
            for p in products:
                cap = by_p.get(p)
                if cap is not None and inv[(w, p)] > int(cap):
                    violations.append("sku cap exceeded")

    total_transfer_related = transfer_cost_total + lane_activation_cost_total
    if obs.budget is not None and total_transfer_related > float(obs.budget) + 1e-9:
        violations.append("budget exceeded")

    shortage_units_total = 0
    total_demand_units = 0
    for w in wh_ids:
        for p in products:
            total_demand_units += int(dem[(w, p)])
            shortage_units_total += int(max(0, dem[(w, p)] - inv[(w, p)]))

    shortage_penalty_total = float(shortage_units_total) * float(obs.penalty_per_unit_shortage)
    total_cost = transfer_cost_total + lane_activation_cost_total + shortage_penalty_total

    fill_rate = 1.0
    if total_demand_units > 0:
        fill_rate = float((total_demand_units - shortage_units_total) / total_demand_units)

    ok = len(violations) == 0
    return float(total_cost), float(fill_rate), ok


def _score_from_obs(obs) -> float:
    if obs.score is not None:
        return float(obs.score)

    optimal_cost = obs.optimal_cost
    optimal_feasible = obs.optimal_feasible

    if not obs.is_feasible:
        return 0.0

    if optimal_feasible is False:
        return 0.0

    if optimal_cost is None or optimal_cost <= 0 or optimal_cost == float("inf"):
        return 0.0

    return max(0.0, min(1.0, optimal_cost / max(obs.total_cost, optimal_cost)))


def _llm_plan(client: OpenAI, model_name: str, obs) -> InventoryTransferAction:
    prompt = {
        "task": "inventory_transfer_planning",
        "instruction": (
            "Return a JSON object with a single key 'transfers' containing a list of transfers. "
            "Each transfer must have keys: from_warehouse, to_warehouse, product, quantity (non-negative int). "
            "Respect all constraints exposed in the observation (budget, caps, min lots, etc.). "
            "If no transfers are needed, return an empty list."
        ),
        "observation": obs.model_dump(),
        "output_schema": {
            "transfers": [
                {
                    "from_warehouse": "str",
                    "to_warehouse": "str",
                    "product": "str",
                    "quantity": "int"
                }
            ]
        },
    }

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a supply chain optimization agent."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or "{}"
    # Be tolerant to models wrapping JSON in text/code fences.
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    data = json.loads(m.group(0) if m else content)
    return InventoryTransferAction.model_validate(data)


def _mip_plan(obs: InventoryTransferObservation, time_limit_ms: int = 800) -> InventoryTransferAction:
    wh_ids = [w.id for w in obs.warehouses]
    products = list(obs.products)

    inv0 = {(w.id, p): int(w.inventory.get(p, 0)) for w in obs.warehouses for p in products}
    dem0 = {(w.id, p): int(w.demand.get(p, 0)) for w in obs.warehouses for p in products}

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        return InventoryTransferAction(transfers=[])
    solver.SetTimeLimit(int(time_limit_ms))

    def _v(expr: pywraplp.LinearExpr | pywraplp.Variable) -> Any:
        return cast(Any, expr)

    x: dict[tuple[str, str, str], pywraplp.Variable] = {}
    y: dict[tuple[str, str], pywraplp.Variable] = {}

    for i in wh_ids:
        for j in wh_ids:
            if i == j:
                continue
            y[(i, j)] = solver.IntVar(0.0, 1.0, f"y_{i}_{j}")
            for p in products:
                x[(i, j, p)] = solver.IntVar(0.0, solver.infinity(), f"x_{i}_{j}_{p}")

    shortage: dict[tuple[str, str], pywraplp.Variable] = {}
    for j in wh_ids:
        for p in products:
            shortage[(j, p)] = solver.NumVar(0.0, solver.infinity(), f"short_{j}_{p}")

    # Inventory availability
    for i in wh_ids:
        for p in products:
            solver.Add(
                solver.Sum([x[(i, j, p)] for j in wh_ids if j != i]) <= inv0[(i, p)]
            )

    # Outbound/inbound capacities
    if obs.outbound_capacity:
        for i, cap in obs.outbound_capacity.items():
            if i in wh_ids:
                solver.Add(
                    solver.Sum(
                        [x[(i, j, p)] for j in wh_ids for p in products if j != i]
                    )
                    <= int(cap)
                )

    if obs.inbound_capacity:
        for j, cap in obs.inbound_capacity.items():
            if j in wh_ids:
                solver.Add(
                    solver.Sum(
                        [x[(i, j, p)] for i in wh_ids for p in products if i != j]
                    )
                    <= int(cap)
                )

    # Shortage definition (flow balance)
    for j in wh_ids:
        for p in products:
            inflow = solver.Sum([x[(i, j, p)] for i in wh_ids if i != j])
            outflow = solver.Sum([x[(j, k, p)] for k in wh_ids if k != j])
            solver.Add(shortage[(j, p)] >= dem0[(j, p)] - (inv0[(j, p)] + inflow - outflow))

    # Lane caps
    if obs.lane_capacity:
        for i, row in obs.lane_capacity.items():
            if i not in wh_ids:
                continue
            for j, by_p in row.items():
                if j not in wh_ids or i == j:
                    continue
                for p, cap in by_p.items():
                    if p in products:
                        solver.Add(_v(x[(i, j, p)]) <= int(cap))

    # SKU caps after transfers
    if obs.sku_capacity:
        for j in wh_ids:
            by_p = obs.sku_capacity.get(j, {})
            for p in products:
                cap = by_p.get(p)
                if cap is None:
                    continue
                inflow = solver.Sum([x[(i, j, p)] for i in wh_ids if i != j])
                outflow = solver.Sum([x[(j, k, p)] for k in wh_ids if k != j])
                solver.Add(inv0[(j, p)] + inflow - outflow <= int(cap))

    # Min transfer lots: enforce x is always a multiple of lot via x = lot * k
    if obs.min_transfer_lot:
        for i in wh_ids:
            for j in wh_ids:
                if i == j:
                    continue
                for p in products:
                    lot = int(obs.min_transfer_lot.get(p, 0) or 0)
                    if lot <= 0:
                        continue
                    max_k = inv0[(i, p)] // lot
                    k = solver.IntVar(0, max_k, f"k_{i}_{j}_{p}")
                    solver.Add(x[(i, j, p)] == lot * k)

    # Lane activation
    for (i, j), yvar in y.items():
        max_out = sum(inv0[(i, p)] for p in products)
        if max_out <= 0:
            solver.Add(yvar == 0)
            continue
        solver.Add(_v(solver.Sum([x[(i, j, p)] for p in products])) <= max_out * _v(yvar))

    # Budget
    if obs.budget is not None:
        solver.Add(
            solver.Sum([
                float(obs.transfer_cost[i][j]) * _v(x[(i, j, p)])
                for (i, j, p) in x.keys()
            ])
            + (
                solver.Sum([
                    float(obs.lane_fixed_cost.get(i, {}).get(j, 0.0)) * _v(y[(i, j)])
                    for (i, j) in y.keys()
                ])
                if obs.lane_fixed_cost
                else 0.0
            )
            <= float(obs.budget)
        )

    solver.Minimize(
        solver.Sum([
            float(obs.transfer_cost[i][j]) * _v(x[(i, j, p)])
            for (i, j, p) in x.keys()
        ])
        + (
            solver.Sum([
                float(obs.lane_fixed_cost.get(i, {}).get(j, 0.0)) * _v(y[(i, j)])
                for (i, j) in y.keys()
            ])
            if obs.lane_fixed_cost
            else 0.0
        )
        + float(obs.penalty_per_unit_shortage)
        * solver.Sum([_v(shortage[(j, p)]) for j in wh_ids for p in products])
    )

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return InventoryTransferAction(transfers=[])

    transfers: list[Transfer] = []
    for (i, j, p), var in x.items():
        q = int(round(var.solution_value()))
        if q > 0:
            transfers.append(Transfer(from_warehouse=i, to_warehouse=j, product=p, quantity=q))

    return InventoryTransferAction(transfers=transfers)


def _load_task_ids() -> list[str]:
    tasks_path = Path(__file__).resolve().parent / "inventory_transfer_env" / "tasks.json"
    if not tasks_path.exists():
        return ["easy", "medium", "hard", "hard_v1", "hard_v2", "hard_v3", "edge_case"]
    return list(json.loads(tasks_path.read_text()).keys())


def _health_ok(base_url: str, timeout_s: float = 5.0) -> bool:
    url = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _is_localhost_url(url: str) -> bool:
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _autostart_server(repo_root: Path) -> tuple[subprocess.Popen[bytes], str]:
    """Start uvicorn in the background and return (proc, base_url) once /health responds."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Server process exited unexpectedly during startup.")
        if _health_ok(base_url, timeout_s=2.0):
            return proc, base_url
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"Server did not become healthy within 30 s on port {port}.")


def main() -> None:
    base_url = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
    print_planner = os.environ.get("PRINT_PLANNER", "0") == "1"

    use_mip = os.environ.get("USE_MIP", "1") == "1"
    mip_time_limit_ms = int(os.environ.get("MIP_TIME_LIMIT_MS", "120000"))

    # Default to LLM mode when variables are present, but fall back safely to greedy if not.
    use_llm = os.environ.get("USE_LLM", "1") == "1"
    force_llm = os.environ.get("FORCE_LLM", "0") == "1"
    print_llm_errors = os.environ.get("PRINT_LLM_ERRORS", "0") == "1"
    api_base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    hf_token = os.environ.get("HF_TOKEN")

    client = None
    if use_llm and api_base_url and model_name and hf_token:
        client = OpenAI(base_url=api_base_url, api_key=hf_token)
    else:
        use_llm = False

    if force_llm and not use_llm:
        missing = [
            k
            for k, v in (("API_BASE_URL", api_base_url), ("MODEL_NAME", model_name), ("HF_TOKEN", hf_token))
            if not v
        ]
        print(f"FORCE_LLM=1 but LLM is disabled (missing env vars: {', '.join(missing)})")

    tasks = _load_task_ids()

    _autostarted_proc: subprocess.Popen[bytes] | None = None
    if not _health_ok(base_url):
        if _is_localhost_url(base_url):
            print(f"No server at {base_url} — auto-starting local server ...")
            repo_root = Path(__file__).resolve().parent
            _autostarted_proc, base_url = _autostart_server(repo_root)
            print(f"Server ready at {base_url}")
        else:
            raise RuntimeError(
                f"Server health check failed at {base_url.rstrip('/')}/health. "
                "Set ENV_BASE_URL to a running deployment."
            )

    results = []
    for task_id in tasks:
        try:
            obs = _reset_once(base_url, task_id)
        except Exception as e:
            print(f"skip {task_id}: reset failed: {e}")
            continue

        greedy_action = _repair_action(obs, _greedy_plan(obs))
        greedy_cost, _, greedy_ok = _simulate_total_cost(obs, greedy_action)

        greedy_improved = _multi_start_improve(obs, greedy_action, seeds_k=10)
        greedy_improved_cost, _, greedy_improved_ok = _simulate_total_cost(obs, greedy_improved)

        mip_action = None
        mip_cost = float("inf")
        mip_ok = False
        if use_mip:
            try:
                mip_raw = _mip_plan(obs, time_limit_ms=mip_time_limit_ms)
                mip_cost, _, mip_ok = _simulate_total_cost(obs, mip_raw)
                # MIP is already globally optimal; skip _repair_action and
                # _multi_start_improve — both call _repair_action internally
                # which processes transfers sequentially and incorrectly drops
                # transfers whose feasibility depends on earlier transfers
                # increasing intermediate inventory (e.g. W1->W8 then W8->W2).
                mip_action = mip_raw if mip_ok else None
            except Exception:
                mip_action = None

        action = InventoryTransferAction(transfers=[])
        planner = "empty"
        best_cost = float("inf")

        if greedy_ok:
            action = greedy_action
            planner = "greedy"
            best_cost = greedy_cost
        if greedy_improved_ok and greedy_improved_cost <= best_cost + 1e-9:
            action = greedy_improved
            planner = "greedy_improve"
            best_cost = greedy_improved_cost

        if mip_action is not None and mip_ok and mip_cost <= best_cost + 1e-9:
            action = mip_action
            planner = "mip"
            best_cost = mip_cost
        if use_llm and client is not None and model_name is not None:
            try:
                llm_action = _llm_plan(client, model_name, obs)
                llm_action = _repair_action(obs, llm_action)
                llm_cost, _, llm_ok = _simulate_total_cost(obs, llm_action)

                llm_improved = _multi_start_improve(obs, llm_action, seeds_k=10)
                llm_improved_cost, _, llm_improved_ok = _simulate_total_cost(obs, llm_improved)

                if force_llm:
                    if llm_improved_ok:
                        action = llm_improved
                        planner = "llm_improve"
                    elif llm_ok:
                        action = llm_action
                        planner = "llm"
                    else:
                        action = InventoryTransferAction(transfers=[])
                        planner = "empty"
                    best_cost = llm_improved_cost if llm_improved_ok else (llm_cost if llm_ok else float("inf"))
                else:
                    if llm_ok and llm_cost <= best_cost + 1e-9:
                        action = llm_action
                        planner = "llm"
                        best_cost = llm_cost

                    if llm_improved_ok and llm_improved_cost <= best_cost + 1e-9:
                        action = llm_improved
                        planner = "llm_improve"
                        best_cost = llm_improved_cost
            except Exception as e:
                if print_llm_errors:
                    print(f"llm_error[{task_id}]={type(e).__name__}: {e}")

        if print_planner:
            print(f"planner[{task_id}]={planner}")

        max_steps = getattr(obs, "max_steps", 1)
        if max_steps > 1:
            # Multi-step episode: keep WS connection open, re-solve MIP each step
            step = _run_multi_step_episode(base_url, task_id, mip_time_limit_ms)
        else:
            step = _step_once(base_url, task_id, action)
        score = _score_from_obs(step.observation)
        results.append((task_id, score, step.observation.total_cost, step.observation))

    for task_id, score, cost, obs in results:
        dq = "DQ" if obs.disqualified else "OK"
        dq_reasons = "; ".join(obs.dq_reasons) if obs.dq_reasons else ""
        print(
            f"{task_id}: score={score:.4f} cost={cost:.2f} {dq} "
            f"opt_cost={obs.optimal_cost} ratio={obs.optimality_ratio} gap={obs.cost_gap} "
            f"fill_rate={obs.fill_rate:.4f} dq_reasons={dq_reasons}"
        )

    if not results:
        if _autostarted_proc is not None:
            _autostarted_proc.terminate()
        raise RuntimeError("No tasks were successfully evaluated.")

    avg = sum(s for _, s, _, _ in results) / len(results)
    print(f"avg_score={avg:.4f}")

    if _autostarted_proc is not None:
        _autostarted_proc.terminate()


if __name__ == "__main__":
    main()
