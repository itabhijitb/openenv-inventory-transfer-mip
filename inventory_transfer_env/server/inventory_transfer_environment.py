from __future__ import annotations

import json
import random
import uuid
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from openenv.core.env_server.interfaces import Action, Environment
from ortools.linear_solver import pywraplp

from ..models import (
    InventoryTransferAction,
    InventoryTransferObservation,
    InventoryTransferState,
    Warehouse,
)


def _validate_square_cost_matrix(cost: dict[str, dict[str, float]], nodes: list[str]) -> None:
    for i in nodes:
        if i not in cost:
            raise ValueError(f"transfer_cost missing row for warehouse '{i}'")
        for j in nodes:
            if j not in cost[i]:
                raise ValueError(f"transfer_cost missing entry cost['{i}']['{j}']")


def _solve_optimal_mip(
    *,
    warehouses: list[Warehouse],
    products: list[str],
    transfer_cost: dict[str, dict[str, float]],
    lane_fixed_cost: dict[str, dict[str, float]] | None,
    penalty_per_unit_shortage: float,
    budget: float | None,
    outbound_capacity: dict[str, int] | None,
    inbound_capacity: dict[str, int] | None,
    sku_capacity: dict[str, dict[str, int]] | None,
    lane_capacity: dict[str, dict[str, dict[str, int]]] | None,
    min_transfer_lot: dict[str, int] | None,
) -> tuple[float, bool]:
    """Solve the lateral transshipment MIP to optimality using OR-Tools SCIP.

    Returns (optimal_cost, feasible). optimal_cost is the true minimum achievable
    total cost (transfer + lane activation + shortage penalty) subject to all
    constraints. feasible=False means no feasible solution exists.

    Lot-size constraint is enforced as x = lot * k (integer multiplier) so the
    MIP optimum is always achievable by a valid agent action.
    """
    wh_ids = [w.id for w in warehouses]
    _validate_square_cost_matrix(transfer_cost, wh_ids)

    inv: dict[tuple[str, str], int] = {}
    dem: dict[tuple[str, str], int] = {}
    for w in warehouses:
        for p in products:
            inv[(w.id, p)] = int(w.inventory.get(p, 0))
            dem[(w.id, p)] = int(w.demand.get(p, 0))

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver is not available")

    x: dict[tuple[str, str, str], pywraplp.Variable] = {}
    y_lane: dict[tuple[str, str], pywraplp.Variable] = {}
    for i in wh_ids:
        for j in wh_ids:
            if i == j:
                continue
            y_lane[(i, j)] = solver.IntVar(0.0, 1.0, f"y_{i}_{j}")
            for p in products:
                x[(i, j, p)] = solver.IntVar(0.0, solver.infinity(), f"x_{i}_{j}_{p}")

    shortage: dict[tuple[str, str], pywraplp.Variable] = {}
    for j in wh_ids:
        for p in products:
            shortage[(j, p)] = solver.NumVar(0.0, solver.infinity(), f"short_{j}_{p}")

    # Inventory availability: outbound per warehouse per product
    for i in wh_ids:
        for p in products:
            solver.Add(
                sum(x[(i, j, p)] for j in wh_ids if j != i) <= inv[(i, p)]
            )

    # Optional aggregate outbound/inbound capacities
    if outbound_capacity:
        for i, cap in outbound_capacity.items():
            if i not in wh_ids:
                continue
            solver.Add(
                sum(
                    x[(i, j, p)]
                    for j in wh_ids
                    for p in products
                    if j != i
                )
                <= cap
            )

    if inbound_capacity:
        for j, cap in inbound_capacity.items():
            if j not in wh_ids:
                continue
            solver.Add(
                sum(
                    x[(i, j, p)]
                    for i in wh_ids
                    for p in products
                    if i != j
                )
                <= cap
            )

    # Shortage definition
    for j in wh_ids:
        for p in products:
            inflow = sum(x[(i, j, p)] for i in wh_ids if i != j)
            outflow = sum(x[(j, k, p)] for k in wh_ids if k != j)
            # shortage >= demand - (inv + inflow - outflow)
            solver.Add(shortage[(j, p)] >= dem[(j, p)] - (inv[(j, p)] + inflow - outflow))

    # Budget constraint
    if budget is not None:
        solver.Add(
            sum(
                transfer_cost[i][j] * x[(i, j, p)]
                for (i, j, p) in x.keys()
            )
            + (
                sum(
                    float(lane_fixed_cost.get(i, {}).get(j, 0.0)) * y_lane[(i, j)]
                    for (i, j) in y_lane.keys()
                )
                if lane_fixed_cost
                else 0.0
            )
            <= budget
        )

    # Lane capacity constraints (by product)
    if lane_capacity:
        for i, row in lane_capacity.items():
            for j, by_product in row.items():
                if i not in wh_ids or j not in wh_ids or i == j:
                    continue
                for p, cap in by_product.items():
                    if p not in products:
                        continue
                    solver.Add(x[(i, j, p)] <= int(cap))

    # SKU capacity constraints at each warehouse after transfers
    if sku_capacity:
        for j in wh_ids:
            by_product = sku_capacity.get(j, {})
            for p in products:
                cap = by_product.get(p)
                if cap is None:
                    continue
                inflow = sum(x[(i, j, p)] for i in wh_ids if i != j)
                outflow = sum(x[(j, k, p)] for k in wh_ids if k != j)
                solver.Add(inv[(j, p)] + inflow - outflow <= int(cap))

    # Min transfer lot: enforce x is an exact multiple of lot via x = lot * k
    if min_transfer_lot:
        for i in wh_ids:
            for j in wh_ids:
                if i == j:
                    continue
                for p in products:
                    lot = int(min_transfer_lot.get(p, 0))
                    if lot <= 0:
                        continue
                    max_k = inv[(i, p)] // lot
                    k = solver.IntVar(0, max_k, f"k_{i}_{j}_{p}")
                    solver.Add(x[(i, j, p)] == lot * k)

    # Lane activation: if any product moves on (i,j), y_lane[i,j] must be 1
    for (i, j) in y_lane.keys():
        max_out = sum(inv[(i, p)] for p in products)
        if max_out <= 0:
            solver.Add(y_lane[(i, j)] == 0)
            continue
        solver.Add(sum(x[(i, j, p)] for p in products) <= max_out * y_lane[(i, j)])

    solver.Minimize(
        sum(transfer_cost[i][j] * x[(i, j, p)] for (i, j, p) in x.keys())
        + (
            sum(
                float(lane_fixed_cost.get(i, {}).get(j, 0.0)) * y_lane[(i, j)]
                for (i, j) in y_lane.keys()
            )
            if lane_fixed_cost
            else 0.0
        )
        + penalty_per_unit_shortage
        * sum(shortage[(j, p)] for j in wh_ids for p in products)
    )

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return float("inf"), False

    return float(solver.Objective().Value()), True


@lru_cache(maxsize=1)
def _load_tasks() -> dict[str, InventoryTransferObservation]:
    tasks_path = Path(__file__).resolve().parent.parent / "tasks.json"
    raw = json.loads(tasks_path.read_text())
    tasks: dict[str, InventoryTransferObservation] = {}
    for task_id, payload in raw.items():
        if isinstance(payload, dict):
            payload = {**payload, "task_id": task_id}
        tasks[task_id] = InventoryTransferObservation.model_validate(payload)
    return tasks


def _build_task_observation(task_id: str) -> InventoryTransferObservation:
    tasks = _load_tasks()
    if task_id not in tasks:
        raise ValueError(f"Unknown task_id '{task_id}'")
    return tasks[task_id].model_copy(deep=True)


def _build_inv_dem(
    warehouses: list[Warehouse], products: list[str]
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    inv: dict[tuple[str, str], int] = {}
    dem: dict[tuple[str, str], int] = {}
    for w in warehouses:
        for p in products:
            inv[(w.id, p)] = int(w.inventory.get(p, 0))
            dem[(w.id, p)] = int(w.demand.get(p, 0))
    return inv, dem


def _attach_optimal_reference(obs: InventoryTransferObservation, problem: InventoryTransferObservation) -> None:
    try:
        optimal_cost, ok = _solve_optimal_mip(
            warehouses=problem.warehouses,
            products=problem.products,
            transfer_cost=problem.transfer_cost,
            lane_fixed_cost=problem.lane_fixed_cost,
            penalty_per_unit_shortage=problem.penalty_per_unit_shortage,
            budget=problem.budget,
            outbound_capacity=problem.outbound_capacity,
            inbound_capacity=problem.inbound_capacity,
            sku_capacity=problem.sku_capacity,
            lane_capacity=problem.lane_capacity,
            min_transfer_lot=problem.min_transfer_lot,
        )
        obs.optimal_cost = optimal_cost
        obs.optimal_feasible = ok
    except Exception as e:
        obs.optimal_cost = None
        obs.optimal_feasible = None
        obs.metadata = {**(obs.metadata or {}), "optimal_cost_error": str(e)}


def _finalize_grading_fields(obs: InventoryTransferObservation) -> None:
    obs.disqualified = not obs.is_feasible
    obs.dq_reasons = list(obs.violations) if obs.disqualified else []
    if obs.optimal_cost is not None and obs.optimal_cost > 0 and obs.optimal_cost != float("inf"):
        obs.optimality_ratio = float(obs.optimal_cost) / max(float(obs.total_cost), float(obs.optimal_cost))
        obs.cost_gap = float(obs.total_cost) - float(obs.optimal_cost)
        if obs.disqualified:
            obs.score = 0.0
        else:
            obs.score = max(0.0, min(1.0, obs.optimality_ratio))
    else:
        obs.score = 0.0


class InventoryTransferEnvironment(Environment):
    def __init__(self):
        self._state = InventoryTransferState(episode_id=str(uuid.uuid4()), step_count=0)
        self._problem: InventoryTransferObservation | None = None

    def reset(self, task_id: str = "easy", **kwargs) -> InventoryTransferObservation:
        """Load task `task_id` and return the initial observation.

        Initialises per-episode state: live inventory (mutated by step()),
        cumulative cost accumulators, and steps_remaining counter.
        Multi-step tasks (max_steps > 1) set done=False until the final step.

        Optional kwargs:
          seed (int): When the task has demand_noise_pct > 0, use this seed to
              sample demand from Normal(mean, mean*noise_pct) per warehouse/product.
              Reproducible: same seed produces identical demand. Omit for
              deterministic behaviour (mean demand used regardless of noise_pct).
        """
        obs = _build_task_observation(task_id)

        seed = kwargs.get("seed", None)
        if seed is not None and obs.demand_noise_pct:
            noise_pct = float(obs.demand_noise_pct)
            rng = random.Random(int(seed))
            perturbed = []
            for w in obs.warehouses:
                new_demand: dict[str, int] = {}
                for p, mean_d in w.demand.items():
                    # Sample demand from Normal; clamp to [0, 2*mean] so negatives/huge values don't arise.
                    sample = rng.gauss(float(mean_d), float(mean_d) * noise_pct)
                    new_demand[p] = max(0, min(int(mean_d * 2), int(round(sample))))
                perturbed.append(Warehouse(id=w.id, inventory=w.inventory, demand=new_demand))
            obs.warehouses = perturbed

        self._problem = deepcopy(obs)
        self._state = InventoryTransferState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            current_inventory={w.id: dict(w.inventory) for w in obs.warehouses},
            cumulative_transfer_cost=0.0,
            cumulative_lane_activation_cost=0.0,
            steps_remaining=obs.max_steps,
        )
        obs.step_number = 0
        obs.done = False
        return obs

    def step(self, action: Action, **kwargs) -> InventoryTransferObservation:
        """Apply `action` and return the updated observation.

        Validates every transfer against inventory, lot sizes, lane/outbound/
        inbound/SKU capacity, and budget (per-step or cumulative). Any violation
        sets disqualified=True and score=0.0.

        On non-final steps: returns reward=-transfer_cost_this_step, done=False.
        On the final step: computes shortage penalty, calls the MIP grader for
        optimal_cost, and returns score = optimal_cost / max(agent_cost, optimal_cost).
        """
        if self._problem is None:
            raise RuntimeError("Environment must be reset() before step().")

        if not isinstance(action, InventoryTransferAction):
            raise ValueError(f"Expected InventoryTransferAction, got {type(action)}")

        self._state.step_count += 1
        self._state.steps_remaining -= 1

        obs = deepcopy(self._problem)
        wh_ids = [w.id for w in obs.warehouses]
        products = list(obs.products)

        # Use live inventory from state (carries across steps); demand is fixed from problem.
        inv: dict[tuple[str, str], int] = {}
        dem: dict[tuple[str, str], int] = {}
        for w in obs.warehouses:
            for p in products:
                inv[(w.id, p)] = int(self._state.current_inventory.get(w.id, {}).get(p, 0))
                dem[(w.id, p)] = int(w.demand.get(p, 0))

        transfer_out: dict[str, int] = {w: 0 for w in wh_ids}
        transfer_in: dict[str, int] = {w: 0 for w in wh_ids}
        lane_used: dict[tuple[str, str, str], int] = {}

        transfer_cost_total = 0.0
        lane_activation_cost_total = 0.0
        violations: list[str] = []
        is_feasible = True

        for t in action.transfers:
            if t.from_warehouse not in wh_ids:
                violations.append(f"unknown from_warehouse '{t.from_warehouse}'")
                is_feasible = False
                continue
            if t.to_warehouse not in wh_ids:
                violations.append(f"unknown to_warehouse '{t.to_warehouse}'")
                is_feasible = False
                continue
            if t.product not in products:
                violations.append(f"unknown product '{t.product}'")
                is_feasible = False
                continue
            if t.from_warehouse == t.to_warehouse:
                continue
            if t.quantity < 0:
                violations.append("negative quantity")
                is_feasible = False
                continue

            if obs.min_transfer_lot:
                lot = int(obs.min_transfer_lot.get(t.product, 0) or 0)
                if lot > 0 and t.quantity > 0 and (t.quantity % lot) != 0:
                    violations.append(
                        f"min transfer lot violation for {t.product}: qty {t.quantity} not multiple of {lot}"
                    )
                    is_feasible = False
                    continue

            if inv[(t.from_warehouse, t.product)] < t.quantity:
                violations.append(
                    f"insufficient inventory for {t.from_warehouse}/{t.product}: have {inv[(t.from_warehouse, t.product)]}, tried {t.quantity}"
                )
                is_feasible = False
                continue

            inv[(t.from_warehouse, t.product)] -= t.quantity
            inv[(t.to_warehouse, t.product)] += t.quantity

            transfer_out[t.from_warehouse] += t.quantity
            transfer_in[t.to_warehouse] += t.quantity

            transfer_cost_total += obs.transfer_cost[t.from_warehouse][t.to_warehouse] * t.quantity

            key = (t.from_warehouse, t.to_warehouse, t.product)
            lane_used[key] = lane_used.get(key, 0) + int(t.quantity)

        if obs.lane_capacity:
            for (i, j, p), used in lane_used.items():
                cap = obs.lane_capacity.get(i, {}).get(j, {}).get(p)
                if cap is not None and used > int(cap):
                    violations.append(
                        f"lane capacity exceeded for {i}->{j}/{p}: {used} > {int(cap)}"
                    )
                    is_feasible = False

        if obs.sku_capacity:
            for w in wh_ids:
                by_product = obs.sku_capacity.get(w, {})
                for p in products:
                    cap = by_product.get(p)
                    if cap is None:
                        continue
                    if inv[(w, p)] > int(cap):
                        violations.append(
                            f"SKU capacity exceeded for {w}/{p}: {inv[(w, p)]} > {int(cap)}"
                        )
                        is_feasible = False

        # Lane fixed activation costs (count each (i,j) once per step if any flow)
        if obs.lane_fixed_cost:
            activated = {(i, j) for (i, j, _p), used in lane_used.items() if used > 0}
            for (i, j) in activated:
                lane_activation_cost_total += float(obs.lane_fixed_cost.get(i, {}).get(j, 0.0))

        if obs.outbound_capacity:
            for w, cap in obs.outbound_capacity.items():
                if transfer_out.get(w, 0) > cap:
                    violations.append(f"outbound capacity exceeded for {w}: {transfer_out[w]} > {cap}")
                    is_feasible = False

        if obs.inbound_capacity:
            for w, cap in obs.inbound_capacity.items():
                if transfer_in.get(w, 0) > cap:
                    violations.append(f"inbound capacity exceeded for {w}: {transfer_in[w]} > {cap}")
                    is_feasible = False

        total_transfer_related = transfer_cost_total + lane_activation_cost_total

        # Per-step budget resets each step; episode budget is cumulative.
        if obs.per_step_budget is not None:
            if total_transfer_related > obs.per_step_budget + 1e-9:
                violations.append(
                    f"per-step budget exceeded: {total_transfer_related:.3f} > {obs.per_step_budget:.3f}"
                )
                is_feasible = False
        elif obs.budget is not None:
            cumulative_spend = (
                self._state.cumulative_transfer_cost
                + self._state.cumulative_lane_activation_cost
                + total_transfer_related
            )
            if cumulative_spend > obs.budget + 1e-9:
                violations.append(
                    f"episode budget exceeded: {cumulative_spend:.3f} > {obs.budget:.3f}"
                )
                is_feasible = False

        # Update live inventory state (applied even if infeasible, matching single-step behaviour)
        for wid in wh_ids:
            for p in products:
                self._state.current_inventory[wid][p] = int(inv[(wid, p)])
        self._state.cumulative_transfer_cost += transfer_cost_total
        self._state.cumulative_lane_activation_cost += lane_activation_cost_total

        is_final = (self._state.steps_remaining <= 0)

        updated_warehouses: list[Warehouse] = [
            Warehouse(
                id=w.id,
                inventory={p: int(inv[(w.id, p)]) for p in products},
                demand=w.demand,
            )
            for w in obs.warehouses
        ]
        obs.warehouses = updated_warehouses
        obs.transfer_cost_total = float(transfer_cost_total)
        obs.lane_activation_cost_total = float(lane_activation_cost_total)

        activated_lanes = {(i, j) for (i, j, _p), used in lane_used.items() if used > 0}
        obs.lanes_activated = int(len(activated_lanes))
        obs.co2_kg = float(sum(
            float(obs.transfer_cost[i][j]) * float(used)
            for (i, j, _p), used in lane_used.items() if used > 0
        ))
        obs.is_feasible = bool(is_feasible)
        obs.violations = violations
        obs.step_number = self._state.step_count

        if is_final:
            shortage_units_total = sum(
                max(0, int(dem[(w, p)]) - int(inv[(w, p)])) for w in wh_ids for p in products
            )
            shortage_penalty_total = shortage_units_total * obs.penalty_per_unit_shortage
            total_demand_units = sum(int(dem[(w, p)]) for w in wh_ids for p in products)
            fulfilled_units = total_demand_units - shortage_units_total
            fill_rate = float(fulfilled_units / total_demand_units) if total_demand_units > 0 else 1.0

            cumulative_cost = (
                self._state.cumulative_transfer_cost
                + self._state.cumulative_lane_activation_cost
                + shortage_penalty_total
            )
            obs.shortage_units_total = int(shortage_units_total)
            obs.shortage_penalty_total = float(shortage_penalty_total)
            obs.total_cost = float(cumulative_cost)
            obs.total_demand_units = int(total_demand_units)
            obs.fulfilled_units = int(fulfilled_units)
            obs.fill_rate = float(fill_rate)
            obs.reward = -float(cumulative_cost) if is_feasible else -float(cumulative_cost) - 1e6
            obs.done = True

            _attach_optimal_reference(obs, self._problem)
            _finalize_grading_fields(obs)
        else:
            # Intermediate step: useful per-step reward signal; shortage assessed at end.
            total_demand_units = sum(int(dem[(w, p)]) for w in wh_ids for p in products)
            obs.shortage_units_total = 0
            obs.shortage_penalty_total = 0.0
            obs.total_cost = float(total_transfer_related)
            obs.total_demand_units = int(total_demand_units)
            obs.fulfilled_units = 0
            obs.fill_rate = 0.0
            obs.reward = -float(total_transfer_related) if is_feasible else -float(total_transfer_related) - 1e6
            obs.done = False

        return obs

    @property
    def state(self) -> InventoryTransferState:
        return self._state
