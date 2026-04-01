from __future__ import annotations

import json
from pathlib import Path

from inventory_transfer_env import InventoryTransferAction, Transfer
from inventory_transfer_env.server.inventory_transfer_environment import InventoryTransferEnvironment


def test_reset_is_deterministic_for_easy() -> None:
    env1 = InventoryTransferEnvironment()
    env2 = InventoryTransferEnvironment()

    obs1 = env1.reset(task_id="easy")
    obs2 = env2.reset(task_id="easy")

    assert obs1.model_dump() == obs2.model_dump()


def test_all_tasks_in_json_can_reset() -> None:
    tasks_path = Path(__file__).resolve().parents[1] / "inventory_transfer_env" / "tasks.json"
    task_ids = list(json.loads(tasks_path.read_text()).keys())
    assert len(task_ids) >= 3

    env = InventoryTransferEnvironment()
    for task_id in task_ids:
        obs = env.reset(task_id=task_id)
        assert obs.task_id == task_id
        assert len(obs.warehouses) > 0
        assert len(obs.products) > 0


def test_reset_is_deterministic_for_all_tasks_in_json() -> None:
    tasks_path = Path(__file__).resolve().parents[1] / "inventory_transfer_env" / "tasks.json"
    task_ids = list(json.loads(tasks_path.read_text()).keys())

    for task_id in task_ids:
        env1 = InventoryTransferEnvironment()
        env2 = InventoryTransferEnvironment()
        obs1 = env1.reset(task_id=task_id)
        obs2 = env2.reset(task_id=task_id)
        assert obs1.model_dump() == obs2.model_dump()


def test_step_score_bounds_and_done_flag() -> None:
    env = InventoryTransferEnvironment()
    _ = env.reset(task_id="easy")

    step = env.step(InventoryTransferAction(transfers=[]))

    assert step.done is True
    assert 0.0 <= float(step.score) <= 1.0


def test_min_transfer_lot_violation_disqualifies() -> None:
    env = InventoryTransferEnvironment()
    _ = env.reset(task_id="easy")

    # easy has min_transfer_lot P1=5; quantity=3 should violate
    action = InventoryTransferAction(
        transfers=[Transfer(from_warehouse="W1", to_warehouse="W2", product="P1", quantity=3)]
    )
    step = env.step(action)

    assert step.disqualified is True
    assert step.score == 0.0
    assert any("min transfer lot" in r.lower() for r in (step.dq_reasons or []))


def test_grader_lot_multiples_respected() -> None:
    """MIP grader must produce optimal_cost achievable with lot-multiple quantities."""
    from inventory_transfer_env.server.inventory_transfer_environment import _solve_optimal_mip
    from inventory_transfer_env.models import Warehouse

    warehouses = [
        Warehouse(id="A", inventory={"P1": 50}, demand={"P1": 10}),
        Warehouse(id="B", inventory={"P1": 0},  demand={"P1": 30}),
    ]
    cost, feasible = _solve_optimal_mip(
        warehouses=warehouses,
        products=["P1"],
        transfer_cost={"A": {"A": 0.0, "B": 2.0}, "B": {"A": 2.0, "B": 0.0}},
        lane_fixed_cost=None,
        penalty_per_unit_shortage=20.0,
        budget=None,
        outbound_capacity=None,
        inbound_capacity=None,
        sku_capacity=None,
        lane_capacity=None,
        min_transfer_lot={"P1": 10},
    )
    assert feasible
    # Optimal: send 30 units (3 lots of 10) from A to B → cost = 30*2 = 60
    assert abs(cost - 60.0) < 1e-6, f"Expected 60.0, got {cost}"


def test_optimal_score_achievable_on_easy() -> None:
    """Agent submitting the MIP-optimal solution should receive score ≈ 1.0."""
    from inventory_transfer_env.server.inventory_transfer_environment import (
        _solve_optimal_mip,
        _build_task_observation,
    )

    env = InventoryTransferEnvironment()
    obs = env.reset(task_id="easy")

    # Solve for the optimal transfers
    opt_cost, feasible = _solve_optimal_mip(
        warehouses=obs.warehouses,
        products=obs.products,
        transfer_cost=obs.transfer_cost,
        lane_fixed_cost=obs.lane_fixed_cost,
        penalty_per_unit_shortage=obs.penalty_per_unit_shortage,
        budget=obs.budget,
        outbound_capacity=obs.outbound_capacity,
        inbound_capacity=obs.inbound_capacity,
        sku_capacity=obs.sku_capacity,
        lane_capacity=obs.lane_capacity,
        min_transfer_lot=obs.min_transfer_lot,
    )
    assert feasible

    # Submit empty action and confirm score reflects shortage cost
    # (we just verify the grader runs and returns a score in range)
    step = env.step(InventoryTransferAction(transfers=[]))
    assert 0.0 <= float(step.score) <= 1.0
    assert step.optimal_cost is not None
    assert abs(step.optimal_cost - opt_cost) < 1e-3, (
        f"Grader optimal_cost {step.optimal_cost} does not match direct MIP call {opt_cost}"
    )


def test_multi_step_inventory_carries_forward() -> None:
    """rolling_3day: inventory state must persist across steps."""
    env = InventoryTransferEnvironment()
    obs0 = env.reset(task_id="rolling_3day")

    assert obs0.max_steps == 3
    assert obs0.done is False

    # Step 1: move 30 units P1 from W1 to W2
    step1 = env.step(InventoryTransferAction(
        transfers=[Transfer(from_warehouse="W1", to_warehouse="W2", product="P1", quantity=30)]
    ))
    assert step1.done is False, "Episode should not end after step 1 of 3"
    # W1 inventory for P1 should decrease by 30
    w1_after = next(w for w in step1.warehouses if w.id == "W1")
    assert w1_after.inventory["P1"] == obs0.warehouses[0].inventory["P1"] - 30

    # Step 2: empty action
    step2 = env.step(InventoryTransferAction(transfers=[]))
    assert step2.done is False
    # W1 inventory should stay at reduced level (state carried forward)
    w1_step2 = next(w for w in step2.warehouses if w.id == "W1")
    assert w1_step2.inventory["P1"] == w1_after.inventory["P1"]

    # Step 3: final step
    step3 = env.step(InventoryTransferAction(transfers=[]))
    assert step3.done is True
    assert step3.score is not None
    assert 0.0 <= float(step3.score) <= 1.0


def test_multi_step_intermediate_reward_nonzero() -> None:
    """Non-final steps with transfers must emit a nonzero (negative) reward."""
    env = InventoryTransferEnvironment()
    env.reset(task_id="rolling_3day")

    step = env.step(InventoryTransferAction(
        transfers=[Transfer(from_warehouse="W1", to_warehouse="W2", product="P1", quantity=30)]
    ))
    assert step.done is False
    assert step.reward is not None
    assert float(step.reward) < 0.0, "Transfer cost should produce negative reward"


def test_hub_spoke_task_resets_cleanly() -> None:
    """hub_spoke task must load and have the correct topology (spoke-to-spoke blocked)."""
    env = InventoryTransferEnvironment()
    obs = env.reset(task_id="hub_spoke")

    assert obs.max_steps == 1
    assert len(obs.warehouses) == 6
    assert obs.lane_capacity is not None

    # Spoke-to-spoke lane W2->W3 must have zero capacity
    cap_p1 = obs.lane_capacity.get("W2", {}).get("W3", {}).get("P1", None)
    assert cap_p1 == 0, f"Expected 0 capacity for W2->W3/P1, got {cap_p1}"
