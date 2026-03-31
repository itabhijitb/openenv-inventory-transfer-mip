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
