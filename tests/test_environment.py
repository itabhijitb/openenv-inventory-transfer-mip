from __future__ import annotations

from inventory_transfer_env import InventoryTransferAction, Transfer
from inventory_transfer_env.server.inventory_transfer_environment import InventoryTransferEnvironment


def test_reset_is_deterministic_for_easy() -> None:
    env1 = InventoryTransferEnvironment()
    env2 = InventoryTransferEnvironment()

    obs1 = env1.reset(task_id="easy")
    obs2 = env2.reset(task_id="easy")

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
