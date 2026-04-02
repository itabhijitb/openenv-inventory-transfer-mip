from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import InventoryTransferAction, InventoryTransferObservation, InventoryTransferState


class InventoryTransferEnv(
    EnvClient[InventoryTransferAction, InventoryTransferObservation, InventoryTransferState]
):
    def _step_payload(self, action: InventoryTransferAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[InventoryTransferObservation]:
        obs = InventoryTransferObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> InventoryTransferState:
        return InventoryTransferState(**payload)
