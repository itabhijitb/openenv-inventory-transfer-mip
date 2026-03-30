from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import InventoryTransferAction, InventoryTransferObservation, InventoryTransferState


class InventoryTransferEnv(
    EnvClient[InventoryTransferAction, InventoryTransferObservation, InventoryTransferState]
):
    def _step_payload(self, action: InventoryTransferAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[InventoryTransferObservation]:
        obs = InventoryTransferObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> InventoryTransferState:
        return InventoryTransferState(**payload)
